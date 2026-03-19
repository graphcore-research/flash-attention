import argparse
import statistics

import torch

from flash_attn.cute.interface import _flash_attn_fwd, flash_attn_func


FP4_GRID = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
    device="cuda",
)

KIND_TO_HEADS = {
    "mha": (4, 4),
    "gqa": (6, 2),
}


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_bools(value: str) -> list[bool]:
    parsed = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized in ("true", "1"):
            parsed.append(True)
        elif normalized in ("false", "0"):
            parsed.append(False)
        else:
            raise ValueError(f"Unsupported boolean value {item!r}. Use true/false.")
    return parsed


def _parse_csv_kinds(value: str) -> list[str]:
    kinds = [item.strip().lower() for item in value.split(",") if item.strip()]
    for kind in kinds:
        if kind not in KIND_TO_HEADS:
            raise ValueError(f"Unsupported kind {kind!r}. Expected one of {tuple(KIND_TO_HEADS)}.")
    return kinds


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)


def _dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID[_unpack_fp4(packed).long()]
    return (
        values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
        * scale.to(torch.float32).unsqueeze(-1)
    ).flatten(start_dim=-2)


def _make_inputs(
    *,
    kind: str,
    seqlen: int,
    head_dim: int,
    causal: bool,
    batch_size: int,
    scale_value: float,
):
    num_heads, num_heads_kv = KIND_TO_HEADS[kind]
    seed = 17_000 + seqlen * 13 + head_dim * 101 + (1 if causal else 0) + num_heads * 7 + num_heads_kv
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    q_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen, num_heads, head_dim // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    k_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen, num_heads_kv, head_dim // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    v_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen, num_heads_kv, head_dim // 2),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    )
    q_scale = torch.full(
        (batch_size, seqlen, num_heads, head_dim // 16),
        scale_value,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    k_scale = torch.full(
        (batch_size, seqlen, num_heads_kv, head_dim // 16),
        scale_value,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    v_scale = torch.full(
        (batch_size, seqlen, num_heads_kv, head_dim // 16),
        scale_value,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    q_ref = _dequantize_fp4(q_packed, q_scale, 16).to(torch.bfloat16)
    k_ref = _dequantize_fp4(k_packed, k_scale, 16).to(torch.bfloat16)
    v_ref = _dequantize_fp4(v_packed, v_scale, 16).to(torch.bfloat16)
    return q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, q_ref, k_ref, v_ref


def _time_ms(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return statistics.median(times_ms)


def _benchmark_case(
    *,
    kind: str,
    seqlen: int,
    head_dim: int,
    causal: bool,
    batch_size: int,
    scale_value: float,
    warmup: int,
    iters: int,
):
    q_packed, k_packed, v_packed, q_scale, k_scale, v_scale, q_ref, k_ref, v_ref = _make_inputs(
        kind=kind,
        seqlen=seqlen,
        head_dim=head_dim,
        causal=causal,
        batch_size=batch_size,
        scale_value=scale_value,
    )

    _flash_attn_fwd.compile_cache.clear()

    def run_fp4():
        return _flash_attn_fwd(
            q_packed,
            k_packed,
            v_ref,
            causal=causal,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    def run_fp4_pv():
        return _flash_attn_fwd(
            q_packed,
            k_packed,
            v_packed,
            causal=causal,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
            use_fp4_pv=True,
            v_scale=v_scale,
        )

    def run_bf16_direct():
        return _flash_attn_fwd(
            q_ref,
            k_ref,
            v_ref,
            causal=causal,
            return_lse=True,
        )

    def run_bf16_public():
        return flash_attn_func(
            q_ref,
            k_ref,
            v_ref,
            causal=causal,
            return_lse=True,
        )

    out_fp4, lse_fp4 = run_fp4()
    out_fp4_pv, lse_fp4_pv = run_fp4_pv()
    out_bf16, lse_bf16 = run_bf16_direct()
    out_public, lse_public = run_bf16_public()
    torch.cuda.synchronize()

    if not torch.isfinite(out_fp4.float()).all().item():
        raise RuntimeError(f"FP4 output contains NaN or Inf for {kind}, d={head_dim}, s={seqlen}, causal={causal}.")
    if not torch.isfinite(lse_fp4.float()).all().item():
        raise RuntimeError(f"FP4 LSE contains NaN or Inf for {kind}, d={head_dim}, s={seqlen}, causal={causal}.")
    if not torch.isfinite(out_fp4_pv.float()).all().item():
        raise RuntimeError(
            f"FP4 PV output contains NaN or Inf for {kind}, d={head_dim}, s={seqlen}, causal={causal}."
        )
    if not torch.isfinite(lse_fp4_pv.float()).all().item():
        raise RuntimeError(
            f"FP4 PV LSE contains NaN or Inf for {kind}, d={head_dim}, s={seqlen}, causal={causal}."
        )

    torch.testing.assert_close(out_fp4.float(), out_bf16.float(), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(lse_fp4.float(), lse_bf16.float(), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(out_fp4_pv.float(), out_bf16.float(), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(lse_fp4_pv.float(), lse_bf16.float(), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(out_public.float(), out_bf16.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse_public.float(), lse_bf16.float(), atol=1e-3, rtol=1e-3)

    fp4_ms = _time_ms(run_fp4, warmup=warmup, iters=iters)
    fp4_pv_ms = _time_ms(run_fp4_pv, warmup=warmup, iters=iters)
    bf16_ms = _time_ms(run_bf16_direct, warmup=warmup, iters=iters)
    public_ms = _time_ms(run_bf16_public, warmup=warmup, iters=iters)

    num_heads, num_heads_kv = KIND_TO_HEADS[kind]
    return {
        "kind": kind,
        "hq": num_heads,
        "hkv": num_heads_kv,
        "d": head_dim,
        "causal": causal,
        "seqlen": seqlen,
        "qk_fp4_ms": fp4_ms,
        "qk_pv_fp4_ms": fp4_pv_ms,
        "bf16_ms": bf16_ms,
        "public_ms": public_ms,
        "qk_fp4_over_bf16": fp4_ms / bf16_ms,
        "qk_pv_fp4_over_bf16": fp4_pv_ms / bf16_ms,
        "qk_fp4_over_public": fp4_ms / public_ms,
        "qk_pv_fp4_over_public": fp4_pv_ms / public_ms,
        "qk_out_max": (out_fp4.float() - out_bf16.float()).abs().max().item(),
        "qk_lse_max": (lse_fp4.float() - lse_bf16.float()).abs().max().item(),
        "pv_out_max": (out_fp4_pv.float() - out_bf16.float()).abs().max().item(),
        "pv_lse_max": (lse_fp4_pv.float() - lse_bf16.float()).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 QK-only and QK+PV forward against BF16 Cute FA4."
    )
    parser.add_argument("--kinds", default="mha,gqa", help="Comma-separated benchmark kinds: mha,gqa")
    parser.add_argument("--head-dims", default="64,128", help="Comma-separated head dims.")
    parser.add_argument("--seqlens", default="128,512,2048", help="Comma-separated sequence lengths.")
    parser.add_argument("--causal-values", default="false,true", help="Comma-separated causal flags.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--scale-value", type=float, default=0.125)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmark_fp4_qk.py.")
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        raise SystemExit(f"NVFP4 benchmark requires SM100/SM110, got {major}.{minor}.")

    kinds = _parse_csv_kinds(args.kinds)
    head_dims = _parse_csv_ints(args.head_dims)
    seqlens = _parse_csv_ints(args.seqlens)
    causal_values = _parse_csv_bools(args.causal_values)

    print(
        "kind,hq,hkv,d,causal,seqlen,"
        "qk_fp4_ms,qk_pv_fp4_ms,bf16_ms,public_ms,"
        "qk_fp4_over_bf16,qk_pv_fp4_over_bf16,qk_fp4_over_public,qk_pv_fp4_over_public,"
        "qk_out_max,qk_lse_max,pv_out_max,pv_lse_max"
    )
    for kind in kinds:
        for head_dim in head_dims:
            for causal in causal_values:
                for seqlen in seqlens:
                    result = _benchmark_case(
                        kind=kind,
                        seqlen=seqlen,
                        head_dim=head_dim,
                        causal=causal,
                        batch_size=args.batch_size,
                        scale_value=args.scale_value,
                        warmup=args.warmup,
                        iters=args.iters,
                    )
                    print(
                        f"{result['kind']},{result['hq']},{result['hkv']},{result['d']},"
                        f"{result['causal']},{result['seqlen']},"
                        f"{result['qk_fp4_ms']:.5f},{result['qk_pv_fp4_ms']:.5f},"
                        f"{result['bf16_ms']:.5f},{result['public_ms']:.5f},"
                        f"{result['qk_fp4_over_bf16']:.3f},{result['qk_pv_fp4_over_bf16']:.3f},"
                        f"{result['qk_fp4_over_public']:.3f},{result['qk_pv_fp4_over_public']:.3f},"
                        f"{result['qk_out_max']:.6f},{result['qk_lse_max']:.6f},"
                        f"{result['pv_out_max']:.6f},{result['pv_lse_max']:.6f}"
                    )


if __name__ == "__main__":
    main()
