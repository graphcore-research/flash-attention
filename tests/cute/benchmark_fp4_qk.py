import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys

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
    values = FP4_GRID.to(device=packed.device)[_unpack_fp4(packed).long()]
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
    # Some environments can launch CUDA kernels but fail during CUDA-local RNG setup.
    # Keep the benchmark itself unchanged by generating inputs on CPU and moving them to CUDA.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.empty(1, device="cuda")

    q_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen, num_heads, head_dim // 2),
        device="cpu",
        dtype=torch.uint8,
        generator=generator,
    ).to(device="cuda")
    k_packed = torch.randint(
        0,
        256,
        (batch_size, seqlen, num_heads_kv, head_dim // 2),
        device="cpu",
        dtype=torch.uint8,
        generator=generator,
    ).to(device="cuda")
    v = (
        torch.randn(
            batch_size,
            seqlen,
            num_heads_kv,
            head_dim,
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        ).to(dtype=torch.bfloat16, device="cuda")
        * 0.25
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
    q_ref = _dequantize_fp4(q_packed, q_scale, 16).to(torch.bfloat16)
    k_ref = _dequantize_fp4(k_packed, k_scale, 16).to(torch.bfloat16)
    return q_packed, k_packed, v, q_scale, k_scale, q_ref, k_ref


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
    skip_qk_baseline_check: bool = False,
):
    q_packed, k_packed, v, q_scale, k_scale, q_ref, k_ref = _make_inputs(
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
            v,
            causal=causal,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=q_scale,
            k_scale=k_scale,
        )

    def run_bf16_direct():
        return _flash_attn_fwd(
            q_ref,
            k_ref,
            v,
            causal=causal,
            return_lse=True,
        )

    def run_bf16_public():
        return flash_attn_func(
            q_ref,
            k_ref,
            v,
            causal=causal,
            return_lse=True,
        )

    out_fp4, lse_fp4 = run_fp4()
    out_bf16, lse_bf16 = run_bf16_direct()
    out_public, lse_public = run_bf16_public()
    torch.cuda.synchronize()

    if not torch.isfinite(out_fp4.float()).all().item():
        raise RuntimeError(f"FP4 output contains NaN or Inf for {kind}, d={head_dim}, s={seqlen}, causal={causal}.")
    if not torch.isfinite(lse_fp4.float()).all().item():
        raise RuntimeError(f"FP4 LSE contains NaN or Inf for {kind}, d={head_dim}, s={seqlen}, causal={causal}.")

    if not skip_qk_baseline_check:
        torch.testing.assert_close(out_fp4.float(), out_bf16.float(), atol=2e-1, rtol=5e-2)
        torch.testing.assert_close(lse_fp4.float(), lse_bf16.float(), atol=2e-1, rtol=5e-2)
    torch.testing.assert_close(out_public.float(), out_bf16.float(), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse_public.float(), lse_bf16.float(), atol=1e-3, rtol=1e-3)

    fp4_ms = _time_ms(run_fp4, warmup=warmup, iters=iters)
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
        "bf16_ms": bf16_ms,
        "public_ms": public_ms,
        "qk_fp4_over_bf16": fp4_ms / bf16_ms,
        "qk_fp4_over_public": fp4_ms / public_ms,
        "qk_out_max": (out_fp4.float() - out_bf16.float()).abs().max().item(),
        "qk_lse_max": (lse_fp4.float() - lse_bf16.float()).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark NVFP4 QK forward against BF16 Cute FA4.")
    parser.add_argument("--kinds", default="mha,gqa", help="Comma-separated benchmark kinds: mha,gqa")
    parser.add_argument("--head-dims", default="64,128", help="Comma-separated head dims.")
    parser.add_argument("--seqlens", default="128,512,2048", help="Comma-separated sequence lengths.")
    parser.add_argument("--causal-values", default="false,true", help="Comma-separated causal flags.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--scale-value", type=float, default=0.125)
    parser.add_argument(
        "--skip-qk-baseline-check",
        action="store_true",
        help="Skip the FP4-vs-BF16 assert so regression rows can still be reported.",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        torch.empty(1, device="cuda")
    except Exception as exc:
        raise SystemExit("CUDA is required for benchmark_fp4_qk.py.") from exc
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        raise SystemExit(f"NVFP4 benchmark requires SM100/SM110, got {major}.{minor}.")

    kinds = _parse_csv_kinds(args.kinds)
    head_dims = _parse_csv_ints(args.head_dims)
    seqlens = _parse_csv_ints(args.seqlens)
    causal_values = _parse_csv_bools(args.causal_values)
    rows = [
        (kind, head_dim, causal, seqlen)
        for kind in kinds
        for head_dim in head_dims
        for causal in causal_values
        for seqlen in seqlens
    ]

    if args.emit_json:
        if len(rows) != 1:
            raise SystemExit("--emit-json expects exactly one benchmark row.")
        kind, head_dim, causal, seqlen = rows[0]
        print(
            json.dumps(
                _benchmark_case(
                    kind=kind,
                    seqlen=seqlen,
                    head_dim=head_dim,
                    causal=causal,
                    batch_size=args.batch_size,
                    scale_value=args.scale_value,
                    warmup=args.warmup,
                    iters=args.iters,
                    skip_qk_baseline_check=args.skip_qk_baseline_check,
                )
            )
        )
        return

    script_path = pathlib.Path(__file__).resolve()

    print(
        "kind,hq,hkv,d,causal,seqlen,qk_fp4_ms,bf16_ms,public_ms,"
        "qk_fp4_over_bf16,qk_fp4_over_public,qk_out_max,qk_lse_max"
    )
    for kind, head_dim, causal, seqlen in rows:
        if len(rows) == 1:
            result = _benchmark_case(
                kind=kind,
                seqlen=seqlen,
                head_dim=head_dim,
                causal=causal,
                batch_size=args.batch_size,
                scale_value=args.scale_value,
                warmup=args.warmup,
                iters=args.iters,
                skip_qk_baseline_check=args.skip_qk_baseline_check,
            )
        else:
            cmd = [
                sys.executable,
                str(script_path),
                "--kinds",
                kind,
                "--head-dims",
                str(head_dim),
                "--seqlens",
                str(seqlen),
                "--causal-values",
                "true" if causal else "false",
                "--batch-size",
                str(args.batch_size),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--scale-value",
                str(args.scale_value),
                "--emit-json",
            ]
            if args.skip_qk_baseline_check:
                cmd.append("--skip-qk-baseline-check")
            child = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=dict(os.environ),
            )
            result = json.loads(child.stdout)
        print(
            f"{result['kind']},{result['hq']},{result['hkv']},{result['d']},"
            f"{result['causal']},{result['seqlen']},"
            f"{result['qk_fp4_ms']:.5f},{result['bf16_ms']:.5f},{result['public_ms']:.5f},"
            f"{result['qk_fp4_over_bf16']:.3f},{result['qk_fp4_over_public']:.3f},"
            f"{result['qk_out_max']:.6f},{result['qk_lse_max']:.6f}"
        )


if __name__ == "__main__":
    main()
