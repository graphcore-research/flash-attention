import argparse
import contextlib
import os
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
)

KIND_TO_HEADS = {
    "mha": (4, 4),
    "gqa": (6, 2),
}

FP4_PV_ENV_KEYS = (
    "FLASH_ATTN_FP4_PV_DIRECT_LOADER",
    "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE",
    "FLASH_ATTN_FP4_PV_ENCODE_CENTRIC",
    "FLASH_ATTN_FP4_PV_FORCE_CTA_DIRECT",
)

FP4_PV_MODE_TO_ENV = {
    "legacy": {},
    "legacy_direct": {
        "FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1",
    },
    "cta": {
        "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1",
    },
    "cta_encode": {
        "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1",
        "FLASH_ATTN_FP4_PV_ENCODE_CENTRIC": "1",
    },
    "cta_direct": {
        "FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1",
        "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1",
    },
    "cta_direct_encode": {
        "FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1",
        "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1",
        "FLASH_ATTN_FP4_PV_ENCODE_CENTRIC": "1",
    },
    "cta_direct_force": {
        "FLASH_ATTN_FP4_PV_DIRECT_LOADER": "1",
        "FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE": "1",
        "FLASH_ATTN_FP4_PV_FORCE_CTA_DIRECT": "1",
    },
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


def _parse_csv_pv_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    for mode in modes:
        if mode not in FP4_PV_MODE_TO_ENV:
            raise ValueError(f"Unsupported PV mode {mode!r}. Expected one of {tuple(FP4_PV_MODE_TO_ENV)}.")
    return modes


@contextlib.contextmanager
def _temporary_fp4_pv_env(overrides: dict[str, str]):
    saved = {key: os.environ.get(key) for key in FP4_PV_ENV_KEYS}
    try:
        for key in FP4_PV_ENV_KEYS:
            if key in overrides:
                os.environ[key] = overrides[key]
            else:
                os.environ.pop(key, None)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)


def _dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed.device)[_unpack_fp4(packed).long()]
    return (
        values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
        * scale.to(torch.float32).unsqueeze(-1)
    ).flatten(start_dim=-2)


def _swizzle_fp4_vt_scale(scale_vt: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
    out = torch.empty_like(scale_vt)
    flat_in = scale_vt.reshape(batch_size, num_heads, head_dim, seqlen_groups)
    flat_out = out.view(batch_size, num_heads, -1)
    for d in range(head_dim):
        tile_m, row_in_tile = divmod(d, 64)
        quad, row_mod16 = divmod(row_in_tile, 16)
        for seq_group in range(seqlen_groups):
            offset = (
                tile_m * 64 * seqlen_groups
                + (seq_group // 4) * 256
                + (seq_group % 4)
                + quad * 4
                + row_mod16 * 16
            )
            flat_out[:, :, offset] = flat_in[:, :, d, seq_group]
    return out


def _unswizzle_fp4_vt_scale(scale_vt: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, head_dim, seqlen_groups = scale_vt.shape
    out = torch.empty_like(scale_vt)
    flat_in = scale_vt.view(batch_size, num_heads, -1)
    logical = out.reshape(batch_size, num_heads, head_dim, seqlen_groups)
    for d in range(head_dim):
        tile_m, row_in_tile = divmod(d, 64)
        quad, row_mod16 = divmod(row_in_tile, 16)
        for seq_group in range(seqlen_groups):
            offset = (
                tile_m * 64 * seqlen_groups
                + (seq_group // 4) * 256
                + (seq_group % 4)
                + quad * 4
                + row_mod16 * 16
            )
            logical[:, :, d, seq_group] = flat_in[:, :, offset]
    return out


def _dequantize_fp4_vt(packed_vt: torch.Tensor, scale_vt: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed_vt.device)[_unpack_fp4(packed_vt).long()]
    logical_scale_vt = _unswizzle_fp4_vt_scale(scale_vt)
    values = values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
    values = values * logical_scale_vt.to(torch.float32).unsqueeze(-1)
    return values.flatten(start_dim=3, end_dim=4).permute(0, 3, 1, 2).contiguous()


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
    seqlen_padded = ((seqlen + 127) // 128) * 128
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
        (batch_size, num_heads_kv, head_dim, seqlen_padded // 2),
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
        (batch_size, num_heads_kv, head_dim, seqlen_padded // 16),
        scale_value,
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    v_scale = _swizzle_fp4_vt_scale(v_scale)
    q_ref = _dequantize_fp4(q_packed, q_scale, 16).to(torch.bfloat16)
    k_ref = _dequantize_fp4(k_packed, k_scale, 16).to(torch.bfloat16)
    v_ref = _dequantize_fp4_vt(v_packed, v_scale, 16)[:, :seqlen].to(torch.bfloat16)
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


def _sanitize_status(exc: BaseException) -> str:
    return f"error_{type(exc).__name__}"


def _numeric_status(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> str:
    try:
        torch.testing.assert_close(out.float(), out_ref.float(), atol=atol, rtol=rtol)
        torch.testing.assert_close(lse.float(), lse_ref.float(), atol=atol, rtol=rtol)
        return "ok"
    except AssertionError:
        return "accuracy_fail"


def _run_fp4_pv_mode(
    *,
    mode: str,
    q_packed: torch.Tensor,
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool,
):
    with _temporary_fp4_pv_env(FP4_PV_MODE_TO_ENV[mode]):
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


def _benchmark_fp4_pv_mode(
    *,
    runner,
    out_bf16: torch.Tensor,
    lse_bf16: torch.Tensor,
    out_legacy: torch.Tensor | None,
    lse_legacy: torch.Tensor | None,
    warmup: int,
    iters: int,
    bf16_ms: float,
    public_ms: float,
    atol: float = 2e-1,
    rtol: float = 5e-2,
):
    try:
        # Measure/report the warmed steady-state kernel, not the very first post-compile launch.
        # Some FP4 PV modes, especially causal CTA quantization, need one warm run before the
        # numerics settle to the values seen by the timed iterations below.
        runner()
        torch.cuda.synchronize()
        out, lse = runner()
        torch.cuda.synchronize()
        if not torch.isfinite(out.float()).all().item():
            raise RuntimeError("non_finite_out")
        if not torch.isfinite(lse.float()).all().item():
            raise RuntimeError("non_finite_lse")
        result = {
            "status": _numeric_status(out, lse, out_bf16, lse_bf16, atol=atol, rtol=rtol),
            "ms": _time_ms(runner, warmup=warmup, iters=iters),
            "out_max": (out.float() - out_bf16.float()).abs().max().item(),
            "lse_max": (lse.float() - lse_bf16.float()).abs().max().item(),
            "out_vs_legacy_max": float("nan"),
            "lse_vs_legacy_max": float("nan"),
        }
        result["over_bf16"] = result["ms"] / bf16_ms
        result["over_public"] = result["ms"] / public_ms
        if out_legacy is not None and lse_legacy is not None:
            result["out_vs_legacy_max"] = (out.float() - out_legacy.float()).abs().max().item()
            result["lse_vs_legacy_max"] = (lse.float() - lse_legacy.float()).abs().max().item()
        return result, out, lse
    except Exception as exc:
        return {
            "status": _sanitize_status(exc),
            "ms": float("nan"),
            "over_bf16": float("nan"),
            "over_public": float("nan"),
            "out_max": float("nan"),
            "lse_max": float("nan"),
            "out_vs_legacy_max": float("nan"),
            "lse_vs_legacy_max": float("nan"),
        }, None, None


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
    pv_modes: list[str],
    skip_qk_baseline_check: bool = False,
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
    pv_atol = 2.5e-1 if causal else 2e-1
    pv_rtol = 5e-2
    pv_results = {}
    legacy_out = None
    legacy_lse = None
    for mode in pv_modes:
        runner = lambda mode=mode: _run_fp4_pv_mode(
            mode=mode,
            q_packed=q_packed,
            k_packed=k_packed,
            v_packed=v_packed,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            causal=causal,
        )
        result, out_mode, lse_mode = _benchmark_fp4_pv_mode(
            runner=runner,
            out_bf16=out_bf16,
            lse_bf16=lse_bf16,
            out_legacy=legacy_out,
            lse_legacy=legacy_lse,
            warmup=warmup,
            iters=iters,
            bf16_ms=bf16_ms,
            public_ms=public_ms,
            atol=pv_atol,
            rtol=pv_rtol,
        )
        pv_results[mode] = result
        if mode == "legacy" and out_mode is not None and lse_mode is not None:
            legacy_out, legacy_lse = out_mode, lse_mode
            pv_results[mode]["out_vs_legacy_max"] = 0.0
            pv_results[mode]["lse_vs_legacy_max"] = 0.0
    if legacy_out is not None and legacy_lse is not None:
        for mode in pv_modes:
            if mode == "legacy" or pv_results[mode]["status"].startswith("error_"):
                continue
            out_mode, lse_mode = _run_fp4_pv_mode(
                mode=mode,
                q_packed=q_packed,
                k_packed=k_packed,
                v_packed=v_packed,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                causal=causal,
            )
            torch.cuda.synchronize()
            pv_results[mode]["out_vs_legacy_max"] = (out_mode.float() - legacy_out.float()).abs().max().item()
            pv_results[mode]["lse_vs_legacy_max"] = (lse_mode.float() - legacy_lse.float()).abs().max().item()

    num_heads, num_heads_kv = KIND_TO_HEADS[kind]
    result = {
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
    for mode, mode_result in pv_results.items():
        prefix = f"pv_{mode}"
        result[f"{prefix}_status"] = mode_result["status"]
        result[f"{prefix}_ms"] = mode_result["ms"]
        result[f"{prefix}_over_bf16"] = mode_result["over_bf16"]
        result[f"{prefix}_over_public"] = mode_result["over_public"]
        result[f"{prefix}_out_max"] = mode_result["out_max"]
        result[f"{prefix}_lse_max"] = mode_result["lse_max"]
        result[f"{prefix}_out_vs_legacy_max"] = mode_result["out_vs_legacy_max"]
        result[f"{prefix}_lse_vs_legacy_max"] = mode_result["lse_vs_legacy_max"]
    return result


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
    parser.add_argument(
        "--pv-modes",
        default="legacy,cta,cta_direct",
        help="Comma-separated FP4 PV modes: legacy,legacy_direct,cta,cta_encode,cta_direct,cta_direct_encode,cta_direct_force",
    )
    parser.add_argument(
        "--skip-qk-baseline-check",
        action="store_true",
        help="Skip the standalone QK-only FP4-vs-BF16 assert so causal PV benchmark rows can still be reported.",
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
    pv_modes = _parse_csv_pv_modes(args.pv_modes)

    header = [
        "kind",
        "hq",
        "hkv",
        "d",
        "causal",
        "seqlen",
        "qk_fp4_ms",
        "bf16_ms",
        "public_ms",
        "qk_fp4_over_bf16",
        "qk_fp4_over_public",
        "qk_out_max",
        "qk_lse_max",
    ]
    for mode in pv_modes:
        prefix = f"pv_{mode}"
        header.extend(
            [
                f"{prefix}_status",
                f"{prefix}_ms",
                f"{prefix}_over_bf16",
                f"{prefix}_over_public",
                f"{prefix}_out_max",
                f"{prefix}_lse_max",
                f"{prefix}_out_vs_legacy_max",
                f"{prefix}_lse_vs_legacy_max",
            ]
        )
    print(",".join(header))
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
                        pv_modes=pv_modes,
                        skip_qk_baseline_check=args.skip_qk_baseline_check,
                    )
                    row = [
                        result["kind"],
                        str(result["hq"]),
                        str(result["hkv"]),
                        str(result["d"]),
                        str(result["causal"]),
                        str(result["seqlen"]),
                        f"{result['qk_fp4_ms']:.5f}",
                        f"{result['bf16_ms']:.5f}",
                        f"{result['public_ms']:.5f}",
                        f"{result['qk_fp4_over_bf16']:.3f}",
                        f"{result['qk_fp4_over_public']:.3f}",
                        f"{result['qk_out_max']:.6f}",
                        f"{result['qk_lse_max']:.6f}",
                    ]
                    for mode in pv_modes:
                        prefix = f"pv_{mode}"
                        row.extend(
                            [
                                result[f"{prefix}_status"],
                                f"{result[f'{prefix}_ms']:.5f}",
                                f"{result[f'{prefix}_over_bf16']:.3f}",
                                f"{result[f'{prefix}_over_public']:.3f}",
                                f"{result[f'{prefix}_out_max']:.6f}",
                                f"{result[f'{prefix}_lse_max']:.6f}",
                                f"{result[f'{prefix}_out_vs_legacy_max']:.6f}",
                                f"{result[f'{prefix}_lse_vs_legacy_max']:.6f}",
                            ]
                        )
                    print(",".join(row))


if __name__ == "__main__":
    main()
