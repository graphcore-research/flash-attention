import argparse
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys

try:
    import cuda.bindings.driver as _pre_torch_cuda_driver
    _pre_torch_cuda_driver.cuInit(0)
except Exception:
    _pre_torch_cuda_driver = None

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


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

DEFAULT_NUM_HEADS = 4
DEFAULT_NUM_HEADS_KV = 4
NVFP4_VEC_SIZE = 16
BOGUS_FAST_QKFAST_MS = 0.05
_FLASH_ATTN_FWD = None
_BENCH_DEVICE = None
_CUDA_DRIVER = False
EXACT_PROFILE_ROW = {
    "kind": "mha",
    "head_dim": 128,
    "seqlen": 512,
    "batch_size": 2,
    "causal": False,
}
EXACT_PROFILE_NCU_KERNEL_NAME_BASE = "demangled"
EXACT_PROFILE_NCU_KERNEL_REGEX = r"regex:.*FP4FlashAttentionForwardSm100PVFused.*"
EXACT_PROFILE_LAUNCH_SKIP = 1
EXACT_PROFILE_LAUNCH_COUNT = 1
EXACT_PROFILE_QKFAST_WARMUP = 2
EXACT_PROFILE_QKFAST_ITERS = 5
EXACT_PROFILE_SKIP_QKFAST_ENV = "FLASH_ATTN_FP4_PROFILE_EXACT_SKIP_QKFAST"
EXACT_SFV_DIRECT_ENV = "FLASH_ATTN_FP4_PV_EXACT_SFV_DIRECT"


def _get_cuda_driver():
    global _CUDA_DRIVER
    if _CUDA_DRIVER is False:
        try:
            import cuda.bindings.driver as cuda_driver
            _CUDA_DRIVER = cuda_driver
        except Exception:
            _CUDA_DRIVER = None
    return _CUDA_DRIVER


def _maybe_cuinit() -> None:
    cuda_driver = _get_cuda_driver()
    if cuda_driver is not None:
        try:
            cuda_driver.cuInit(0)
        except Exception:
            pass


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


def _unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    packed_i32 = packed.to(torch.int32)
    return torch.stack((packed_i32 & 0xF, (packed_i32 >> 4) & 0xF), dim=-1).flatten(start_dim=-2)


def _pack_fp4(indices: torch.Tensor) -> torch.Tensor:
    return (indices[..., ::2] | (indices[..., 1::2] << 4)).contiguous()


def _nearest_fp4_indices(values: torch.Tensor) -> torch.Tensor:
    grid = FP4_GRID.to(device=values.device)
    return (values.unsqueeze(-1) - grid).abs().argmin(dim=-1).to(torch.uint8)


def _nvfp4_scale_from_amax(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale_fp32 = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    scale_fp8 = scale_fp32.to(torch.float8_e4m3fn)
    return scale_fp8, torch.where(amax > 0, scale_fp8.to(torch.float32), torch.ones_like(scale_fp32))


def _quantize_nvfp4_lastdim(x: torch.Tensor, *, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"Expected last dim divisible by {block_size}, got {x.shape[-1]}.")
    blocks = x.to(torch.float32).unflatten(-1, (-1, block_size))
    scale_fp8, scale_fp32 = _nvfp4_scale_from_amax(blocks.abs().amax(dim=-1))
    quantized = blocks / scale_fp32.unsqueeze(-1)
    packed = _pack_fp4(_nearest_fp4_indices(quantized)).flatten(start_dim=-2)
    return packed.to(torch.uint8).contiguous(), scale_fp8.contiguous()


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


def _dequantize_fp4(packed: torch.Tensor, scale: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed.device)[_unpack_fp4(packed).long()]
    return (
        values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
        * scale.to(torch.float32).unsqueeze(-1)
    ).flatten(start_dim=-2)


def _dequantize_fp4_vt(packed_vt: torch.Tensor, scale_vt: torch.Tensor, sf_vec_size: int) -> torch.Tensor:
    values = FP4_GRID.to(device=packed_vt.device)[_unpack_fp4(packed_vt).long()]
    logical_scale_vt = _unswizzle_fp4_vt_scale(scale_vt)
    values = values.unflatten(dim=-1, sizes=(-1, sf_vec_size))
    values = values * logical_scale_vt.to(torch.float32).unsqueeze(-1)
    return values.flatten(start_dim=3, end_dim=4).permute(0, 3, 1, 2).contiguous()


def _get_benchmark_device() -> int:
    global _BENCH_DEVICE
    if _BENCH_DEVICE is not None:
        return _BENCH_DEVICE
    last_exc = None
    for device_idx in range(8):
        try:
            torch.cuda.set_device(device_idx)
            torch.empty(1, device="cuda")
            _BENCH_DEVICE = device_idx
            return device_idx
        except Exception as exc:
            last_exc = exc
    device_count = None
    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = None
    cuda_driver = _get_cuda_driver()
    if (device_count is None or device_count <= 0) and cuda_driver is not None:
        _maybe_cuinit()
        try:
            device_count = int(cuda_driver.cuDeviceGetCount()[1])
        except Exception:
            device_count = None
    if device_count is None or device_count <= 0:
        try:
            device_count = torch.cuda.device_count()
        except Exception:
            device_count = None
    if device_count <= 0:
        raise RuntimeError("CUDA is required for benchmark_fp4_pv.py.")
    for device_idx in range(device_count):
        try:
            torch.cuda.set_device(device_idx)
            torch.empty(1, device="cuda")
            _BENCH_DEVICE = device_idx
            return device_idx
        except Exception as exc:
            last_exc = exc
    raise RuntimeError("No usable CUDA device found for benchmark_fp4_pv.py.") from last_exc


def _enumerate_usable_devices() -> list[int]:
    devices = []
    last_exc = None
    _maybe_cuinit()
    for device_idx in range(8):
        try:
            torch.cuda.set_device(device_idx)
            torch.empty(1, device="cuda")
            devices.append(device_idx)
        except Exception as exc:
            last_exc = exc
    if devices:
        return devices
    raise RuntimeError("No usable CUDA device found for benchmark_fp4_pv.py.") from last_exc


def _is_known_bogus_fast_qkfast(
    *,
    qkfast_ms: float,
    seqlen: int,
    head_dim: int,
    causal: bool,
    num_heads: int,
    num_heads_kv: int,
) -> bool:
    return (
        not causal
        and head_dim == 128
        and seqlen == 512
        and num_heads == num_heads_kv
        and qkfast_ms < BOGUS_FAST_QKFAST_MS
    )


def _make_inputs(*, seqlen: int, head_dim: int, batch_size: int, num_heads: int, num_heads_kv: int):
    seed = 31_000 + seqlen * 13 + head_dim * 101 + batch_size * 17
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.cuda.set_device(_get_benchmark_device())
    torch.empty(1, device="cuda")

    q_bf16 = (
        torch.randn(
            batch_size,
            seqlen,
            num_heads,
            head_dim,
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        ).to(dtype=torch.bfloat16, device="cuda")
        * 0.25
    )
    k_bf16 = (
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
    v_bf16 = (
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

    q_qkfast_packed, q_qkfast_scale = _quantize_nvfp4_lastdim(q_bf16.float(), block_size=NVFP4_VEC_SIZE)
    k_qkfast_packed, k_qkfast_scale = _quantize_nvfp4_lastdim(k_bf16.float(), block_size=NVFP4_VEC_SIZE)

    seqlen_padded = ((seqlen + 127) // 128) * 128
    v_vt = torch.zeros(
        batch_size,
        num_heads_kv,
        head_dim,
        seqlen_padded,
        device="cuda",
        dtype=torch.float32,
    )
    v_vt[..., :seqlen] = v_bf16.permute(0, 2, 3, 1).to(torch.float32)
    v_pv_packed, v_scale_logical = _quantize_nvfp4_lastdim(v_vt, block_size=NVFP4_VEC_SIZE)
    v_pv_scale = _swizzle_fp4_vt_scale(v_scale_logical)

    return {
        "q_bf16": q_bf16,
        "k_bf16": k_bf16,
        "v_bf16": v_bf16,
        "q_qkfast_packed": q_qkfast_packed,
        "k_qkfast_packed": k_qkfast_packed,
        "q_qkfast_scale": q_qkfast_scale,
        "k_qkfast_scale": k_qkfast_scale,
        "v_pv_packed": v_pv_packed,
        "v_pv_scale": v_pv_scale,
    }

def _get_flash_attn_fwd():
    global _FLASH_ATTN_FWD
    if _FLASH_ATTN_FWD is None:
        from flash_attn.cute.interface import _flash_attn_fwd as flash_attn_fwd

        _FLASH_ATTN_FWD = flash_attn_fwd
    return _FLASH_ATTN_FWD


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


def _median_metric(samples: list[dict], key: str) -> float:
    values = [sample[key] for sample in samples if isinstance(sample.get(key), (int, float)) and math.isfinite(sample[key])]
    return statistics.median(values) if values else math.nan


def _max_metric(samples: list[dict], key: str) -> float:
    values = [sample[key] for sample in samples if isinstance(sample.get(key), (int, float)) and math.isfinite(sample[key])]
    return max(values) if values else math.nan


def _exact_profile_metadata(*, device_idx: int) -> dict:
    return {
        "profile_mode": "exact-lane",
        "profile_row": "mha_d128_s512_b2_noncausal",
        "profile_device": device_idx,
        "ncu_kernel_name_base": EXACT_PROFILE_NCU_KERNEL_NAME_BASE,
        "ncu_kernel_regex": EXACT_PROFILE_NCU_KERNEL_REGEX,
        "profile_launch_skip": EXACT_PROFILE_LAUNCH_SKIP,
        "profile_launch_count": EXACT_PROFILE_LAUNCH_COUNT,
        "profile_qkfast_warmup": EXACT_PROFILE_QKFAST_WARMUP,
        "profile_qkfast_iters": EXACT_PROFILE_QKFAST_ITERS,
        "profile_skip_qkfast_env": EXACT_PROFILE_SKIP_QKFAST_ENV,
    }


def _benchmark_case(
    *,
    seqlen: int,
    head_dim: int,
    causal: bool,
    batch_size: int,
    num_heads: int,
    num_heads_kv: int,
    warmup: int,
    iters: int,
    skip_baseline_check: bool = False,
    compare_mode: str = "full",
    device_idx: int | None = None,
    pv_tile_mn: tuple[int, int] | None = None,
):
    global _BENCH_DEVICE
    if device_idx is not None:
        _BENCH_DEVICE = device_idx
    profile_exact = compare_mode == "profile-exact"
    skip_qkfast_profile_baseline = (
        profile_exact and os.environ.get(EXACT_PROFILE_SKIP_QKFAST_ENV) == "1"
    )
    if profile_exact:
        if device_idx is None:
            raise RuntimeError("profile-exact mode requires an explicit device_idx.")
        if (
            head_dim != EXACT_PROFILE_ROW["head_dim"]
            or seqlen != EXACT_PROFILE_ROW["seqlen"]
            or batch_size != EXACT_PROFILE_ROW["batch_size"]
            or causal != EXACT_PROFILE_ROW["causal"]
            or num_heads != num_heads_kv
        ):
            raise RuntimeError("profile-exact mode is only supported on the must-win exact row.")
    tensors = _make_inputs(
        seqlen=seqlen,
        head_dim=head_dim,
        batch_size=batch_size,
        num_heads=num_heads,
        num_heads_kv=num_heads_kv,
    )
    flash_attn_fwd = _get_flash_attn_fwd()
    flash_attn_fwd.compile_cache.clear()

    def run_qkfast():
        return flash_attn_fwd(
            tensors["q_qkfast_packed"],
            tensors["k_qkfast_packed"],
            tensors["v_bf16"],
            causal=causal,
            return_lse=True,
            fp4_qk_format="nvfp4",
            q_scale=tensors["q_qkfast_scale"],
            k_scale=tensors["k_qkfast_scale"],
        )

    def run_pv_fp4():
        return flash_attn_fwd(
            tensors["q_qkfast_packed"],
            tensors["k_qkfast_packed"],
            tensors["v_pv_packed"],
            causal=causal,
            return_lse=True,
            tile_mn=pv_tile_mn,
            fp4_qk_format="nvfp4",
            q_scale=tensors["q_qkfast_scale"],
            k_scale=tensors["k_qkfast_scale"],
            use_fp4_pv=True,
            v_scale=tensors["v_pv_scale"],
        )

    def run_bf16_direct():
        return flash_attn_fwd(
            tensors["q_bf16"],
            tensors["k_bf16"],
            tensors["v_bf16"],
            causal=causal,
            return_lse=True,
        )

    include_bf16 = compare_mode == "full"

    # The exact-Sage must-win lane requires the very first fused PV launch in a
    # fresh process to be numerically clean. Run it before any other path so
    # warmup from qkfast cannot mask a first-launch bug.
    out_fp4, lse_fp4 = run_pv_fp4()
    out_qkfast, lse_qkfast = run_qkfast()
    out_bf16 = None
    lse_bf16 = None
    if include_bf16:
        out_bf16, lse_bf16 = run_bf16_direct()
    torch.cuda.synchronize()

    if not torch.isfinite(out_qkfast.float()).all().item():
        raise RuntimeError(f"QK-fast output contains NaN or Inf for d={head_dim}, s={seqlen}, causal={causal}.")
    if not torch.isfinite(lse_qkfast.float()).all().item():
        raise RuntimeError(f"QK-fast LSE contains NaN or Inf for d={head_dim}, s={seqlen}, causal={causal}.")
    if not torch.isfinite(out_fp4.float()).all().item():
        raise RuntimeError(f"FP4 PV output contains NaN or Inf for d={head_dim}, s={seqlen}, causal={causal}.")
    if not torch.isfinite(lse_fp4.float()).all().item():
        raise RuntimeError(f"FP4 PV LSE contains NaN or Inf for d={head_dim}, s={seqlen}, causal={causal}.")

    if include_bf16 and not skip_baseline_check:
        torch.testing.assert_close(out_qkfast.float(), out_bf16.float(), atol=2e-1, rtol=5e-2)
        torch.testing.assert_close(lse_qkfast.float(), lse_bf16.float(), atol=2e-1, rtol=5e-2)
        torch.testing.assert_close(out_fp4.float(), out_bf16.float(), atol=2e-1, rtol=5e-2)
        torch.testing.assert_close(lse_fp4.float(), lse_bf16.float(), atol=2e-1, rtol=5e-2)

    if profile_exact and not skip_qkfast_profile_baseline:
        qkfast_ms = _time_ms(
            run_qkfast,
            warmup=EXACT_PROFILE_QKFAST_WARMUP,
            iters=EXACT_PROFILE_QKFAST_ITERS,
        )
        fp4_ms = _time_ms(run_pv_fp4, warmup=1, iters=1)
    elif profile_exact:
        qkfast_ms = math.nan
        fp4_ms = _time_ms(run_pv_fp4, warmup=1, iters=1)
    else:
        qkfast_ms = _time_ms(run_qkfast, warmup=warmup, iters=iters)
        fp4_ms = _time_ms(run_pv_fp4, warmup=warmup, iters=iters)
    bf16_ms = math.nan
    if include_bf16:
        bf16_ms = _time_ms(run_bf16_direct, warmup=warmup, iters=iters)
    if not skip_qkfast_profile_baseline:
        if _is_known_bogus_fast_qkfast(
            qkfast_ms=qkfast_ms,
            seqlen=seqlen,
            head_dim=head_dim,
            causal=causal,
            num_heads=num_heads,
            num_heads_kv=num_heads_kv,
        ):
            raise RuntimeError(
                f"QK-fast baseline looks bogus-fast on device {_get_benchmark_device()} "
                f"(qkfast_ms={qkfast_ms:.5f}) for d={head_dim}, s={seqlen}, causal={causal}."
            )

    result = {
        "kind": "mha" if num_heads == num_heads_kv else "gqa",
        "device": _get_benchmark_device(),
        "hq": num_heads,
        "hkv": num_heads_kv,
        "d": head_dim,
        "causal": causal,
        "seqlen": seqlen,
        "qkfast_ms": qkfast_ms,
        "pv_fused_ms": fp4_ms,
        "bf16_ms": bf16_ms,
        "qkfast_over_bf16": qkfast_ms / bf16_ms if math.isfinite(qkfast_ms) and math.isfinite(bf16_ms) else math.nan,
        "pv_fused_over_qkfast": fp4_ms / qkfast_ms if math.isfinite(qkfast_ms) else math.nan,
        "pv_fused_over_bf16": fp4_ms / bf16_ms if math.isfinite(bf16_ms) else math.nan,
        "qkfast_out_max": (out_qkfast.float() - out_bf16.float()).abs().max().item() if out_bf16 is not None else math.nan,
        "qkfast_lse_max": (lse_qkfast.float() - lse_bf16.float()).abs().max().item() if lse_bf16 is not None else math.nan,
        "pv_fused_out_max": (out_fp4.float() - out_bf16.float()).abs().max().item() if out_bf16 is not None else math.nan,
        "pv_fused_lse_max": (lse_fp4.float() - lse_bf16.float()).abs().max().item() if lse_bf16 is not None else math.nan,
        "pv_fused_impl": "cute",
        "compare_mode": compare_mode,
        "exact_sfv_direct": os.environ.get(EXACT_SFV_DIRECT_ENV) == "1",
    }
    if profile_exact:
        result.update(_exact_profile_metadata(device_idx=_get_benchmark_device()))
        result["profile_skip_qkfast"] = skip_qkfast_profile_baseline
    return result


def _aggregate_benchmark_runs(samples: list[dict], failures: list[dict] | None = None) -> dict:
    if not samples:
        raise ValueError("Expected at least one successful sample to aggregate.")
    failures = failures or []
    first = samples[0]
    aggregated = {
        "kind": first["kind"],
        "device": first.get("device", -1),
        "hq": first["hq"],
        "hkv": first["hkv"],
        "d": first["d"],
        "causal": first["causal"],
        "seqlen": first["seqlen"],
        "success_count": len(samples),
        "failure_count": len(failures),
        "qkfast_ms": _median_metric(samples, "qkfast_ms"),
        "pv_fused_ms": _median_metric(samples, "pv_fused_ms"),
        "bf16_ms": _median_metric(samples, "bf16_ms"),
        "qkfast_out_max": _max_metric(samples, "qkfast_out_max"),
        "qkfast_lse_max": _max_metric(samples, "qkfast_lse_max"),
        "pv_fused_out_max": _max_metric(samples, "pv_fused_out_max"),
        "pv_fused_lse_max": _max_metric(samples, "pv_fused_lse_max"),
        "pv_fused_impl": first.get("pv_fused_impl", "cute"),
        "compare_mode": first.get("compare_mode", "full"),
        "exact_sfv_direct": bool(first.get("exact_sfv_direct", False)),
    }
    aggregated["qkfast_over_bf16"] = aggregated["qkfast_ms"] / aggregated["bf16_ms"] if math.isfinite(aggregated["bf16_ms"]) else math.nan
    aggregated["pv_fused_over_qkfast"] = aggregated["pv_fused_ms"] / aggregated["qkfast_ms"]
    aggregated["pv_fused_over_bf16"] = aggregated["pv_fused_ms"] / aggregated["bf16_ms"] if math.isfinite(aggregated["bf16_ms"]) else math.nan
    return aggregated


def _run_row_fresh_processes(
    *,
    script_path: pathlib.Path,
    head_dim: int,
    causal: bool,
    seqlen: int,
    batch_size: int,
    num_heads: int,
    num_heads_kv: int,
    device_idx: int,
    warmup: int,
    iters: int,
    skip_baseline_check: bool,
    fresh_runs: int,
    max_attempts: int,
    compare_mode: str,
    pv_tile_mn: tuple[int, int] | None = None,
) -> dict:
    if compare_mode == "profile-exact":
        raise RuntimeError("profile-exact mode does not use the fresh-process row aggregator.")
    successes = []
    failures = []
    while len(successes) < fresh_runs and len(successes) + len(failures) < max_attempts:
        cmd = [
            sys.executable,
            str(script_path),
            "--head-dims",
            str(head_dim),
            "--seqlens",
            str(seqlen),
            "--causal-values",
            "true" if causal else "false",
            "--batch-size",
            str(batch_size),
            "--device",
            str(device_idx),
            "--num-heads",
            str(num_heads),
            "--num-heads-kv",
            str(num_heads_kv),
            "--warmup",
            str(warmup),
            "--iters",
            str(iters),
            "--emit-json",
            "--compare-mode",
            compare_mode,
        ]
        if pv_tile_mn is not None:
            cmd.extend(
                [
                    "--pv-tile-m",
                    str(pv_tile_mn[0]),
                    "--pv-tile-n",
                    str(pv_tile_mn[1]),
                ]
            )
        if skip_baseline_check:
            cmd.append("--skip-baseline-check")
        child = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        if child.returncode == 0:
            stdout_lines = [line for line in child.stdout.splitlines() if line.strip()]
            if not stdout_lines:
                failures.append({"returncode": child.returncode, "stderr": child.stderr.strip(), "stdout": child.stdout.strip()})
                continue
            successes.append(json.loads(stdout_lines[-1]))
        else:
            failures.append(
                {
                    "returncode": child.returncode,
                    "stderr": child.stderr.strip(),
                    "stdout": child.stdout.strip(),
                }
            )
    if len(successes) < fresh_runs:
        failure_summary = failures[-1]["stderr"] if failures else "unknown failure"
        raise RuntimeError(
            f"benchmark_fp4_pv.py needed {fresh_runs} successful fresh-process samples but only got "
            f"{len(successes)} after {len(successes) + len(failures)} attempts. Last failure: {failure_summary}"
        )
    return _aggregate_benchmark_runs(successes, failures)


def main():
    parser = argparse.ArgumentParser(description="Benchmark isolated NVFP4 PV forward against QK-fast and BF16 Cute FA4.")
    parser.add_argument("--head-dims", default="128", help="Comma-separated head dims.")
    parser.add_argument("--seqlens", default="512", help="Comma-separated sequence lengths.")
    parser.add_argument("--causal-values", default="false", help="Comma-separated causal flags.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pv-tile-m", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pv-tile-n", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-heads-kv", type=int, default=DEFAULT_NUM_HEADS_KV)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--fresh-runs", type=int, default=5, help="Required successful fresh-process runs per row.")
    parser.add_argument("--max-attempts", type=int, default=10, help="Maximum subprocess attempts per row before failing.")
    parser.add_argument(
        "--compare-mode",
        choices=("full", "fused-only", "profile-exact"),
        default="full",
        help="Use 'fused-only' for primary perf work so only qkfast vs fused PV are measured in each subprocess.",
    )
    parser.add_argument(
        "--skip-baseline-check",
        action="store_true",
        help="Skip the FP4-vs-BF16 assert so diagnostic rows can still be reported.",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    try:
        _maybe_cuinit()
        if args.device is not None:
            global _BENCH_DEVICE
            _BENCH_DEVICE = args.device
        torch.cuda.set_device(_get_benchmark_device())
        torch.empty(1, device="cuda")
    except Exception as exc:
        raise SystemExit("CUDA is required for benchmark_fp4_pv.py.") from exc
    major, minor = torch.cuda.get_device_capability()
    if major not in (10, 11):
        raise SystemExit(f"FP4 PV benchmark requires SM100/SM110, got {major}.{minor}.")

    head_dims = _parse_csv_ints(args.head_dims)
    seqlens = _parse_csv_ints(args.seqlens)
    causal_values = _parse_csv_bools(args.causal_values)
    if (args.pv_tile_m is None) ^ (args.pv_tile_n is None):
        raise SystemExit("Pass both --pv-tile-m and --pv-tile-n together.")
    pv_tile_mn = (
        (args.pv_tile_m, args.pv_tile_n)
        if args.pv_tile_m is not None and args.pv_tile_n is not None
        else None
    )
    rows = [
        (head_dim, causal, seqlen)
        for head_dim in head_dims
        for causal in causal_values
        for seqlen in seqlens
    ]

    if args.emit_json or args.compare_mode == "profile-exact":
        if len(rows) != 1:
            raise SystemExit("--emit-json expects exactly one benchmark row.")
        head_dim, causal, seqlen = rows[0]
        print(
            json.dumps(
                _benchmark_case(
                    seqlen=seqlen,
                    head_dim=head_dim,
                    causal=causal,
                    batch_size=args.batch_size,
                    num_heads=args.num_heads,
                    num_heads_kv=args.num_heads_kv,
                    device_idx=args.device,
                    warmup=args.warmup,
                    iters=args.iters,
                    skip_baseline_check=args.skip_baseline_check,
                    compare_mode=args.compare_mode,
                    pv_tile_mn=pv_tile_mn,
                )
            )
        )
        return

    script_path = pathlib.Path(__file__).resolve()
    candidate_devices = [args.device] if args.device is not None else _enumerate_usable_devices()
    print(
        "device,kind,hq,hkv,d,causal,seqlen,pv_fused_impl,exact_sfv_direct,success_count,failure_count,"
        "qkfast_ms,pv_fused_ms,bf16_ms,qkfast_over_bf16,pv_fused_over_qkfast,pv_fused_over_bf16,"
        "qkfast_out_max,qkfast_lse_max,pv_fused_out_max,pv_fused_lse_max"
    )
    saw_clean_device = False
    failed_clean_device = False
    for head_dim, causal, seqlen in rows:
        for device_idx in candidate_devices:
            try:
                result = _run_row_fresh_processes(
                    script_path=script_path,
                    head_dim=head_dim,
                    causal=causal,
                    seqlen=seqlen,
                    batch_size=args.batch_size,
                    num_heads=args.num_heads,
                    num_heads_kv=args.num_heads_kv,
                    device_idx=device_idx,
                    warmup=args.warmup,
                    iters=args.iters,
                    skip_baseline_check=args.skip_baseline_check,
                    fresh_runs=args.fresh_runs,
                    max_attempts=args.max_attempts,
                    compare_mode=args.compare_mode,
                    pv_tile_mn=pv_tile_mn,
                )
            except RuntimeError as exc:
                if "bogus-fast" in str(exc):
                    continue
                raise
            saw_clean_device = True
            if result["pv_fused_over_qkfast"] >= 1.0:
                failed_clean_device = True
            print(
                f"{result['device']},{result['kind']},{result['hq']},{result['hkv']},{result['d']},"
                f"{result['causal']},{result['seqlen']},{result['pv_fused_impl']},{result['exact_sfv_direct']},{result['success_count']},{result['failure_count']},"
                f"{result['qkfast_ms']:.5f},{result['pv_fused_ms']:.5f},{result['bf16_ms']:.5f},"
                f"{result['qkfast_over_bf16']:.3f},{result['pv_fused_over_qkfast']:.3f},{result['pv_fused_over_bf16']:.3f},"
                f"{result['qkfast_out_max']:.6f},{result['qkfast_lse_max']:.6f},{result['pv_fused_out_max']:.6f},{result['pv_fused_lse_max']:.6f}"
            )
    if not saw_clean_device:
        raise SystemExit("benchmark_fp4_pv.py did not find any clean GPU for this row.")
    if failed_clean_device:
        raise SystemExit("At least one clean GPU failed the pv_fused_over_qkfast < 1.0 gate.")


if __name__ == "__main__":
    main()
