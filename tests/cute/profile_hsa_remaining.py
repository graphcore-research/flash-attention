import argparse
import ast
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attn.cute import hsa_cached_2d_forward_analysis as cached_2d
from flash_attn.cute.hsa_explicit_2d_sparse_analysis import (
    analyze_explicit_2d_sparse_forward,
    summarize_explicit_2d_sparse_forward,
)


class _FakeCudaTensor:
    is_cuda = True
    dtype = torch.bfloat16

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    return value


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


class _SyntheticGrid:
    def __init__(self, direct_plan: dict[str, Any]):
        self.forward_execution_plan = {"direct_execution_plan": direct_plan}


class _Runtime:
    def __init__(self, direct_plan: dict[str, Any]):
        self.forward_synthetic_grid = _SyntheticGrid(direct_plan)


def _make_representative_runtime() -> _Runtime:
    packed_q = 2
    row_k_cap = 4
    direct_plan = {
        "bucket_size": [1],
        "bucket_packed_q": [packed_q],
        "bucket_packed_k": [row_k_cap],
        "bucket_q_row_range": [(0, packed_q)],
        "bucket_q_row_idx": torch.tensor([0, 1], dtype=torch.int32),
        "row_compact_plan": {
            "bucket_row_k_range": [(0, packed_q * row_k_cap)],
            "bucket_row_k_length_range": [(0, packed_q)],
            "bucket_row_k_cap": [row_k_cap],
            "bucket_row_k_row_idx": torch.tensor(
                [
                    0,
                    1,
                    2,
                    3,
                    1,
                    2,
                    3,
                    4,
                ],
                dtype=torch.int32,
            ),
            "bucket_row_k_length": torch.tensor([4, 4], dtype=torch.int32),
        },
    }
    return _Runtime(direct_plan)


def _profile_cached_payload_cache_probe(*, steps: int, vary_shapes: bool = False) -> dict[str, Any]:
    runtime = _make_representative_runtime()
    cached_2d.reset_cached_direct_2d_forward_payload_cache_stats(runtime)
    steps = max(1, int(steps))
    payload_ids = []
    first = None
    last = None
    for step_idx in range(steps):
        q_rows = 2 + (step_idx % 2 if vary_shapes else 0)
        q = torch.empty((1, q_rows, 2, 64), dtype=torch.bfloat16)
        k = torch.empty((1, 5, 2, 64), dtype=torch.bfloat16)
        v = torch.empty((1, 5, 2, 64), dtype=torch.bfloat16)
        payload = cached_2d.build_cached_direct_2d_forward_payload(runtime, q, k, v)
        if first is None:
            first = payload
        last = payload
        payload_ids.append(id(payload))
    stats = cached_2d.get_cached_direct_2d_forward_payload_cache_stats(runtime)
    return {
        "steps": steps,
        "vary_shapes": bool(vary_shapes),
        "first_status": first.get("status") if isinstance(first, dict) else "missing",
        "last_is_first": last is first,
        "unique_payload_objects": len(set(payload_ids)),
        "stats": stats,
    }


def _base_direct_final_payload() -> dict[str, Any]:
    return {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 5, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 5,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([1, 1], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }


def _direct_final_reason(payload: dict[str, Any]) -> str | None:
    fake = _FakeCudaTensor((int(payload["total_rows"]), 2, 64))
    return cached_2d._cached_direct_final_residual_support_reason(payload, fake, fake, fake)


def _profile_residual_blockers() -> dict[str, Any]:
    within_group = _base_direct_final_payload()
    within_group["range_scatter_q_row_idx"] = torch.tensor(
        [[4, 4, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32,
    )
    within_group["range_scatter_q_length"] = torch.tensor([2], dtype=torch.int32)

    serial_groups = _base_direct_final_payload()
    serial_groups["range_scatter_q_row_idx"] = torch.tensor(
        [
            [4, -1, -1, -1, -1, -1, -1, -1],
            [4, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    serial_ranges = cached_2d._direct_final_residual_group_ranges_without_duplicate_rows(
        serial_groups,
        "range_scatter_q_row_idx",
        "range_scatter_q_length",
        device=torch.device("cpu"),
    )
    within_group_plan = cached_2d._direct_final_residual_dispatch_plan_without_duplicate_rows(
        within_group,
        "range_scatter_q_row_idx",
        "range_scatter_q_length",
        device=torch.device("cpu"),
    )

    with _temporary_env({"FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_DUP_SERIAL_MAX_RANGES": "1"}):
        range_limited_reason = _direct_final_reason(serial_groups)

    return {
        "point": "duplicate rows inside one residual kernel",
        "within_group_reason": _direct_final_reason(within_group),
        "within_group_duplicate_detected": cached_2d._direct_final_has_duplicate_residual_rows_within_kernel(
            within_group,
            torch.device("cpu"),
        ),
        "within_group_serializable": cached_2d._direct_final_can_serialize_duplicate_residual_rows(
            within_group,
            torch.device("cpu"),
        ),
        "within_group_dispatch_plan": within_group_plan,
        "serial_group_reason": _direct_final_reason(serial_groups),
        "serial_group_duplicate_detected": cached_2d._direct_final_has_duplicate_residual_rows_within_kernel(
            serial_groups,
            torch.device("cpu"),
        ),
        "serial_group_serializable": cached_2d._direct_final_can_serialize_duplicate_residual_rows(
            serial_groups,
            torch.device("cpu"),
        ),
        "serial_group_dispatch_ranges": serial_ranges,
        "range_limited_reason": range_limited_reason,
        "status": (
            "fixed_for_cross_group_and_within_group_duplicates_via_serial_dispatch"
            if _direct_final_reason(serial_groups) is None
            and _direct_final_reason(within_group) is None
            else "unexpected_gate_state"
        ),
    }


def _owned_backward_payload(num_rows: int) -> dict[str, torch.Tensor | str]:
    return {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.arange(num_rows, dtype=torch.int32),
        "owned_occurrence_ptr": torch.arange(num_rows + 1, dtype=torch.int32),
        "owned_occurrence_kind": torch.zeros((num_rows,), dtype=torch.int32),
        "owned_occurrence_range_idx": torch.zeros((num_rows,), dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.zeros((num_rows,), dtype=torch.int32),
        "owned_occurrence_col_idx": torch.zeros((num_rows,), dtype=torch.int32),
    }


def _profile_backward_gates() -> dict[str, Any]:
    small_owned = _owned_backward_payload(128)
    large_owned = _owned_backward_payload(129)
    partial_owned = _owned_backward_payload(128)
    small_k = torch.empty((128, 2, 64), dtype=torch.bfloat16)
    large_k = torch.empty((129, 2, 64), dtype=torch.bfloat16)
    partial_k = torch.empty((256, 2, 64), dtype=torch.bfloat16)
    return {
        "point": "backward split and DK/DV ownership",
        "key_owned_ready_small": cached_2d._can_use_cached_backward_key_owned_dkdv(small_owned),
        "key_owned_all_kv_small": cached_2d._cached_backward_key_owned_overwrites_all_kv_rows(small_owned, small_k),
        "key_owned_auto_small": cached_2d._auto_use_cached_backward_key_owned_dkdv(small_owned, small_k),
        "key_owned_auto_large_129_rows": cached_2d._auto_use_cached_backward_key_owned_dkdv(large_owned, large_k),
        "key_owned_all_kv_partial": cached_2d._cached_backward_key_owned_overwrites_all_kv_rows(partial_owned, partial_k),
        "key_owned_auto_partial": cached_2d._auto_use_cached_backward_key_owned_dkdv(partial_owned, partial_k),
        "fused_zero_helper_4096_rows": cached_2d._use_cached_fused_grad_helper(
            "FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO",
            torch.arange(4096, dtype=torch.int32),
        ),
        "fused_zero_helper_65536_rows": cached_2d._use_cached_fused_grad_helper(
            "FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO",
            torch.arange(65536, dtype=torch.int32),
        ),
        "status": "gated_to_small_all_kv_owned; tile_atomic_default_for_common_large_or_partial_cases",
    }


def _profile_online_combine_cast() -> dict[str, Any]:
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 6,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 1,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[4, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([1], dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[5, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_packed_q_length": torch.tensor([1], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }
    requires_online_combine = cached_2d._direct_final_requires_online_combine(
        payload,
        torch.device("cpu"),
        base_source="fused",
    )
    with _temporary_env({"FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE": "0"}):
        disabled_reason = _direct_final_reason(payload)
    with _temporary_env({"FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE": "1"}):
        enabled_reason = _direct_final_reason(payload)
    return {
        "point": "FP32 online combine cast-out",
        "requires_online_combine": requires_online_combine,
        "reason_when_online_combine_disabled": disabled_reason,
        "reason_when_online_combine_enabled": enabled_reason,
        "status": "fp32_semantics_preserved; cast_out_still_required_when_overlap_hits",
    }


def _profile_d128_routing(target_head_dims: list[int], support_values: list[int]) -> dict[str, Any]:
    rows = []
    for head_dim in target_head_dims:
        for support_k in support_values:
            if head_dim == 128 and support_k > 128:
                status = "gated_wide_d128_requires_focused_probe"
            elif head_dim == 128:
                status = "within_default_d128_scalar_cap"
            else:
                status = "d64_or_non_d128"
            rows.append(
                {
                    "head_dim": head_dim,
                    "support_k": support_k,
                    "status": status,
                }
            )
    return {
        "point": "D128/wider support routing",
        "target_rows": rows,
        "wide_d128_count": sum(1 for row in rows if row["status"] == "gated_wide_d128_requires_focused_probe"),
        "status": "counted_target_configs; run optional D128 probe only if wide_d128_count is nonzero",
    }


def _profile_2d_cases(args: argparse.Namespace) -> dict[str, Any]:
    cache_probe = _profile_cached_payload_cache_probe(
        steps=args.cache_probe_steps,
        vary_shapes=args.cache_probe_vary_shapes,
    )
    if args.no_cuda or not torch.cuda.is_available():
        return {
            "point": "2D compact payload construction",
            "status": "skipped_no_cuda",
            "online_payload_build_in_hot_timing": False,
            "cached_payload_cache_probe": cache_probe,
        }
    cases = []
    for seqlen in _parse_int_list(args.seqlens):
        support_k = args.support_k if args.support_k > 0 else (64 if seqlen <= 4096 else 128)
        check_correctness = bool(args.check_correctness and seqlen <= args.correctness_max_seqlen)
        report = analyze_explicit_2d_sparse_forward(
            case_family=args.case_family,
            seqlen=seqlen,
            heads=args.heads,
            head_dim=args.head_dim,
            packed_q=args.packed_q,
            support_k=support_k,
            islands_per_row=args.islands_per_row,
            island_width=args.island_width,
            row_shift=args.row_shift,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            variants=("direct_2d_compact", "fa4_packed"),
            device=torch.device("cuda"),
            seed=args.seed,
            check_correctness=check_correctness,
        )
        summary = summarize_explicit_2d_sparse_forward(report)
        geometry = report.get("geometry", {})
        direct = report.get("results", {}).get("direct_2d_compact", {})
        fa4 = report.get("results", {}).get("fa4_packed", {})
        cases.append(
            {
                "seqlen": seqlen,
                "support_k": support_k,
                "check_correctness": check_correctness,
                "payload_s": report.get("payload_build_excluding_diagnostic_geometry_seconds"),
                "diagnostic_geometry_s": report.get("diagnostic_geometry_seconds"),
                "build_s": report.get("payload_build_seconds"),
                "direct_2d_compact_ms": direct.get("fwd_ms"),
                "fa4_packed_ms": fa4.get("fwd_ms"),
                "speedup_vs_fa4_packed": direct.get("speedup_vs_fa4_packed"),
                "compact_buckets": geometry.get("direct_2d_compact_buckets_compacted"),
                "passthrough_buckets": geometry.get("direct_2d_compact_buckets_passthrough"),
                "avg_union_k": geometry.get("direct_2d_compact_avg_union_k"),
                "max_union_k": geometry.get("direct_2d_compact_max_union_k"),
                "best_variant": summary.get("best_variant", {}).get("name"),
            }
        )
    return {
        "point": "2D compact payload construction",
        "status": "measured_representative_cases",
        "online_payload_build_in_hot_timing": False,
        "cached_payload_cache_probe": cache_probe,
        "cases": cases,
    }


def _profile_arhsa(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        "tests/cute/benchmark_arhsa_walk.py",
        "--n-queries",
        str(args.arhsa_queries),
        "--n-heads",
        str(args.arhsa_heads),
        "--head-dim-v",
        "64",
        "--leaves-per-query",
        str(args.arhsa_leaves_per_query),
        "--n-iters",
        "3",
        "--graph-mode",
        "level_dag",
        "--level-range-kernels",
        "--incoming-packed-step",
        "--query-warp-readout",
        "--query-warp-fused-bwd",
        "--skip-torch",
        "--skip-custom-fwd-bwd",
        "--no-check",
        "--no-memory",
        "--iters",
        str(args.arhsa_iters),
        "--warmup",
        str(args.arhsa_warmup),
    ]
    if args.arhsa_reuse_forward_denom:
        command.append("--reuse-forward-denom")
    result: dict[str, Any] = {
        "point": "AR-HSA readout backward",
        "denom_retained_by_caller": bool(args.arhsa_reuse_forward_denom),
        "denom_reuse_path_available": True,
        "command": command,
    }
    if not args.include_arhsa_probe:
        result["status"] = "probe_not_run"
        return result
    if args.no_cuda or not torch.cuda.is_available():
        result["status"] = "skipped_no_cuda"
        return result
    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.arhsa_timeout_s,
        check=False,
    )
    result["returncode"] = proc.returncode
    result["status"] = "measured" if proc.returncode == 0 else "failed"
    result["stdout_tail"] = proc.stdout[-4000:]
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            result["parsed"] = parsed
            break
    return result


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile remaining HSA fusion gates and fallback frequencies.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-cuda", action="store_true", help="Skip CUDA timing probes and report gate counters only.")
    parser.add_argument("--seqlens", default="4096", help="Comma-separated 2D sequence lengths.")
    parser.add_argument("--case-family", choices=("disjoint_confetti", "compact_control"), default="compact_control")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--packed-q", type=int, default=16)
    parser.add_argument("--support-k", type=int, default=-1)
    parser.add_argument("--islands-per-row", type=int, default=8)
    parser.add_argument("--island-width", type=int, default=4)
    parser.add_argument("--row-shift", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--benchmark-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--correctness-max-seqlen", type=int, default=4096)
    parser.add_argument("--cache-probe-steps", type=int, default=8)
    parser.add_argument("--cache-probe-vary-shapes", action="store_true")
    parser.add_argument("--target-head-dims", default="64,128")
    parser.add_argument("--target-support-k", default="64,128,512")
    parser.add_argument("--include-arhsa-probe", action="store_true")
    parser.add_argument("--arhsa-queries", type=int, default=16384)
    parser.add_argument("--arhsa-heads", type=int, default=4)
    parser.add_argument("--arhsa-leaves-per-query", type=int, default=4)
    parser.add_argument("--arhsa-iters", type=int, default=3)
    parser.add_argument("--arhsa-warmup", type=int, default=1)
    parser.add_argument("--arhsa-timeout-s", type=int, default=180)
    parser.add_argument("--arhsa-reuse-forward-denom", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _make_parser().parse_args(argv)
    report = {
        "residual_duplicate_rows": _profile_residual_blockers(),
        "compact_payload": _profile_2d_cases(args),
        "backward_split": _profile_backward_gates(),
        "online_combine_cast": _profile_online_combine_cast(),
        "d128_routing": _profile_d128_routing(
            _parse_int_list(args.target_head_dims),
            _parse_int_list(args.target_support_k),
        ),
        "arhsa_readout_backward": _profile_arhsa(args),
    }
    if args.json:
        print(json.dumps(_jsonable(report), sort_keys=True))
    else:
        for key, payload in report.items():
            print(f"{key}: {payload.get('status', 'unknown')}")
            if key == "compact_payload":
                for case in payload.get("cases", []):
                    print(
                        "  "
                        f"seq={case['seqlen']} payload_s={case['payload_s']:.3f} "
                        f"direct_ms={case['direct_2d_compact_ms']:.3f} "
                        f"fa4_ms={case['fa4_packed_ms']:.3f} "
                        f"speedup={case['speedup_vs_fa4_packed']:.2f}x"
                    )
    return report


if __name__ == "__main__":
    main()
