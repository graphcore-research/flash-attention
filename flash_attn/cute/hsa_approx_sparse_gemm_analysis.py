from __future__ import annotations

import math
import time
from typing import Any

import torch


def _measure_ms(fn, warmup_iters: int, benchmark_iters: int) -> float:
    warmup_iters = max(int(warmup_iters), 0)
    benchmark_iters = max(int(benchmark_iters), 1)
    for _ in range(warmup_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / benchmark_iters


def _prune_dense_matrix_topk_per_group(
    x: torch.Tensor,
    *,
    group_size: int = 4,
    nnz_per_group: int = 2,
) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("expected a rank-2 dense matrix")
    if group_size <= 0 or nnz_per_group <= 0 or nnz_per_group > group_size:
        raise ValueError("invalid semistructured sparsity group configuration")
    rows, cols = x.shape
    if cols % group_size != 0:
        raise ValueError("column dimension must be divisible by group_size")
    view = x.reshape(rows, cols // group_size, group_size)
    keep_idx = torch.topk(view.abs(), k=nnz_per_group, dim=-1, sorted=False).indices
    keep_mask = torch.zeros_like(view, dtype=torch.bool)
    keep_mask.scatter_(-1, keep_idx, True)
    pruned = view.clone()
    pruned.masked_fill_(~keep_mask, 0)
    return pruned.reshape(rows, cols)


def _pad_rows_to_multiple(x: torch.Tensor, row_multiple: int) -> torch.Tensor:
    if row_multiple <= 0:
        raise ValueError("row_multiple must be positive")
    rows = x.shape[0]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    if padded_rows == rows:
        return x.contiguous()
    out = torch.zeros((padded_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    out[:rows] = x
    return out.contiguous()


def _compress_sparse24_operand(x: torch.Tensor):
    from torch.sparse import to_sparse_semi_structured

    last_exc: Exception | None = None
    for row_multiple in (16, 32, 64, 128):
        padded = _pad_rows_to_multiple(x, row_multiple)
        try:
            return to_sparse_semi_structured(padded), padded.shape[0]
        except Exception as exc:  # pragma: no cover - device/library dependent
            last_exc = exc
    if last_exc is None:
        raise RuntimeError("failed to compress sparse24 operand")
    raise RuntimeError(f"failed to compress sparse24 operand: {last_exc}") from last_exc


def _dense_scores(q_hqd: torch.Tensor, k_hkd: torch.Tensor) -> torch.Tensor:
    return torch.bmm(q_hqd, k_hkd.transpose(1, 2))


def _sparse_scores_precompressed(problem: dict[str, Any]) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    k_len = int(problem["k_len"])
    for head_idx, sparse_operand in enumerate(problem["sparse_heads"]):
        rhs = problem["q_rhs_cols"][head_idx]
        head_scores = sparse_operand @ rhs
        scores.append(head_scores[:k_len].transpose(0, 1).contiguous())
    return torch.stack(scores, dim=0)


def _sparse_scores_end_to_end(
    problem: dict[str, Any],
    *,
    group_size: int,
    nnz_per_group: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    k_len = int(problem["k_len"])
    for head_idx in range(problem["k_hkd"].shape[0]):
        pruned = _prune_dense_matrix_topk_per_group(
            problem["k_hkd"][head_idx],
            group_size=group_size,
            nnz_per_group=nnz_per_group,
        )
        sparse_operand, _ = _compress_sparse24_operand(pruned)
        rhs = problem["q_rhs_cols"][head_idx]
        head_scores = sparse_operand @ rhs
        scores.append(head_scores[:k_len].transpose(0, 1).contiguous())
    return torch.stack(scores, dim=0)


def _masked_forward_from_scores(problem: dict[str, Any], scores_hqk: torch.Tensor, *, softmax_scale: float) -> torch.Tensor:
    mask_hqk = problem["mask_hqk"]
    scaled = scores_hqk.float() * float(softmax_scale)
    scaled = scaled.masked_fill(~mask_hqk, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.bmm(probs, problem["v_hkd_float"])


def _select_forward_row_compact_plan(runtime) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    forward_metadata = getattr(runtime, "forward_synthetic_grid", None)
    if forward_metadata is None or forward_metadata.forward_execution_plan is None:
        return None, None, "missing_forward_execution_plan"
    execution_plan = forward_metadata.forward_execution_plan
    direct_plan = execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return None, None, "missing_direct_execution_plan"
    union_plan = direct_plan.get("union_row_compact_plan")
    if union_plan is not None and union_plan.get("row_compact_plan") is not None:
        return union_plan, union_plan["row_compact_plan"], "union_row_compact"
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is not None:
        return direct_plan, row_plan, "direct_row_compact"
    return None, None, "missing_row_compact_plan"


def _materialize_sparse_gemm_microproblems(
    schedule,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group_size: int,
    nnz_per_group: int,
    max_buckets: int | None,
    max_members: int | None,
) -> dict[str, Any]:
    import flash_attn.cute.hsa as hsa_module

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    bucket_plan, row_plan, plan_mode = _select_forward_row_compact_plan(runtime)
    if bucket_plan is None or row_plan is None:
        return {
            "status": plan_mode,
            "problems": [],
            "sampled_buckets": 0,
            "sampled_members": 0,
            "available_buckets": 0,
            "available_members": 0,
            "plan_mode": plan_mode,
        }

    q_flat = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3]).contiguous()
    k_flat = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3]).contiguous()
    v_flat = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3]).contiguous()

    bucket_q_row_range = bucket_plan["bucket_q_row_range"]
    bucket_k_row_range = bucket_plan["bucket_k_row_range"]
    bucket_q_length_range = bucket_plan["bucket_q_length_range"]
    bucket_k_length_range = bucket_plan["bucket_k_length_range"]
    bucket_q_row_idx = bucket_plan["bucket_q_row_idx"]
    bucket_k_row_idx = bucket_plan["bucket_k_row_idx"]
    bucket_q_length = bucket_plan["bucket_q_length"]
    bucket_k_length = bucket_plan["bucket_k_length"]
    bucket_packed_q = bucket_plan["bucket_packed_q"]
    bucket_packed_k = bucket_plan["bucket_packed_k"]

    row_bucket_row_k_cap = row_plan["bucket_row_k_cap"]
    row_bucket_row_k_to_union_range = row_plan["bucket_row_k_to_union_range"]
    row_bucket_row_k_length_range = row_plan["bucket_row_k_length_range"]
    row_bucket_row_k_to_union_idx = row_plan["bucket_row_k_to_union_idx"]
    row_bucket_row_k_length = row_plan["bucket_row_k_length"]

    available_buckets = len(bucket_packed_q)
    available_members = int(bucket_q_length.numel())
    sampled_buckets = 0
    sampled_members = 0
    retained_abs_sum = 0.0
    dense_abs_sum = 0.0
    retained_sq_sum = 0.0
    dense_sq_sum = 0.0
    problems: list[dict[str, Any]] = []

    for bucket_idx in range(available_buckets):
        if max_buckets is not None and sampled_buckets >= max_buckets:
            break
        packed_q = int(bucket_packed_q[bucket_idx])
        packed_k = int(bucket_packed_k[bucket_idx])
        row_k_cap = int(row_bucket_row_k_cap[bucket_idx])
        if packed_q <= 0 or packed_k <= 0 or row_k_cap <= 0:
            continue

        q_row_start, q_row_end = bucket_q_row_range[bucket_idx]
        k_row_start, k_row_end = bucket_k_row_range[bucket_idx]
        q_length_start, q_length_end = bucket_q_length_range[bucket_idx]
        k_length_start, k_length_end = bucket_k_length_range[bucket_idx]
        row_k_to_union_start, row_k_to_union_end = row_bucket_row_k_to_union_range[bucket_idx]
        row_k_length_start, row_k_length_end = row_bucket_row_k_length_range[bucket_idx]

        bucket_size = q_length_end - q_length_start
        if bucket_size <= 0:
            continue
        q_row_idx = bucket_q_row_idx[q_row_start:q_row_end].contiguous().view(bucket_size, packed_q)
        union_k_row_idx = bucket_k_row_idx[k_row_start:k_row_end].contiguous().view(bucket_size, packed_k)
        q_lengths = bucket_q_length[q_length_start:q_length_end].contiguous()
        union_k_lengths = bucket_k_length[k_length_start:k_length_end].contiguous()
        row_k_to_union = row_bucket_row_k_to_union_idx[row_k_to_union_start:row_k_to_union_end].contiguous().view(
            bucket_size, packed_q, row_k_cap
        )
        row_k_lengths = row_bucket_row_k_length[row_k_length_start:row_k_length_end].contiguous().view(
            bucket_size, packed_q
        )

        bucket_sampled = False
        for member_idx in range(bucket_size):
            if max_members is not None and sampled_members >= max_members:
                break
            q_len = int(q_lengths[member_idx])
            k_len = int(union_k_lengths[member_idx])
            if q_len <= 0 or k_len <= 0:
                continue

            q_rows = q_row_idx[member_idx, :q_len].long()
            k_rows = union_k_row_idx[member_idx, :k_len].long()
            if torch.any(q_rows < 0) or torch.any(k_rows < 0):
                continue

            q_sel = q_flat.index_select(0, q_rows).contiguous()
            k_sel = k_flat.index_select(0, k_rows).contiguous()
            v_sel = v_flat.index_select(0, k_rows).contiguous()
            k_expanded = hsa_module._expand_kv_to_q_heads(k_sel, q_sel.shape[1]).contiguous()
            v_expanded = hsa_module._expand_kv_to_q_heads(v_sel, q_sel.shape[1]).contiguous()

            q_hqd = q_sel.permute(1, 0, 2).contiguous()
            k_hkd = k_expanded.permute(1, 0, 2).contiguous()
            v_hkd = v_expanded.permute(1, 0, 2).contiguous()
            if k_hkd.shape[-1] % group_size != 0:
                return {
                    "status": "head_dim_not_divisible_by_group_size",
                    "problems": [],
                    "sampled_buckets": sampled_buckets,
                    "sampled_members": sampled_members,
                    "available_buckets": available_buckets,
                    "available_members": available_members,
                    "plan_mode": plan_mode,
                }

            mask = torch.zeros((q_len, k_len), dtype=torch.bool, device=q.device)
            for q_slot in range(q_len):
                row_len = int(row_k_lengths[member_idx, q_slot])
                if row_len <= 0:
                    continue
                allowed = row_k_to_union[member_idx, q_slot, :row_len].long()
                allowed = allowed[allowed >= 0]
                if allowed.numel() > 0:
                    mask[q_slot, allowed] = True

            sparse_heads = []
            pruned_hkd = torch.empty_like(k_hkd)
            for head_idx in range(k_hkd.shape[0]):
                pruned = _prune_dense_matrix_topk_per_group(
                    k_hkd[head_idx],
                    group_size=group_size,
                    nnz_per_group=nnz_per_group,
                )
                sparse_operand, _ = _compress_sparse24_operand(pruned)
                sparse_heads.append(sparse_operand)
                pruned_hkd[head_idx] = pruned
                retained_abs_sum += pruned.abs().sum().item()
                dense_abs_sum += k_hkd[head_idx].abs().sum().item()
                retained_sq_sum += pruned.float().square().sum().item()
                dense_sq_sum += k_hkd[head_idx].float().square().sum().item()

            problems.append(
                {
                    "q_hqd": q_hqd,
                    "q_rhs_cols": q_hqd.transpose(1, 2).contiguous(),
                    "k_hkd": k_hkd,
                    "v_hkd_float": v_hkd.float().contiguous(),
                    "mask_hqk": mask.unsqueeze(0).expand(q_hqd.shape[0], q_len, k_len).contiguous(),
                    "k_len": k_len,
                    "q_len": q_len,
                    "sparse_heads": sparse_heads,
                }
            )
            sampled_members += 1
            bucket_sampled = True

        if bucket_sampled:
            sampled_buckets += 1
        if max_members is not None and sampled_members >= max_members:
            break

    return {
        "status": "measured" if problems else "no_row_compact_members",
        "problems": problems,
        "sampled_buckets": sampled_buckets,
        "sampled_members": sampled_members,
        "available_buckets": available_buckets,
        "available_members": available_members,
        "plan_mode": plan_mode,
        "retained_abs_sum": retained_abs_sum,
        "dense_abs_sum": dense_abs_sum,
        "retained_sq_sum": retained_sq_sum,
        "dense_sq_sum": dense_sq_sum,
    }


def analyze_hsa_approx_sparse_gemm_forward(
    schedule,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_synthetic_grid: bool = True,
    group_size: int = 4,
    nnz_per_group: int = 2,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    max_buckets: int | None = None,
    max_members: int | None = 256,
) -> dict[str, Any]:
    if not use_synthetic_grid:
        return {
            "status": "requires_synthetic_grid",
            "group_size": group_size,
            "nnz_per_group": nnz_per_group,
        }

    try:
        materialized = _materialize_sparse_gemm_microproblems(
            schedule,
            q,
            k,
            v,
            group_size=group_size,
            nnz_per_group=nnz_per_group,
            max_buckets=max_buckets,
            max_members=max_members,
        )
    except Exception as exc:  # pragma: no cover - benchmark-only failure path
        return {
            "status": f"materialize_failed_{type(exc).__name__}",
            "group_size": group_size,
            "nnz_per_group": nnz_per_group,
        }
    report: dict[str, Any] = {
        "status": materialized["status"],
        "group_size": group_size,
        "nnz_per_group": nnz_per_group,
        "warmup_iters": int(warmup_iters),
        "benchmark_iters": int(benchmark_iters),
        "sampled_buckets": int(materialized.get("sampled_buckets", 0)),
        "sampled_members": int(materialized.get("sampled_members", 0)),
        "available_buckets": int(materialized.get("available_buckets", 0)),
        "available_members": int(materialized.get("available_members", 0)),
        "plan_mode": materialized.get("plan_mode", "unknown"),
    }
    if materialized["status"] != "measured":
        return report

    problems = materialized["problems"]
    if not problems:
        report["status"] = "no_row_compact_members"
        return report

    softmax_scale = q.shape[-1] ** (-0.5)
    avg_q_len = sum(int(problem["q_len"]) for problem in problems) / len(problems)
    avg_k_len = sum(int(problem["k_len"]) for problem in problems) / len(problems)
    max_k_len = max(int(problem["k_len"]) for problem in problems)
    retained_abs_sum = float(materialized["retained_abs_sum"])
    dense_abs_sum = float(materialized["dense_abs_sum"])
    retained_sq_sum = float(materialized["retained_sq_sum"])
    dense_sq_sum = float(materialized["dense_sq_sum"])

    max_diff = 0.0
    sum_abs_diff = 0.0
    sum_abs_dense = 0.0
    total_elements = 0
    for problem in problems:
        dense_scores = _dense_scores(problem["q_hqd"], problem["k_hkd"])
        dense_out = _masked_forward_from_scores(problem, dense_scores, softmax_scale=softmax_scale)
        sparse_scores = _sparse_scores_precompressed(problem)
        sparse_out = _masked_forward_from_scores(problem, sparse_scores, softmax_scale=softmax_scale)
        diff = (dense_out - sparse_out).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        sum_abs_diff += float(diff.sum().item())
        sum_abs_dense += float(dense_out.abs().sum().item())
        total_elements += diff.numel()

    def _run_dense_qk():
        last = None
        for problem in problems:
            last = _dense_scores(problem["q_hqd"], problem["k_hkd"])
        return last

    def _run_sparse_qk_precompressed():
        last = None
        for problem in problems:
            last = _sparse_scores_precompressed(problem)
        return last

    def _run_sparse_qk_end_to_end():
        last = None
        for problem in problems:
            last = _sparse_scores_end_to_end(
                problem,
                group_size=group_size,
                nnz_per_group=nnz_per_group,
            )
        return last

    def _run_dense_fwd():
        last = None
        for problem in problems:
            last = _masked_forward_from_scores(
                problem,
                _dense_scores(problem["q_hqd"], problem["k_hkd"]),
                softmax_scale=softmax_scale,
            )
        return last

    def _run_sparse_fwd_precompressed():
        last = None
        for problem in problems:
            last = _masked_forward_from_scores(
                problem,
                _sparse_scores_precompressed(problem),
                softmax_scale=softmax_scale,
            )
        return last

    def _run_sparse_fwd_end_to_end():
        last = None
        for problem in problems:
            last = _masked_forward_from_scores(
                problem,
                _sparse_scores_end_to_end(
                    problem,
                    group_size=group_size,
                    nnz_per_group=nnz_per_group,
                ),
                softmax_scale=softmax_scale,
            )
        return last

    try:
        dense_qk_ms = _measure_ms(_run_dense_qk, warmup_iters, benchmark_iters)
        sparse_qk_precompressed_ms = _measure_ms(_run_sparse_qk_precompressed, warmup_iters, benchmark_iters)
        sparse_qk_end_to_end_ms = _measure_ms(_run_sparse_qk_end_to_end, warmup_iters, benchmark_iters)
        dense_fwd_ms = _measure_ms(_run_dense_fwd, warmup_iters, benchmark_iters)
        sparse_fwd_precompressed_ms = _measure_ms(_run_sparse_fwd_precompressed, warmup_iters, benchmark_iters)
        sparse_fwd_end_to_end_ms = _measure_ms(_run_sparse_fwd_end_to_end, warmup_iters, benchmark_iters)
    except Exception as exc:  # pragma: no cover - benchmark-only failure path
        report["status"] = f"benchmark_failed_{type(exc).__name__}"
        return report

    report.update(
        {
            "status": "measured",
            "avg_q_len": avg_q_len,
            "avg_union_k_len": avg_k_len,
            "max_union_k_len": max_k_len,
            "dense_qk_ms": dense_qk_ms,
            "sparse_qk_precompressed_ms": sparse_qk_precompressed_ms,
            "sparse_qk_end_to_end_ms": sparse_qk_end_to_end_ms,
            "dense_fwd_ms": dense_fwd_ms,
            "sparse_fwd_precompressed_ms": sparse_fwd_precompressed_ms,
            "sparse_fwd_end_to_end_ms": sparse_fwd_end_to_end_ms,
            "sparse_qk_precompressed_speedup": dense_qk_ms / sparse_qk_precompressed_ms
            if sparse_qk_precompressed_ms > 0
            else float("inf"),
            "sparse_qk_end_to_end_speedup": dense_qk_ms / sparse_qk_end_to_end_ms
            if sparse_qk_end_to_end_ms > 0
            else float("inf"),
            "sparse_fwd_precompressed_speedup": dense_fwd_ms / sparse_fwd_precompressed_ms
            if sparse_fwd_precompressed_ms > 0
            else float("inf"),
            "sparse_fwd_end_to_end_speedup": dense_fwd_ms / sparse_fwd_end_to_end_ms
            if sparse_fwd_end_to_end_ms > 0
            else float("inf"),
            "output_max_diff": max_diff,
            "output_mean_diff": (sum_abs_diff / total_elements) if total_elements > 0 else 0.0,
            "output_relative_l1_error": (sum_abs_diff / sum_abs_dense) if sum_abs_dense > 0 else 0.0,
            "retained_abs_fraction": (retained_abs_sum / dense_abs_sum) if dense_abs_sum > 0 else 0.0,
            "retained_l2_fraction": math.sqrt(retained_sq_sum / dense_sq_sum) if dense_sq_sum > 0 else 0.0,
        }
    )
    return report


def summarize_hsa_approx_sparse_gemm_forward(report: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": str(report.get("status", "unknown")),
        "group_size": int(report.get("group_size", 4)),
        "nnz_per_group": int(report.get("nnz_per_group", 2)),
        "plan_mode": str(report.get("plan_mode", "unknown")),
        "sampled_buckets": int(report.get("sampled_buckets", 0)),
        "sampled_members": int(report.get("sampled_members", 0)),
        "available_buckets": int(report.get("available_buckets", 0)),
        "available_members": int(report.get("available_members", 0)),
        "avg_q_len": float(report.get("avg_q_len", 0.0)),
        "avg_union_k_len": float(report.get("avg_union_k_len", 0.0)),
        "max_union_k_len": int(report.get("max_union_k_len", 0)),
        "dense_qk_ms": float(report.get("dense_qk_ms", float("nan"))),
        "sparse_qk_precompressed_ms": float(report.get("sparse_qk_precompressed_ms", float("nan"))),
        "sparse_qk_end_to_end_ms": float(report.get("sparse_qk_end_to_end_ms", float("nan"))),
        "dense_fwd_ms": float(report.get("dense_fwd_ms", float("nan"))),
        "sparse_fwd_precompressed_ms": float(report.get("sparse_fwd_precompressed_ms", float("nan"))),
        "sparse_fwd_end_to_end_ms": float(report.get("sparse_fwd_end_to_end_ms", float("nan"))),
        "sparse_qk_precompressed_speedup": float(report.get("sparse_qk_precompressed_speedup", float("nan"))),
        "sparse_qk_end_to_end_speedup": float(report.get("sparse_qk_end_to_end_speedup", float("nan"))),
        "sparse_fwd_precompressed_speedup": float(report.get("sparse_fwd_precompressed_speedup", float("nan"))),
        "sparse_fwd_end_to_end_speedup": float(report.get("sparse_fwd_end_to_end_speedup", float("nan"))),
        "output_max_diff": float(report.get("output_max_diff", float("nan"))),
        "output_mean_diff": float(report.get("output_mean_diff", float("nan"))),
        "output_relative_l1_error": float(report.get("output_relative_l1_error", float("nan"))),
        "retained_abs_fraction": float(report.get("retained_abs_fraction", float("nan"))),
        "retained_l2_fraction": float(report.get("retained_l2_fraction", float("nan"))),
    }
    return summary


__all__ = [
    "analyze_hsa_approx_sparse_gemm_forward",
    "summarize_hsa_approx_sparse_gemm_forward",
]
