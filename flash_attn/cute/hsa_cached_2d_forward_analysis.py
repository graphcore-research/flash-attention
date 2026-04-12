from __future__ import annotations

from typing import Any

import torch

from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
    _can_use_synthetic_2d_masked_fwd,
    _run_synthetic_combine_scatter_rows_kernel,
    _run_synthetic_2d_masked_fwd_kernel,
    _run_synthetic_2d_masked_gather_fwd_kernel,
    _run_synthetic_2d_masked_gather_scatter_fwd_kernel,
    _run_synthetic_pack_kv_rows_kernel,
    _run_synthetic_pack_rows_kernel,
)
from flash_attn.cute.hsa_explicit_2d_sparse_analysis import _partition_packed_rows_by_support
from flash_attn.cute.hsa_shared_sparse_gemm_analysis import _encode_mask_rows_to_words


def _flatten_row_tensor(rows: torch.Tensor) -> torch.Tensor:
    if rows.ndim == 4:
        return rows.reshape(-1, rows.shape[2], rows.shape[3]).contiguous()
    if rows.ndim == 3:
        return rows.contiguous()
    raise ValueError(f"expected rank-3 or rank-4 row tensor, got shape {tuple(rows.shape)}")


def _extract_bucket_live_row_supports(
    direct_plan: dict[str, Any],
    row_plan: dict[str, Any],
    bucket_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    bucket_size = int(direct_plan["bucket_size"][bucket_idx])
    packed_q = int(direct_plan["bucket_packed_q"][bucket_idx])
    packed_k = int(direct_plan["bucket_packed_k"][bucket_idx])
    if bucket_size <= 0 or packed_q <= 0:
        empty_rows = torch.empty((0,), dtype=torch.int32)
        empty_support = torch.empty((0, 0), dtype=torch.int32)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)
        return empty_rows, empty_support, empty_mask, 0, packed_k

    q_row_start, q_row_end = direct_plan["bucket_q_row_range"][bucket_idx]
    q_row_idx = direct_plan["bucket_q_row_idx"][q_row_start:q_row_end].contiguous().view(bucket_size, packed_q)
    if "bucket_q_length_range" in direct_plan and "bucket_q_length" in direct_plan:
        q_length_start, q_length_end = direct_plan["bucket_q_length_range"][bucket_idx]
        q_length = direct_plan["bucket_q_length"][q_length_start:q_length_end].contiguous().view(bucket_size)
        q_slot_valid = torch.arange(packed_q, device=q_row_idx.device).unsqueeze(0) < q_length.unsqueeze(1)
    else:
        q_slot_valid = torch.ones((bucket_size, packed_q), dtype=torch.bool, device=q_row_idx.device)

    row_k_start, row_k_end = row_plan["bucket_row_k_range"][bucket_idx]
    row_k_length_start, row_k_length_end = row_plan["bucket_row_k_length_range"][bucket_idx]
    row_k_cap = int(row_plan["bucket_row_k_cap"][bucket_idx])
    row_k_row_idx = row_plan["bucket_row_k_row_idx"][row_k_start:row_k_end].contiguous().view(
        bucket_size, packed_q, row_k_cap
    )
    row_k_length = row_plan["bucket_row_k_length"][row_k_length_start:row_k_length_end].contiguous().view(
        bucket_size, packed_q
    )

    q_rows_flat = q_row_idx.reshape(-1).to(dtype=torch.int32)
    support_rows_flat = row_k_row_idx.reshape(-1, row_k_cap).to(dtype=torch.int32)
    support_lengths_flat = row_k_length.reshape(-1).to(dtype=torch.int32)
    valid_q_mask = torch.logical_and(q_rows_flat >= 0, q_slot_valid.reshape(-1))
    positive_length_mask = support_lengths_flat > 0
    zero_support_rows_count = int(torch.logical_and(valid_q_mask, torch.logical_not(positive_length_mask)).sum().item())
    live_mask = torch.logical_and(valid_q_mask, positive_length_mask)
    if not bool(live_mask.any().item()):
        empty_rows = torch.empty((0,), dtype=torch.int32, device=q_row_idx.device)
        empty_support = torch.empty((0, row_k_cap), dtype=torch.int32, device=q_row_idx.device)
        empty_mask = torch.empty((0, row_k_cap), dtype=torch.bool, device=q_row_idx.device)
        return empty_rows, empty_support, empty_mask, zero_support_rows_count, packed_k

    live_q_rows = q_rows_flat[live_mask].contiguous()
    live_support_rows = support_rows_flat[live_mask].contiguous()
    live_support_lengths = support_lengths_flat[live_mask].contiguous()
    support_valid = torch.arange(row_k_cap, device=q_row_idx.device).unsqueeze(0) < live_support_lengths.unsqueeze(1)
    support_valid = torch.logical_and(support_valid, live_support_rows >= 0)
    nonempty_live_mask = torch.any(support_valid, dim=1)
    zero_support_rows_count += int(torch.logical_not(nonempty_live_mask).sum().item())
    return (
        live_q_rows[nonempty_live_mask].contiguous(),
        live_support_rows[nonempty_live_mask].contiguous(),
        support_valid[nonempty_live_mask].contiguous(),
        zero_support_rows_count,
        packed_k,
    )


def _group_rows_by_support_patterns(bucket_mask: torch.Tensor, *, max_rows_per_group: int) -> list[list[int]]:
    valid_q = int(bucket_mask.shape[0])
    if valid_q <= 0:
        return []
    if valid_q <= max_rows_per_group:
        return [list(range(valid_q))]

    unique_patterns, inverse = torch.unique(bucket_mask, dim=0, sorted=False, return_inverse=True)
    if int(unique_patterns.shape[0]) == valid_q:
        return _partition_packed_rows_by_support(bucket_mask, max_rows_per_group=max_rows_per_group)

    inverse_rows = inverse.detach().cpu().tolist()
    pattern_order: list[int] = []
    seen_patterns: set[int] = set()
    for pattern_idx in inverse_rows:
        pattern_idx = int(pattern_idx)
        if pattern_idx in seen_patterns:
            continue
        seen_patterns.add(pattern_idx)
        pattern_order.append(pattern_idx)
    ordered_patterns = unique_patterns.index_select(
        0,
        torch.tensor(pattern_order, dtype=torch.long, device=bucket_mask.device),
    )
    pattern_groups = _partition_packed_rows_by_support(
        ordered_patterns,
        max_rows_per_group=min(max_rows_per_group, int(ordered_patterns.shape[0])),
    )
    old_to_new = {pattern_idx: new_idx for new_idx, pattern_idx in enumerate(pattern_order)}
    rows_by_pattern: list[list[int]] = [[] for _ in range(int(ordered_patterns.shape[0]))]
    for row_idx, pattern_idx in enumerate(inverse_rows):
        rows_by_pattern[old_to_new[int(pattern_idx)]].append(row_idx)

    row_groups: list[list[int]] = []
    for pattern_group in pattern_groups:
        pattern_rows: list[int] = []
        for pattern_idx in pattern_group:
            pattern_rows.extend(rows_by_pattern[pattern_idx])
        for row_start in range(0, len(pattern_rows), max_rows_per_group):
            row_group = pattern_rows[row_start : row_start + max_rows_per_group]
            if row_group:
                row_groups.append(row_group)
    return row_groups


def _group_rows_by_support_span(
    live_support_rows: torch.Tensor,
    live_support_valid: torch.Tensor,
    *,
    max_rows_per_group: int,
) -> list[list[int]]:
    live_row_count = int(live_support_rows.shape[0])
    if live_row_count <= 0:
        return []
    if live_row_count <= max_rows_per_group:
        return [list(range(live_row_count))]

    support_min = torch.where(
        live_support_valid,
        live_support_rows,
        torch.iinfo(live_support_rows.dtype).max,
    ).amin(dim=1)
    support_max = torch.where(
        live_support_valid,
        live_support_rows,
        torch.iinfo(live_support_rows.dtype).min,
    ).amax(dim=1)
    sort_key = support_min.to(dtype=torch.int64) * (1 << 32) + support_max.to(dtype=torch.int64)
    ordered_rows = torch.argsort(sort_key, stable=True).detach().cpu().tolist()
    return [
        ordered_rows[row_start : row_start + max_rows_per_group]
        for row_start in range(0, len(ordered_rows), max_rows_per_group)
    ]


def _group_support_lists_by_span(
    support_lists: list[list[int]],
    *,
    max_rows_per_group: int,
    max_union_k: int,
) -> list[list[int]]:
    valid_indices = [idx for idx, support in enumerate(support_lists) if support]
    if not valid_indices:
        return []
    if len(valid_indices) <= max_rows_per_group and len(set().union(*(set(support_lists[idx]) for idx in valid_indices))) <= max_union_k:
        return [valid_indices]

    ordered_rows = sorted(
        valid_indices,
        key=lambda idx: (support_lists[idx][0], support_lists[idx][-1], len(support_lists[idx]), idx),
    )
    row_groups: list[list[int]] = []
    current_group: list[int] = []
    current_union: set[int] = set()
    for row_idx in ordered_rows:
        support_set = set(support_lists[row_idx])
        candidate_union_size = len(current_union | support_set)
        if current_group and (len(current_group) >= max_rows_per_group or candidate_union_size > max_union_k):
            row_groups.append(current_group)
            current_group = [row_idx]
            current_union = set(support_set)
            continue
        current_group.append(row_idx)
        current_union.update(support_set)
    if current_group:
        row_groups.append(current_group)
    return row_groups


def _append_group_entries(
    *,
    device: torch.device,
    q_rows: list[int],
    support_lists: list[list[int]],
    row_groups: list[list[int]],
    group_q_rows: list[torch.Tensor],
    group_k_rows: list[torch.Tensor],
    group_masks: list[torch.Tensor],
    group_fill: list[float],
):
    for row_group in row_groups:
        if not row_group:
            continue
        union_rows = sorted({support_row for row_idx in row_group for support_row in support_lists[row_idx]})
        union_k = len(union_rows)
        if union_k <= 0:
            continue
        union_pos = {row_idx: pos for pos, row_idx in enumerate(union_rows)}
        mask = torch.zeros((len(row_group), union_k), dtype=torch.bool, device=device)
        for row_pos, entry_idx in enumerate(row_group):
            for support_row in support_lists[entry_idx]:
                mask[row_pos, union_pos[support_row]] = True
        q_tensor = torch.tensor([q_rows[entry_idx] for entry_idx in row_group], dtype=torch.int32, device=device)
        k_tensor = torch.tensor(union_rows, dtype=torch.int32, device=device)
        group_q_rows.append(q_tensor.contiguous())
        group_k_rows.append(k_tensor.contiguous())
        group_masks.append(mask.contiguous())
        group_fill.append(float(mask.sum().item()) / max(1, len(row_group) * union_k))


def _build_range_execution_metadata(
    group_q_rows: list[torch.Tensor],
    combine_group_ranges: list[tuple[int, int]],
) -> list[dict[str, int | bool]]:
    row_counts: dict[int, int] = {}
    for q_rows in group_q_rows:
        for q_row in q_rows.detach().cpu().tolist():
            q_row_int = int(q_row)
            if q_row_int < 0:
                continue
            row_counts[q_row_int] = row_counts.get(q_row_int, 0) + 1

    range_execution: list[dict[str, int | bool]] = []
    for group_start, group_end in combine_group_ranges:
        range_rows: list[int] = []
        for group_idx in range(int(group_start), int(group_end)):
            range_rows.extend(
                int(q_row)
                for q_row in group_q_rows[group_idx].detach().cpu().tolist()
                if int(q_row) >= 0
            )
        scatter_only = bool(range_rows) and all(row_counts.get(q_row, 0) == 1 for q_row in range_rows)
        range_execution.append(
            {
                "group_start": int(group_start),
                "group_end": int(group_end),
                "scatter_only": scatter_only,
            }
        )
    return range_execution


def build_cached_direct_2d_forward_payload(
    runtime,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    max_rows_per_group: int = 16,
    max_merged_support_rows: int = 128,
    max_merged_support_growth_ratio: float = 999.0,
    max_merged_support_increase: int = 1_000_000,
) -> dict[str, Any]:
    q_flat = _flatten_row_tensor(q)
    k_flat = _flatten_row_tensor(k)
    v_flat = _flatten_row_tensor(v)
    metadata = None if runtime is None else getattr(runtime, "forward_synthetic_grid", None)
    if metadata is None or getattr(metadata, "forward_execution_plan", None) is None:
        return {"status": "not_applicable", "reason": "cached schedule is missing synthetic-grid forward metadata"}
    direct_plan = metadata.forward_execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return {"status": "not_applicable", "reason": "cached schedule does not expose a direct execution plan"}
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is None:
        return {"status": "not_applicable", "reason": "cached direct execution plan is missing a row-compact plan"}
    if max_rows_per_group <= 0:
        raise ValueError("max_rows_per_group must be positive")

    device = q_flat.device
    group_q_rows: list[torch.Tensor] = []
    group_k_rows: list[torch.Tensor] = []
    group_masks: list[torch.Tensor] = []
    group_fill: list[float] = []
    combine_group_ranges: list[tuple[int, int]] = []
    bucket_entries: list[dict[str, Any]] = []
    qgroup_bucket_entries: dict[int, list[tuple[int, list[int]]]] = {}
    baseline_packed_k_sum = 0
    baseline_live_row_count = 0
    zero_support_rows_count = 0
    total_live_pairs = 0
    active_bucket_count = 0
    merged_row_count = 0
    fallback_row_count = 0
    merged_qgroup_bucket_count = 0
    bucket_qgroup_bucket_idx = direct_plan.get("bucket_qgroup_bucket_idx")

    for bucket_idx in range(len(direct_plan["bucket_size"])):
        live_q_rows, live_support_rows, live_support_valid, zero_support_rows, packed_k = _extract_bucket_live_row_supports(
            direct_plan,
            row_plan,
            bucket_idx,
        )
        zero_support_rows_count += int(zero_support_rows)
        live_row_count = int(live_q_rows.numel())
        if live_row_count <= 0:
            continue
        active_bucket_count += 1
        baseline_packed_k_sum += packed_k * live_row_count
        baseline_live_row_count += live_row_count
        total_live_pairs += int(live_support_valid.sum().item())
        live_q_rows_cpu = live_q_rows.detach().cpu().tolist()
        live_support_rows_cpu = live_support_rows.detach().cpu().tolist()
        live_support_valid_cpu = live_support_valid.detach().cpu().tolist()
        support_lists = [
            [int(support_row) for support_row, valid in zip(row_supports, row_valid, strict=True) if valid and int(support_row) >= 0]
            for row_supports, row_valid in zip(live_support_rows_cpu, live_support_valid_cpu, strict=True)
        ]
        q_rows = [int(q_row) for q_row in live_q_rows_cpu]
        qgroup_bucket_idx = bucket_idx
        if bucket_qgroup_bucket_idx is not None:
            qgroup_bucket_idx = int(bucket_qgroup_bucket_idx[bucket_idx])
        bucket_entries.append(
            {
                "bucket_idx": bucket_idx,
                "qgroup_bucket_idx": qgroup_bucket_idx,
                "q_rows": q_rows,
                "support_lists": support_lists,
            }
        )
        qgroup_entries = qgroup_bucket_entries.setdefault(qgroup_bucket_idx, [])
        qgroup_entries.extend(zip(q_rows, support_lists, strict=True))

    unmerged_rows_by_qgroup: dict[int, set[int]] = {}
    for qgroup_bucket_idx in sorted(qgroup_bucket_entries):
        row_support_map: dict[int, set[int]] = {}
        row_max_bucket_support: dict[int, int] = {}
        for q_row, support_list in qgroup_bucket_entries[qgroup_bucket_idx]:
            if not support_list:
                continue
            row_support_map.setdefault(int(q_row), set()).update(int(support_row) for support_row in support_list)
            row_max_bucket_support[int(q_row)] = max(
                row_max_bucket_support.get(int(q_row), 0),
                len(support_list),
            )
        merged_q_rows: list[int] = []
        merged_support_lists: list[list[int]] = []
        unmerged_rows: set[int] = set()
        for q_row in sorted(row_support_map):
            support_list = sorted(row_support_map[q_row])
            max_bucket_support = max(1, int(row_max_bucket_support.get(q_row, len(support_list))))
            merged_support_len = len(support_list)
            growth_ratio = merged_support_len / max_bucket_support
            support_increase = merged_support_len - max_bucket_support
            if (
                merged_support_len > max_merged_support_rows
                or growth_ratio > max_merged_support_growth_ratio
                or support_increase > max_merged_support_increase
            ):
                unmerged_rows.add(int(q_row))
                continue
            merged_q_rows.append(int(q_row))
            merged_support_lists.append(support_list)
        if merged_q_rows:
            group_start = len(group_q_rows)
            row_groups = _group_support_lists_by_span(
                merged_support_lists,
                max_rows_per_group=min(max_rows_per_group, len(merged_q_rows)),
                max_union_k=max_merged_support_rows,
            )
            _append_group_entries(
                device=device,
                q_rows=merged_q_rows,
                support_lists=merged_support_lists,
                row_groups=row_groups,
                group_q_rows=group_q_rows,
                group_k_rows=group_k_rows,
                group_masks=group_masks,
                group_fill=group_fill,
            )
            if len(group_q_rows) > group_start:
                combine_group_ranges.append((group_start, len(group_q_rows)))
                merged_qgroup_bucket_count += 1
                merged_row_count += len(merged_q_rows)
        if unmerged_rows:
            unmerged_rows_by_qgroup[qgroup_bucket_idx] = unmerged_rows

    for bucket_entry in bucket_entries:
        unmerged_rows = unmerged_rows_by_qgroup.get(bucket_entry["qgroup_bucket_idx"])
        if not unmerged_rows:
            continue
        fallback_q_rows = [
            q_row
            for q_row in bucket_entry["q_rows"]
            if q_row in unmerged_rows
        ]
        if not fallback_q_rows:
            continue
        fallback_support_lists = [
            support_list
            for q_row, support_list in zip(bucket_entry["q_rows"], bucket_entry["support_lists"], strict=True)
            if q_row in unmerged_rows and support_list
        ]
        fallback_q_rows = [
            q_row
            for q_row, support_list in zip(bucket_entry["q_rows"], bucket_entry["support_lists"], strict=True)
            if q_row in unmerged_rows and support_list
        ]
        if not fallback_q_rows:
            continue
        group_start = len(group_q_rows)
        row_groups = _group_support_lists_by_span(
            fallback_support_lists,
            max_rows_per_group=min(max_rows_per_group, len(fallback_q_rows)),
            max_union_k=max_merged_support_rows,
        )
        _append_group_entries(
            device=device,
            q_rows=fallback_q_rows,
            support_lists=fallback_support_lists,
            row_groups=row_groups,
            group_q_rows=group_q_rows,
            group_k_rows=group_k_rows,
            group_masks=group_masks,
            group_fill=group_fill,
        )
        if len(group_q_rows) > group_start:
            combine_group_ranges.append((group_start, len(group_q_rows)))
            fallback_row_count += len(fallback_q_rows)

    group_count = len(group_q_rows)
    if group_count <= 0:
        return {"status": "not_applicable", "reason": "cached direct buckets did not yield any live 2D forward groups"}

    rows_per_group = max(int(group_q_rows[group_idx].numel()) for group_idx in range(group_count))
    max_union_k = max(int(group_k_rows[group_idx].numel()) for group_idx in range(group_count))
    q_row_idx = torch.full((group_count, rows_per_group), -1, dtype=torch.int32, device=device)
    k_row_idx = torch.full((group_count, max_union_k), -1, dtype=torch.int32, device=device)
    mask_bool = torch.zeros((group_count, rows_per_group, max_union_k), dtype=torch.bool, device=device)
    q_length = torch.zeros((group_count,), dtype=torch.int32, device=device)
    k_length = torch.zeros((group_count,), dtype=torch.int32, device=device)

    for group_idx in range(group_count):
        row_count = int(group_q_rows[group_idx].numel())
        union_k = int(group_k_rows[group_idx].numel())
        q_row_idx[group_idx, :row_count] = group_q_rows[group_idx]
        k_row_idx[group_idx, :union_k] = group_k_rows[group_idx]
        mask_bool[group_idx, :row_count, :union_k] = group_masks[group_idx]
        q_length[group_idx] = row_count
        k_length[group_idx] = union_k

    mask_words = _encode_mask_rows_to_words(mask_bool.view(group_count * rows_per_group, max_union_k)).view(
        group_count,
        rows_per_group,
        -1,
    )
    range_execution = _build_range_execution_metadata(group_q_rows, combine_group_ranges)
    scatter_only_ranges = sum(1 for entry in range_execution if bool(entry["scatter_only"]))
    scatter_only_rows = sum(
        int(payload_q_rows.numel())
        for group_idx, payload_q_rows in enumerate(group_q_rows)
        if any(
            int(entry["group_start"]) <= group_idx < int(entry["group_end"]) and bool(entry["scatter_only"])
            for entry in range_execution
        )
    )
    avg_union_k = float(k_length.float().mean().item()) if group_count > 0 else 0.0
    avg_baseline_k = float(baseline_packed_k_sum / baseline_live_row_count) if baseline_live_row_count > 0 else 0.0
    total_group_area = sum(
        int(q_length[group_idx].item()) * int(k_length[group_idx].item()) for group_idx in range(group_count)
    )
    geometry = {
        "cached_direct_2d_buckets": active_bucket_count,
        "cached_direct_2d_groups": group_count,
        "cached_direct_2d_rows_per_group": rows_per_group,
        "cached_direct_2d_avg_union_k": avg_union_k,
        "cached_direct_2d_max_union_k": max_union_k,
        "cached_direct_2d_avg_group_fill": float(sum(group_fill) / len(group_fill)) if group_fill else 0.0,
        "cached_direct_2d_support_reduction": (
            0.0 if avg_baseline_k <= 0.0 else 1.0 - (avg_union_k / avg_baseline_k)
        ),
        "cached_direct_2d_zero_support_rows": zero_support_rows_count,
        "cached_direct_2d_live_pairs": total_live_pairs,
        "cached_direct_2d_case_fill_rate": total_live_pairs / max(1, total_group_area),
        "cached_direct_2d_cross_bucket_merged_rows": merged_row_count,
        "cached_direct_2d_cross_bucket_fallback_rows": fallback_row_count,
        "cached_direct_2d_cross_bucket_merged_qgroup_buckets": merged_qgroup_bucket_count,
        "cached_direct_2d_scatter_only_ranges": scatter_only_ranges,
        "cached_direct_2d_scatter_only_rows": scatter_only_rows,
    }
    return {
        "status": "ready",
        "reason": "cached direct buckets were repacked into 2D forward groups",
        "packed_q": rows_per_group,
        "support_rows": max_union_k,
        "q_row_idx": q_row_idx.contiguous(),
        "k_row_idx": k_row_idx.contiguous(),
        "mask_words": mask_words.contiguous(),
        "mask_bool": mask_bool.contiguous(),
        "q_length": q_length.contiguous(),
        "k_length": k_length.contiguous(),
        "total_rows": int(q_flat.shape[0]),
        "combine_group_ranges": combine_group_ranges,
        "range_execution": range_execution,
        "geometry": geometry,
        "_workspace": {},
    }


def _get_cached_direct_2d_buffers(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    workspace = payload.setdefault("_workspace", {})
    key = (
        str(q_flat.device),
        q_flat.dtype,
        k_flat.dtype,
        v_flat.dtype,
        q_flat.shape[1],
        q_flat.shape[2],
        k_flat.shape[1],
        k_flat.shape[2],
        v_flat.shape[1],
        v_flat.shape[2],
        int(payload["packed_q"]),
        int(payload["support_rows"]),
        int(payload["q_row_idx"].shape[0]),
        int(payload["total_rows"]),
    )
    buffers = workspace.get(key)
    if buffers is None:
        group_count = int(payload["q_row_idx"].shape[0])
        packed_q = int(payload["packed_q"])
        support_rows = int(payload["support_rows"])
        buffers = (
            torch.empty((group_count * packed_q, q_flat.shape[1], q_flat.shape[2]), dtype=q_flat.dtype, device=q_flat.device),
            torch.empty((group_count * support_rows, k_flat.shape[1], k_flat.shape[2]), dtype=k_flat.dtype, device=k_flat.device),
            torch.empty((group_count * support_rows, v_flat.shape[1], v_flat.shape[2]), dtype=v_flat.dtype, device=v_flat.device),
            torch.zeros((int(payload["total_rows"]), q_flat.shape[1], v_flat.shape[2]), dtype=torch.float32, device=v_flat.device),
            torch.empty((int(payload["total_rows"]), q_flat.shape[1]), dtype=torch.float32, device=v_flat.device),
        )
        workspace[key] = buffers
    return buffers


def run_cached_direct_2d_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    if payload.get("status") != "ready":
        reason = str(payload.get("reason", "cached direct 2D payload is unavailable"))
        raise RuntimeError(reason)

    q_flat = _flatten_row_tensor(q)
    k_flat = _flatten_row_tensor(k)
    v_flat = _flatten_row_tensor(v)
    packed_q = int(payload["packed_q"])
    packed_k = int(payload["support_rows"])
    if softmax_scale is None:
        softmax_scale = q_flat.shape[-1] ** (-0.5)
    if not _can_use_synthetic_2d_masked_fwd(q_flat, k_flat, v_flat, packed_q=packed_q, packed_k=packed_k):
        raise RuntimeError("cached_direct_2d_forward_unsupported")

    q_buf_flat, k_buf_flat, v_buf_flat, out_flat, lse_flat = _get_cached_direct_2d_buffers(payload, q_flat, k_flat, v_flat)
    out_flat.zero_()
    lse_flat.fill_(float("-inf"))
    group_count = int(payload["q_row_idx"].shape[0])
    range_execution = payload.get("range_execution")
    if not range_execution:
        range_execution = [
            {
                "group_start": 0,
                "group_end": group_count,
                "scatter_only": False,
            }
        ]

    def _run_range_packed(group_start: int, group_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        range_group_count = group_end - group_start
        try:
            return _run_synthetic_2d_masked_gather_fwd_kernel(
                q_flat,
                k_flat,
                v_flat,
                payload["q_row_idx"][group_start:group_end],
                payload["k_row_idx"][group_start:group_end],
                payload["q_length"][group_start:group_end],
                payload["k_length"][group_start:group_end],
                payload["mask_words"][group_start:group_end],
                softmax_scale=float(softmax_scale),
            )
        except Exception:
            q_buf_range = q_buf_flat[: range_group_count * packed_q]
            k_buf_range = k_buf_flat[: range_group_count * packed_k]
            v_buf_range = v_buf_flat[: range_group_count * packed_k]
            _run_synthetic_pack_rows_kernel(
                q_flat,
                payload["q_row_idx"][group_start:group_end].reshape(-1).contiguous(),
                q_buf_range,
            )
            _run_synthetic_pack_kv_rows_kernel(
                k_flat,
                v_flat,
                payload["k_row_idx"][group_start:group_end].reshape(-1).contiguous(),
                k_buf_range,
                v_buf_range,
            )
            q_buf = q_buf_range.view(range_group_count, packed_q, q_flat.shape[1], q_flat.shape[2])
            k_buf = k_buf_range.view(range_group_count, packed_k, k_flat.shape[1], k_flat.shape[2])
            v_buf = v_buf_range.view(range_group_count, packed_k, v_flat.shape[1], v_flat.shape[2])
            return _run_synthetic_2d_masked_fwd_kernel(
                q_buf,
                k_buf,
                v_buf,
                payload["q_length"][group_start:group_end],
                payload["k_length"][group_start:group_end],
                payload["mask_words"][group_start:group_end],
                softmax_scale=float(softmax_scale),
            )

    for range_entry in range_execution:
        group_start = int(range_entry["group_start"])
        group_end = int(range_entry["group_end"])
        group_start = int(group_start)
        group_end = int(group_end)
        if group_end <= group_start:
            continue
        if bool(range_entry.get("scatter_only")):
            try:
                _run_synthetic_2d_masked_gather_scatter_fwd_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    payload["q_row_idx"][group_start:group_end],
                    payload["k_row_idx"][group_start:group_end],
                    payload["q_length"][group_start:group_end],
                    payload["k_length"][group_start:group_end],
                    payload["mask_words"][group_start:group_end],
                    out_flat,
                    lse_flat,
                    softmax_scale=float(softmax_scale),
                )
                continue
            except Exception:
                pass
        packed_out, packed_lse = _run_range_packed(group_start, group_end)
        range_group_count = group_end - group_start
        packed_out_flat = packed_out.view(range_group_count * packed_q, packed_out.shape[2], packed_out.shape[3]).contiguous()
        packed_lse_flat = packed_lse.view(range_group_count * packed_q, packed_lse.shape[2]).contiguous()
        _run_synthetic_combine_scatter_rows_kernel(
            packed_out_flat,
            packed_lse_flat,
            payload["q_row_idx"][group_start:group_end].reshape(-1).contiguous(),
            out_flat,
            lse_flat,
        )
    if q.ndim == 4:
        return out_flat.to(dtype=v.dtype).view(q.shape[0], q.shape[1], q.shape[2], v.shape[3]).contiguous()
    return out_flat.to(dtype=v.dtype).contiguous()
