from __future__ import annotations

import math
from typing import Any

import torch

from flash_attn.cute.hsa_shared_sparse_gemm_analysis import (
    _encode_mask_rows_to_words,
    _measure_ms,
    _run_custom_masked_bucket_forward,
    _run_fa4_packed_bucket_forward,
    _run_shared_cta_bucket_forward,
)


def _normalize_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if isinstance(device, torch.device) else torch.device(device)


def _default_dtype_for_device(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _mask_intervals(mask_row: torch.Tensor) -> list[tuple[int, int]]:
    values = [bool(value) for value in mask_row.detach().cpu().tolist()]
    intervals: list[tuple[int, int]] = []
    start = None
    for idx, is_on in enumerate(values):
        if is_on and start is None:
            start = idx
        elif not is_on and start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(values)))
    return intervals


def _average_pairwise_row_jaccard(mask_rows: torch.Tensor) -> float:
    rows = mask_rows.shape[0]
    if rows <= 1:
        return 0.0
    total = 0.0
    count = 0
    for row_idx in range(rows):
        left = mask_rows[row_idx]
        for other_idx in range(row_idx + 1, rows):
            right = mask_rows[other_idx]
            intersection = int(torch.logical_and(left, right).sum().item())
            union = int(torch.logical_or(left, right).sum().item())
            total += 0.0 if union <= 0 else intersection / union
            count += 1
    return total / count if count > 0 else 0.0


def _mask_geometry(mask_bool: torch.Tensor, q_length: torch.Tensor) -> dict[str, float | int]:
    total_live_pairs = int(mask_bool.sum().item())
    support_width = int(mask_bool.shape[2]) if mask_bool.ndim == 3 else 0
    valid_rows = int(q_length.sum().item())
    rows_per_bucket = int(mask_bool.shape[1]) if mask_bool.ndim == 3 else 0
    num_buckets = int(mask_bool.shape[0]) if mask_bool.ndim == 3 else 0
    island_counts: list[int] = []
    gap_lengths: list[int] = []
    overlap_scores: list[float] = []

    for bucket_idx in range(num_buckets):
        valid_q = int(q_length[bucket_idx].item())
        if valid_q <= 0:
            continue
        bucket_masks = mask_bool[bucket_idx, :valid_q]
        overlap_scores.append(_average_pairwise_row_jaccard(bucket_masks))
        for row_idx in range(valid_q):
            intervals = _mask_intervals(bucket_masks[row_idx])
            island_counts.append(len(intervals))
            for interval_idx in range(len(intervals) - 1):
                gap_lengths.append(intervals[interval_idx + 1][0] - intervals[interval_idx][1])

    packed_area = max(1, num_buckets * rows_per_bucket * support_width)
    return {
        "num_buckets": num_buckets,
        "valid_rows": valid_rows,
        "live_pairs": total_live_pairs,
        "fill_rate": total_live_pairs / packed_area,
        "support_width": support_width,
        "avg_islands_per_row": float(sum(island_counts) / len(island_counts)) if island_counts else 0.0,
        "max_islands_per_row": max(island_counts) if island_counts else 0,
        "avg_gap": float(sum(gap_lengths) / len(gap_lengths)) if gap_lengths else 0.0,
        "max_gap": max(gap_lengths) if gap_lengths else 0,
        "avg_pairwise_row_jaccard": float(sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0,
    }


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _build_disjoint_confetti_row_mask(
    *,
    support_k: int,
    islands_per_row: int,
    island_width: int,
    row_seed: int,
    row_shift: int,
) -> torch.Tensor:
    if support_k <= 0:
        return torch.zeros((0,), dtype=torch.bool)
    max_start = max(0, support_k - island_width)
    search_space = max_start + 1
    stride = max(island_width + 1, support_k // max(1, islands_per_row))
    intervals: list[tuple[int, int]] = []
    base_seed = (row_seed * max(1, row_shift)) % max(1, search_space)
    for island_idx in range(islands_per_row):
        found = None
        preferred = (base_seed + island_idx * stride) % max(1, search_space)
        for attempt in range(search_space):
            start = (preferred + attempt) % max(1, search_space)
            candidate = (start, start + island_width)
            if any(_intervals_overlap(candidate, existing) for existing in intervals):
                continue
            found = candidate
            break
        if found is None:
            break
        intervals.append(found)
    mask = torch.zeros((support_k,), dtype=torch.bool)
    for start, end in intervals:
        mask[start:end] = True
    return mask


def _build_compact_control_row_mask(
    *,
    support_k: int,
    live_per_row: int,
    row_seed: int,
    row_shift: int,
) -> torch.Tensor:
    if support_k <= 0:
        return torch.zeros((0,), dtype=torch.bool)
    width = min(support_k, max(0, live_per_row))
    max_start = max(0, support_k - width)
    start = (row_seed * max(1, row_shift)) % max(1, max_start + 1)
    mask = torch.zeros((support_k,), dtype=torch.bool)
    mask[start : start + width] = True
    return mask


def _flatten_valid_packed_rows(packed_out: torch.Tensor, q_length: torch.Tensor) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for bucket_idx, q_count in enumerate(q_length.detach().cpu().tolist()):
        valid_q = int(q_count)
        if valid_q > 0:
            rows.append(packed_out[bucket_idx, :valid_q].float().contiguous())
    if not rows:
        return torch.empty(
            (packed_out.shape[2], 0, packed_out.shape[3]),
            dtype=torch.float32,
            device=packed_out.device,
        )
    return torch.cat(rows, dim=0).permute(1, 0, 2).contiguous()


def _total_rows_from_q_row_idx(
    q_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    *,
    fallback_total_rows: int,
) -> int:
    max_row = -1
    for bucket_idx, q_count in enumerate(q_length.detach().cpu().tolist()):
        valid_q = int(q_count)
        if valid_q <= 0:
            continue
        valid_rows = q_row_idx[bucket_idx, :valid_q]
        if int(valid_rows.numel()) > 0:
            max_row = max(max_row, int(valid_rows.max().item()))
    return max(int(fallback_total_rows), max_row + 1)


def scatter_explicit_packed_rows(
    packed_out: torch.Tensor,
    q_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    *,
    total_rows: int,
) -> torch.Tensor:
    scattered = torch.zeros(
        (total_rows, packed_out.shape[2], packed_out.shape[3]),
        dtype=torch.float32,
        device=packed_out.device,
    )
    if int(packed_out.shape[0]) <= 0 or int(packed_out.shape[1]) <= 0:
        return scattered
    slot_valid = torch.arange(int(packed_out.shape[1]), device=q_length.device).unsqueeze(0) < q_length.unsqueeze(1)
    target_rows = q_row_idx.to(device=packed_out.device)[slot_valid.to(device=packed_out.device)].long()
    if int(target_rows.numel()) <= 0:
        return scattered
    if bool((target_rows < 0).any().item()) or bool((target_rows >= total_rows).any().item()):
        raise RuntimeError("explicit_2d_q_row_idx_out_of_bounds")
    source_rows = packed_out[slot_valid.to(device=packed_out.device)].float()
    scattered.index_copy_(0, target_rows, source_rows)
    return scattered


def _build_micro_bucket(
    *,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    mask_bool: torch.Tensor,
    q_length: torch.Tensor,
    q_row_idx: torch.Tensor,
) -> dict[str, Any]:
    num_buckets, _, num_heads, head_dim = q_buf.shape
    support_k = int(k_buf.shape[1])
    micro_bucket_count = sum((int(q_count.item()) + 1) // 2 for q_count in q_length)
    micro_q_buf = torch.zeros((micro_bucket_count, 2, num_heads, head_dim), dtype=q_buf.dtype, device=q_buf.device)
    micro_k_buf = torch.zeros((micro_bucket_count, support_k, num_heads, head_dim), dtype=k_buf.dtype, device=k_buf.device)
    micro_v_buf = torch.zeros((micro_bucket_count, support_k, num_heads, head_dim), dtype=v_buf.dtype, device=v_buf.device)
    micro_mask_bool = torch.zeros((micro_bucket_count, 2, support_k), dtype=torch.bool, device=q_buf.device)
    micro_q_length = torch.zeros((micro_bucket_count,), dtype=torch.int32, device=q_buf.device)
    micro_k_length = torch.full((micro_bucket_count,), support_k, dtype=torch.int32, device=q_buf.device)
    micro_q_row_idx = torch.full((micro_bucket_count, 2), -1, dtype=torch.int32, device=q_buf.device)

    micro_idx = 0
    for bucket_idx in range(num_buckets):
        valid_q = int(q_length[bucket_idx].item())
        for row_start in range(0, valid_q, 2):
            row_count = min(2, valid_q - row_start)
            micro_q_buf[micro_idx, :row_count] = q_buf[bucket_idx, row_start : row_start + row_count]
            micro_k_buf[micro_idx] = k_buf[bucket_idx]
            micro_v_buf[micro_idx] = v_buf[bucket_idx]
            micro_mask_bool[micro_idx, :row_count] = mask_bool[bucket_idx, row_start : row_start + row_count]
            micro_q_length[micro_idx] = row_count
            micro_q_row_idx[micro_idx, :row_count] = q_row_idx[bucket_idx, row_start : row_start + row_count]
            micro_idx += 1

    micro_mask_words = _encode_mask_rows_to_words(micro_mask_bool.reshape(micro_bucket_count * 2, support_k)).view(
        micro_bucket_count,
        2,
        -1,
    )
    return {
        "packed_q": 2,
        "support_rows": support_k,
        "custom_q_buf": micro_q_buf.contiguous(),
        "custom_k_buf": micro_k_buf.contiguous(),
        "custom_v_buf": micro_v_buf.contiguous(),
        "custom_mask_bool": micro_mask_bool.contiguous(),
        "custom_mask_words": micro_mask_words.contiguous(),
        "custom_q_length": micro_q_length.contiguous(),
        "custom_k_length": micro_k_length.contiguous(),
        "q_row_idx": micro_q_row_idx.contiguous(),
    }


def _partition_packed_rows_by_support(
    bucket_mask: torch.Tensor,
    *,
    max_rows_per_group: int,
) -> list[list[int]]:
    valid_q = int(bucket_mask.shape[0])
    remaining = list(range(valid_q))
    groups: list[list[int]] = []

    while remaining:
        seed_row = remaining.pop(0)
        group = [seed_row]
        union_mask = bucket_mask[seed_row].clone()
        while remaining and len(group) < max_rows_per_group:
            best_idx = 0
            best_key: tuple[int, int, int] | None = None
            for candidate_pos, candidate_row in enumerate(remaining):
                candidate_mask = bucket_mask[candidate_row]
                merged_union = torch.logical_or(union_mask, candidate_mask)
                union_k = int(merged_union.sum().item())
                overlap_k = int(torch.logical_and(union_mask, candidate_mask).sum().item())
                candidate_key = (union_k, -overlap_k, candidate_row)
                if best_key is None or candidate_key < best_key:
                    best_idx = candidate_pos
                    best_key = candidate_key
            chosen_row = remaining.pop(best_idx)
            group.append(chosen_row)
            union_mask = torch.logical_or(union_mask, bucket_mask[chosen_row])
        groups.append(group)
    return groups


def _build_direct_2d_bucket(
    *,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    mask_bool: torch.Tensor,
    q_length: torch.Tensor,
    q_row_idx: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    num_buckets, _, num_heads, head_dim = q_buf.shape
    support_k = int(k_buf.shape[1])
    max_rows_per_group = min(8, int(q_buf.shape[1]))
    group_rows_per_bucket: list[list[list[int]]] = []
    group_count = 0
    union_cols_per_group: list[torch.Tensor] = []
    group_row_counts: list[int] = []
    group_fill: list[float] = []
    max_union_k = 0

    for bucket_idx in range(num_buckets):
        valid_q = int(q_length[bucket_idx].item())
        if valid_q <= 0:
            group_rows_per_bucket.append([])
            continue
        groups = _partition_packed_rows_by_support(
            mask_bool[bucket_idx, :valid_q],
            max_rows_per_group=max_rows_per_group,
        )
        group_rows_per_bucket.append(groups)
        group_count += len(groups)
        for group_rows in groups:
            row_count = len(group_rows)
            group_mask = mask_bool[bucket_idx, group_rows]
            union_cols = torch.nonzero(torch.any(group_mask, dim=0), as_tuple=False).flatten()
            union_k = int(union_cols.numel())
            max_union_k = max(max_union_k, union_k)
            group_fill_denom = max(1, row_count * max(1, union_k))
            group_fill.append(float(group_mask.sum().item()) / group_fill_denom)
            group_row_counts.append(row_count)
            union_cols_per_group.append(union_cols)

    rows_per_group = max(1, max((len(group_rows) for groups in group_rows_per_bucket for group_rows in groups), default=0))
    max_union_k = max(1, max_union_k)
    direct_q_buf = torch.zeros((group_count, rows_per_group, num_heads, head_dim), dtype=q_buf.dtype, device=q_buf.device)
    direct_k_buf = torch.zeros((group_count, max_union_k, num_heads, head_dim), dtype=k_buf.dtype, device=k_buf.device)
    direct_v_buf = torch.zeros((group_count, max_union_k, num_heads, head_dim), dtype=v_buf.dtype, device=v_buf.device)
    direct_mask_bool = torch.zeros((group_count, rows_per_group, max_union_k), dtype=torch.bool, device=q_buf.device)
    direct_q_length = torch.zeros((group_count,), dtype=torch.int32, device=q_buf.device)
    direct_k_length = torch.zeros((group_count,), dtype=torch.int32, device=q_buf.device)
    direct_q_row_idx = torch.full((group_count, rows_per_group), -1, dtype=torch.int32, device=q_buf.device)

    group_idx = 0
    for bucket_idx in range(num_buckets):
        for group_rows in group_rows_per_bucket[bucket_idx]:
            row_count = len(group_rows)
            union_cols = union_cols_per_group[group_idx]
            union_k = int(union_cols.numel())
            direct_q_buf[group_idx, :row_count] = q_buf[bucket_idx, group_rows]
            direct_q_length[group_idx] = row_count
            direct_q_row_idx[group_idx, :row_count] = q_row_idx[bucket_idx, group_rows]
            if union_k > 0:
                direct_k_buf[group_idx, :union_k] = k_buf[bucket_idx, union_cols]
                direct_v_buf[group_idx, :union_k] = v_buf[bucket_idx, union_cols]
                direct_mask_bool[group_idx, :row_count, :union_k] = mask_bool[
                    bucket_idx,
                    group_rows,
                ][:, union_cols]
            direct_k_length[group_idx] = union_k
            group_idx += 1

    direct_mask_words = _encode_mask_rows_to_words(
        direct_mask_bool.reshape(group_count * rows_per_group, max_union_k)
    ).view(
        group_count,
        rows_per_group,
        -1,
    )
    direct_bucket = {
        "packed_q": rows_per_group,
        "support_rows": max_union_k,
        "custom_q_buf": direct_q_buf.contiguous(),
        "custom_k_buf": direct_k_buf.contiguous(),
        "custom_v_buf": direct_v_buf.contiguous(),
        "custom_mask_bool": direct_mask_bool.contiguous(),
        "custom_mask_words": direct_mask_words.contiguous(),
        "custom_q_length": direct_q_length.contiguous(),
        "custom_k_length": direct_k_length.contiguous(),
        "q_row_idx": direct_q_row_idx.contiguous(),
        "total_rows": _total_rows_from_q_row_idx(
            q_row_idx,
            q_length,
            fallback_total_rows=int(q_length.sum().item()),
        ),
    }
    avg_union_k = (
        float(direct_k_length.float().mean().item()) if group_count > 0 else 0.0
    )
    total_row_area = sum(row_count * max_union_k for row_count in group_row_counts)
    total_live_pairs = int(mask_bool.sum().item())
    direct_geometry = {
        "direct_2d_groups": group_count,
        "direct_2d_rows_per_group": rows_per_group,
        "direct_2d_avg_union_k": avg_union_k,
        "direct_2d_max_union_k": max_union_k,
        "direct_2d_avg_group_fill": float(sum(group_fill) / len(group_fill)) if group_fill else 0.0,
        "direct_2d_case_fill_rate": (
            total_live_pairs / max(1, total_row_area)
        ),
        "direct_2d_support_reduction": (
            0.0 if support_k <= 0 else 1.0 - (avg_union_k / float(support_k))
        ),
    }
    return direct_bucket, direct_geometry


def _build_direct_2d_compact_payload(
    *,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    mask_bool: torch.Tensor,
    q_length: torch.Tensor,
    q_row_idx: torch.Tensor,
    tile_k: int = 32,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    num_buckets, packed_q, num_heads, head_dim = q_buf.shape
    support_k = int(k_buf.shape[1])
    device = q_buf.device
    full_tile_span = max(1, math.ceil(support_k / max(1, tile_k)))
    bucket_union_cols: list[torch.Tensor] = []
    bucket_union_k: list[int] = []
    bucket_tile_spans: list[int] = []
    buckets_by_tile_span: dict[int, list[int]] = {}
    full_bucket_indices: list[int] = []
    compact_fill: list[float] = []

    for bucket_idx in range(num_buckets):
        valid_q = int(q_length[bucket_idx].item())
        if valid_q <= 0:
            bucket_union_cols.append(torch.empty((0,), dtype=torch.long, device=device))
            bucket_union_k.append(0)
            bucket_tile_spans.append(0)
            continue
        union_cols = torch.nonzero(torch.any(mask_bool[bucket_idx, :valid_q], dim=0), as_tuple=False).flatten()
        union_k = int(union_cols.numel())
        tile_span = max(1, math.ceil(union_k / max(1, tile_k)))
        bucket_union_cols.append(union_cols)
        bucket_union_k.append(union_k)
        bucket_tile_spans.append(tile_span)
        effective_support = support_k
        if tile_span < full_tile_span:
            buckets_by_tile_span.setdefault(tile_span, []).append(bucket_idx)
            effective_support = max(1, union_k)
        else:
            full_bucket_indices.append(bucket_idx)
        compact_fill_denom = max(1, valid_q * effective_support)
        compact_fill.append(float(mask_bool[bucket_idx, :valid_q].sum().item()) / compact_fill_denom)

    compact_groups: list[dict[str, Any]] = []
    for tile_span in sorted(buckets_by_tile_span):
        bucket_indices = buckets_by_tile_span[tile_span]
        if not bucket_indices:
            continue
        bucket_index_tensor = torch.tensor(bucket_indices, dtype=torch.long, device=device)
        max_union_k = max(bucket_union_k[bucket_idx] for bucket_idx in bucket_indices)
        group_q_buf = q_buf.index_select(0, bucket_index_tensor).contiguous()
        group_q_length = q_length.index_select(0, bucket_index_tensor).contiguous()
        group_k_buf = torch.zeros(
            (len(bucket_indices), max_union_k, num_heads, head_dim),
            dtype=k_buf.dtype,
            device=device,
        )
        group_v_buf = torch.zeros(
            (len(bucket_indices), max_union_k, num_heads, head_dim),
            dtype=v_buf.dtype,
            device=device,
        )
        group_mask_bool = torch.zeros(
            (len(bucket_indices), packed_q, max_union_k),
            dtype=torch.bool,
            device=device,
        )
        group_k_length = torch.zeros((len(bucket_indices),), dtype=torch.int32, device=device)

        for local_idx, bucket_idx in enumerate(bucket_indices):
            union_cols = bucket_union_cols[bucket_idx]
            union_k = int(union_cols.numel())
            if union_k <= 0:
                continue
            group_k_buf[local_idx, :union_k] = k_buf[bucket_idx, union_cols]
            group_v_buf[local_idx, :union_k] = v_buf[bucket_idx, union_cols]
            group_mask_bool[local_idx, :, :union_k] = mask_bool[bucket_idx, :, union_cols]
            group_k_length[local_idx] = union_k

        group_mask_words = _encode_mask_rows_to_words(
            group_mask_bool.reshape(len(bucket_indices) * packed_q, max_union_k)
        ).view(
            len(bucket_indices),
            packed_q,
            -1,
        )
        compact_groups.append(
            {
                "bucket_indices": bucket_index_tensor.contiguous(),
                "packed_q": packed_q,
                "support_rows": max_union_k,
                "custom_q_buf": group_q_buf,
                "custom_k_buf": group_k_buf.contiguous(),
                "custom_v_buf": group_v_buf.contiguous(),
                "custom_mask_bool": group_mask_bool.contiguous(),
                "custom_mask_words": group_mask_words.contiguous(),
                "custom_q_length": group_q_length,
                "custom_k_length": group_k_length.contiguous(),
            }
        )

    if full_bucket_indices:
        bucket_index_tensor = torch.tensor(full_bucket_indices, dtype=torch.long, device=device)
        group_k_length = torch.full((len(full_bucket_indices),), support_k, dtype=torch.int32, device=device)
        group_mask_bool = mask_bool.index_select(0, bucket_index_tensor).contiguous()
        group_mask_words = _encode_mask_rows_to_words(
            group_mask_bool.reshape(len(full_bucket_indices) * packed_q, support_k)
        ).view(
            len(full_bucket_indices),
            packed_q,
            -1,
        )
        compact_groups.append(
            {
                "bucket_indices": bucket_index_tensor.contiguous(),
                "packed_q": packed_q,
                "support_rows": support_k,
                "custom_q_buf": q_buf.index_select(0, bucket_index_tensor).contiguous(),
                "custom_k_buf": k_buf.index_select(0, bucket_index_tensor).contiguous(),
                "custom_v_buf": v_buf.index_select(0, bucket_index_tensor).contiguous(),
                "custom_mask_bool": group_mask_bool,
                "custom_mask_words": group_mask_words.contiguous(),
                "custom_q_length": q_length.index_select(0, bucket_index_tensor).contiguous(),
                "custom_k_length": group_k_length,
            }
        )

    valid_union_k = [value for value in bucket_union_k if value > 0]
    total_live_pairs = int(mask_bool.sum().item())
    total_compact_area = sum(
        int(q_length[bucket_idx].item())
        * (
            max(1, bucket_union_k[bucket_idx])
            if bucket_tile_spans[bucket_idx] < full_tile_span
            else support_k
        )
        for bucket_idx in range(num_buckets)
        if int(q_length[bucket_idx].item()) > 0
    )
    compact_geometry = {
        "direct_2d_compact_launch_groups": len(compact_groups),
        "direct_2d_compact_avg_union_k": (
            float(sum(valid_union_k) / len(valid_union_k)) if valid_union_k else 0.0
        ),
        "direct_2d_compact_max_union_k": max(valid_union_k) if valid_union_k else 0,
        "direct_2d_compact_avg_tile_span": (
            float(sum(span for span in bucket_tile_spans if span > 0) / len(valid_union_k)) if valid_union_k else 0.0
        ),
        "direct_2d_compact_avg_group_fill": float(sum(compact_fill) / len(compact_fill)) if compact_fill else 0.0,
        "direct_2d_compact_case_fill_rate": total_live_pairs / max(1, total_compact_area),
        "direct_2d_compact_buckets_compacted": sum(len(bucket_indices) for bucket_indices in buckets_by_tile_span.values()),
        "direct_2d_compact_buckets_passthrough": len(full_bucket_indices),
    }
    return {
        "groups": compact_groups,
        "total_buckets": num_buckets,
        "packed_q": packed_q,
        "custom_q_length": q_length.contiguous(),
        "q_row_idx": q_row_idx.contiguous(),
        "total_rows": _total_rows_from_q_row_idx(
            q_row_idx,
            q_length,
            fallback_total_rows=int(q_length.sum().item()),
        ),
    }, compact_geometry


def _build_shared_support_buckets(
    *,
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    mask_bool: torch.Tensor,
    q_length: torch.Tensor,
) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    num_buckets = int(q_buf.shape[0])
    support_k = int(k_buf.shape[1])
    for bucket_idx in range(num_buckets):
        valid_q = int(q_length[bucket_idx].item())
        if valid_q <= 0:
            continue
        num_qgroups = (valid_q + 1) // 2
        shared_q_buf = torch.zeros(
            (num_qgroups, 2, q_buf.shape[2], q_buf.shape[3]),
            dtype=q_buf.dtype,
            device=q_buf.device,
        )
        shared_mask_bool = torch.zeros((num_qgroups, 2, support_k), dtype=torch.bool, device=q_buf.device)
        shared_q_length = torch.zeros((num_qgroups,), dtype=torch.int32, device=q_buf.device)
        q_cursor = 0
        for qgroup_idx in range(num_qgroups):
            q_count = min(2, valid_q - q_cursor)
            shared_q_buf[qgroup_idx, :q_count] = q_buf[bucket_idx, q_cursor : q_cursor + q_count]
            shared_mask_bool[qgroup_idx, :q_count] = mask_bool[bucket_idx, q_cursor : q_cursor + q_count]
            shared_q_length[qgroup_idx] = q_count
            q_cursor += q_count
        shared_mask_words = _encode_mask_rows_to_words(shared_mask_bool.view(num_qgroups * 2, support_k)).view(
            num_qgroups,
            2,
            -1,
        )
        buckets.append(
            {
                "shared_cta_q_buf": shared_q_buf.contiguous(),
                "shared_k_expanded": k_buf[bucket_idx].contiguous(),
                "shared_v_expanded": v_buf[bucket_idx].contiguous(),
                "shared_cta_q_length": shared_q_length.contiguous(),
                "shared_cta_mask_words": shared_mask_words.contiguous(),
                "shared_cta_qgroups_per_cta": 2,
            }
        )
    return buckets


def build_explicit_2d_sparse_case(
    *,
    case_family: str,
    seqlen: int,
    heads: int,
    head_dim: int,
    packed_q: int,
    support_k: int,
    islands_per_row: int,
    island_width: int,
    row_shift: int,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    if case_family not in {"disjoint_confetti", "compact_control"}:
        raise ValueError(f"unsupported case_family {case_family!r}")
    if seqlen <= 0 or heads <= 0 or head_dim <= 0 or packed_q <= 0 or support_k <= 0:
        raise ValueError("seqlen, heads, head_dim, packed_q, and support_k must be positive")
    if islands_per_row <= 0 or island_width <= 0:
        raise ValueError("islands_per_row and island_width must be positive")

    device = _normalize_device(device)
    dtype = _default_dtype_for_device(device) if dtype is None else dtype
    num_buckets = (seqlen + packed_q - 1) // packed_q
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)

    k_buf = torch.randn((num_buckets, support_k, heads, head_dim), dtype=dtype, device=device, generator=generator)
    v_buf = torch.randn((num_buckets, support_k, heads, head_dim), dtype=dtype, device=device, generator=generator)
    q_buf = torch.randn((num_buckets, packed_q, heads, head_dim), dtype=dtype, device=device, generator=generator)
    q_length_cpu = torch.full((num_buckets,), packed_q, dtype=torch.int32)
    tail_q = seqlen - (num_buckets - 1) * packed_q
    if 0 < tail_q < packed_q:
        q_length_cpu[-1] = int(tail_q)
        q_buf[-1, tail_q:] = 0
    q_length = q_length_cpu.to(device=device)
    k_length = torch.full((num_buckets,), support_k, dtype=torch.int32, device=device)
    q_row_idx = torch.arange(num_buckets * packed_q, dtype=torch.int32, device=device).view(num_buckets, packed_q)
    q_row_idx = q_row_idx.masked_fill(q_row_idx >= seqlen, -1)

    live_per_row = min(support_k, islands_per_row * island_width)
    if case_family == "compact_control":
        width = min(support_k, max(0, live_per_row))
        global_rows = q_row_idx.to(dtype=torch.int64)
        row_valid = global_rows >= 0
        max_start = max(0, support_k - width)
        starts = (global_rows.clamp_min(0) * max(1, row_shift // 2)) % max(1, max_start + 1)
        cols = torch.arange(support_k, dtype=torch.int64, device=device).view(1, 1, support_k)
        mask_bool = row_valid.unsqueeze(-1) & (cols >= starts.unsqueeze(-1)) & (cols < (starts + width).unsqueeze(-1))
    else:
        mask_bool_cpu = torch.zeros((num_buckets, packed_q, support_k), dtype=torch.bool)
        for global_row in range(seqlen):
            bucket_idx, row_idx = divmod(global_row, packed_q)
            mask_bool_cpu[bucket_idx, row_idx] = _build_disjoint_confetti_row_mask(
                support_k=support_k,
                islands_per_row=islands_per_row,
                island_width=island_width,
                row_seed=global_row,
                row_shift=row_shift,
            )
        mask_bool = mask_bool_cpu.to(device=device)

    mask_words = _encode_mask_rows_to_words(mask_bool.reshape(num_buckets * packed_q, support_k)).view(
        num_buckets,
        packed_q,
        -1,
    )
    full_bucket = {
        "packed_q": packed_q,
        "support_rows": support_k,
        "custom_q_buf": q_buf.contiguous(),
        "custom_k_buf": k_buf.contiguous(),
        "custom_v_buf": v_buf.contiguous(),
        "custom_mask_bool": mask_bool.contiguous(),
        "custom_mask_words": mask_words.contiguous(),
        "custom_q_length": q_length.contiguous(),
        "custom_k_length": k_length.contiguous(),
        "q_row_idx": q_row_idx.contiguous(),
        "total_rows": _total_rows_from_q_row_idx(
            q_row_idx,
            q_length,
            fallback_total_rows=seqlen,
        ),
    }
    geometry = _mask_geometry(mask_bool, q_length)
    geometry.update(
        {
            "case_family": case_family,
            "seqlen": seqlen,
            "heads": heads,
            "head_dim": head_dim,
            "packed_q": packed_q,
            "support_k": support_k,
            "requested_islands_per_row": islands_per_row,
            "requested_island_width": island_width,
            "row_shift": row_shift,
            "live_pairs_per_row": live_per_row,
        }
    )
    direct_2d_bucket, direct_2d_geometry = _build_direct_2d_bucket(
        q_buf=q_buf,
        k_buf=k_buf,
        v_buf=v_buf,
        mask_bool=mask_bool,
        q_length=q_length,
        q_row_idx=q_row_idx,
    )
    direct_2d_compact_payload, direct_2d_compact_geometry = _build_direct_2d_compact_payload(
        q_buf=q_buf,
        k_buf=k_buf,
        v_buf=v_buf,
        mask_bool=mask_bool,
        q_length=q_length,
        q_row_idx=q_row_idx,
    )
    geometry.update(direct_2d_geometry)
    geometry.update(direct_2d_compact_geometry)
    return {
        "case_family": case_family,
        "seqlen": seqlen,
        "heads": heads,
        "head_dim": head_dim,
        "packed_q": packed_q,
        "support_k": support_k,
        "islands_per_row": islands_per_row,
        "island_width": island_width,
        "row_shift": row_shift,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "full_bucket": full_bucket,
        "direct_2d_bucket": direct_2d_bucket,
        "direct_2d_compact_payload": direct_2d_compact_payload,
        "micro_bucket": _build_micro_bucket(
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
            mask_bool=mask_bool,
            q_length=q_length,
            q_row_idx=q_row_idx,
        ),
        "shared_support_buckets": _build_shared_support_buckets(
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
            mask_bool=mask_bool,
            q_length=q_length,
        ),
        "geometry": geometry,
    }


def _run_dense_explicit_bucket_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    q_buf = bucket["custom_q_buf"].float()
    k_buf = bucket["custom_k_buf"].float()
    v_buf = bucket["custom_v_buf"].float()
    mask = bucket["custom_mask_bool"]
    num_buckets, packed_q, num_heads, _ = q_buf.shape
    support_k = int(k_buf.shape[1])
    q_bhd = q_buf.permute(0, 2, 1, 3).reshape(num_buckets * num_heads, packed_q, q_buf.shape[3])
    k_bhd = k_buf.permute(0, 2, 1, 3).reshape(num_buckets * num_heads, support_k, k_buf.shape[3])
    v_bhd = v_buf.permute(0, 2, 1, 3).reshape(num_buckets * num_heads, support_k, v_buf.shape[3])
    mask_bhqk = mask.unsqueeze(1).expand(num_buckets, num_heads, packed_q, support_k).reshape(
        num_buckets * num_heads,
        packed_q,
        support_k,
    )
    scaled = torch.bmm(q_bhd, k_bhd.transpose(1, 2)) * float(softmax_scale)
    scaled = scaled.masked_fill(~mask_bhqk, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.bmm(probs, v_bhd)
    packed_out = out.view(num_buckets, num_heads, packed_q, v_buf.shape[3]).permute(0, 2, 1, 3).contiguous()
    return _flatten_valid_packed_rows(packed_out, bucket["custom_q_length"])


def _run_direct_2d_packed_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
        _can_use_synthetic_2d_masked_fwd,
        _run_synthetic_2d_masked_fwd_kernel,
    )

    q_buf = bucket["custom_q_buf"]
    k_buf = bucket["custom_k_buf"]
    v_buf = bucket["custom_v_buf"]
    packed_q = int(bucket["packed_q"])
    packed_k = int(bucket["support_rows"])
    if not _can_use_synthetic_2d_masked_fwd(
        q_buf,
        k_buf,
        v_buf,
        packed_q=packed_q,
        packed_k=packed_k,
    ):
        raise RuntimeError("direct_2d_masked_unsupported")
    packed_out, _ = _run_synthetic_2d_masked_fwd_kernel(
        q_buf,
        k_buf,
        v_buf,
        bucket["custom_q_length"],
        bucket["custom_k_length"],
        bucket["custom_mask_words"],
        softmax_scale=softmax_scale,
    )
    return packed_out


def _run_direct_2d_bucket_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    packed_out = _run_direct_2d_packed_forward(bucket, softmax_scale=softmax_scale)
    total_rows = bucket.get("total_rows")
    if total_rows is None:
        return _flatten_valid_packed_rows(packed_out, bucket["custom_q_length"])
    return scatter_explicit_packed_rows(
        packed_out,
        bucket["q_row_idx"],
        bucket["custom_q_length"],
        total_rows=int(total_rows),
    ).permute(1, 0, 2).contiguous()


def _run_direct_2d_compact_forward(case_payload: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    compact_payload = case_payload["direct_2d_compact_payload"]
    full_bucket = case_payload["full_bucket"]
    packed_out = torch.zeros(
        (
            int(compact_payload["total_buckets"]),
            int(compact_payload["packed_q"]),
            full_bucket["custom_q_buf"].shape[2],
            full_bucket["custom_v_buf"].shape[3],
        ),
        dtype=torch.float32,
        device=full_bucket["custom_q_buf"].device,
    )
    for group in compact_payload["groups"]:
        group_out = _run_direct_2d_packed_forward(group, softmax_scale=softmax_scale).float()
        packed_out.index_copy_(0, group["bucket_indices"].long(), group_out)
    if "q_row_idx" in compact_payload and "total_rows" in compact_payload:
        return scatter_explicit_packed_rows(
            packed_out,
            compact_payload["q_row_idx"],
            compact_payload["custom_q_length"],
            total_rows=int(compact_payload["total_rows"]),
        ).permute(1, 0, 2).contiguous()
    return _flatten_valid_packed_rows(packed_out, compact_payload["custom_q_length"])


def _run_explicit_shared_support_forward(case_payload: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for bucket in case_payload["shared_support_buckets"]:
        bucket_out = _run_shared_cta_bucket_forward(bucket, softmax_scale=softmax_scale)
        rows.append(bucket_out)
    if not rows:
        full_bucket = case_payload["full_bucket"]
        return torch.empty(
            (full_bucket["custom_q_buf"].shape[2], 0, full_bucket["custom_v_buf"].shape[3]),
            dtype=torch.float32,
            device=full_bucket["custom_q_buf"].device,
        )
    return torch.cat(rows, dim=1).contiguous()


def _status_from_exc(exc: Exception) -> str:
    message = str(exc).strip().lower()
    if "unsupported" in message or isinstance(exc, NotImplementedError):
        return "unsupported_shape"
    if "compile" in message:
        return "compile_failed"
    return f"failed_{type(exc).__name__}"


def analyze_explicit_2d_sparse_forward(
    *,
    case_family: str,
    seqlen: int,
    heads: int,
    head_dim: int,
    packed_q: int,
    support_k: int,
    islands_per_row: int,
    island_width: int,
    row_shift: int,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    variants: tuple[str, ...] = ("dense", "custom_masked", "fa4_packed", "direct_2d"),
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    valid_variants = {"dense", "custom_masked", "fa4_packed", "direct_2d", "direct_2d_compact", "shared_support"}
    if any(variant not in valid_variants for variant in variants):
        unknown = sorted(set(variants) - valid_variants)
        raise ValueError(f"unknown variants {unknown}")

    case_payload = build_explicit_2d_sparse_case(
        case_family=case_family,
        seqlen=seqlen,
        heads=heads,
        head_dim=head_dim,
        packed_q=packed_q,
        support_k=support_k,
        islands_per_row=islands_per_row,
        island_width=island_width,
        row_shift=row_shift,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    full_bucket = case_payload["full_bucket"]
    softmax_scale = head_dim ** (-0.5)
    dense_out = _run_dense_explicit_bucket_forward(full_bucket, softmax_scale=softmax_scale)

    runners = {
        "dense": lambda: _run_dense_explicit_bucket_forward(full_bucket, softmax_scale=softmax_scale),
        "custom_masked": lambda: _run_custom_masked_bucket_forward(
            case_payload["micro_bucket"],
            softmax_scale=softmax_scale,
        ),
        "fa4_packed": lambda: _run_fa4_packed_bucket_forward(
            full_bucket,
            softmax_scale=softmax_scale,
        ),
        "direct_2d": lambda: _run_direct_2d_bucket_forward(
            full_bucket,
            softmax_scale=softmax_scale,
        ),
        "direct_2d_compact": lambda: _run_direct_2d_compact_forward(
            case_payload,
            softmax_scale=softmax_scale,
        ),
        "shared_support": lambda: _run_explicit_shared_support_forward(
            case_payload,
            softmax_scale=softmax_scale,
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        runner = runners[variant]
        try:
            out = runner()
            diff = (dense_out.float() - out.float()).abs()
            results[variant] = {
                "status": "measured",
                "fwd_ms": _measure_ms(runner, warmup_iters, benchmark_iters),
                "output_max_diff": float(diff.max().item()) if diff.numel() > 0 else 0.0,
                "output_mean_diff": float(diff.mean().item()) if diff.numel() > 0 else 0.0,
            }
        except Exception as exc:  # pragma: no cover - benchmark-only failure path
            results[variant] = {
                "status": _status_from_exc(exc),
                "error": f"{type(exc).__name__}: {exc}",
                "fwd_ms": float("nan"),
                "output_max_diff": float("nan"),
                "output_mean_diff": float("nan"),
            }

    dense_ms = results.get("dense", {}).get("fwd_ms")
    custom_masked_ms = results.get("custom_masked", {}).get("fwd_ms")
    fa4_ms = results.get("fa4_packed", {}).get("fwd_ms")
    for payload in results.values():
        if isinstance(dense_ms, float) and dense_ms > 0 and isinstance(payload.get("fwd_ms"), float) and payload["fwd_ms"] > 0:
            payload["speedup_vs_dense"] = dense_ms / payload["fwd_ms"]
        if (
            isinstance(custom_masked_ms, float)
            and custom_masked_ms > 0
            and isinstance(payload.get("fwd_ms"), float)
            and payload["fwd_ms"] > 0
        ):
            payload["speedup_vs_custom_masked"] = custom_masked_ms / payload["fwd_ms"]
        if isinstance(fa4_ms, float) and fa4_ms > 0 and isinstance(payload.get("fwd_ms"), float) and payload["fwd_ms"] > 0:
            payload["speedup_vs_fa4_packed"] = fa4_ms / payload["fwd_ms"]

    go_no_go = {
        "status": "not_applicable",
        "reason": "direct_2d, custom_masked, and fa4_packed must all measure successfully",
    }
    direct_2d = results.get("direct_2d")
    custom_masked = results.get("custom_masked")
    fa4_packed = results.get("fa4_packed")
    if (
        direct_2d is not None
        and custom_masked is not None
        and fa4_packed is not None
        and direct_2d.get("status") == "measured"
        and custom_masked.get("status") == "measured"
        and fa4_packed.get("status") == "measured"
    ):
        if (
            float(direct_2d["fwd_ms"]) < float(custom_masked["fwd_ms"])
            and float(direct_2d["fwd_ms"]) < float(fa4_packed["fwd_ms"])
        ):
            go_no_go = {"status": "pass", "reason": "direct_2d beat both custom_masked and fa4_packed"}
        else:
            go_no_go = {"status": "fail", "reason": "direct_2d did not beat both custom_masked and fa4_packed"}

    return {
        "status": "measured",
        "case_family": case_family,
        "dtype": case_payload["dtype"],
        "device": case_payload["device"],
        "variants": list(variants),
        "geometry": case_payload["geometry"],
        "results": results,
        "go_no_go": go_no_go,
    }


def summarize_explicit_2d_sparse_forward(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", {})
    measured = [
        (name, payload)
        for name, payload in results.items()
        if isinstance(payload, dict) and payload.get("status") == "measured"
    ]
    summary = {
        "status": str(report.get("status", "unknown")),
        "case_family": str(report.get("case_family", "unknown")),
        "go_no_go": report.get("go_no_go", {}),
    }
    if measured:
        best_name, best_payload = min(measured, key=lambda item: float(item[1]["fwd_ms"]))
        summary["best_variant"] = {
            "name": best_name,
            "fwd_ms": float(best_payload["fwd_ms"]),
        }
    geometry = report.get("geometry")
    if isinstance(geometry, dict):
        summary["geometry"] = {
            "live_pairs": int(geometry.get("live_pairs", 0)),
            "fill_rate": float(geometry.get("fill_rate", 0.0)),
            "avg_islands_per_row": float(geometry.get("avg_islands_per_row", 0.0)),
            "avg_pairwise_row_jaccard": float(geometry.get("avg_pairwise_row_jaccard", 0.0)),
        }
    for variant_name in ("direct_2d", "direct_2d_compact"):
        variant = results.get(variant_name)
        if not isinstance(variant, dict):
            continue
        variant_summary = {
            "status": str(variant.get("status", "unknown")),
        }
        for key in (
            "fwd_ms",
            "output_max_diff",
            "output_mean_diff",
            "speedup_vs_custom_masked",
            "speedup_vs_fa4_packed",
        ):
            value = variant.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                variant_summary[key] = float(value)
        if "error" in variant:
            variant_summary["error"] = str(variant["error"])
        summary[variant_name] = variant_summary
    return summary


def summarize_explicit_2d_sparse_suite(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    primary_confetti_cases: list[str] = []
    primary_confetti_failures: list[dict[str, str]] = []
    min_speedup_vs_custom_masked: float | None = None
    min_speedup_vs_fa4_packed: float | None = None

    for case_idx, case_payload in enumerate(case_payloads):
        name = str(case_payload.get("name", f"case_{case_idx}"))
        report = case_payload.get("report", case_payload)
        summary = case_payload.get("summary")
        if not isinstance(summary, dict):
            summary = summarize_explicit_2d_sparse_forward(report)
        go_no_go = summary.get("go_no_go", report.get("go_no_go", {}))
        status = str(go_no_go.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

        case_family = str(summary.get("case_family", report.get("case_family", "unknown")))
        if case_family != "disjoint_confetti":
            continue

        primary_confetti_cases.append(name)
        if status != "pass":
            primary_confetti_failures.append(
                {
                    "name": name,
                    "status": status,
                    "reason": str(go_no_go.get("reason", "unknown")),
                }
            )
            continue

        direct_2d = summary.get("direct_2d", {})
        speedup_vs_custom_masked = direct_2d.get("speedup_vs_custom_masked")
        speedup_vs_fa4_packed = direct_2d.get("speedup_vs_fa4_packed")
        if isinstance(speedup_vs_custom_masked, (int, float)) and math.isfinite(float(speedup_vs_custom_masked)):
            value = float(speedup_vs_custom_masked)
            min_speedup_vs_custom_masked = value if min_speedup_vs_custom_masked is None else min(min_speedup_vs_custom_masked, value)
        if isinstance(speedup_vs_fa4_packed, (int, float)) and math.isfinite(float(speedup_vs_fa4_packed)):
            value = float(speedup_vs_fa4_packed)
            min_speedup_vs_fa4_packed = value if min_speedup_vs_fa4_packed is None else min(min_speedup_vs_fa4_packed, value)

    if not primary_confetti_cases:
        primary_confetti_go_no_go = {
            "status": "not_applicable",
            "reason": "no disjoint_confetti cases were benchmarked",
        }
    elif primary_confetti_failures:
        primary_confetti_go_no_go = {
            "status": "fail",
            "reason": "one or more disjoint_confetti cases failed the direct_2d gate",
            "required_cases": len(primary_confetti_cases),
            "passing_cases": len(primary_confetti_cases) - len(primary_confetti_failures),
            "failing_cases": primary_confetti_failures,
        }
    else:
        primary_confetti_go_no_go = {
            "status": "pass",
            "reason": "all disjoint_confetti cases passed the direct_2d gate",
            "required_cases": len(primary_confetti_cases),
            "passing_cases": len(primary_confetti_cases),
            "evaluated_cases": primary_confetti_cases,
        }
        if min_speedup_vs_custom_masked is not None:
            primary_confetti_go_no_go["min_direct_2d_speedup_vs_custom_masked"] = min_speedup_vs_custom_masked
        if min_speedup_vs_fa4_packed is not None:
            primary_confetti_go_no_go["min_direct_2d_speedup_vs_fa4_packed"] = min_speedup_vs_fa4_packed

    runtime_routing_recommendation = {
        "status": "candidate" if primary_confetti_go_no_go.get("status") == "pass" else "do_not_route",
        "reason": (
            "primary disjoint_confetti cases passed; keep this path benchmark-only until runtime routing is explicitly planned"
            if primary_confetti_go_no_go.get("status") == "pass"
            else "keep this path benchmark-only; direct_2d has not cleared the primary disjoint_confetti gate"
        ),
    }

    return {
        "num_cases": len(case_payloads),
        "case_go_no_go_counts": status_counts,
        "primary_confetti_go_no_go": primary_confetti_go_no_go,
        "runtime_routing_recommendation": runtime_routing_recommendation,
    }
