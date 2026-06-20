from __future__ import annotations

import math
import os
import time
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
    mask_float = mask_rows.to(dtype=torch.float32)
    intersection = mask_float @ mask_float.transpose(0, 1)
    row_live = mask_float.sum(dim=1)
    union = row_live[:, None] + row_live[None, :] - intersection
    pair_mask = torch.triu(torch.ones((rows, rows), dtype=torch.bool, device=mask_rows.device), diagonal=1)
    pair_union = union[pair_mask]
    pair_intersection = intersection[pair_mask]
    scores = torch.where(pair_union > 0, pair_intersection / pair_union.clamp_min(1.0), torch.zeros_like(pair_union))
    return float(scores.mean().item()) if int(scores.numel()) > 0 else 0.0


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
    result = {
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
    return result


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


def _disjoint_confetti_offsets_have_no_overlap(
    *,
    support_k: int,
    islands_per_row: int,
    island_width: int,
) -> bool:
    if support_k <= 0 or islands_per_row <= 0 or island_width <= 0:
        return False
    if islands_per_row * island_width > support_k:
        return False
    max_start = max(0, support_k - island_width)
    search_space = max_start + 1
    stride = max(island_width + 1, support_k // max(1, islands_per_row))
    offsets = sorted({(island_idx * stride) % max(1, search_space) for island_idx in range(islands_per_row)})
    if len(offsets) != islands_per_row:
        return False
    if islands_per_row == 1:
        return True
    for idx, start in enumerate(offsets):
        next_start = offsets[(idx + 1) % len(offsets)]
        gap = (next_start - start) % search_space
        if gap < island_width:
            return False
    return True


def _build_disjoint_confetti_mask_vectorized(
    *,
    q_row_idx: torch.Tensor,
    support_k: int,
    islands_per_row: int,
    island_width: int,
    row_shift: int,
) -> torch.Tensor:
    if not _disjoint_confetti_offsets_have_no_overlap(
        support_k=support_k,
        islands_per_row=islands_per_row,
        island_width=island_width,
    ):
        raise ValueError("disjoint_confetti_vectorized_requires_non_overlapping_offsets")
    num_buckets, packed_q = q_row_idx.shape
    flat_rows = q_row_idx.reshape(-1).to(dtype=torch.int64)
    row_valid = flat_rows >= 0
    max_start = max(0, support_k - island_width)
    search_space = max_start + 1
    stride = max(island_width + 1, support_k // max(1, islands_per_row))
    base = (flat_rows.clamp_min(0) * max(1, row_shift)) % max(1, search_space)
    offsets = (
        torch.arange(islands_per_row, dtype=torch.int64, device=q_row_idx.device)
        * int(stride)
    ) % max(1, search_space)
    starts = (base.unsqueeze(1) + offsets.unsqueeze(0)) % max(1, search_space)
    cols = starts.unsqueeze(-1) + torch.arange(island_width, dtype=torch.int64, device=q_row_idx.device).view(1, 1, -1)
    cols = cols.reshape(flat_rows.numel(), islands_per_row * island_width)
    mask_2d = torch.zeros((flat_rows.numel(), support_k), dtype=torch.bool, device=q_row_idx.device)
    mask_2d.scatter_(1, cols, True)
    if bool((~row_valid).any().item()):
        mask_2d[~row_valid] = False
    return mask_2d.view(num_buckets, packed_q, support_k).contiguous()


def _fast_disjoint_confetti_geometry(
    *,
    q_row_idx: torch.Tensor,
    num_buckets: int,
    packed_q: int,
    support_k: int,
    islands_per_row: int,
    island_width: int,
    row_shift: int,
) -> dict[str, float | int]:
    flat_rows = q_row_idx.reshape(-1).to(dtype=torch.int64)
    valid_rows = flat_rows[flat_rows >= 0]
    valid_row_count = int(valid_rows.numel())
    live_per_row = min(support_k, islands_per_row * island_width)
    total_live_pairs = valid_row_count * live_per_row
    max_start = max(0, support_k - island_width)
    search_space = max_start + 1
    stride = max(island_width + 1, support_k // max(1, islands_per_row))
    if valid_row_count > 0 and islands_per_row > 1:
        base = (valid_rows * max(1, row_shift)) % max(1, search_space)
        offsets = (
            torch.arange(islands_per_row, dtype=torch.int64, device=q_row_idx.device)
            * int(stride)
        ) % max(1, search_space)
        starts = ((base.unsqueeze(1) + offsets.unsqueeze(0)) % max(1, search_space)).sort(dim=1).values
        gaps = starts[:, 1:] - (starts[:, :-1] + int(island_width))
        avg_gap = float(gaps.float().mean().item()) if int(gaps.numel()) > 0 else 0.0
        max_gap = int(gaps.max().item()) if int(gaps.numel()) > 0 else 0
    else:
        avg_gap = 0.0
        max_gap = 0
    return {
        "num_buckets": num_buckets,
        "valid_rows": valid_row_count,
        "live_pairs": total_live_pairs,
        "fill_rate": total_live_pairs / max(1, num_buckets * packed_q * support_k),
        "support_width": support_k,
        "avg_islands_per_row": float(islands_per_row if valid_row_count > 0 else 0),
        "max_islands_per_row": int(islands_per_row if valid_row_count > 0 else 0),
        "avg_gap": avg_gap,
        "max_gap": max_gap,
        "avg_pairwise_row_jaccard": -1.0,
    }


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
    if int(packed_out.shape[0]) > 0 and int(packed_out.shape[1]) > 0:
        packed_q = int(packed_out.shape[1])
        total_rows = int(q_length.sum().item())
        if total_rows <= int(packed_out.shape[0]) * packed_q:
            full_rows = total_rows // packed_q
            tail_rows = total_rows - full_rows * packed_q
            if (
                (full_rows == 0 or bool((q_length[:full_rows] == packed_q).all().item()))
                and (
                    full_rows >= int(q_length.numel())
                    or tail_rows == 0
                    or int(q_length[full_rows].item()) == tail_rows
                )
                and (
                    full_rows + (1 if tail_rows > 0 else 0) >= int(q_length.numel())
                    or bool((q_length[full_rows + (1 if tail_rows > 0 else 0):] == 0).all().item())
                )
            ):
                flat_rows = packed_out.reshape(-1, packed_out.shape[2], packed_out.shape[3])[:total_rows].float()
                return flat_rows.permute(1, 0, 2).contiguous()
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
    result = {
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
    return result


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
    mask_words: torch.Tensor | None = None,
    q_length: torch.Tensor,
    q_row_idx: torch.Tensor,
    tile_k: int = 32,
    total_rows: int | None = None,
    contiguous_q_rows: bool = False,
) -> tuple[dict[str, Any], dict[str, float | int]]:
    num_buckets, packed_q, num_heads, head_dim = q_buf.shape
    support_k = int(k_buf.shape[1])
    device = q_buf.device
    full_tile_span = max(1, math.ceil(support_k / max(1, tile_k)))
    if int(num_buckets) > 0:
        q_length_i64 = q_length.to(dtype=torch.int64).clamp_min(0)
        valid_row_mask = torch.arange(packed_q, device=device).view(1, packed_q) < q_length_i64.view(-1, 1)
        valid_mask_bool = mask_bool & valid_row_mask.unsqueeze(-1)
        bucket_has_any = valid_mask_bool.any(dim=1)
        bucket_union_k_tensor = bucket_has_any.sum(dim=1).to(dtype=torch.int32)
        valid_bucket = q_length > 0
        bucket_tile_span_tensor = ((bucket_union_k_tensor + int(tile_k) - 1) // max(1, int(tile_k))).clamp_min(1)
        all_valid_buckets_full_span = bool(
            ((~valid_bucket) | (bucket_tile_span_tensor >= int(full_tile_span))).all().item()
        )
        if all_valid_buckets_full_span:
            if mask_words is None:
                mask_words = _encode_mask_rows_to_words(mask_bool.reshape(num_buckets * packed_q, support_k)).view(
                    num_buckets,
                    packed_q,
                    -1,
                )
            bucket_index_tensor = torch.arange(num_buckets, dtype=torch.long, device=device)
            group_k_length = torch.full((num_buckets,), support_k, dtype=torch.int32, device=device)
            compact_groups = [
                {
                    "bucket_indices": bucket_index_tensor,
                    "packed_q": packed_q,
                    "support_rows": support_k,
                    "custom_q_buf": q_buf.contiguous(),
                    "custom_k_buf": k_buf.contiguous(),
                    "custom_v_buf": v_buf.contiguous(),
                    "custom_mask_bool": mask_bool.contiguous(),
                    "custom_mask_words": mask_words.contiguous(),
                    "custom_q_length": q_length.contiguous(),
                    "custom_k_length": group_k_length,
                    "is_full_passthrough_order": True,
                }
            ]
            valid_union_k = bucket_union_k_tensor[valid_bucket].to(dtype=torch.float32)
            total_live_pairs = int(mask_bool.sum().item())
            total_row_area = int(q_length.sum().item()) * support_k
            resolved_total_rows = (
                int(total_rows)
                if total_rows is not None
                else _total_rows_from_q_row_idx(
                    q_row_idx,
                    q_length,
                    fallback_total_rows=int(q_length.sum().item()),
                )
            )
            compact_geometry = {
                "direct_2d_compact_launch_groups": 1,
                "direct_2d_compact_avg_union_k": float(valid_union_k.mean().item()) if int(valid_union_k.numel()) else 0.0,
                "direct_2d_compact_max_union_k": int(bucket_union_k_tensor[valid_bucket].max().item()) if bool(valid_bucket.any().item()) else 0,
                "direct_2d_compact_avg_tile_span": float(bucket_tile_span_tensor[valid_bucket].to(dtype=torch.float32).mean().item()) if bool(valid_bucket.any().item()) else 0.0,
                "direct_2d_compact_avg_group_fill": total_live_pairs / max(1, total_row_area),
                "direct_2d_compact_case_fill_rate": total_live_pairs / max(1, total_row_area),
                "direct_2d_compact_buckets_compacted": 0,
                "direct_2d_compact_buckets_passthrough": int(valid_bucket.sum().item()),
            }
            result = {
                "groups": compact_groups,
                "total_buckets": num_buckets,
                "packed_q": packed_q,
                "custom_q_length": q_length.contiguous(),
                "q_row_idx": q_row_idx.contiguous(),
                "total_rows": resolved_total_rows,
                "q_row_idx_is_contiguous": bool(contiguous_q_rows),
            }
            return result, compact_geometry

    if int(num_buckets) > 0:
        compact_bucket_mask = valid_bucket & (bucket_tile_span_tensor < int(full_tile_span))
        full_bucket_tensor = torch.nonzero(valid_bucket & ~compact_bucket_mask, as_tuple=False).flatten()
        buckets_by_tile_span: dict[int, torch.Tensor] = {}
        for span_tensor in torch.unique(bucket_tile_span_tensor[compact_bucket_mask]).tolist():
            tile_span = int(span_tensor)
            buckets_by_tile_span[tile_span] = torch.nonzero(
                compact_bucket_mask & (bucket_tile_span_tensor == tile_span),
                as_tuple=False,
            ).flatten()

        live_pairs_by_bucket = valid_mask_bool.sum(dim=(1, 2)).to(dtype=torch.float32)
        effective_support = torch.where(
            compact_bucket_mask,
            bucket_union_k_tensor.clamp_min(1),
            torch.full_like(bucket_union_k_tensor, int(support_k)),
        ).to(dtype=torch.int64)
        compact_area_by_bucket = q_length_i64 * effective_support
        valid_compact_area = compact_area_by_bucket[valid_bucket]
        valid_compact_fill = live_pairs_by_bucket[valid_bucket] / valid_compact_area.clamp_min(1).to(dtype=torch.float32)
        valid_union_k_tensor = bucket_union_k_tensor[valid_bucket]
    else:
        compact_bucket_mask = torch.empty((0,), dtype=torch.bool, device=device)
        full_bucket_tensor = torch.empty((0,), dtype=torch.long, device=device)
        buckets_by_tile_span = {}
        compact_area_by_bucket = torch.empty((0,), dtype=torch.int64, device=device)
        valid_compact_fill = torch.empty((0,), dtype=torch.float32, device=device)
        valid_union_k_tensor = torch.empty((0,), dtype=torch.int32, device=device)

    compact_groups: list[dict[str, Any]] = []
    for tile_span in sorted(buckets_by_tile_span):
        bucket_index_tensor = buckets_by_tile_span[tile_span].to(dtype=torch.long)
        num_group_buckets = int(bucket_index_tensor.numel())
        if num_group_buckets == 0:
            continue
        group_union_k = bucket_union_k_tensor.index_select(0, bucket_index_tensor)
        max_union_k = int(group_union_k.max().item()) if num_group_buckets > 0 else 0
        group_q_buf = q_buf.index_select(0, bucket_index_tensor).contiguous()
        group_q_length = q_length.index_select(0, bucket_index_tensor).contiguous()
        if max_union_k > 0:
            selected_has_any = bucket_has_any.index_select(0, bucket_index_tensor)
            selected_has_any_i64 = selected_has_any.to(dtype=torch.int64)
            union_slot = selected_has_any_i64.cumsum(dim=1) - 1
            union_col_matrix = torch.zeros((num_group_buckets, max_union_k), dtype=torch.long, device=device)
            nonzero_pairs = torch.nonzero(selected_has_any, as_tuple=False)
            if int(nonzero_pairs.numel()) > 0:
                nz_bucket = nonzero_pairs[:, 0]
                nz_col = nonzero_pairs[:, 1]
                nz_slot = union_slot[nz_bucket, nz_col]
                union_col_matrix[nz_bucket, nz_slot] = nz_col
            union_valid = torch.arange(max_union_k, dtype=torch.int32, device=device).view(1, max_union_k) < group_union_k.view(
                -1,
                1,
            )
            gather_kv_idx = union_col_matrix.view(num_group_buckets, max_union_k, 1, 1).expand(
                -1,
                -1,
                num_heads,
                head_dim,
            )
            group_k_buf = torch.gather(k_buf.index_select(0, bucket_index_tensor), 1, gather_kv_idx).contiguous()
            group_v_buf = torch.gather(v_buf.index_select(0, bucket_index_tensor), 1, gather_kv_idx).contiguous()
            group_k_buf.masked_fill_(~union_valid.view(num_group_buckets, max_union_k, 1, 1), 0)
            group_v_buf.masked_fill_(~union_valid.view(num_group_buckets, max_union_k, 1, 1), 0)
            gather_mask_idx = union_col_matrix.view(num_group_buckets, 1, max_union_k).expand(
                -1,
                packed_q,
                -1,
            )
            group_mask_bool = torch.gather(
                mask_bool.index_select(0, bucket_index_tensor),
                2,
                gather_mask_idx,
            ).contiguous()
            group_mask_bool &= union_valid.view(num_group_buckets, 1, max_union_k)
            group_k_length = group_union_k.contiguous()
        else:
            group_k_buf = torch.zeros(
                (num_group_buckets, max_union_k, num_heads, head_dim),
                dtype=k_buf.dtype,
                device=device,
            )
            group_v_buf = torch.zeros(
                (num_group_buckets, max_union_k, num_heads, head_dim),
                dtype=v_buf.dtype,
                device=device,
            )
            group_mask_bool = torch.zeros(
                (num_group_buckets, packed_q, max_union_k),
                dtype=torch.bool,
                device=device,
            )
            group_k_length = torch.zeros((num_group_buckets,), dtype=torch.int32, device=device)

        group_mask_words = _encode_mask_rows_to_words(
            group_mask_bool.reshape(num_group_buckets * packed_q, max_union_k)
        ).view(
            num_group_buckets,
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

    if int(full_bucket_tensor.numel()) > 0:
        bucket_index_tensor = full_bucket_tensor.to(dtype=torch.long)
        num_full_buckets = int(bucket_index_tensor.numel())
        group_k_length = torch.full((num_full_buckets,), support_k, dtype=torch.int32, device=device)
        group_mask_bool = mask_bool.index_select(0, bucket_index_tensor).contiguous()
        group_mask_words = _encode_mask_rows_to_words(
            group_mask_bool.reshape(num_full_buckets * packed_q, support_k)
        ).view(
            num_full_buckets,
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

    valid_union_k_positive = valid_union_k_tensor[valid_union_k_tensor > 0]
    total_live_pairs = int(mask_bool.sum().item())
    total_compact_area = int(compact_area_by_bucket[valid_bucket].sum().item()) if int(num_buckets) > 0 else 0
    compact_geometry = {
        "direct_2d_compact_launch_groups": len(compact_groups),
        "direct_2d_compact_avg_union_k": (
            float(valid_union_k_positive.to(dtype=torch.float32).mean().item()) if int(valid_union_k_positive.numel()) else 0.0
        ),
        "direct_2d_compact_max_union_k": int(valid_union_k_positive.max().item()) if int(valid_union_k_positive.numel()) else 0,
        "direct_2d_compact_avg_tile_span": (
            float(bucket_tile_span_tensor[valid_bucket].to(dtype=torch.float32).mean().item()) if bool(valid_bucket.any().item()) else 0.0
        ),
        "direct_2d_compact_avg_group_fill": float(valid_compact_fill.mean().item()) if int(valid_compact_fill.numel()) else 0.0,
        "direct_2d_compact_case_fill_rate": total_live_pairs / max(1, total_compact_area),
        "direct_2d_compact_buckets_compacted": int(compact_bucket_mask.sum().item()) if int(num_buckets) > 0 else 0,
        "direct_2d_compact_buckets_passthrough": int(full_bucket_tensor.numel()),
    }
    result = {
        "groups": compact_groups,
        "total_buckets": num_buckets,
        "packed_q": packed_q,
        "custom_q_length": q_length.contiguous(),
        "q_row_idx": q_row_idx.contiguous(),
        "total_rows": (
            int(total_rows)
            if total_rows is not None
            else _total_rows_from_q_row_idx(
                q_row_idx,
                q_length,
                fallback_total_rows=int(q_length.sum().item()),
            )
        ),
        "q_row_idx_is_contiguous": bool(contiguous_q_rows),
    }, compact_geometry
    return result


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
    payload_variants: tuple[str, ...] | None = None,
    fast_geometry: bool = False,
) -> dict[str, Any]:
    if case_family not in {"disjoint_confetti", "compact_control"}:
        raise ValueError(f"unsupported case_family {case_family!r}")
    if seqlen <= 0 or heads <= 0 or head_dim <= 0 or packed_q <= 0 or support_k <= 0:
        raise ValueError("seqlen, heads, head_dim, packed_q, and support_k must be positive")
    if islands_per_row <= 0 or island_width <= 0:
        raise ValueError("islands_per_row and island_width must be positive")
    if payload_variants is None:
        payload_variant_set = {
            "custom_masked",
            "direct_2d",
            "direct_2d_compact",
            "shared_support",
        }
    else:
        payload_variant_set = set(payload_variants)

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
        if _disjoint_confetti_offsets_have_no_overlap(
            support_k=support_k,
            islands_per_row=islands_per_row,
            island_width=island_width,
        ):
            mask_bool = _build_disjoint_confetti_mask_vectorized(
                q_row_idx=q_row_idx,
                support_k=support_k,
                islands_per_row=islands_per_row,
                island_width=island_width,
                row_shift=row_shift,
            )
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
    k_row_idx = torch.arange(num_buckets * support_k, dtype=torch.int32, device=device).view(num_buckets, support_k)
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
        "custom_k_row_idx": k_row_idx.contiguous(),
        "total_rows": seqlen,
    }
    if device.type == "cuda":
        torch.cuda.synchronize()
    diagnostic_geometry_t0 = time.perf_counter()
    if fast_geometry and case_family == "disjoint_confetti" and _disjoint_confetti_offsets_have_no_overlap(
        support_k=support_k,
        islands_per_row=islands_per_row,
        island_width=island_width,
    ):
        geometry = _fast_disjoint_confetti_geometry(
            q_row_idx=q_row_idx,
            num_buckets=num_buckets,
            packed_q=packed_q,
            support_k=support_k,
            islands_per_row=islands_per_row,
            island_width=island_width,
            row_shift=row_shift,
        )
    else:
        geometry = _mask_geometry(mask_bool, q_length)
    if device.type == "cuda":
        torch.cuda.synchronize()
    diagnostic_geometry_seconds = time.perf_counter() - diagnostic_geometry_t0
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
    result = {
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
        "geometry": geometry,
        "diagnostic_geometry_seconds": diagnostic_geometry_seconds,
    }
    if "direct_2d" in payload_variant_set:
        direct_2d_bucket, direct_2d_geometry = _build_direct_2d_bucket(
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
            mask_bool=mask_bool,
            q_length=q_length,
            q_row_idx=q_row_idx,
        )
        geometry.update(direct_2d_geometry)
        result["direct_2d_bucket"] = direct_2d_bucket
    if "direct_2d_compact" in payload_variant_set:
        direct_2d_compact_payload, direct_2d_compact_geometry = _build_direct_2d_compact_payload(
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
            mask_bool=mask_bool,
            mask_words=mask_words,
            q_length=q_length,
            q_row_idx=q_row_idx,
            total_rows=seqlen,
            contiguous_q_rows=True,
        )
        geometry.update(direct_2d_compact_geometry)
        result["direct_2d_compact_payload"] = direct_2d_compact_payload
    if "custom_masked" in payload_variant_set:
        result["micro_bucket"] = _build_micro_bucket(
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
            mask_bool=mask_bool,
            q_length=q_length,
            q_row_idx=q_row_idx,
        )
    if "shared_support" in payload_variant_set:
        result["shared_support_buckets"] = _build_shared_support_buckets(
            q_buf=q_buf,
            k_buf=k_buf,
            v_buf=v_buf,
            mask_bool=mask_bool,
            q_length=q_length,
        )
    return result


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


def _direct_2d_tc_mode() -> str:
    value = os.environ.get("FLASH_ATTN_HSA_EXPLICIT_DIRECT_2D_TC", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "on"
    if value in {"0", "false", "no", "off"}:
        return "off"
    return "auto"


def _can_use_direct_2d_tc_forward(bucket: dict[str, Any]) -> bool:
    mode = _direct_2d_tc_mode()
    if mode == "off":
        return False
    q_buf = bucket.get("custom_q_buf")
    k_buf = bucket.get("custom_k_buf")
    v_buf = bucket.get("custom_v_buf")
    if not all(isinstance(tensor, torch.Tensor) for tensor in (q_buf, k_buf, v_buf)):
        return False
    if not all(tensor.is_cuda for tensor in (q_buf, k_buf, v_buf)):
        return False
    head_dim = int(q_buf.shape[-1])
    support_rows = int(bucket.get("support_rows", 0))
    tc_support_rows = 0 < support_rows <= 128
    if head_dim == 64:
        try:
            high_support_max = int(os.environ.get("FLASH_ATTN_HSA_EXPLICIT_DIRECT_2D_TC_HIGH_SUPPORT_MAX", "1024"))
        except ValueError:
            high_support_max = 1024
        high_support_max = max(512, min(high_support_max, 2048))
        tc_support_rows = tc_support_rows or (512 <= support_rows <= high_support_max)
    return (
        int(bucket.get("packed_q", 0)) == 16
        and tc_support_rows
        and head_dim in (64, 128)
        and int(k_buf.shape[-1]) == head_dim
        and int(v_buf.shape[-1]) == head_dim
        and q_buf.dtype in (torch.float16, torch.bfloat16)
        and k_buf.dtype == q_buf.dtype
        and v_buf.dtype == q_buf.dtype
        and isinstance(bucket.get("q_row_idx"), torch.Tensor)
        and isinstance(bucket.get("custom_k_row_idx"), torch.Tensor)
        and isinstance(bucket.get("custom_q_length"), torch.Tensor)
        and isinstance(bucket.get("custom_k_length"), torch.Tensor)
        and isinstance(bucket.get("custom_mask_words"), torch.Tensor)
    )


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
    groups = compact_payload["groups"]
    if (
        len(groups) == 1
        and bool(groups[0].get("is_full_passthrough_order", False))
        and bool(compact_payload.get("q_row_idx_is_contiguous", False))
    ):
        if _can_use_direct_2d_tc_forward(full_bucket):
            return _run_direct_2d_tc_forward(full_bucket, softmax_scale=softmax_scale)
        packed_out = _run_direct_2d_packed_forward(groups[0], softmax_scale=softmax_scale).float()
        return _flatten_valid_packed_rows(packed_out, compact_payload["custom_q_length"])
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
    for group in groups:
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


def _run_direct_2d_tc_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
        _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel,
    )

    q_buf = bucket["custom_q_buf"]
    k_buf = bucket["custom_k_buf"]
    v_buf = bucket["custom_v_buf"]
    q_flat = q_buf.reshape(-1, q_buf.shape[2], q_buf.shape[3]).contiguous()
    k_flat = k_buf.reshape(-1, k_buf.shape[2], k_buf.shape[3]).contiguous()
    v_flat = v_buf.reshape(-1, v_buf.shape[2], v_buf.shape[3]).contiguous()
    total_rows = int(bucket.get("total_rows", q_flat.shape[0]))
    out_flat = torch.empty((q_flat.shape[0], q_flat.shape[1], v_flat.shape[2]), dtype=torch.float32, device=q_flat.device)
    lse_flat = torch.empty((q_flat.shape[0], q_flat.shape[1]), dtype=torch.float32, device=q_flat.device)
    _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel(
        q_flat,
        k_flat,
        v_flat,
        bucket["q_row_idx"],
        bucket["custom_k_row_idx"],
        bucket["custom_q_length"],
        bucket["custom_k_length"],
        bucket["custom_mask_words"],
        out_flat,
        lse_flat,
        softmax_scale=softmax_scale,
    )
    return out_flat[:total_rows].permute(1, 0, 2).contiguous()


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
    check_correctness: bool = True,
) -> dict[str, Any]:
    valid_variants = {
        "dense",
        "custom_masked",
        "fa4_packed",
        "direct_2d",
        "direct_2d_compact",
        "direct_2d_tc",
        "shared_support",
    }
    if any(variant not in valid_variants for variant in variants):
        unknown = sorted(set(variants) - valid_variants)
        raise ValueError(f"unknown variants {unknown}")

    normalized_device = _normalize_device(device)
    payload_t0 = time.perf_counter()
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
        device=normalized_device,
        dtype=dtype,
        seed=seed,
        payload_variants=tuple(variants),
        fast_geometry=not check_correctness,
    )
    if normalized_device.type == "cuda":
        torch.cuda.synchronize()
    payload_build_seconds = time.perf_counter() - payload_t0
    diagnostic_geometry_seconds = float(case_payload.get("diagnostic_geometry_seconds", 0.0))
    payload_build_excluding_diagnostic_geometry_seconds = max(
        0.0,
        payload_build_seconds - diagnostic_geometry_seconds,
    )
    full_bucket = case_payload["full_bucket"]
    softmax_scale = head_dim ** (-0.5)
    dense_out = (
        _run_dense_explicit_bucket_forward(full_bucket, softmax_scale=softmax_scale)
        if check_correctness
        else None
    )

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
        "direct_2d_tc": lambda: _run_direct_2d_tc_forward(
            full_bucket,
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
            if dense_out is not None:
                diff = (dense_out.float() - out.float()).abs()
                output_max_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
                output_mean_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0
            else:
                output_max_diff = float("nan")
                output_mean_diff = float("nan")
            results[variant] = {
                "status": "measured",
                "fwd_ms": _measure_ms(runner, warmup_iters, benchmark_iters),
                "output_max_diff": output_max_diff,
                "output_mean_diff": output_mean_diff,
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
        "payload_build_seconds": payload_build_seconds,
        "payload_build_total_seconds": payload_build_seconds,
        "payload_build_excluding_diagnostic_geometry_seconds": payload_build_excluding_diagnostic_geometry_seconds,
        "diagnostic_geometry_seconds": diagnostic_geometry_seconds,
        "check_correctness": bool(check_correctness),
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
    for variant_name in ("direct_2d", "direct_2d_compact", "direct_2d_tc"):
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
