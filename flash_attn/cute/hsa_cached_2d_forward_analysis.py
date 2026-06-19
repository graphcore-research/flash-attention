from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON_LSE_PUBLIC_TO_FLAT = True
except Exception:  # pragma: no cover - optional Triton runtime
    triton = None
    tl = None
    _HAS_TRITON_LSE_PUBLIC_TO_FLAT = False

from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
    _can_use_synthetic_2d_masked_fwd,
    _run_cached_cast_two_rows_kernel,
    _run_cached_cast_three_rows_kernel,
    _run_cached_cast_rows_kernel,
    _run_cached_finalize_output_rows_kernel,
    _run_cached_init_output_rows_kernel,
    _run_cached_lse_flat_to_public_kernel,
    _run_cached_lse_public_to_flat_kernel,
    _run_cached_zero_three_rows_kernel,
    _run_cached_zero_two_rows_kernel,
    _run_cached_zero_rows_kernel,
    _run_synthetic_2d_exact_gather_scatter_tc_fwd_kernel,
    _run_synthetic_2d_exact_tail_gather_scatter_tc_fwd_kernel,
    _run_synthetic_combine_scatter_rows_kernel,
    _run_synthetic_2d_masked_fwd_kernel,
    _run_synthetic_2d_masked_gather_combine_fwd_kernel,
    _run_synthetic_2d_masked_gather_fwd_kernel,
    _run_synthetic_2d_masked_gather_scatter_fwd_kernel,
    _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel,
    _run_synthetic_pack_kv_rows_kernel,
    _run_synthetic_pack_rows_kernel,
)
from flash_attn.cute.hsa_explicit_2d_sparse_analysis import _partition_packed_rows_by_support


if _HAS_TRITON_LSE_PUBLIC_TO_FLAT:

    @triton.jit
    def _triton_lse_public_to_flat_kernel(
        src_public_lse,
        dst_flat_lse,
        total_elems: tl.constexpr,
        seqlen: tl.constexpr,
        num_heads: tl.constexpr,
        block_elems: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_elems + tl.arange(0, block_elems)
        mask = offsets < total_elems
        head_idx = offsets % num_heads
        flat_row = offsets // num_heads
        batch_idx = flat_row // seqlen
        token_idx = flat_row - batch_idx * seqlen
        src_offsets = batch_idx * num_heads * seqlen + head_idx * seqlen + token_idx
        values = tl.load(src_public_lse + src_offsets, mask=mask, other=0.0)
        tl.store(dst_flat_lse + offsets, values, mask=mask)


@dataclass(frozen=True)
class CachedPackingPolicy:
    direct_density_threshold: float = 0.85
    window_density_threshold: float = 0.60
    k_gap_threshold: float = 1.25
    max_rows_per_group: int = 16
    tile_k: int = 32
    max_union_k_direct: int = 64
    max_union_k_2d: int = 1024
    union_kernel: str = "tc16x32"
    exact_kernel_family: str = "tc8x8"
    exact_min_rows: int = 6
    residual_mode: str = "fused_tail"


_EXACT_KERNEL_SPECS = {
    "tc16x16": {
        "rows_per_range": 16,
        "keys_per_tile": 16,
        "default_min_rows": 8,
    },
    "tc8x8": {
        "rows_per_range": 8,
        "keys_per_tile": 8,
        "default_min_rows": 6,
    },
}


def _resolve_exact_kernel_spec(
    family: str,
    *,
    min_rows: int | None = None,
) -> dict[str, int | str]:
    family_name = str(family)
    if family_name not in _EXACT_KERNEL_SPECS:
        raise ValueError(f"unsupported exact_kernel_family {family_name!r}")
    base = dict(_EXACT_KERNEL_SPECS[family_name])
    resolved_min_rows = int(base["default_min_rows"] if min_rows is None else min_rows)
    rows_per_range = int(base["rows_per_range"])
    if resolved_min_rows <= 0 or resolved_min_rows > rows_per_range:
        raise ValueError(
            f"exact_min_rows must be between 1 and {rows_per_range} for {family_name}"
        )
    base["family"] = family_name
    base["min_rows"] = resolved_min_rows
    return base


def _flatten_row_tensor(rows: torch.Tensor) -> torch.Tensor:
    if rows.ndim == 4:
        return rows.reshape(-1, rows.shape[2], rows.shape[3]).contiguous()
    if rows.ndim == 3:
        return rows.contiguous()
    raise ValueError(f"expected rank-3 or rank-4 row tensor, got shape {tuple(rows.shape)}")


def _env_mode(name: str, default: str = "auto") -> str:
    value = os.environ.get(name, default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "on"
    if value in {"0", "false", "no", "off"}:
        return "off"
    return "auto"


def _is_env_forced_on(name: str) -> bool:
    return _env_mode(name) == "on"


def _is_env_enabled(name: str, default: str = "auto") -> bool:
    return _env_mode(name, default=default) != "off"


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


def _build_group_mask_words(
    support_lists: list[list[int]],
    row_group: list[int],
    union_rows: list[int],
) -> tuple[torch.Tensor, float]:
    union_pos = {row_idx: pos for pos, row_idx in enumerate(union_rows)}
    word_count = max(1, (len(union_rows) + 31) // 32)
    row_word_values: list[list[int]] = []
    live_pairs = 0
    for entry_idx in row_group:
        row_words = [0] * word_count
        row_live_pairs = 0
        for support_row in support_lists[entry_idx]:
            support_pos = union_pos[int(support_row)]
            row_words[support_pos // 32] |= 1 << (support_pos % 32)
            row_live_pairs += 1
        row_word_values.append(row_words)
        live_pairs += row_live_pairs
    mask_words = torch.tensor(row_word_values, dtype=torch.int64).to(dtype=torch.int32)
    fill = float(live_pairs) / max(1, len(row_group) * len(union_rows))
    return mask_words, fill


def _decode_mask_words_to_bool(mask_words: torch.Tensor, width: int) -> torch.Tensor:
    if width <= 0:
        return torch.zeros((int(mask_words.shape[0]), 0), dtype=torch.bool)
    bit_offsets = torch.arange(32, dtype=torch.int64, device=mask_words.device).view(1, 1, 32)
    mask_words_u32 = mask_words.to(dtype=torch.int64) & ((1 << 32) - 1)
    unpacked = ((mask_words_u32.unsqueeze(-1) >> bit_offsets) & 1).to(dtype=torch.bool)
    return unpacked.view(int(mask_words.shape[0]), -1)[:, :width].contiguous()


def _extract_exact_dense_tile_ranges(
    q_rows: list[int],
    support_lists: list[list[int]],
    *,
    rows_per_range: int,
    keys_per_tile: int,
    min_rows: int,
) -> tuple[list[dict[str, Any]], list[list[int]], int]:
    if rows_per_range <= 0 or keys_per_tile <= 0 or min_rows <= 0:
        raise ValueError("exact dense extractor parameters must be positive")
    residual_sets = [set(int(value) for value in support_list) for support_list in support_lists]
    exact_ranges: list[dict[str, Any]] = []
    exact_live_pairs = 0

    while True:
        active_rows = [row_idx for row_idx, support in enumerate(residual_sets) if len(support) >= keys_per_tile]
        if len(active_rows) < min_rows:
            break

        best_group: list[int] | None = None
        best_common: set[int] | None = None
        best_score: tuple[int, int, int, int] | None = None

        for seed_idx in active_rows:
            seed_support = residual_sets[seed_idx]
            candidate_rows = sorted(
                (row_idx for row_idx in active_rows if row_idx != seed_idx),
                key=lambda row_idx: (
                    len(seed_support.intersection(residual_sets[row_idx])),
                    len(residual_sets[row_idx]),
                    -row_idx,
                ),
                reverse=True,
            )
            subgroup = [seed_idx]
            common_support = set(seed_support)
            for row_idx in candidate_rows:
                if len(subgroup) >= rows_per_range:
                    break
                candidate_common = common_support.intersection(residual_sets[row_idx])
                if len(candidate_common) < keys_per_tile:
                    continue
                subgroup.append(row_idx)
                common_support = candidate_common
            if len(subgroup) < min_rows:
                continue
            tile_count = len(common_support) // keys_per_tile
            if tile_count <= 0:
                continue
            score = (
                len(subgroup) * tile_count * keys_per_tile,
                len(subgroup),
                tile_count,
                len(common_support),
            )
            if best_score is None or score > best_score:
                best_group = subgroup
                best_common = common_support
                best_score = score

        if best_group is None or best_common is None or best_score is None:
            break

        sorted_common = sorted(best_common)
        tile_count = best_score[2]
        tiles = [
            sorted_common[tile_idx * keys_per_tile : (tile_idx + 1) * keys_per_tile]
            for tile_idx in range(tile_count)
        ]
        covered_keys = set(key for tile in tiles for key in tile)
        exact_ranges.append(
            {
                "q_rows": [int(q_rows[row_idx]) for row_idx in best_group],
                "tiles": tiles,
            }
        )
        exact_live_pairs += len(best_group) * len(covered_keys)
        for row_idx in best_group:
            residual_sets[row_idx].difference_update(covered_keys)

    residual_support_lists = [sorted(support_set) for support_set in residual_sets]
    return exact_ranges, residual_support_lists, exact_live_pairs


def _build_masked_tail_tiles(
    tail_support_lists: list[list[int]],
    *,
    rows_per_range: int,
    keys_per_tile: int,
) -> tuple[list[list[int]], list[list[int]], int]:
    if rows_per_range <= 0 or keys_per_tile <= 0:
        raise ValueError("tail tile builder parameters must be positive")
    flat_support_rows: list[int] = []
    flat_row_bits: list[int] = []
    for row_idx, support_list in enumerate(tail_support_lists):
        if row_idx >= rows_per_range:
            break
        if not support_list:
            continue
        row_bit = 1 << row_idx
        flat_support_rows.extend(int(support_row) for support_row in support_list)
        flat_row_bits.extend(row_bit for _ in support_list)
    if not flat_support_rows:
        return [], [], 0
    support_row_tensor = torch.tensor(flat_support_rows, dtype=torch.int32)
    row_bit_tensor = torch.tensor(flat_row_bits, dtype=torch.int64)
    sort_idx = torch.argsort(support_row_tensor, stable=True)
    sorted_support_rows = support_row_tensor[sort_idx]
    sorted_row_bits = row_bit_tensor[sort_idx]
    unique_support_rows, counts = torch.unique_consecutive(sorted_support_rows, return_counts=True)
    counts64 = counts.to(dtype=torch.int64)
    prefix_sums = torch.cumsum(sorted_row_bits, dim=0)
    segment_ends = torch.cumsum(counts64, dim=0) - 1
    segment_starts = segment_ends - counts64 + 1
    previous_prefix = torch.zeros_like(segment_ends)
    has_previous = segment_starts > 0
    if bool(has_previous.any().item()):
        previous_prefix[has_previous] = prefix_sums[segment_starts[has_previous] - 1]
    row_masks = prefix_sums[segment_ends] - previous_prefix
    support_row_list = [int(support_row) for support_row in unique_support_rows.tolist()]
    row_mask_list = [int(row_mask) for row_mask in row_masks.tolist()]
    tail_tiles: list[list[int]] = []
    tail_mask_rows: list[list[int]] = []
    tail_live_pairs = 0
    for chunk_start in range(0, len(support_row_list), keys_per_tile):
        chunk_support_rows = support_row_list[chunk_start : chunk_start + keys_per_tile]
        chunk_row_masks = row_mask_list[chunk_start : chunk_start + keys_per_tile]
        tile_rows = [-1] * keys_per_tile
        row_masks = [0] * rows_per_range
        tail_live_pairs += int(sum(int(row_mask).bit_count() for row_mask in chunk_row_masks))
        for col_idx, (support_row, row_mask) in enumerate(zip(chunk_support_rows, chunk_row_masks, strict=True)):
            tile_rows[col_idx] = int(support_row)
            while row_mask:
                lowest_bit = row_mask & -row_mask
                row_idx = lowest_bit.bit_length() - 1
                if row_idx < rows_per_range:
                    row_masks[row_idx] |= 1 << col_idx
                row_mask ^= lowest_bit
        tail_tiles.append(tile_rows)
        tail_mask_rows.append(row_masks)
    return tail_tiles, tail_mask_rows, tail_live_pairs


def _build_rowwise_tail_tiles(
    tail_support_lists: list[list[int]],
    *,
    rows_per_range: int,
    keys_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    tail_tile_tensors: list[torch.Tensor] = []
    tail_mask_tensors: list[torch.Tensor] = []
    tail_live_pairs = 0
    for row_idx, support_list in enumerate(tail_support_lists):
        if row_idx >= rows_per_range:
            break
        if not support_list:
            continue
        support_tensor = torch.tensor(support_list, dtype=torch.int32)
        support_count = int(support_tensor.numel())
        if support_count <= 0:
            continue
        tile_count = (support_count + keys_per_tile - 1) // keys_per_tile
        padded = torch.full((tile_count * keys_per_tile,), -1, dtype=torch.int32)
        padded[:support_count] = support_tensor
        tile_rows_tensor = padded.view(tile_count, keys_per_tile)
        row_mask_tensor = torch.zeros((tile_count, rows_per_range), dtype=torch.int32)
        row_mask_tensor[:, row_idx] = (1 << keys_per_tile) - 1
        tail_count = support_count - ((tile_count - 1) * keys_per_tile)
        row_mask_tensor[-1, row_idx] = (1 << tail_count) - 1
        tail_tile_tensors.append(tile_rows_tensor)
        tail_mask_tensors.append(row_mask_tensor)
        tail_live_pairs += support_count
    if not tail_tile_tensors:
        return (
            torch.empty((0, keys_per_tile), dtype=torch.int32),
            torch.empty((0, rows_per_range), dtype=torch.int32),
            0,
        )
    return (
        torch.cat(tail_tile_tensors, dim=0).contiguous(),
        torch.cat(tail_mask_tensors, dim=0).contiguous(),
        tail_live_pairs,
    )


def _extract_fused_exact_tail_ranges(
    q_rows: list[int],
    support_lists: list[list[int]],
    *,
    rows_per_range: int,
    keys_per_tile: int,
    min_rows: int,
) -> tuple[list[dict[str, Any]], int, int]:
    if rows_per_range <= 0 or keys_per_tile <= 0 or min_rows <= 0:
        raise ValueError("fused exact/tail extractor parameters must be positive")
    support_sets = [set(int(value) for value in support_list) for support_list in support_lists]
    active_rows = [row_idx for row_idx, support_set in enumerate(support_sets) if support_set]
    fused_ranges: list[dict[str, Any]] = []
    exact_live_pairs = 0
    tail_live_pairs = 0
    rowwise_tail_live_pairs_threshold = int(
        os.environ.get(
            "FLASH_ATTN_HSA_FUSED_TAIL_ROWWISE_LIVE_PAIRS_THRESHOLD",
            "1024",
        )
    )

    def _ordered_rows(row_indices: list[int]) -> list[int]:
        return sorted(
            row_indices,
            key=lambda idx: (
                support_lists[idx][0] if support_lists[idx] else -1,
                support_lists[idx][-1] if support_lists[idx] else -1,
                len(support_lists[idx]),
                idx,
            ),
        )

    ordered_active_rows = _ordered_rows(active_rows)
    for group_start in range(0, len(ordered_active_rows), rows_per_range):
        ordered_group = ordered_active_rows[group_start : group_start + rows_per_range]
        if not ordered_group:
            continue
        exact_tiles: list[list[int]] = []
        covered_keys: set[int] = set()
        group_live_pairs = int(sum(len(support_sets[row_idx]) for row_idx in ordered_group))
        use_rowwise_tail = (
            rowwise_tail_live_pairs_threshold > 0
            and group_live_pairs >= rowwise_tail_live_pairs_threshold
        )
        if not use_rowwise_tail and len(ordered_group) >= min_rows:
            common_support = set(support_sets[ordered_group[0]])
            for row_idx in ordered_group[1:]:
                common_support.intersection_update(support_sets[row_idx])
                if len(common_support) < keys_per_tile:
                    break
            if len(common_support) >= keys_per_tile:
                sorted_common = sorted(common_support)
                tile_count = len(sorted_common) // keys_per_tile
                exact_tiles = [
                    sorted_common[tile_idx * keys_per_tile : (tile_idx + 1) * keys_per_tile]
                    for tile_idx in range(tile_count)
                ]
                covered_keys = {key for tile in exact_tiles for key in tile}
                exact_live_pairs += len(ordered_group) * len(covered_keys)
        tail_support_lists = [
            sorted(support_sets[row_idx].difference(covered_keys))
            for row_idx in ordered_group
        ]
        if use_rowwise_tail:
            tail_tile_tensor, tail_mask_tensor, group_tail_live_pairs = _build_rowwise_tail_tiles(
                tail_support_lists,
                rows_per_range=rows_per_range,
                keys_per_tile=keys_per_tile,
            )
            tail_tiles: list[list[int]] = []
            tail_mask_rows: list[list[int]] = []
        else:
            tail_tiles, tail_mask_rows, group_tail_live_pairs = _build_masked_tail_tiles(
                tail_support_lists,
                rows_per_range=rows_per_range,
                keys_per_tile=keys_per_tile,
            )
            tail_tile_tensor = None
            tail_mask_tensor = None
        fused_ranges.append(
            {
                "q_rows": [int(q_rows[row_idx]) for row_idx in ordered_group],
                "exact_tiles": exact_tiles,
                "tail_tiles": tail_tiles,
                "tail_mask_rows": tail_mask_rows,
                "tail_tile_tensor": tail_tile_tensor,
                "tail_mask_tensor": tail_mask_tensor,
                }
            )
        tail_live_pairs += int(group_tail_live_pairs)

    return fused_ranges, exact_live_pairs, tail_live_pairs


def _materialize_exact_dense_range_tensors(
    *,
    device: torch.device,
    exact_ranges: list[dict[str, Any]],
    rows_per_range: int,
    keys_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_count = len(exact_ranges)
    tile_count = sum(len(range_entry["tiles"]) for range_entry in exact_ranges)
    q_row_idx_cpu = [[-1] * rows_per_range for _ in range(range_count)]
    q_length_cpu = [0] * range_count
    tile_ptr_cpu = [0] * (range_count + 1)
    tile_k_row_idx_cpu = [[-1] * keys_per_tile for _ in range(tile_count)]
    tile_offset = 0
    for range_idx, range_entry in enumerate(exact_ranges):
        q_rows = [int(value) for value in range_entry["q_rows"]]
        tiles = range_entry["tiles"]
        q_length_cpu[range_idx] = len(q_rows)
        q_row_idx_cpu[range_idx][: len(q_rows)] = q_rows
        for tile in tiles:
            tile_values = [int(value) for value in tile]
            tile_k_row_idx_cpu[tile_offset][: len(tile_values)] = tile_values
            tile_offset += 1
        tile_ptr_cpu[range_idx + 1] = tile_offset
    return (
        torch.tensor(q_row_idx_cpu, dtype=torch.int32, device=device).contiguous(),
        torch.tensor(q_length_cpu, dtype=torch.int32, device=device).contiguous(),
        torch.tensor(tile_ptr_cpu, dtype=torch.int32, device=device).contiguous(),
        torch.tensor(tile_k_row_idx_cpu, dtype=torch.int32, device=device).contiguous(),
    )


def _materialize_fused_exact_tail_range_tensors(
    *,
    device: torch.device,
    fused_ranges: list[dict[str, Any]],
    rows_per_range: int,
    keys_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_count = len(fused_ranges)
    exact_tile_count = sum(len(range_entry.get("exact_tiles", ())) for range_entry in fused_ranges)
    tail_tile_count = 0
    for range_entry in fused_ranges:
        tail_tile_tensor = range_entry.get("tail_tile_tensor")
        if isinstance(tail_tile_tensor, torch.Tensor):
            tail_tile_count += int(tail_tile_tensor.shape[0])
        else:
            tail_tile_count += len(range_entry.get("tail_tiles", ()))
    q_row_idx_cpu = torch.full((range_count, rows_per_range), -1, dtype=torch.int32)
    q_length_cpu = torch.zeros((range_count,), dtype=torch.int32)
    exact_tile_ptr_cpu = torch.zeros((range_count + 1,), dtype=torch.int32)
    exact_k_row_idx_cpu = torch.full((exact_tile_count, keys_per_tile), -1, dtype=torch.int32)
    tail_tile_ptr_cpu = torch.zeros((range_count + 1,), dtype=torch.int32)
    tail_k_row_idx_cpu = torch.full((tail_tile_count, keys_per_tile), -1, dtype=torch.int32)
    tail_mask_words_cpu = torch.zeros((tail_tile_count, rows_per_range, 1), dtype=torch.int32)
    exact_tile_offset = 0
    tail_tile_offset = 0
    for range_idx, range_entry in enumerate(fused_ranges):
        q_rows = [int(value) for value in range_entry["q_rows"]]
        exact_tiles = [[int(value) for value in tile] for tile in range_entry.get("exact_tiles", ())]
        q_length_cpu[range_idx] = int(len(q_rows))
        if q_rows:
            q_row_idx_cpu[range_idx, : len(q_rows)] = torch.tensor(q_rows, dtype=torch.int32)
        if exact_tiles:
            exact_tile_tensor = torch.tensor(exact_tiles, dtype=torch.int32)
            exact_tile_count_range = int(exact_tile_tensor.shape[0])
            exact_k_row_idx_cpu[exact_tile_offset : exact_tile_offset + exact_tile_count_range, : exact_tile_tensor.shape[1]] = exact_tile_tensor
            exact_tile_offset += exact_tile_count_range
        exact_tile_ptr_cpu[range_idx + 1] = int(exact_tile_offset)

        tail_tile_tensor = range_entry.get("tail_tile_tensor")
        tail_mask_tensor = range_entry.get("tail_mask_tensor")
        if isinstance(tail_tile_tensor, torch.Tensor) and isinstance(tail_mask_tensor, torch.Tensor):
            tail_tile_tensor = tail_tile_tensor.to(dtype=torch.int32, device="cpu").contiguous()
            tail_mask_tensor = tail_mask_tensor.to(dtype=torch.int32, device="cpu").contiguous()
            tail_tile_count_range = int(tail_tile_tensor.shape[0])
            if tail_tile_count_range > 0:
                tail_k_row_idx_cpu[tail_tile_offset : tail_tile_offset + tail_tile_count_range] = tail_tile_tensor
                tail_mask_words_cpu[tail_tile_offset : tail_tile_offset + tail_tile_count_range, :, 0] = tail_mask_tensor
                tail_tile_offset += tail_tile_count_range
        else:
            tail_tiles = [
                [int(value) for value in tail_tile]
                for tail_tile in range_entry.get("tail_tiles", ())
            ]
            tail_mask_rows = [
                [int(value) for value in row_masks]
                for row_masks in range_entry.get("tail_mask_rows", ())
            ]
            if tail_tiles:
                tail_tile_tensor = torch.tensor(tail_tiles, dtype=torch.int32)
                tail_mask_tensor = torch.tensor(tail_mask_rows, dtype=torch.int32)
                tail_tile_count_range = int(tail_tile_tensor.shape[0])
                tail_k_row_idx_cpu[tail_tile_offset : tail_tile_offset + tail_tile_count_range, : tail_tile_tensor.shape[1]] = tail_tile_tensor
                tail_mask_words_cpu[tail_tile_offset : tail_tile_offset + tail_tile_count_range, :, 0] = tail_mask_tensor
                tail_tile_offset += tail_tile_count_range
        tail_tile_ptr_cpu[range_idx + 1] = int(tail_tile_offset)
    return (
        q_row_idx_cpu.to(device=device).contiguous(),
        q_length_cpu.to(device=device).contiguous(),
        exact_tile_ptr_cpu.to(device=device).contiguous(),
        exact_k_row_idx_cpu.to(device=device).contiguous(),
        tail_tile_ptr_cpu.to(device=device).contiguous(),
        tail_k_row_idx_cpu.to(device=device).contiguous(),
        tail_mask_words_cpu.to(device=device).contiguous(),
    )


def _materialize_cached_group_tensors(
    *,
    device: torch.device,
    group_q_rows: list[list[int]],
    group_k_rows: list[list[int]],
    group_mask_words: list[torch.Tensor],
    include_mask_bool: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, int, int]:
    group_count = len(group_q_rows)
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    rows_per_group = max(len(group_q_rows[group_idx]) for group_idx in range(group_count))
    max_union_k = max(len(group_k_rows[group_idx]) for group_idx in range(group_count))
    max_mask_words = max(int(group_mask_words[group_idx].shape[1]) for group_idx in range(group_count))

    q_row_idx_cpu = torch.full((group_count, rows_per_group), -1, dtype=torch.int32)
    k_row_idx_cpu = torch.full((group_count, max_union_k), -1, dtype=torch.int32)
    mask_words_cpu = torch.zeros((group_count, rows_per_group, max_mask_words), dtype=torch.int32)
    q_length_cpu = torch.zeros((group_count,), dtype=torch.int32)
    k_length_cpu = torch.zeros((group_count,), dtype=torch.int32)
    mask_bool_cpu = (
        torch.zeros((group_count, rows_per_group, max_union_k), dtype=torch.bool)
        if include_mask_bool
        else None
    )

    for group_idx in range(group_count):
        row_count = len(group_q_rows[group_idx])
        union_k = len(group_k_rows[group_idx])
        word_count = int(group_mask_words[group_idx].shape[1])
        q_row_idx_cpu[group_idx, :row_count] = torch.tensor(group_q_rows[group_idx], dtype=torch.int32)
        k_row_idx_cpu[group_idx, :union_k] = torch.tensor(group_k_rows[group_idx], dtype=torch.int32)
        mask_words_cpu[group_idx, :row_count, :word_count] = group_mask_words[group_idx]
        q_length_cpu[group_idx] = row_count
        k_length_cpu[group_idx] = union_k
        if mask_bool_cpu is not None:
            mask_bool_cpu[group_idx, :row_count, :union_k] = _decode_mask_words_to_bool(
                group_mask_words[group_idx],
                union_k,
            )

    return (
        q_row_idx_cpu.to(device=device).contiguous(),
        k_row_idx_cpu.to(device=device).contiguous(),
        mask_words_cpu.to(device=device).contiguous(),
        None if mask_bool_cpu is None else mask_bool_cpu.to(device=device).contiguous(),
        q_length_cpu.to(device=device).contiguous(),
        k_length_cpu.to(device=device).contiguous(),
        rows_per_group,
        max_union_k,
    )


def _append_group_entries(
    *,
    q_rows: list[int],
    support_lists: list[list[int]],
    row_groups: list[list[int]],
    group_q_rows: list[list[int]],
    group_k_rows: list[list[int]],
    group_mask_words: list[torch.Tensor],
    group_fill: list[float],
    group_families: list[str] | None = None,
    family: str | None = None,
):
    for row_group in row_groups:
        if not row_group:
            continue
        union_rows = _merge_support_segments([support_lists[row_idx] for row_idx in row_group])
        union_k = len(union_rows)
        if union_k <= 0:
            continue
        mask_words, fill = _build_group_mask_words(support_lists, row_group, union_rows)
        group_q_rows.append([q_rows[entry_idx] for entry_idx in row_group])
        group_k_rows.append(union_rows)
        group_mask_words.append(mask_words.contiguous())
        group_fill.append(fill)
        if group_families is not None and family is not None:
            group_families.append(family)


def _build_range_execution_metadata(
    group_q_rows: list[list[int]],
    combine_group_ranges: list[tuple[int, int]],
    *,
    group_families: list[str] | None = None,
    additional_row_counts: dict[int, int] | None = None,
) -> list[dict[str, int | bool | str]]:
    row_counts: dict[int, int] = {} if additional_row_counts is None else dict(additional_row_counts)
    for q_rows in group_q_rows:
        for q_row in q_rows:
            q_row_int = int(q_row)
            if q_row_int < 0:
                continue
            row_counts[q_row_int] = row_counts.get(q_row_int, 0) + 1

    range_execution: list[dict[str, int | bool | str]] = []
    for group_start, group_end in combine_group_ranges:
        range_rows: list[int] = []
        for group_idx in range(int(group_start), int(group_end)):
            range_rows.extend(
                int(q_row) for q_row in group_q_rows[group_idx] if int(q_row) >= 0
            )
        scatter_only = bool(range_rows) and all(row_counts.get(q_row, 0) == 1 for q_row in range_rows)
        range_entry: dict[str, int | bool | str] = {
            "group_start": int(group_start),
            "group_end": int(group_end),
            "scatter_only": scatter_only,
        }
        if group_families is not None and int(group_start) < len(group_families):
            range_entry["family"] = str(group_families[int(group_start)])
        range_execution.append(range_entry)
    return _merge_adjacent_range_execution_metadata(range_execution)


def _merge_adjacent_range_execution_metadata(
    range_execution: list[dict[str, int | bool | str]],
) -> list[dict[str, int | bool | str]]:
    if len(range_execution) <= 1:
        return [dict(entry) for entry in range_execution]

    merged: list[dict[str, int | bool | str]] = [dict(range_execution[0])]
    for entry in range_execution[1:]:
        current = dict(entry)
        previous = merged[-1]
        if (
            int(previous["group_end"]) == int(current["group_start"])
            and bool(previous.get("scatter_only")) == bool(current.get("scatter_only"))
            and previous.get("family") == current.get("family")
        ):
            previous["group_end"] = int(current["group_end"])
            continue
        merged.append(current)
    return merged


def _annotate_range_execution_kernels(
    range_execution: list[dict[str, int | bool | str]],
    *,
    rows_per_group: int,
    max_union_k: int,
    tile_k: int,
    union_kernel: str,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> list[dict[str, int | bool | str]]:
    annotated: list[dict[str, int | bool | str]] = []
    can_tc_union = (
        str(union_kernel) == "tc16x32"
        and int(rows_per_group) == 16
        and int(tile_k) == 32
        and 0 < int(max_union_k) <= 128
        and _can_use_synthetic_2d_masked_fwd(
            q_flat,
            k_flat,
            v_flat,
            packed_q=int(rows_per_group),
            packed_k=int(max_union_k),
        )
    )
    for entry in range_execution:
        next_entry = dict(entry)
        if (
            bool(next_entry.get("scatter_only"))
            and str(next_entry.get("family", "union_2d")) == "union_2d"
            and can_tc_union
        ):
            next_entry["kernel_kind"] = "tc_scatter"
        elif bool(next_entry.get("scatter_only")):
            next_entry["kernel_kind"] = "scatter"
        else:
            next_entry["kernel_kind"] = "packed"
        annotated.append(next_entry)
    return annotated


_RANGE_KERNEL_KIND_CODES = {
    "packed": 0,
    "scatter": 1,
    "tc_scatter": 2,
}


def _materialize_range_execution_tensors(
    range_execution: list[dict[str, int | bool | str]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    starts = [int(entry["group_start"]) for entry in range_execution]
    ends = [int(entry["group_end"]) for entry in range_execution]
    scatter_only = [1 if bool(entry.get("scatter_only")) else 0 for entry in range_execution]
    kernel_kind = [
        int(_RANGE_KERNEL_KIND_CODES.get(str(entry.get("kernel_kind", "packed")), 0))
        for entry in range_execution
    ]
    return {
        "range_group_start": torch.tensor(starts, dtype=torch.int32, device=device).contiguous(),
        "range_group_end": torch.tensor(ends, dtype=torch.int32, device=device).contiguous(),
        "range_scatter_only": torch.tensor(scatter_only, dtype=torch.int32, device=device).contiguous(),
        "range_kernel_kind": torch.tensor(kernel_kind, dtype=torch.int32, device=device).contiguous(),
    }


def _materialize_range_kernel_group_tensors(
    *,
    q_row_idx: torch.Tensor,
    k_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    mask_words: torch.Tensor,
    range_execution: list[dict[str, int | bool | str]],
) -> dict[str, torch.Tensor | int]:
    result: dict[str, torch.Tensor | int] = {}
    for kernel_kind in ("tc_scatter", "scatter", "packed"):
        q_chunks: list[torch.Tensor] = []
        k_chunks: list[torch.Tensor] = []
        q_length_chunks: list[torch.Tensor] = []
        k_length_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        union_group_count = 0
        union_row_count = 0
        for entry in range_execution:
            if str(entry.get("kernel_kind", "packed")) != kernel_kind:
                continue
            group_start = int(entry["group_start"])
            group_end = int(entry["group_end"])
            if group_end <= group_start:
                continue
            q_chunks.append(q_row_idx[group_start:group_end])
            k_chunks.append(k_row_idx[group_start:group_end])
            q_length_chunks.append(q_length[group_start:group_end])
            k_length_chunks.append(k_length[group_start:group_end])
            mask_chunks.append(mask_words[group_start:group_end])
            if str(entry.get("family", "union_2d")) == "union_2d":
                union_group_count += group_end - group_start
                union_row_count += int(q_length[group_start:group_end].sum().item())
        prefix = f"range_{kernel_kind}"
        if q_chunks:
            grouped_q = torch.cat(q_chunks, dim=0).contiguous()
            grouped_k = torch.cat(k_chunks, dim=0).contiguous()
            grouped_q_length = torch.cat(q_length_chunks, dim=0).contiguous()
            grouped_k_length = torch.cat(k_length_chunks, dim=0).contiguous()
            grouped_mask = torch.cat(mask_chunks, dim=0).contiguous()
        else:
            grouped_q = torch.empty((0, q_row_idx.shape[1]), dtype=q_row_idx.dtype, device=q_row_idx.device)
            grouped_k = torch.empty((0, k_row_idx.shape[1]), dtype=k_row_idx.dtype, device=k_row_idx.device)
            grouped_q_length = torch.empty((0,), dtype=q_length.dtype, device=q_length.device)
            grouped_k_length = torch.empty((0,), dtype=k_length.dtype, device=k_length.device)
            grouped_mask = torch.empty(
                (0, mask_words.shape[1], mask_words.shape[2]),
                dtype=mask_words.dtype,
                device=mask_words.device,
            )
        result[f"{prefix}_q_row_idx"] = grouped_q
        result[f"{prefix}_k_row_idx"] = grouped_k
        result[f"{prefix}_q_length"] = grouped_q_length
        result[f"{prefix}_k_length"] = grouped_k_length
        result[f"{prefix}_mask_words"] = grouped_mask
        result[f"{prefix}_q_row_idx_flat"] = grouped_q.view(-1)
        result[f"{prefix}_k_row_idx_flat"] = grouped_k.view(-1)
        result[f"{prefix}_group_count"] = int(grouped_q.shape[0])
        result[f"{prefix}_row_count"] = int(grouped_q_length.sum().item()) if int(grouped_q_length.numel()) > 0 else 0
        result[f"{prefix}_union_group_count"] = int(union_group_count)
        result[f"{prefix}_union_row_count"] = int(union_row_count)
    return result


def _coerce_cached_packing_policy(
    policy: CachedPackingPolicy | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> CachedPackingPolicy:
    resolved = CachedPackingPolicy() if policy is None else policy
    if overrides:
        resolved = replace(resolved, **overrides)
    if not (0.0 < float(resolved.direct_density_threshold) <= 1.0):
        raise ValueError("direct_density_threshold must be in (0, 1]")
    if not (0.0 < float(resolved.window_density_threshold) <= 1.0):
        raise ValueError("window_density_threshold must be in (0, 1]")
    if float(resolved.window_density_threshold) > float(resolved.direct_density_threshold):
        raise ValueError("window_density_threshold must be <= direct_density_threshold")
    if float(resolved.k_gap_threshold) <= 0.0:
        raise ValueError("k_gap_threshold must be positive")
    if int(resolved.max_rows_per_group) <= 0 or int(resolved.max_rows_per_group) > 16:
        raise ValueError("max_rows_per_group must be between 1 and 16")
    if int(resolved.tile_k) not in {16, 32}:
        raise ValueError("tile_k must be one of 16 or 32")
    if int(resolved.max_union_k_direct) <= 0 or int(resolved.max_union_k_direct) > 128:
        raise ValueError("max_union_k_direct must be between 1 and 128")
    if int(resolved.max_union_k_2d) <= 0 or int(resolved.max_union_k_2d) > 1024:
        raise ValueError("max_union_k_2d must be between 1 and 1024")
    if int(resolved.max_union_k_direct) > int(resolved.max_union_k_2d):
        raise ValueError("max_union_k_direct must be <= max_union_k_2d")
    if str(resolved.union_kernel) not in {"tc16x32", "scalar"}:
        raise ValueError("union_kernel must be one of tc16x32 or scalar")
    _resolve_exact_kernel_spec(
        str(resolved.exact_kernel_family),
        min_rows=int(resolved.exact_min_rows),
    )
    if str(resolved.residual_mode) not in {"fused_tail", "masked_union"}:
        raise ValueError("residual_mode must be one of fused_tail or masked_union")
    return resolved


def _summarize_bucket_support_geometry(
    q_rows: list[int],
    support_lists: list[list[int]],
) -> dict[str, float | int]:
    live_pairs = sum(len(support_list) for support_list in support_lists)
    num_live_rows = len(q_rows)
    q_extent = max(q_rows) - min(q_rows) + 1 if q_rows else 0
    max_row_support = max((len(support_list) for support_list in support_lists), default=0)
    live_k_min = min((int(support_list[0]) for support_list in support_lists if support_list), default=0)
    live_k_max = max((int(support_list[-1]) for support_list in support_lists if support_list), default=-1)
    approx_live_pairs_threshold = int(
        os.environ.get(
            "FLASH_ATTN_HSA_CACHED_PACKING_APPROX_STATS_LIVE_PAIRS_THRESHOLD",
            "4096",
        )
    )
    if live_pairs >= approx_live_pairs_threshold and max_row_support > 0 and live_k_max >= live_k_min:
        num_live_k = max_row_support
        k_extent = live_k_max - live_k_min + 1
        active_density = live_pairs / max(1, num_live_rows * max(1, num_live_k)) if num_live_rows > 0 else 0.0
        return {
            "num_live_rows": num_live_rows,
            "num_live_k": num_live_k,
            "live_pairs": live_pairs,
            "k_extent": k_extent,
            "q_extent": q_extent,
            "active_density": active_density,
            "k_gap_ratio": (k_extent / max(1, num_live_k)) if num_live_k > 0 else 0.0,
            "q_gap_ratio": (q_extent / max(1, num_live_rows)) if num_live_rows > 0 else 0.0,
            "max_row_support": max_row_support,
        }
    intervals: list[tuple[int, int]] = []
    for support_list in support_lists:
        if not support_list:
            continue
        start = prev = int(support_list[0])
        for support_row in support_list[1:]:
            support_row = int(support_row)
            if support_row == prev + 1:
                prev = support_row
                continue
            intervals.append((start, prev))
            start = prev = support_row
        intervals.append((start, prev))
    if intervals:
        intervals.sort(key=lambda item: (item[0], item[1]))
        merged_start, merged_end = intervals[0]
        num_live_k = 0
        for start, end in intervals[1:]:
            if start <= merged_end + 1:
                if end > merged_end:
                    merged_end = end
                continue
            num_live_k += merged_end - merged_start + 1
            merged_start, merged_end = start, end
        num_live_k += merged_end - merged_start + 1
        k_extent = intervals[-1][1] - intervals[0][0] + 1
    else:
        num_live_k = 0
        k_extent = 0
    active_density = live_pairs / max(1, num_live_rows * max(1, num_live_k)) if num_live_rows > 0 else 0.0
    return {
        "num_live_rows": num_live_rows,
        "num_live_k": num_live_k,
        "live_pairs": live_pairs,
        "k_extent": k_extent,
        "q_extent": q_extent,
        "active_density": active_density,
        "k_gap_ratio": (k_extent / max(1, num_live_k)) if num_live_k > 0 else 0.0,
        "q_gap_ratio": (q_extent / max(1, num_live_rows)) if num_live_rows > 0 else 0.0,
        "max_row_support": max_row_support,
    }


def _merge_support_segments(segments: list[list[int]]) -> list[int]:
    if not segments:
        return []
    if len(segments) == 1:
        return [int(value) for value in segments[0]]
    flat_values = [int(value) for segment in segments for value in segment]
    if not flat_values:
        return []
    merged_values = torch.unique(torch.tensor(flat_values, dtype=torch.int32), sorted=True)
    return [int(value) for value in merged_values.tolist()]


def _choose_cached_packing_family(
    bucket_stats: dict[str, float | int],
    policy: CachedPackingPolicy,
) -> str:
    num_live_k = int(bucket_stats["num_live_k"])
    active_density = float(bucket_stats["active_density"])
    k_gap_ratio = float(bucket_stats["k_gap_ratio"])
    if (
        num_live_k > 0
        and num_live_k <= int(policy.max_union_k_direct)
        and active_density >= float(policy.direct_density_threshold)
        and k_gap_ratio <= float(policy.k_gap_threshold)
    ):
        return "direct_passthrough"
    if active_density >= float(policy.window_density_threshold) and k_gap_ratio > float(policy.k_gap_threshold):
        return "k_window"
    return "union_2d"


def _build_bucket_mask_from_support_lists(
    support_lists: list[list[int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    union_rows = sorted({support_row for support_list in support_lists for support_row in support_list})
    if not union_rows:
        return torch.zeros((len(support_lists), 0), dtype=torch.bool, device=device)
    union_pos = {support_row: pos for pos, support_row in enumerate(union_rows)}
    mask = torch.zeros((len(support_lists), len(union_rows)), dtype=torch.bool, device=device)
    for row_idx, support_list in enumerate(support_lists):
        for support_row in support_list:
            mask[row_idx, union_pos[int(support_row)]] = True
    return mask


def _append_group_range(
    *,
    group_start: int,
    family: str,
    group_q_rows: list[torch.Tensor],
    combine_group_ranges: list[tuple[int, int]],
    group_families: list[str],
):
    if len(group_q_rows) <= group_start:
        return
    group_end = len(group_q_rows)
    combine_group_ranges.append((group_start, group_end))
    group_families.extend([family] * (group_end - group_start))


def _append_direct_passthrough_groups(
    *,
    q_rows: list[int],
    support_lists: list[list[int]],
    policy: CachedPackingPolicy,
    group_q_rows: list[list[int]],
    group_k_rows: list[list[int]],
    group_mask_words: list[torch.Tensor],
    group_fill: list[float],
    combine_group_ranges: list[tuple[int, int]],
    group_families: list[str],
) -> int:
    row_groups = _group_support_lists_by_span(
        support_lists,
        max_rows_per_group=min(int(policy.max_rows_per_group), len(q_rows)),
        max_union_k=max(1, int(policy.max_union_k_direct)),
    )
    group_start = len(group_q_rows)
    _append_group_entries(
        q_rows=q_rows,
        support_lists=support_lists,
        row_groups=row_groups,
        group_q_rows=group_q_rows,
        group_k_rows=group_k_rows,
        group_mask_words=group_mask_words,
        group_fill=group_fill,
    )
    _append_group_range(
        group_start=group_start,
        family="direct_passthrough",
        group_q_rows=group_q_rows,
        combine_group_ranges=combine_group_ranges,
        group_families=group_families,
    )
    return len(group_q_rows) - group_start


def _append_k_window_groups(
    *,
    q_rows: list[int],
    support_lists: list[list[int]],
    policy: CachedPackingPolicy,
    group_q_rows: list[list[int]],
    group_k_rows: list[list[int]],
    group_mask_words: list[torch.Tensor],
    group_fill: list[float],
    combine_group_ranges: list[tuple[int, int]],
    group_families: list[str],
) -> int:
    max_union_k = max(1, int(policy.max_union_k_2d))
    expanded_q_rows: list[int] = []
    expanded_support_lists: list[list[int]] = []
    for q_row, support_list in zip(q_rows, support_lists, strict=True):
        if len(support_list) <= max_union_k:
            expanded_q_rows.append(q_row)
            expanded_support_lists.append(support_list)
            continue
        for segment_start in range(0, len(support_list), max_union_k):
            segment = support_list[segment_start : segment_start + max_union_k]
            if not segment:
                continue
            expanded_q_rows.append(q_row)
            expanded_support_lists.append(segment)
    row_groups = _group_support_lists_by_span(
        expanded_support_lists,
        max_rows_per_group=min(int(policy.max_rows_per_group), len(expanded_q_rows)),
        max_union_k=max_union_k,
    )
    group_start = len(group_q_rows)
    _append_group_entries(
        q_rows=expanded_q_rows,
        support_lists=expanded_support_lists,
        row_groups=row_groups,
        group_q_rows=group_q_rows,
        group_k_rows=group_k_rows,
        group_mask_words=group_mask_words,
        group_fill=group_fill,
    )
    _append_group_range(
        group_start=group_start,
        family="k_window",
        group_q_rows=group_q_rows,
        combine_group_ranges=combine_group_ranges,
        group_families=group_families,
    )
    return len(group_q_rows) - group_start


def _finalize_generalized_cached_forward_payload(
    *,
    device: torch.device,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    group_q_rows: list[list[int]],
    group_k_rows: list[list[int]],
    group_mask_words: list[torch.Tensor],
    group_fill: list[float],
    combine_group_ranges: list[tuple[int, int]],
    group_families: list[str],
    tile_k: int,
    geometry_base: dict[str, Any],
    reason: str,
    exact_ranges: list[dict[str, Any]] | None = None,
    exact_live_pairs: int = 0,
    fused_ranges: list[dict[str, Any]] | None = None,
    fused_exact_live_pairs: int = 0,
    fused_tail_live_pairs: int = 0,
    union_kernel: str = "scalar",
    exact_kernel_family: str = "tc16x16",
    exact_rows_per_range: int = 16,
    exact_keys_per_tile: int = 16,
    exact_min_rows: int = 8,
    residual_mode: str = "masked_union",
    include_mask_bool: bool = False,
) -> dict[str, Any]:
    group_count = len(group_q_rows)
    exact_ranges = [] if exact_ranges is None else exact_ranges
    fused_ranges = [] if fused_ranges is None else fused_ranges
    exact_range_count = len(exact_ranges)
    fused_range_count = len(fused_ranges)
    if group_count <= 0 and exact_range_count <= 0 and fused_range_count <= 0:
        return {"status": "not_applicable", "reason": "cached generalized packing did not yield any forward groups"}

    if group_count > 0:
        exact_row_counts: dict[int, int] = {}
        for range_entry in exact_ranges:
            for q_row in range_entry["q_rows"]:
                q_row_int = int(q_row)
                if q_row_int < 0:
                    continue
                exact_row_counts[q_row_int] = exact_row_counts.get(q_row_int, 0) + 1
        q_row_idx, k_row_idx, mask_words, mask_bool, q_length, k_length, rows_per_group, max_union_k = (
            _materialize_cached_group_tensors(
                device=device,
                group_q_rows=group_q_rows,
                group_k_rows=group_k_rows,
                group_mask_words=group_mask_words,
                include_mask_bool=include_mask_bool,
            )
        )
        range_execution = _build_range_execution_metadata(
            group_q_rows,
            combine_group_ranges,
            group_families=group_families,
            additional_row_counts=exact_row_counts,
        )
        range_execution = _annotate_range_execution_kernels(
            range_execution,
            rows_per_group=int(rows_per_group),
            max_union_k=int(max_union_k),
            tile_k=int(tile_k),
            union_kernel=str(union_kernel),
            q_flat=q_flat,
            k_flat=k_flat,
            v_flat=v_flat,
        )
    else:
        q_row_idx = torch.empty((0, int(geometry_base.get("cached_pack_policy", {}).get("max_rows_per_group", 16))), dtype=torch.int32, device=device)
        k_row_idx = torch.empty((0, 0), dtype=torch.int32, device=device)
        mask_words = torch.empty((0, 0, 0), dtype=torch.int32, device=device)
        mask_bool = None
        q_length = torch.empty((0,), dtype=torch.int32, device=device)
        k_length = torch.empty((0,), dtype=torch.int32, device=device)
        rows_per_group = int(geometry_base.get("cached_pack_policy", {}).get("max_rows_per_group", 16))
        max_union_k = 0
        range_execution = []
    scatter_only_ranges = sum(1 for entry in range_execution if bool(entry["scatter_only"]))
    scatter_only_rows = 0
    family_group_counts = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_q_rows = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_group_area = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_group_fill_sums = {family: 0.0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_union_k_sums = {family: 0.0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_scatter_only_rows = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    grouped_hardware_area = 0
    for group_idx, family in enumerate(group_families):
        q_count = len(group_q_rows[group_idx])
        group_area = q_count * len(group_k_rows[group_idx])
        family_group_counts[family] = family_group_counts.get(family, 0) + 1
        family_q_rows[family] = family_q_rows.get(family, 0) + q_count
        family_group_area[family] = family_group_area.get(family, 0) + group_area
        family_group_fill_sums[family] = family_group_fill_sums.get(family, 0.0) + float(group_fill[group_idx])
        family_union_k_sums[family] = family_union_k_sums.get(family, 0.0) + float(len(group_k_rows[group_idx]))
        grouped_hardware_area += group_area
        is_scatter_only = any(
            int(entry["group_start"]) <= group_idx < int(entry["group_end"]) and bool(entry["scatter_only"])
            for entry in range_execution
        )
        if is_scatter_only:
            scatter_only_rows += q_count
            family_scatter_only_rows[family] = family_scatter_only_rows.get(family, 0) + q_count

    if exact_range_count > 0:
        exact_output_row_count = sum(
            1
            for range_entry in exact_ranges
            for q_row in range_entry.get("q_rows", [])
            if int(q_row) >= 0
        )
        exact_q_row_idx, exact_q_length, exact_tile_ptr, exact_k_row_idx = _materialize_exact_dense_range_tensors(
            device=device,
            exact_ranges=exact_ranges,
            rows_per_range=int(exact_rows_per_range),
            keys_per_tile=int(exact_keys_per_tile),
        )
        exact_tile_count = int(exact_k_row_idx.shape[0])
        exact_hardware_slots = exact_tile_count * int(exact_rows_per_range) * int(exact_keys_per_tile)
        exact_coverage_frac = float(exact_live_pairs) / max(1, int(geometry_base.get("cached_generalized_live_pairs", 0)))
        residual_live_pairs = max(0, int(geometry_base.get("cached_generalized_live_pairs", 0)) - int(exact_live_pairs))
    else:
        exact_q_row_idx = torch.empty((0, int(exact_rows_per_range)), dtype=torch.int32, device=device)
        exact_q_length = torch.empty((0,), dtype=torch.int32, device=device)
        exact_tile_ptr = torch.zeros((1,), dtype=torch.int32, device=device)
        exact_k_row_idx = torch.empty((0, int(exact_keys_per_tile)), dtype=torch.int32, device=device)
        exact_tile_count = 0
        exact_hardware_slots = 0
        exact_coverage_frac = 0.0
        exact_output_row_count = 0
        residual_live_pairs = int(geometry_base.get("cached_generalized_live_pairs", 0))
    if fused_range_count > 0:
        fused_output_row_count = sum(
            1
            for range_entry in fused_ranges
            for q_row in range_entry.get("q_rows", [])
            if int(q_row) >= 0
        )
        (
            fused_q_row_idx,
            fused_q_length,
            fused_exact_tile_ptr,
            fused_exact_k_row_idx,
            fused_tail_tile_ptr,
            fused_tail_k_row_idx,
            fused_tail_mask_words,
        ) = _materialize_fused_exact_tail_range_tensors(
            device=device,
            fused_ranges=fused_ranges,
            rows_per_range=int(exact_rows_per_range),
            keys_per_tile=int(exact_keys_per_tile),
        )
        tail_only_range_count = sum(1 for range_entry in fused_ranges if not range_entry.get("exact_tiles"))
        fused_total_live_pairs = int(fused_exact_live_pairs) + int(fused_tail_live_pairs)
        residual_live_pairs = max(
            0,
            int(geometry_base.get("cached_generalized_live_pairs", 0)) - fused_total_live_pairs,
        )
        fused_tail_tile_count = int(fused_tail_k_row_idx.shape[0])
        fused_tail_slots = fused_tail_tile_count * int(exact_rows_per_range) * int(exact_keys_per_tile)
    else:
        fused_q_row_idx = torch.empty((0, int(exact_rows_per_range)), dtype=torch.int32, device=device)
        fused_q_length = torch.empty((0,), dtype=torch.int32, device=device)
        fused_exact_tile_ptr = torch.zeros((1,), dtype=torch.int32, device=device)
        fused_exact_k_row_idx = torch.empty((0, int(exact_keys_per_tile)), dtype=torch.int32, device=device)
        fused_tail_tile_ptr = torch.zeros((1,), dtype=torch.int32, device=device)
        fused_tail_k_row_idx = torch.empty((0, int(exact_keys_per_tile)), dtype=torch.int32, device=device)
        fused_tail_mask_words = torch.empty((0, int(exact_rows_per_range), 1), dtype=torch.int32, device=device)
        tail_only_range_count = 0
        fused_total_live_pairs = int(exact_live_pairs) + int(residual_live_pairs)
        fused_output_row_count = 0
        fused_tail_tile_count = 0
        fused_tail_slots = 0
    legacy_union_group_count = sum(1 for family in group_families if family == "union_2d")
    geometry = dict(geometry_base)
    geometry.update(
        {
            "cached_generalized_groups": group_count,
            "cached_generalized_rows_per_group": rows_per_group,
            "cached_generalized_avg_union_k": float(k_length.float().mean().item()) if group_count > 0 else 0.0,
            "cached_generalized_max_union_k": max_union_k,
            "cached_generalized_grouped_hardware_area": int(grouped_hardware_area),
            "cached_generalized_total_hardware_area": int(geometry_base.get("cached_generalized_total_hardware_area", 0)),
            "cached_generalized_total_area_reduction": (
                0.0
                if int(geometry_base.get("cached_generalized_baseline_packed_area", 0)) <= 0
                else 1.0
                - (
                    float(geometry_base.get("cached_generalized_total_hardware_area", 0))
                    / float(geometry_base["cached_generalized_baseline_packed_area"])
                )
            ),
            "cached_generalized_avg_group_fill": float(sum(group_fill) / len(group_fill)) if group_fill else 0.0,
            "cached_generalized_scatter_only_ranges": scatter_only_ranges,
            "cached_generalized_scatter_only_rows": scatter_only_rows,
            "family_group_counts": family_group_counts,
            "family_q_rows": family_q_rows,
            "family_group_area": family_group_area,
            "family_avg_union_k": {
                family: (
                    family_union_k_sums[family] / family_group_counts[family]
                    if family_group_counts[family] > 0
                    else 0.0
                )
                for family in family_group_counts
            },
            "family_avg_q_rows": {
                family: (
                    family_q_rows[family] / family_group_counts[family]
                    if family_group_counts[family] > 0
                    else 0.0
                )
                for family in family_group_counts
            },
            "family_avg_group_area": {
                family: (
                    family_group_area[family] / family_group_counts[family]
                    if family_group_counts[family] > 0
                    else 0.0
                )
                for family in family_group_counts
            },
            "family_avg_group_fill": {
                family: (
                    family_group_fill_sums[family] / family_group_counts[family]
                    if family_group_counts[family] > 0
                    else 0.0
                )
                for family in family_group_counts
            },
            "family_scatter_only_rows": family_scatter_only_rows,
            "union_kernel": str(union_kernel),
            "union_tc_group_count": 0,
            "union_tc_row_count": 0,
            "union_scalar_fallback_group_count": 0,
            "union_scalar_fallback_row_count": 0,
            "exact_dense_range_count": exact_range_count,
            "exact_dense_tile_count": exact_tile_count,
            "exact_dense_output_row_count": int(exact_output_row_count),
            "exact_dense_live_pairs": int(exact_live_pairs),
            "exact_dense_slots": int(exact_hardware_slots),
            "exact_dense_hardware_fill": float(exact_live_pairs) / max(1, exact_hardware_slots),
            "exact_dense_coverage_frac": exact_coverage_frac,
            "residual_live_pairs": int(residual_live_pairs),
            "residual_coverage_frac": float(residual_live_pairs) / max(1, int(geometry_base.get("cached_generalized_live_pairs", 0))),
            "exact_kernel_family": str(exact_kernel_family),
            "residual_mode": str(residual_mode),
            "fused_range_count": int(fused_range_count),
            "fused_output_row_count": int(fused_output_row_count),
            "fused_exact_live_pairs": int(fused_exact_live_pairs),
            "fused_tail_live_pairs": int(fused_tail_live_pairs),
            "fused_exact_coverage_frac": float(fused_exact_live_pairs) / max(1, int(geometry_base.get("cached_generalized_live_pairs", 0))),
            "fused_tail_coverage_frac": float(fused_tail_live_pairs) / max(1, int(geometry_base.get("cached_generalized_live_pairs", 0))),
            "fused_total_coverage_frac": float(fused_total_live_pairs) / max(1, int(geometry_base.get("cached_generalized_live_pairs", 0))),
            "fused_tail_tile_count": int(fused_tail_tile_count),
            "fused_tail_slots": int(fused_tail_slots),
            "fused_tail_hardware_fill": float(fused_tail_live_pairs) / max(1, fused_tail_slots),
            "tail_only_range_count": int(tail_only_range_count),
            "legacy_residual_fallback_range_count": int(legacy_union_group_count),
            "cached_payload_includes_mask_bool": bool(include_mask_bool),
        }
    )
    payload = {
        "status": "ready",
        "reason": reason,
        "packed_q": rows_per_group,
        "support_rows": max_union_k,
        "tile_k": int(tile_k),
        "q_row_idx": q_row_idx.contiguous(),
        "k_row_idx": k_row_idx.contiguous(),
        "q_row_idx_flat": q_row_idx.contiguous().view(-1),
        "k_row_idx_flat": k_row_idx.contiguous().view(-1),
        "mask_words": mask_words.contiguous(),
        "q_length": q_length.contiguous(),
        "k_length": k_length.contiguous(),
        "total_rows": int(q_flat.shape[0]),
        "all_row_idx": torch.arange(int(q_flat.shape[0]), dtype=torch.int32, device=device).contiguous(),
        "combine_group_ranges": combine_group_ranges,
        "range_execution": range_execution,
        **_materialize_range_execution_tensors(range_execution, device=device),
        **_materialize_range_kernel_group_tensors(
            q_row_idx=q_row_idx,
            k_row_idx=k_row_idx,
            q_length=q_length,
            k_length=k_length,
            mask_words=mask_words,
            range_execution=range_execution,
        ),
        "union_kernel": str(union_kernel),
        "geometry": geometry,
        "_workspace": {},
        "exact_dense_q_row_idx": exact_q_row_idx.contiguous(),
        "exact_dense_q_length": exact_q_length.contiguous(),
        "exact_dense_tile_ptr": exact_tile_ptr.contiguous(),
        "exact_dense_k_row_idx": exact_k_row_idx.contiguous(),
        "exact_dense_output_row_count": int(exact_output_row_count),
        "exact_dense_rows_per_range": int(exact_rows_per_range),
        "exact_dense_keys_per_tile": int(exact_keys_per_tile),
        "exact_dense_min_rows": int(exact_min_rows),
        "exact_kernel_family": str(exact_kernel_family),
        "residual_mode": str(residual_mode),
        "fused_q_row_idx": fused_q_row_idx.contiguous(),
        "fused_q_length": fused_q_length.contiguous(),
        "fused_output_row_count": int(fused_output_row_count),
        "fused_exact_tile_ptr": fused_exact_tile_ptr.contiguous(),
        "fused_exact_k_row_idx": fused_exact_k_row_idx.contiguous(),
        "fused_tail_tile_ptr": fused_tail_tile_ptr.contiguous(),
        "fused_tail_k_row_idx": fused_tail_k_row_idx.contiguous(),
        "fused_tail_mask_words": fused_tail_mask_words.contiguous(),
    }
    if mask_bool is not None:
        payload["mask_bool"] = mask_bool.contiguous()
    if str(payload.get("residual_mode", "")) == "fused_tail":
        backward_payload = build_cached_generalized_backward_payload(payload)
        if isinstance(backward_payload, dict):
            payload["cached_generalized_backward_payload"] = backward_payload
    return payload


def build_cached_generalized_packed_forward_payload(
    runtime,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    policy: CachedPackingPolicy | None = None,
    policy_overrides: dict[str, Any] | None = None,
    include_mask_bool: bool = False,
) -> dict[str, Any]:
    q_flat = _flatten_row_tensor(q)
    k_flat = _flatten_row_tensor(k)
    v_flat = _flatten_row_tensor(v)
    resolved_policy = _coerce_cached_packing_policy(policy, overrides=policy_overrides)
    exact_spec = _resolve_exact_kernel_spec(
        str(resolved_policy.exact_kernel_family),
        min_rows=int(resolved_policy.exact_min_rows),
    )
    metadata = None if runtime is None else getattr(runtime, "forward_synthetic_grid", None)
    if metadata is None or getattr(metadata, "forward_execution_plan", None) is None:
        return {"status": "not_applicable", "reason": "cached schedule is missing synthetic-grid forward metadata"}
    direct_plan = metadata.forward_execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return {"status": "not_applicable", "reason": "cached schedule does not expose a direct execution plan"}
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is None:
        return {"status": "not_applicable", "reason": "cached direct execution plan is missing a row-compact plan"}

    device = q_flat.device
    group_q_rows: list[list[int]] = []
    group_k_rows: list[list[int]] = []
    group_mask_words: list[torch.Tensor] = []
    group_fill: list[float] = []
    group_families: list[str] = []
    combine_group_ranges: list[tuple[int, int]] = []
    bucket_qgroup_bucket_idx = direct_plan.get("bucket_qgroup_bucket_idx")
    family_counts = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_live_pairs = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_bucket_stats = {
        family: {"active_density": 0.0, "k_gap_ratio": 0.0, "q_gap_ratio": 0.0, "count": 0}
        for family in ("direct_passthrough", "k_window", "union_2d")
    }
    direct_bucket_entries: list[dict[str, Any]] = []
    kwindow_bucket_entries: list[dict[str, Any]] = []
    union_bucket_entries: list[dict[str, Any]] = []
    baseline_live_row_count = 0
    baseline_packed_k_sum = 0
    zero_support_rows_count = 0
    total_live_pairs = 0
    total_group_area = 0
    exact_ranges: list[dict[str, Any]] = []
    exact_live_pairs = 0
    fused_ranges: list[dict[str, Any]] = []
    fused_exact_live_pairs = 0
    fused_tail_live_pairs = 0
    use_fused_tail = (
        str(resolved_policy.residual_mode) == "fused_tail"
        and str(exact_spec["family"]) == "tc8x8"
        and q_flat.shape[-1] == 64
        and q_flat.dtype in {torch.bfloat16, torch.float16}
    )
    disable_direct_for_fused_tail = use_fused_tail and os.environ.get(
        "FLASH_ATTN_HSA_FUSED_TAIL_DISABLE_DIRECT",
        "0",
    ).strip().lower() not in {"0", "false", "off", "no"}
    precomputed_q_rows = direct_plan.get("bucket_live_q_rows_list")
    precomputed_support_lists = direct_plan.get("bucket_live_support_lists")

    for bucket_idx in range(len(direct_plan["bucket_size"])):
        packed_k = int(direct_plan["bucket_packed_k"][bucket_idx])
        q_rows: list[int]
        support_lists: list[list[int]]
        if (
            isinstance(precomputed_q_rows, list)
            and isinstance(precomputed_support_lists, list)
            and bucket_idx < len(precomputed_q_rows)
            and bucket_idx < len(precomputed_support_lists)
        ):
            q_rows = precomputed_q_rows[bucket_idx]
            support_lists = precomputed_support_lists[bucket_idx]
        else:
            live_q_rows, live_support_rows, live_support_valid, zero_support_rows, _ = _extract_bucket_live_row_supports(
                direct_plan,
                row_plan,
                bucket_idx,
            )
            zero_support_rows_count += int(zero_support_rows)
            live_row_count = int(live_q_rows.numel())
            if live_row_count <= 0:
                continue
            live_q_rows_cpu = [int(value) for value in live_q_rows.detach().cpu().tolist()]
            live_support_rows_cpu = live_support_rows.detach().cpu().tolist()
            live_support_valid_cpu = live_support_valid.detach().cpu().tolist()
            support_lists = [
                [int(support_row) for support_row, valid in zip(row_supports, row_valid, strict=True) if valid and int(support_row) >= 0]
                for row_supports, row_valid in zip(live_support_rows_cpu, live_support_valid_cpu, strict=True)
            ]
            q_rows = [int(q_row) for q_row in live_q_rows_cpu]
        if q_rows:
            filtered_pairs = [(q_row, support_rows) for q_row, support_rows in zip(q_rows, support_lists, strict=True) if support_rows]
            zero_support_rows_count += len(q_rows) - len(filtered_pairs)
            q_rows = [q_row for q_row, _support_rows in filtered_pairs]
            support_lists = [support_rows for _q_row, support_rows in filtered_pairs]
        live_row_count = len(q_rows)
        if not q_rows:
            continue
        baseline_live_row_count += live_row_count
        baseline_packed_k_sum += int(packed_k) * live_row_count
        bucket_stats = _summarize_bucket_support_geometry(q_rows, support_lists)
        family = _choose_cached_packing_family(bucket_stats, resolved_policy)
        if disable_direct_for_fused_tail and family == "direct_passthrough":
            family = "union_2d"
        family_counts[family] = family_counts.get(family, 0) + 1
        family_live_pairs[family] = family_live_pairs.get(family, 0) + int(bucket_stats["live_pairs"])
        total_live_pairs += int(bucket_stats["live_pairs"])
        family_bucket_stats[family]["active_density"] += float(bucket_stats["active_density"])
        family_bucket_stats[family]["k_gap_ratio"] += float(bucket_stats["k_gap_ratio"])
        family_bucket_stats[family]["q_gap_ratio"] += float(bucket_stats["q_gap_ratio"])
        family_bucket_stats[family]["count"] += 1
        qgroup_bucket_idx = bucket_idx if bucket_qgroup_bucket_idx is None else int(bucket_qgroup_bucket_idx[bucket_idx])
        if family == "direct_passthrough":
            direct_bucket_entries.append(
                {
                    "bucket_idx": bucket_idx,
                    "qgroup_bucket_idx": qgroup_bucket_idx,
                    "q_rows": q_rows,
                    "support_lists": support_lists,
                }
            )
            continue
        if family == "k_window":
            kwindow_bucket_entries.append(
                {
                    "bucket_idx": bucket_idx,
                    "qgroup_bucket_idx": qgroup_bucket_idx,
                    "q_rows": q_rows,
                    "support_lists": support_lists,
                }
            )
            continue
        union_bucket_entries.append(
            {
                "bucket_idx": bucket_idx,
                "qgroup_bucket_idx": qgroup_bucket_idx,
                "q_rows": q_rows,
                "support_lists": support_lists,
            }
        )

    merged_direct_q_rows: list[int] = []
    merged_direct_support_lists: list[list[int]] = []
    for bucket_entry in direct_bucket_entries:
        for q_row, support_list in zip(bucket_entry["q_rows"], bucket_entry["support_lists"], strict=True):
            if not support_list:
                continue
            merged_direct_q_rows.append(int(q_row))
            merged_direct_support_lists.append(list(support_list))
    if merged_direct_q_rows:
        _append_direct_passthrough_groups(
            q_rows=merged_direct_q_rows,
            support_lists=merged_direct_support_lists,
            policy=resolved_policy,
            group_q_rows=group_q_rows,
            group_k_rows=group_k_rows,
            group_mask_words=group_mask_words,
            group_fill=group_fill,
            combine_group_ranges=combine_group_ranges,
            group_families=group_families,
        )

    merged_kwindow_q_rows: list[int] = []
    merged_kwindow_support_lists: list[list[int]] = []
    for bucket_entry in kwindow_bucket_entries:
        for q_row, support_list in zip(bucket_entry["q_rows"], bucket_entry["support_lists"], strict=True):
            if not support_list:
                continue
            merged_kwindow_q_rows.append(int(q_row))
            merged_kwindow_support_lists.append(list(support_list))
    if merged_kwindow_q_rows:
        _append_k_window_groups(
            q_rows=merged_kwindow_q_rows,
            support_lists=merged_kwindow_support_lists,
            policy=resolved_policy,
            group_q_rows=group_q_rows,
            group_k_rows=group_k_rows,
            group_mask_words=group_mask_words,
            group_fill=group_fill,
            combine_group_ranges=combine_group_ranges,
            group_families=group_families,
        )

    union_entries_by_qgroup: dict[int, list[tuple[int, list[int]]]] = {}
    for bucket_entry in union_bucket_entries:
        qgroup_entries = union_entries_by_qgroup.setdefault(bucket_entry["qgroup_bucket_idx"], [])
        qgroup_entries.extend(zip(bucket_entry["q_rows"], bucket_entry["support_lists"], strict=True))

    for qgroup_bucket_idx in sorted(union_entries_by_qgroup):
        row_support_segments: dict[int, list[list[int]]] = {}
        for q_row, support_list in union_entries_by_qgroup[qgroup_bucket_idx]:
            if not support_list:
                continue
            row_support_segments.setdefault(int(q_row), []).append(support_list)
        merged_q_rows = sorted(row_support_segments)
        merged_support_lists = [
            _merge_support_segments(row_support_segments[q_row]) for q_row in merged_q_rows
        ]
        if use_fused_tail:
            if not merged_q_rows:
                continue
            row_groups = _group_support_lists_by_span(
                merged_support_lists,
                max_rows_per_group=min(int(resolved_policy.max_rows_per_group), len(merged_q_rows)),
                max_union_k=max(
                    int(resolved_policy.max_union_k_2d),
                    max((len(support_list) for support_list in merged_support_lists), default=0),
                ),
            )
            for row_group in row_groups:
                if not row_group:
                    continue
                grouped_q_rows = [merged_q_rows[row_idx] for row_idx in row_group]
                grouped_support_lists = [merged_support_lists[row_idx] for row_idx in row_group]
                group_fused_ranges, group_exact_live_pairs, group_tail_live_pairs = _extract_fused_exact_tail_ranges(
                    grouped_q_rows,
                    grouped_support_lists,
                    rows_per_range=int(exact_spec["rows_per_range"]),
                    keys_per_tile=int(exact_spec["keys_per_tile"]),
                    min_rows=int(exact_spec["min_rows"]),
                )
                fused_ranges.extend(group_fused_ranges)
                fused_exact_live_pairs += int(group_exact_live_pairs)
                fused_tail_live_pairs += int(group_tail_live_pairs)
            continue

        fallback_rows: set[int] = set()
        filtered_q_rows: list[int] = []
        filtered_support_lists: list[list[int]] = []
        for q_row, support_list in zip(merged_q_rows, merged_support_lists, strict=True):
            if len(support_list) > int(resolved_policy.max_union_k_2d):
                fallback_rows.add(q_row)
                continue
            filtered_q_rows.append(q_row)
            filtered_support_lists.append(support_list)
        residual_q_rows: list[int] = []
        residual_support_lists: list[list[int]] = []
        if filtered_q_rows:
            row_groups = _group_support_lists_by_span(
                filtered_support_lists,
                max_rows_per_group=min(int(resolved_policy.max_rows_per_group), len(filtered_q_rows)),
                max_union_k=max(
                    int(resolved_policy.max_union_k_2d),
                    max((len(support_list) for support_list in filtered_support_lists), default=0),
                ),
            )
            for row_group in row_groups:
                if not row_group:
                    continue
                grouped_q_rows = [filtered_q_rows[row_idx] for row_idx in row_group]
                grouped_support_lists = [filtered_support_lists[row_idx] for row_idx in row_group]
                group_exact_ranges, group_residual_support_lists, group_exact_live_pairs = _extract_exact_dense_tile_ranges(
                    grouped_q_rows,
                    grouped_support_lists,
                    rows_per_range=int(exact_spec["rows_per_range"]),
                    keys_per_tile=int(exact_spec["keys_per_tile"]),
                    min_rows=int(exact_spec["min_rows"]),
                )
                exact_ranges.extend(group_exact_ranges)
                exact_live_pairs += int(group_exact_live_pairs)
                for q_row, support_list in zip(grouped_q_rows, group_residual_support_lists, strict=True):
                    if not support_list:
                        continue
                    residual_q_rows.append(q_row)
                    residual_support_lists.append(support_list)
        if residual_q_rows:
            row_groups = _group_support_lists_by_span(
                residual_support_lists,
                max_rows_per_group=min(int(resolved_policy.max_rows_per_group), len(residual_q_rows)),
                max_union_k=max(
                    int(resolved_policy.max_union_k_2d),
                    max((len(support_list) for support_list in residual_support_lists), default=0),
                ),
            )
            group_start = len(group_q_rows)
            _append_group_entries(
                q_rows=residual_q_rows,
                support_lists=residual_support_lists,
                row_groups=row_groups,
                group_q_rows=group_q_rows,
                group_k_rows=group_k_rows,
                group_mask_words=group_mask_words,
                group_fill=group_fill,
            )
            _append_group_range(
                group_start=group_start,
                family="union_2d",
                group_q_rows=group_q_rows,
                combine_group_ranges=combine_group_ranges,
                group_families=group_families,
            )

    if use_fused_tail:
        exact_ranges = [
            {
                "q_rows": [int(q_row) for q_row in range_entry["q_rows"]],
                "tiles": [[int(value) for value in tile] for tile in range_entry.get("exact_tiles", ())],
            }
            for range_entry in fused_ranges
            if range_entry.get("exact_tiles")
        ]
        exact_live_pairs = int(fused_exact_live_pairs)

    for group_idx in range(len(group_q_rows)):
        total_group_area += len(group_q_rows[group_idx]) * len(group_k_rows[group_idx])
    if fused_ranges:
        total_group_area += (
            sum(len(range_entry.get("exact_tiles", ())) for range_entry in fused_ranges)
            * int(exact_spec["rows_per_range"])
            * int(exact_spec["keys_per_tile"])
        )
        total_group_area += int(fused_tail_live_pairs)

    geometry_base = {
        "cached_generalized_buckets": sum(family_counts.values()),
        "cached_generalized_zero_support_rows": zero_support_rows_count,
        "cached_generalized_live_pairs": total_live_pairs,
        "cached_generalized_baseline_packed_area": int(baseline_packed_k_sum),
        "cached_generalized_total_hardware_area": int(total_group_area),
        "cached_generalized_case_fill_rate": total_live_pairs / max(1, total_group_area),
        "cached_generalized_support_reduction": (
            0.0
            if baseline_live_row_count <= 0
            else 1.0
            - (
                sum(len(group_k_rows[group_idx]) * len(group_q_rows[group_idx]) for group_idx in range(len(group_q_rows)))
                / max(1, baseline_packed_k_sum)
            )
        ),
        "family_counts": family_counts,
        "family_live_pairs": family_live_pairs,
        "family_avg_active_density": {
            family: (
                family_bucket_stats[family]["active_density"] / family_bucket_stats[family]["count"]
                if family_bucket_stats[family]["count"] > 0
                else 0.0
            )
            for family in family_bucket_stats
        },
        "family_avg_k_gap_ratio": {
            family: (
                family_bucket_stats[family]["k_gap_ratio"] / family_bucket_stats[family]["count"]
                if family_bucket_stats[family]["count"] > 0
                else 0.0
            )
            for family in family_bucket_stats
        },
        "family_avg_q_gap_ratio": {
            family: (
                family_bucket_stats[family]["q_gap_ratio"] / family_bucket_stats[family]["count"]
                if family_bucket_stats[family]["count"] > 0
                else 0.0
            )
            for family in family_bucket_stats
        },
        "cached_pack_policy": asdict(resolved_policy),
    }
    return _finalize_generalized_cached_forward_payload(
        device=device,
        q_flat=q_flat,
        k_flat=k_flat,
        v_flat=v_flat,
        group_q_rows=group_q_rows,
        group_k_rows=group_k_rows,
        group_mask_words=group_mask_words,
        group_fill=group_fill,
        combine_group_ranges=combine_group_ranges,
        group_families=group_families,
        tile_k=int(resolved_policy.tile_k),
        geometry_base=geometry_base,
        reason="cached direct buckets were packed by a policy-selected family mix",
        exact_ranges=exact_ranges,
        exact_live_pairs=exact_live_pairs,
        fused_ranges=fused_ranges,
        fused_exact_live_pairs=int(fused_exact_live_pairs),
        fused_tail_live_pairs=int(fused_tail_live_pairs),
        union_kernel=str(resolved_policy.union_kernel),
        exact_kernel_family=str(exact_spec["family"]),
        exact_rows_per_range=int(exact_spec["rows_per_range"]),
        exact_keys_per_tile=int(exact_spec["keys_per_tile"]),
        exact_min_rows=int(exact_spec["min_rows"]),
        residual_mode=str(resolved_policy.residual_mode),
        include_mask_bool=include_mask_bool,
    )


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
    include_mask_bool: bool = False,
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
    group_q_rows: list[list[int]] = []
    group_k_rows: list[list[int]] = []
    group_mask_words: list[torch.Tensor] = []
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
                q_rows=merged_q_rows,
                support_lists=merged_support_lists,
                row_groups=row_groups,
                group_q_rows=group_q_rows,
                group_k_rows=group_k_rows,
                group_mask_words=group_mask_words,
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
            q_rows=fallback_q_rows,
            support_lists=fallback_support_lists,
            row_groups=row_groups,
            group_q_rows=group_q_rows,
            group_k_rows=group_k_rows,
            group_mask_words=group_mask_words,
            group_fill=group_fill,
        )
        if len(group_q_rows) > group_start:
            combine_group_ranges.append((group_start, len(group_q_rows)))
            fallback_row_count += len(fallback_q_rows)

    group_count = len(group_q_rows)
    if group_count <= 0:
        return {"status": "not_applicable", "reason": "cached direct buckets did not yield any live 2D forward groups"}

    q_row_idx, k_row_idx, mask_words, mask_bool, q_length, k_length, rows_per_group, max_union_k = (
        _materialize_cached_group_tensors(
            device=device,
            group_q_rows=group_q_rows,
            group_k_rows=group_k_rows,
            group_mask_words=group_mask_words,
            include_mask_bool=include_mask_bool,
        )
    )
    range_execution = _build_range_execution_metadata(group_q_rows, combine_group_ranges)
    range_execution = _annotate_range_execution_kernels(
        range_execution,
        rows_per_group=int(rows_per_group),
        max_union_k=int(max_union_k),
        tile_k=32,
        union_kernel="tc16x32",
        q_flat=q_flat,
        k_flat=k_flat,
        v_flat=v_flat,
    )
    scatter_only_ranges = sum(1 for entry in range_execution if bool(entry["scatter_only"]))
    scatter_only_rows = sum(
        len(payload_q_rows)
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
    payload = {
        "status": "ready",
        "reason": "cached direct buckets were repacked into 2D forward groups",
        "packed_q": rows_per_group,
        "support_rows": max_union_k,
        "tile_k": 32,
        "q_row_idx": q_row_idx.contiguous(),
        "k_row_idx": k_row_idx.contiguous(),
        "q_row_idx_flat": q_row_idx.contiguous().view(-1),
        "k_row_idx_flat": k_row_idx.contiguous().view(-1),
        "mask_words": mask_words.contiguous(),
        "q_length": q_length.contiguous(),
        "k_length": k_length.contiguous(),
        "total_rows": int(q_flat.shape[0]),
        "all_row_idx": torch.arange(int(q_flat.shape[0]), dtype=torch.int32, device=device).contiguous(),
        "combine_group_ranges": combine_group_ranges,
        "range_execution": range_execution,
        **_materialize_range_execution_tensors(range_execution, device=device),
        **_materialize_range_kernel_group_tensors(
            q_row_idx=q_row_idx,
            k_row_idx=k_row_idx,
            q_length=q_length,
            k_length=k_length,
            mask_words=mask_words,
            range_execution=range_execution,
        ),
        "geometry": geometry,
        "_workspace": {},
    }
    if mask_bool is not None:
        payload["mask_bool"] = mask_bool.contiguous()
    if str(payload.get("residual_mode", "")) == "fused_tail":
        backward_payload = build_cached_generalized_backward_payload(payload)
        if isinstance(backward_payload, dict):
            payload["cached_generalized_backward_payload"] = backward_payload
    return payload


def attach_precomputed_cached_generalized_forward_payload(
    schedule: Any,
    cached_payload: dict[str, Any],
    *,
    forward_block_q: int,
    logical_block_q: int = -1,
    logical_block_k: int = -1,
    max_packed_k: int = -1,
    max_direct_segments: int = -1,
    replace: bool = True,
) -> Any:
    """Attach a prebuilt cached generalized payload to a schedule.

    This keeps schedule/payload construction out of the per-step path while
    preserving the resolver format consumed by the HSA runtime.
    """

    if not isinstance(cached_payload, dict) or cached_payload.get("status") != "ready":
        raise ValueError("cached_payload must be a ready cached generalized forward payload")
    entry = {
        "forward_block_q": int(forward_block_q),
        "logical_block_q": int(logical_block_q),
        "logical_block_k": int(logical_block_k),
        "max_packed_k": int(max_packed_k),
        "max_direct_segments": int(max_direct_segments),
        "cached_generalized_forward_payload": cached_payload,
    }
    container = getattr(schedule, "_precomputed_forward_direct_plan_payload", None)
    next_container = dict(container) if isinstance(container, dict) else {}
    entries = []
    if isinstance(container, dict) and isinstance(container.get("entries"), list):
        entries = list(container["entries"])
    match_key = (
        int(forward_block_q),
        int(logical_block_q),
        int(logical_block_k),
        int(max_packed_k),
        int(max_direct_segments),
    )
    replaced = False
    if replace:
        for idx, existing in enumerate(entries):
            if not isinstance(existing, dict):
                continue
            existing_key = (
                int(existing.get("forward_block_q", -1)),
                int(existing.get("logical_block_q", -1)),
                int(existing.get("logical_block_k", -1)),
                int(existing.get("max_packed_k", -1)),
                int(existing.get("max_direct_segments", -1)),
            )
            if existing_key == match_key:
                merged = dict(existing)
                merged.update(entry)
                entries[idx] = merged
                replaced = True
                break
    if not replaced:
        entries.append(entry)
    next_container["entries"] = entries
    setattr(schedule, "_precomputed_forward_direct_plan_payload", next_container)
    for attr in (
        "_precomputed_cached_generalized_forward_payload_device_cache",
        "_resolved_cached_generalized_forward_payload_fast_cache",
        "_resolved_cached_generalized_forward_payload_cache",
    ):
        if hasattr(schedule, attr):
            delattr(schedule, attr)
    return schedule


def _build_cached_generalized_masked_union_row_compact_backward_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    q_row_idx = payload.get("q_row_idx")
    k_row_idx = payload.get("k_row_idx")
    mask_words = payload.get("mask_words")
    q_length = payload.get("q_length")
    k_length = payload.get("k_length")
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (q_row_idx, k_row_idx, mask_words, q_length, k_length)
    ):
        return None

    device = q_row_idx.device
    rows_per_member = 2
    union_chunk_cap = 16

    member_q_row_idx: list[list[int]] = []
    member_q_length: list[int] = []
    member_row_k_row_idx: list[list[list[int]]] = []
    member_row_k_to_union_idx: list[list[list[int]]] = []
    member_union_k_row_idx: list[list[int]] = []
    member_union_to_row_slot: list[list[list[int]]] = []
    member_row_k_length: list[list[int]] = []
    member_union_k_length: list[int] = []
    unique_key_occurrences: dict[int, list[tuple[int, int]]] = {}

    def append_member(
        pair_q_rows: list[int],
        union_rows: list[int],
        row_support_lists: list[list[int]],
    ) -> None:
        if not pair_q_rows or not union_rows:
            return
        member_idx = len(member_q_row_idx)
        q_pair = [int(value) for value in pair_q_rows[:rows_per_member]]
        q_pair.extend([-1] * (rows_per_member - len(q_pair)))
        member_q_row_idx.append(q_pair)
        member_q_length.append(min(rows_per_member, len(pair_q_rows)))

        union_rows = [int(value) for value in union_rows[:union_chunk_cap] if int(value) >= 0]
        if not union_rows:
            return
        union_index = {int(key_row): idx for idx, key_row in enumerate(union_rows)}
        padded_union_rows = union_rows + [-1] * (union_chunk_cap - len(union_rows))
        member_union_k_row_idx.append(padded_union_rows)
        member_union_k_length.append(len(union_rows))

        row_k_rows_entry: list[list[int]] = []
        row_k_to_union_entry: list[list[int]] = []
        union_to_row_entry: list[list[int]] = []
        row_k_length_entry: list[int] = []
        for row_slot in range(rows_per_member):
            support_rows = (
                [int(value) for value in row_support_lists[row_slot]]
                if row_slot < len(row_support_lists)
                else []
            )
            support_rows = [value for value in support_rows if value in union_index]
            row_k_length_entry.append(len(support_rows))
            row_k_rows_entry.append(support_rows + [-1] * (union_chunk_cap - len(support_rows)))
            row_k_to_union_entry.append(
                [union_index[value] for value in support_rows] + [-1] * (union_chunk_cap - len(support_rows))
            )
            union_to_row = [-1] * union_chunk_cap
            for row_local_slot, key_row in enumerate(support_rows):
                union_to_row[union_index[key_row]] = row_local_slot
            union_to_row_entry.append(union_to_row)
        member_row_k_row_idx.append(row_k_rows_entry)
        member_row_k_to_union_idx.append(row_k_to_union_entry)
        member_union_to_row_slot.append(union_to_row_entry)
        member_row_k_length.append(row_k_length_entry)

        for union_idx, key_row in enumerate(union_rows):
            unique_key_occurrences.setdefault(int(key_row), []).append((member_idx, union_idx))

    group_count = int(q_row_idx.shape[0])
    for group_idx in range(group_count):
        q_length_value = int(q_length[group_idx].item())
        k_length_value = int(k_length[group_idx].item())
        if q_length_value <= 0 or k_length_value <= 0:
            continue
        group_q_rows = [int(value) for value in q_row_idx[group_idx, :q_length_value].detach().to("cpu").tolist()]
        group_k_rows = [int(value) for value in k_row_idx[group_idx, :k_length_value].detach().to("cpu").tolist()]
        if not group_q_rows or not group_k_rows:
            continue
        word_cols = (k_length_value + 31) // 32
        group_mask_words = mask_words[group_idx, :q_length_value, :word_cols].detach().to("cpu").contiguous()
        mask_bool = _decode_mask_words_to_bool(group_mask_words, k_length_value)
        support_lists: list[list[int]] = []
        for row_idx in range(q_length_value):
            row_support = [
                group_k_rows[col_idx]
                for col_idx, keep in enumerate(mask_bool[row_idx].tolist())
                if keep
            ]
            support_lists.append(row_support)
        for pair_start in range(0, len(group_q_rows), rows_per_member):
            pair_q_rows = group_q_rows[pair_start : pair_start + rows_per_member]
            pair_support_lists = support_lists[pair_start : pair_start + len(pair_q_rows)]
            union_rows_all: list[int] = []
            seen_union_rows: set[int] = set()
            for support in pair_support_lists:
                for key_row in support:
                    key_row = int(key_row)
                    if key_row not in seen_union_rows:
                        seen_union_rows.add(key_row)
                        union_rows_all.append(key_row)
            if not union_rows_all:
                continue
            for chunk_start in range(0, len(union_rows_all), union_chunk_cap):
                union_chunk = union_rows_all[chunk_start : chunk_start + union_chunk_cap]
                union_chunk_set = set(union_chunk)
                row_support_chunk = [
                    [key_row for key_row in support if key_row in union_chunk_set]
                    for support in pair_support_lists
                ]
                append_member(pair_q_rows, union_chunk, row_support_chunk)

    if not member_q_row_idx:
        return None

    unique_key_row_idx_list: list[int] = []
    unique_key_member_idx_list: list[int] = []
    unique_key_union_idx_list: list[int] = []
    unique_key_occurrence_row_ptr_list = [0]
    max_unique_key_occurrences = 0
    for key_row in sorted(unique_key_occurrences):
        occurrences = unique_key_occurrences[key_row]
        max_unique_key_occurrences = max(max_unique_key_occurrences, len(occurrences))
        unique_key_row_idx_list.append(int(key_row))
        for member_idx, union_idx in occurrences:
            unique_key_member_idx_list.append(int(member_idx))
            unique_key_union_idx_list.append(int(union_idx))
        unique_key_occurrence_row_ptr_list.append(len(unique_key_member_idx_list))

    return {
        "status": "ready",
        "backward_kernel_family": "cached_masked_union_row_compact",
        "rows_per_member": int(rows_per_member),
        "union_chunk_cap": int(union_chunk_cap),
        "q_row_idx": torch.tensor(member_q_row_idx, dtype=torch.int32, device=device).contiguous(),
        "q_length": torch.tensor(member_q_length, dtype=torch.int32, device=device).contiguous(),
        "row_k_row_idx": torch.tensor(member_row_k_row_idx, dtype=torch.int32, device=device).contiguous(),
        "row_k_to_union_idx": torch.tensor(member_row_k_to_union_idx, dtype=torch.int32, device=device).contiguous(),
        "union_k_row_idx": torch.tensor(member_union_k_row_idx, dtype=torch.int32, device=device).contiguous(),
        "union_to_row_slot": torch.tensor(member_union_to_row_slot, dtype=torch.int32, device=device).contiguous(),
        "row_k_length": torch.tensor(member_row_k_length, dtype=torch.int32, device=device).contiguous(),
        "union_k_length": torch.tensor(member_union_k_length, dtype=torch.int32, device=device).contiguous(),
        "unique_key_row_idx": torch.tensor(unique_key_row_idx_list, dtype=torch.int32, device=device).contiguous(),
        "unique_key_member_idx": torch.tensor(unique_key_member_idx_list, dtype=torch.int32, device=device).contiguous(),
        "unique_key_union_idx": torch.tensor(unique_key_union_idx_list, dtype=torch.int32, device=device).contiguous(),
        "unique_key_occurrence_row_ptr": torch.tensor(
            unique_key_occurrence_row_ptr_list,
            dtype=torch.int32,
            device=device,
        ).contiguous(),
        "max_unique_key_occurrences": int(max_unique_key_occurrences),
        "_workspace": {},
    }


def build_cached_generalized_backward_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        return None
    if str(payload.get("exact_kernel_family", "")) != "tc8x8":
        return None
    residual_mode = str(payload.get("residual_mode", ""))
    if residual_mode == "masked_union":
        return _build_cached_generalized_masked_union_row_compact_backward_payload(payload)
    if residual_mode != "fused_tail":
        return None
    if int(payload.get("exact_dense_rows_per_range", 0)) != 8:
        return None
    if int(payload.get("exact_dense_keys_per_tile", 0)) != 8:
        return None

    fused_q_row_idx = payload.get("fused_q_row_idx")
    fused_q_length = payload.get("fused_q_length")
    fused_exact_tile_ptr = payload.get("fused_exact_tile_ptr")
    fused_exact_k_row_idx = payload.get("fused_exact_k_row_idx")
    fused_tail_tile_ptr = payload.get("fused_tail_tile_ptr")
    fused_tail_k_row_idx = payload.get("fused_tail_k_row_idx")
    fused_tail_mask_words = payload.get("fused_tail_mask_words")
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (
            fused_q_row_idx,
            fused_q_length,
            fused_exact_tile_ptr,
            fused_exact_k_row_idx,
            fused_tail_tile_ptr,
            fused_tail_k_row_idx,
            fused_tail_mask_words,
        )
    ):
        return None

    device = fused_q_row_idx.device
    q_row_idx_cpu = fused_q_row_idx.detach().to("cpu").tolist()
    q_length_cpu = fused_q_length.detach().to("cpu").tolist()
    exact_tile_ptr_cpu = fused_exact_tile_ptr.detach().to("cpu").tolist()
    exact_k_row_idx_cpu = fused_exact_k_row_idx.detach().to("cpu").tolist()
    tail_tile_ptr_cpu = fused_tail_tile_ptr.detach().to("cpu").tolist()
    tail_k_row_idx_cpu = fused_tail_k_row_idx.detach().to("cpu").tolist()
    tail_mask_words_cpu = fused_tail_mask_words.detach().to("cpu").tolist()

    q_rows_flat: list[int] = []
    for range_idx, q_length_value in enumerate(q_length_cpu):
        for row_idx in range(min(8, int(q_length_value))):
            q_row = int(q_row_idx_cpu[range_idx][row_idx])
            if q_row >= 0:
                q_rows_flat.append(q_row)
    if len(q_rows_flat) != len(set(q_rows_flat)):
        return None

    class _LocalKOverflow(Exception):
        pass

    max_local_k_per_range = 64
    owned_occurrences: dict[int, list[tuple[int, int, int, int]]] = {}
    range_local_k_ptr = [0]
    range_local_k_row_idx: list[int] = []
    exact_tile_local_k_idx = [[-1 for _ in range(8)] for _ in range(len(exact_k_row_idx_cpu))]
    tail_tile_local_k_idx = [[-1 for _ in range(8)] for _ in range(len(tail_k_row_idx_cpu))]
    try:
        for range_idx, q_length_value in enumerate(q_length_cpu):
            q_length_value = int(q_length_value)
            if q_length_value <= 0:
                range_local_k_ptr.append(len(range_local_k_row_idx))
                continue
            local_k_index: dict[int, int] = {}

            def _get_local_k_idx(key_row: int) -> int:
                local_idx = local_k_index.get(key_row)
                if local_idx is None:
                    local_idx = len(local_k_index)
                    if local_idx >= max_local_k_per_range:
                        raise _LocalKOverflow(
                            f"cached_generalized_backward_local_k_overflow range={range_idx} count>{max_local_k_per_range}"
                        )
                    local_k_index[key_row] = local_idx
                    range_local_k_row_idx.append(int(key_row))
                return local_idx

            exact_start = int(exact_tile_ptr_cpu[range_idx])
            exact_end = int(exact_tile_ptr_cpu[range_idx + 1])
            for tile_idx in range(exact_start, exact_end):
                tile_rows = exact_k_row_idx_cpu[tile_idx]
                for col_idx, key_row in enumerate(tile_rows):
                    key_row = int(key_row)
                    if key_row < 0:
                        continue
                    exact_tile_local_k_idx[tile_idx][col_idx] = _get_local_k_idx(key_row)
                    owned_occurrences.setdefault(key_row, []).append((0, int(range_idx), int(tile_idx), int(col_idx)))

            tail_start = int(tail_tile_ptr_cpu[range_idx])
            tail_end = int(tail_tile_ptr_cpu[range_idx + 1])
            for tile_idx in range(tail_start, tail_end):
                tile_rows = tail_k_row_idx_cpu[tile_idx]
                row_masks = [int(mask_words[0]) for mask_words in tail_mask_words_cpu[tile_idx][:q_length_value]]
                active_cols = 0
                for row_mask in row_masks:
                    active_cols |= int(row_mask)
                for col_idx, key_row in enumerate(tile_rows):
                    key_row = int(key_row)
                    if key_row < 0 or ((active_cols >> col_idx) & 1) == 0:
                        continue
                    tail_tile_local_k_idx[tile_idx][col_idx] = _get_local_k_idx(key_row)
                    owned_occurrences.setdefault(key_row, []).append((1, int(range_idx), int(tile_idx), int(col_idx)))
            range_local_k_ptr.append(len(range_local_k_row_idx))
    except _LocalKOverflow:
        return None

    owned_k_row_idx: list[int] = []
    owned_occurrence_ptr = [0]
    owned_occurrence_kind: list[int] = []
    owned_occurrence_range_idx: list[int] = []
    owned_occurrence_tile_idx: list[int] = []
    owned_occurrence_col_idx: list[int] = []
    for key_row in sorted(owned_occurrences):
        owned_k_row_idx.append(int(key_row))
        for kind, range_idx, tile_idx, col_idx in owned_occurrences[key_row]:
            owned_occurrence_kind.append(int(kind))
            owned_occurrence_range_idx.append(int(range_idx))
            owned_occurrence_tile_idx.append(int(tile_idx))
            owned_occurrence_col_idx.append(int(col_idx))
        owned_occurrence_ptr.append(len(owned_occurrence_kind))

    return {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "rows_per_range": 8,
        "keys_per_tile": 8,
        "head_dim": 64,
        "max_local_k_per_range": max_local_k_per_range,
        "range_local_k_ptr": torch.tensor(range_local_k_ptr, dtype=torch.int32, device=device).contiguous(),
        "range_local_k_row_idx": torch.tensor(range_local_k_row_idx, dtype=torch.int32, device=device).contiguous(),
        "exact_tile_local_k_idx": torch.tensor(exact_tile_local_k_idx, dtype=torch.int32, device=device).contiguous(),
        "tail_tile_local_k_idx": torch.tensor(tail_tile_local_k_idx, dtype=torch.int32, device=device).contiguous(),
        "owned_k_row_idx": torch.tensor(owned_k_row_idx, dtype=torch.int32, device=device).contiguous(),
        "owned_occurrence_ptr": torch.tensor(owned_occurrence_ptr, dtype=torch.int32, device=device).contiguous(),
        "owned_occurrence_kind": torch.tensor(owned_occurrence_kind, dtype=torch.int32, device=device).contiguous(),
        "owned_occurrence_range_idx": torch.tensor(
            owned_occurrence_range_idx, dtype=torch.int32, device=device
        ).contiguous(),
        "owned_occurrence_tile_idx": torch.tensor(
            owned_occurrence_tile_idx, dtype=torch.int32, device=device
        ).contiguous(),
        "owned_occurrence_col_idx": torch.tensor(
            owned_occurrence_col_idx, dtype=torch.int32, device=device
        ).contiguous(),
    }


def _get_cached_direct_2d_output_buffers(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace = payload.setdefault("_workspace", {})
    key = (
        "output",
        str(q_flat.device),
        q_flat.dtype,
        v_flat.dtype,
        q_flat.shape[1],
        v_flat.shape[2],
        int(payload["total_rows"]),
    )
    buffers = workspace.get(key)
    if buffers is None:
        buffers = (
            torch.zeros((int(payload["total_rows"]), q_flat.shape[1], v_flat.shape[2]), dtype=torch.float32, device=v_flat.device),
            torch.empty((int(payload["total_rows"]), q_flat.shape[1]), dtype=torch.float32, device=v_flat.device),
        )
        workspace[key] = buffers
    return buffers


def _get_cached_direct_2d_pack_buffers(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    workspace = payload.setdefault("_workspace", {})
    key = (
        "pack",
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
        )
        workspace[key] = buffers
    return buffers


def _get_cached_all_row_idx(payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    row_idx = payload.get("all_row_idx")
    total_rows = int(payload["total_rows"])
    if isinstance(row_idx, torch.Tensor) and row_idx.device == device and int(row_idx.numel()) == total_rows:
        return row_idx
    workspace = payload.setdefault("_workspace", {})
    key = ("all_row_idx", str(device), total_rows)
    cached = workspace.get(key) if isinstance(workspace, dict) else None
    if isinstance(cached, torch.Tensor) and cached.device == device and int(cached.numel()) == total_rows:
        return cached
    row_idx = torch.arange(total_rows, dtype=torch.int32, device=device).contiguous()
    if isinstance(workspace, dict):
        workspace[key] = row_idx
    payload["all_row_idx"] = row_idx
    return row_idx


def _payload_row_device(payload: dict[str, Any], fallback: Any | None = None) -> torch.device:
    device = getattr(fallback, "device", None)
    if isinstance(device, torch.device):
        return device
    for key in (
        "fused_q_row_idx",
        "exact_dense_q_row_idx",
        "range_tc_scatter_q_row_idx",
        "range_scatter_q_row_idx",
        "range_packed_q_row_idx",
        "q_row_idx",
    ):
        tensor = payload.get(key)
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return torch.device("cpu")


def _valid_unique_payload_rows(
    row_idx: torch.Tensor,
    *,
    total_rows: int,
    device: torch.device,
) -> torch.Tensor:
    rows = row_idx.reshape(-1).to(device=device, dtype=torch.int32)
    if int(rows.numel()) == 0:
        return rows
    rows = rows[(rows >= 0) & (rows < int(total_rows))]
    if int(rows.numel()) == 0:
        return rows
    return torch.unique(rows, sorted=True)


def _get_direct_final_base_row_idx(
    payload: dict[str, Any],
    device: torch.device,
    *,
    base_source: str = "auto",
) -> torch.Tensor:
    total_rows = int(payload["total_rows"])
    if base_source == "auto":
        base_source = (
            "fused"
            if int(getattr(payload.get("fused_q_row_idx"), "shape", [0])[0]) > 0
            else "exact_dense"
        )
    key_name = "fused_q_row_idx" if base_source == "fused" else "exact_dense_q_row_idx"
    row_idx = payload.get(key_name)
    if not isinstance(row_idx, torch.Tensor):
        return torch.empty((0,), dtype=torch.int32, device=device)
    workspace = payload.setdefault("_workspace", {})
    cache_key = ("direct_final_base_rows", str(device), base_source, total_rows, int(row_idx.data_ptr()))
    cached = workspace.get(cache_key) if isinstance(workspace, dict) else None
    if isinstance(cached, torch.Tensor) and cached.device == device:
        return cached
    rows = _valid_unique_payload_rows(row_idx, total_rows=total_rows, device=device)
    if isinstance(workspace, dict):
        workspace[cache_key] = rows
    return rows


def _get_direct_final_residual_row_idx(
    payload: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    total_rows = int(payload["total_rows"])
    row_tensors = [
        payload.get("range_tc_scatter_q_row_idx"),
        payload.get("range_scatter_q_row_idx"),
        payload.get("range_packed_q_row_idx"),
    ]
    tensors = [tensor for tensor in row_tensors if isinstance(tensor, torch.Tensor) and int(tensor.numel()) > 0]
    if not tensors:
        return torch.empty((0,), dtype=torch.int32, device=device)
    workspace = payload.setdefault("_workspace", {})
    cache_key = (
        "direct_final_residual_rows",
        str(device),
        total_rows,
        tuple(int(tensor.data_ptr()) for tensor in tensors),
    )
    cached = workspace.get(cache_key) if isinstance(workspace, dict) else None
    if isinstance(cached, torch.Tensor) and cached.device == device:
        return cached
    rows = _valid_unique_payload_rows(
        torch.cat([tensor.reshape(-1).to(device=device, dtype=torch.int32) for tensor in tensors]),
        total_rows=total_rows,
        device=device,
    )
    if isinstance(workspace, dict):
        workspace[cache_key] = rows
    return rows


def _get_direct_final_missing_init_row_idx(
    payload: dict[str, Any],
    device: torch.device,
    *,
    base_source: str = "auto",
) -> torch.Tensor:
    total_rows = int(payload["total_rows"])
    base_rows = _get_direct_final_base_row_idx(payload, device, base_source=base_source)
    residual_rows = _get_direct_final_residual_row_idx(payload, device)
    if int(residual_rows.numel()) == 0:
        return residual_rows
    workspace = payload.setdefault("_workspace", {})
    cache_key = (
        "direct_final_missing_init_rows",
        str(device),
        base_source,
        total_rows,
        int(base_rows.data_ptr()) if int(base_rows.numel()) > 0 else 0,
        int(residual_rows.data_ptr()),
    )
    cached = workspace.get(cache_key) if isinstance(workspace, dict) else None
    if isinstance(cached, torch.Tensor) and cached.device == device:
        return cached
    if int(base_rows.numel()) == 0:
        missing_rows = residual_rows
    else:
        missing_rows = residual_rows[~torch.isin(residual_rows, base_rows)]
    if isinstance(workspace, dict):
        workspace[cache_key] = missing_rows
    return missing_rows


def _direct_final_base_residual_union_row_count(
    payload: dict[str, Any],
    device: torch.device,
    *,
    base_source: str = "auto",
) -> int:
    base_rows = _get_direct_final_base_row_idx(payload, device, base_source=base_source)
    residual_rows = _get_direct_final_residual_row_idx(payload, device)
    if int(base_rows.numel()) == 0:
        return int(residual_rows.numel())
    if int(residual_rows.numel()) == 0:
        return int(base_rows.numel())
    return int(torch.unique(torch.cat([base_rows, residual_rows]), sorted=False).numel())


def _slice_cached_flat_row_idx(
    payload: dict[str, Any],
    *,
    flat_key: str,
    matrix_key: str,
    group_start: int,
    group_end: int,
    width: int,
) -> torch.Tensor:
    flat = payload.get(flat_key)
    if isinstance(flat, torch.Tensor):
        return flat[int(group_start) * int(width) : int(group_end) * int(width)]
    return payload[matrix_key][group_start:group_end].reshape(-1).contiguous()


def _get_cached_direct_2d_final_buffers(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(
        (int(payload["total_rows"]), q_flat.shape[1], v_flat.shape[2]),
        dtype=v_flat.dtype,
        device=v_flat.device,
    )
    lse = torch.empty(
        (int(payload["total_rows"]), q_flat.shape[1]),
        dtype=torch.float32,
        device=v_flat.device,
    )
    return out, lse


def _get_cached_backward_accum_buffers(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    workspace = _get_cached_backward_workspace(payload)
    key = (
        "backward_accum",
        str(q_flat.device),
        q_flat.shape,
        k_flat.shape,
        v_flat.shape,
    )
    buffers = workspace.get(key)
    if buffers is None:
        buffers = (
            torch.empty_like(q_flat, dtype=torch.float32),
            torch.empty_like(k_flat, dtype=torch.float32),
            torch.empty_like(v_flat, dtype=torch.float32),
        )
        workspace[key] = buffers
    return buffers


def _get_cached_backward_kv_accum_buffers(
    payload: dict[str, Any],
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    workspace = _get_cached_backward_workspace(payload)
    key = (
        "backward_kv_accum",
        str(k_flat.device),
        k_flat.shape,
        v_flat.shape,
    )
    buffers = workspace.get(key)
    if buffers is None:
        buffers = (
            torch.empty_like(k_flat, dtype=torch.float32),
            torch.empty_like(v_flat, dtype=torch.float32),
        )
        workspace[key] = buffers
    return buffers


def _finalize_cached_backward_grads(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    dq_acc: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_idx = _get_cached_all_row_idx(payload, q_flat.device)
    dq = torch.empty_like(q_flat)
    dk = torch.empty_like(k_flat)
    dv = torch.empty_like(v_flat)
    if q_flat.is_cuda and _is_env_enabled("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_FINALIZE"):
        _run_cached_cast_three_rows_kernel(dq_acc, dk_acc, dv_acc, row_idx, dq, dk, dv)
    else:
        _run_cached_cast_rows_kernel(dq_acc, row_idx, dq)
        _run_cached_cast_rows_kernel(dk_acc, row_idx, dk)
        _run_cached_cast_rows_kernel(dv_acc, row_idx, dv)
    return dq, dk, dv


def _finalize_cached_backward_kv_grads(
    payload: dict[str, Any],
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_idx = _get_cached_all_row_idx(payload, k_flat.device)
    dk = torch.empty_like(k_flat)
    dv = torch.empty_like(v_flat)
    if k_flat.is_cuda and _is_env_enabled("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_FINALIZE"):
        _run_cached_cast_two_rows_kernel(dk_acc, dv_acc, row_idx, dk, dv)
    else:
        _run_cached_cast_rows_kernel(dk_acc, row_idx, dk)
        _run_cached_cast_rows_kernel(dv_acc, row_idx, dv)
    return dk, dv


def _can_use_triton_lse_public_to_flat(
    q: torch.Tensor,
    q_flat: torch.Tensor,
    lse: torch.Tensor,
) -> bool:
    if not _HAS_TRITON_LSE_PUBLIC_TO_FLAT:
        return False
    if not _is_env_enabled("FLASH_ATTN_HSA_CACHED_BWD_TRITON_LSE_PUBLIC_TO_FLAT", default="on"):
        return False
    if q.ndim != 4 or lse.ndim != 3:
        return False
    if not (q_flat.is_cuda and lse.is_cuda):
        return False
    if lse.dtype != torch.float32 or not lse.is_contiguous():
        return False
    batch = int(q.shape[0])
    seqlen = int(q.shape[1])
    num_heads = int(q.shape[2])
    if num_heads not in (8, 16):
        return False
    return (
        int(lse.shape[0]) == batch
        and int(lse.shape[1]) == num_heads
        and int(lse.shape[2]) == seqlen
        and int(q_flat.shape[0]) == batch * seqlen
    )


def _run_triton_lse_public_to_flat(
    src_public_lse: torch.Tensor,
    dst_flat_lse: torch.Tensor,
) -> None:
    if not _HAS_TRITON_LSE_PUBLIC_TO_FLAT:
        raise RuntimeError("Triton public-LSE-to-flat kernel is unavailable")
    batch = int(src_public_lse.shape[0])
    num_heads = int(src_public_lse.shape[1])
    seqlen = int(src_public_lse.shape[2])
    total_elems = batch * seqlen * num_heads
    block_elems = 2048
    grid = (triton.cdiv(total_elems, block_elems),)
    _triton_lse_public_to_flat_kernel[grid](
        src_public_lse,
        dst_flat_lse,
        total_elems,
        seqlen,
        num_heads,
        block_elems,
        num_warps=4,
    )


def _get_cached_lse_flat_for_backward_buffer(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    workspace = payload.setdefault("_workspace", {})
    shape = (int(q_flat.shape[0]), int(num_heads))
    cache_key = ("lse_flat_for_backward", str(q_flat.device), shape)
    cached = workspace.get(cache_key) if isinstance(workspace, dict) else None
    if (
        isinstance(cached, torch.Tensor)
        and tuple(cached.shape) == shape
        and cached.device == q_flat.device
        and cached.dtype == torch.float32
    ):
        return cached
    lse_flat = torch.empty(shape, dtype=torch.float32, device=q_flat.device)
    if isinstance(workspace, dict):
        workspace[cache_key] = lse_flat
    return lse_flat


def _flatten_cached_lse_for_backward(
    q: torch.Tensor,
    q_flat: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    triton_checked: bool = False,
) -> torch.Tensor:
    if lse.ndim == 2 and int(lse.shape[0]) == int(q_flat.shape[0]):
        return lse.float() if lse.dtype != torch.float32 else lse
    if triton_checked or _can_use_triton_lse_public_to_flat(q, q_flat, lse):
        expected_shape = (int(q_flat.shape[0]), int(q.shape[2]))
        if (
            out is not None
            and tuple(out.shape) == expected_shape
            and out.device == lse.device
            and out.dtype == torch.float32
            and out.is_contiguous()
        ):
            lse_flat = out
        else:
            lse_flat = torch.empty(expected_shape, dtype=torch.float32, device=lse.device)
        _run_triton_lse_public_to_flat(lse, lse_flat)
        return lse_flat
    if (
        q.ndim == 4
        and _is_env_enabled("FLASH_ATTN_HSA_CACHED_BWD_CUTE_LSE_PUBLIC_TO_FLAT", default="off")
        and q_flat.is_cuda
        and lse.is_cuda
        and lse.ndim == 3
        and int(lse.shape[0]) == int(q.shape[0])
        and int(lse.shape[1]) == int(q.shape[2])
        and int(lse.shape[2]) == int(q.shape[1])
    ):
        lse_flat = torch.empty(
            (int(q_flat.shape[0]), int(q.shape[2])),
            dtype=torch.float32,
            device=lse.device,
        )
        _run_cached_lse_public_to_flat_kernel(lse, lse_flat)
        return lse_flat
    if q.ndim == 4:
        return lse.permute(0, 2, 1).contiguous().view(-1, q.shape[2]).float()
    return lse.transpose(0, 1).contiguous().float()


def _cached_backward_dq_overwrites_all_rows(payload: dict[str, Any], q_flat: torch.Tensor) -> bool:
    if str(payload.get("residual_mode", "")) != "fused_tail":
        return False
    if int(payload.get("fused_output_row_count", -1)) != int(payload.get("total_rows", -2)):
        return False
    if int(payload.get("total_rows", -1)) != int(q_flat.shape[0]):
        return False
    fused_q_row_idx = payload.get("fused_q_row_idx")
    fused_q_length = payload.get("fused_q_length")
    if not isinstance(fused_q_row_idx, torch.Tensor) or not isinstance(fused_q_length, torch.Tensor):
        return False
    if int(fused_q_row_idx.shape[0]) <= 0:
        return False
    return True


def _use_cached_backward_direct_dq(payload: dict[str, Any], q_flat: torch.Tensor) -> bool:
    if not q_flat.is_cuda:
        return False
    if not _is_env_enabled("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD"):
        return False
    return _cached_backward_dq_overwrites_all_rows(payload, q_flat)


def _zero_cached_backward_kv_accum_buffers(
    payload: dict[str, Any],
    k_flat: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
) -> None:
    if k_flat.is_cuda:
        all_row_idx = _get_cached_all_row_idx(payload, k_flat.device)
        if _is_env_enabled("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO"):
            _run_cached_zero_two_rows_kernel(all_row_idx, dk_acc, dv_acc)
            return
        _run_cached_zero_rows_kernel(all_row_idx, dk_acc)
        _run_cached_zero_rows_kernel(all_row_idx, dv_acc)
        return
    dk_acc.zero_()
    dv_acc.zero_()


def _zero_cached_backward_kv_final_buffers(
    payload: dict[str, Any],
    k_flat: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
) -> None:
    if k_flat.is_cuda:
        all_row_idx = _get_cached_all_row_idx(payload, k_flat.device)
        if _is_env_enabled("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO"):
            _run_cached_zero_two_rows_kernel(all_row_idx, dk, dv)
            return
        _run_cached_zero_rows_kernel(all_row_idx, dk)
        _run_cached_zero_rows_kernel(all_row_idx, dv)
        return
    dk.zero_()
    dv.zero_()


def _can_use_cached_backward_key_owned_dkdv(backward_payload: dict[str, Any] | None) -> bool:
    if not isinstance(backward_payload, dict) or backward_payload.get("status") != "ready":
        return False
    if str(backward_payload.get("backward_kernel_family", "")) != "cached_tc8x8_fused":
        return False
    required = (
        "owned_k_row_idx",
        "owned_occurrence_ptr",
        "owned_occurrence_kind",
        "owned_occurrence_range_idx",
        "owned_occurrence_tile_idx",
        "owned_occurrence_col_idx",
    )
    if not all(isinstance(backward_payload.get(key), torch.Tensor) for key in required):
        return False
    return int(backward_payload["owned_k_row_idx"].numel()) > 0


def _cached_backward_key_owned_overwrites_all_kv_rows(
    backward_payload: dict[str, Any] | None,
    k_flat: torch.Tensor,
) -> bool:
    if not _can_use_cached_backward_key_owned_dkdv(backward_payload):
        return False
    owned_k_row_idx = backward_payload["owned_k_row_idx"]
    # The payload builder materializes this list from a Python dict keyed by
    # key row, then sorts it. Equal cardinality means every KV row has exactly
    # one owning CTA and the key-owned kernel overwrites the whole DK/DV output.
    return int(owned_k_row_idx.numel()) == int(k_flat.shape[0])


def _auto_use_cached_backward_key_owned_dkdv(
    backward_payload: dict[str, Any] | None,
    k_flat: torch.Tensor,
) -> bool:
    if not _cached_backward_key_owned_overwrites_all_kv_rows(backward_payload, k_flat):
        return False
    try:
        max_rows = int(os.environ.get("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED_MAX_ROWS", "2048"))
    except ValueError:
        max_rows = 2048
    return int(k_flat.shape[0]) <= max(0, max_rows)


def _zero_cached_backward_accum_buffers(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    dq_acc: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
) -> None:
    if q_flat.is_cuda:
        all_row_idx = _get_cached_all_row_idx(payload, q_flat.device)
        if _cached_backward_dq_overwrites_all_rows(payload, q_flat):
            if _is_env_enabled("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO"):
                _run_cached_zero_two_rows_kernel(all_row_idx, dk_acc, dv_acc)
                return
            _run_cached_zero_rows_kernel(all_row_idx, dk_acc)
            _run_cached_zero_rows_kernel(all_row_idx, dv_acc)
            return
        if _is_env_enabled("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO"):
            _run_cached_zero_three_rows_kernel(all_row_idx, dq_acc, dk_acc, dv_acc)
            return
        _run_cached_zero_rows_kernel(all_row_idx, dq_acc)
        _run_cached_zero_rows_kernel(all_row_idx, dk_acc)
        _run_cached_zero_rows_kernel(all_row_idx, dv_acc)
        return
    dq_acc.zero_()
    dk_acc.zero_()
    dv_acc.zero_()


def _cached_union_range_counts(
    payload: dict[str, Any],
    *,
    group_start: int,
    group_end: int,
) -> tuple[int, int]:
    row_count = int(payload["q_length"][group_start:group_end].sum().item())
    return int(group_end - group_start), row_count


def _can_use_cached_union_tc(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    *,
    group_start: int,
    group_end: int,
    range_entry: dict[str, Any],
) -> bool:
    if str(payload.get("union_kernel", "scalar")) != "tc16x32":
        return False
    if str(range_entry.get("family", "union_2d")) != "union_2d":
        return False
    if not bool(range_entry.get("scatter_only")):
        return False
    if int(payload["packed_q"]) != 16 or int(payload.get("tile_k", 32)) != 32:
        return False
    if int(payload["support_rows"]) <= 0 or int(payload["support_rows"]) > 128:
        return False
    if not _can_use_synthetic_2d_masked_fwd(
        q_flat,
        k_flat,
        v_flat,
        packed_q=int(payload["packed_q"]),
        packed_k=int(payload["support_rows"]),
    ):
        return False
    range_q_lengths = payload["q_length"][group_start:group_end]
    if int(range_q_lengths.numel()) <= 0 or int(range_q_lengths.max().item()) > 16:
        return False
    return True


def _record_union_runtime_geometry(
    payload: dict[str, Any],
    *,
    tc_group_count: int,
    tc_row_count: int,
    scalar_group_count: int,
    scalar_row_count: int,
) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return
    geometry["union_kernel"] = str(payload.get("union_kernel", geometry.get("union_kernel", "scalar")))
    geometry["union_tc_group_count"] = int(tc_group_count)
    geometry["union_tc_row_count"] = int(tc_row_count)
    geometry["union_scalar_fallback_group_count"] = int(scalar_group_count)
    geometry["union_scalar_fallback_row_count"] = int(scalar_row_count)


def _record_exact_dense_runtime_geometry(
    payload: dict[str, Any],
    *,
    exact_range_count: int,
    exact_row_count: int,
) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return
    geometry["exact_dense_runtime_range_count"] = int(exact_range_count)
    geometry["exact_dense_runtime_row_count"] = int(exact_row_count)


def _record_fused_runtime_geometry(
    payload: dict[str, Any],
    *,
    fused_range_count: int,
    fused_row_count: int,
) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return
    geometry["fused_runtime_range_count"] = int(fused_range_count)
    geometry["fused_runtime_row_count"] = int(fused_row_count)


def _record_cached_forward_path(
    payload: dict[str, Any],
    *,
    path: str,
    reason: str | None = None,
) -> None:
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return
    geometry["cached_forward_runtime_path"] = str(path)
    if reason is not None:
        geometry["cached_forward_runtime_fallback_reason"] = str(reason)


def _run_cached_fused_exact_tail_ranges(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[int, int]:
    fused_q_row_idx = payload.get("fused_q_row_idx")
    fused_q_length = payload.get("fused_q_length")
    fused_exact_tile_ptr = payload.get("fused_exact_tile_ptr")
    fused_exact_k_row_idx = payload.get("fused_exact_k_row_idx")
    fused_tail_tile_ptr = payload.get("fused_tail_tile_ptr")
    fused_tail_k_row_idx = payload.get("fused_tail_k_row_idx")
    fused_tail_mask_words = payload.get("fused_tail_mask_words")
    if not isinstance(fused_q_row_idx, torch.Tensor) or int(fused_q_row_idx.shape[0]) <= 0:
        return 0, 0
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (
            fused_q_length,
            fused_exact_tile_ptr,
            fused_exact_k_row_idx,
            fused_tail_tile_ptr,
            fused_tail_k_row_idx,
            fused_tail_mask_words,
        )
    ):
        return 0, 0
    if str(payload.get("exact_kernel_family", "")) != "tc8x8":
        return 0, 0
    if int(payload.get("exact_dense_rows_per_range", 0)) != 8 or int(payload.get("exact_dense_keys_per_tile", 0)) != 8:
        return 0, 0
    if str(payload.get("residual_mode", "masked_union")) != "fused_tail":
        return 0, 0
    _run_synthetic_2d_exact_tail_gather_scatter_tc_fwd_kernel(
        q_flat,
        k_flat,
        v_flat,
        fused_q_row_idx,
        fused_q_length,
        fused_exact_tile_ptr,
        fused_exact_k_row_idx,
        fused_tail_tile_ptr,
        fused_tail_k_row_idx,
        fused_tail_mask_words,
        out_flat,
        lse_flat,
        softmax_scale=float(softmax_scale),
    )
    return int(fused_q_row_idx.shape[0]), int(payload.get("fused_output_row_count", 0))


def _run_cached_exact_dense_ranges(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[int, int]:
    exact_q_row_idx = payload.get("exact_dense_q_row_idx")
    exact_q_length = payload.get("exact_dense_q_length")
    exact_tile_ptr = payload.get("exact_dense_tile_ptr")
    exact_k_row_idx = payload.get("exact_dense_k_row_idx")
    if not isinstance(exact_q_row_idx, torch.Tensor) or int(exact_q_row_idx.shape[0]) <= 0:
        return 0, 0
    if not isinstance(exact_q_length, torch.Tensor) or not isinstance(exact_tile_ptr, torch.Tensor):
        return 0, 0
    if not isinstance(exact_k_row_idx, torch.Tensor):
        return 0, 0
    exact_family = str(payload.get("exact_kernel_family", "tc16x16"))
    exact_rows_per_range = int(payload.get("exact_dense_rows_per_range", 0))
    exact_keys_per_tile = int(payload.get("exact_dense_keys_per_tile", 0))
    exact_spec = _resolve_exact_kernel_spec(exact_family)
    if (
        exact_rows_per_range != int(exact_spec["rows_per_range"])
        or exact_keys_per_tile != int(exact_spec["keys_per_tile"])
    ):
        return 0, 0
    _run_synthetic_2d_exact_gather_scatter_tc_fwd_kernel(
        q_flat,
        k_flat,
        v_flat,
        exact_q_row_idx,
        exact_q_length,
        exact_tile_ptr,
        exact_k_row_idx,
        out_flat,
        lse_flat,
        softmax_scale=float(softmax_scale),
        kernel_family=exact_family,
    )
    return int(exact_q_row_idx.shape[0]), int(payload.get("exact_dense_output_row_count", 0))


def _run_cached_masked_payload_forward(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    *,
    softmax_scale: float,
    force_combine_scatter: bool = False,
) -> tuple[int, int, int, int]:
    packed_q = int(payload["packed_q"])
    packed_k = int(payload["support_rows"])
    tile_k = int(payload.get("tile_k", 32))
    group_count = int(payload["q_row_idx"].shape[0])
    if group_count <= 0:
        return 0, 0, 0, 0

    pack_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    range_execution = payload.get("range_execution")
    if not range_execution:
        range_execution = [
            {
                "group_start": 0,
                "group_end": group_count,
                "scatter_only": False,
            }
        ]
    union_tc_group_count = 0
    union_tc_row_count = 0
    union_scalar_group_count = 0
    union_scalar_row_count = 0
    try:
        group_chunk_limit = max(1, int(os.environ.get("FLASH_ATTN_HSA_CACHED_MASKED_GROUP_CHUNK", "512")))
    except ValueError:
        group_chunk_limit = 128

    def _run_range_packed(group_start: int, group_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        range_group_count = group_end - group_start
        if _can_use_synthetic_2d_masked_fwd(
            q_flat,
            k_flat,
            v_flat,
            packed_q=packed_q,
            packed_k=packed_k,
        ):
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
                tile_k=tile_k,
            )
        nonlocal pack_buffers
        if pack_buffers is None:
            pack_buffers = _get_cached_direct_2d_pack_buffers(payload, q_flat, k_flat, v_flat)
        q_buf_flat, k_buf_flat, v_buf_flat = pack_buffers
        q_buf_range = q_buf_flat[: range_group_count * packed_q]
        k_buf_range = k_buf_flat[: range_group_count * packed_k]
        v_buf_range = v_buf_flat[: range_group_count * packed_k]
        _run_synthetic_pack_rows_kernel(
            q_flat,
            _slice_cached_flat_row_idx(
                payload,
                flat_key="q_row_idx_flat",
                matrix_key="q_row_idx",
                group_start=group_start,
                group_end=group_end,
                width=packed_q,
            ),
            q_buf_range,
        )
        _run_synthetic_pack_kv_rows_kernel(
            k_flat,
            v_flat,
            _slice_cached_flat_row_idx(
                payload,
                flat_key="k_row_idx_flat",
                matrix_key="k_row_idx",
                group_start=group_start,
                group_end=group_end,
                width=packed_k,
            ),
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
            tile_k=tile_k,
        )

    def _run_masked_group_range(group_start: int, group_end: int, range_entry: dict[str, Any]) -> None:
        nonlocal union_tc_group_count, union_tc_row_count, union_scalar_group_count, union_scalar_row_count
        if group_end <= group_start:
            return
        kernel_kind = str(range_entry.get("kernel_kind", ""))
        if not kernel_kind:
            if _can_use_cached_union_tc(
                payload,
                q_flat,
                k_flat,
                v_flat,
                group_start=group_start,
                group_end=group_end,
                range_entry=range_entry,
            ):
                kernel_kind = "tc_scatter"
            elif bool(range_entry.get("scatter_only")):
                kernel_kind = "scatter"
            else:
                kernel_kind = "packed"
        if kernel_kind == "tc_scatter":
            if force_combine_scatter:
                _run_synthetic_2d_masked_gather_combine_fwd_kernel(
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
                    tile_k=tile_k,
                )
                tc_groups, tc_rows = _cached_union_range_counts(payload, group_start=group_start, group_end=group_end)
                union_tc_group_count += tc_groups
                union_tc_row_count += tc_rows
                return
            _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel(
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
                tile_k=tile_k,
            )
            tc_groups, tc_rows = _cached_union_range_counts(payload, group_start=group_start, group_end=group_end)
            union_tc_group_count += tc_groups
            union_tc_row_count += tc_rows
            return
        if kernel_kind == "scatter":
            if str(range_entry.get("family", "union_2d")) == "union_2d":
                scalar_groups, scalar_rows = _cached_union_range_counts(
                    payload,
                    group_start=group_start,
                    group_end=group_end,
                )
                union_scalar_group_count += scalar_groups
                union_scalar_row_count += scalar_rows
            if force_combine_scatter:
                _run_synthetic_2d_masked_gather_combine_fwd_kernel(
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
                    tile_k=tile_k,
                )
                return
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
                tile_k=tile_k,
            )
            return
        if kernel_kind != "packed":
            raise RuntimeError(f"unsupported cached 2D range kernel kind: {kernel_kind}")
        if _can_use_synthetic_2d_masked_fwd(
            q_flat,
            k_flat,
            v_flat,
            packed_q=packed_q,
            packed_k=packed_k,
        ):
            _run_synthetic_2d_masked_gather_combine_fwd_kernel(
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
                tile_k=tile_k,
            )
            return
        packed_out, packed_lse = _run_range_packed(group_start, group_end)
        range_group_count = group_end - group_start
        packed_out_flat = packed_out.view(range_group_count * packed_q, packed_out.shape[2], packed_out.shape[3]).contiguous()
        packed_lse_flat = packed_lse.view(range_group_count * packed_q, packed_lse.shape[2]).contiguous()
        _run_synthetic_combine_scatter_rows_kernel(
            packed_out_flat,
            packed_lse_flat,
            _slice_cached_flat_row_idx(
                payload,
                flat_key="q_row_idx_flat",
                matrix_key="q_row_idx",
                group_start=group_start,
                group_end=group_end,
                width=packed_q,
            ),
            out_flat,
            lse_flat,
        )

    def _run_grouped_packed(
        q_row_idx: torch.Tensor,
        k_row_idx: torch.Tensor,
        q_length: torch.Tensor,
        k_length: torch.Tensor,
        mask_words: torch.Tensor,
        q_row_idx_flat: torch.Tensor,
        k_row_idx_flat: torch.Tensor,
    ) -> None:
        nonlocal pack_buffers
        grouped_count = int(q_row_idx.shape[0])
        if grouped_count <= 0:
            return
        if _can_use_synthetic_2d_masked_fwd(
            q_flat,
            k_flat,
            v_flat,
            packed_q=packed_q,
            packed_k=packed_k,
        ):
            _run_synthetic_2d_masked_gather_combine_fwd_kernel(
                q_flat,
                k_flat,
                v_flat,
                q_row_idx,
                k_row_idx,
                q_length,
                k_length,
                mask_words,
                out_flat,
                lse_flat,
                softmax_scale=float(softmax_scale),
                tile_k=tile_k,
            )
            return
        else:
            if pack_buffers is None:
                pack_buffers = _get_cached_direct_2d_pack_buffers(payload, q_flat, k_flat, v_flat)
            q_buf_flat, k_buf_flat, v_buf_flat = pack_buffers
            q_buf_range = q_buf_flat[: grouped_count * packed_q]
            k_buf_range = k_buf_flat[: grouped_count * packed_k]
            v_buf_range = v_buf_flat[: grouped_count * packed_k]
            _run_synthetic_pack_rows_kernel(q_flat, q_row_idx_flat, q_buf_range)
            _run_synthetic_pack_kv_rows_kernel(k_flat, v_flat, k_row_idx_flat, k_buf_range, v_buf_range)
            q_buf = q_buf_range.view(grouped_count, packed_q, q_flat.shape[1], q_flat.shape[2])
            k_buf = k_buf_range.view(grouped_count, packed_k, k_flat.shape[1], k_flat.shape[2])
            v_buf = v_buf_range.view(grouped_count, packed_k, v_flat.shape[1], v_flat.shape[2])
            packed_out, packed_lse = _run_synthetic_2d_masked_fwd_kernel(
                q_buf,
                k_buf,
                v_buf,
                q_length,
                k_length,
                mask_words,
                softmax_scale=float(softmax_scale),
                tile_k=tile_k,
            )
        packed_out_flat = packed_out.view(grouped_count * packed_q, packed_out.shape[2], packed_out.shape[3]).contiguous()
        packed_lse_flat = packed_lse.view(grouped_count * packed_q, packed_lse.shape[2]).contiguous()
        _run_synthetic_combine_scatter_rows_kernel(
            packed_out_flat,
            packed_lse_flat,
            q_row_idx_flat,
            out_flat,
            lse_flat,
        )

    grouped_keys = (
        "range_tc_scatter_q_row_idx",
        "range_scatter_q_row_idx",
        "range_packed_q_row_idx",
    )
    if all(isinstance(payload.get(key), torch.Tensor) for key in grouped_keys):
        tc_q_row_idx = payload["range_tc_scatter_q_row_idx"]
        if int(tc_q_row_idx.shape[0]) > 0:
            _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel(
                q_flat,
                k_flat,
                v_flat,
                tc_q_row_idx,
                payload["range_tc_scatter_k_row_idx"],
                payload["range_tc_scatter_q_length"],
                payload["range_tc_scatter_k_length"],
                payload["range_tc_scatter_mask_words"],
                out_flat,
                lse_flat,
                softmax_scale=float(softmax_scale),
                tile_k=tile_k,
            )
            union_tc_group_count += int(payload.get("range_tc_scatter_group_count", int(tc_q_row_idx.shape[0])))
            union_tc_row_count += int(payload.get("range_tc_scatter_row_count", 0))
        scatter_q_row_idx = payload["range_scatter_q_row_idx"]
        if int(scatter_q_row_idx.shape[0]) > 0:
            _run_synthetic_2d_masked_gather_scatter_fwd_kernel(
                q_flat,
                k_flat,
                v_flat,
                scatter_q_row_idx,
                payload["range_scatter_k_row_idx"],
                payload["range_scatter_q_length"],
                payload["range_scatter_k_length"],
                payload["range_scatter_mask_words"],
                out_flat,
                lse_flat,
                softmax_scale=float(softmax_scale),
                tile_k=tile_k,
            )
            union_scalar_group_count += int(payload.get("range_scatter_union_group_count", 0))
            union_scalar_row_count += int(payload.get("range_scatter_union_row_count", 0))
        _run_grouped_packed(
            payload["range_packed_q_row_idx"],
            payload["range_packed_k_row_idx"],
            payload["range_packed_q_length"],
            payload["range_packed_k_length"],
            payload["range_packed_mask_words"],
            payload["range_packed_q_row_idx_flat"],
            payload["range_packed_k_row_idx_flat"],
        )
        return union_tc_group_count, union_tc_row_count, union_scalar_group_count, union_scalar_row_count

    for range_entry in range_execution:
        group_start = int(range_entry["group_start"])
        group_end = int(range_entry["group_end"])
        if group_end <= group_start:
            continue
        if group_end - group_start <= group_chunk_limit:
            _run_masked_group_range(group_start, group_end, range_entry)
            continue
        for chunk_start in range(group_start, group_end, group_chunk_limit):
            chunk_end = min(group_end, chunk_start + group_chunk_limit)
            _run_masked_group_range(chunk_start, chunk_end, range_entry)
    return union_tc_group_count, union_tc_row_count, union_scalar_group_count, union_scalar_row_count


def format_cached_public_lse(
    q: torch.Tensor,
    lse_final_flat: torch.Tensor,
) -> torch.Tensor:
    if q.ndim == 4:
        if (
            q.is_cuda
            and lse_final_flat.is_cuda
            and lse_final_flat.ndim == 2
            and lse_final_flat.is_contiguous()
            and int(lse_final_flat.shape[0]) == int(q.shape[0]) * int(q.shape[1])
            and int(lse_final_flat.shape[1]) == int(q.shape[2])
        ):
            public_lse = torch.empty(
                (q.shape[0], q.shape[2], q.shape[1]),
                device=lse_final_flat.device,
                dtype=lse_final_flat.dtype,
            )
            _run_cached_lse_flat_to_public_kernel(lse_final_flat, public_lse)
            return public_lse
        return lse_final_flat.view(q.shape[0], q.shape[1], q.shape[2]).permute(0, 2, 1).contiguous()
    return lse_final_flat.transpose(0, 1).contiguous()


def _format_cached_forward_result(
    q: torch.Tensor,
    out_final_flat: torch.Tensor,
    lse_final_flat: torch.Tensor,
    *,
    return_lse: bool,
    lse_layout: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if q.ndim == 4:
        out = out_final_flat.view(q.shape[0], q.shape[1], q.shape[2], out_final_flat.shape[2])
        if not return_lse:
            return out
        if str(lse_layout) == "flat":
            return out, lse_final_flat
        return out, format_cached_public_lse(q, lse_final_flat)
    if not return_lse:
        return out_final_flat
    if str(lse_layout) == "flat":
        return out_final_flat, lse_final_flat
    return out_final_flat, format_cached_public_lse(q, lse_final_flat)


def _cached_monolithic_forward_support_reason(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> str | None:
    if not q_flat.is_cuda:
        return "requires_cuda"
    if q_flat.dtype not in {torch.float16, torch.bfloat16}:
        return f"unsupported_q_dtype_{q_flat.dtype}"
    if k_flat.dtype != q_flat.dtype or v_flat.dtype != q_flat.dtype:
        return "mixed_qkv_dtype"
    if int(q_flat.shape[-1]) != 64 or int(k_flat.shape[-1]) != 64 or int(v_flat.shape[-1]) != 64:
        return "requires_head_dim_64"
    if str(payload.get("residual_mode", "")) != "fused_tail":
        return "requires_fused_tail_residual_mode"
    if str(payload.get("exact_kernel_family", "")) != "tc8x8":
        return "requires_tc8x8_exact_kernel"
    if int(payload.get("exact_dense_rows_per_range", 0)) != 8 or int(payload.get("exact_dense_keys_per_tile", 0)) != 8:
        return "requires_8x8_exact_tail_tiles"
    fused_q_row_idx = payload.get("fused_q_row_idx")
    if not isinstance(fused_q_row_idx, torch.Tensor) or int(fused_q_row_idx.shape[0]) <= 0:
        return "missing_fused_ranges"
    if int(getattr(payload.get("q_row_idx"), "shape", [0])[0]) != 0:
        return "residual_groups_present"
    if int(payload.get("fused_output_row_count", -1)) != int(payload["total_rows"]):
        return "incomplete_fused_output_row_coverage"
    geometry = payload.get("geometry")
    if isinstance(geometry, dict) and float(geometry.get("fused_total_coverage_frac", 0.0)) < 0.999999:
        return "incomplete_fused_pair_coverage"
    return None


def _cached_direct_final_residual_support_reason(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
) -> str | None:
    base_reason = _cached_monolithic_forward_support_reason(payload, q_flat, k_flat, v_flat)
    if base_reason is None:
        return None
    allowed_base_reasons = {
        "missing_fused_ranges",
        "residual_groups_present",
        "incomplete_fused_output_row_coverage",
        "incomplete_fused_pair_coverage",
    }
    if base_reason not in allowed_base_reasons:
        return base_reason
    if str(payload.get("residual_mode", "")) != "fused_tail":
        return "requires_fused_tail_residual_mode"
    base_output_row_count = int(payload.get("fused_output_row_count", 0))
    if base_output_row_count <= 0:
        base_output_row_count = int(payload.get("exact_dense_output_row_count", 0))
    residual_row_count = int(payload.get("range_tc_scatter_row_count", 0)) + int(
        payload.get("range_scatter_row_count", 0)
    )
    packed_group_count = int(payload.get("range_packed_group_count", 0))
    packed_tensor_count = int(getattr(payload.get("range_packed_q_row_idx"), "shape", [0])[0])
    if packed_group_count != 0 or packed_tensor_count != 0:
        grouped_keys = (
            "range_tc_scatter_q_row_idx",
            "range_scatter_q_row_idx",
            "range_packed_q_row_idx",
        )
        if not all(isinstance(payload.get(key), torch.Tensor) for key in grouped_keys):
            return "missing_grouped_residual_tensors"
        if base_output_row_count != int(payload["total_rows"]):
            device = _payload_row_device(payload, q_flat)
            union_row_count = _direct_final_base_residual_union_row_count(payload, device)
            if union_row_count != int(payload["total_rows"]):
                if residual_row_count != 0:
                    return "mixed_residual_incomplete_direct_final_row_coverage"
                return "packed_residual_incomplete_direct_final_row_coverage"
        return None
    if residual_row_count <= 0:
        return "missing_scatter_residual_rows"
    covered_rows = base_output_row_count + residual_row_count
    if covered_rows != int(payload["total_rows"]):
        return "incomplete_direct_final_row_coverage"
    grouped_keys = (
        "range_tc_scatter_q_row_idx",
        "range_scatter_q_row_idx",
        "range_packed_q_row_idx",
    )
    if not all(isinstance(payload.get(key), torch.Tensor) for key in grouped_keys):
        return "missing_grouped_residual_tensors"
    return None


def _run_cached_monolithic_fused_tail_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    *,
    softmax_scale: float,
    return_lse: bool,
    lse_layout: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    env_name = "FLASH_ATTN_HSA_CACHED_MONOLITHIC_FWD"
    if not _is_env_enabled(env_name):
        return None
    support_reason = _cached_monolithic_forward_support_reason(payload, q_flat, k_flat, v_flat)
    if support_reason is not None:
        if _is_env_forced_on(env_name):
            direct_final_reason = _cached_direct_final_residual_support_reason(payload, q_flat, k_flat, v_flat)
            if direct_final_reason is not None:
                raise RuntimeError(f"cached_monolithic_fwd_unsupported_{support_reason}")
        return None
    out_final_flat, lse_final_flat = _get_cached_direct_2d_final_buffers(payload, q_flat, v_flat)
    fused_range_count, fused_row_count = _run_cached_fused_exact_tail_ranges(
        payload,
        q_flat,
        k_flat,
        v_flat,
        out_final_flat,
        lse_final_flat,
        softmax_scale=float(softmax_scale),
    )
    if fused_range_count <= 0 or fused_row_count != int(payload["total_rows"]):
        if _is_env_forced_on(env_name):
            raise RuntimeError("cached_monolithic_fwd_failed_incomplete_runtime_coverage")
        return None
    _record_fused_runtime_geometry(payload, fused_range_count=fused_range_count, fused_row_count=fused_row_count)
    _record_exact_dense_runtime_geometry(payload, exact_range_count=0, exact_row_count=0)
    _record_union_runtime_geometry(payload, tc_group_count=0, tc_row_count=0, scalar_group_count=0, scalar_row_count=0)
    _record_cached_forward_path(payload, path="monolithic_fused_tail")
    return _format_cached_forward_result(q, out_final_flat, lse_final_flat, return_lse=return_lse, lse_layout=lse_layout)


def _run_cached_direct_final_residual_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    *,
    softmax_scale: float,
    return_lse: bool,
    lse_layout: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
    env_name = "FLASH_ATTN_HSA_CACHED_MONOLITHIC_FWD"
    if not _is_env_enabled(env_name):
        return None
    support_reason = _cached_direct_final_residual_support_reason(payload, q_flat, k_flat, v_flat)
    if support_reason is not None:
        if _is_env_forced_on(env_name):
            raise RuntimeError(f"cached_direct_final_residual_fwd_unsupported_{support_reason}")
        _record_cached_forward_path(payload, path="split_fallback", reason=support_reason)
        return None

    out_final_flat, lse_final_flat = _get_cached_direct_2d_final_buffers(payload, q_flat, v_flat)
    fused_range_count = 0
    fused_row_count = 0
    exact_range_count = 0
    exact_row_count = 0
    if int(getattr(payload.get("fused_q_row_idx"), "shape", [0])[0]) > 0:
        fused_range_count, fused_row_count = _run_cached_fused_exact_tail_ranges(
            payload,
            q_flat,
            k_flat,
            v_flat,
            out_final_flat,
            lse_final_flat,
            softmax_scale=float(softmax_scale),
        )
    if fused_range_count <= 0 and int(getattr(payload.get("exact_dense_q_row_idx"), "shape", [0])[0]) > 0:
        exact_range_count, exact_row_count = _run_cached_exact_dense_ranges(
            payload,
            q_flat,
            k_flat,
            v_flat,
            out_final_flat,
            lse_final_flat,
            softmax_scale=float(softmax_scale),
        )
    base_row_count = int(fused_row_count) if int(fused_row_count) > 0 else int(exact_row_count)
    residual_row_count = int(payload.get("range_tc_scatter_row_count", 0)) + int(
        payload.get("range_scatter_row_count", 0)
    )
    packed_group_count = int(payload.get("range_packed_group_count", 0))
    packed_tensor_count = int(getattr(payload.get("range_packed_q_row_idx"), "shape", [0])[0])
    force_combine_scatter = (
        residual_row_count > 0
        and (packed_group_count > 0 or packed_tensor_count > 0)
    )
    has_packed_residual = packed_group_count > 0 or packed_tensor_count > 0
    base_source = "fused" if int(fused_row_count) > 0 else "exact_dense"
    initialized_residual_rows = 0
    if has_packed_residual and base_row_count != int(payload["total_rows"]):
        union_row_count = _direct_final_base_residual_union_row_count(
            payload,
            out_final_flat.device,
            base_source=base_source,
        )
        if union_row_count != int(payload["total_rows"]):
            reason = (
                "mixed_residual_incomplete_direct_final_runtime_coverage"
                if force_combine_scatter
                else "packed_residual_incomplete_direct_final_runtime_coverage"
            )
            if _is_env_forced_on(env_name):
                raise RuntimeError(f"cached_direct_final_residual_fwd_{reason}")
            _record_cached_forward_path(payload, path="split_fallback", reason=reason)
            return None
        init_row_idx = _get_direct_final_missing_init_row_idx(
            payload,
            out_final_flat.device,
            base_source=base_source,
        )
        initialized_residual_rows = int(init_row_idx.numel())
        if initialized_residual_rows > 0:
            _run_cached_init_output_rows_kernel(init_row_idx, out_final_flat, lse_final_flat)
    union_tc_group_count, union_tc_row_count, union_scalar_group_count, union_scalar_row_count = (
        _run_cached_masked_payload_forward(
            payload,
            q_flat,
            k_flat,
            v_flat,
            out_final_flat,
            lse_final_flat,
            softmax_scale=float(softmax_scale),
            force_combine_scatter=force_combine_scatter,
        )
    )
    output_coverage = (
        int(payload["total_rows"])
        if has_packed_residual and base_row_count != int(payload["total_rows"])
        else base_row_count
        if force_combine_scatter
        else base_row_count + residual_row_count
    )
    if output_coverage != int(payload["total_rows"]):
        if _is_env_forced_on(env_name):
            raise RuntimeError("cached_direct_final_residual_fwd_incomplete_runtime_coverage")
        _record_cached_forward_path(payload, path="split_fallback", reason="incomplete_runtime_coverage")
        return None
    _record_union_runtime_geometry(
        payload,
        tc_group_count=union_tc_group_count,
        tc_row_count=union_tc_row_count,
        scalar_group_count=union_scalar_group_count,
        scalar_row_count=union_scalar_row_count,
    )
    _record_exact_dense_runtime_geometry(payload, exact_range_count=exact_range_count, exact_row_count=exact_row_count)
    _record_fused_runtime_geometry(payload, fused_range_count=fused_range_count, fused_row_count=fused_row_count)
    _record_cached_forward_path(
        payload,
        path="direct_final_residual",
        initialized_residual_rows=initialized_residual_rows,
    )
    return _format_cached_forward_result(q, out_final_flat, lse_final_flat, return_lse=return_lse, lse_layout=lse_layout)


def run_cached_direct_2d_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    lse_layout: str = "public",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    has_exact_dense = int(getattr(payload.get("exact_dense_q_row_idx"), "shape", [0])[0]) > 0
    has_fused_ranges = int(getattr(payload.get("fused_q_row_idx"), "shape", [0])[0]) > 0
    if not has_exact_dense and not has_fused_ranges and int(payload["q_row_idx"].shape[0]) <= 0:
        raise RuntimeError("cached_direct_2d_forward_empty")

    monolithic_result = _run_cached_monolithic_fused_tail_forward(
        payload,
        q,
        q_flat,
        k_flat,
        v_flat,
        softmax_scale=float(softmax_scale),
        return_lse=return_lse,
        lse_layout=lse_layout,
    )
    if monolithic_result is not None:
        return monolithic_result

    direct_final_residual_result = _run_cached_direct_final_residual_forward(
        payload,
        q,
        q_flat,
        k_flat,
        v_flat,
        softmax_scale=float(softmax_scale),
        return_lse=return_lse,
        lse_layout=lse_layout,
    )
    if direct_final_residual_result is not None:
        return direct_final_residual_result

    out_flat, lse_flat = _get_cached_direct_2d_output_buffers(payload, q_flat, v_flat)
    all_row_idx = _get_cached_all_row_idx(payload, q_flat.device)
    _run_cached_init_output_rows_kernel(all_row_idx, out_flat, lse_flat)
    fused_range_count, fused_row_count = _run_cached_fused_exact_tail_ranges(
        payload,
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        lse_flat,
        softmax_scale=float(softmax_scale),
    )
    exact_range_count = 0
    exact_row_count = 0
    if fused_range_count <= 0:
        exact_range_count, exact_row_count = _run_cached_exact_dense_ranges(
            payload,
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            lse_flat,
            softmax_scale=float(softmax_scale),
        )
    union_tc_group_count, union_tc_row_count, union_scalar_group_count, union_scalar_row_count = (
        _run_cached_masked_payload_forward(
            payload,
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            lse_flat,
            softmax_scale=float(softmax_scale),
        )
    )
    _record_union_runtime_geometry(
        payload,
        tc_group_count=union_tc_group_count,
        tc_row_count=union_tc_row_count,
        scalar_group_count=union_scalar_group_count,
        scalar_row_count=union_scalar_row_count,
    )
    _record_exact_dense_runtime_geometry(
        payload,
        exact_range_count=exact_range_count,
        exact_row_count=exact_row_count,
    )
    _record_fused_runtime_geometry(
        payload,
        fused_range_count=fused_range_count,
        fused_row_count=fused_row_count,
    )
    _record_cached_forward_path(payload, path="split_fallback")
    out_final_flat, lse_final_flat = _get_cached_direct_2d_final_buffers(payload, q_flat, v_flat)
    _run_cached_finalize_output_rows_kernel(out_flat, lse_flat, all_row_idx, out_final_flat, lse_final_flat)
    return _format_cached_forward_result(q, out_final_flat, lse_final_flat, return_lse=return_lse, lse_layout=lse_layout)


def run_cached_generalized_packed_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    return_lse: bool = False,
    lse_layout: str = "public",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return run_cached_direct_2d_forward(
        payload,
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        lse_layout=lse_layout,
    )


def _expand_kv_to_q_heads_local(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    if int(x.shape[2]) == int(num_q_heads):
        return x
    repeat_factor = int(num_q_heads) // int(x.shape[2])
    return x.repeat_interleave(repeat_factor, dim=2)


def _collapse_q_to_kv_heads_local(x: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    if int(x.shape[2]) == int(num_kv_heads):
        return x
    repeat_factor = int(x.shape[2]) // int(num_kv_heads)
    return x.view(x.shape[0], x.shape[1], num_kv_heads, repeat_factor, x.shape[3]).sum(dim=3)


def can_use_cached_generalized_fused_backward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    deterministic: bool = False,
) -> bool:
    if not _is_env_enabled("FLASH_ATTN_HSA_CACHED_MONOLITHIC_BWD"):
        return False
    if payload.get("status") != "ready" or deterministic:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.shape[-1] != 64 or k.shape[-1] != 64:
        return False
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if str(payload.get("exact_kernel_family", "")) != "tc8x8":
        return False
    residual_mode = str(payload.get("residual_mode", ""))
    backward_payload = payload.get("cached_generalized_backward_payload")
    if residual_mode == "masked_union":
        if not q.is_cuda:
            return False
        if (
            not isinstance(backward_payload, dict)
            or backward_payload.get("status") != "ready"
            or str(backward_payload.get("backward_kernel_family", "")) != "cached_masked_union_row_compact"
        ):
            backward_payload = build_cached_generalized_backward_payload(payload)
            if (
                not isinstance(backward_payload, dict)
                or backward_payload.get("status") != "ready"
                or str(backward_payload.get("backward_kernel_family", "")) != "cached_masked_union_row_compact"
            ):
                return False
            payload["cached_generalized_backward_payload"] = backward_payload
        q_row_idx = backward_payload.get("q_row_idx")
        union_k_row_idx = backward_payload.get("union_k_row_idx")
        unique_key_occurrence_row_ptr = backward_payload.get("unique_key_occurrence_row_ptr")
        if not all(
            isinstance(tensor, torch.Tensor)
            for tensor in (q_row_idx, union_k_row_idx, unique_key_occurrence_row_ptr)
        ):
            return False
        if int(q_row_idx.shape[0]) <= 0 or int(union_k_row_idx.shape[1]) > 16:
            return False
        return True
    if residual_mode != "fused_tail":
        return False
    if int(payload.get("exact_dense_rows_per_range", 0)) != 8:
        return False
    if int(payload.get("exact_dense_keys_per_tile", 0)) != 8:
        return False
    fused_q_row_idx = payload.get("fused_q_row_idx")
    fused_exact_tile_ptr = payload.get("fused_exact_tile_ptr")
    fused_tail_tile_ptr = payload.get("fused_tail_tile_ptr")
    if not all(isinstance(tensor, torch.Tensor) for tensor in (fused_q_row_idx, fused_exact_tile_ptr, fused_tail_tile_ptr)):
        return False
    if int(fused_q_row_idx.shape[0]) <= 0:
        return False
    geometry = payload.get("geometry")
    if not isinstance(geometry, dict):
        return False
    min_coverage = float(
        os.environ.get(
            "FLASH_ATTN_HSA_CACHED_GENERALIZED_FUSED_BWD_MIN_COVERAGE",
            "0.999",
        )
    )
    if float(geometry.get("fused_total_coverage_frac", 0.0)) < min_coverage:
        return False
    if int(geometry.get("legacy_residual_fallback_range_count", 0)) != 0:
        return False
    if q.is_cuda:
        use_tile_atomic_dkdv = os.environ.get(
            "FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_TILE_ATOMICS",
            "1",
        ).strip().lower() not in {"0", "false", "off", "no"}
        if not use_tile_atomic_dkdv and (
            not isinstance(backward_payload, dict) or backward_payload.get("status") != "ready"
        ):
            backward_payload = build_cached_generalized_backward_payload(payload)
            if not isinstance(backward_payload, dict) or backward_payload.get("status") != "ready":
                return False
            payload["cached_generalized_backward_payload"] = backward_payload
        if isinstance(backward_payload, dict):
            if str(backward_payload.get("backward_kernel_family", "")) != "cached_tc8x8_fused":
                return False
            if int(backward_payload.get("rows_per_range", 0)) != 8:
                return False
            if int(backward_payload.get("keys_per_tile", 0)) != 8:
                return False
    return True


def _get_cached_backward_workspace(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = payload.setdefault("_workspace", {})
    if not isinstance(workspace, dict):
        workspace = {}
        payload["_workspace"] = workspace
    return workspace


def _get_tile_range_index(
    payload: dict[str, Any],
    *,
    ptr_key: str,
) -> torch.Tensor:
    workspace = _get_cached_backward_workspace(payload)
    cache_key = f"backward_{ptr_key}_range_idx"
    cached = workspace.get(cache_key)
    ptr = payload[ptr_key]
    if isinstance(cached, torch.Tensor) and cached.device == ptr.device:
        return cached
    tile_counts = (ptr[1:] - ptr[:-1]).to(dtype=torch.long)
    if int(tile_counts.numel()) <= 0 or int(tile_counts.sum().item()) <= 0:
        cached = torch.empty((0,), dtype=torch.long, device=ptr.device)
    else:
        cached = torch.repeat_interleave(
            torch.arange(int(tile_counts.shape[0]), device=ptr.device, dtype=torch.long),
            tile_counts,
        ).contiguous()
    workspace[cache_key] = cached
    return cached


def _decode_tail_mask_words_chunk(mask_words: torch.Tensor, width: int) -> torch.Tensor:
    if int(mask_words.numel()) <= 0:
        return torch.zeros((int(mask_words.shape[0]), int(mask_words.shape[1]), width), dtype=torch.bool, device=mask_words.device)
    flat_mask = mask_words.reshape(-1, mask_words.shape[-1]).contiguous()
    decoded = _decode_mask_words_to_bool(flat_mask, width)
    return decoded.view(int(mask_words.shape[0]), int(mask_words.shape[1]), width).contiguous()


def _build_cached_generalized_pair_tile_backward_payload(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = _get_cached_backward_workspace(payload)
    fused_q_row_idx = payload["fused_q_row_idx"]
    cache_key = ("pair_tile_backward_payload", str(fused_q_row_idx.device))
    cached = workspace.get(cache_key)
    if isinstance(cached, dict):
        cached_q_row_idx = cached.get("q_row_idx")
        if isinstance(cached_q_row_idx, torch.Tensor) and cached_q_row_idx.device == fused_q_row_idx.device:
            return cached

    device = fused_q_row_idx.device
    q_row_idx_cpu = fused_q_row_idx.detach().to("cpu").tolist()
    q_length_cpu = payload["fused_q_length"].detach().to("cpu").tolist()
    exact_tile_ptr_cpu = payload["fused_exact_tile_ptr"].detach().to("cpu").tolist()
    exact_k_row_idx_cpu = payload["fused_exact_k_row_idx"].detach().to("cpu").tolist()
    tail_tile_ptr_cpu = payload["fused_tail_tile_ptr"].detach().to("cpu").tolist()
    tail_k_row_idx_cpu = payload["fused_tail_k_row_idx"].detach().to("cpu").tolist()
    tail_mask_words_cpu = payload["fused_tail_mask_words"].detach().to("cpu").tolist()

    member_q_row_idx: list[list[int]] = []
    member_q_length: list[int] = []
    member_row_k_row_idx: list[list[list[int]]] = []
    member_row_k_to_union_idx: list[list[list[int]]] = []
    member_union_k_row_idx: list[list[int]] = []
    member_union_to_row_slot: list[list[list[int]]] = []
    member_row_k_length: list[list[int]] = []
    member_union_k_length: list[int] = []
    unique_key_occurrences: dict[int, list[tuple[int, int]]] = {}

    def append_member(
        pair_q_rows: list[int],
        union_rows: list[int],
        row_support_lists: list[list[int]],
    ) -> None:
        if not pair_q_rows or not union_rows:
            return
        member_idx = len(member_q_row_idx)
        q_pair = [int(value) for value in pair_q_rows[:2]]
        q_pair.extend([-1] * (2 - len(q_pair)))
        member_q_row_idx.append(q_pair)
        member_q_length.append(min(2, len(pair_q_rows)))

        union_rows = [int(value) for value in union_rows[:8] if int(value) >= 0]
        union_index = {int(key_row): idx for idx, key_row in enumerate(union_rows)}
        padded_union_rows = union_rows + [-1] * (8 - len(union_rows))
        member_union_k_row_idx.append(padded_union_rows)
        member_union_k_length.append(len(union_rows))

        row_k_rows_entry: list[list[int]] = []
        row_k_to_union_entry: list[list[int]] = []
        union_to_row_entry: list[list[int]] = []
        row_k_length_entry: list[int] = []
        for row_slot in range(2):
            support_rows = [int(value) for value in row_support_lists[row_slot]] if row_slot < len(row_support_lists) else []
            support_rows = [value for value in support_rows if value in union_index]
            row_k_length_entry.append(len(support_rows))
            row_k_rows_entry.append(support_rows + [-1] * (8 - len(support_rows)))
            row_k_to_union_entry.append([union_index[value] for value in support_rows] + [-1] * (8 - len(support_rows)))
            union_to_row = [-1] * 8
            for row_local_slot, key_row in enumerate(support_rows):
                union_to_row[union_index[key_row]] = row_local_slot
            union_to_row_entry.append(union_to_row)
        member_row_k_row_idx.append(row_k_rows_entry)
        member_row_k_to_union_idx.append(row_k_to_union_entry)
        member_union_to_row_slot.append(union_to_row_entry)
        member_row_k_length.append(row_k_length_entry)

        for union_idx, key_row in enumerate(union_rows):
            unique_key_occurrences.setdefault(int(key_row), []).append((member_idx, union_idx))

    for range_idx, q_length_value in enumerate(q_length_cpu):
        q_length_value = int(q_length_value)
        if q_length_value <= 0:
            continue
        range_q_rows = [int(value) for value in q_row_idx_cpu[range_idx][:q_length_value] if int(value) >= 0]
        if not range_q_rows:
            continue
        exact_start = int(exact_tile_ptr_cpu[range_idx])
        exact_end = int(exact_tile_ptr_cpu[range_idx + 1])
        for tile_idx in range(exact_start, exact_end):
            union_rows = [int(value) for value in exact_k_row_idx_cpu[tile_idx] if int(value) >= 0]
            if not union_rows:
                continue
            for pair_start in range(0, len(range_q_rows), 2):
                pair_q_rows = range_q_rows[pair_start : pair_start + 2]
                row_support_lists = [list(union_rows) for _ in pair_q_rows]
                append_member(pair_q_rows, union_rows, row_support_lists)

        tail_start = int(tail_tile_ptr_cpu[range_idx])
        tail_end = int(tail_tile_ptr_cpu[range_idx + 1])
        for tile_idx in range(tail_start, tail_end):
            tile_rows = [int(value) for value in tail_k_row_idx_cpu[tile_idx] if int(value) >= 0]
            if not tile_rows:
                continue
            row_masks = [int(mask_words[0]) for mask_words in tail_mask_words_cpu[tile_idx][: len(range_q_rows)]]
            for pair_start in range(0, len(range_q_rows), 2):
                pair_q_rows = range_q_rows[pair_start : pair_start + 2]
                pair_masks = row_masks[pair_start : pair_start + len(pair_q_rows)]
                active_cols = [
                    col_idx
                    for col_idx, key_row in enumerate(tile_rows)
                    if any(((int(row_mask) >> col_idx) & 1) for row_mask in pair_masks) and int(key_row) >= 0
                ]
                if not active_cols:
                    continue
                union_rows = [tile_rows[col_idx] for col_idx in active_cols]
                row_support_lists = []
                for local_row_idx, row_mask in enumerate(pair_masks):
                    del local_row_idx
                    row_support_lists.append(
                        [
                            tile_rows[col_idx]
                            for col_idx in active_cols
                            if ((int(row_mask) >> col_idx) & 1) and int(tile_rows[col_idx]) >= 0
                        ]
                    )
                append_member(pair_q_rows, union_rows, row_support_lists)

    if member_q_row_idx:
        q_row_idx = torch.tensor(member_q_row_idx, dtype=torch.int32, device=device)
        q_length = torch.tensor(member_q_length, dtype=torch.int32, device=device)
        row_k_row_idx = torch.tensor(member_row_k_row_idx, dtype=torch.int32, device=device)
        row_k_to_union_idx = torch.tensor(member_row_k_to_union_idx, dtype=torch.int32, device=device)
        union_k_row_idx = torch.tensor(member_union_k_row_idx, dtype=torch.int32, device=device)
        union_to_row_slot = torch.tensor(member_union_to_row_slot, dtype=torch.int32, device=device)
        row_k_length = torch.tensor(member_row_k_length, dtype=torch.int32, device=device)
        union_k_length = torch.tensor(member_union_k_length, dtype=torch.int32, device=device)
    else:
        q_row_idx = torch.empty((0, 2), dtype=torch.int32, device=device)
        q_length = torch.empty((0,), dtype=torch.int32, device=device)
        row_k_row_idx = torch.empty((0, 2, 8), dtype=torch.int32, device=device)
        row_k_to_union_idx = torch.empty((0, 2, 8), dtype=torch.int32, device=device)
        union_k_row_idx = torch.empty((0, 8), dtype=torch.int32, device=device)
        union_to_row_slot = torch.empty((0, 2, 8), dtype=torch.int32, device=device)
        row_k_length = torch.empty((0, 2), dtype=torch.int32, device=device)
        union_k_length = torch.empty((0,), dtype=torch.int32, device=device)

    unique_key_row_idx_list: list[int] = []
    unique_key_member_idx_list: list[int] = []
    unique_key_union_idx_list: list[int] = []
    unique_key_occurrence_row_ptr_list = [0]
    max_unique_key_occurrences = 0
    for key_row in sorted(unique_key_occurrences):
        occurrences = unique_key_occurrences[key_row]
        max_unique_key_occurrences = max(max_unique_key_occurrences, len(occurrences))
        unique_key_row_idx_list.append(int(key_row))
        for member_idx, union_idx in occurrences:
            unique_key_member_idx_list.append(int(member_idx))
            unique_key_union_idx_list.append(int(union_idx))
        unique_key_occurrence_row_ptr_list.append(len(unique_key_member_idx_list))

    built = {
        "q_row_idx": q_row_idx.contiguous(),
        "q_length": q_length.contiguous(),
        "row_k_row_idx": row_k_row_idx.contiguous(),
        "row_k_to_union_idx": row_k_to_union_idx.contiguous(),
        "union_k_row_idx": union_k_row_idx.contiguous(),
        "union_to_row_slot": union_to_row_slot.contiguous(),
        "row_k_length": row_k_length.contiguous(),
        "union_k_length": union_k_length.contiguous(),
        "unique_key_row_idx": torch.tensor(unique_key_row_idx_list, dtype=torch.int32, device=device).contiguous(),
        "unique_key_member_idx": torch.tensor(unique_key_member_idx_list, dtype=torch.int32, device=device).contiguous(),
        "unique_key_union_idx": torch.tensor(unique_key_union_idx_list, dtype=torch.int32, device=device).contiguous(),
        "unique_key_occurrence_row_ptr": torch.tensor(
            unique_key_occurrence_row_ptr_list,
            dtype=torch.int32,
            device=device,
        ).contiguous(),
        "max_unique_key_occurrences": int(max_unique_key_occurrences),
    }
    workspace[cache_key] = built
    return built


def _accumulate_cached_generalized_tile_backward(
    *,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    dq_acc: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
    q_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    tile_k_row_idx: torch.Tensor,
    tile_range_idx: torch.Tensor,
    softmax_scale: float,
    tail_mask_words: torch.Tensor | None = None,
    chunk_tiles: int = 2048,
) -> None:
    total_tiles = int(tile_k_row_idx.shape[0])
    if total_tiles <= 0:
        return
    device = q_flat.device
    num_q_heads = int(q_flat.shape[1])
    num_kv_heads = int(k_flat.shape[1])
    q_slots = int(q_row_idx.shape[1])
    k_slots = int(tile_k_row_idx.shape[1])
    q_offsets = torch.arange(q_slots, device=device, dtype=torch.int32).view(1, q_slots)
    for tile_start in range(0, total_tiles, chunk_tiles):
        tile_end = min(total_tiles, tile_start + chunk_tiles)
        chunk_range_idx = tile_range_idx[tile_start:tile_end]
        chunk_q_row_idx = q_row_idx.index_select(0, chunk_range_idx).contiguous()
        chunk_q_length = q_length.index_select(0, chunk_range_idx).contiguous()
        q_valid = q_offsets < chunk_q_length.unsqueeze(1)
        q_index = chunk_q_row_idx.clamp_min(0).to(dtype=torch.long)
        q_sel = q_flat.index_select(0, q_index.reshape(-1)).view(tile_end - tile_start, q_slots, num_q_heads, q_flat.shape[-1]).contiguous()
        out_sel = out_flat.index_select(0, q_index.reshape(-1)).view(tile_end - tile_start, q_slots, num_q_heads, out_flat.shape[-1]).contiguous()
        dout_sel = dout_flat.index_select(0, q_index.reshape(-1)).view(tile_end - tile_start, q_slots, num_q_heads, dout_flat.shape[-1]).contiguous()
        lse_sel = lse_flat.index_select(0, q_index.reshape(-1)).view(tile_end - tile_start, q_slots, num_q_heads).contiguous()

        chunk_k_row_idx = tile_k_row_idx[tile_start:tile_end].contiguous()
        k_valid = chunk_k_row_idx >= 0
        k_index = chunk_k_row_idx.clamp_min(0).to(dtype=torch.long)
        k_sel = k_flat.index_select(0, k_index.reshape(-1)).view(tile_end - tile_start, k_slots, num_kv_heads, k_flat.shape[-1]).contiguous()
        v_sel = v_flat.index_select(0, k_index.reshape(-1)).view(tile_end - tile_start, k_slots, num_kv_heads, v_flat.shape[-1]).contiguous()
        k_q_heads = _expand_kv_to_q_heads_local(k_sel, num_q_heads).permute(0, 2, 1, 3).contiguous()
        v_q_heads = _expand_kv_to_q_heads_local(v_sel, num_q_heads).permute(0, 2, 1, 3).contiguous()

        q_heads = q_sel.permute(0, 2, 1, 3).contiguous()
        out_heads = out_sel.permute(0, 2, 1, 3).contiguous()
        dout_heads = dout_sel.permute(0, 2, 1, 3).contiguous()
        lse_heads = lse_sel.permute(0, 2, 1).unsqueeze(-1).float()

        scores = torch.matmul(q_heads, k_q_heads.transpose(-1, -2)).float()
        scores.mul_(float(softmax_scale))
        pair_mask = torch.logical_and(q_valid.unsqueeze(-1), k_valid.unsqueeze(1))
        if tail_mask_words is not None:
            tail_mask = _decode_tail_mask_words_chunk(tail_mask_words[tile_start:tile_end], k_slots)
            pair_mask = torch.logical_and(pair_mask, tail_mask)
        scores.masked_fill_(~pair_mask.unsqueeze(1), float("-inf"))
        probs = torch.exp(scores - lse_heads)
        probs.mul_(pair_mask.unsqueeze(1))
        dprob = torch.matmul(dout_heads, v_q_heads.transpose(-1, -2)).float()
        delta = (out_sel.float() * dout_sel.float()).sum(dim=-1).permute(0, 2, 1).unsqueeze(-1)
        dscores = probs * (dprob - delta)

        dq_chunk = torch.matmul(dscores.to(dtype=q_heads.dtype), k_q_heads).float()
        dq_chunk.mul_(float(softmax_scale))
        dq_chunk = dq_chunk.permute(0, 2, 1, 3).contiguous()
        dq_chunk.mul_(q_valid.unsqueeze(-1).unsqueeze(-1))

        dk_chunk = torch.matmul(dscores.transpose(-1, -2).to(dtype=q_heads.dtype), q_heads).float()
        dk_chunk.mul_(float(softmax_scale))
        dk_chunk = _collapse_q_to_kv_heads_local(
            dk_chunk.permute(0, 2, 1, 3).contiguous(),
            num_kv_heads,
        )
        dk_chunk.mul_(k_valid.unsqueeze(-1).unsqueeze(-1))

        dv_chunk = torch.matmul(probs.transpose(-1, -2).to(dtype=q_heads.dtype), dout_heads).float()
        dv_chunk = _collapse_q_to_kv_heads_local(
            dv_chunk.permute(0, 2, 1, 3).contiguous(),
            num_kv_heads,
        )
        dv_chunk.mul_(k_valid.unsqueeze(-1).unsqueeze(-1))

        dq_acc.index_add_(0, q_index.reshape(-1), dq_chunk.reshape(-1, num_q_heads, q_flat.shape[-1]))
        dk_acc.index_add_(0, k_index.reshape(-1), dk_chunk.reshape(-1, num_kv_heads, k_flat.shape[-1]))
        dv_acc.index_add_(0, k_index.reshape(-1), dv_chunk.reshape(-1, num_kv_heads, v_flat.shape[-1]))


def _materialize_range_tile_chunk(
    tile_ptr: torch.Tensor,
    tile_k_row_idx: torch.Tensor,
    *,
    range_start: int,
    range_end: int,
    tail_mask_words: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    chunk_range_count = max(0, int(range_end) - int(range_start))
    counts = (tile_ptr[range_start + 1 : range_end + 1] - tile_ptr[range_start:range_end]).to(dtype=torch.long)
    if chunk_range_count <= 0 or int(counts.numel()) <= 0:
        empty_rows = torch.empty((0, 0, int(tile_k_row_idx.shape[-1])), dtype=tile_k_row_idx.dtype, device=tile_k_row_idx.device)
        empty_masks = None
        if tail_mask_words is not None:
            empty_masks = torch.empty(
                (0, 0, int(tail_mask_words.shape[-2]), int(tail_mask_words.shape[-1])),
                dtype=tail_mask_words.dtype,
                device=tail_mask_words.device,
            )
        return counts, empty_rows, empty_masks
    max_tiles = int(counts.max().item()) if int(counts.numel()) > 0 else 0
    if max_tiles <= 0:
        empty_rows = torch.empty((chunk_range_count, 0, int(tile_k_row_idx.shape[-1])), dtype=tile_k_row_idx.dtype, device=tile_k_row_idx.device)
        empty_masks = None
        if tail_mask_words is not None:
            empty_masks = torch.empty(
                (chunk_range_count, 0, int(tail_mask_words.shape[-2]), int(tail_mask_words.shape[-1])),
                dtype=tail_mask_words.dtype,
                device=tail_mask_words.device,
            )
        return counts, empty_rows, empty_masks

    padded_k_rows = torch.full(
        (chunk_range_count, max_tiles, int(tile_k_row_idx.shape[-1])),
        -1,
        dtype=tile_k_row_idx.dtype,
        device=tile_k_row_idx.device,
    )
    padded_tail_masks = None
    if tail_mask_words is not None:
        padded_tail_masks = torch.zeros(
            (chunk_range_count, max_tiles, int(tail_mask_words.shape[-2]), int(tail_mask_words.shape[-1])),
            dtype=tail_mask_words.dtype,
            device=tail_mask_words.device,
        )
    for local_range_idx in range(chunk_range_count):
        tile_start = int(tile_ptr[range_start + local_range_idx].item())
        tile_end = int(tile_ptr[range_start + local_range_idx + 1].item())
        if tile_end <= tile_start:
            continue
        tile_count = tile_end - tile_start
        padded_k_rows[local_range_idx, :tile_count].copy_(tile_k_row_idx[tile_start:tile_end])
        if padded_tail_masks is not None:
            padded_tail_masks[local_range_idx, :tile_count].copy_(tail_mask_words[tile_start:tile_end])
    return counts, padded_k_rows, padded_tail_masks


def _accumulate_cached_generalized_range_tiles(
    *,
    q_heads: torch.Tensor,
    out_sel: torch.Tensor,
    dout_sel: torch.Tensor,
    lse_heads: torch.Tensor,
    q_valid: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
    q_index: torch.Tensor,
    dq_acc: torch.Tensor,
    softmax_scale: float,
    tile_counts: torch.Tensor,
    padded_k_rows: torch.Tensor,
    tail_mask_words: torch.Tensor | None = None,
) -> None:
    chunk_range_count = int(q_heads.shape[0])
    if chunk_range_count <= 0 or int(padded_k_rows.shape[1]) <= 0:
        return
    num_q_heads = int(q_heads.shape[1])
    num_kv_heads = int(k_flat.shape[1])
    q_slots = int(q_heads.shape[2])
    k_slots = int(padded_k_rows.shape[-1])
    max_tiles = int(padded_k_rows.shape[1])
    tile_active = torch.arange(max_tiles, device=q_heads.device, dtype=torch.long).view(1, max_tiles) < tile_counts.unsqueeze(1)
    k_valid = padded_k_rows >= 0
    k_index = padded_k_rows.clamp_min(0).to(dtype=torch.long)
    k_sel = k_flat.index_select(0, k_index.reshape(-1)).view(
        chunk_range_count,
        max_tiles,
        k_slots,
        num_kv_heads,
        k_flat.shape[-1],
    ).contiguous()
    v_sel = v_flat.index_select(0, k_index.reshape(-1)).view(
        chunk_range_count,
        max_tiles,
        k_slots,
        num_kv_heads,
        v_flat.shape[-1],
    ).contiguous()
    k_q_heads = _expand_kv_to_q_heads_local(
        k_sel.view(chunk_range_count * max_tiles, k_slots, num_kv_heads, k_flat.shape[-1]),
        num_q_heads,
    ).view(chunk_range_count, max_tiles, k_slots, num_q_heads, k_flat.shape[-1]).permute(0, 1, 3, 2, 4).contiguous()
    v_q_heads = _expand_kv_to_q_heads_local(
        v_sel.view(chunk_range_count * max_tiles, k_slots, num_kv_heads, v_flat.shape[-1]),
        num_q_heads,
    ).view(chunk_range_count, max_tiles, k_slots, num_q_heads, v_flat.shape[-1]).permute(0, 1, 3, 2, 4).contiguous()

    q_heads_t = q_heads.unsqueeze(1)
    dout_heads = dout_sel.permute(0, 2, 1, 3).contiguous().unsqueeze(1)
    scores = torch.matmul(q_heads_t, k_q_heads.transpose(-1, -2)).float()
    scores.mul_(float(softmax_scale))
    pair_mask = torch.logical_and(
        q_valid[:, None, None, :, None],
        torch.logical_and(tile_active[:, :, None, None, None], k_valid[:, :, None, None, :]),
    )
    if tail_mask_words is not None:
        tail_mask = _decode_tail_mask_words_chunk(
            tail_mask_words.view(chunk_range_count * max_tiles, q_slots, tail_mask_words.shape[-1]).contiguous(),
            k_slots,
        ).view(chunk_range_count, max_tiles, q_slots, k_slots)
        pair_mask = torch.logical_and(pair_mask, tail_mask[:, :, None, :, :])
    scores.masked_fill_(~pair_mask, float("-inf"))
    probs = torch.exp(scores - lse_heads.unsqueeze(1))
    probs.mul_(pair_mask)
    dprob = torch.matmul(dout_heads, v_q_heads.transpose(-1, -2)).float()
    delta = (out_sel.float() * dout_sel.float()).sum(dim=-1).permute(0, 2, 1).unsqueeze(1).unsqueeze(-1)
    dscores = probs * (dprob - delta)

    dq_chunk = torch.matmul(dscores.to(dtype=q_heads.dtype), k_q_heads).float()
    dq_chunk.mul_(float(softmax_scale))
    dq_chunk = dq_chunk.sum(dim=1).permute(0, 2, 1, 3).contiguous()
    dq_chunk.mul_(q_valid.unsqueeze(-1).unsqueeze(-1))
    dq_acc.index_add_(0, q_index.reshape(-1), dq_chunk.reshape(-1, num_q_heads, q_heads.shape[-1]))

    dk_chunk = torch.matmul(
        dscores.transpose(-1, -2).to(dtype=q_heads.dtype),
        q_heads_t.expand(-1, max_tiles, -1, -1, -1),
    ).float()
    dk_chunk.mul_(float(softmax_scale))
    dk_chunk = _collapse_q_to_kv_heads_local(
        dk_chunk.permute(0, 1, 3, 2, 4).reshape(chunk_range_count, max_tiles * k_slots, num_q_heads, q_heads.shape[-1]).contiguous(),
        num_kv_heads,
    ).view(chunk_range_count, max_tiles, k_slots, num_kv_heads, q_heads.shape[-1]).contiguous()
    dk_chunk.mul_(torch.logical_and(tile_active.unsqueeze(-1), k_valid).unsqueeze(-1).unsqueeze(-1))
    dk_acc.index_add_(0, k_index.reshape(-1), dk_chunk.reshape(-1, num_kv_heads, q_heads.shape[-1]))

    dv_chunk = torch.matmul(
        probs.transpose(-1, -2).to(dtype=q_heads.dtype),
        dout_heads.expand(-1, max_tiles, -1, -1, -1),
    ).float()
    dv_chunk = _collapse_q_to_kv_heads_local(
        dv_chunk.permute(0, 1, 3, 2, 4).reshape(chunk_range_count, max_tiles * k_slots, num_q_heads, dout_sel.shape[-1]).contiguous(),
        num_kv_heads,
    ).view(chunk_range_count, max_tiles, k_slots, num_kv_heads, dout_sel.shape[-1]).contiguous()
    dv_chunk.mul_(torch.logical_and(tile_active.unsqueeze(-1), k_valid).unsqueeze(-1).unsqueeze(-1))
    dv_acc.index_add_(0, k_index.reshape(-1), dv_chunk.reshape(-1, num_kv_heads, dout_sel.shape[-1]))


def _accumulate_cached_generalized_range_backward(
    *,
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    dq_acc: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
    softmax_scale: float,
    chunk_ranges: int = 1024,
) -> None:
    fused_q_row_idx = payload["fused_q_row_idx"]
    fused_q_length = payload["fused_q_length"]
    range_count = int(fused_q_row_idx.shape[0])
    if range_count <= 0:
        return
    num_q_heads = int(q_flat.shape[1])
    q_offsets = torch.arange(int(fused_q_row_idx.shape[1]), device=q_flat.device, dtype=torch.int32).view(1, -1)
    for range_start in range(0, range_count, chunk_ranges):
        range_end = min(range_count, range_start + chunk_ranges)
        chunk_q_row_idx = fused_q_row_idx[range_start:range_end].contiguous()
        chunk_q_length = fused_q_length[range_start:range_end].contiguous()
        q_valid = q_offsets < chunk_q_length.unsqueeze(1)
        q_index = chunk_q_row_idx.clamp_min(0).to(dtype=torch.long)
        chunk_rows = range_end - range_start
        q_sel = q_flat.index_select(0, q_index.reshape(-1)).view(chunk_rows, int(fused_q_row_idx.shape[1]), num_q_heads, q_flat.shape[-1]).contiguous()
        out_sel = out_flat.index_select(0, q_index.reshape(-1)).view(chunk_rows, int(fused_q_row_idx.shape[1]), num_q_heads, out_flat.shape[-1]).contiguous()
        dout_sel = dout_flat.index_select(0, q_index.reshape(-1)).view(chunk_rows, int(fused_q_row_idx.shape[1]), num_q_heads, dout_flat.shape[-1]).contiguous()
        lse_heads = lse_flat.index_select(0, q_index.reshape(-1)).view(chunk_rows, int(fused_q_row_idx.shape[1]), num_q_heads).permute(0, 2, 1).unsqueeze(-1).float()
        q_heads = q_sel.permute(0, 2, 1, 3).contiguous()

        exact_counts, exact_k_rows, _ = _materialize_range_tile_chunk(
            payload["fused_exact_tile_ptr"],
            payload["fused_exact_k_row_idx"],
            range_start=range_start,
            range_end=range_end,
        )
        if int(exact_k_rows.shape[1]) > 0:
            _accumulate_cached_generalized_range_tiles(
                q_heads=q_heads,
                out_sel=out_sel,
                dout_sel=dout_sel,
                lse_heads=lse_heads,
                q_valid=q_valid,
                k_flat=k_flat,
                v_flat=v_flat,
                dk_acc=dk_acc,
                dv_acc=dv_acc,
                q_index=q_index,
                dq_acc=dq_acc,
                softmax_scale=float(softmax_scale),
                tile_counts=exact_counts,
                padded_k_rows=exact_k_rows,
            )

        tail_counts, tail_k_rows, tail_masks = _materialize_range_tile_chunk(
            payload["fused_tail_tile_ptr"],
            payload["fused_tail_k_row_idx"],
            range_start=range_start,
            range_end=range_end,
            tail_mask_words=payload["fused_tail_mask_words"],
        )
        if int(tail_k_rows.shape[1]) > 0 and tail_masks is not None:
            _accumulate_cached_generalized_range_tiles(
                q_heads=q_heads,
                out_sel=out_sel,
                dout_sel=dout_sel,
                lse_heads=lse_heads,
                q_valid=q_valid,
                k_flat=k_flat,
                v_flat=v_flat,
                dk_acc=dk_acc,
                dv_acc=dv_acc,
                q_index=q_index,
                dq_acc=dq_acc,
                softmax_scale=float(softmax_scale),
                tile_counts=tail_counts,
                padded_k_rows=tail_k_rows,
                tail_mask_words=tail_masks,
            )


def run_cached_generalized_packed_backward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_env_enabled("FLASH_ATTN_HSA_CACHED_MONOLITHIC_BWD"):
        raise RuntimeError("cached_generalized_fused_backward_disabled")
    if not can_use_cached_generalized_fused_backward(payload, q, k, v, deterministic=deterministic):
        raise RuntimeError("cached_generalized_fused_backward_unsupported")

    q_flat = _flatten_row_tensor(q)
    k_flat = _flatten_row_tensor(k)
    v_flat = _flatten_row_tensor(v)
    out_flat = _flatten_row_tensor(out)
    dout_flat = _flatten_row_tensor(dout)
    use_triton_lse_flat = _can_use_triton_lse_public_to_flat(q, q_flat, lse)
    lse_flat_out = (
        _get_cached_lse_flat_for_backward_buffer(payload, q_flat, int(q.shape[2]))
        if use_triton_lse_flat
        else None
    )
    lse_flat = _flatten_cached_lse_for_backward(
        q,
        q_flat,
        lse,
        out=lse_flat_out,
        triton_checked=use_triton_lse_flat,
    )
    if softmax_scale is None:
        softmax_scale = q_flat.shape[-1] ** (-0.5)

    if q_flat.is_cuda:
        backward_payload = payload.get("cached_generalized_backward_payload")
        if not isinstance(backward_payload, dict) or backward_payload.get("status") != "ready":
            backward_payload = build_cached_generalized_backward_payload(payload)
            if isinstance(backward_payload, dict):
                payload["cached_generalized_backward_payload"] = backward_payload
        if isinstance(backward_payload, dict) and str(backward_payload.get("backward_kernel_family", "")) == "cached_masked_union_row_compact":
            from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
                _run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel,
            )

            dq_acc, dk_acc, dv_acc = _get_cached_backward_accum_buffers(payload, q_flat, k_flat, v_flat)
            _zero_cached_backward_accum_buffers(payload, q_flat, dq_acc, dk_acc, dv_acc)
            _run_synthetic_direct_row_micro_bwd_kernel_row_compact_one_kernel(
                q_flat,
                k_flat,
                v_flat,
                out_flat,
                dout_flat,
                lse_flat,
                backward_payload["q_row_idx"],
                backward_payload["row_k_row_idx"],
                backward_payload["union_k_row_idx"],
                backward_payload["row_k_to_union_idx"],
                backward_payload["union_to_row_slot"],
                backward_payload["q_length"],
                backward_payload["row_k_length"],
                backward_payload["union_k_length"],
                backward_payload.get("unique_key_row_idx"),
                backward_payload.get("unique_key_member_idx"),
                backward_payload.get("unique_key_union_idx"),
                backward_payload.get("unique_key_occurrence_row_ptr"),
                dq_acc,
                dk_acc,
                dv_acc,
                softmax_scale=float(softmax_scale),
                max_unique_key_occurrences=int(backward_payload.get("max_unique_key_occurrences", 0)),
                workspace=_get_cached_backward_workspace(payload),
            )
            dq, dk, dv = _finalize_cached_backward_grads(payload, q_flat, k_flat, v_flat, dq_acc, dk_acc, dv_acc)
            return (
                dq.view_as(q),
                dk.view_as(k),
                dv.view_as(v),
            )
        local_k_chunk = int(os.environ.get("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_LOCAL_K_CHUNK", "8"))
        use_tile_atomic_dkdv = os.environ.get(
            "FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_TILE_ATOMICS",
            "1",
        ).strip().lower() not in {"0", "false", "off", "no"}
        key_owned_mode = _env_mode("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED", default="auto")
        use_key_owned_dkdv = key_owned_mode != "off"
        use_direct_dq = _use_cached_backward_direct_dq(payload, q_flat)
        backward_payload = None
        if use_key_owned_dkdv or not use_tile_atomic_dkdv:
            backward_payload = payload.get("cached_generalized_backward_payload")
            if (
                not isinstance(backward_payload, dict)
                or backward_payload.get("status") != "ready"
            ):
                backward_payload = build_cached_generalized_backward_payload(payload)
                if not isinstance(backward_payload, dict) or backward_payload.get("status") != "ready":
                    if use_key_owned_dkdv:
                        use_key_owned_dkdv = False
                    else:
                        raise RuntimeError("cached_generalized_fused_backward_missing_payload")
                payload["cached_generalized_backward_payload"] = backward_payload
        if use_key_owned_dkdv and not _can_use_cached_backward_key_owned_dkdv(backward_payload):
            use_key_owned_dkdv = False
        if use_key_owned_dkdv and key_owned_mode == "auto" and not _auto_use_cached_backward_key_owned_dkdv(
            backward_payload,
            k_flat,
        ):
            use_key_owned_dkdv = False
        if not use_key_owned_dkdv and not use_tile_atomic_dkdv:
            if (
                not isinstance(backward_payload, dict)
                or "range_local_k_ptr" not in backward_payload
                or "range_local_k_row_idx" not in backward_payload
                or "exact_tile_local_k_idx" not in backward_payload
                or "tail_tile_local_k_idx" not in backward_payload
            ):
                backward_payload = build_cached_generalized_backward_payload(payload)
                if not isinstance(backward_payload, dict) or backward_payload.get("status") != "ready":
                    raise RuntimeError("cached_generalized_fused_backward_missing_payload")
                payload["cached_generalized_backward_payload"] = backward_payload
        if (not use_key_owned_dkdv) and use_direct_dq:
            dq = torch.empty_like(q_flat)
            dk_acc, dv_acc = _get_cached_backward_kv_accum_buffers(payload, k_flat, v_flat)
            _zero_cached_backward_kv_accum_buffers(payload, k_flat, dk_acc, dv_acc)
            dq_rows = dq
        elif not use_key_owned_dkdv:
            dq_acc, dk_acc, dv_acc = _get_cached_backward_accum_buffers(payload, q_flat, k_flat, v_flat)
            _zero_cached_backward_accum_buffers(payload, q_flat, dq_acc, dk_acc, dv_acc)
            dq_rows = dq_acc
        elif use_direct_dq:
            dq = torch.empty_like(q_flat)
            dk = torch.empty_like(k_flat)
            dv = torch.empty_like(v_flat)
            if not _cached_backward_key_owned_overwrites_all_kv_rows(backward_payload, k_flat):
                _zero_cached_backward_kv_final_buffers(payload, k_flat, dk, dv)
            dq_rows = dq
            dk_acc = dk
            dv_acc = dv
        else:
            dq_acc = _get_cached_backward_accum_buffers(payload, q_flat, k_flat, v_flat)[0]
            if q_flat.is_cuda:
                _run_cached_zero_rows_kernel(_get_cached_all_row_idx(payload, q_flat.device), dq_acc)
            else:
                dq_acc.zero_()
            dk = torch.empty_like(k_flat)
            dv = torch.empty_like(v_flat)
            if not _cached_backward_key_owned_overwrites_all_kv_rows(backward_payload, k_flat):
                _zero_cached_backward_kv_final_buffers(payload, k_flat, dk, dv)
            dq_rows = dq_acc
            dk_acc = dk
            dv_acc = dv
        from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
            _run_cached_generalized_fused_bwd_dkdv_kernel,
            _run_cached_generalized_fused_bwd_dkdv_range_kernel,
            _run_cached_generalized_fused_bwd_dq_kernel,
        )

        _run_cached_generalized_fused_bwd_dq_kernel(
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            dout_flat,
            lse_flat,
            payload["fused_q_row_idx"],
            payload["fused_q_length"],
            payload["fused_exact_tile_ptr"],
            payload["fused_exact_k_row_idx"],
            payload["fused_tail_tile_ptr"],
            payload["fused_tail_k_row_idx"],
            payload["fused_tail_mask_words"],
            dq_rows,
            dk_acc,
            dv_acc,
            softmax_scale=float(softmax_scale),
            compute_dkdv=not use_key_owned_dkdv,
        )
        if use_key_owned_dkdv:
            _run_cached_generalized_fused_bwd_dkdv_kernel(
                q_flat,
                k_flat,
                v_flat,
                out_flat,
                dout_flat,
                lse_flat,
                payload["fused_q_row_idx"],
                payload["fused_q_length"],
                payload["fused_tail_mask_words"],
                backward_payload["owned_k_row_idx"],
                backward_payload["owned_occurrence_ptr"],
                backward_payload["owned_occurrence_kind"],
                backward_payload["owned_occurrence_range_idx"],
                backward_payload["owned_occurrence_tile_idx"],
                backward_payload["owned_occurrence_col_idx"],
                dk_acc,
                dv_acc,
                softmax_scale=float(softmax_scale),
            )
            if use_direct_dq:
                return (
                    dq.view_as(q),
                    dk.view_as(k),
                    dv.view_as(v),
                )
            dq = torch.empty_like(q_flat)
            _run_cached_cast_rows_kernel(dq_acc, _get_cached_all_row_idx(payload, q_flat.device), dq)
            return (
                dq.view_as(q),
                dk.view_as(k),
                dv.view_as(v),
            )
        if (not use_tile_atomic_dkdv) and int(backward_payload["range_local_k_row_idx"].numel()) > 0:
            _run_cached_generalized_fused_bwd_dkdv_range_kernel(
                q_flat,
                k_flat,
                v_flat,
                out_flat,
                dout_flat,
                lse_flat,
                payload["fused_q_row_idx"],
                payload["fused_q_length"],
                payload["fused_exact_tile_ptr"],
                backward_payload["exact_tile_local_k_idx"],
                payload["fused_tail_tile_ptr"],
                backward_payload["tail_tile_local_k_idx"],
                payload["fused_tail_mask_words"],
                backward_payload["range_local_k_ptr"],
                backward_payload["range_local_k_row_idx"],
                dk_acc,
                dv_acc,
                softmax_scale=float(softmax_scale),
                local_k_chunk=local_k_chunk,
            )
        if use_direct_dq:
            dk, dv = _finalize_cached_backward_kv_grads(payload, k_flat, v_flat, dk_acc, dv_acc)
            return (
                dq.view_as(q),
                dk.view_as(k),
                dv.view_as(v),
            )
        dq, dk, dv = _finalize_cached_backward_grads(payload, q_flat, k_flat, v_flat, dq_acc, dk_acc, dv_acc)
        return (
            dq.view_as(q),
            dk.view_as(k),
            dv.view_as(v),
        )

    dq_acc, dk_acc, dv_acc = _get_cached_backward_accum_buffers(payload, q_flat, k_flat, v_flat)
    _zero_cached_backward_accum_buffers(payload, q_flat, dq_acc, dk_acc, dv_acc)
    fused_q_row_idx = payload["fused_q_row_idx"]
    fused_q_length = payload["fused_q_length"]
    exact_tile_range_idx = _get_tile_range_index(payload, ptr_key="fused_exact_tile_ptr")
    if int(exact_tile_range_idx.numel()) > 0:
        _accumulate_cached_generalized_tile_backward(
            q_flat=q_flat,
            k_flat=k_flat,
            v_flat=v_flat,
            out_flat=out_flat.float(),
            dout_flat=dout_flat,
            lse_flat=lse_flat,
            dq_acc=dq_acc,
            dk_acc=dk_acc,
            dv_acc=dv_acc,
            q_row_idx=fused_q_row_idx,
            q_length=fused_q_length,
            tile_k_row_idx=payload["fused_exact_k_row_idx"],
            tile_range_idx=exact_tile_range_idx,
            softmax_scale=float(softmax_scale),
            tail_mask_words=None,
        )
    tail_tile_range_idx = _get_tile_range_index(payload, ptr_key="fused_tail_tile_ptr")
    if int(tail_tile_range_idx.numel()) > 0:
        _accumulate_cached_generalized_tile_backward(
            q_flat=q_flat,
            k_flat=k_flat,
            v_flat=v_flat,
            out_flat=out_flat.float(),
            dout_flat=dout_flat,
            lse_flat=lse_flat,
            dq_acc=dq_acc,
            dk_acc=dk_acc,
            dv_acc=dv_acc,
            q_row_idx=fused_q_row_idx,
            q_length=fused_q_length,
            tile_k_row_idx=payload["fused_tail_k_row_idx"],
            tile_range_idx=tail_tile_range_idx,
            softmax_scale=float(softmax_scale),
            tail_mask_words=payload["fused_tail_mask_words"],
        )
    return (
        dq_acc.to(dtype=q.dtype).view_as(q),
        dk_acc.to(dtype=k.dtype).view_as(k),
        dv_acc.to(dtype=v.dtype).view_as(v),
    )
