from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch

from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
    _can_use_synthetic_2d_masked_fwd,
    _run_synthetic_2d_exact_gather_scatter_tc_fwd_kernel,
    _run_synthetic_2d_exact_tail_gather_scatter_tc_fwd_kernel,
    _run_synthetic_combine_scatter_rows_kernel,
    _run_synthetic_2d_masked_fwd_kernel,
    _run_synthetic_2d_masked_gather_fwd_kernel,
    _run_synthetic_2d_masked_gather_scatter_fwd_kernel,
    _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel,
    _run_synthetic_pack_kv_rows_kernel,
    _run_synthetic_pack_rows_kernel,
)
from flash_attn.cute.hsa_explicit_2d_sparse_analysis import _partition_packed_rows_by_support


@dataclass(frozen=True)
class CachedPackingPolicy:
    direct_density_threshold: float = 0.85
    window_density_threshold: float = 0.60
    k_gap_threshold: float = 1.25
    max_rows_per_group: int = 16
    tile_k: int = 32
    max_union_k_direct: int = 64
    max_union_k_2d: int = 128
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
    bit_offsets = torch.arange(32, dtype=torch.int64).view(1, 1, 32)
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
    key_row_masks: dict[int, int] = {}
    for row_idx, support_list in enumerate(tail_support_lists):
        if row_idx >= rows_per_range:
            break
        for support_row in support_list:
            key_row_masks[int(support_row)] = key_row_masks.get(int(support_row), 0) | (1 << row_idx)
    if not key_row_masks:
        return [], [], 0

    key_entries = sorted(
        key_row_masks.items(),
        key=lambda item: (-int(item[1]).bit_count(), int(item[1]), int(item[0])),
    )
    tail_tiles: list[list[int]] = []
    tail_mask_rows: list[list[int]] = []
    tail_live_pairs = 0
    for chunk_start in range(0, len(key_entries), keys_per_tile):
        chunk = key_entries[chunk_start : chunk_start + keys_per_tile]
        tile_rows = [-1] * keys_per_tile
        row_masks = [0] * rows_per_range
        for col_idx, (support_row, row_mask) in enumerate(chunk):
            tile_rows[col_idx] = int(support_row)
            tail_live_pairs += int(row_mask).bit_count()
            for row_idx in range(rows_per_range):
                if (int(row_mask) >> row_idx) & 1:
                    row_masks[row_idx] |= 1 << col_idx
        tail_tiles.append(tile_rows)
        tail_mask_rows.append(row_masks)
    return tail_tiles, tail_mask_rows, tail_live_pairs


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
    remaining_rows = [row_idx for row_idx, support_set in enumerate(support_sets) if support_set]
    fused_ranges: list[dict[str, Any]] = []
    exact_live_pairs = 0
    tail_live_pairs = 0

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

    while remaining_rows:
        active_rows = [row_idx for row_idx in remaining_rows if len(support_sets[row_idx]) >= keys_per_tile]
        best_group: list[int] | None = None
        best_common: set[int] | None = None
        best_score: tuple[int, int, int, int] | None = None

        for seed_idx in active_rows:
            seed_support = support_sets[seed_idx]
            candidate_rows = sorted(
                (row_idx for row_idx in active_rows if row_idx != seed_idx),
                key=lambda row_idx: (
                    len(seed_support.intersection(support_sets[row_idx])),
                    len(support_sets[row_idx]),
                    -row_idx,
                ),
                reverse=True,
            )
            subgroup = [seed_idx]
            common_support = set(seed_support)
            for row_idx in candidate_rows:
                if len(subgroup) >= rows_per_range:
                    break
                candidate_common = common_support.intersection(support_sets[row_idx])
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
            tail_group = _ordered_rows(remaining_rows[:rows_per_range] if len(remaining_rows) <= rows_per_range else _ordered_rows(remaining_rows)[:rows_per_range])
            tail_support_lists = [sorted(support_sets[row_idx]) for row_idx in tail_group]
            tail_tiles, tail_mask_rows, group_tail_live_pairs = _build_masked_tail_tiles(
                tail_support_lists,
                rows_per_range=rows_per_range,
                keys_per_tile=keys_per_tile,
            )
            fused_ranges.append(
                {
                    "q_rows": [int(q_rows[row_idx]) for row_idx in tail_group],
                    "exact_tiles": [],
                    "tail_tiles": tail_tiles,
                    "tail_mask_rows": tail_mask_rows,
                }
            )
            tail_live_pairs += int(group_tail_live_pairs)
            remaining_set = set(tail_group)
            remaining_rows = [row_idx for row_idx in remaining_rows if row_idx not in remaining_set]
            continue

        ordered_group = _ordered_rows(best_group)
        sorted_common = sorted(best_common)
        tile_count = best_score[2]
        exact_tiles = [
            sorted_common[tile_idx * keys_per_tile : (tile_idx + 1) * keys_per_tile]
            for tile_idx in range(tile_count)
        ]
        covered_keys = set(key for tile in exact_tiles for key in tile)
        tail_support_lists = [
            sorted(support_sets[row_idx].difference(covered_keys))
            for row_idx in ordered_group
        ]
        tail_tiles, tail_mask_rows, group_tail_live_pairs = _build_masked_tail_tiles(
            tail_support_lists,
            rows_per_range=rows_per_range,
            keys_per_tile=keys_per_tile,
        )
        fused_ranges.append(
            {
                "q_rows": [int(q_rows[row_idx]) for row_idx in ordered_group],
                "exact_tiles": exact_tiles,
                "tail_tiles": tail_tiles,
                "tail_mask_rows": tail_mask_rows,
            }
        )
        exact_live_pairs += len(ordered_group) * len(covered_keys)
        tail_live_pairs += int(group_tail_live_pairs)
        remaining_set = set(best_group)
        remaining_rows = [row_idx for row_idx in remaining_rows if row_idx not in remaining_set]

    return fused_ranges, exact_live_pairs, tail_live_pairs


def _materialize_exact_dense_range_tensors(
    *,
    device: torch.device,
    exact_ranges: list[dict[str, Any]],
    rows_per_range: int,
    keys_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_count = len(exact_ranges)
    q_row_idx = torch.full((range_count, rows_per_range), -1, dtype=torch.int32, device=device)
    q_length = torch.zeros((range_count,), dtype=torch.int32, device=device)
    tile_ptr = torch.zeros((range_count + 1,), dtype=torch.int32, device=device)
    tile_count = sum(len(range_entry["tiles"]) for range_entry in exact_ranges)
    tile_k_row_idx = torch.full((tile_count, keys_per_tile), -1, dtype=torch.int32, device=device)

    tile_offset = 0
    for range_idx, range_entry in enumerate(exact_ranges):
        q_rows = [int(value) for value in range_entry["q_rows"]]
        tiles = range_entry["tiles"]
        q_length[range_idx] = len(q_rows)
        q_row_idx[range_idx, : len(q_rows)] = torch.tensor(q_rows, dtype=torch.int32, device=device)
        for tile in tiles:
            tile_k_row_idx[tile_offset] = torch.tensor(tile, dtype=torch.int32, device=device)
            tile_offset += 1
        tile_ptr[range_idx + 1] = tile_offset
    return q_row_idx, q_length, tile_ptr, tile_k_row_idx


def _materialize_fused_exact_tail_range_tensors(
    *,
    device: torch.device,
    fused_ranges: list[dict[str, Any]],
    rows_per_range: int,
    keys_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    range_count = len(fused_ranges)
    q_row_idx = torch.full((range_count, rows_per_range), -1, dtype=torch.int32, device=device)
    q_length = torch.zeros((range_count,), dtype=torch.int32, device=device)
    exact_tile_ptr = torch.zeros((range_count + 1,), dtype=torch.int32, device=device)
    exact_tile_count = sum(len(range_entry.get("exact_tiles", ())) for range_entry in fused_ranges)
    exact_k_row_idx = torch.full((exact_tile_count, keys_per_tile), -1, dtype=torch.int32, device=device)
    tail_tile_ptr = torch.zeros((range_count + 1,), dtype=torch.int32, device=device)
    tail_tile_count = sum(len(range_entry.get("tail_tiles", ())) for range_entry in fused_ranges)
    tail_k_row_idx = torch.full((tail_tile_count, keys_per_tile), -1, dtype=torch.int32, device=device)
    tail_mask_words = torch.zeros((tail_tile_count, rows_per_range, 1), dtype=torch.int32, device=device)

    exact_tile_offset = 0
    tail_tile_offset = 0
    for range_idx, range_entry in enumerate(fused_ranges):
        q_rows = [int(value) for value in range_entry["q_rows"]]
        exact_tiles = [[int(value) for value in tile] for tile in range_entry.get("exact_tiles", ())]
        tail_tiles = [
            [int(value) for value in tail_tile]
            for tail_tile in range_entry.get("tail_tiles", ())
        ]
        tail_mask_rows = [
            [int(value) for value in row_masks]
            for row_masks in range_entry.get("tail_mask_rows", ())
        ]
        q_length[range_idx] = len(q_rows)
        if q_rows:
            q_row_idx[range_idx, : len(q_rows)] = torch.tensor(q_rows, dtype=torch.int32, device=device)
        for tile in exact_tiles:
            exact_k_row_idx[exact_tile_offset, : len(tile)] = torch.tensor(tile, dtype=torch.int32, device=device)
            exact_tile_offset += 1
        exact_tile_ptr[range_idx + 1] = exact_tile_offset
        for tile_rows, row_masks in zip(tail_tiles, tail_mask_rows, strict=True):
            tail_k_row_idx[tail_tile_offset, : len(tile_rows)] = torch.tensor(
                tile_rows,
                dtype=torch.int32,
                device=device,
            )
            tail_mask_words[tail_tile_offset, : len(row_masks), 0] = torch.tensor(
                row_masks,
                dtype=torch.int32,
                device=device,
            )
            tail_tile_offset += 1
        tail_tile_ptr[range_idx + 1] = tail_tile_offset
    return (
        q_row_idx,
        q_length,
        exact_tile_ptr,
        exact_k_row_idx,
        tail_tile_ptr,
        tail_k_row_idx,
        tail_mask_words,
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
        union_rows = sorted({support_row for row_idx in row_group for support_row in support_lists[row_idx]})
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

    range_execution: list[dict[str, int | bool]] = []
    for group_start, group_end in combine_group_ranges:
        range_rows: list[int] = []
        for group_idx in range(int(group_start), int(group_end)):
            range_rows.extend(
                int(q_row) for q_row in group_q_rows[group_idx] if int(q_row) >= 0
            )
        scatter_only = bool(range_rows) and all(row_counts.get(q_row, 0) == 1 for q_row in range_rows)
        range_execution.append(
            {
                "group_start": int(group_start),
                "group_end": int(group_end),
                "scatter_only": scatter_only,
                **(
                    {}
                    if group_families is None or int(group_start) >= len(group_families)
                    else {"family": str(group_families[int(group_start)])}
                ),
            }
        )
    return range_execution


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
    if int(resolved.max_union_k_2d) <= 0 or int(resolved.max_union_k_2d) > 128:
        raise ValueError("max_union_k_2d must be between 1 and 128")
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
    live_k_rows = sorted({support_row for support_list in support_lists for support_row in support_list})
    num_live_rows = len(q_rows)
    num_live_k = len(live_k_rows)
    k_extent = live_k_rows[-1] - live_k_rows[0] + 1 if live_k_rows else 0
    q_extent = max(q_rows) - min(q_rows) + 1 if q_rows else 0
    max_row_support = max((len(support_list) for support_list in support_lists), default=0)
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
    device = torch.device("cpu")
    bucket_mask = _build_bucket_mask_from_support_lists(support_lists, device=device)
    row_groups = _group_rows_by_support_patterns(
        bucket_mask,
        max_rows_per_group=min(int(policy.max_rows_per_group), len(q_rows)),
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
    max_union_k = max(
        int(policy.max_union_k_direct),
        max((len(support_list) for support_list in support_lists), default=0),
    )
    row_groups = _group_support_lists_by_span(
        support_lists,
        max_rows_per_group=min(int(policy.max_rows_per_group), len(q_rows)),
        max_union_k=max_union_k,
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
    include_mask_bool: bool = True,
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
    family_union_k_sums = {family: 0.0 for family in ("direct_passthrough", "k_window", "union_2d")}
    family_scatter_only_rows = {family: 0 for family in ("direct_passthrough", "k_window", "union_2d")}
    for group_idx, family in enumerate(group_families):
        q_count = len(group_q_rows[group_idx])
        family_group_counts[family] = family_group_counts.get(family, 0) + 1
        family_union_k_sums[family] = family_union_k_sums.get(family, 0.0) + float(len(group_k_rows[group_idx]))
        is_scatter_only = any(
            int(entry["group_start"]) <= group_idx < int(entry["group_end"]) and bool(entry["scatter_only"])
            for entry in range_execution
        )
        if is_scatter_only:
            scatter_only_rows += q_count
            family_scatter_only_rows[family] = family_scatter_only_rows.get(family, 0) + q_count

    if exact_range_count > 0:
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
        residual_live_pairs = int(geometry_base.get("cached_generalized_live_pairs", 0))
    if fused_range_count > 0:
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
            "cached_generalized_avg_group_fill": float(sum(group_fill) / len(group_fill)) if group_fill else 0.0,
            "cached_generalized_scatter_only_ranges": scatter_only_ranges,
            "cached_generalized_scatter_only_rows": scatter_only_rows,
            "family_group_counts": family_group_counts,
            "family_avg_union_k": {
                family: (
                    family_union_k_sums[family] / family_group_counts[family]
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
            "exact_dense_live_pairs": int(exact_live_pairs),
            "exact_dense_slots": int(exact_hardware_slots),
            "exact_dense_hardware_fill": float(exact_live_pairs) / max(1, exact_hardware_slots),
            "exact_dense_coverage_frac": exact_coverage_frac,
            "residual_live_pairs": int(residual_live_pairs),
            "residual_coverage_frac": float(residual_live_pairs) / max(1, int(geometry_base.get("cached_generalized_live_pairs", 0))),
            "exact_kernel_family": str(exact_kernel_family),
            "residual_mode": str(residual_mode),
            "fused_range_count": int(fused_range_count),
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
        "mask_words": mask_words.contiguous(),
        "q_length": q_length.contiguous(),
        "k_length": k_length.contiguous(),
        "total_rows": int(q_flat.shape[0]),
        "combine_group_ranges": combine_group_ranges,
        "range_execution": range_execution,
        "union_kernel": str(union_kernel),
        "geometry": geometry,
        "_workspace": {},
        "exact_dense_q_row_idx": exact_q_row_idx.contiguous(),
        "exact_dense_q_length": exact_q_length.contiguous(),
        "exact_dense_tile_ptr": exact_tile_ptr.contiguous(),
        "exact_dense_k_row_idx": exact_k_row_idx.contiguous(),
        "exact_dense_rows_per_range": int(exact_rows_per_range),
        "exact_dense_keys_per_tile": int(exact_keys_per_tile),
        "exact_dense_min_rows": int(exact_min_rows),
        "exact_kernel_family": str(exact_kernel_family),
        "residual_mode": str(residual_mode),
        "fused_q_row_idx": fused_q_row_idx.contiguous(),
        "fused_q_length": fused_q_length.contiguous(),
        "fused_exact_tile_ptr": fused_exact_tile_ptr.contiguous(),
        "fused_exact_k_row_idx": fused_exact_k_row_idx.contiguous(),
        "fused_tail_tile_ptr": fused_tail_tile_ptr.contiguous(),
        "fused_tail_k_row_idx": fused_tail_k_row_idx.contiguous(),
        "fused_tail_mask_words": fused_tail_mask_words.contiguous(),
    }
    if mask_bool is not None:
        payload["mask_bool"] = mask_bool.contiguous()
    return payload


def build_cached_generalized_packed_forward_payload(
    runtime,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    policy: CachedPackingPolicy | None = None,
    policy_overrides: dict[str, Any] | None = None,
    include_mask_bool: bool = True,
) -> dict[str, Any]:
    del k, v
    q_flat = _flatten_row_tensor(q)
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
        baseline_live_row_count += live_row_count
        baseline_packed_k_sum += int(packed_k) * live_row_count
        live_q_rows_cpu = [int(value) for value in live_q_rows.detach().cpu().tolist()]
        live_support_rows_cpu = live_support_rows.detach().cpu().tolist()
        live_support_valid_cpu = live_support_valid.detach().cpu().tolist()
        support_lists = [
            [int(support_row) for support_row, valid in zip(row_supports, row_valid, strict=True) if valid and int(support_row) >= 0]
            for row_supports, row_valid in zip(live_support_rows_cpu, live_support_valid_cpu, strict=True)
        ]
        q_rows = [int(q_row) for q_row in live_q_rows_cpu]
        if not q_rows:
            continue
        bucket_stats = _summarize_bucket_support_geometry(q_rows, support_lists)
        family = _choose_cached_packing_family(bucket_stats, resolved_policy)
        family_counts[family] = family_counts.get(family, 0) + 1
        family_live_pairs[family] = family_live_pairs.get(family, 0) + int(bucket_stats["live_pairs"])
        total_live_pairs += int(bucket_stats["live_pairs"])
        family_bucket_stats[family]["active_density"] += float(bucket_stats["active_density"])
        family_bucket_stats[family]["k_gap_ratio"] += float(bucket_stats["k_gap_ratio"])
        family_bucket_stats[family]["q_gap_ratio"] += float(bucket_stats["q_gap_ratio"])
        family_bucket_stats[family]["count"] += 1
        qgroup_bucket_idx = bucket_idx if bucket_qgroup_bucket_idx is None else int(bucket_qgroup_bucket_idx[bucket_idx])
        if family == "direct_passthrough":
            _append_direct_passthrough_groups(
                q_rows=q_rows,
                support_lists=support_lists,
                policy=resolved_policy,
                group_q_rows=group_q_rows,
                group_k_rows=group_k_rows,
                group_mask_words=group_mask_words,
                group_fill=group_fill,
                combine_group_ranges=combine_group_ranges,
                group_families=group_families,
            )
            continue
        if family == "k_window":
            _append_k_window_groups(
                q_rows=q_rows,
                support_lists=support_lists,
                policy=resolved_policy,
                group_q_rows=group_q_rows,
                group_k_rows=group_k_rows,
                group_mask_words=group_mask_words,
                group_fill=group_fill,
                combine_group_ranges=combine_group_ranges,
                group_families=group_families,
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

    union_entries_by_qgroup: dict[int, list[tuple[int, list[int]]]] = {}
    for bucket_entry in union_bucket_entries:
        qgroup_entries = union_entries_by_qgroup.setdefault(int(bucket_entry["qgroup_bucket_idx"]), [])
        qgroup_entries.extend(zip(bucket_entry["q_rows"], bucket_entry["support_lists"], strict=True))

    for qgroup_bucket_idx in sorted(union_entries_by_qgroup):
        row_support_map: dict[int, set[int]] = {}
        for q_row, support_list in union_entries_by_qgroup[qgroup_bucket_idx]:
            if not support_list:
                continue
            row_support_map.setdefault(int(q_row), set()).update(int(support_row) for support_row in support_list)
        merged_q_rows = [int(q_row) for q_row in sorted(row_support_map)]
        merged_support_lists = [sorted(row_support_map[q_row]) for q_row in sorted(row_support_map)]
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
                fallback_rows.add(int(q_row))
                continue
            filtered_q_rows.append(int(q_row))
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
                    residual_q_rows.append(int(q_row))
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
    include_mask_bool: bool = True,
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
    if mask_bool is not None:
        payload["mask_bool"] = mask_bool.contiguous()
    return payload


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
    return int(fused_q_row_idx.shape[0]), int(fused_q_length.sum().item())


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
    return int(exact_q_row_idx.shape[0]), int(exact_q_length.sum().item())


def _run_cached_masked_payload_forward(
    payload: dict[str, Any],
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[int, int, int, int]:
    packed_q = int(payload["packed_q"])
    packed_k = int(payload["support_rows"])
    tile_k = int(payload.get("tile_k", 32))
    group_count = int(payload["q_row_idx"].shape[0])
    if group_count <= 0:
        return 0, 0, 0, 0

    q_buf_flat, k_buf_flat, v_buf_flat, _, _ = _get_cached_direct_2d_buffers(payload, q_flat, k_flat, v_flat)
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
                tile_k=tile_k,
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
                tile_k=tile_k,
            )

    for range_entry in range_execution:
        group_start = int(range_entry["group_start"])
        group_end = int(range_entry["group_end"])
        if group_end <= group_start:
            continue
        if _can_use_cached_union_tc(
            payload,
            q_flat,
            k_flat,
            v_flat,
            group_start=group_start,
            group_end=group_end,
            range_entry=range_entry,
        ):
            try:
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
                continue
            except Exception:
                pass
        if bool(range_entry.get("scatter_only")):
            if str(range_entry.get("family", "union_2d")) == "union_2d":
                scalar_groups, scalar_rows = _cached_union_range_counts(
                    payload,
                    group_start=group_start,
                    group_end=group_end,
                )
                union_scalar_group_count += scalar_groups
                union_scalar_row_count += scalar_rows
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
                    tile_k=tile_k,
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
    return union_tc_group_count, union_tc_row_count, union_scalar_group_count, union_scalar_row_count


def run_cached_direct_2d_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    return_lse: bool = False,
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
    if packed_k > 0 and not _can_use_synthetic_2d_masked_fwd(q_flat, k_flat, v_flat, packed_q=packed_q, packed_k=packed_k):
        raise RuntimeError("cached_direct_2d_forward_unsupported")
    if not has_exact_dense and not has_fused_ranges and int(payload["q_row_idx"].shape[0]) <= 0:
        raise RuntimeError("cached_direct_2d_forward_empty")

    q_buf_flat, k_buf_flat, v_buf_flat, out_flat, lse_flat = _get_cached_direct_2d_buffers(payload, q_flat, k_flat, v_flat)
    del q_buf_flat, k_buf_flat, v_buf_flat
    out_flat.zero_()
    lse_flat.fill_(float("-inf"))
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
    if q.ndim == 4:
        out = out_flat.to(dtype=v.dtype).view(q.shape[0], q.shape[1], q.shape[2], v.shape[3]).contiguous()
        if not return_lse:
            return out
        lse = lse_flat.view(q.shape[0], q.shape[1], q.shape[2]).permute(0, 2, 1).contiguous()
        return out, lse
    out = out_flat.to(dtype=v.dtype).contiguous()
    if not return_lse:
        return out
    return out, lse_flat.transpose(0, 1).contiguous()


def run_cached_generalized_packed_forward(
    payload: dict[str, Any],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return run_cached_direct_2d_forward(
        payload,
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
    )
