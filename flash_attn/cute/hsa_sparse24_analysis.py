from __future__ import annotations

from collections import Counter
from typing import Any

import torch


def _load_hsa_module():
    import flash_attn.cute.hsa as hsa_module

    return hsa_module


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _top_reason(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    reason, _ = max(counter.items(), key=lambda item: (item[1], item[0]))
    return reason


def _word_to_u32(word: int) -> int:
    return int(word) & 0xFFFFFFFF


def _decode_mask_words(
    mask_words: torch.Tensor | list[int],
    *,
    rows: int,
    cols: int,
    words_per_row: int,
) -> torch.Tensor:
    mask = torch.zeros((rows, cols), dtype=torch.bool)
    if rows <= 0 or cols <= 0:
        return mask
    for row_idx in range(rows):
        row_word_start = row_idx * words_per_row
        for word_idx in range(words_per_row):
            word = _word_to_u32(mask_words[row_word_start + word_idx])
            col_base = word_idx * 32
            if col_base >= cols:
                continue
            for bit_idx in range(min(32, cols - col_base)):
                if (word >> bit_idx) & 1:
                    mask[row_idx, col_base + bit_idx] = True
    return mask


def _classify_axis_support(
    *,
    operand: str,
    length: int,
    active_count: int,
    group_size: int,
    nnz_per_group: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if active_count <= 0:
        reasons.append("empty_segment")
    elif active_count == length:
        reasons.append("dense_segment")
    if length % group_size != 0:
        reasons.append("unaligned_tail")
    if length > 0 and active_count > (length // group_size) * nnz_per_group:
        reasons.append("gt2_per_group")
    return {
        "operand": operand,
        "support_length": length,
        "support_nnz": active_count,
        "support_density": (active_count / length) if length > 0 else 0.0,
        "eligible": len(reasons) == 0,
        "failure_reasons": reasons,
    }


def _classify_mask_matrix(
    mask: torch.Tensor,
    *,
    group_size: int = 4,
    nnz_per_group: int = 2,
) -> dict[str, Any]:
    mask = mask.to(dtype=torch.bool, device="cpu")
    rows, cols = mask.shape
    total_pairs = rows * cols
    allowed_pairs = int(mask.sum().item())
    fill = (allowed_pairs / total_pairs) if total_pairs > 0 else 0.0
    row_identical = bool(rows > 0 and torch.equal(mask, mask[:1].expand_as(mask)))
    col_identical = bool(cols > 0 and torch.equal(mask, mask[:, :1].expand_as(mask)))

    axis_candidates: list[dict[str, Any]] = []
    if row_identical and rows > 0:
        axis_candidates.append(
            _classify_axis_support(
                operand="B",
                length=cols,
                active_count=int(mask[0].sum().item()),
                group_size=group_size,
                nnz_per_group=nnz_per_group,
            )
        )
    if col_identical and cols > 0:
        axis_candidates.append(
            _classify_axis_support(
                operand="A",
                length=rows,
                active_count=int(mask[:, 0].sum().item()),
                group_size=group_size,
                nnz_per_group=nnz_per_group,
            )
        )

    eligible_candidates = [candidate for candidate in axis_candidates if candidate["eligible"]]
    if eligible_candidates:
        chosen = min(
            eligible_candidates,
            key=lambda candidate: (
                candidate["support_density"],
                candidate["support_nnz"],
                candidate["operand"],
            ),
        )
        return {
            "rows": rows,
            "cols": cols,
            "allowed_pairs": allowed_pairs,
            "total_pairs": total_pairs,
            "fill": fill,
            "row_identical": row_identical,
            "col_identical": col_identical,
            "eligible": True,
            "operand": chosen["operand"],
            "support_length": chosen["support_length"],
            "support_nnz": chosen["support_nnz"],
            "support_density": chosen["support_density"],
            "failure_reasons": [],
            "axis_candidates": axis_candidates,
        }

    failure_reasons: list[str] = []
    if not axis_candidates:
        if total_pairs > 0 and allowed_pairs == total_pairs:
            failure_reasons.append("dense_segment")
        else:
            failure_reasons.extend(["row_dependent_columns", "requires_output_mask"])
    else:
        seen = set()
        for candidate in axis_candidates:
            for reason in candidate["failure_reasons"]:
                if reason not in seen:
                    failure_reasons.append(reason)
                    seen.add(reason)

    chosen = axis_candidates[0] if axis_candidates else None
    return {
        "rows": rows,
        "cols": cols,
        "allowed_pairs": allowed_pairs,
        "total_pairs": total_pairs,
        "fill": fill,
        "row_identical": row_identical,
        "col_identical": col_identical,
        "eligible": False,
        "operand": None if chosen is None else chosen["operand"],
        "support_length": None if chosen is None else chosen["support_length"],
        "support_nnz": None if chosen is None else chosen["support_nnz"],
        "support_density": None if chosen is None else chosen["support_density"],
        "failure_reasons": failure_reasons,
        "axis_candidates": axis_candidates,
    }


def _summarize_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
    failure_reasons: Counter[str] = Counter()
    eligible_pairs = 0
    total_pairs = 0
    fills: list[float] = []
    for segment in segments:
        total_pairs += int(segment["total_pairs"])
        eligible_pairs += int(segment["allowed_pairs"]) if segment["eligible"] else 0
        fills.append(float(segment["fill"]))
        for reason in segment["failure_reasons"]:
            failure_reasons[reason] += 1

    return {
        "segments": len(segments),
        "eligible_segments": sum(1 for segment in segments if segment["eligible"]),
        "total_pairs": total_pairs,
        "eligible_pairs": eligible_pairs,
        "pair_coverage": (eligible_pairs / total_pairs) if total_pairs > 0 else 0.0,
        "avg_fill": _mean_or_zero(fills),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "top_failure_reason": _top_reason(failure_reasons),
    }


def _mask_kind_name(kind: int) -> str:
    hsa_module = _load_hsa_module()
    if kind == hsa_module._HSA_FWD_TILE_AFFINE_PREFIX:
        return "affine_prefix"
    if kind == hsa_module._HSA_FWD_TILE_ROW_PREFIX:
        return "row_prefix"
    if kind == hsa_module._HSA_FWD_TILE_BITMAP:
        return "bitmap"
    if kind == hsa_module._HSA_FWD_TILE_NONE:
        return "none"
    return f"kind_{kind}"


def _materialize_forward_partial_mask(
    *,
    tile_masks: Any,
    block_id: int,
    q_len: int,
    k_len: int,
) -> tuple[torch.Tensor, int]:
    hsa_module = _load_hsa_module()
    kind = int(tile_masks.tile_kind[block_id].item())
    if kind == hsa_module._HSA_FWD_TILE_AFFINE_PREFIX:
        affine_base = int(tile_masks.affine_base[block_id].item())
        prefix = torch.clamp(
            affine_base + torch.arange(q_len, dtype=torch.int32),
            min=0,
            max=k_len,
        )
        mask = torch.arange(k_len, dtype=torch.int32)[None, :] < prefix[:, None]
        return mask, kind
    if kind == hsa_module._HSA_FWD_TILE_ROW_PREFIX:
        prefix_start = int(tile_masks.row_prefix_row_ptr[block_id].item())
        prefix = tile_masks.row_prefix_len[prefix_start:prefix_start + q_len].to(dtype=torch.int32, device="cpu")
        prefix = torch.clamp(prefix, min=0, max=k_len)
        mask = torch.arange(k_len, dtype=torch.int32)[None, :] < prefix[:, None]
        return mask, kind
    if kind == hsa_module._HSA_FWD_TILE_BITMAP:
        word_start = int(tile_masks.bitmap_word_row_ptr[block_id].item())
        words = tile_masks.bitmap_words[word_start:word_start + q_len * tile_masks.words_per_row]
        mask = _decode_mask_words(
            words,
            rows=q_len,
            cols=k_len,
            words_per_row=tile_masks.words_per_row,
        )
        return mask, kind
    return torch.zeros((q_len, k_len), dtype=torch.bool), kind


def _analyze_physical_forward_tiles(
    schedule,
    runtime,
    *,
    group_size: int,
    nnz_per_group: int,
) -> dict[str, Any]:
    sparse_tensors = runtime.forward_sparse
    tile_masks = runtime.forward_tile_masks
    q_block_size, k_block_size = sparse_tensors.block_size
    seqlen = int(schedule.seqlen)

    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = sparse_tensors.full_block_idx.detach().cpu()

    tile_masks_cpu = type("TileMasksCPU", (), {})()
    tile_masks_cpu.block_id_table = tile_masks.block_id_table.detach().cpu()
    tile_masks_cpu.tile_kind = tile_masks.tile_kind.detach().cpu()
    tile_masks_cpu.affine_base = tile_masks.affine_base.detach().cpu()
    tile_masks_cpu.row_prefix_row_ptr = tile_masks.row_prefix_row_ptr.detach().cpu()
    tile_masks_cpu.row_prefix_len = tile_masks.row_prefix_len.detach().cpu()
    tile_masks_cpu.bitmap_word_row_ptr = tile_masks.bitmap_word_row_ptr.detach().cpu()
    tile_masks_cpu.bitmap_words = tile_masks.bitmap_words.detach().cpu()
    tile_masks_cpu.words_per_row = tile_masks.words_per_row

    segments: list[dict[str, Any]] = []
    partial_segments = 0
    full_segments = 0
    batch_size = int(schedule.batch_size)
    num_q_blocks = int(mask_block_cnt.shape[2])
    for batch_idx in range(batch_size):
        for q_block in range(num_q_blocks):
            q_start = q_block * q_block_size
            q_end = min(seqlen, q_start + q_block_size)
            q_len = q_end - q_start

            full_count = int(full_block_cnt[batch_idx, 0, q_block].item())
            for offset in range(full_count):
                k_block = int(full_block_idx[batch_idx, 0, q_block, offset].item())
                k_start = k_block * k_block_size
                k_end = min(seqlen, k_start + k_block_size)
                k_len = k_end - k_start
                classification = _classify_mask_matrix(
                    torch.ones((q_len, k_len), dtype=torch.bool),
                    group_size=group_size,
                    nnz_per_group=nnz_per_group,
                )
                segments.append(
                    {
                        "segment_kind": "full_tile",
                        "mask_kind": "full",
                        "batch_idx": batch_idx,
                        "q_block": q_block,
                        "k_block": k_block,
                        **classification,
                    }
                )
                full_segments += 1

            mask_count = int(mask_block_cnt[batch_idx, 0, q_block].item())
            for offset in range(mask_count):
                k_block = int(mask_block_idx[batch_idx, 0, q_block, offset].item())
                k_start = k_block * k_block_size
                k_end = min(seqlen, k_start + k_block_size)
                k_len = k_end - k_start
                block_id = int(tile_masks_cpu.block_id_table[batch_idx, q_block, k_block].item())
                mask, kind = _materialize_forward_partial_mask(
                    tile_masks=tile_masks_cpu,
                    block_id=block_id,
                    q_len=q_len,
                    k_len=k_len,
                )
                classification = _classify_mask_matrix(
                    mask,
                    group_size=group_size,
                    nnz_per_group=nnz_per_group,
                )
                segments.append(
                    {
                        "segment_kind": "partial_tile",
                        "mask_kind": _mask_kind_name(kind),
                        "batch_idx": batch_idx,
                        "q_block": q_block,
                        "k_block": k_block,
                        "block_id": block_id,
                        **classification,
                    }
                )
                partial_segments += 1

    summary = _summarize_segments(segments)
    summary["partial_segments"] = partial_segments
    summary["full_segments"] = full_segments
    return {
        "segments": segments,
        "summary": summary,
    }


def _analyze_bucket_like_segments(
    *,
    packed_q: list[int] | torch.Tensor,
    packed_k: list[int] | torch.Tensor,
    dense: list[bool] | torch.Tensor,
    row_ptr: list[int] | torch.Tensor,
    words_per_row: list[int] | torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    mask_word_row_ptr: list[int] | torch.Tensor,
    mask_words: torch.Tensor,
    group_size: int,
    nnz_per_group: int,
    label_prefix: str,
    bucket_fill: list[float] | torch.Tensor | None = None,
) -> dict[str, Any]:
    packed_q_list = [int(value) for value in packed_q]
    packed_k_list = [int(value) for value in packed_k]
    dense_list = [bool(value) for value in dense]
    row_ptr_list = [int(value) for value in row_ptr]
    words_per_row_list = [int(value) for value in words_per_row]
    mask_word_row_ptr_list = [int(value) for value in mask_word_row_ptr]
    q_length_cpu = q_length.detach().cpu()
    k_length_cpu = k_length.detach().cpu()
    mask_words_cpu = mask_words.detach().cpu()
    bucket_fill_list = None if bucket_fill is None else [float(value) for value in bucket_fill]

    segments: list[dict[str, Any]] = []
    per_bucket: list[dict[str, Any]] = []
    for bucket_idx, bucket_packed_q in enumerate(packed_q_list):
        bucket_packed_k = packed_k_list[bucket_idx]
        bucket_dense = dense_list[bucket_idx]
        bucket_words_per_row = words_per_row_list[bucket_idx]
        data_start = row_ptr_list[bucket_idx]
        data_end = row_ptr_list[bucket_idx + 1]
        bucket_size = data_end - data_start
        mask_word_start = mask_word_row_ptr_list[bucket_idx]
        bucket_segments: list[dict[str, Any]] = []
        for member_idx in range(bucket_size):
            q_len = int(q_length_cpu[data_start + member_idx].item())
            k_len = int(k_length_cpu[data_start + member_idx].item())
            if bucket_dense:
                mask = torch.ones((q_len, k_len), dtype=torch.bool)
            else:
                row_word_count = bucket_packed_q * bucket_words_per_row
                member_word_start = mask_word_start + member_idx * row_word_count
                member_words = mask_words_cpu[member_word_start:member_word_start + row_word_count]
                mask = _decode_mask_words(
                    member_words,
                    rows=bucket_packed_q,
                    cols=bucket_packed_k,
                    words_per_row=bucket_words_per_row,
                )[:q_len, :k_len]
            classification = _classify_mask_matrix(
                mask,
                group_size=group_size,
                nnz_per_group=nnz_per_group,
            )
            segment = {
                "segment_kind": f"{label_prefix}_segment",
                "bucket_idx": bucket_idx,
                "member_idx": member_idx,
                "bucket_packed_q": bucket_packed_q,
                "bucket_packed_k": bucket_packed_k,
                "bucket_dense": bucket_dense,
                **classification,
            }
            segments.append(segment)
            bucket_segments.append(segment)

        bucket_summary = _summarize_segments(bucket_segments)
        bucket_summary.update(
            {
                "bucket_idx": bucket_idx,
                "bucket_size": bucket_size,
                "packed_q": bucket_packed_q,
                "packed_k": bucket_packed_k,
                "dense": bucket_dense,
            }
        )
        if bucket_fill_list is not None:
            bucket_summary["reported_fill"] = bucket_fill_list[bucket_idx]
        per_bucket.append(bucket_summary)

    summary = _summarize_segments(segments)
    if bucket_fill_list is not None:
        summary["avg_reported_fill"] = _mean_or_zero(bucket_fill_list)
    return {
        "segments": segments,
        "per_bucket": per_bucket,
        "summary": summary,
    }


def _analyze_synthetic_forward_segments(
    runtime,
    *,
    use_synthetic_grid: bool,
    group_size: int,
    nnz_per_group: int,
) -> dict[str, Any]:
    if not use_synthetic_grid or runtime.forward_synthetic_grid is None:
        return {
            "enabled": False,
            "bucket_analysis": {"segments": [], "per_bucket": [], "summary": _summarize_segments([])},
            "direct_analysis": {"segments": [], "per_bucket": [], "summary": _summarize_segments([])},
            "summary": {
                "segments": 0,
                "eligible_segments": 0,
                "total_pairs": 0,
                "eligible_pairs": 0,
                "pair_coverage": 0.0,
                "avg_tile_fill_before_packing": 0.0,
                "avg_bucket_fill_after_packing": 0.0,
                "direct_segments": 0,
                "direct_eligible_segments": 0,
                "direct_total_pairs": 0,
                "direct_eligible_pairs": 0,
                "direct_pair_coverage": 0.0,
                "avg_direct_fill": 0.0,
                "failure_reasons": {},
                "top_failure_reason": "none",
            },
        }

    metadata = runtime.forward_synthetic_grid
    bucket_analysis = _analyze_bucket_like_segments(
        packed_q=metadata.bucket_packed_q.detach().cpu(),
        packed_k=metadata.bucket_packed_k.detach().cpu(),
        dense=metadata.bucket_dense.detach().cpu(),
        row_ptr=metadata.bucket_row_ptr.detach().cpu(),
        words_per_row=metadata.bucket_words_per_row.detach().cpu(),
        q_length=metadata.bucket_q_length,
        k_length=metadata.bucket_k_length,
        mask_word_row_ptr=metadata.bucket_mask_word_row_ptr.detach().cpu(),
        mask_words=metadata.bucket_mask_words,
        group_size=group_size,
        nnz_per_group=nnz_per_group,
        label_prefix="synthetic_bucket",
        bucket_fill=metadata.bucket_fill.detach().cpu(),
    )

    direct_plan = metadata.forward_execution_plan.get("direct_execution_plan") if metadata.forward_execution_plan else None
    if direct_plan is not None:
        direct_dense = direct_plan["bucket_dense"]
        direct_analysis = _analyze_bucket_like_segments(
            packed_q=direct_plan["bucket_packed_q"],
            packed_k=direct_plan["bucket_packed_k"],
            dense=direct_dense,
            row_ptr=[0, *torch.tensor(direct_plan["bucket_size"], dtype=torch.int32).cumsum(0).tolist()],
            words_per_row=direct_plan["bucket_words_per_row"],
            q_length=direct_plan["bucket_q_length"],
            k_length=direct_plan["bucket_k_length"],
            mask_word_row_ptr=[0, *torch.tensor(
                [
                    end - start
                    for start, end in direct_plan["bucket_mask_word_range"]
                ],
                dtype=torch.int32,
            ).cumsum(0).tolist()],
            mask_words=direct_plan["bucket_mask_words"],
            group_size=group_size,
            nnz_per_group=nnz_per_group,
            label_prefix="synthetic_direct",
            bucket_fill=direct_plan["bucket_fill"],
        )
    else:
        direct_analysis = {"segments": [], "per_bucket": [], "summary": _summarize_segments([])}

    failure_reasons: Counter[str] = Counter()
    for source in (bucket_analysis["summary"], direct_analysis["summary"]):
        for reason, count in source["failure_reasons"].items():
            failure_reasons[reason] += int(count)

    summary = {
        "segments": int(bucket_analysis["summary"]["segments"]),
        "eligible_segments": int(bucket_analysis["summary"]["eligible_segments"]),
        "total_pairs": int(bucket_analysis["summary"]["total_pairs"]),
        "eligible_pairs": int(bucket_analysis["summary"]["eligible_pairs"]),
        "pair_coverage": float(bucket_analysis["summary"]["pair_coverage"]),
        "avg_tile_fill_before_packing": float(metadata.tile_fill.float().mean().item()) if metadata.tile_fill is not None and metadata.tile_fill.numel() > 0 else 0.0,
        "avg_bucket_fill_after_packing": float(metadata.bucket_fill.float().mean().item()) if metadata.bucket_fill is not None and metadata.bucket_fill.numel() > 0 else 0.0,
        "direct_segments": int(direct_analysis["summary"]["segments"]),
        "direct_eligible_segments": int(direct_analysis["summary"]["eligible_segments"]),
        "direct_total_pairs": int(direct_analysis["summary"]["total_pairs"]),
        "direct_eligible_pairs": int(direct_analysis["summary"]["eligible_pairs"]),
        "direct_pair_coverage": float(direct_analysis["summary"]["pair_coverage"]),
        "avg_direct_fill": float(direct_analysis["summary"].get("avg_reported_fill", 0.0)),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "top_failure_reason": _top_reason(failure_reasons),
    }

    return {
        "enabled": True,
        "bucket_analysis": bucket_analysis,
        "direct_analysis": direct_analysis,
        "summary": summary,
    }


def analyze_hsa_sparse24_feasibility(
    schedule,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    use_synthetic_grid: bool = True,
    group_size: int = 4,
    nnz_per_group: int = 2,
) -> dict[str, Any]:
    hsa_module = _load_hsa_module()
    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    if use_synthetic_grid:
        hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)

    physical = _analyze_physical_forward_tiles(
        schedule,
        runtime,
        group_size=group_size,
        nnz_per_group=nnz_per_group,
    )
    synthetic = _analyze_synthetic_forward_segments(
        runtime,
        use_synthetic_grid=use_synthetic_grid,
        group_size=group_size,
        nnz_per_group=nnz_per_group,
    )

    return {
        "group_size": group_size,
        "nnz_per_group": nnz_per_group,
        "use_synthetic_grid": bool(use_synthetic_grid),
        "physical": physical,
        "synthetic": synthetic,
    }


def summarize_hsa_sparse24_feasibility(report: dict[str, Any]) -> dict[str, Any]:
    physical_summary = report["physical"]["summary"]
    synthetic_summary = report["synthetic"]["summary"]

    summary = {
        "group_size": int(report["group_size"]),
        "nnz_per_group": int(report["nnz_per_group"]),
        "physical_segments": int(physical_summary["segments"]),
        "physical_partial_segments": int(physical_summary.get("partial_segments", 0)),
        "physical_full_segments": int(physical_summary.get("full_segments", 0)),
        "physical_eligible_segments": int(physical_summary["eligible_segments"]),
        "physical_total_pairs": int(physical_summary["total_pairs"]),
        "physical_eligible_pairs": int(physical_summary["eligible_pairs"]),
        "physical_pair_coverage": float(physical_summary["pair_coverage"]),
        "physical_avg_fill": float(physical_summary["avg_fill"]),
        "physical_top_failure_reason": physical_summary["top_failure_reason"],
        "synthetic_segments": int(synthetic_summary["segments"]),
        "synthetic_eligible_segments": int(synthetic_summary["eligible_segments"]),
        "synthetic_total_pairs": int(synthetic_summary["total_pairs"]),
        "synthetic_eligible_pairs": int(synthetic_summary["eligible_pairs"]),
        "synthetic_pair_coverage": float(synthetic_summary["pair_coverage"]),
        "synthetic_avg_tile_fill_before_packing": float(synthetic_summary["avg_tile_fill_before_packing"]),
        "synthetic_avg_bucket_fill_after_packing": float(synthetic_summary["avg_bucket_fill_after_packing"]),
        "synthetic_top_failure_reason": synthetic_summary["top_failure_reason"],
        "synthetic_direct_segments": int(synthetic_summary["direct_segments"]),
        "synthetic_direct_eligible_segments": int(synthetic_summary["direct_eligible_segments"]),
        "synthetic_direct_total_pairs": int(synthetic_summary["direct_total_pairs"]),
        "synthetic_direct_eligible_pairs": int(synthetic_summary["direct_eligible_pairs"]),
        "synthetic_direct_pair_coverage": float(synthetic_summary["direct_pair_coverage"]),
        "synthetic_avg_direct_fill": float(synthetic_summary["avg_direct_fill"]),
    }

    for prefix, reasons in (
        ("physical", physical_summary["failure_reasons"]),
        ("synthetic", synthetic_summary["failure_reasons"]),
    ):
        for reason in (
            "row_dependent_columns",
            "requires_output_mask",
            "gt2_per_group",
            "unaligned_tail",
            "dense_segment",
        ):
            summary[f"{prefix}_{reason}"] = int(reasons.get(reason, 0))

    return summary


__all__ = [
    "analyze_hsa_sparse24_feasibility",
    "summarize_hsa_sparse24_feasibility",
]
