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
    for bucket_idx, q_count in enumerate(q_length.detach().cpu().tolist()):
        valid_q = int(q_count)
        if valid_q <= 0:
            continue
        target_rows = q_row_idx[bucket_idx, :valid_q].long()
        scattered.index_copy_(0, target_rows, packed_out[bucket_idx, :valid_q].float())
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

    q_buf = torch.zeros((num_buckets, packed_q, heads, head_dim), dtype=dtype, device=device)
    k_buf = torch.randn((num_buckets, support_k, heads, head_dim), dtype=dtype, device=device, generator=generator)
    v_buf = torch.randn((num_buckets, support_k, heads, head_dim), dtype=dtype, device=device, generator=generator)
    mask_bool = torch.zeros((num_buckets, packed_q, support_k), dtype=torch.bool, device=device)
    q_length = torch.zeros((num_buckets,), dtype=torch.int32, device=device)
    k_length = torch.full((num_buckets,), support_k, dtype=torch.int32, device=device)
    q_row_idx = torch.full((num_buckets, packed_q), -1, dtype=torch.int32, device=device)

    live_per_row = min(support_k, islands_per_row * island_width)
    global_row = 0
    for bucket_idx in range(num_buckets):
        valid_q = min(packed_q, seqlen - bucket_idx * packed_q)
        q_length[bucket_idx] = valid_q
        for row_idx in range(valid_q):
            q_buf[bucket_idx, row_idx] = torch.randn(
                (heads, head_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            q_row_idx[bucket_idx, row_idx] = global_row
            if case_family == "disjoint_confetti":
                row_mask = _build_disjoint_confetti_row_mask(
                    support_k=support_k,
                    islands_per_row=islands_per_row,
                    island_width=island_width,
                    row_seed=global_row,
                    row_shift=row_shift,
                )
            else:
                row_mask = _build_compact_control_row_mask(
                    support_k=support_k,
                    live_per_row=live_per_row,
                    row_seed=global_row,
                    row_shift=max(1, row_shift // 2),
                )
            mask_bool[bucket_idx, row_idx] = row_mask.to(device=device)
            global_row += 1

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


def _run_direct_2d_bucket_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
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
    return _flatten_valid_packed_rows(packed_out, bucket["custom_q_length"])


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
    valid_variants = {"dense", "custom_masked", "fa4_packed", "direct_2d", "shared_support"}
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
    fa4_ms = results.get("fa4_packed", {}).get("fwd_ms")
    for payload in results.values():
        if isinstance(dense_ms, float) and dense_ms > 0 and isinstance(payload.get("fwd_ms"), float) and payload["fwd_ms"] > 0:
            payload["speedup_vs_dense"] = dense_ms / payload["fwd_ms"]
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
    return summary
