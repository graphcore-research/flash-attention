from __future__ import annotations

import math
import time
from collections import defaultdict
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


def _encode_mask_rows_to_words(mask_rows: torch.Tensor) -> torch.Tensor:
    if mask_rows.ndim != 2:
        raise ValueError("expected a rank-2 boolean mask")
    rows, cols = mask_rows.shape
    words_per_row = (cols + 31) // 32
    if words_per_row <= 0:
        return torch.empty((rows, 0), dtype=torch.int32, device=mask_rows.device)
    out = torch.zeros((rows, words_per_row), dtype=torch.int32, device=mask_rows.device)
    for word_idx in range(words_per_row):
        col_start = word_idx * 32
        col_end = min(col_start + 32, cols)
        if col_end <= col_start:
            continue
        bits = mask_rows[:, col_start:col_end].to(dtype=torch.int64)
        bit_weights = (1 << torch.arange(col_end - col_start, device=mask_rows.device, dtype=torch.int64)).view(1, -1)
        out[:, word_idx] = (bits * bit_weights).sum(dim=1).to(dtype=torch.int32)
    return out.contiguous()


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
        except Exception as exc:  # pragma: no cover - backend dependent
            last_exc = exc
    if last_exc is None:
        raise RuntimeError("failed to compress sparse24 operand")
    raise RuntimeError(f"failed to compress sparse24 operand: {last_exc}") from last_exc


def _split_dense_matrix_main_and_residual(
    x: torch.Tensor,
    *,
    group_size: int = 4,
    nnz_per_group: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    main = view.clone()
    residual = view.clone()
    main.masked_fill_(~keep_mask, 0)
    residual.masked_fill_(keep_mask, 0)
    return main.reshape(rows, cols), residual.reshape(rows, cols)


def _dense_scores(q_hqd: torch.Tensor, k_hkd: torch.Tensor) -> torch.Tensor:
    return torch.bmm(q_hqd, k_hkd.transpose(1, 2))


def _masked_forward_from_scores(bucket: dict[str, Any], scores_hqk: torch.Tensor, *, softmax_scale: float) -> torch.Tensor:
    scaled = scores_hqk.float() * float(softmax_scale)
    scaled = scaled.masked_fill(~bucket["mask_hqk"], float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.bmm(probs, bucket["v_hkd_float"])


def _tensor_int_tuple(x: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in x.detach().cpu().tolist())


def _support_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(entry["packed_q"]),
        int(entry["q_count"]),
        _tensor_int_tuple(entry["union_rows"]),
        tuple(_word_to_u32(word) for word in entry["mask_words"].detach().cpu().tolist()),
    )


def _make_support_group(entries: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(entries, key=lambda entry: int(entry["qgroup_idx"]))
    support_row_set: set[int] = set()
    for entry in members:
        support_row_set.update(int(row_idx) for row_idx in entry["union_rows"].detach().cpu().tolist())
    qgroup_ids = [int(entry["qgroup_idx"]) for entry in members]
    return {
        "packed_q": int(members[0]["packed_q"]),
        "members": members,
        "qgroup_ids": qgroup_ids,
        "support_rows": tuple(sorted(support_row_set)),
        "support_row_set": frozenset(support_row_set),
    }


def _group_qgroups_by_exact_support(
    entries: list[dict[str, Any]],
    *,
    max_qgroups_per_bucket: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in sorted(entries, key=lambda item: int(item["qgroup_idx"])):
        grouped[_support_signature(entry)].append(entry)

    groups: list[dict[str, Any]] = []
    for members in grouped.values():
        members = sorted(members, key=lambda item: int(item["qgroup_idx"]))
        for chunk_start in range(0, len(members), max_qgroups_per_bucket):
            groups.append(_make_support_group(members[chunk_start:chunk_start + max_qgroups_per_bucket]))
    groups.sort(key=lambda group: (group["packed_q"], group["qgroup_ids"][0]))
    return groups


def _candidate_merge_key(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    min_jaccard: float,
    max_support_rows: int,
    max_qgroups_per_bucket: int,
) -> tuple[Any, ...] | None:
    if left["packed_q"] != right["packed_q"]:
        return None
    merged_qgroup_ids = sorted(list(left["qgroup_ids"]) + list(right["qgroup_ids"]))
    if len(merged_qgroup_ids) > max_qgroups_per_bucket:
        return None
    union_set = left["support_row_set"] | right["support_row_set"]
    union_len = len(union_set)
    if union_len <= 0 or union_len > max_support_rows:
        return None
    intersection_len = len(left["support_row_set"] & right["support_row_set"])
    jaccard = intersection_len / union_len
    if jaccard < min_jaccard:
        return None
    return (
        jaccard,
        len(merged_qgroup_ids),
        -union_len,
        -merged_qgroup_ids[0],
        tuple(-qgroup_idx for qgroup_idx in merged_qgroup_ids),
    )


def _greedy_merge_support_groups(
    groups: list[dict[str, Any]],
    *,
    min_jaccard: float,
    max_support_rows: int,
    max_qgroups_per_bucket: int,
) -> list[dict[str, Any]]:
    working = [_make_support_group(group["members"]) for group in groups]
    working.sort(key=lambda group: (group["packed_q"], group["qgroup_ids"][0]))

    while True:
        best_pair: tuple[int, int] | None = None
        best_key: tuple[Any, ...] | None = None
        for left_idx in range(len(working)):
            for right_idx in range(left_idx + 1, len(working)):
                candidate_key = _candidate_merge_key(
                    working[left_idx],
                    working[right_idx],
                    min_jaccard=min_jaccard,
                    max_support_rows=max_support_rows,
                    max_qgroups_per_bucket=max_qgroups_per_bucket,
                )
                if candidate_key is None:
                    continue
                if best_key is None or candidate_key > best_key:
                    best_key = candidate_key
                    best_pair = (left_idx, right_idx)
        if best_pair is None:
            break

        left_idx, right_idx = best_pair
        merged_group = _make_support_group(working[left_idx]["members"] + working[right_idx]["members"])
        working[left_idx] = merged_group
        del working[right_idx]
        working.sort(key=lambda group: (group["packed_q"], group["qgroup_ids"][0]))
    return working


def _coarse_window_merge_support_groups(
    groups: list[dict[str, Any]],
    *,
    max_support_rows: int,
    max_qgroups_per_bucket: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    working = [_make_support_group(group["members"]) for group in groups]
    working.sort(key=lambda group: (group["packed_q"], group["qgroup_ids"][0]))

    start_idx = 0
    while start_idx < len(working):
        seed = working[start_idx]
        current_members = list(seed["members"])
        current_support = set(seed["support_row_set"])
        current_packed_q = int(seed["packed_q"])
        end_idx = start_idx + 1

        while end_idx < len(working):
            candidate = working[end_idx]
            if int(candidate["packed_q"]) != current_packed_q:
                break
            candidate_members = list(candidate["members"])
            if len(current_members) + len(candidate_members) > max_qgroups_per_bucket:
                break
            candidate_support = current_support | set(candidate["support_row_set"])
            if len(candidate_support) > max_support_rows:
                break
            current_members.extend(candidate_members)
            current_support = candidate_support
            end_idx += 1

        merged.append(_make_support_group(current_members))
        start_idx = end_idx
    return merged


def _sparse_scores_from_heads(
    sparse_heads: list[Any],
    q_rhs_cols: torch.Tensor,
    *,
    support_rows: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for head_idx, sparse_operand in enumerate(sparse_heads):
        head_scores = sparse_operand @ q_rhs_cols[head_idx]
        scores.append(head_scores[:support_rows].transpose(0, 1).contiguous())
    return torch.stack(scores, dim=0)


def _dense_scores_from_heads(
    dense_heads: list[torch.Tensor],
    q_rhs_cols: torch.Tensor,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for head_idx, dense_operand in enumerate(dense_heads):
        head_scores = dense_operand @ q_rhs_cols[head_idx]
        scores.append(head_scores.transpose(0, 1).contiguous())
    return torch.stack(scores, dim=0)


def _validated_scores_from_heads(
    *,
    dense_heads: list[torch.Tensor],
    sparse_heads: list[Any],
    sparse_valid: list[bool],
    q_rhs_cols: torch.Tensor,
    support_rows: int,
) -> torch.Tensor:
    scores: list[torch.Tensor] = []
    for head_idx, dense_operand in enumerate(dense_heads):
        if sparse_valid[head_idx]:
            head_scores = sparse_heads[head_idx] @ q_rhs_cols[head_idx]
            scores.append(head_scores[:support_rows].transpose(0, 1).contiguous())
        else:
            head_scores = dense_operand @ q_rhs_cols[head_idx]
            scores.append(head_scores.transpose(0, 1).contiguous())
    return torch.stack(scores, dim=0)


def _combine_packed_out_lse(
    acc_out: torch.Tensor | None,
    acc_lse: torch.Tensor | None,
    tile_out: torch.Tensor,
    tile_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if acc_out is None or acc_lse is None:
        return tile_out.clone(), tile_lse.clone()
    combined_lse = torch.logaddexp(acc_lse, tile_lse)
    acc_valid = torch.isfinite(acc_lse)
    tile_valid = torch.isfinite(tile_lse)
    acc_scale = torch.where(acc_valid, torch.exp(acc_lse - combined_lse), torch.zeros_like(combined_lse))
    tile_scale = torch.where(tile_valid, torch.exp(tile_lse - combined_lse), torch.zeros_like(combined_lse))
    combined_out = acc_out * acc_scale.unsqueeze(-1) + tile_out * tile_scale.unsqueeze(-1)
    combined_out = torch.where(
        torch.isfinite(combined_lse).unsqueeze(-1),
        combined_out,
        torch.zeros_like(combined_out),
    )
    return combined_out, combined_lse


def _run_custom_masked_bucket_forward(
    bucket: dict[str, Any],
    *,
    softmax_scale: float,
    tile_k: int = 32,
) -> torch.Tensor:
    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
        _can_use_synthetic_micro_fwd,
        _run_synthetic_micro_fwd_masked_kernel,
    )

    q_buf = bucket["custom_q_buf"]
    k_buf = bucket["custom_k_buf"]
    v_buf = bucket["custom_v_buf"]
    packed_q = int(bucket["packed_q"])
    packed_k = int(bucket["support_rows"])
    if tile_k <= 0 or tile_k > 32:
        raise ValueError("custom masked tile_k must be in [1, 32]")
    if not _can_use_synthetic_micro_fwd(q_buf, k_buf[:, : min(packed_k, tile_k)], v_buf[:, : min(packed_k, tile_k)], packed_q=packed_q, packed_k=min(packed_k, tile_k)):
        raise RuntimeError("custom_masked_micro_unsupported")
    mask_bool = bucket["custom_mask_bool"]
    packed_out = None
    packed_lse = None
    for tile_start in range(0, packed_k, tile_k):
        tile_end = min(tile_start + tile_k, packed_k)
        tile_width = tile_end - tile_start
        tile_mask_words = _encode_mask_rows_to_words(
            mask_bool[:, :, tile_start:tile_end].reshape(mask_bool.shape[0] * mask_bool.shape[1], tile_width)
        ).view(mask_bool.shape[0], mask_bool.shape[1], -1)
        tile_k_length = torch.full(
            bucket["custom_k_length"].shape,
            tile_width,
            dtype=bucket["custom_k_length"].dtype,
            device=bucket["custom_k_length"].device,
        )
        tile_out, tile_lse = _run_synthetic_micro_fwd_masked_kernel(
            q_buf,
            k_buf[:, tile_start:tile_end].contiguous(),
            v_buf[:, tile_start:tile_end].contiguous(),
            bucket["custom_q_length"],
            tile_k_length,
            tile_mask_words.contiguous(),
            softmax_scale=softmax_scale,
        )
        packed_out, packed_lse = _combine_packed_out_lse(packed_out, packed_lse, tile_out, tile_lse)

    assert packed_out is not None

    out_rows: list[torch.Tensor] = []
    for qgroup_idx, q_count in enumerate(bucket["custom_q_length"].detach().cpu().tolist()):
        valid_q = int(q_count)
        if valid_q > 0:
            out_rows.append(packed_out[qgroup_idx, :valid_q].contiguous())
    if not out_rows:
        return torch.empty(
            (q_buf.shape[2], 0, v_buf.shape[3]),
            dtype=torch.float32,
            device=q_buf.device,
        )
    return torch.cat(out_rows, dim=0).permute(1, 0, 2).contiguous()


def _run_fa4_packed_bucket_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    import flash_attn.cute.hsa as hsa_module

    _flash_attn_fwd = hsa_module._lazy_cute_imports()[5]
    words_per_row = int(bucket["custom_mask_words"].shape[2])
    q_length = hsa_module._tag_aux_tensor(bucket["custom_q_length"], leading_dim=0)
    k_length = hsa_module._tag_aux_tensor(bucket["custom_k_length"], leading_dim=0)
    mask_words = hsa_module._tag_aux_tensor(bucket["custom_mask_words"], leading_dim=2)
    mask_mod = hsa_module.get_hsa_synthetic_packed_bitmap_mask_mod(words_per_row)
    out_bucket, _ = _flash_attn_fwd(
        bucket["custom_q_buf"],
        bucket["custom_k_buf"],
        bucket["custom_v_buf"],
        softmax_scale=softmax_scale,
        causal=False,
        m_block_size=128,
        n_block_size=128,
        pack_gqa=False,
        mask_mod=mask_mod,
        aux_tensors=[q_length, k_length, mask_words],
        return_lse=True,
    )
    out_rows: list[torch.Tensor] = []
    for qgroup_idx, q_count in enumerate(bucket["custom_q_length"].detach().cpu().tolist()):
        valid_q = int(q_count)
        if valid_q > 0:
            out_rows.append(out_bucket[qgroup_idx, :valid_q].float().contiguous())
    if not out_rows:
        return torch.empty(
            (bucket["custom_q_buf"].shape[2], 0, bucket["custom_v_buf"].shape[3]),
            dtype=torch.float32,
            device=bucket["custom_q_buf"].device,
        )
    return torch.cat(out_rows, dim=0).permute(1, 0, 2).contiguous()


def _run_shared_cta_bucket_forward(bucket: dict[str, Any], *, softmax_scale: float) -> torch.Tensor:
    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _run_shared_support_bucket_fwd_kernel

    out_bucket, _ = _run_shared_support_bucket_fwd_kernel(
        bucket["shared_cta_q_buf"],
        bucket["shared_k_expanded"],
        bucket["shared_v_expanded"],
        bucket["shared_cta_q_length"],
        bucket["shared_cta_mask_words"],
        softmax_scale=softmax_scale,
        qgroups_per_cta=int(bucket.get("shared_cta_qgroups_per_cta", 2)),
    )
    out_rows: list[torch.Tensor] = []
    for qgroup_idx, q_count in enumerate(bucket["shared_cta_q_length"].detach().cpu().tolist()):
        valid_q = int(q_count)
        if valid_q > 0:
            out_rows.append(out_bucket[qgroup_idx, :valid_q].float().contiguous())
    if not out_rows:
        return torch.empty(
            (bucket["shared_cta_q_buf"].shape[2], 0, bucket["shared_v_expanded"].shape[2]),
            dtype=torch.float32,
            device=bucket["shared_cta_q_buf"].device,
        )
    return torch.cat(out_rows, dim=0).permute(1, 0, 2).contiguous()


def _materialize_shared_support_buckets(
    schedule,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group_size: int,
    nnz_per_group: int,
    bucketizer: str,
    min_jaccard: float,
    max_support_rows: int,
    max_qgroups_per_bucket: int,
    max_buckets: int | None,
    max_qgroups: int | None,
    backend_validation_atol: float,
    include_sparse_payload: bool = True,
) -> dict[str, Any]:
    import flash_attn.cute.hsa as hsa_module

    runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    metadata = runtime.forward_synthetic_grid
    if metadata is None or metadata.forward_execution_plan is None:
        return {"status": "missing_forward_execution_plan", "buckets": []}

    support_plan = metadata.forward_execution_plan.get("qgroup_union_support_plan")
    if support_plan is None:
        return {"status": "missing_qgroup_union_support_plan", "buckets": []}

    q_flat = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3]).contiguous()
    k_flat = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3]).contiguous()
    v_flat = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3]).contiguous()

    total_qgroups = int(metadata.qgroup_length.numel())
    qgroup_limit = total_qgroups if max_qgroups is None else min(total_qgroups, max_qgroups)
    support_entries: list[dict[str, Any]] = []
    for qgroup_idx in range(qgroup_limit):
        q_count = int(metadata.qgroup_length[qgroup_idx])
        packed_q = int(metadata.qgroup_packed_q[qgroup_idx])
        union_k_length = int(support_plan["qgroup_union_k_length"][qgroup_idx])
        if q_count <= 0 or packed_q <= 0 or union_k_length <= 0:
            continue
        q_row_start, q_row_end = support_plan["qgroup_row_range"][qgroup_idx]
        union_row_start, union_row_end = support_plan["qgroup_union_k_row_range"][qgroup_idx]
        mask_word_start, mask_word_end = support_plan["qgroup_mask_word_range"][qgroup_idx]
        support_entries.append(
            {
                "qgroup_idx": int(qgroup_idx),
                "packed_q": packed_q,
                "q_count": q_count,
                "q_rows": metadata.qgroup_rows[q_row_start:q_row_end].contiguous(),
                "union_rows": support_plan["qgroup_union_k_row_idx"][
                    union_row_start : union_row_start + union_k_length
                ].contiguous(),
                "union_k_length": union_k_length,
                "mask_words": support_plan["qgroup_mask_words"][mask_word_start:mask_word_end].contiguous(),
                "words_per_row": int(support_plan["qgroup_words_per_row"][qgroup_idx]),
                "allowed_pairs": int(support_plan["qgroup_allowed_pairs"][qgroup_idx]),
                "fill": float(support_plan["qgroup_fill"][qgroup_idx]),
            }
        )

    if not support_entries:
        return {
            "status": "no_qgroup_support_entries",
            "buckets": [],
            "available_qgroups": total_qgroups,
            "sampled_qgroups": 0,
        }

    exact_groups = _group_qgroups_by_exact_support(
        support_entries,
        max_qgroups_per_bucket=max_qgroups_per_bucket,
    )
    if bucketizer == "coarse_window":
        merged_groups = _coarse_window_merge_support_groups(
            exact_groups,
            max_support_rows=max_support_rows,
            max_qgroups_per_bucket=max_qgroups_per_bucket,
        )
    else:
        merged_groups = _greedy_merge_support_groups(
            exact_groups,
            min_jaccard=min_jaccard,
            max_support_rows=max_support_rows,
            max_qgroups_per_bucket=max_qgroups_per_bucket,
        )
    if max_buckets is not None:
        merged_groups = merged_groups[:max_buckets]

    buckets: list[dict[str, Any]] = []
    retained_abs_sum = 0.0
    dense_abs_sum = 0.0
    retained_sq_sum = 0.0
    dense_sq_sum = 0.0
    sampled_qgroups = 0
    total_heads = 0
    valid_main_heads = 0
    valid_residual_heads = 0
    entry_by_qgroup = {int(entry["qgroup_idx"]): entry for entry in support_entries}

    for group in merged_groups:
        member_entries = [entry_by_qgroup[qgroup_idx] for qgroup_idx in group["qgroup_ids"]]
        packed_q = int(group["packed_q"])
        support_rows = torch.tensor(group["support_rows"], dtype=torch.long, device=q.device)
        support_rows_count = int(support_rows.numel())
        if support_rows_count <= 0:
            continue
        if support_rows_count > max_support_rows:
            continue
        if q.shape[-1] % group_size != 0:
            return {
                "status": "head_dim_not_divisible_by_group_size",
                "buckets": [],
                "available_qgroups": total_qgroups,
                "sampled_qgroups": 0,
            }

        support_slot = {int(row_idx): slot for slot, row_idx in enumerate(group["support_rows"])}
        q_row_tensors: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        custom_q_buf_parts: list[torch.Tensor] = []
        custom_mask_bool_parts: list[torch.Tensor] = []
        custom_mask_word_parts: list[torch.Tensor] = []
        custom_q_lengths: list[int] = []
        total_queries = 0
        for entry in member_entries:
            q_rows = entry["q_rows"][:entry["q_count"]].long()
            q_row_tensors.append(q_rows)
            total_queries += int(entry["q_count"])

            local_mask = _decode_mask_words(
                entry["mask_words"],
                rows=int(entry["packed_q"]),
                cols=int(entry["union_k_length"]),
                words_per_row=int(entry["words_per_row"]),
            )[: int(entry["q_count"])]
            local_to_shared = [support_slot[int(row_idx)] for row_idx in entry["union_rows"].detach().cpu().tolist()]
            member_mask = torch.zeros((int(entry["q_count"]), support_rows_count), dtype=torch.bool, device=q.device)
            for q_slot in range(int(entry["q_count"])):
                allowed_local = torch.nonzero(local_mask[q_slot], as_tuple=False).flatten().tolist()
                for local_idx in allowed_local:
                    member_mask[q_slot, local_to_shared[local_idx]] = True
            mask_rows.append(member_mask)

            entry_q_buf = torch.zeros(
                (packed_q, q.shape[2], q.shape[3]),
                dtype=q.dtype,
                device=q.device,
            )
            if int(entry["q_count"]) > 0:
                entry_q_buf[: int(entry["q_count"])] = q_flat.index_select(0, q_rows).contiguous()
            custom_q_buf_parts.append(entry_q_buf)

            entry_mask_padded = torch.zeros((packed_q, support_rows_count), dtype=torch.bool, device=q.device)
            entry_mask_padded[: int(entry["q_count"])] = member_mask
            custom_mask_bool_parts.append(entry_mask_padded)
            custom_mask_word_parts.append(_encode_mask_rows_to_words(entry_mask_padded))
            custom_q_lengths.append(int(entry["q_count"]))

        if total_queries <= 0:
            continue

        q_row_idx = torch.cat(q_row_tensors, dim=0)
        q_bucket = q_flat.index_select(0, q_row_idx).contiguous()
        k_bucket = k_flat.index_select(0, support_rows).contiguous()
        v_bucket = v_flat.index_select(0, support_rows).contiguous()
        k_expanded = hsa_module._expand_kv_to_q_heads(k_bucket, q_bucket.shape[1]).contiguous()
        v_expanded = hsa_module._expand_kv_to_q_heads(v_bucket, q_bucket.shape[1]).contiguous()
        q_hqd = q_bucket.permute(1, 0, 2).contiguous()
        k_hkd = k_expanded.permute(1, 0, 2).contiguous()
        v_hkd_float = v_expanded.permute(1, 0, 2).float().contiguous()
        mask_hqk = (
            torch.cat(mask_rows, dim=0)
            .unsqueeze(0)
            .expand(q_hqd.shape[0], -1, -1)
            .contiguous()
        )
        custom_q_buf = torch.stack(custom_q_buf_parts, dim=0).contiguous()
        custom_mask_bool = torch.stack(custom_mask_bool_parts, dim=0).contiguous()
        custom_k_buf = k_bucket.unsqueeze(0).expand(len(member_entries), -1, -1, -1).contiguous()
        custom_v_buf = v_bucket.unsqueeze(0).expand(len(member_entries), -1, -1, -1).contiguous()
        shared_k_expanded = k_expanded.contiguous()
        shared_v_expanded = v_expanded.contiguous()
        custom_mask_words = torch.stack(custom_mask_word_parts, dim=0).contiguous()
        custom_q_length = torch.tensor(custom_q_lengths, dtype=torch.int32, device=q.device)
        custom_k_length = torch.full(
            (len(member_entries),),
            support_rows_count,
            dtype=torch.int32,
            device=q.device,
        )
        shared_cta_qgroups = (total_queries + 1) // 2
        shared_cta_q_buf = torch.zeros(
            (shared_cta_qgroups, 2, q.shape[2], q.shape[3]),
            dtype=q.dtype,
            device=q.device,
        )
        shared_cta_mask_bool = torch.zeros(
            (shared_cta_qgroups, 2, support_rows_count),
            dtype=torch.bool,
            device=q.device,
        )
        shared_cta_q_lengths: list[int] = []
        q_cursor = 0
        for qgroup_idx in range(shared_cta_qgroups):
            q_count = min(2, total_queries - q_cursor)
            if q_count > 0:
                shared_cta_q_buf[qgroup_idx, :q_count] = q_bucket[q_cursor : q_cursor + q_count]
                shared_cta_mask_bool[qgroup_idx, :q_count] = mask_hqk[0, q_cursor : q_cursor + q_count]
            shared_cta_q_lengths.append(int(q_count))
            q_cursor += q_count
        shared_cta_mask_words = _encode_mask_rows_to_words(
            shared_cta_mask_bool.view(shared_cta_qgroups * 2, support_rows_count)
        ).view(shared_cta_qgroups, 2, -1).contiguous()
        shared_cta_q_length = torch.tensor(shared_cta_q_lengths, dtype=torch.int32, device=q.device)

        main_sparse_heads: list[Any] = []
        residual_sparse_heads: list[Any] = []
        main_dense_heads: list[torch.Tensor] = []
        residual_dense_heads: list[torch.Tensor] = []
        main_sparse_valid: list[bool] = []
        residual_sparse_valid: list[bool] = []
        main_validated_scores_hqk_parts: list[torch.Tensor] = []
        residual_validated_scores_hqk_parts: list[torch.Tensor] = []
        if include_sparse_payload:
            for head_idx in range(k_hkd.shape[0]):
                main_dense, residual_dense = _split_dense_matrix_main_and_residual(
                    k_hkd[head_idx],
                    group_size=group_size,
                    nnz_per_group=nnz_per_group,
                )
                main_sparse, _ = _compress_sparse24_operand(main_dense)
                residual_sparse, _ = _compress_sparse24_operand(residual_dense)
                main_sparse_heads.append(main_sparse)
                residual_sparse_heads.append(residual_sparse)
                main_dense_heads.append(main_dense)
                residual_dense_heads.append(residual_dense)
                retained_abs_sum += main_dense.abs().sum().item()
                dense_abs_sum += k_hkd[head_idx].abs().sum().item()
                retained_sq_sum += main_dense.float().square().sum().item()
                dense_sq_sum += k_hkd[head_idx].float().square().sum().item()
                rhs = q_hqd[head_idx].transpose(0, 1)
                main_dense_scores = main_dense @ rhs
                main_sparse_scores = (main_sparse @ rhs)[:support_rows_count]
                main_sparse_scores_replay = (main_sparse @ rhs)[:support_rows_count]
                residual_dense_scores = residual_dense @ rhs
                residual_sparse_scores = (residual_sparse @ rhs)[:support_rows_count]
                residual_sparse_scores_replay = (residual_sparse @ rhs)[:support_rows_count]
                main_sparse_valid.append(
                    max(
                        float((main_dense_scores - main_sparse_scores).abs().max().item()),
                        float((main_dense_scores - main_sparse_scores_replay).abs().max().item()),
                        float((main_sparse_scores - main_sparse_scores_replay).abs().max().item()),
                    )
                    <= backend_validation_atol
                )
                residual_sparse_valid.append(
                    max(
                        float((residual_dense_scores - residual_sparse_scores).abs().max().item()),
                        float((residual_dense_scores - residual_sparse_scores_replay).abs().max().item()),
                        float((residual_sparse_scores - residual_sparse_scores_replay).abs().max().item()),
                    )
                    <= backend_validation_atol
                )
                main_validated_scores_hqk_parts.append(
                    (main_sparse_scores if main_sparse_valid[-1] else main_dense_scores)
                    .transpose(0, 1)
                    .contiguous()
                    .clone()
                )
                residual_validated_scores_hqk_parts.append(
                    (residual_sparse_scores if residual_sparse_valid[-1] else residual_dense_scores)
                    .transpose(0, 1)
                    .contiguous()
                    .clone()
                )

        sampled_qgroups += len(member_entries)
        total_heads += len(main_sparse_valid)
        valid_main_heads += sum(1 for is_valid in main_sparse_valid if is_valid)
        valid_residual_heads += sum(1 for is_valid in residual_sparse_valid if is_valid)
        buckets.append(
            {
                "group": group,
                "packed_q": packed_q,
                "q_hqd": q_hqd,
                "q_rhs_cols": q_hqd.transpose(1, 2).contiguous(),
                "k_hkd": k_hkd,
                "v_hkd_float": v_hkd_float,
                "mask_hqk": mask_hqk,
                "custom_q_buf": custom_q_buf,
                "custom_mask_bool": custom_mask_bool,
                "custom_k_buf": custom_k_buf,
                "custom_v_buf": custom_v_buf,
                "shared_k_expanded": shared_k_expanded,
                "shared_v_expanded": shared_v_expanded,
                "custom_mask_words": custom_mask_words,
                "custom_q_length": custom_q_length,
                "custom_k_length": custom_k_length,
                "shared_cta_q_buf": shared_cta_q_buf,
                "shared_cta_mask_words": shared_cta_mask_words,
                "shared_cta_q_length": shared_cta_q_length,
                "main_sparse_heads": main_sparse_heads,
                "residual_sparse_heads": residual_sparse_heads,
                "main_dense_heads": main_dense_heads,
                "residual_dense_heads": residual_dense_heads,
                "main_sparse_valid": main_sparse_valid,
                "residual_sparse_valid": residual_sparse_valid,
                "main_validated_scores_hqk": (
                    torch.stack(main_validated_scores_hqk_parts, dim=0)
                    if main_validated_scores_hqk_parts
                    else None
                ),
                "residual_validated_scores_hqk": (
                    torch.stack(residual_validated_scores_hqk_parts, dim=0)
                    if residual_validated_scores_hqk_parts
                    else None
                ),
                "num_queries": total_queries,
                "support_rows": support_rows_count,
                "num_qgroups": len(member_entries),
            }
        )

    return {
        "status": "measured" if buckets else "no_shared_support_buckets",
        "buckets": buckets,
        "available_qgroups": total_qgroups,
        "sampled_qgroups": sampled_qgroups,
        "retained_abs_sum": retained_abs_sum,
        "dense_abs_sum": dense_abs_sum,
        "retained_sq_sum": retained_sq_sum,
        "dense_sq_sum": dense_sq_sum,
        "exact_group_count": len(exact_groups),
        "merged_group_count": len(merged_groups),
        "total_heads": total_heads,
        "valid_main_heads": valid_main_heads,
        "valid_residual_heads": valid_residual_heads,
    }


def analyze_hsa_shared_sparse_gemm_forward(
    schedule,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    use_synthetic_grid: bool = True,
    group_size: int = 4,
    nnz_per_group: int = 2,
    min_jaccard: float = 0.5,
    max_support_rows: int = 128,
    max_qgroups_per_bucket: int = 32,
    max_buckets: int | None = None,
    max_qgroups: int | None = None,
    bucketizer: str = "exact_jaccard",
    backend_validation_atol: float = 0.5,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    enable_sparse_payload: bool = True,
    enable_custom_masked: bool = True,
    enable_fa4_packed: bool = True,
    enable_shared_cta: bool = True,
    shared_cta_qgroups_per_cta: int = 2,
) -> dict[str, Any]:
    if not use_synthetic_grid:
        return {
            "status": "requires_synthetic_grid",
            "group_size": group_size,
            "nnz_per_group": nnz_per_group,
        }
    if bucketizer not in {"exact_jaccard", "coarse_window"}:
        return {
            "status": "unsupported_bucketizer",
            "group_size": group_size,
            "nnz_per_group": nnz_per_group,
            "bucketizer": str(bucketizer),
        }

    try:
        materialized = _materialize_shared_support_buckets(
            schedule,
            q,
            k,
            v,
            group_size=group_size,
            nnz_per_group=nnz_per_group,
            bucketizer=bucketizer,
            min_jaccard=min_jaccard,
            max_support_rows=max_support_rows,
            max_qgroups_per_bucket=max_qgroups_per_bucket,
            max_buckets=max_buckets,
            max_qgroups=max_qgroups,
            backend_validation_atol=backend_validation_atol,
            include_sparse_payload=enable_sparse_payload,
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
        "bucketizer": str(bucketizer),
        "min_jaccard": float(min_jaccard),
        "max_support_rows": int(max_support_rows),
        "max_qgroups_per_bucket": int(max_qgroups_per_bucket),
        "backend_validation_atol": float(backend_validation_atol),
        "warmup_iters": int(warmup_iters),
        "benchmark_iters": int(benchmark_iters),
        "enable_sparse_payload": bool(enable_sparse_payload),
        "enable_custom_masked": bool(enable_custom_masked),
        "enable_fa4_packed": bool(enable_fa4_packed),
        "enable_shared_cta": bool(enable_shared_cta),
        "shared_cta_qgroups_per_cta": int(shared_cta_qgroups_per_cta),
        "bucket_count": len(materialized.get("buckets", [])),
        "available_qgroups": int(materialized.get("available_qgroups", 0)),
        "sampled_qgroups": int(materialized.get("sampled_qgroups", 0)),
        "exact_group_count": int(materialized.get("exact_group_count", 0)),
        "merged_group_count": int(materialized.get("merged_group_count", 0)),
        "total_heads": int(materialized.get("total_heads", 0)),
        "valid_main_heads": int(materialized.get("valid_main_heads", 0)),
        "valid_residual_heads": int(materialized.get("valid_residual_heads", 0)),
    }
    if materialized["status"] != "measured":
        return report

    buckets = materialized["buckets"]
    for bucket in buckets:
        bucket["shared_cta_qgroups_per_cta"] = int(shared_cta_qgroups_per_cta)
    if not buckets:
        report["status"] = "no_shared_support_buckets"
        return report

    softmax_scale = q.shape[-1] ** (-0.5)
    avg_queries_per_bucket = sum(bucket["num_queries"] for bucket in buckets) / len(buckets)
    avg_support_rows = sum(bucket["support_rows"] for bucket in buckets) / len(buckets)
    max_support_rows_seen = max(bucket["support_rows"] for bucket in buckets)
    retained_abs_sum = float(materialized["retained_abs_sum"])
    dense_abs_sum = float(materialized["dense_abs_sum"])
    retained_sq_sum = float(materialized["retained_sq_sum"])
    dense_sq_sum = float(materialized["dense_sq_sum"])

    main_output_max_diff = 0.0
    main_output_sum_abs_diff = 0.0
    exact_output_max_diff = 0.0
    exact_output_sum_abs_diff = 0.0
    custom_masked_status = "not_run"
    custom_masked_output_max_diff = 0.0
    custom_masked_output_sum_abs_diff = 0.0
    custom_masked_output_elements = 0
    fa4_packed_status = "not_run"
    fa4_packed_output_max_diff = 0.0
    fa4_packed_output_sum_abs_diff = 0.0
    fa4_packed_output_elements = 0
    shared_cta_status = "not_run"
    shared_cta_output_max_diff = 0.0
    shared_cta_output_sum_abs_diff = 0.0
    shared_cta_output_elements = 0
    main_output_elements = 0
    exact_output_elements = 0

    for bucket in buckets:
        dense_scores = _dense_scores(bucket["q_hqd"], bucket["k_hkd"])
        dense_out = _masked_forward_from_scores(bucket, dense_scores, softmax_scale=softmax_scale)
        if enable_sparse_payload:
            main_scores = bucket["main_validated_scores_hqk"]
            exact_scores = main_scores + bucket["residual_validated_scores_hqk"]
            main_out = _masked_forward_from_scores(bucket, main_scores, softmax_scale=softmax_scale)
            exact_out = _masked_forward_from_scores(bucket, exact_scores, softmax_scale=softmax_scale)

            main_diff = (dense_out - main_out).abs()
            exact_diff = (dense_out - exact_out).abs()
            main_output_max_diff = max(main_output_max_diff, float(main_diff.max().item()))
            main_output_sum_abs_diff += float(main_diff.sum().item())
            main_output_elements += main_diff.numel()
            exact_output_max_diff = max(exact_output_max_diff, float(exact_diff.max().item()))
            exact_output_sum_abs_diff += float(exact_diff.sum().item())
            exact_output_elements += exact_diff.numel()
        if enable_custom_masked and custom_masked_status in {"not_run", "measured"}:
            try:
                custom_out = _run_custom_masked_bucket_forward(bucket, softmax_scale=softmax_scale)
                custom_diff = (dense_out - custom_out).abs()
                custom_masked_status = "measured"
                custom_masked_output_max_diff = max(
                    custom_masked_output_max_diff,
                    float(custom_diff.max().item()) if custom_diff.numel() > 0 else 0.0,
                )
                custom_masked_output_sum_abs_diff += float(custom_diff.sum().item())
                custom_masked_output_elements += custom_diff.numel()
            except Exception as exc:  # pragma: no cover - benchmark-only failure path
                custom_masked_status = f"custom_masked_failed_{type(exc).__name__}"
        if enable_fa4_packed and fa4_packed_status in {"not_run", "measured"}:
            try:
                fa4_out = _run_fa4_packed_bucket_forward(bucket, softmax_scale=softmax_scale)
                fa4_diff = (dense_out - fa4_out).abs()
                fa4_packed_status = "measured"
                fa4_packed_output_max_diff = max(
                    fa4_packed_output_max_diff,
                    float(fa4_diff.max().item()) if fa4_diff.numel() > 0 else 0.0,
                )
                fa4_packed_output_sum_abs_diff += float(fa4_diff.sum().item())
                fa4_packed_output_elements += fa4_diff.numel()
            except Exception as exc:  # pragma: no cover - benchmark-only failure path
                fa4_packed_status = f"fa4_packed_failed_{type(exc).__name__}"
        if enable_shared_cta and shared_cta_status in {"not_run", "measured"}:
            try:
                shared_cta_out = _run_shared_cta_bucket_forward(bucket, softmax_scale=softmax_scale)
                shared_cta_diff = (dense_out - shared_cta_out).abs()
                shared_cta_status = "measured"
                shared_cta_output_max_diff = max(
                    shared_cta_output_max_diff,
                    float(shared_cta_diff.max().item()) if shared_cta_diff.numel() > 0 else 0.0,
                )
                shared_cta_output_sum_abs_diff += float(shared_cta_diff.sum().item())
                shared_cta_output_elements += shared_cta_diff.numel()
            except Exception as exc:  # pragma: no cover - benchmark-only failure path
                shared_cta_status = f"shared_cta_failed_{type(exc).__name__}"

    def _run_dense_qk():
        last = None
        for bucket in buckets:
            last = _dense_scores(bucket["q_hqd"], bucket["k_hkd"])
        return last

    def _run_sparse_main_qk():
        last = None
        for bucket in buckets:
            last = _validated_scores_from_heads(
                dense_heads=bucket["main_dense_heads"],
                sparse_heads=bucket["main_sparse_heads"],
                sparse_valid=bucket["main_sparse_valid"],
                q_rhs_cols=bucket["q_rhs_cols"],
                support_rows=bucket["support_rows"],
            )
        return last

    def _run_sparse_exact_qk():
        last = None
        for bucket in buckets:
            last = _validated_scores_from_heads(
                dense_heads=bucket["main_dense_heads"],
                sparse_heads=bucket["main_sparse_heads"],
                sparse_valid=bucket["main_sparse_valid"],
                q_rhs_cols=bucket["q_rhs_cols"],
                support_rows=bucket["support_rows"],
            ) + _validated_scores_from_heads(
                dense_heads=bucket["residual_dense_heads"],
                sparse_heads=bucket["residual_sparse_heads"],
                sparse_valid=bucket["residual_sparse_valid"],
                q_rhs_cols=bucket["q_rhs_cols"],
                support_rows=bucket["support_rows"],
            )
        return last

    def _run_dense_fwd():
        last = None
        for bucket in buckets:
            last = _masked_forward_from_scores(
                bucket,
                _dense_scores(bucket["q_hqd"], bucket["k_hkd"]),
                softmax_scale=softmax_scale,
            )
        return last

    def _run_sparse_main_fwd():
        last = None
        for bucket in buckets:
            last = _masked_forward_from_scores(
                bucket,
                _validated_scores_from_heads(
                    dense_heads=bucket["main_dense_heads"],
                    sparse_heads=bucket["main_sparse_heads"],
                    sparse_valid=bucket["main_sparse_valid"],
                    q_rhs_cols=bucket["q_rhs_cols"],
                    support_rows=bucket["support_rows"],
                ),
                softmax_scale=softmax_scale,
            )
        return last

    def _run_sparse_exact_fwd():
        last = None
        for bucket in buckets:
            last = _masked_forward_from_scores(
                bucket,
                _validated_scores_from_heads(
                    dense_heads=bucket["main_dense_heads"],
                    sparse_heads=bucket["main_sparse_heads"],
                    sparse_valid=bucket["main_sparse_valid"],
                    q_rhs_cols=bucket["q_rhs_cols"],
                    support_rows=bucket["support_rows"],
                ) + _validated_scores_from_heads(
                    dense_heads=bucket["residual_dense_heads"],
                    sparse_heads=bucket["residual_sparse_heads"],
                    sparse_valid=bucket["residual_sparse_valid"],
                    q_rhs_cols=bucket["q_rhs_cols"],
                    support_rows=bucket["support_rows"],
                ),
                softmax_scale=softmax_scale,
            )
        return last

    def _run_custom_masked_fwd():
        last = None
        for bucket in buckets:
            last = _run_custom_masked_bucket_forward(bucket, softmax_scale=softmax_scale)
        return last

    def _run_fa4_packed_fwd():
        last = None
        for bucket in buckets:
            last = _run_fa4_packed_bucket_forward(bucket, softmax_scale=softmax_scale)
        return last

    def _run_shared_cta_fwd():
        last = None
        for bucket in buckets:
            last = _run_shared_cta_bucket_forward(bucket, softmax_scale=softmax_scale)
        return last

    try:
        dense_qk_ms = _measure_ms(_run_dense_qk, warmup_iters, benchmark_iters)
        dense_fwd_ms = _measure_ms(_run_dense_fwd, warmup_iters, benchmark_iters)
        if enable_sparse_payload:
            sparse_main_qk_ms = _measure_ms(_run_sparse_main_qk, warmup_iters, benchmark_iters)
            sparse_exact_qk_ms = _measure_ms(_run_sparse_exact_qk, warmup_iters, benchmark_iters)
            sparse_main_fwd_ms = _measure_ms(_run_sparse_main_fwd, warmup_iters, benchmark_iters)
            sparse_exact_fwd_ms = _measure_ms(_run_sparse_exact_fwd, warmup_iters, benchmark_iters)
        else:
            sparse_main_qk_ms = float("nan")
            sparse_exact_qk_ms = float("nan")
            sparse_main_fwd_ms = float("nan")
            sparse_exact_fwd_ms = float("nan")
            main_output_max_diff = float("nan")
            main_output_sum_abs_diff = 0.0
            main_output_elements = 0
            exact_output_max_diff = float("nan")
            exact_output_sum_abs_diff = 0.0
            exact_output_elements = 0
    except Exception as exc:  # pragma: no cover - benchmark-only failure path
        report["status"] = f"benchmark_failed_{type(exc).__name__}"
        return report

    custom_masked_fwd_ms = float("nan")
    custom_masked_fwd_speedup = float("nan")
    if enable_custom_masked and custom_masked_status == "measured":
        try:
            custom_masked_fwd_ms = _measure_ms(_run_custom_masked_fwd, warmup_iters, benchmark_iters)
            custom_masked_fwd_speedup = dense_fwd_ms / custom_masked_fwd_ms if custom_masked_fwd_ms > 0 else float("inf")
        except Exception as exc:  # pragma: no cover - benchmark-only failure path
            custom_masked_status = f"custom_masked_benchmark_failed_{type(exc).__name__}"
            custom_masked_fwd_ms = float("nan")
            custom_masked_fwd_speedup = float("nan")

    fa4_packed_fwd_ms = float("nan")
    fa4_packed_fwd_speedup = float("nan")
    if enable_fa4_packed and fa4_packed_status == "measured":
        try:
            fa4_packed_fwd_ms = _measure_ms(_run_fa4_packed_fwd, warmup_iters, benchmark_iters)
            fa4_packed_fwd_speedup = dense_fwd_ms / fa4_packed_fwd_ms if fa4_packed_fwd_ms > 0 else float("inf")
        except Exception as exc:  # pragma: no cover - benchmark-only failure path
            fa4_packed_status = f"fa4_packed_benchmark_failed_{type(exc).__name__}"
            fa4_packed_fwd_ms = float("nan")
            fa4_packed_fwd_speedup = float("nan")

    shared_cta_fwd_ms = float("nan")
    shared_cta_fwd_speedup = float("nan")
    if enable_shared_cta and shared_cta_status == "measured":
        try:
            shared_cta_fwd_ms = _measure_ms(_run_shared_cta_fwd, warmup_iters, benchmark_iters)
            shared_cta_fwd_speedup = dense_fwd_ms / shared_cta_fwd_ms if shared_cta_fwd_ms > 0 else float("inf")
        except Exception as exc:  # pragma: no cover - benchmark-only failure path
            shared_cta_status = f"shared_cta_benchmark_failed_{type(exc).__name__}"
            shared_cta_fwd_ms = float("nan")
            shared_cta_fwd_speedup = float("nan")

    report.update(
        {
            "status": "measured",
            "avg_queries_per_bucket": avg_queries_per_bucket,
            "avg_support_rows": avg_support_rows,
            "max_support_rows_seen": int(max_support_rows_seen),
            "dense_qk_ms": dense_qk_ms,
            "sparse_main_qk_ms": sparse_main_qk_ms,
            "sparse_exact_qk_ms": sparse_exact_qk_ms,
            "dense_fwd_ms": dense_fwd_ms,
            "sparse_main_fwd_ms": sparse_main_fwd_ms,
            "sparse_exact_fwd_ms": sparse_exact_fwd_ms,
            "sparse_main_qk_speedup": dense_qk_ms / sparse_main_qk_ms if sparse_main_qk_ms > 0 else float("inf"),
            "sparse_exact_qk_speedup": dense_qk_ms / sparse_exact_qk_ms if sparse_exact_qk_ms > 0 else float("inf"),
            "sparse_main_fwd_speedup": dense_fwd_ms / sparse_main_fwd_ms
            if sparse_main_fwd_ms > 0
            else float("inf"),
            "sparse_exact_fwd_speedup": dense_fwd_ms / sparse_exact_fwd_ms
            if sparse_exact_fwd_ms > 0
            else float("inf"),
            "main_output_max_diff": main_output_max_diff,
            "main_output_mean_diff": main_output_sum_abs_diff / main_output_elements
            if main_output_elements > 0
            else 0.0,
            "exact_output_max_diff": exact_output_max_diff,
            "exact_output_mean_diff": exact_output_sum_abs_diff / exact_output_elements
            if exact_output_elements > 0
            else 0.0,
            "custom_masked_status": str(custom_masked_status),
            "custom_masked_fwd_ms": custom_masked_fwd_ms,
            "custom_masked_fwd_speedup": custom_masked_fwd_speedup,
            "custom_masked_output_max_diff": custom_masked_output_max_diff,
            "custom_masked_output_mean_diff": custom_masked_output_sum_abs_diff / custom_masked_output_elements
            if custom_masked_output_elements > 0
            else 0.0,
            "fa4_packed_status": str(fa4_packed_status),
            "fa4_packed_fwd_ms": fa4_packed_fwd_ms,
            "fa4_packed_fwd_speedup": fa4_packed_fwd_speedup,
            "fa4_packed_output_max_diff": fa4_packed_output_max_diff,
            "fa4_packed_output_mean_diff": fa4_packed_output_sum_abs_diff / fa4_packed_output_elements
            if fa4_packed_output_elements > 0
            else 0.0,
            "shared_cta_status": str(shared_cta_status),
            "shared_cta_fwd_ms": shared_cta_fwd_ms,
            "shared_cta_fwd_speedup": shared_cta_fwd_speedup,
            "shared_cta_output_max_diff": shared_cta_output_max_diff,
            "shared_cta_output_mean_diff": shared_cta_output_sum_abs_diff / shared_cta_output_elements
            if shared_cta_output_elements > 0
            else 0.0,
            "main_retained_abs_fraction": retained_abs_sum / dense_abs_sum if dense_abs_sum > 0 else 0.0,
            "main_retained_l2_fraction": math.sqrt(retained_sq_sum / dense_sq_sum) if dense_sq_sum > 0 else 0.0,
            "total_heads": int(report.get("total_heads", 0)),
            "valid_main_heads": int(report.get("valid_main_heads", 0)),
            "valid_residual_heads": int(report.get("valid_residual_heads", 0)),
        }
    )
    return report


def summarize_hsa_shared_sparse_gemm_forward(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": str(report.get("status", "unknown")),
        "group_size": int(report.get("group_size", 4)),
        "nnz_per_group": int(report.get("nnz_per_group", 2)),
        "bucketizer": str(report.get("bucketizer", "exact_jaccard")),
        "min_jaccard": float(report.get("min_jaccard", 0.5)),
        "max_support_rows": int(report.get("max_support_rows", 0)),
        "max_qgroups_per_bucket": int(report.get("max_qgroups_per_bucket", 0)),
        "backend_validation_atol": float(report.get("backend_validation_atol", 0.5)),
        "enable_sparse_payload": bool(report.get("enable_sparse_payload", True)),
        "enable_custom_masked": bool(report.get("enable_custom_masked", True)),
        "enable_fa4_packed": bool(report.get("enable_fa4_packed", True)),
        "enable_shared_cta": bool(report.get("enable_shared_cta", True)),
        "shared_cta_qgroups_per_cta": int(report.get("shared_cta_qgroups_per_cta", 2)),
        "bucket_count": int(report.get("bucket_count", 0)),
        "available_qgroups": int(report.get("available_qgroups", 0)),
        "sampled_qgroups": int(report.get("sampled_qgroups", 0)),
        "exact_group_count": int(report.get("exact_group_count", 0)),
        "merged_group_count": int(report.get("merged_group_count", 0)),
        "total_heads": int(report.get("total_heads", 0)),
        "valid_main_heads": int(report.get("valid_main_heads", 0)),
        "valid_residual_heads": int(report.get("valid_residual_heads", 0)),
        "avg_queries_per_bucket": float(report.get("avg_queries_per_bucket", 0.0)),
        "avg_support_rows": float(report.get("avg_support_rows", 0.0)),
        "max_support_rows_seen": int(report.get("max_support_rows_seen", 0)),
        "dense_qk_ms": float(report.get("dense_qk_ms", float("nan"))),
        "sparse_main_qk_ms": float(report.get("sparse_main_qk_ms", float("nan"))),
        "sparse_exact_qk_ms": float(report.get("sparse_exact_qk_ms", float("nan"))),
        "dense_fwd_ms": float(report.get("dense_fwd_ms", float("nan"))),
        "sparse_main_fwd_ms": float(report.get("sparse_main_fwd_ms", float("nan"))),
        "sparse_exact_fwd_ms": float(report.get("sparse_exact_fwd_ms", float("nan"))),
        "sparse_main_qk_speedup": float(report.get("sparse_main_qk_speedup", float("nan"))),
        "sparse_exact_qk_speedup": float(report.get("sparse_exact_qk_speedup", float("nan"))),
        "sparse_main_fwd_speedup": float(report.get("sparse_main_fwd_speedup", float("nan"))),
        "sparse_exact_fwd_speedup": float(report.get("sparse_exact_fwd_speedup", float("nan"))),
        "custom_masked_status": str(report.get("custom_masked_status", "not_run")),
        "custom_masked_fwd_ms": float(report.get("custom_masked_fwd_ms", float("nan"))),
        "custom_masked_fwd_speedup": float(report.get("custom_masked_fwd_speedup", float("nan"))),
        "custom_masked_output_max_diff": float(report.get("custom_masked_output_max_diff", float("nan"))),
        "custom_masked_output_mean_diff": float(report.get("custom_masked_output_mean_diff", float("nan"))),
        "fa4_packed_status": str(report.get("fa4_packed_status", "not_run")),
        "fa4_packed_fwd_ms": float(report.get("fa4_packed_fwd_ms", float("nan"))),
        "fa4_packed_fwd_speedup": float(report.get("fa4_packed_fwd_speedup", float("nan"))),
        "fa4_packed_output_max_diff": float(report.get("fa4_packed_output_max_diff", float("nan"))),
        "fa4_packed_output_mean_diff": float(report.get("fa4_packed_output_mean_diff", float("nan"))),
        "shared_cta_status": str(report.get("shared_cta_status", "not_run")),
        "shared_cta_fwd_ms": float(report.get("shared_cta_fwd_ms", float("nan"))),
        "shared_cta_fwd_speedup": float(report.get("shared_cta_fwd_speedup", float("nan"))),
        "shared_cta_output_max_diff": float(report.get("shared_cta_output_max_diff", float("nan"))),
        "shared_cta_output_mean_diff": float(report.get("shared_cta_output_mean_diff", float("nan"))),
        "main_output_max_diff": float(report.get("main_output_max_diff", float("nan"))),
        "main_output_mean_diff": float(report.get("main_output_mean_diff", float("nan"))),
        "exact_output_max_diff": float(report.get("exact_output_max_diff", float("nan"))),
        "exact_output_mean_diff": float(report.get("exact_output_mean_diff", float("nan"))),
        "main_retained_abs_fraction": float(report.get("main_retained_abs_fraction", float("nan"))),
        "main_retained_l2_fraction": float(report.get("main_retained_l2_fraction", float("nan"))),
    }


__all__ = [
    "analyze_hsa_shared_sparse_gemm_forward",
    "summarize_hsa_shared_sparse_gemm_forward",
]
