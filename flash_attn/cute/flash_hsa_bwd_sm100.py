from typing import Optional

import torch

from flash_attn.cute.flash_hsa_fwd_sm100 import _materialize_runtime_state


def _load_hsa_module():
    import flash_attn.cute.hsa as hsa_mod

    return hsa_mod


def _lazy_cute_imports():
    return _load_hsa_module()._lazy_cute_imports()


def _is_supported_packed_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        return False
    arch = torch.cuda.get_device_capability(q.device)
    if arch[0] not in (10, 11):
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        return False
    return True


def _gather_batch_tensors(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
):
    q_indices = batch.q_indices.long()
    k_indices = batch.k_indices.long()
    q_sel = q_flat.index_select(0, q_indices.reshape(-1)).view(
        q_indices.shape[0], q_indices.shape[1], q_flat.shape[1], q_flat.shape[2]
    ).contiguous()
    k_sel = k_flat.index_select(0, k_indices.reshape(-1)).view(
        k_indices.shape[0], k_indices.shape[1], k_flat.shape[1], k_flat.shape[2]
    ).contiguous()
    v_sel = v_flat.index_select(0, k_indices.reshape(-1)).view(
        k_indices.shape[0], k_indices.shape[1], v_flat.shape[1], v_flat.shape[2]
    ).contiguous()
    out_sel = out_flat.index_select(0, q_indices.reshape(-1)).view(
        q_indices.shape[0], q_indices.shape[1], out_flat.shape[1], out_flat.shape[2]
    ).contiguous()
    dout_sel = dout_flat.index_select(0, q_indices.reshape(-1)).view(
        q_indices.shape[0], q_indices.shape[1], dout_flat.shape[1], dout_flat.shape[2]
    ).contiguous()
    lse_sel = lse_flat.index_select(0, q_indices.reshape(-1)).view(
        q_indices.shape[0], q_indices.shape[1], lse_flat.shape[1]
    ).permute(0, 2, 1).contiguous()
    return q_indices, k_indices, q_sel, k_sel, v_sel, out_sel, dout_sel, lse_sel


def _build_sentence_prefix_len(batch) -> torch.Tensor:
    row_offsets = torch.arange(batch.q_indices.shape[1], device=batch.q_indices.device, dtype=torch.int32).view(1, -1)
    prefix_len = row_offsets + 1
    prefix_len = torch.minimum(prefix_len, batch.k_length.unsqueeze(1))
    prefix_len = torch.where(
        row_offsets < batch.q_length.unsqueeze(1),
        prefix_len,
        torch.zeros_like(prefix_len),
    )
    return prefix_len.contiguous()


def _run_panel_batch_cute(
    q_sel: torch.Tensor,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    out_sel: torch.Tensor,
    dout_sel: torch.Tensor,
    lse_sel: torch.Tensor,
    prefix_len: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    softmax_scale: float,
    deterministic: bool,
):
    hsa_mod = _load_hsa_module()
    _, _, _, _, _, _, flash_attn_bwd = _lazy_cute_imports()
    q_length_table = q_length.unsqueeze(1).expand(-1, prefix_len.shape[1]).contiguous()
    k_length_table = k_length.unsqueeze(1).expand(-1, prefix_len.shape[1]).contiguous()
    prefix_len.__leading_dim__ = 1
    q_length_table.__leading_dim__ = 1
    k_length_table.__leading_dim__ = 1
    aux_tensors = [prefix_len, q_length_table, k_length_table]
    return flash_attn_bwd(
        q_sel,
        k_sel,
        v_sel,
        out_sel,
        dout_sel,
        lse_sel,
        softmax_scale=softmax_scale,
        causal=False,
        pack_gqa=False,
        deterministic=deterministic,
        mask_mod=hsa_mod.get_hsa_panel_prefix_mask_mod(),
        aux_tensors=aux_tensors,
    )


def run_hsa_bwd_sm100_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    softmax_scale: float,
    deterministic: bool,
):
    if not _is_supported_packed_bwd(q, k, v):
        raise NotImplementedError("Packed HSA backward requires CUDA SM100+/fp16 or bf16 fixed-length tensors")

    hsa_mod = _load_hsa_module()
    hybrid_schedule = hsa_mod._get_hsa_hybrid_backward_schedule(schedule)
    sentence_batches, anchor_batches = hsa_mod._get_hsa_hybrid_backward_batches(schedule, hybrid_schedule)

    total_rows = schedule.num_rows
    head_dim = q.shape[-1]
    head_dim_v = v.shape[-1]

    q_flat = q.reshape(total_rows, q.shape[2], head_dim)
    k_flat = k.reshape(total_rows, k.shape[2], head_dim)
    v_flat = v.reshape(total_rows, v.shape[2], head_dim_v)
    out_flat = out.reshape(total_rows, out.shape[2], out.shape[3])
    dout_flat = dout.reshape(total_rows, dout.shape[2], dout.shape[3])
    lse_flat = lse.permute(0, 2, 1).contiguous().view(total_rows, q.shape[2]).float()

    dq_acc = torch.zeros_like(q_flat, dtype=torch.float32)
    dk_acc = torch.zeros_like(k_flat, dtype=torch.float32)
    dv_acc = torch.zeros_like(v_flat, dtype=torch.float32)

    for batch in sentence_batches:
        q_indices, k_indices, q_sel, k_sel, v_sel, out_sel, dout_sel, lse_sel = _gather_batch_tensors(
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            dout_flat,
            lse_flat,
            batch,
        )
        dq, dk, dv = _run_panel_batch_cute(
            q_sel,
            k_sel,
            v_sel,
            out_sel,
            dout_sel,
            lse_sel,
            _build_sentence_prefix_len(batch),
            batch.q_length,
            batch.k_length,
            softmax_scale,
            deterministic,
        )
        q_valid = torch.arange(q_indices.shape[1], device=q.device).view(1, -1) < batch.q_length.unsqueeze(1)
        k_valid = torch.arange(k_indices.shape[1], device=q.device).view(1, -1) < batch.k_length.unsqueeze(1)
        dq_acc.index_add_(0, q_indices[q_valid].long(), dq[q_valid].float())
        dk_acc.index_add_(0, k_indices[k_valid].long(), dk[k_valid].float())
        dv_acc.index_add_(0, k_indices[k_valid].long(), dv[k_valid].float())

    for batch in anchor_batches:
        q_indices, k_indices, q_sel, k_sel, v_sel, out_sel, dout_sel, lse_sel = _gather_batch_tensors(
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            dout_flat,
            lse_flat,
            batch,
        )
        dq, dk, dv = _run_panel_batch_cute(
            q_sel,
            k_sel,
            v_sel,
            out_sel,
            dout_sel,
            lse_sel,
            batch.prefix_len,
            batch.q_length,
            batch.k_length,
            softmax_scale,
            deterministic,
        )
        q_valid = torch.arange(q_indices.shape[1], device=q.device).view(1, -1) < batch.q_length.unsqueeze(1)
        k_valid = torch.arange(k_indices.shape[1], device=q.device).view(1, -1) < batch.k_length.unsqueeze(1)
        dq_acc.index_add_(0, q_indices[q_valid].long(), dq[q_valid].float())
        dk_acc.index_add_(0, k_indices[k_valid].long(), dk[k_valid].float())
        dv_acc.index_add_(0, k_indices[k_valid].long(), dv[k_valid].float())

    return (
        dq_acc.view_as(q).to(dtype=q.dtype),
        dk_acc.view_as(k).to(dtype=k.dtype),
        dv_acc.view_as(v).to(dtype=v.dtype),
    )


def run_hsa_bwd_sm100_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    total_lse_flat: torch.Tensor,
    sentence_lse: torch.Tensor,
    section_prefix_lse: torch.Tensor,
    document_prefix_lse: torch.Tensor,
    schedule,
    softmax_scale: float,
    deterministic: bool,
):
    hsa_mod = _load_hsa_module()
    runtime_state = _materialize_runtime_state(schedule)
    total_rows = schedule.num_rows

    q_flat = q.reshape(total_rows, q.shape[2], q.shape[3])
    k_flat = k.reshape(total_rows, k.shape[2], k.shape[3])
    v_flat = v.reshape(total_rows, v.shape[2], v.shape[3])
    out_total_flat = out.reshape(total_rows, out.shape[2], out.shape[3]).float()
    dout_flat = dout.reshape(total_rows, dout.shape[2], dout.shape[3]).float()

    dq_acc = torch.zeros_like(q_flat, dtype=torch.float32)
    dk_acc = torch.zeros_like(k_flat, dtype=torch.float32)
    dv_acc = torch.zeros_like(v_flat, dtype=torch.float32)

    sentence_weights = hsa_mod._stream_weights(
        total_lse_flat,
        runtime_state["sentence_stream"].row_indices,
        sentence_lse,
    )
    sentence_dout = torch.zeros_like(dout_flat)
    if sentence_weights.numel() > 0:
        row_idx = runtime_state["sentence_stream"].row_indices.long()
        sentence_dout.index_copy_(
            0,
            row_idx,
            sentence_weights.unsqueeze(-1) * dout_flat.index_select(0, row_idx),
        )
        dq, dk, dv = hsa_mod._run_varlen_fa4_bwd(
            q_flat,
            k_flat,
            v_flat,
            out_total_flat,
            sentence_dout,
            sentence_lse,
            runtime_state["sentence_stream"],
            softmax_scale,
            deterministic,
        )
        dq_acc.index_add_(0, runtime_state["sentence_stream"].query_indices.long(), dq.float())
        dk_acc.index_add_(0, runtime_state["sentence_stream"].key_indices.long(), dk.float())
        dv_acc.index_add_(0, runtime_state["sentence_stream"].key_indices.long(), dv.float())

    section_prefix_weights = hsa_mod._stream_weights(
        total_lse_flat,
        runtime_state["section_prefix_stream"].row_indices,
        section_prefix_lse,
    )
    section_prefix_dout = torch.zeros_like(dout_flat)
    if section_prefix_weights.numel() > 0:
        row_idx = runtime_state["section_prefix_stream"].row_indices.long()
        section_prefix_dout.index_copy_(
            0,
            row_idx,
            section_prefix_weights.unsqueeze(-1) * dout_flat.index_select(0, row_idx),
        )
        dq, dk, dv = hsa_mod._run_varlen_fa4_bwd(
            q_flat,
            k_flat,
            v_flat,
            out_total_flat,
            section_prefix_dout,
            section_prefix_lse,
            runtime_state["section_prefix_stream"],
            softmax_scale,
            deterministic,
        )
        dq_acc.index_add_(0, runtime_state["section_prefix_stream"].query_indices.long(), dq.float())
        dk_acc.index_add_(0, runtime_state["section_prefix_stream"].key_indices.long(), dk.float())
        dv_acc.index_add_(0, runtime_state["section_prefix_stream"].key_indices.long(), dv.float())

    hsa_mod._accumulate_self_stream_grads(
        dq_acc,
        dk_acc,
        dv_acc,
        q_flat,
        k_flat,
        v_flat,
        out_total_flat,
        dout_flat,
        total_lse_flat,
        runtime_state["section_self_indices"],
        softmax_scale,
    )

    document_prefix_weights = hsa_mod._stream_weights(
        total_lse_flat,
        runtime_state["document_prefix_stream"].row_indices,
        document_prefix_lse,
    )
    document_prefix_dout = torch.zeros_like(dout_flat)
    if document_prefix_weights.numel() > 0:
        row_idx = runtime_state["document_prefix_stream"].row_indices.long()
        document_prefix_dout.index_copy_(
            0,
            row_idx,
            document_prefix_weights.unsqueeze(-1) * dout_flat.index_select(0, row_idx),
        )
        dq, dk, dv = hsa_mod._run_varlen_fa4_bwd(
            q_flat,
            k_flat,
            v_flat,
            out_total_flat,
            document_prefix_dout,
            document_prefix_lse,
            runtime_state["document_prefix_stream"],
            softmax_scale,
            deterministic,
        )
        dq_acc.index_add_(0, runtime_state["document_prefix_stream"].query_indices.long(), dq.float())
        dk_acc.index_add_(0, runtime_state["document_prefix_stream"].key_indices.long(), dk.float())
        dv_acc.index_add_(0, runtime_state["document_prefix_stream"].key_indices.long(), dv.float())

    hsa_mod._accumulate_self_stream_grads(
        dq_acc,
        dk_acc,
        dv_acc,
        q_flat,
        k_flat,
        v_flat,
        out_total_flat,
        dout_flat,
        total_lse_flat,
        runtime_state["document_self_indices"],
        softmax_scale,
    )

    return (
        dq_acc.view_as(q).to(dtype=q.dtype),
        dk_acc.view_as(k).to(dtype=k.dtype),
        dv_acc.view_as(v).to(dtype=v.dtype),
    )


def run_hsa_bwd_sm100_blocksparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    softmax_scale: float,
    deterministic: bool,
    keep_ids: torch.Tensor | None = None,
    hash_ids: torch.Tensor | None = None,
):
    hsa_mod = _load_hsa_module()
    return hsa_mod._run_hsa_blocksparse_backward(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        softmax_scale,
        deterministic,
        keep_ids,
        hash_ids,
    )
