import torch

from flash_attn.cute.flash_hsa_fwd_sm100 import _materialize_runtime_state


def _load_hsa_module():
    import flash_attn.cute.hsa as hsa_mod

    return hsa_mod


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
    )
