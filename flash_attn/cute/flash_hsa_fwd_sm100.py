import os
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import torch

from flash_attn.cute.block_sparsity import normalize_block_sparse_config, to_cute_block_sparse_tensors
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import get_aux_tensor_metadata, to_cute_aux_tensor, to_cute_tensor
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100


def _iter_segment_rows(segment_ptr: torch.Tensor, segment_pos: torch.Tensor):
    ptr = segment_ptr.detach().cpu().tolist()
    pos = segment_pos.detach().cpu().tolist()
    for segment_id in range(len(ptr) - 1):
        yield segment_id, pos[ptr[segment_id] : ptr[segment_id + 1]]


def _load_hsa_module():
    import flash_attn.cute.hsa as hsa_mod

    return hsa_mod


@dataclass
class HSARuntimeState:
    sentence_stream: object
    sentence_stream_indices_long: torch.Tensor
    sentence_stream_key_indices_long: torch.Tensor
    sentence_stream_row_indices_long: torch.Tensor
    section_prefix_stream: object
    section_prefix_query_indices_long: torch.Tensor
    section_prefix_key_indices_long: torch.Tensor
    section_prefix_row_indices_long: torch.Tensor
    document_prefix_stream: object
    document_prefix_query_indices_long: torch.Tensor
    document_prefix_key_indices_long: torch.Tensor
    document_prefix_row_indices_long: torch.Tensor
    section_self_indices: torch.Tensor
    section_self_indices_long: torch.Tensor
    document_self_indices: torch.Tensor
    document_self_indices_long: torch.Tensor

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _materialize_runtime_state(schedule):
    hsa_mod = _load_hsa_module()
    cache = getattr(schedule, "_hsa_runtime_cache", None)
    device_key = str(schedule.sentence_start.device)
    if cache is not None and device_key in cache:
        return cache[device_key]

    sentence_query_indices: list[int] = []
    sentence_cu = [0]
    max_sentence = 0
    for _, segment in _iter_segment_rows(schedule.sentence_segment_ptr, schedule.sentence_segment_pos):
        sentence_query_indices.extend(segment)
        sentence_cu.append(sentence_cu[-1] + len(segment))
        max_sentence = max(max_sentence, len(segment))
    sentence_stream = hsa_mod._make_stream_pack(
        query_indices=sentence_query_indices,
        key_indices=sentence_query_indices,
        row_indices=sentence_query_indices,
        cu_seqlens_q=sentence_cu,
        cu_seqlens_k=sentence_cu,
        max_seqlen_q=max_sentence,
        max_seqlen_k=max_sentence,
        device=schedule.sentence_start.device,
    )

    def _materialize_prefix_stream(segment_ptr: torch.Tensor, segment_pos: torch.Tensor):
        query_indices: list[int] = []
        key_indices: list[int] = []
        cu_q = [0]
        cu_k = [0]
        max_seqlen = 0
        for _, segment in _iter_segment_rows(segment_ptr, segment_pos):
            if len(segment) <= 1:
                continue
            q_chunk = segment[1:]
            k_chunk = segment[:-1]
            query_indices.extend(q_chunk)
            key_indices.extend(k_chunk)
            cu_q.append(cu_q[-1] + len(q_chunk))
            cu_k.append(cu_k[-1] + len(k_chunk))
            max_seqlen = max(max_seqlen, len(q_chunk))
        return hsa_mod._make_stream_pack(
            query_indices=query_indices,
            key_indices=key_indices,
            row_indices=query_indices,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            device=schedule.sentence_start.device,
        )

    section_prefix_stream = _materialize_prefix_stream(
        schedule.section_segment_ptr,
        schedule.section_segment_pos,
    )
    document_prefix_stream = _materialize_prefix_stream(
        schedule.document_segment_ptr,
        schedule.document_segment_pos,
    )
    section_self_indices = torch.nonzero(
        schedule.section_self_allowed,
        as_tuple=False,
    ).flatten().to(dtype=torch.int32, device=schedule.sentence_start.device)
    document_self_indices = torch.nonzero(
        schedule.document_self_allowed,
        as_tuple=False,
    ).flatten().to(dtype=torch.int32, device=schedule.sentence_start.device)
    runtime_state = HSARuntimeState(
        sentence_stream=sentence_stream,
        sentence_stream_indices_long=sentence_stream.query_indices.long(),
        sentence_stream_key_indices_long=sentence_stream.key_indices.long(),
        sentence_stream_row_indices_long=sentence_stream.row_indices.long(),
        section_prefix_stream=section_prefix_stream,
        section_prefix_query_indices_long=section_prefix_stream.query_indices.long(),
        section_prefix_key_indices_long=section_prefix_stream.key_indices.long(),
        section_prefix_row_indices_long=section_prefix_stream.row_indices.long(),
        document_prefix_stream=document_prefix_stream,
        document_prefix_query_indices_long=document_prefix_stream.query_indices.long(),
        document_prefix_key_indices_long=document_prefix_stream.key_indices.long(),
        document_prefix_row_indices_long=document_prefix_stream.row_indices.long(),
        section_self_indices=section_self_indices,
        section_self_indices_long=section_self_indices.long(),
        document_self_indices=document_self_indices,
        document_self_indices_long=document_self_indices.long(),
    )
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_runtime_cache", cache)
    cache[device_key] = runtime_state
    return runtime_state


def run_hsa_fwd_sm100_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule,
    softmax_scale: float,
):
    hsa_mod = _load_hsa_module()
    runtime_state = _materialize_runtime_state(schedule)

    bsz, seqlen, num_q_heads, _ = q.shape
    head_dim_v = v.shape[-1]
    total_rows = bsz * seqlen

    q_flat = q.reshape(total_rows, num_q_heads, q.shape[-1])
    k_flat = k.reshape(total_rows, k.shape[-2], k.shape[-1])
    v_flat = v.reshape(total_rows, v.shape[-2], head_dim_v)

    total_out = torch.zeros(total_rows, num_q_heads, head_dim_v, dtype=torch.float32, device=q.device)
    total_lse = torch.full((total_rows, num_q_heads), float("-inf"), dtype=torch.float32, device=q.device)

    sentence_out, sentence_lse = hsa_mod._run_varlen_fa4_stream(
        q_flat,
        k_flat,
        v_flat,
        runtime_state["sentence_stream"],
        softmax_scale,
    )
    hsa_mod._combine_stream(
        total_out,
        total_lse,
        runtime_state["sentence_stream"].row_indices,
        sentence_out,
        sentence_lse,
    )

    section_prefix_out, section_prefix_lse = hsa_mod._run_varlen_fa4_stream(
        q_flat,
        k_flat,
        v_flat,
        runtime_state["section_prefix_stream"],
        softmax_scale,
    )
    hsa_mod._combine_stream(
        total_out,
        total_lse,
        runtime_state["section_prefix_stream"].row_indices,
        section_prefix_out,
        section_prefix_lse,
    )

    section_self_out, section_self_lse = hsa_mod._self_stream_forward(
        q_flat,
        k_flat,
        v_flat,
        runtime_state["section_self_indices"],
        softmax_scale,
    )
    hsa_mod._combine_stream(
        total_out,
        total_lse,
        runtime_state["section_self_indices"],
        section_self_out,
        section_self_lse,
    )

    document_prefix_out, document_prefix_lse = hsa_mod._run_varlen_fa4_stream(
        q_flat,
        k_flat,
        v_flat,
        runtime_state["document_prefix_stream"],
        softmax_scale,
    )
    hsa_mod._combine_stream(
        total_out,
        total_lse,
        runtime_state["document_prefix_stream"].row_indices,
        document_prefix_out,
        document_prefix_lse,
    )

    document_self_out, document_self_lse = hsa_mod._self_stream_forward(
        q_flat,
        k_flat,
        v_flat,
        runtime_state["document_self_indices"],
        softmax_scale,
    )
    hsa_mod._combine_stream(
        total_out,
        total_lse,
        runtime_state["document_self_indices"],
        document_self_out,
        document_self_lse,
    )

    out = total_out.to(dtype=q.dtype).view(bsz, seqlen, num_q_heads, head_dim_v)
    lse = total_lse.view(bsz, seqlen, num_q_heads).permute(0, 2, 1).contiguous()
    return out, lse, sentence_lse, section_prefix_lse, document_prefix_lse


class FlashHSAForwardSm100(FlashAttentionForwardSm100):
    """Dedicated HSA forward kernel wrapper for direct CuTe compilation."""

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        *,
        qhead_per_kvhead: int,
        q_stage: int,
        q_subtile_factor: int,
        mask_mod,
    ):
        super().__init__(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            is_split_kv=False,
            pack_gqa=False,
            m_block_size=128,
            n_block_size=128,
            q_stage=q_stage,
            is_persistent=True,
            mask_mod=mask_mod,
            has_aux_tensors=True,
            q_subtile_factor=q_subtile_factor,
            use_2cta_instrs=False,
        )


def _run_hsa_fwd_sm100_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule,
    softmax_scale: float,
):
    hsa_mod = _load_hsa_module()
    runtime = hsa_mod._get_hsa_block_sparse_runtime(schedule, q, k)
    sparse_tensors_torch = hsa_mod._to_block_sparse_tensors_torch(runtime.forward_sparse)

    q = q.contiguous() if q.stride(-1) != 1 else q
    k = k.contiguous() if k.stride(-1) != 1 else k
    v = v.contiguous() if v.stride(-1) != 1 else v

    batch_size, seqlen_q, num_q_heads, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_kv_heads = k.shape[2]
    head_dim_v = v.shape[-1]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    q_stage = runtime.forward_block_q // 128

    normalized_sparse, block_sparse_broadcast_pattern, q_subtile_factor = normalize_block_sparse_config(
        sparse_tensors_torch,
        batch_size=batch_size,
        num_head=num_q_heads,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        block_size=(128, 128),
        q_stage=q_stage,
    )

    out = torch.empty_like(q[..., :head_dim_v])
    lse = torch.empty((batch_size, num_q_heads, seqlen_q), dtype=torch.float32, device=q.device)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    aux_tensors = runtime.forward_aux_tensors
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    compile_key = (
        "tile_mask_v1",
        q.dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        q_stage,
        q_subtile_factor,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        torch.cuda.get_device_capability(q.device),
    )

    if compile_key not in _run_hsa_fwd_sm100_direct.compile_cache:
        q_tensor, k_tensor, v_tensor, out_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)
        sparse_tensors = to_cute_block_sparse_tensors(normalized_sparse)
        cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]
        fa_fwd = FlashHSAForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            q_stage=q_stage,
            q_subtile_factor=q_subtile_factor,
            mask_mod=hsa_mod.get_hsa_forward_tile_mask_mod(
                runtime.forward_tile_masks.q_block_size,
                runtime.forward_tile_masks.k_block_size,
            ),
        )
        _run_hsa_fwd_sm100_direct.compile_cache[compile_key] = hsa_mod._lazy_cute_imports()[1].compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            out_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            sparse_tensors,
            cute_aux_tensors,
            options="--enable-tvm-ffi",
        )

    _run_hsa_fwd_sm100_direct.compile_cache[compile_key](
        q.detach(),
        k.detach(),
        v.detach(),
        out.detach(),
        lse,
        softmax_scale,
        current_stream,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        normalized_sparse[:4],
        aux_tensors,
    )
    return out, lse


_run_hsa_fwd_sm100_direct.compile_cache = get_jit_cache("hsa_fwd")


def run_hsa_fwd_sm100_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule,
    softmax_scale: float,
):
    return _run_hsa_fwd_sm100_direct(q, k, v, schedule, softmax_scale)


def run_hsa_fwd_sm100_blocksparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule,
    softmax_scale: float,
):
    sentence_lse = torch.empty(0, dtype=torch.float32, device=q.device)
    sentence_q_stream = torch.empty(0, dtype=q.dtype, device=q.device)
    sentence_k_stream = torch.empty(0, dtype=k.dtype, device=k.device)
    sentence_v_stream = torch.empty(0, dtype=v.dtype, device=v.device)
    sentence_out_stream = torch.empty(0, dtype=q.dtype, device=q.device)
    if os.environ.get("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "0") == "1":
        hsa_mod = _load_hsa_module()
        runtime_state = _materialize_runtime_state(schedule)
        total_rows = schedule.num_rows
        q_flat = q.reshape(total_rows, q.shape[2], q.shape[3])
        k_flat = k.reshape(total_rows, k.shape[2], k.shape[3])
        v_flat = v.reshape(total_rows, v.shape[2], v.shape[3])
        cache_sentence_streams = os.environ.get("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "0") == "1"
        packed_q_stream = q_flat.index_select(0, runtime_state["sentence_stream_indices_long"]).contiguous()
        packed_k_stream = k_flat.index_select(0, runtime_state["sentence_stream_key_indices_long"]).contiguous()
        packed_v_stream = v_flat.index_select(0, runtime_state["sentence_stream_key_indices_long"]).contiguous()
        _, _, _, _, _, flash_attn_fwd, _ = hsa_mod._lazy_cute_imports()
        packed_out_stream, sentence_lse = flash_attn_fwd(
            packed_q_stream,
            packed_k_stream,
            packed_v_stream,
            cu_seqlens_q=runtime_state["sentence_stream"].cu_seqlens_q,
            cu_seqlens_k=runtime_state["sentence_stream"].cu_seqlens_k,
            max_seqlen_q=runtime_state["sentence_stream"].max_seqlen_q,
            max_seqlen_k=runtime_state["sentence_stream"].max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            return_lse=True,
        )
        sentence_lse = sentence_lse.detach()
        if cache_sentence_streams:
            sentence_q_stream = packed_q_stream.detach()
            sentence_k_stream = packed_k_stream.detach()
            sentence_v_stream = packed_v_stream.detach()
            sentence_out_stream = packed_out_stream.detach()
    if os.environ.get("FLASH_ATTN_HSA_USE_FUSED_FWD", "0") == "1":
        out, lse = run_hsa_fwd_sm100_fused(q, k, v, schedule, softmax_scale)
        return out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream
    hsa_mod = _load_hsa_module()
    out, lse = hsa_mod._run_hsa_blocksparse_forward(q, k, v, schedule, softmax_scale)
    return out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream
