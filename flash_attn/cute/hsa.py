import math
from functools import lru_cache
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import fast_sampling
from flash_attn.cute.interface import flash_attn_func


@lru_cache(maxsize=1)
def get_hsa_mask_mod():
    """Return the CuTe mask_mod implementing nanochat's decoder-HDT attention."""

    @fast_sampling
    @cute.jit
    def _hsa_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        keep_ids = aux_tensors[0]
        hash_ids = aux_tensors[1]

        b = batch[0]
        q_idx = m_idx[0]
        kv_idx = n_idx[0]
        safe_q_idx = q_idx % seqlen_info.seqlen_q
        safe_kv_idx = kv_idx % seqlen_info.seqlen_k
        in_bounds = (q_idx < seqlen_info.seqlen_q) & (kv_idx < seqlen_info.seqlen_k)

        q_ki0 = utils.scalar_to_ssa(keep_ids[b, 0, safe_q_idx], cutlass.Int32)
        q_ki1 = utils.scalar_to_ssa(keep_ids[b, 1, safe_q_idx], cutlass.Int32)
        q_ki2 = utils.scalar_to_ssa(keep_ids[b, 2, safe_q_idx], cutlass.Int32)
        k_ki0 = utils.scalar_to_ssa(keep_ids[b, 0, safe_kv_idx], cutlass.Int32)
        k_ki1 = utils.scalar_to_ssa(keep_ids[b, 1, safe_kv_idx], cutlass.Int32)
        k_ki2 = utils.scalar_to_ssa(keep_ids[b, 2, safe_kv_idx], cutlass.Int32)

        q_h0 = utils.scalar_to_ssa(hash_ids[b, 0, safe_q_idx], cutlass.Int32)
        q_h1 = utils.scalar_to_ssa(hash_ids[b, 1, safe_q_idx], cutlass.Int32)
        q_h2 = utils.scalar_to_ssa(hash_ids[b, 2, safe_q_idx], cutlass.Int32)
        k_h0 = utils.scalar_to_ssa(hash_ids[b, 0, safe_kv_idx], cutlass.Int32)
        k_h1 = utils.scalar_to_ssa(hash_ids[b, 1, safe_kv_idx], cutlass.Int32)
        k_h2 = utils.scalar_to_ssa(hash_ids[b, 2, safe_kv_idx], cutlass.Int32)

        same_sent = q_h0 == k_h0
        same_sec = q_h1 == k_h1
        same_doc = q_h2 == k_h2

        level0 = (
            (q_ki0 != 0)
            & (k_ki0 != 0)
            & same_sent
            & ((q_ki1 == 0) | (k_ki1 == 0))
        )
        level1 = (q_ki1 != 0) & (k_ki1 != 0) & same_sec
        level2 = (q_ki2 != 0) & (k_ki2 != 0) & same_doc
        causal = kv_idx <= q_idx
        return in_bounds & causal & (level0 | level1 | level2)

    return _hsa_mask


def compute_hsa_mask(keep_ids: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
    """Return the additive float mask used by decoder-HDT attention."""
    keep_ids = keep_ids.to(dtype=torch.int32)
    hash_ids = hash_ids.to(dtype=torch.int32)
    bsz, _, seqlen = keep_ids.shape
    device = keep_ids.device

    ki0 = keep_ids[:, 0].bool()
    ki1 = keep_ids[:, 1].bool()
    ki2 = keep_ids[:, 2].bool()

    h0 = hash_ids[:, 0]
    h1 = hash_ids[:, 1]
    h2 = hash_ids[:, 2]

    pos = torch.arange(seqlen, device=device)
    causal = pos[None, :, None] >= pos[None, None, :]

    same_sent = h0.unsqueeze(2) == h0.unsqueeze(1)
    same_sec = h1.unsqueeze(2) == h1.unsqueeze(1)
    same_doc = h2.unsqueeze(2) == h2.unsqueeze(1)

    cls_mask = (
        ki0.unsqueeze(2)
        & ki1.unsqueeze(2)
        & ki0.unsqueeze(1)
        & ki1.unsqueeze(1)
    )

    level0 = ki0.unsqueeze(2) & ki0.unsqueeze(1) & same_sent & ~cls_mask
    level1 = ki1.unsqueeze(2) & ki1.unsqueeze(1) & same_sec
    level2 = ki2.unsqueeze(2) & ki2.unsqueeze(1) & same_doc
    attend = causal & (level0 | level1 | level2)

    mask = torch.zeros(bsz, seqlen, seqlen, device=device, dtype=torch.float32)
    mask.masked_fill_(~attend, float("-inf"))
    return mask


def hsa_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Explicit masked-softmax reference for HSA attention."""
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    if q_ref.size(1) != k_ref.size(1):
        repeat_factor = q_ref.size(1) // k_ref.size(1)
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=1)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=1)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    scores = scores + compute_hsa_mask(keep_ids, hash_ids).unsqueeze(1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v_ref)
    return out.transpose(1, 2).to(dtype=q.dtype)


def flash_attn_hsa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """Public fixed-length HSA training entrypoint built on top of flash_attn_func."""
    keep_ids = keep_ids if keep_ids.dtype == torch.int32 else keep_ids.to(dtype=torch.int32)
    hash_ids = hash_ids if hash_ids.dtype == torch.int32 else hash_ids.to(dtype=torch.int32)
    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        deterministic=deterministic,
        mask_mod=get_hsa_mask_mod(),
        aux_tensors=[keep_ids.contiguous(), hash_ids.contiguous()],
        return_lse=return_lse,
    )

