import math
from dataclasses import dataclass
from typing import Optional

import torch

from flash_attn.cute.cache_utils import get_jit_cache
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


class FlashHSABackwardSm100:
    """Internal-only monolithic HSA backward kernel wrapper on SM100/SM110."""

    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        *,
        qhead_per_kvhead: int,
        k_block_size: int,
        anchor_row_panel_size: int,
        deterministic: bool,
    ):
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.k_block_size = k_block_size
        self.anchor_row_panel_size = anchor_row_panel_size
        self.deterministic = deterministic
        self.tile_m = 128
        self.tile_n = k_block_size


@dataclass
class HSAMonolithicBackwardLaunchPlan:
    arch: int
    dtype: torch.dtype
    head_dim: int
    head_dim_v: int
    num_q_heads: int
    num_kv_heads: int
    qhead_per_kvhead: int
    tile_m: int
    tile_n: int
    k_block_size: int
    anchor_row_panel_size: int
    deterministic: bool
    dkv_postprocess: bool
    num_k_blocks: int
    num_sentence_full_desc: int
    num_sentence_tail_desc: int
    num_anchor_full_desc: int
    num_anchor_tail_desc: int
    total_anchor_q_rows: int
    total_anchor_tail_prefix_rows: int
    dq_accum_shape: tuple[int, ...]
    dpsum_shape: tuple[int, ...]
    lse_log2_shape: tuple[int, ...]
    dk_accum_shape: Optional[tuple[int, ...]] = None
    dv_accum_shape: Optional[tuple[int, ...]] = None
    dQ_semaphore_shape: Optional[tuple[int, ...]] = None
    dK_semaphore_shape: Optional[tuple[int, ...]] = None
    dV_semaphore_shape: Optional[tuple[int, ...]] = None


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _build_hsa_bwd_monolithic_launch_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    monolithic_schedule,
    deterministic: bool,
) -> HSAMonolithicBackwardLaunchPlan:
    batch_size, seqlen_q, num_q_heads, head_dim = q.shape
    seqlen_k, num_kv_heads, head_dim_v = k.shape[1], k.shape[2], v.shape[3]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    tile_m = 128
    tile_n = monolithic_schedule.k_block_size
    seqlen_q_rounded = _round_up(seqlen_q, tile_m)
    seqlen_k_rounded = _round_up(seqlen_k, tile_n)
    head_dim_rounded = _round_up(head_dim, 32)
    head_dim_v_rounded = _round_up(head_dim_v, 32)
    dkv_postprocess = qhead_per_kvhead > 1

    dQ_semaphore_shape = None
    dK_semaphore_shape = None
    dV_semaphore_shape = None
    if deterministic:
        dQ_semaphore_shape = (batch_size, num_q_heads, seqlen_q_rounded // tile_m, 1)
        if dkv_postprocess:
            dK_semaphore_shape = (batch_size, num_kv_heads, seqlen_k_rounded // tile_n, 2)
            dV_semaphore_shape = (batch_size, num_kv_heads, seqlen_k_rounded // tile_n, 2)

    arch = torch.cuda.get_device_capability(q.device)[0] * 10 if q.is_cuda else 0
    return HSAMonolithicBackwardLaunchPlan(
        arch=arch,
        dtype=q.dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        qhead_per_kvhead=qhead_per_kvhead,
        tile_m=tile_m,
        tile_n=tile_n,
        k_block_size=monolithic_schedule.k_block_size,
        anchor_row_panel_size=monolithic_schedule.anchor_row_panel_size,
        deterministic=deterministic,
        dkv_postprocess=dkv_postprocess,
        num_k_blocks=monolithic_schedule.num_k_blocks,
        num_sentence_full_desc=int(monolithic_schedule.sentence_full_q_start.numel()),
        num_sentence_tail_desc=int(monolithic_schedule.sentence_tail_q_start.numel()),
        num_anchor_full_desc=int(monolithic_schedule.anchor_full_q_row_start.numel()),
        num_anchor_tail_desc=int(monolithic_schedule.anchor_tail_q_row_start.numel()),
        total_anchor_q_rows=int(monolithic_schedule.anchor_q_indices.numel()),
        total_anchor_tail_prefix_rows=int(monolithic_schedule.anchor_prefix_len.numel()),
        dq_accum_shape=(batch_size, num_q_heads, seqlen_q_rounded * head_dim_rounded),
        dpsum_shape=(batch_size, num_q_heads, seqlen_q_rounded),
        lse_log2_shape=(batch_size, num_q_heads, seqlen_q_rounded),
        dk_accum_shape=(
            (batch_size, num_kv_heads, seqlen_k_rounded * head_dim_rounded) if dkv_postprocess else None
        ),
        dv_accum_shape=(
            (batch_size, num_kv_heads, seqlen_k_rounded * head_dim_v_rounded) if dkv_postprocess else None
        ),
        dQ_semaphore_shape=dQ_semaphore_shape,
        dK_semaphore_shape=dK_semaphore_shape,
        dV_semaphore_shape=dV_semaphore_shape,
    )


def _allocate_hsa_bwd_monolithic_workspaces(
    plan: HSAMonolithicBackwardLaunchPlan,
    *,
    device: torch.device | str,
) -> dict[str, Optional[torch.Tensor]]:
    workspaces: dict[str, Optional[torch.Tensor]] = {
        "dq_accum": torch.empty(plan.dq_accum_shape, dtype=torch.float32, device=device),
        "dpsum": torch.empty(plan.dpsum_shape, dtype=torch.float32, device=device),
        "lse_log2": torch.empty(plan.lse_log2_shape, dtype=torch.float32, device=device),
        "dk_accum": None,
        "dv_accum": None,
        "dQ_semaphore": None,
        "dK_semaphore": None,
        "dV_semaphore": None,
    }
    if plan.dk_accum_shape is not None:
        workspaces["dk_accum"] = torch.zeros(plan.dk_accum_shape, dtype=torch.float32, device=device)
    if plan.dv_accum_shape is not None:
        workspaces["dv_accum"] = torch.zeros(plan.dv_accum_shape, dtype=torch.float32, device=device)
    if plan.dQ_semaphore_shape is not None:
        workspaces["dQ_semaphore"] = torch.zeros(plan.dQ_semaphore_shape, dtype=torch.int32, device=device)
    if plan.dK_semaphore_shape is not None:
        workspaces["dK_semaphore"] = torch.zeros(plan.dK_semaphore_shape, dtype=torch.int32, device=device)
    if plan.dV_semaphore_shape is not None:
        workspaces["dV_semaphore"] = torch.zeros(plan.dV_semaphore_shape, dtype=torch.int32, device=device)
    return workspaces


def _prepare_hsa_bwd_monolithic_workspaces(
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    workspaces: dict[str, Optional[torch.Tensor]],
):
    assert workspaces["dq_accum"] is not None
    assert workspaces["dpsum"] is not None
    assert workspaces["lse_log2"] is not None
    workspaces["dq_accum"].zero_()
    workspaces["dpsum"].zero_()
    workspaces["lse_log2"].zero_()
    if workspaces["dk_accum"] is not None:
        workspaces["dk_accum"].zero_()
    if workspaces["dv_accum"] is not None:
        workspaces["dv_accum"].zero_()
    if workspaces["dQ_semaphore"] is not None:
        workspaces["dQ_semaphore"].zero_()
    if workspaces["dK_semaphore"] is not None:
        workspaces["dK_semaphore"].zero_()
    if workspaces["dV_semaphore"] is not None:
        workspaces["dV_semaphore"].zero_()

    seqlen = out.shape[1]
    dpsum = (out.float() * dout.float()).sum(dim=-1).permute(0, 2, 1).contiguous()
    workspaces["dpsum"][:, :, :seqlen].copy_(dpsum)
    workspaces["lse_log2"][:, :, :seqlen].copy_(lse.float() * math.log2(math.e))


def _run_hsa_monolithic_panel_math(
    q_sel: torch.Tensor,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    out_sel: torch.Tensor,
    dout_sel: torch.Tensor,
    lse_sel: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hsa_mod = _load_hsa_module()
    batch_count, q_len = q_sel.shape[0], q_sel.shape[1]
    k_len = k_sel.shape[1]
    num_q_heads = q_sel.shape[2]
    q_float = q_sel.float().contiguous()
    k_expanded = hsa_mod._expand_kv_to_q_heads(
        k_sel.reshape(-1, k_sel.shape[2], k_sel.shape[3]).float(),
        num_q_heads,
    ).view(batch_count, k_len, num_q_heads, k_sel.shape[3]).contiguous()
    v_expanded = hsa_mod._expand_kv_to_q_heads(
        v_sel.reshape(-1, v_sel.shape[2], v_sel.shape[3]).float(),
        num_q_heads,
    ).view(batch_count, k_len, num_q_heads, v_sel.shape[3]).contiguous()
    out_float = out_sel.float().contiguous()
    dout_float = dout_sel.float().contiguous()

    q_hqd = q_float.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, q_len, q_sel.shape[3]).contiguous()
    k_hkd = k_expanded.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, k_len, k_sel.shape[3]).contiguous()
    v_hkd = v_expanded.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, k_len, v_sel.shape[3]).contiguous()
    scores = torch.bmm(q_hqd, k_hkd.transpose(1, 2)) * softmax_scale
    lse_expanded = lse_sel.permute(0, 2, 1).reshape(batch_count * num_q_heads, q_len, 1)
    probs = torch.exp(scores - lse_expanded)

    dout_hqd = dout_float.permute(0, 2, 1, 3).reshape(
        batch_count * num_q_heads, q_len, dout_sel.shape[3]
    ).contiguous()
    dprob = torch.bmm(dout_hqd, v_hkd.transpose(1, 2))
    delta = (out_float * dout_float).sum(dim=-1).permute(0, 2, 1).reshape(
        batch_count * num_q_heads, q_len, 1
    )
    dscores = probs * (dprob - delta)

    dq = torch.bmm(dscores, k_hkd).view(batch_count, num_q_heads, q_len, q_sel.shape[3]).permute(0, 2, 1, 3)
    dq = dq.contiguous() * softmax_scale
    dk_expanded = torch.bmm(dscores.transpose(1, 2), q_hqd)
    dk_expanded = dk_expanded.view(batch_count, num_q_heads, k_len, k_sel.shape[3]).permute(0, 2, 1, 3).contiguous()
    dk_expanded = dk_expanded * softmax_scale
    dv_expanded = torch.bmm(probs.transpose(1, 2), dout_hqd)
    dv_expanded = dv_expanded.view(batch_count, num_q_heads, k_len, v_sel.shape[3]).permute(0, 2, 1, 3).contiguous()
    dk = hsa_mod._collapse_q_to_kv_heads(
        dk_expanded.view(batch_count * k_len, num_q_heads, k_sel.shape[3]),
        k_sel.shape[2],
    ).view(batch_count, k_len, k_sel.shape[2], k_sel.shape[3])
    dv = hsa_mod._collapse_q_to_kv_heads(
        dv_expanded.view(batch_count * k_len, num_q_heads, v_sel.shape[3]),
        v_sel.shape[2],
    ).view(batch_count, k_len, v_sel.shape[2], v_sel.shape[3])
    return dq, dk, dv


def _run_hsa_bwd_monolithic_main_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    launch_plan: HSAMonolithicBackwardLaunchPlan,
    softmax_scale: float,
    workspaces: dict[str, Optional[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seqlen_q, num_q_heads, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_kv_heads = k.shape[2]
    head_dim_v = v.shape[-1]
    total_rows = schedule.num_rows
    device = q.device

    q_flat = q.reshape(total_rows, num_q_heads, head_dim)
    k_flat = k.reshape(total_rows, num_kv_heads, head_dim)
    v_flat = v.reshape(total_rows, num_kv_heads, head_dim_v)
    out_flat = out.reshape(total_rows, num_q_heads, head_dim_v).float()
    dout_flat = dout.reshape(total_rows, num_q_heads, head_dim_v).float()
    lse_flat = lse.permute(0, 2, 1).contiguous().view(total_rows, num_q_heads).float()

    dq_acc_rows = torch.zeros_like(q_flat, dtype=torch.float32)
    dk_acc_rows = torch.zeros_like(k_flat, dtype=torch.float32)
    dv_acc_rows = torch.zeros_like(v_flat, dtype=torch.float32)

    def _apply_descriptor(query_rows: torch.Tensor, key_rows: torch.Tensor):
        if query_rows.numel() == 0 or key_rows.numel() == 0:
            return
        q_sel = q_flat.index_select(0, query_rows).unsqueeze(0)
        k_sel = k_flat.index_select(0, key_rows).unsqueeze(0)
        v_sel = v_flat.index_select(0, key_rows).unsqueeze(0)
        out_sel = out_flat.index_select(0, query_rows).unsqueeze(0)
        dout_sel = dout_flat.index_select(0, query_rows).unsqueeze(0)
        lse_sel = lse_flat.index_select(0, query_rows).unsqueeze(0)
        dq_part, dk_part, dv_part = _run_hsa_monolithic_panel_math(
            q_sel,
            k_sel,
            v_sel,
            out_sel,
            dout_sel,
            lse_sel,
            softmax_scale,
        )
        dq_acc_rows.index_add_(0, query_rows, dq_part[0].float())
        dk_acc_rows.index_add_(0, key_rows, dk_part[0].float())
        dv_acc_rows.index_add_(0, key_rows, dv_part[0].float())

    for global_k_block in range(monolithic_schedule.num_k_blocks):
        batch_idx = global_k_block // monolithic_schedule.blocks_per_batch
        block_k_start = (global_k_block % monolithic_schedule.blocks_per_batch) * monolithic_schedule.k_block_size
        block_flat_start = batch_idx * seqlen_k + block_k_start

        sent_full_start = int(monolithic_schedule.sentence_full_kblock_row_ptr[global_k_block].item())
        sent_full_end = int(monolithic_schedule.sentence_full_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_full_start, sent_full_end):
            q_start = int(monolithic_schedule.sentence_full_q_start[desc_idx].item())
            q_len = int(monolithic_schedule.sentence_full_q_len[desc_idx].item())
            k_local_start = int(monolithic_schedule.sentence_full_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.sentence_full_k_len[desc_idx].item())
            query_rows = torch.arange(
                batch_idx * seqlen_q + q_start,
                batch_idx * seqlen_q + q_start + q_len,
                device=device,
                dtype=torch.long,
            )
            key_rows = torch.arange(
                block_flat_start + k_local_start,
                block_flat_start + k_local_start + k_len,
                device=device,
                dtype=torch.long,
            )
            _apply_descriptor(query_rows, key_rows)

        sent_tail_start = int(monolithic_schedule.sentence_tail_kblock_row_ptr[global_k_block].item())
        sent_tail_end = int(monolithic_schedule.sentence_tail_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_tail_start, sent_tail_end):
            q_start = int(monolithic_schedule.sentence_tail_q_start[desc_idx].item())
            q_len = int(monolithic_schedule.sentence_tail_q_len[desc_idx].item())
            k_local_start = int(monolithic_schedule.sentence_tail_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.sentence_tail_k_len[desc_idx].item())
            row0_prefix_len = int(monolithic_schedule.sentence_tail_row0_prefix_len[desc_idx].item())
            query_rows = torch.arange(
                batch_idx * seqlen_q + q_start,
                batch_idx * seqlen_q + q_start + q_len,
                device=device,
                dtype=torch.long,
            )
            for q_offset in range(q_len):
                prefix = min(k_len, row0_prefix_len + q_offset)
                if prefix <= 0:
                    continue
                key_rows = torch.arange(
                    block_flat_start + k_local_start,
                    block_flat_start + k_local_start + prefix,
                    device=device,
                    dtype=torch.long,
                )
                _apply_descriptor(query_rows[q_offset : q_offset + 1], key_rows)

        anchor_full_start = int(monolithic_schedule.anchor_full_kblock_row_ptr[global_k_block].item())
        anchor_full_end = int(monolithic_schedule.anchor_full_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_full_start, anchor_full_end):
            q_row_start = int(monolithic_schedule.anchor_full_q_row_start[desc_idx].item())
            q_row_count = int(monolithic_schedule.anchor_full_q_row_count[desc_idx].item())
            k_local_start = int(monolithic_schedule.anchor_full_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.anchor_full_k_len[desc_idx].item())
            query_rows = monolithic_schedule.anchor_q_indices[q_row_start : q_row_start + q_row_count].long()
            key_rows = torch.arange(
                block_flat_start + k_local_start,
                block_flat_start + k_local_start + k_len,
                device=device,
                dtype=torch.long,
            )
            _apply_descriptor(query_rows, key_rows)

        anchor_tail_start = int(monolithic_schedule.anchor_tail_kblock_row_ptr[global_k_block].item())
        anchor_tail_end = int(monolithic_schedule.anchor_tail_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_tail_start, anchor_tail_end):
            q_row_start = int(monolithic_schedule.anchor_tail_q_row_start[desc_idx].item())
            q_row_count = int(monolithic_schedule.anchor_tail_q_row_count[desc_idx].item())
            k_local_start = int(monolithic_schedule.anchor_tail_k_local_start[desc_idx].item())
            prefix_row_start = int(monolithic_schedule.anchor_tail_prefix_row_start[desc_idx].item())
            query_rows = monolithic_schedule.anchor_q_indices[q_row_start : q_row_start + q_row_count].long()
            prefix_rows = monolithic_schedule.anchor_prefix_len[prefix_row_start : prefix_row_start + q_row_count].long()
            for query_row, prefix in zip(query_rows.tolist(), prefix_rows.tolist()):
                if prefix <= 0:
                    continue
                key_rows = torch.arange(
                    block_flat_start + k_local_start,
                    block_flat_start + k_local_start + prefix,
                    device=device,
                    dtype=torch.long,
                )
                _apply_descriptor(torch.tensor([query_row], device=device, dtype=torch.long), key_rows)

    dq_accum = workspaces["dq_accum"]
    assert dq_accum is not None
    head_dim_rounded = _round_up(head_dim, 32)
    seqlen_q_rounded = _round_up(seqlen_q, launch_plan.tile_m)
    dq_staging = dq_accum.view(batch_size, num_q_heads, seqlen_q_rounded, head_dim_rounded)
    dq_staging.zero_()
    dq_staging[:, :, :seqlen_q, :head_dim] = dq_acc_rows.view(batch_size, seqlen_q, num_q_heads, head_dim).permute(0, 2, 1, 3)
    dq = dq_staging[:, :, :seqlen_q, :head_dim].permute(0, 2, 1, 3).contiguous().to(dtype=q.dtype)

    if launch_plan.dkv_postprocess:
        assert workspaces["dk_accum"] is not None and workspaces["dv_accum"] is not None
        seqlen_k_rounded = _round_up(seqlen_k, launch_plan.tile_n)
        head_dim_k_rounded = _round_up(head_dim, 32)
        head_dim_v_rounded = _round_up(head_dim_v, 32)
        dk_staging = workspaces["dk_accum"].view(batch_size, num_kv_heads, seqlen_k_rounded, head_dim_k_rounded)
        dv_staging = workspaces["dv_accum"].view(batch_size, num_kv_heads, seqlen_k_rounded, head_dim_v_rounded)
        dk_staging.zero_()
        dv_staging.zero_()
        dk_staging[:, :, :seqlen_k, :head_dim] = dk_acc_rows.view(batch_size, seqlen_k, num_kv_heads, head_dim).permute(0, 2, 1, 3)
        dv_staging[:, :, :seqlen_k, :head_dim_v] = dv_acc_rows.view(batch_size, seqlen_k, num_kv_heads, head_dim_v).permute(0, 2, 1, 3)
        dk = dk_staging[:, :, :seqlen_k, :head_dim].permute(0, 2, 1, 3).contiguous().to(dtype=k.dtype)
        dv = dv_staging[:, :, :seqlen_k, :head_dim_v].permute(0, 2, 1, 3).contiguous().to(dtype=v.dtype)
    else:
        dk = dk_acc_rows.view_as(k).to(dtype=k.dtype)
        dv = dv_acc_rows.view_as(v).to(dtype=v.dtype)

    return dq, dk, dv


def _build_hsa_bwd_monolithic_compile_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    monolithic_schedule,
    deterministic: bool,
):
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    return (
        q.dtype,
        q.shape[-1],
        v.shape[-1],
        qhead_per_kvhead,
        monolithic_schedule.k_block_size,
        monolithic_schedule.anchor_row_panel_size,
        deterministic,
        torch.cuda.get_device_capability(q.device),
        num_q_heads == num_kv_heads,
    )


def run_hsa_bwd_sm100_monolithic(
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
        raise NotImplementedError("Monolithic HSA backward scaffold requires CUDA SM100+/fp16 or bf16 tensors")

    hsa_mod = _load_hsa_module()
    monolithic_schedule = hsa_mod._get_hsa_monolithic_backward_schedule(schedule)
    launch_plan = _build_hsa_bwd_monolithic_launch_plan(q, k, v, monolithic_schedule, deterministic)
    compile_key = _build_hsa_bwd_monolithic_compile_key(q, k, v, monolithic_schedule, deterministic)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = FlashHSABackwardSm100(
            q.shape[-1],
            v.shape[-1],
            qhead_per_kvhead=q.shape[2] // k.shape[2],
            k_block_size=monolithic_schedule.k_block_size,
            anchor_row_panel_size=monolithic_schedule.anchor_row_panel_size,
            deterministic=deterministic,
        )
    run_hsa_bwd_sm100_monolithic.launch_plan_cache[compile_key] = launch_plan
    workspaces = _allocate_hsa_bwd_monolithic_workspaces(launch_plan, device=q.device)
    _prepare_hsa_bwd_monolithic_workspaces(out, dout, lse, workspaces)
    return _run_hsa_bwd_monolithic_main_torch(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        launch_plan,
        softmax_scale,
        workspaces,
    )


run_hsa_bwd_sm100_monolithic.compile_cache = get_jit_cache("hsa_bwd_monolithic")
run_hsa_bwd_sm100_monolithic.launch_plan_cache = {}


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
    """Prototype/reference packed backward path using gathered CuTe panel calls."""
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
