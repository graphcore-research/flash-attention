from __future__ import annotations

import torch


def _ensure_runtime(schedule, runtime, q, k):
    if runtime is None:
        import flash_attn.cute.hsa as hsa_module

        runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    import flash_attn.cute.hsa as hsa_module

    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    return runtime


def summarize_synthetic_grid(runtime) -> dict[str, float | int]:
    synthetic = runtime.synthetic_grid
    if synthetic is None:
        return {
            "logical_block_q": 0,
            "logical_block_k": 0,
            "physical_block_q": 0,
            "physical_block_k": 0,
            "num_tiles": 0,
            "avg_q_rows": 0.0,
            "avg_k_rows": 0.0,
            "avg_logical_pairs": 0.0,
        }

    q_rows_per_tile = synthetic.tile_q_row_ptr[1:] - synthetic.tile_q_row_ptr[:-1]
    k_rows_per_tile = synthetic.tile_k_row_ptr[1:] - synthetic.tile_k_row_ptr[:-1]
    logical_pairs_per_tile = synthetic.tile_logical_pair_row_ptr[1:] - synthetic.tile_logical_pair_row_ptr[:-1]

    def _mean_or_zero(tensor: torch.Tensor) -> float:
        return float(tensor.float().mean().item()) if tensor.numel() > 0 else 0.0

    return {
        "logical_block_q": synthetic.logical_block_q,
        "logical_block_k": synthetic.logical_block_k,
        "physical_block_q": synthetic.physical_block_q,
        "physical_block_k": synthetic.physical_block_k,
        "num_tiles": synthetic.num_tiles,
        "avg_q_rows": _mean_or_zero(q_rows_per_tile),
        "avg_k_rows": _mean_or_zero(k_rows_per_tile),
        "avg_logical_pairs": _mean_or_zero(logical_pairs_per_tile),
    }


def run_hsa_fwd_sm100_synthetic_grid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule,
    softmax_scale: float,
    *,
    runtime=None,
):
    runtime = _ensure_runtime(schedule, runtime, q, k)
    from flash_attn.cute.flash_hsa_fwd_sm100 import run_hsa_fwd_sm100_blocksparse

    return run_hsa_fwd_sm100_blocksparse(q, k, v, schedule, softmax_scale)


def run_hsa_bwd_sm100_synthetic_grid(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    sentence_lse,
    sentence_q_stream,
    sentence_k_stream,
    sentence_v_stream,
    sentence_out_stream,
    schedule,
    softmax_scale: float,
    deterministic: bool,
    keep_ids=None,
    hash_ids=None,
    *,
    runtime=None,
):
    runtime = _ensure_runtime(schedule, runtime, q, k)
    from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_blocksparse

    return run_hsa_bwd_sm100_blocksparse(
        q,
        k,
        v,
        out,
        dout,
        lse,
        sentence_lse,
        sentence_q_stream,
        sentence_k_stream,
        sentence_v_stream,
        sentence_out_stream,
        schedule,
        softmax_scale,
        deterministic,
        keep_ids,
        hash_ids,
        runtime=runtime,
    )
