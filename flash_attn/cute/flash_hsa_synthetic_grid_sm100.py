from __future__ import annotations

import math

import torch

try:
    import cutlass
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Boolean, Float32, Int32

    from flash_attn.cute.cache_utils import get_jit_cache
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
    from flash_attn.cute import utils
    from flash_attn.cute.utils import scalar_to_ssa, ssa_to_scalar

    _HAS_CUTE_RUNTIME = True
except Exception:  # pragma: no cover - CPU-only guard
    _HAS_CUTE_RUNTIME = False

    class _FakeCuda:
        class CUstream:  # noqa: D401 - simple placeholder
            """CPU placeholder."""

    cuda = _FakeCuda()


def _load_hsa_module():
    import flash_attn.cute.hsa as hsa_module

    return hsa_module


def _require_cute_runtime():
    if not _HAS_CUTE_RUNTIME:
        raise NotImplementedError("Synthetic packed CuTE path requires CUDA/CuTE runtime")


def _ensure_runtime(schedule, runtime, q, k):
    hsa_module = _load_hsa_module()
    if runtime is None:
        runtime = hsa_module._get_hsa_block_sparse_runtime(schedule, q, k)
    hsa_module._ensure_hsa_synthetic_grid_metadata(schedule, runtime)
    return runtime


def _is_mixed_schedule(schedule) -> bool:
    hsa_module = _load_hsa_module()
    return not hsa_module._schedule_has_only_sentence_backward_families(schedule)


def _mean_or_zero(tensor: torch.Tensor) -> float:
    return float(tensor.float().mean().item()) if tensor.numel() > 0 else 0.0


def _quantile_or_zero(tensor: torch.Tensor, quantile: float) -> float:
    if tensor.numel() <= 0:
        return 0.0
    return float(torch.quantile(tensor.float(), quantile).item())


def _summarize_one_grid(metadata) -> dict[str, float | int]:
    if metadata is None:
        return {
            "num_tiles": 0,
            "num_qgroups": 0,
            "num_buckets": 0,
            "avg_q_rows": 0.0,
            "avg_k_rows": 0.0,
            "avg_logical_pairs": 0.0,
            "avg_packed_q": 0.0,
            "avg_packed_k": 0.0,
            "avg_fill": 0.0,
            "fill_p50": 0.0,
            "fill_p90": 0.0,
            "avg_num_splits": 0.0,
            "dense_tiles": 0,
        }

    q_rows_per_tile = metadata.tile_q_row_ptr[1:] - metadata.tile_q_row_ptr[:-1]
    k_rows_per_tile = metadata.tile_k_row_ptr[1:] - metadata.tile_k_row_ptr[:-1]
    logical_pairs_per_tile = metadata.tile_logical_pair_row_ptr[1:] - metadata.tile_logical_pair_row_ptr[:-1]
    if (
        metadata.bucket_packed_q is not None
        and metadata.bucket_packed_q.numel() > 0
        and (metadata.bucket_fill is not None or metadata.bucket_allowed_pairs is not None)
    ):
        bucket_sizes = metadata.bucket_row_ptr[1:] - metadata.bucket_row_ptr[:-1]
        if metadata.bucket_fill is not None:
            fill = metadata.bucket_fill.float()
        elif metadata.bucket_allowed_pairs is not None:
            packed_area = bucket_sizes.float() * metadata.bucket_packed_q.float() * metadata.bucket_packed_k.float()
            fill = metadata.bucket_allowed_pairs.float() / torch.clamp(packed_area, min=1.0)
        else:
            packed_area = metadata.bucket_packed_q.float() * metadata.bucket_packed_k.float()
            fill = metadata.tile_allowed_pairs.float() / torch.clamp(packed_area, min=1.0)
        avg_packed_q = _mean_or_zero(metadata.bucket_packed_q)
        avg_packed_k = _mean_or_zero(metadata.bucket_packed_k)
        num_buckets = int(metadata.bucket_packed_q.numel())
    else:
        if metadata.tile_fill is not None:
            fill = metadata.tile_fill.float()
        else:
            packed_area = metadata.tile_packed_q.float() * metadata.tile_packed_k.float()
            fill = metadata.tile_allowed_pairs.float() / torch.clamp(packed_area, min=1.0)
        avg_packed_q = _mean_or_zero(metadata.tile_packed_q)
        avg_packed_k = _mean_or_zero(metadata.tile_packed_k)
        num_buckets = 0
    return {
        "num_tiles": metadata.num_tiles,
        "num_qgroups": int(metadata.qgroup_length.numel()) if metadata.qgroup_length is not None else 0,
        "num_buckets": num_buckets,
        "avg_q_rows": _mean_or_zero(q_rows_per_tile),
        "avg_k_rows": _mean_or_zero(k_rows_per_tile),
        "avg_logical_pairs": _mean_or_zero(logical_pairs_per_tile),
        "avg_packed_q": avg_packed_q,
        "avg_packed_k": avg_packed_k,
        "avg_fill": _mean_or_zero(fill),
        "fill_p50": _quantile_or_zero(fill, 0.5),
        "fill_p90": _quantile_or_zero(fill, 0.9),
        "avg_num_splits": _mean_or_zero(metadata.qgroup_num_splits)
        if metadata.qgroup_num_splits is not None
        else 0.0,
        "dense_tiles": int(metadata.tile_dense.sum().item()) if metadata.tile_dense.numel() > 0 else 0,
    }


def summarize_synthetic_grid(runtime) -> dict[str, float | int]:
    backward = _summarize_one_grid(runtime.backward_synthetic_grid)
    forward = _summarize_one_grid(runtime.forward_synthetic_grid)
    reference = runtime.forward_synthetic_grid if runtime.forward_synthetic_grid is not None else runtime.backward_synthetic_grid
    direct_plan = None
    if runtime.forward_synthetic_grid is not None and runtime.forward_synthetic_grid.forward_execution_plan is not None:
        direct_plan = runtime.forward_synthetic_grid.forward_execution_plan.get("direct_execution_plan")
    if reference is None:
        return {
            "logical_block_q": 0,
            "logical_block_k": 0,
            "max_packed_k": 0,
            "max_direct_segments": 0,
            "physical_block_q": 0,
            "physical_block_k": 0,
            "num_tiles": 0,
            "avg_q_rows": 0.0,
            "avg_k_rows": 0.0,
            "avg_logical_pairs": 0.0,
            "forward_tiles": 0,
            "forward_qgroups": 0,
            "backward_tiles": 0,
            "backward_qgroups": 0,
            "forward_buckets": 0,
            "backward_buckets": 0,
            "forward_avg_packed_q": 0.0,
            "forward_avg_packed_k": 0.0,
            "backward_avg_packed_q": 0.0,
            "backward_avg_packed_k": 0.0,
            "forward_avg_fill": 0.0,
            "forward_fill_p50": 0.0,
            "forward_fill_p90": 0.0,
            "backward_avg_fill": 0.0,
            "backward_fill_p50": 0.0,
            "backward_fill_p90": 0.0,
            "forward_avg_num_splits": 0.0,
            "backward_avg_num_splits": 0.0,
            "forward_avg_direct_segments": 0.0,
            "forward_avg_segment_k_length": 0.0,
            "forward_avg_segment_fill": 0.0,
            "forward_segment_fill_p50": 0.0,
            "forward_segment_fill_p90": 0.0,
            "forward_avg_row_k": 0.0,
            "forward_row_k_p50": 0.0,
            "forward_row_k_p90": 0.0,
            "forward_avg_union_k": 0.0,
        }

    if direct_plan is not None and len(direct_plan["bucket_packed_k"]) > 0:
        segment_k = torch.tensor(direct_plan["bucket_packed_k"], dtype=torch.float32)
        segment_fill = torch.tensor(direct_plan["bucket_fill"], dtype=torch.float32)
        qgroup_num_segments = torch.tensor(direct_plan["qgroup_num_segments"], dtype=torch.float32)
        max_direct_segments = int(direct_plan["max_direct_segments"])
        row_plan = direct_plan.get("row_compact_plan")
        if row_plan is not None and row_plan["bucket_row_k_length"].numel() > 0:
            row_k = row_plan["bucket_row_k_length"].detach().cpu().float()
        else:
            row_k = torch.empty(0, dtype=torch.float32)
        union_k = direct_plan["bucket_k_length"].detach().cpu().float() if len(direct_plan["bucket_k_length"]) > 0 else torch.empty(0, dtype=torch.float32)
    else:
        segment_k = torch.empty(0, dtype=torch.float32)
        segment_fill = torch.empty(0, dtype=torch.float32)
        qgroup_num_segments = torch.empty(0, dtype=torch.float32)
        row_k = torch.empty(0, dtype=torch.float32)
        union_k = torch.empty(0, dtype=torch.float32)
        max_direct_segments = 0

    return {
        "logical_block_q": reference.logical_block_q,
        "logical_block_k": reference.logical_block_k,
        "max_packed_k": 0 if reference.max_packed_k is None else reference.max_packed_k,
        "max_direct_segments": max_direct_segments if reference.max_direct_segments is None else reference.max_direct_segments,
        "physical_block_q": reference.physical_block_q,
        "physical_block_k": reference.physical_block_k,
        "num_tiles": forward["num_tiles"],
        "avg_q_rows": forward["avg_q_rows"],
        "avg_k_rows": forward["avg_k_rows"],
        "avg_logical_pairs": forward["avg_logical_pairs"],
        "forward_tiles": forward["num_tiles"],
        "forward_qgroups": forward["num_qgroups"],
        "backward_tiles": backward["num_tiles"],
        "backward_qgroups": backward["num_qgroups"],
        "forward_buckets": forward["num_buckets"],
        "backward_buckets": backward["num_buckets"],
        "forward_avg_packed_q": forward["avg_packed_q"],
        "forward_avg_packed_k": forward["avg_packed_k"],
        "backward_avg_packed_q": backward["avg_packed_q"],
        "backward_avg_packed_k": backward["avg_packed_k"],
        "forward_avg_fill": forward["avg_fill"],
        "forward_fill_p50": forward["fill_p50"],
        "forward_fill_p90": forward["fill_p90"],
        "backward_avg_fill": backward["avg_fill"],
        "backward_fill_p50": backward["fill_p50"],
        "backward_fill_p90": backward["fill_p90"],
        "forward_avg_num_splits": forward["avg_num_splits"],
        "backward_avg_num_splits": backward["avg_num_splits"],
        "forward_dense_tiles": forward["dense_tiles"],
        "backward_dense_tiles": backward["dense_tiles"],
        "forward_avg_direct_segments": _mean_or_zero(qgroup_num_segments),
        "forward_avg_segment_k_length": _mean_or_zero(segment_k),
        "forward_avg_segment_fill": _mean_or_zero(segment_fill),
        "forward_segment_fill_p50": _quantile_or_zero(segment_fill, 0.5),
        "forward_segment_fill_p90": _quantile_or_zero(segment_fill, 0.9),
        "forward_avg_row_k": _mean_or_zero(row_k),
        "forward_row_k_p50": _quantile_or_zero(row_k, 0.5),
        "forward_row_k_p90": _quantile_or_zero(row_k, 0.9),
        "forward_avg_union_k": _mean_or_zero(union_k),
    }


def _empty_sentence_cache(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    device = q.device
    return (
        torch.empty(0, dtype=torch.float32, device=device),
        torch.empty(0, q.shape[2], q.shape[3], dtype=q.dtype, device=device),
        torch.empty(0, k.shape[2], k.shape[3], dtype=k.dtype, device=device),
        torch.empty(0, v.shape[2], v.shape[3], dtype=v.dtype, device=device),
        torch.empty(0, q.shape[2], v.shape[3], dtype=q.dtype, device=device),
    )


_LOG2_E = math.log2(math.e)


class FlashHSASyntheticPackRowsSm100:
    """Pack flat rows into a contiguous staged tensor, zeroing padded slots."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcRows: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstPacked: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_heads = mSrcRows.shape[1]
        head_dim = mSrcRows.shape[2]
        total_tasks = num_stream_rows * num_heads * head_dim
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcRows,
            mRowIdx,
            mDstPacked,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSrcRows: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstPacked: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcRows.shape[1])
            head_dim = Int32(mSrcRows.shape[2])
            elems_per_row = num_heads * head_dim
            stream_row = task_idx // elems_per_row
            rem = task_idx - stream_row * elems_per_row
            head_idx = rem // head_dim
            dim_idx = rem - head_idx * head_dim
            global_row = Int32(mRowIdx[stream_row])
            value = Float32(0.0)
            if global_row >= Int32(0):
                value = Float32(mSrcRows[global_row, head_idx, dim_idx])
            mDstPacked[stream_row, head_idx, dim_idx] = value.to(mDstPacked.element_type)


class FlashHSASyntheticPackKVRowsSm100:
    """Pack flat K/V rows into contiguous staged tensors in one pass."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcKRows: cute.Tensor,
        mSrcVRows: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstKPacked: cute.Tensor,
        mDstVPacked: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_heads = mSrcKRows.shape[1]
        head_dim = mSrcKRows.shape[2]
        total_tasks = num_stream_rows * num_heads * head_dim
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcKRows,
            mSrcVRows,
            mRowIdx,
            mDstKPacked,
            mDstVPacked,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSrcKRows: cute.Tensor,
        mSrcVRows: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstKPacked: cute.Tensor,
        mDstVPacked: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcKRows.shape[1])
            head_dim = Int32(mSrcKRows.shape[2])
            elems_per_row = num_heads * head_dim
            stream_row = task_idx // elems_per_row
            rem = task_idx - stream_row * elems_per_row
            head_idx = rem // head_dim
            dim_idx = rem - head_idx * head_dim
            global_row = Int32(mRowIdx[stream_row])
            k_value = Float32(0.0)
            v_value = Float32(0.0)
            if global_row >= Int32(0):
                k_value = Float32(mSrcKRows[global_row, head_idx, dim_idx])
                v_value = Float32(mSrcVRows[global_row, head_idx, dim_idx])
            mDstKPacked[stream_row, head_idx, dim_idx] = k_value.to(mDstKPacked.element_type)
            mDstVPacked[stream_row, head_idx, dim_idx] = v_value.to(mDstVPacked.element_type)


class FlashHSASyntheticScatterRowsSm100:
    """Scatter packed rows back to flat output rows, skipping padded slots."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcPacked: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_heads = mSrcPacked.shape[1]
        head_dim = mSrcPacked.shape[2]
        total_tasks = num_stream_rows * num_heads * head_dim
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcPacked,
            mRowIdx,
            mDstRows,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSrcPacked: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstRows: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcPacked.shape[1])
            head_dim = Int32(mSrcPacked.shape[2])
            elems_per_row = num_heads * head_dim
            stream_row = task_idx // elems_per_row
            rem = task_idx - stream_row * elems_per_row
            head_idx = rem // head_dim
            dim_idx = rem - head_idx * head_dim
            global_row = Int32(mRowIdx[stream_row])
            if global_row >= Int32(0):
                mDstRows[global_row, head_idx, dim_idx] = Float32(
                    mSrcPacked[stream_row, head_idx, dim_idx]
                ).to(mDstRows.element_type)


class FlashHSASyntheticScatterLSESm100:
    """Scatter packed LSE rows back to flat [B*T, H] rows."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_heads = mSrcLSE.shape[1]
        total_tasks = num_stream_rows * num_heads
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcLSE,
            mRowIdx,
            mDstLSE,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSrcLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstLSE: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcLSE.shape[1])
            stream_row = task_idx // num_heads
            head_idx = task_idx - stream_row * num_heads
            global_row = Int32(mRowIdx[stream_row])
            if global_row >= Int32(0):
                mDstLSE[global_row, head_idx] = Float32(mSrcLSE[stream_row, head_idx]).to(mDstLSE.element_type)


class FlashHSASyntheticCombineScatterRowsSm100:
    """Combine packed split outputs directly into final out/LSE rows."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcPacked: cute.Tensor,
        mSrcLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstRows: cute.Tensor,
        mDstLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_heads = mSrcPacked.shape[1]
        total_tasks = num_stream_rows * num_heads
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcPacked,
            mSrcLSE,
            mRowIdx,
            mDstRows,
            mDstLSE,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSrcPacked: cute.Tensor,
        mSrcLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstRows: cute.Tensor,
        mDstLSE: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcPacked.shape[1])
            stream_row = task_idx // num_heads
            head_idx = task_idx - stream_row * num_heads
            global_row = Int32(mRowIdx[stream_row])
            if global_row >= Int32(0):
                row_lse = Float32(mSrcLSE[stream_row, head_idx])
                prev_lse = Float32(mDstLSE[global_row, head_idx])
                next_lse = -Float32.inf
                prev_weight = Float32(0.0)
                row_weight = Float32(0.0)
                max_lse = prev_lse if prev_lse > row_lse else row_lse
                if max_lse == -Float32.inf:
                    next_lse = -Float32.inf
                else:
                    log2_e = math.log2(math.e)
                    prev_scale = cute.math.exp2(prev_lse * log2_e - max_lse * log2_e, fastmath=True)
                    row_scale = cute.math.exp2(row_lse * log2_e - max_lse * log2_e, fastmath=True)
                    sum_scale = 0.0
                    sum_scale += prev_scale
                    sum_scale += row_scale
                    next_lse = ssa_to_scalar(cute.math.log(scalar_to_ssa(sum_scale, Float32), fastmath=True)) + max_lse
                    inv_sum = 0.0 if (sum_scale == 0.0 or sum_scale != sum_scale) else 1.0 / sum_scale
                    prev_weight = prev_scale * inv_sum
                    row_weight = row_scale * inv_sum
                for dim_idx in range(mSrcPacked.shape[2]):
                    prev_out = Float32(mDstRows[global_row, head_idx, dim_idx])
                    row_out = Float32(mSrcPacked[stream_row, head_idx, dim_idx])
                    combined_out = prev_out * prev_weight + row_out * row_weight
                    mDstRows[global_row, head_idx, dim_idx] = combined_out.to(mDstRows.element_type)
                mDstLSE[global_row, head_idx] = next_lse.to(mDstLSE.element_type)


class FlashHSASyntheticMicroFwdDenseSm100:
    """Specialized forward kernel for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQPacked: cute.Tensor,
        mKPacked: cute.Tensor,
        mVPacked: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        softmax_scale: Float32,
        mOutPacked: cute.Tensor,
        mLSEPacked: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = mQPacked.shape[0]
        grid_y = mQPacked.shape[2]
        self.kernel(
            mQPacked,
            mKPacked,
            mVPacked,
            mQLength,
            mKLength,
            softmax_scale,
            mOutPacked,
            mLSEPacked,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQPacked: cute.Tensor,
        mKPacked: cute.Tensor,
        mVPacked: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        softmax_scale: Float32,
        mOutPacked: cute.Tensor,
        mLSEPacked: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        if warp_idx < Int32(2):
            q_length = Int32(mQLength[qgroup_idx])
            if warp_idx >= q_length:
                if lane == Int32(0):
                    mLSEPacked[qgroup_idx, warp_idx, head_idx] = -Float32.inf
                for dim_idx in range(lane, mOutPacked.shape[3], cute.arch.WARP_SIZE):
                    mOutPacked[qgroup_idx, warp_idx, head_idx, dim_idx] = Float32(0.0).to(
                        mOutPacked.element_type
                    )
            else:
                k_length = Int32(mKLength[qgroup_idx])
                row_idx = warp_idx
                score = -Float32.inf
                if lane < k_length:
                    if mQPacked.shape[3] in (64, 128):
                        score = Float32(0.0)
                        for dim_idx in range(0, mQPacked.shape[3], 4):
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 0]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 0]
                            )
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 1]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 1]
                            )
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 2]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 2]
                            )
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 3]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 3]
                            )
                    else:
                        score = Float32(0.0)
                        for dim_idx in range(mQPacked.shape[3]):
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx]
                            )
                    score *= softmax_scale
                row_max = utils.warp_reduce(score, utils.fmax)

                if row_max == -Float32.inf:
                    if lane == Int32(0):
                        mLSEPacked[qgroup_idx, row_idx, head_idx] = -Float32.inf
                    for dim_idx in range(lane, mOutPacked.shape[3], cute.arch.WARP_SIZE):
                        mOutPacked[qgroup_idx, row_idx, head_idx, dim_idx] = Float32(0.0).to(
                            mOutPacked.element_type
                        )
                else:
                    prob = Float32(0.0)
                    if lane < k_length:
                        prob = cute.math.exp2((score - row_max) * Float32(_LOG2_E), fastmath=True)
                    row_sum = utils.warp_reduce(prob, lambda a, b: a + b)
                    inv_row_sum = Float32(0.0) if row_sum == Float32(0.0) else Float32(1.0) / row_sum
                    for dim_idx in range(lane, mOutPacked.shape[3], cute.arch.WARP_SIZE):
                        acc_val = Float32(0.0)
                        for k_idx in range(mKPacked.shape[1]):
                            prob_k = utils.shuffle_sync(prob, k_idx)
                            acc_val += prob_k * Float32(mVPacked[qgroup_idx, k_idx, head_idx, dim_idx])
                        mOutPacked[qgroup_idx, row_idx, head_idx, dim_idx] = (acc_val * inv_row_sum).to(
                            mOutPacked.element_type
                        )
                    if lane == Int32(0):
                        lse = row_max + ssa_to_scalar(
                            cute.math.log(scalar_to_ssa(row_sum, Float32), fastmath=True)
                        )
                        mLSEPacked[qgroup_idx, row_idx, head_idx] = lse.to(mLSEPacked.element_type)


class FlashHSASyntheticMicroFwdMaskedSm100:
    """Masked synthetic micro forward for one-launch 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQPacked: cute.Tensor,
        mKPacked: cute.Tensor,
        mVPacked: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        mMaskWords: cute.Tensor,
        softmax_scale: Float32,
        mOutPacked: cute.Tensor,
        mLSEPacked: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = mQPacked.shape[0]
        grid_y = mQPacked.shape[2]
        self.kernel(
            mQPacked,
            mKPacked,
            mVPacked,
            mQLength,
            mKLength,
            mMaskWords,
            softmax_scale,
            mOutPacked,
            mLSEPacked,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQPacked: cute.Tensor,
        mKPacked: cute.Tensor,
        mVPacked: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        mMaskWords: cute.Tensor,
        softmax_scale: Float32,
        mOutPacked: cute.Tensor,
        mLSEPacked: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        if warp_idx < Int32(2):
            q_length = Int32(mQLength[qgroup_idx])
            if warp_idx >= q_length:
                if lane == Int32(0):
                    mLSEPacked[qgroup_idx, warp_idx, head_idx] = -Float32.inf
                for dim_idx in range(lane, mOutPacked.shape[3], cute.arch.WARP_SIZE):
                    mOutPacked[qgroup_idx, warp_idx, head_idx, dim_idx] = Float32(0.0).to(
                        mOutPacked.element_type
                    )
            else:
                k_length = Int32(mKLength[qgroup_idx])
                row_idx = warp_idx
                allowed = False
                if lane < k_length:
                    word_idx = lane // 32
                    bit_idx = lane % 32
                    mask_word = cutlass.Uint32(mMaskWords[qgroup_idx, row_idx, word_idx])
                    bit = utils.shr_u32(mask_word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)
                    allowed = bit != cutlass.Uint32(0)
                score = -Float32.inf
                if allowed:
                    if mQPacked.shape[3] in (64, 128):
                        score = Float32(0.0)
                        for dim_idx in range(0, mQPacked.shape[3], 4):
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 0]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 0]
                            )
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 1]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 1]
                            )
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 2]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 2]
                            )
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx + 3]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx + 3]
                            )
                    else:
                        score = Float32(0.0)
                        for dim_idx in range(mQPacked.shape[3]):
                            score += Float32(mQPacked[qgroup_idx, row_idx, head_idx, dim_idx]) * Float32(
                                mKPacked[qgroup_idx, lane, head_idx, dim_idx]
                            )
                    score *= softmax_scale
                row_max = utils.warp_reduce(score, utils.fmax)

                if row_max == -Float32.inf:
                    if lane == Int32(0):
                        mLSEPacked[qgroup_idx, row_idx, head_idx] = -Float32.inf
                    for dim_idx in range(lane, mOutPacked.shape[3], cute.arch.WARP_SIZE):
                        mOutPacked[qgroup_idx, row_idx, head_idx, dim_idx] = Float32(0.0).to(
                            mOutPacked.element_type
                        )
                else:
                    prob = Float32(0.0)
                    if allowed:
                        prob = cute.math.exp2((score - row_max) * Float32(_LOG2_E), fastmath=True)
                    row_sum = utils.warp_reduce(prob, lambda a, b: a + b)
                    inv_row_sum = Float32(0.0) if row_sum == Float32(0.0) else Float32(1.0) / row_sum
                    for dim_idx in range(lane, mOutPacked.shape[3], cute.arch.WARP_SIZE):
                        acc_val = Float32(0.0)
                        for k_idx in range(mKPacked.shape[1]):
                            prob_k = utils.shuffle_sync(prob, k_idx)
                            acc_val += prob_k * Float32(mVPacked[qgroup_idx, k_idx, head_idx, dim_idx])
                        mOutPacked[qgroup_idx, row_idx, head_idx, dim_idx] = (acc_val * inv_row_sum).to(
                            mOutPacked.element_type
                        )
                    if lane == Int32(0):
                        lse = row_max + ssa_to_scalar(
                            cute.math.log(scalar_to_ssa(row_sum, Float32), fastmath=True)
                        )
                        mLSEPacked[qgroup_idx, row_idx, head_idx] = lse.to(mLSEPacked.element_type)


def _run_synthetic_pack_rows_kernel(
    src_rows: torch.Tensor,
    row_idx: torch.Tensor,
    dst_packed: torch.Tensor,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_pack_rows_v1",
        src_rows.dtype,
        dst_packed.dtype,
        src_rows.shape[1],
        src_rows.shape[2],
        torch.cuda.get_device_capability(src_rows.device),
    )
    if compile_key not in _run_synthetic_pack_rows_kernel.compile_cache:
        kernel = FlashHSASyntheticPackRowsSm100()
        _run_synthetic_pack_rows_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(src_rows),
            to_cute_tensor(row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(dst_packed),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_pack_rows_kernel.compile_cache[compile_key](
        src_rows,
        row_idx,
        dst_packed,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_pack_rows_kernel.compile_cache = get_jit_cache("hsa_synth_pack_rows")


def _run_synthetic_pack_kv_rows_kernel(
    src_k_rows: torch.Tensor,
    src_v_rows: torch.Tensor,
    row_idx: torch.Tensor,
    dst_k_packed: torch.Tensor,
    dst_v_packed: torch.Tensor,
):
    _require_cute_runtime()
    if src_k_rows.shape[1:] != src_v_rows.shape[1:] or dst_k_packed.shape[1:] != dst_v_packed.shape[1:]:
        raise ValueError("Synthetic fused KV pack requires matching K/V shapes")
    compile_key = (
        "synthetic_pack_kv_rows_v1",
        src_k_rows.dtype,
        src_v_rows.dtype,
        dst_k_packed.dtype,
        dst_v_packed.dtype,
        src_k_rows.shape[1],
        src_k_rows.shape[2],
        torch.cuda.get_device_capability(src_k_rows.device),
    )
    if compile_key not in _run_synthetic_pack_kv_rows_kernel.compile_cache:
        kernel = FlashHSASyntheticPackKVRowsSm100()
        _run_synthetic_pack_kv_rows_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(src_k_rows),
            to_cute_tensor(src_v_rows),
            to_cute_tensor(row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(dst_k_packed),
            to_cute_tensor(dst_v_packed),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_pack_kv_rows_kernel.compile_cache[compile_key](
        src_k_rows,
        src_v_rows,
        row_idx,
        dst_k_packed,
        dst_v_packed,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_pack_kv_rows_kernel.compile_cache = get_jit_cache("hsa_synth_pack_kv_rows")


def _run_synthetic_scatter_rows_kernel(
    src_packed: torch.Tensor,
    row_idx: torch.Tensor,
    dst_rows: torch.Tensor,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_scatter_rows_v1",
        src_packed.dtype,
        dst_rows.dtype,
        src_packed.shape[1],
        src_packed.shape[2],
        torch.cuda.get_device_capability(src_packed.device),
    )
    if compile_key not in _run_synthetic_scatter_rows_kernel.compile_cache:
        kernel = FlashHSASyntheticScatterRowsSm100()
        _run_synthetic_scatter_rows_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(src_packed),
            to_cute_tensor(row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(dst_rows),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_scatter_rows_kernel.compile_cache[compile_key](
        src_packed,
        row_idx,
        dst_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_scatter_rows_kernel.compile_cache = get_jit_cache("hsa_synth_scatter_rows")


def _run_synthetic_scatter_lse_kernel(
    src_lse: torch.Tensor,
    row_idx: torch.Tensor,
    dst_lse: torch.Tensor,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_scatter_lse_v1",
        src_lse.dtype,
        dst_lse.dtype,
        src_lse.shape[1],
        torch.cuda.get_device_capability(src_lse.device),
    )
    if compile_key not in _run_synthetic_scatter_lse_kernel.compile_cache:
        kernel = FlashHSASyntheticScatterLSESm100()
        _run_synthetic_scatter_lse_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(src_lse, assumed_align=4),
            to_cute_tensor(row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(dst_lse, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_scatter_lse_kernel.compile_cache[compile_key](
        src_lse,
        row_idx,
        dst_lse,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_scatter_lse_kernel.compile_cache = get_jit_cache("hsa_synth_scatter_lse")


def _run_synthetic_combine_scatter_rows_kernel(
    src_packed: torch.Tensor,
    src_lse: torch.Tensor,
    row_idx: torch.Tensor,
    dst_rows: torch.Tensor,
    dst_lse: torch.Tensor,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_combine_scatter_rows_v1",
        src_packed.dtype,
        src_lse.dtype,
        dst_rows.dtype,
        dst_lse.dtype,
        src_packed.shape[1],
        src_packed.shape[2],
        torch.cuda.get_device_capability(src_packed.device),
    )
    if compile_key not in _run_synthetic_combine_scatter_rows_kernel.compile_cache:
        kernel = FlashHSASyntheticCombineScatterRowsSm100()
        _run_synthetic_combine_scatter_rows_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(src_packed),
            to_cute_tensor(src_lse, assumed_align=4),
            to_cute_tensor(row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(dst_rows, assumed_align=4),
            to_cute_tensor(dst_lse, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_combine_scatter_rows_kernel.compile_cache[compile_key](
        src_packed,
        src_lse,
        row_idx,
        dst_rows,
        dst_lse,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_combine_scatter_rows_kernel.compile_cache = get_jit_cache("hsa_synth_combine_scatter_rows")


class FlashHSASyntheticDirectCombineRowsSm100:
    """Combine direct micro partial outputs from a temporary global buffer."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcRows: cute.Tensor,
        mSrcLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstRows: cute.Tensor,
        mDstLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_heads = mSrcRows.shape[1]
        total_tasks = num_stream_rows * num_heads
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcRows,
            mSrcLSE,
            mRowIdx,
            mDstRows,
            mDstLSE,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mSrcRows: cute.Tensor,
        mSrcLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mDstRows: cute.Tensor,
        mDstLSE: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcRows.shape[1])
            stream_row = task_idx // num_heads
            head_idx = task_idx - stream_row * num_heads
            global_row = Int32(mRowIdx[stream_row])
            if global_row >= Int32(0):
                row_lse = Float32(mSrcLSE[global_row, head_idx])
                prev_lse = Float32(mDstLSE[global_row, head_idx])
                next_lse = -Float32.inf
                prev_weight = Float32(0.0)
                row_weight = Float32(0.0)
                max_lse = prev_lse if prev_lse > row_lse else row_lse
                if max_lse != -Float32.inf:
                    prev_scale = cute.math.exp2(prev_lse * Float32(_LOG2_E) - max_lse * Float32(_LOG2_E), fastmath=True)
                    row_scale = cute.math.exp2(row_lse * Float32(_LOG2_E) - max_lse * Float32(_LOG2_E), fastmath=True)
                    sum_scale = prev_scale + row_scale
                    next_lse = ssa_to_scalar(cute.math.log(scalar_to_ssa(sum_scale, Float32), fastmath=True)) + max_lse
                    inv_sum = 0.0 if (sum_scale == 0.0 or sum_scale != sum_scale) else 1.0 / sum_scale
                    prev_weight = prev_scale * inv_sum
                    row_weight = row_scale * inv_sum
                for dim_idx in range(mSrcRows.shape[2]):
                    prev_out = Float32(mDstRows[global_row, head_idx, dim_idx])
                    row_out = Float32(mSrcRows[global_row, head_idx, dim_idx])
                    mDstRows[global_row, head_idx, dim_idx] = (
                        prev_out * prev_weight + row_out * row_weight
                    ).to(mDstRows.element_type)
                mDstLSE[global_row, head_idx] = next_lse.to(mDstLSE.element_type)


def _run_synthetic_direct_combine_rows_kernel(
    src_rows: torch.Tensor,
    src_lse: torch.Tensor,
    row_idx: torch.Tensor,
    dst_rows: torch.Tensor,
    dst_lse: torch.Tensor,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_combine_rows_v1",
        src_rows.dtype,
        src_lse.dtype,
        dst_rows.dtype,
        dst_lse.dtype,
        src_rows.shape[1],
        src_rows.shape[2],
        torch.cuda.get_device_capability(src_rows.device),
    )
    if compile_key not in _run_synthetic_direct_combine_rows_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectCombineRowsSm100()
        _run_synthetic_direct_combine_rows_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(src_rows, assumed_align=4),
            to_cute_tensor(src_lse, assumed_align=4),
            to_cute_tensor(row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(dst_rows, assumed_align=4),
            to_cute_tensor(dst_lse, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_combine_rows_kernel.compile_cache[compile_key](
        src_rows,
        src_lse,
        row_idx,
        dst_rows,
        dst_lse,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_combine_rows_kernel.compile_cache = get_jit_cache("hsa_synth_direct_combine_rows")


def _can_use_synthetic_micro_fwd(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    *,
    packed_q: int,
    packed_k: int,
) -> bool:
    return (
        packed_q <= 2
        and packed_k > 0
        and q_rows.dtype in (torch.float16, torch.bfloat16)
        and k_rows.dtype == q_rows.dtype
        and v_rows.dtype == q_rows.dtype
        and q_rows.shape[-1] <= 128
        and v_rows.shape[-1] <= 128
    )


def _synthetic_micro_fwd_enabled() -> bool:
    import os

    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _synthetic_micro_bwd_enabled() -> bool:
    import os

    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_synthetic_qgroups_per_cta() -> int:
    import os

    value = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA", "2").strip().lower()
    try:
        parsed = int(value)
    except ValueError:
        parsed = 2
    if parsed not in (1, 2, 4):
        parsed = 2
    return parsed


def _get_synthetic_qgroups_per_cta_bwd() -> int:
    import os

    value = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_QGROUPS_PER_CTA_BWD", "2").strip().lower()
    try:
        parsed = int(value)
    except ValueError:
        parsed = 2
    if parsed not in (1, 2, 4):
        parsed = 2
    return parsed


def _synthetic_fused_bwd_enabled() -> bool:
    import os

    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_FUSED_BWD", "off").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_synthetic_one_kernel_bwd_mode() -> str:
    import os

    value = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ONE_KERNEL_BWD", "off").strip().lower()
    if value not in {"off", "on", "auto"}:
        value = "off"
    return value


def _synthetic_split_bwd_enabled() -> bool:
    import os

    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_SPLIT_BWD", "off").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_synthetic_row_bwd_accum_mode() -> str:
    import os

    value = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_ROW_BWD_ACCUM_MODE", "row_local").strip().lower()
    if value not in {"row_local", "union_local"}:
        value = "row_local"
    return value


def _get_synthetic_short_bwd_mode() -> str:
    import os

    value = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_SHORT_BWD", "off").strip().lower()
    if value not in {"auto", "on", "off"}:
        value = "off"
    return value


def _can_use_synthetic_micro_bwd(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    *,
    packed_q: int,
    packed_k: int,
) -> bool:
    return (
        _can_use_synthetic_micro_fwd(q_rows, k_rows, v_rows, packed_q=packed_q, packed_k=packed_k)
        and packed_q <= 2
    )


@cute.jit
def _direct_micro_dot_qk(
    mQRows: cute.Tensor,
    mKRows: cute.Tensor,
    global_q_row: Int32,
    key_row: Int32,
    head_idx: Int32,
):
    score = Float32(0.0)
    if mQRows.shape[2] in (64, 128):
        for dim_idx in range(0, mQRows.shape[2], 4):
            score += Float32(mQRows[global_q_row, head_idx, dim_idx + 0]) * Float32(
                mKRows[key_row, head_idx, dim_idx + 0]
            )
            score += Float32(mQRows[global_q_row, head_idx, dim_idx + 1]) * Float32(
                mKRows[key_row, head_idx, dim_idx + 1]
            )
            score += Float32(mQRows[global_q_row, head_idx, dim_idx + 2]) * Float32(
                mKRows[key_row, head_idx, dim_idx + 2]
            )
            score += Float32(mQRows[global_q_row, head_idx, dim_idx + 3]) * Float32(
                mKRows[key_row, head_idx, dim_idx + 3]
            )
    else:
        for dim_idx in range(mQRows.shape[2]):
            score += Float32(mQRows[global_q_row, head_idx, dim_idx]) * Float32(
                mKRows[key_row, head_idx, dim_idx]
            )
    return score


@cute.jit
def _direct_micro_dot_dout_v(
    mdORows: cute.Tensor,
    mVRows: cute.Tensor,
    global_q_row: Int32,
    key_row: Int32,
    head_idx: Int32,
):
    dprob = Float32(0.0)
    if mdORows.shape[2] in (64, 128):
        for dim_idx in range(0, mdORows.shape[2], 4):
            dprob += Float32(mdORows[global_q_row, head_idx, dim_idx + 0]) * Float32(
                mVRows[key_row, head_idx, dim_idx + 0]
            )
            dprob += Float32(mdORows[global_q_row, head_idx, dim_idx + 1]) * Float32(
                mVRows[key_row, head_idx, dim_idx + 1]
            )
            dprob += Float32(mdORows[global_q_row, head_idx, dim_idx + 2]) * Float32(
                mVRows[key_row, head_idx, dim_idx + 2]
            )
            dprob += Float32(mdORows[global_q_row, head_idx, dim_idx + 3]) * Float32(
                mVRows[key_row, head_idx, dim_idx + 3]
            )
    else:
        for dim_idx in range(mdORows.shape[2]):
            dprob += Float32(mdORows[global_q_row, head_idx, dim_idx]) * Float32(
                mVRows[key_row, head_idx, dim_idx]
            )
    return dprob


@cute.jit
def _direct_micro_is_allowed(
    mMaskWords: cute.Tensor,
    qgroup_idx: Int32,
    row_idx: Int32,
    key_slot: Int32,
):
    word_idx = key_slot // 32
    bit_idx = key_slot % 32
    mask_word = cutlass.Uint32(mMaskWords[qgroup_idx, row_idx, word_idx])
    bit = utils.shr_u32(mask_word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)
    return bit != cutlass.Uint32(0)


class FlashHSASyntheticDirectMicroFwdDenseSm100:
    """Direct sparse forward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        softmax_scale: Float32,
        mOutRows: cute.Tensor,
        mLSERows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = mQRowIdx.shape[0]
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mQRowIdx,
            mKRowIdx,
            mQLength,
            mKLength,
            softmax_scale,
            mOutRows,
            mLSERows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        softmax_scale: Float32,
        mOutRows: cute.Tensor,
        mLSERows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(mQRows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sK = smem.allocate_tensor(mKRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        sV = smem.allocate_tensor(mVRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        q_length = Int32(mQLength[qgroup_idx])
        for elem_idx in cutlass.range(tidx, Int32(2) * Int32(mQRows.shape[2]), self.num_threads, unroll=1):
            row_idx = elem_idx // Int32(mQRows.shape[2])
            dim_idx = elem_idx - row_idx * Int32(mQRows.shape[2])
            if row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                if global_q_row >= Int32(0):
                    sQ[row_idx, dim_idx] = mQRows[global_q_row, head_idx, dim_idx]
                else:
                    sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
            else:
                sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
        cute.arch.barrier()
        if warp_idx < Int32(2):
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if warp_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, warp_idx])
                active_row = global_q_row >= Int32(0)
            k_length = Int32(mKLength[qgroup_idx])
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            out0 = Float32(0.0)
            out1 = Float32(0.0)
            out2 = Float32(0.0)
            out3 = Float32(0.0)
            for chunk_start in range(0, mKRowIdx.shape[1], cute.arch.WARP_SIZE):
                for elem_idx in cutlass.range(
                    tidx,
                    cute.arch.WARP_SIZE * Int32(mKRows.shape[2]),
                    self.num_threads,
                    unroll=1,
                ):
                    key_rel = elem_idx // Int32(mKRows.shape[2])
                    dim_idx = elem_idx - key_rel * Int32(mKRows.shape[2])
                    key_slot_load = Int32(chunk_start) + key_rel
                    if key_slot_load < k_length:
                        key_row_load = Int32(mKRowIdx[qgroup_idx, key_slot_load])
                        if key_row_load >= Int32(0):
                            sK[key_rel, dim_idx] = mKRows[key_row_load, head_idx, dim_idx]
                            sV[key_rel, dim_idx] = mVRows[key_row_load, head_idx, dim_idx]
                        else:
                            sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                            sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                    else:
                        sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                cute.arch.barrier()
                if active_row:
                    key_slot = Int32(chunk_start) + lane
                    key_row = Int32(-1)
                    score = -Float32.inf
                    if key_slot < k_length:
                        key_row = Int32(mKRowIdx[qgroup_idx, key_slot])
                        if key_row >= Int32(0):
                            score = Float32(0.0)
                            if mQRows.shape[2] in (64, 128):
                                for dim_idx in range(0, mQRows.shape[2], 4):
                                    score += Float32(sQ[warp_idx, dim_idx + 0]) * Float32(sK[lane, dim_idx + 0])
                                    score += Float32(sQ[warp_idx, dim_idx + 1]) * Float32(sK[lane, dim_idx + 1])
                                    score += Float32(sQ[warp_idx, dim_idx + 2]) * Float32(sK[lane, dim_idx + 2])
                                    score += Float32(sQ[warp_idx, dim_idx + 3]) * Float32(sK[lane, dim_idx + 3])
                            else:
                                for dim_idx in range(mQRows.shape[2]):
                                    score += Float32(sQ[warp_idx, dim_idx]) * Float32(sK[lane, dim_idx])
                            score *= softmax_scale
                    chunk_max = utils.warp_reduce(score, utils.fmax)
                    if chunk_max != -Float32.inf:
                        prob = Float32(0.0)
                        if key_row >= Int32(0):
                            prob = cute.math.exp2((score - chunk_max) * Float32(_LOG2_E), fastmath=True)
                        chunk_sum = utils.warp_reduce(prob, lambda a, b: a + b)
                        next_max = row_max if row_max > chunk_max else chunk_max
                        prev_scale = Float32(0.0) if row_max == -Float32.inf else cute.math.exp2(
                            (row_max - next_max) * Float32(_LOG2_E), fastmath=True
                        )
                        chunk_scale = cute.math.exp2((chunk_max - next_max) * Float32(_LOG2_E), fastmath=True)
                        dim0 = lane
                        dim1 = lane + cute.arch.WARP_SIZE
                        dim2 = dim1 + cute.arch.WARP_SIZE
                        dim3 = dim2 + cute.arch.WARP_SIZE
                        chunk_out0 = Float32(0.0)
                        chunk_out1 = Float32(0.0)
                        chunk_out2 = Float32(0.0)
                        chunk_out3 = Float32(0.0)
                        for k_rel in range(cute.arch.WARP_SIZE):
                            prob_k = utils.shuffle_sync(prob, k_rel)
                            key_row_k = Int32(utils.shuffle_sync(key_row, k_rel))
                            if key_row_k >= Int32(0):
                                if dim0 < mVRows.shape[2]:
                                    chunk_out0 += prob_k * Float32(sV[k_rel, dim0])
                                if dim1 < mVRows.shape[2]:
                                    chunk_out1 += prob_k * Float32(sV[k_rel, dim1])
                                if dim2 < mVRows.shape[2]:
                                    chunk_out2 += prob_k * Float32(sV[k_rel, dim2])
                                if dim3 < mVRows.shape[2]:
                                    chunk_out3 += prob_k * Float32(sV[k_rel, dim3])
                        out0 = out0 * prev_scale + chunk_out0 * chunk_scale
                        out1 = out1 * prev_scale + chunk_out1 * chunk_scale
                        out2 = out2 * prev_scale + chunk_out2 * chunk_scale
                        out3 = out3 * prev_scale + chunk_out3 * chunk_scale
                        row_sum = row_sum * prev_scale + chunk_sum * chunk_scale
                        row_max = next_max
                cute.arch.barrier()
            if active_row:
                if row_max == -Float32.inf or row_sum == Float32(0.0):
                    if lane == Int32(0):
                        mLSERows[global_q_row, head_idx] = -Float32.inf
                    dim0 = lane
                    dim1 = lane + cute.arch.WARP_SIZE
                    dim2 = dim1 + cute.arch.WARP_SIZE
                    dim3 = dim2 + cute.arch.WARP_SIZE
                    if dim0 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim0] = Float32(0.0).to(mOutRows.element_type)
                    if dim1 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim1] = Float32(0.0).to(mOutRows.element_type)
                    if dim2 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim2] = Float32(0.0).to(mOutRows.element_type)
                    if dim3 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim3] = Float32(0.0).to(mOutRows.element_type)
                else:
                    inv_row_sum = Float32(1.0) / row_sum
                    dim0 = lane
                    dim1 = lane + cute.arch.WARP_SIZE
                    dim2 = dim1 + cute.arch.WARP_SIZE
                    dim3 = dim2 + cute.arch.WARP_SIZE
                    if dim0 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim0] = (out0 * inv_row_sum).to(mOutRows.element_type)
                    if dim1 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim1] = (out1 * inv_row_sum).to(mOutRows.element_type)
                    if dim2 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim2] = (out2 * inv_row_sum).to(mOutRows.element_type)
                    if dim3 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim3] = (out3 * inv_row_sum).to(mOutRows.element_type)
                    if lane == Int32(0):
                        mLSERows[global_q_row, head_idx] = (
                            row_max + ssa_to_scalar(cute.math.log(scalar_to_ssa(row_sum, Float32), fastmath=True))
                        ).to(mLSERows.element_type)


class FlashHSASyntheticDirectMicroFwdMaskedSm100:
    """Masked direct sparse forward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        mMaskWords: cute.Tensor,
        softmax_scale: Float32,
        mOutRows: cute.Tensor,
        mLSERows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = mQRowIdx.shape[0]
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mQRowIdx,
            mKRowIdx,
            mQLength,
            mKLength,
            mMaskWords,
            softmax_scale,
            mOutRows,
            mLSERows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        mMaskWords: cute.Tensor,
        softmax_scale: Float32,
        mOutRows: cute.Tensor,
        mLSERows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(mQRows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sK = smem.allocate_tensor(mKRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        sV = smem.allocate_tensor(mVRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        q_length = Int32(mQLength[qgroup_idx])
        for elem_idx in cutlass.range(tidx, Int32(2) * Int32(mQRows.shape[2]), self.num_threads, unroll=1):
            row_idx = elem_idx // Int32(mQRows.shape[2])
            dim_idx = elem_idx - row_idx * Int32(mQRows.shape[2])
            if row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                if global_q_row >= Int32(0):
                    sQ[row_idx, dim_idx] = mQRows[global_q_row, head_idx, dim_idx]
                else:
                    sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
            else:
                sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
        cute.arch.barrier()
        if warp_idx < Int32(2):
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if warp_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, warp_idx])
                active_row = global_q_row >= Int32(0)
            k_length = Int32(mKLength[qgroup_idx])
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            out0 = Float32(0.0)
            out1 = Float32(0.0)
            out2 = Float32(0.0)
            out3 = Float32(0.0)
            for chunk_start in range(0, mKRowIdx.shape[1], cute.arch.WARP_SIZE):
                for elem_idx in cutlass.range(
                    tidx,
                    cute.arch.WARP_SIZE * Int32(mKRows.shape[2]),
                    self.num_threads,
                    unroll=1,
                ):
                    key_rel = elem_idx // Int32(mKRows.shape[2])
                    dim_idx = elem_idx - key_rel * Int32(mKRows.shape[2])
                    key_slot_load = Int32(chunk_start) + key_rel
                    if key_slot_load < k_length:
                        key_row_load = Int32(mKRowIdx[qgroup_idx, key_slot_load])
                        if key_row_load >= Int32(0):
                            sK[key_rel, dim_idx] = mKRows[key_row_load, head_idx, dim_idx]
                            sV[key_rel, dim_idx] = mVRows[key_row_load, head_idx, dim_idx]
                        else:
                            sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                            sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                    else:
                        sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                cute.arch.barrier()
                if active_row:
                    key_slot = Int32(chunk_start) + lane
                    key_row = Int32(-1)
                    score = -Float32.inf
                    if key_slot < k_length and _direct_micro_is_allowed(mMaskWords, Int32(qgroup_idx), warp_idx, key_slot):
                        key_row = Int32(mKRowIdx[qgroup_idx, key_slot])
                        if key_row >= Int32(0):
                            score = Float32(0.0)
                            if mQRows.shape[2] in (64, 128):
                                for dim_idx in range(0, mQRows.shape[2], 4):
                                    score += Float32(sQ[warp_idx, dim_idx + 0]) * Float32(sK[lane, dim_idx + 0])
                                    score += Float32(sQ[warp_idx, dim_idx + 1]) * Float32(sK[lane, dim_idx + 1])
                                    score += Float32(sQ[warp_idx, dim_idx + 2]) * Float32(sK[lane, dim_idx + 2])
                                    score += Float32(sQ[warp_idx, dim_idx + 3]) * Float32(sK[lane, dim_idx + 3])
                            else:
                                for dim_idx in range(mQRows.shape[2]):
                                    score += Float32(sQ[warp_idx, dim_idx]) * Float32(sK[lane, dim_idx])
                            score *= softmax_scale
                    chunk_max = utils.warp_reduce(score, utils.fmax)
                    if chunk_max != -Float32.inf:
                        prob = Float32(0.0)
                        if key_row >= Int32(0):
                            prob = cute.math.exp2((score - chunk_max) * Float32(_LOG2_E), fastmath=True)
                        chunk_sum = utils.warp_reduce(prob, lambda a, b: a + b)
                        next_max = row_max if row_max > chunk_max else chunk_max
                        prev_scale = Float32(0.0) if row_max == -Float32.inf else cute.math.exp2(
                            (row_max - next_max) * Float32(_LOG2_E), fastmath=True
                        )
                        chunk_scale = cute.math.exp2((chunk_max - next_max) * Float32(_LOG2_E), fastmath=True)
                        dim0 = lane
                        dim1 = lane + cute.arch.WARP_SIZE
                        dim2 = dim1 + cute.arch.WARP_SIZE
                        dim3 = dim2 + cute.arch.WARP_SIZE
                        chunk_out0 = Float32(0.0)
                        chunk_out1 = Float32(0.0)
                        chunk_out2 = Float32(0.0)
                        chunk_out3 = Float32(0.0)
                        for k_rel in range(cute.arch.WARP_SIZE):
                            prob_k = utils.shuffle_sync(prob, k_rel)
                            key_row_k = Int32(utils.shuffle_sync(key_row, k_rel))
                            if key_row_k >= Int32(0):
                                if dim0 < mVRows.shape[2]:
                                    chunk_out0 += prob_k * Float32(sV[k_rel, dim0])
                                if dim1 < mVRows.shape[2]:
                                    chunk_out1 += prob_k * Float32(sV[k_rel, dim1])
                                if dim2 < mVRows.shape[2]:
                                    chunk_out2 += prob_k * Float32(sV[k_rel, dim2])
                                if dim3 < mVRows.shape[2]:
                                    chunk_out3 += prob_k * Float32(sV[k_rel, dim3])
                        out0 = out0 * prev_scale + chunk_out0 * chunk_scale
                        out1 = out1 * prev_scale + chunk_out1 * chunk_scale
                        out2 = out2 * prev_scale + chunk_out2 * chunk_scale
                        out3 = out3 * prev_scale + chunk_out3 * chunk_scale
                        row_sum = row_sum * prev_scale + chunk_sum * chunk_scale
                        row_max = next_max
                cute.arch.barrier()
            if active_row:
                if row_max == -Float32.inf or row_sum == Float32(0.0):
                    if lane == Int32(0):
                        mLSERows[global_q_row, head_idx] = -Float32.inf
                    dim0 = lane
                    dim1 = lane + cute.arch.WARP_SIZE
                    dim2 = dim1 + cute.arch.WARP_SIZE
                    dim3 = dim2 + cute.arch.WARP_SIZE
                    if dim0 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim0] = Float32(0.0).to(mOutRows.element_type)
                    if dim1 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim1] = Float32(0.0).to(mOutRows.element_type)
                    if dim2 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim2] = Float32(0.0).to(mOutRows.element_type)
                    if dim3 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim3] = Float32(0.0).to(mOutRows.element_type)
                else:
                    inv_row_sum = Float32(1.0) / row_sum
                    dim0 = lane
                    dim1 = lane + cute.arch.WARP_SIZE
                    dim2 = dim1 + cute.arch.WARP_SIZE
                    dim3 = dim2 + cute.arch.WARP_SIZE
                    if dim0 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim0] = (out0 * inv_row_sum).to(mOutRows.element_type)
                    if dim1 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim1] = (out1 * inv_row_sum).to(mOutRows.element_type)
                    if dim2 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim2] = (out2 * inv_row_sum).to(mOutRows.element_type)
                    if dim3 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim3] = (out3 * inv_row_sum).to(mOutRows.element_type)
                    if lane == Int32(0):
                        mLSERows[global_q_row, head_idx] = (
                            row_max + ssa_to_scalar(cute.math.log(scalar_to_ssa(row_sum, Float32), fastmath=True))
                        ).to(mLSERows.element_type)


class FlashHSASyntheticDirectMicroBwdDenseSm100:
    """Direct sparse backward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, num_threads: int = 64):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = mQRowIdx.shape[0]
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mKRowIdx,
            mQLength,
            mKLength,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(mQRows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sO = smem.allocate_tensor(mOutRows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sdO = smem.allocate_tensor(mdORows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sK = smem.allocate_tensor(mKRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        sV = smem.allocate_tensor(mVRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        q_length = Int32(mQLength[qgroup_idx])
        for elem_idx in cutlass.range(tidx, Int32(2) * Int32(mQRows.shape[2]), self.num_threads, unroll=1):
            row_idx = elem_idx // Int32(mQRows.shape[2])
            dim_idx = elem_idx - row_idx * Int32(mQRows.shape[2])
            if row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                if global_q_row >= Int32(0):
                    sQ[row_idx, dim_idx] = mQRows[global_q_row, head_idx, dim_idx]
                else:
                    sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
            else:
                sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
        for elem_idx in cutlass.range(tidx, Int32(2) * Int32(mOutRows.shape[2]), self.num_threads, unroll=1):
            row_idx = elem_idx // Int32(mOutRows.shape[2])
            dim_idx = elem_idx - row_idx * Int32(mOutRows.shape[2])
            if row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                if global_q_row >= Int32(0):
                    sO[row_idx, dim_idx] = mOutRows[global_q_row, head_idx, dim_idx]
                    sdO[row_idx, dim_idx] = mdORows[global_q_row, head_idx, dim_idx]
                else:
                    sO[row_idx, dim_idx] = Float32(0.0).to(sO.element_type)
                    sdO[row_idx, dim_idx] = Float32(0.0).to(sdO.element_type)
            else:
                sO[row_idx, dim_idx] = Float32(0.0).to(sO.element_type)
                sdO[row_idx, dim_idx] = Float32(0.0).to(sdO.element_type)
        cute.arch.barrier()
        if warp_idx < Int32(2):
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if warp_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, warp_idx])
                active_row = global_q_row >= Int32(0)
            dpsum = Float32(0.0)
            lse_log2 = Float32(0.0)
            if active_row:
                dpsum_partial = Float32(0.0)
                for dim_idx in range(lane, mOutRows.shape[2], cute.arch.WARP_SIZE):
                    dpsum_partial += Float32(sO[warp_idx, dim_idx]) * Float32(sdO[warp_idx, dim_idx])
                dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b)
                lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
            scale_log2 = softmax_scale * Float32(_LOG2_E)
            k_length = Int32(mKLength[qgroup_idx])
            dq0 = Float32(0.0)
            dq1 = Float32(0.0)
            dq2 = Float32(0.0)
            dq3 = Float32(0.0)
            for chunk_start in range(0, mKRowIdx.shape[1], cute.arch.WARP_SIZE):
                for elem_idx in cutlass.range(
                    tidx,
                    cute.arch.WARP_SIZE * Int32(mKRows.shape[2]),
                    self.num_threads,
                    unroll=1,
                ):
                    key_rel = elem_idx // Int32(mKRows.shape[2])
                    dim_idx = elem_idx - key_rel * Int32(mKRows.shape[2])
                    key_slot_load = Int32(chunk_start) + key_rel
                    if key_slot_load < k_length:
                        key_row_load = Int32(mKRowIdx[qgroup_idx, key_slot_load])
                        if key_row_load >= Int32(0):
                            sK[key_rel, dim_idx] = mKRows[key_row_load, head_idx, dim_idx]
                            sV[key_rel, dim_idx] = mVRows[key_row_load, head_idx, dim_idx]
                        else:
                            sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                            sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                    else:
                        sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                cute.arch.barrier()
                ds_scaled = Float32(0.0)
                if active_row:
                    key_slot = Int32(chunk_start) + lane
                    key_row = Int32(-1)
                    prob = Float32(0.0)
                    if key_slot < k_length:
                        key_row = Int32(mKRowIdx[qgroup_idx, key_slot])
                        if key_row >= Int32(0):
                            score = Float32(0.0)
                            if mQRows.shape[2] in (64, 128):
                                for dim_idx in range(0, mQRows.shape[2], 4):
                                    score += Float32(sQ[warp_idx, dim_idx + 0]) * Float32(sK[lane, dim_idx + 0])
                                    score += Float32(sQ[warp_idx, dim_idx + 1]) * Float32(sK[lane, dim_idx + 1])
                                    score += Float32(sQ[warp_idx, dim_idx + 2]) * Float32(sK[lane, dim_idx + 2])
                                    score += Float32(sQ[warp_idx, dim_idx + 3]) * Float32(sK[lane, dim_idx + 3])
                            else:
                                for dim_idx in range(mQRows.shape[2]):
                                    score += Float32(sQ[warp_idx, dim_idx]) * Float32(sK[lane, dim_idx])
                            prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                            dprob = Float32(0.0)
                            if mdORows.shape[2] in (64, 128):
                                for dim_idx in range(0, mdORows.shape[2], 4):
                                    dprob += Float32(sdO[warp_idx, dim_idx + 0]) * Float32(sV[lane, dim_idx + 0])
                                    dprob += Float32(sdO[warp_idx, dim_idx + 1]) * Float32(sV[lane, dim_idx + 1])
                                    dprob += Float32(sdO[warp_idx, dim_idx + 2]) * Float32(sV[lane, dim_idx + 2])
                                    dprob += Float32(sdO[warp_idx, dim_idx + 3]) * Float32(sV[lane, dim_idx + 3])
                            else:
                                for dim_idx in range(mdORows.shape[2]):
                                    dprob += Float32(sdO[warp_idx, dim_idx]) * Float32(sV[lane, dim_idx])
                            ds_scaled = prob * (dprob - dpsum) * softmax_scale
                            for dim_idx in range(lane, mQRows.shape[2], cute.arch.WARP_SIZE):
                                utils.atomic_add_fp32(
                                    ds_scaled * Float32(sQ[warp_idx, dim_idx]),
                                    utils.elem_pointer(mdKRows, (key_row, head_idx, dim_idx)),
                                )
                            for dim_idx in range(lane, mVRows.shape[2], cute.arch.WARP_SIZE):
                                utils.atomic_add_fp32(
                                    prob * Float32(sdO[warp_idx, dim_idx]),
                                    utils.elem_pointer(mdVRows, (key_row, head_idx, dim_idx)),
                                )
                dim0 = lane
                dim1 = lane + cute.arch.WARP_SIZE
                dim2 = dim1 + cute.arch.WARP_SIZE
                dim3 = dim2 + cute.arch.WARP_SIZE
                if active_row:
                    for k_rel in range(cute.arch.WARP_SIZE):
                        ds_k = utils.shuffle_sync(ds_scaled, k_rel)
                        if dim0 < mQRows.shape[2]:
                            dq0 += ds_k * Float32(sK[k_rel, dim0])
                        if dim1 < mQRows.shape[2]:
                            dq1 += ds_k * Float32(sK[k_rel, dim1])
                        if dim2 < mQRows.shape[2]:
                            dq2 += ds_k * Float32(sK[k_rel, dim2])
                        if dim3 < mQRows.shape[2]:
                            dq3 += ds_k * Float32(sK[k_rel, dim3])
                cute.arch.barrier()
            if active_row:
                dim0 = lane
                dim1 = lane + cute.arch.WARP_SIZE
                dim2 = dim1 + cute.arch.WARP_SIZE
                dim3 = dim2 + cute.arch.WARP_SIZE
                if dim0 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq0,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim0)),
                    )
                if dim1 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq1,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim1)),
                    )
                if dim2 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq2,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim2)),
                    )
                if dim3 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq3,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim3)),
                    )


class FlashHSASyntheticDirectMicroBwdMaskedSm100:
    """Masked direct sparse backward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, num_threads: int = 64):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        mMaskWords: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = mQRowIdx.shape[0]
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mKRowIdx,
            mQLength,
            mKLength,
            mMaskWords,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mKRowIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mKLength: cute.Tensor,
        mMaskWords: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(mQRows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sO = smem.allocate_tensor(mOutRows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sdO = smem.allocate_tensor(mdORows.element_type, cute.make_layout((2, 128)), byte_alignment=16)
        sK = smem.allocate_tensor(mKRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        sV = smem.allocate_tensor(mVRows.element_type, cute.make_layout((32, 128)), byte_alignment=16)
        q_length = Int32(mQLength[qgroup_idx])
        for elem_idx in cutlass.range(tidx, Int32(2) * Int32(mQRows.shape[2]), self.num_threads, unroll=1):
            row_idx = elem_idx // Int32(mQRows.shape[2])
            dim_idx = elem_idx - row_idx * Int32(mQRows.shape[2])
            if row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                if global_q_row >= Int32(0):
                    sQ[row_idx, dim_idx] = mQRows[global_q_row, head_idx, dim_idx]
                else:
                    sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
            else:
                sQ[row_idx, dim_idx] = Float32(0.0).to(sQ.element_type)
        for elem_idx in cutlass.range(tidx, Int32(2) * Int32(mOutRows.shape[2]), self.num_threads, unroll=1):
            row_idx = elem_idx // Int32(mOutRows.shape[2])
            dim_idx = elem_idx - row_idx * Int32(mOutRows.shape[2])
            if row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                if global_q_row >= Int32(0):
                    sO[row_idx, dim_idx] = mOutRows[global_q_row, head_idx, dim_idx]
                    sdO[row_idx, dim_idx] = mdORows[global_q_row, head_idx, dim_idx]
                else:
                    sO[row_idx, dim_idx] = Float32(0.0).to(sO.element_type)
                    sdO[row_idx, dim_idx] = Float32(0.0).to(sdO.element_type)
            else:
                sO[row_idx, dim_idx] = Float32(0.0).to(sO.element_type)
                sdO[row_idx, dim_idx] = Float32(0.0).to(sdO.element_type)
        cute.arch.barrier()
        if warp_idx < Int32(2):
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if warp_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, warp_idx])
                active_row = global_q_row >= Int32(0)
            dpsum = Float32(0.0)
            lse_log2 = Float32(0.0)
            if active_row:
                dpsum_partial = Float32(0.0)
                for dim_idx in range(lane, mOutRows.shape[2], cute.arch.WARP_SIZE):
                    dpsum_partial += Float32(sO[warp_idx, dim_idx]) * Float32(sdO[warp_idx, dim_idx])
                dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b)
                lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
            scale_log2 = softmax_scale * Float32(_LOG2_E)
            k_length = Int32(mKLength[qgroup_idx])
            dq0 = Float32(0.0)
            dq1 = Float32(0.0)
            dq2 = Float32(0.0)
            dq3 = Float32(0.0)
            for chunk_start in range(0, mKRowIdx.shape[1], cute.arch.WARP_SIZE):
                for elem_idx in cutlass.range(
                    tidx,
                    cute.arch.WARP_SIZE * Int32(mKRows.shape[2]),
                    self.num_threads,
                    unroll=1,
                ):
                    key_rel = elem_idx // Int32(mKRows.shape[2])
                    dim_idx = elem_idx - key_rel * Int32(mKRows.shape[2])
                    key_slot_load = Int32(chunk_start) + key_rel
                    if key_slot_load < k_length:
                        key_row_load = Int32(mKRowIdx[qgroup_idx, key_slot_load])
                        if key_row_load >= Int32(0):
                            sK[key_rel, dim_idx] = mKRows[key_row_load, head_idx, dim_idx]
                            sV[key_rel, dim_idx] = mVRows[key_row_load, head_idx, dim_idx]
                        else:
                            sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                            sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                    else:
                        sK[key_rel, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[key_rel, dim_idx] = Float32(0.0).to(sV.element_type)
                cute.arch.barrier()
                ds_scaled = Float32(0.0)
                if active_row:
                    key_slot = Int32(chunk_start) + lane
                    key_row = Int32(-1)
                    prob = Float32(0.0)
                    if key_slot < k_length and _direct_micro_is_allowed(mMaskWords, Int32(qgroup_idx), warp_idx, key_slot):
                        key_row = Int32(mKRowIdx[qgroup_idx, key_slot])
                        if key_row >= Int32(0):
                            score = Float32(0.0)
                            if mQRows.shape[2] in (64, 128):
                                for dim_idx in range(0, mQRows.shape[2], 4):
                                    score += Float32(sQ[warp_idx, dim_idx + 0]) * Float32(sK[lane, dim_idx + 0])
                                    score += Float32(sQ[warp_idx, dim_idx + 1]) * Float32(sK[lane, dim_idx + 1])
                                    score += Float32(sQ[warp_idx, dim_idx + 2]) * Float32(sK[lane, dim_idx + 2])
                                    score += Float32(sQ[warp_idx, dim_idx + 3]) * Float32(sK[lane, dim_idx + 3])
                            else:
                                for dim_idx in range(mQRows.shape[2]):
                                    score += Float32(sQ[warp_idx, dim_idx]) * Float32(sK[lane, dim_idx])
                            prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                            dprob = Float32(0.0)
                            if mdORows.shape[2] in (64, 128):
                                for dim_idx in range(0, mdORows.shape[2], 4):
                                    dprob += Float32(sdO[warp_idx, dim_idx + 0]) * Float32(sV[lane, dim_idx + 0])
                                    dprob += Float32(sdO[warp_idx, dim_idx + 1]) * Float32(sV[lane, dim_idx + 1])
                                    dprob += Float32(sdO[warp_idx, dim_idx + 2]) * Float32(sV[lane, dim_idx + 2])
                                    dprob += Float32(sdO[warp_idx, dim_idx + 3]) * Float32(sV[lane, dim_idx + 3])
                            else:
                                for dim_idx in range(mdORows.shape[2]):
                                    dprob += Float32(sdO[warp_idx, dim_idx]) * Float32(sV[lane, dim_idx])
                            ds_scaled = prob * (dprob - dpsum) * softmax_scale
                            for dim_idx in range(lane, mQRows.shape[2], cute.arch.WARP_SIZE):
                                utils.atomic_add_fp32(
                                    ds_scaled * Float32(sQ[warp_idx, dim_idx]),
                                    utils.elem_pointer(mdKRows, (key_row, head_idx, dim_idx)),
                                )
                            for dim_idx in range(lane, mVRows.shape[2], cute.arch.WARP_SIZE):
                                utils.atomic_add_fp32(
                                    prob * Float32(sdO[warp_idx, dim_idx]),
                                    utils.elem_pointer(mdVRows, (key_row, head_idx, dim_idx)),
                                )
                dim0 = lane
                dim1 = lane + cute.arch.WARP_SIZE
                dim2 = dim1 + cute.arch.WARP_SIZE
                dim3 = dim2 + cute.arch.WARP_SIZE
                if active_row:
                    for k_rel in range(cute.arch.WARP_SIZE):
                        ds_k = utils.shuffle_sync(ds_scaled, k_rel)
                        if dim0 < mQRows.shape[2]:
                            dq0 += ds_k * Float32(sK[k_rel, dim0])
                        if dim1 < mQRows.shape[2]:
                            dq1 += ds_k * Float32(sK[k_rel, dim1])
                        if dim2 < mQRows.shape[2]:
                            dq2 += ds_k * Float32(sK[k_rel, dim2])
                        if dim3 < mQRows.shape[2]:
                            dq3 += ds_k * Float32(sK[k_rel, dim3])
                cute.arch.barrier()
            if active_row:
                dim0 = lane
                dim1 = lane + cute.arch.WARP_SIZE
                dim2 = dim1 + cute.arch.WARP_SIZE
                dim3 = dim2 + cute.arch.WARP_SIZE
                if dim0 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq0,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim0)),
                    )
                if dim1 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq1,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim1)),
                    )
                if dim2 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq2,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim2)),
                    )
                if dim3 < mdQRows.shape[2]:
                    utils.atomic_add_fp32(
                        dq3,
                        utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim3)),
                    )


def _run_synthetic_direct_micro_fwd_dense_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    k_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    out_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_micro_fwd_dense_v2",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        q_row_idx.shape[1],
        k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_micro_fwd_dense_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectMicroFwdDenseSm100()
        _run_synthetic_direct_micro_fwd_dense_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(k_row_idx, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(out_rows, assumed_align=4),
            to_cute_tensor(lse_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_micro_fwd_dense_kernel.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        Float32(softmax_scale),
        out_rows,
        lse_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_micro_fwd_dense_kernel.compile_cache = get_jit_cache("hsa_synth_direct_micro_fwd_dense")


def _run_synthetic_direct_micro_fwd_masked_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    k_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    mask_words: torch.Tensor,
    out_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_micro_fwd_masked_v2",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        q_row_idx.shape[1],
        k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        mask_words.shape[2],
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_micro_fwd_masked_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectMicroFwdMaskedSm100()
        _run_synthetic_direct_micro_fwd_masked_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(k_row_idx, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(k_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(mask_words, assumed_align=4, leading_dim=2),
            Float32(softmax_scale),
            to_cute_tensor(out_rows, assumed_align=4),
            to_cute_tensor(lse_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_micro_fwd_masked_kernel.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        Float32(softmax_scale),
        out_rows,
        lse_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_micro_fwd_masked_kernel.compile_cache = get_jit_cache("hsa_synth_direct_micro_fwd_masked")


def _run_synthetic_direct_micro_bwd_dense_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    k_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_micro_bwd_dense_v2",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_micro_bwd_dense_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectMicroBwdDenseSm100()
        _run_synthetic_direct_micro_bwd_dense_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(k_row_idx, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_micro_bwd_dense_kernel.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_micro_bwd_dense_kernel.compile_cache = get_jit_cache("hsa_synth_direct_micro_bwd_dense")


def _run_synthetic_direct_micro_bwd_masked_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    k_row_idx: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    mask_words: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_micro_bwd_masked_v2",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        mask_words.shape[2],
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_micro_bwd_masked_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectMicroBwdMaskedSm100()
        _run_synthetic_direct_micro_bwd_masked_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(k_row_idx, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(k_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(mask_words, assumed_align=4, leading_dim=2),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_micro_bwd_masked_kernel.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_micro_bwd_masked_kernel.compile_cache = get_jit_cache("hsa_synth_direct_micro_bwd_masked")


class FlashHSASyntheticDirectRowMicroFwdSm100:
    """Row-compact direct sparse forward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mOutRows: cute.Tensor,
        mLSERows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = (mQRowIdx.shape[0] + self.qgroups_per_cta - 1) // self.qgroups_per_cta
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mQRowIdx,
            mRowKRowIdx,
            mUnionKRowIdx,
            mRowKToUnionIdx,
            mQLength,
            mRowKLength,
            mUnionKLength,
            softmax_scale,
            mOutRows,
            mLSERows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mOutRows: cute.Tensor,
        mLSERows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_pair_idx, head_idx, _ = cute.arch.block_idx()
        subwarp_idx = tidx // Int32(16)
        qgroup_in_cta = subwarp_idx // Int32(2)
        row_idx = subwarp_idx % Int32(2)
        lane = tidx % Int32(16)
        qgroup_base = qgroup_pair_idx * Int32(self.qgroups_per_cta)
        qgroup_count = Int32(mQRowIdx.shape[0])
        dim0 = lane * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        smem = cutlass.utils.SmemAllocator()
        sScores = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.qgroups_per_cta * 2, 16)), byte_alignment=16
        )
        sProbs = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.qgroups_per_cta * 2, 16)), byte_alignment=16
        )
        sK = smem.allocate_tensor(
            mKRows.element_type, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        sV = smem.allocate_tensor(
            mVRows.element_type, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        for elem_idx in cutlass.range(
            tidx, Int32(self.qgroups_per_cta) * Int32(16) * Int32(64), self.num_threads, unroll=1
        ):
            qgroup_load = elem_idx // (Int32(16) * Int32(64))
            rem = elem_idx - qgroup_load * Int32(16) * Int32(64)
            union_slot = rem // Int32(64)
            dim_idx = rem - union_slot * Int32(64)
            qgroup_idx = qgroup_base + qgroup_load
            if qgroup_idx < qgroup_count:
                union_k_length = Int32(mUnionKLength[qgroup_idx])
                if union_slot < union_k_length:
                    union_key_row = Int32(mUnionKRowIdx[qgroup_idx, union_slot])
                    if union_key_row >= Int32(0):
                        sK[qgroup_load, union_slot, dim_idx] = mKRows[union_key_row, head_idx, dim_idx]
                        sV[qgroup_load, union_slot, dim_idx] = mVRows[union_key_row, head_idx, dim_idx]
                    else:
                        sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
                else:
                    sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                    sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
            else:
                sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
        cute.arch.barrier()
        if subwarp_idx < Int32(self.qgroups_per_cta * 2):
            qgroup_idx = qgroup_base + qgroup_in_cta
            qgroup_valid = qgroup_idx < qgroup_count
            q_length = Int32(0)
            if qgroup_valid:
                q_length = Int32(mQLength[qgroup_idx])
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if qgroup_valid and row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                active_row = global_q_row >= Int32(0)
            row_k_length = Int32(0)
            if qgroup_valid:
                row_k_length = Int32(mRowKLength[qgroup_idx, row_idx])
            q0 = Float32(0.0)
            q1 = Float32(0.0)
            q2 = Float32(0.0)
            q3 = Float32(0.0)
            if active_row:
                if dim0 < mQRows.shape[2]:
                    q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                if dim1 < mQRows.shape[2]:
                    q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                if dim2 < mQRows.shape[2]:
                    q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                if dim3 < mQRows.shape[2]:
                    q3 = Float32(mQRows[global_q_row, head_idx, dim3])
            for key_slot in range(mRowKRowIdx.shape[2]):
                score_partial = Float32(0.0)
                union_slot = Int32(-1)
                if active_row and Int32(key_slot) < row_k_length:
                    union_slot = Int32(mRowKToUnionIdx[qgroup_idx, row_idx, key_slot])
                    if union_slot >= Int32(0):
                        if dim0 < mQRows.shape[2]:
                            score_partial += q0 * Float32(sK[qgroup_in_cta, union_slot, dim0])
                        if dim1 < mQRows.shape[2]:
                            score_partial += q1 * Float32(sK[qgroup_in_cta, union_slot, dim1])
                        if dim2 < mQRows.shape[2]:
                            score_partial += q2 * Float32(sK[qgroup_in_cta, union_slot, dim2])
                        if dim3 < mQRows.shape[2]:
                            score_partial += q3 * Float32(sK[qgroup_in_cta, union_slot, dim3])
                score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
                if lane == Int32(0):
                    sScores[subwarp_idx, key_slot] = (
                        score * softmax_scale if active_row and Int32(key_slot) < row_k_length and union_slot >= Int32(0) else -Float32.inf
                    )
                    sProbs[subwarp_idx, key_slot] = Float32(0.0)
            cute.arch.sync_warp()
            row_max = -Float32.inf
            row_sum = Float32(0.0)
            if lane == Int32(0):
                if active_row:
                    for key_slot in range(mRowKRowIdx.shape[2]):
                        if Int32(key_slot) < row_k_length:
                            score = Float32(sScores[subwarp_idx, key_slot])
                            row_max = score if row_max == -Float32.inf or score > row_max else row_max
                    if row_max != -Float32.inf:
                        for key_slot in range(mRowKRowIdx.shape[2]):
                            if Int32(key_slot) < row_k_length:
                                prob = cute.math.exp2(
                                    (Float32(sScores[subwarp_idx, key_slot]) - row_max) * Float32(_LOG2_E), fastmath=True
                                )
                                sProbs[subwarp_idx, key_slot] = prob
                                row_sum += prob
            cute.arch.sync_warp()
            if active_row:
                row_max = utils.shuffle_sync(row_max, 0, width=16)
                row_sum = utils.shuffle_sync(row_sum, 0, width=16)
                if row_max == -Float32.inf or row_sum == Float32(0.0):
                    if lane == Int32(0):
                        mLSERows[global_q_row, head_idx] = -Float32.inf
                    if dim0 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim0] = Float32(0.0).to(mOutRows.element_type)
                    if dim1 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim1] = Float32(0.0).to(mOutRows.element_type)
                    if dim2 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim2] = Float32(0.0).to(mOutRows.element_type)
                    if dim3 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim3] = Float32(0.0).to(mOutRows.element_type)
                else:
                    inv_row_sum = Float32(1.0) / row_sum
                    out0 = Float32(0.0)
                    out1 = Float32(0.0)
                    out2 = Float32(0.0)
                    out3 = Float32(0.0)
                    for key_slot in range(mRowKRowIdx.shape[2]):
                        if Int32(key_slot) < row_k_length:
                            union_slot = Int32(mRowKToUnionIdx[qgroup_idx, row_idx, key_slot])
                            prob = Float32(sProbs[subwarp_idx, key_slot]) * inv_row_sum
                            if union_slot >= Int32(0):
                                if dim0 < mOutRows.shape[2]:
                                    out0 += prob * Float32(sV[qgroup_in_cta, union_slot, dim0])
                                if dim1 < mOutRows.shape[2]:
                                    out1 += prob * Float32(sV[qgroup_in_cta, union_slot, dim1])
                                if dim2 < mOutRows.shape[2]:
                                    out2 += prob * Float32(sV[qgroup_in_cta, union_slot, dim2])
                                if dim3 < mOutRows.shape[2]:
                                    out3 += prob * Float32(sV[qgroup_in_cta, union_slot, dim3])
                    if dim0 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim0] = out0.to(mOutRows.element_type)
                    if dim1 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim1] = out1.to(mOutRows.element_type)
                    if dim2 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim2] = out2.to(mOutRows.element_type)
                    if dim3 < mOutRows.shape[2]:
                        mOutRows[global_q_row, head_idx, dim3] = out3.to(mOutRows.element_type)
                    if lane == Int32(0):
                        mLSERows[global_q_row, head_idx] = (
                            row_max + ssa_to_scalar(cute.math.log(scalar_to_ssa(row_sum, Float32), fastmath=True))
                        ).to(mLSERows.element_type)


class FlashHSASyntheticDirectRowMicroBwdShortSm100:
    """Union-centric short backward for small-key row-compact 2x2 synthetic buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        del mRowKRowIdx, mRowKToUnionIdx, mRowKLength
        grid_x = (mQRowIdx.shape[0] + self.qgroups_per_cta - 1) // self.qgroups_per_cta
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mUnionKRowIdx,
            mUnionToRowSlot,
            mQLength,
            mUnionKLength,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_pair_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row_idx = lane // Int32(16)
        lane16 = lane % Int32(16)
        dim0 = lane16 * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        qgroup_base = qgroup_pair_idx * Int32(self.qgroups_per_cta)
        qgroup_idx = qgroup_base + warp_idx
        qgroup_count = Int32(mQRowIdx.shape[0])
        qgroup_valid = qgroup_idx < qgroup_count
        q_length = Int32(0)
        if qgroup_valid:
            q_length = Int32(mQLength[qgroup_idx])
        global_q_row = Int32(-1)
        active_row = Boolean(False)
        if qgroup_valid and row_idx < q_length:
            global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
            active_row = global_q_row >= Int32(0)

        q0 = Float32(0.0)
        q1 = Float32(0.0)
        q2 = Float32(0.0)
        q3 = Float32(0.0)
        do0 = Float32(0.0)
        do1 = Float32(0.0)
        do2 = Float32(0.0)
        do3 = Float32(0.0)
        out0 = Float32(0.0)
        out1 = Float32(0.0)
        out2 = Float32(0.0)
        out3 = Float32(0.0)
        if active_row:
            if dim0 < mQRows.shape[2]:
                q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                do0 = Float32(mdORows[global_q_row, head_idx, dim0])
            if dim1 < mQRows.shape[2]:
                q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                do1 = Float32(mdORows[global_q_row, head_idx, dim1])
            if dim2 < mQRows.shape[2]:
                q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                do2 = Float32(mdORows[global_q_row, head_idx, dim2])
            if dim3 < mQRows.shape[2]:
                q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                do3 = Float32(mdORows[global_q_row, head_idx, dim3])

        dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
        dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
        lse_log2 = Float32(0.0)
        if active_row:
            lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
        scale_log2 = softmax_scale * Float32(_LOG2_E)
        dq0 = Float32(0.0)
        dq1 = Float32(0.0)
        dq2 = Float32(0.0)
        dq3 = Float32(0.0)
        union_k_length = Int32(0)
        if qgroup_valid:
            union_k_length = Int32(mUnionKLength[qgroup_idx])

        for union_slot in range(mUnionKRowIdx.shape[1]):
            union_slot_i = Int32(union_slot)
            union_key_row = Int32(-1)
            if qgroup_valid and union_slot_i < union_k_length:
                union_key_row = Int32(mUnionKRowIdx[qgroup_idx, union_slot_i])
            participates = Boolean(False)
            score_partial = Float32(0.0)
            dprob_partial = Float32(0.0)
            if active_row and union_slot_i < union_k_length and union_key_row >= Int32(0):
                row_slot = Int32(mUnionToRowSlot[qgroup_idx, row_idx, union_slot_i])
                participates = row_slot >= Int32(0)
                if participates:
                    if dim0 < mQRows.shape[2]:
                        kval0 = Float32(mKRows[union_key_row, head_idx, dim0])
                        vval0 = Float32(mVRows[union_key_row, head_idx, dim0])
                        score_partial += q0 * kval0
                        dprob_partial += do0 * vval0
                    if dim1 < mQRows.shape[2]:
                        kval1 = Float32(mKRows[union_key_row, head_idx, dim1])
                        vval1 = Float32(mVRows[union_key_row, head_idx, dim1])
                        score_partial += q1 * kval1
                        dprob_partial += do1 * vval1
                    if dim2 < mQRows.shape[2]:
                        kval2 = Float32(mKRows[union_key_row, head_idx, dim2])
                        vval2 = Float32(mVRows[union_key_row, head_idx, dim2])
                        score_partial += q2 * kval2
                        dprob_partial += do2 * vval2
                    if dim3 < mQRows.shape[2]:
                        kval3 = Float32(mKRows[union_key_row, head_idx, dim3])
                        vval3 = Float32(mVRows[union_key_row, head_idx, dim3])
                        score_partial += q3 * kval3
                        dprob_partial += do3 * vval3
            score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
            dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
            prob = Float32(0.0)
            ds_scaled = Float32(0.0)
            if lane16 == Int32(0) and participates:
                prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                ds_scaled = prob * (dprob - dpsum) * softmax_scale
            prob = utils.shuffle_sync(prob, 0, width=16)
            ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)

            dk0 = Float32(0.0)
            dk1 = Float32(0.0)
            dk2 = Float32(0.0)
            dk3 = Float32(0.0)
            dv0 = Float32(0.0)
            dv1 = Float32(0.0)
            dv2 = Float32(0.0)
            dv3 = Float32(0.0)
            if participates:
                if dim0 < mQRows.shape[2]:
                    kval0 = Float32(mKRows[union_key_row, head_idx, dim0])
                    dq0 += ds_scaled * kval0
                    dk0 = ds_scaled * q0
                    dv0 = prob * do0
                if dim1 < mQRows.shape[2]:
                    kval1 = Float32(mKRows[union_key_row, head_idx, dim1])
                    dq1 += ds_scaled * kval1
                    dk1 = ds_scaled * q1
                    dv1 = prob * do1
                if dim2 < mQRows.shape[2]:
                    kval2 = Float32(mKRows[union_key_row, head_idx, dim2])
                    dq2 += ds_scaled * kval2
                    dk2 = ds_scaled * q2
                    dv2 = prob * do2
                if dim3 < mQRows.shape[2]:
                    kval3 = Float32(mKRows[union_key_row, head_idx, dim3])
                    dq3 += ds_scaled * kval3
                    dk3 = ds_scaled * q3
                    dv3 = prob * do3
            partner_lane = lane16 + (Int32(1) - row_idx) * Int32(16)
            partner_dk0 = utils.shuffle_sync(dk0, partner_lane, width=32)
            partner_dk1 = utils.shuffle_sync(dk1, partner_lane, width=32)
            partner_dk2 = utils.shuffle_sync(dk2, partner_lane, width=32)
            partner_dk3 = utils.shuffle_sync(dk3, partner_lane, width=32)
            partner_dv0 = utils.shuffle_sync(dv0, partner_lane, width=32)
            partner_dv1 = utils.shuffle_sync(dv1, partner_lane, width=32)
            partner_dv2 = utils.shuffle_sync(dv2, partner_lane, width=32)
            partner_dv3 = utils.shuffle_sync(dv3, partner_lane, width=32)
            if lane < Int32(16) and union_slot_i < union_k_length and union_key_row >= Int32(0):
                if dim0 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk0 + partner_dk0,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim0)),
                    )
                    utils.atomic_add_fp32(
                        dv0 + partner_dv0,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim0)),
                    )
                if dim1 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk1 + partner_dk1,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim1)),
                    )
                    utils.atomic_add_fp32(
                        dv1 + partner_dv1,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim1)),
                    )
                if dim2 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk2 + partner_dk2,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim2)),
                    )
                    utils.atomic_add_fp32(
                        dv2 + partner_dv2,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim2)),
                    )
                if dim3 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk3 + partner_dk3,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim3)),
                    )
                    utils.atomic_add_fp32(
                        dv3 + partner_dv3,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim3)),
                    )

        if active_row:
            if dim0 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim0] = dq0
            if dim1 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim1] = dq1
            if dim2 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim2] = dq2
            if dim3 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim3] = dq3


class FlashHSASyntheticDirectRowMicroBwdFusedSm100:
    """One-qgroup union-centric fused backward for small-key row-compact 2x2 buckets."""

    arch = 100

    def __init__(self):
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        del mRowKRowIdx, mRowKToUnionIdx, mRowKLength
        grid_x = mQRowIdx.shape[0]
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mUnionKRowIdx,
            mUnionToRowSlot,
            mQLength,
            mUnionKLength,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_idx, head_idx, _ = cute.arch.block_idx()
        lane = tidx % cute.arch.WARP_SIZE
        row_idx = lane // Int32(16)
        lane16 = lane % Int32(16)
        dim0 = lane16 * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        qgroup_count = Int32(mQRowIdx.shape[0])
        qgroup_valid = qgroup_idx < qgroup_count

        q_length = Int32(0)
        if qgroup_valid:
            q_length = Int32(mQLength[qgroup_idx])
        global_q_row = Int32(-1)
        active_row = Boolean(False)
        if qgroup_valid and row_idx < q_length:
            global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
            active_row = global_q_row >= Int32(0)

        q0 = Float32(0.0)
        q1 = Float32(0.0)
        q2 = Float32(0.0)
        q3 = Float32(0.0)
        do0 = Float32(0.0)
        do1 = Float32(0.0)
        do2 = Float32(0.0)
        do3 = Float32(0.0)
        out0 = Float32(0.0)
        out1 = Float32(0.0)
        out2 = Float32(0.0)
        out3 = Float32(0.0)
        if active_row:
            if dim0 < mQRows.shape[2]:
                q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                do0 = Float32(mdORows[global_q_row, head_idx, dim0])
            if dim1 < mQRows.shape[2]:
                q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                do1 = Float32(mdORows[global_q_row, head_idx, dim1])
            if dim2 < mQRows.shape[2]:
                q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                do2 = Float32(mdORows[global_q_row, head_idx, dim2])
            if dim3 < mQRows.shape[2]:
                q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                do3 = Float32(mdORows[global_q_row, head_idx, dim3])

        dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
        dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
        lse_log2 = Float32(0.0)
        if active_row:
            lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
        scale_log2 = softmax_scale * Float32(_LOG2_E)
        dq0 = Float32(0.0)
        dq1 = Float32(0.0)
        dq2 = Float32(0.0)
        dq3 = Float32(0.0)
        union_k_length = Int32(0)
        if qgroup_valid:
            union_k_length = Int32(mUnionKLength[qgroup_idx])

        for union_slot in range(mUnionKRowIdx.shape[1]):
            union_slot_i = Int32(union_slot)
            union_key_row = Int32(-1)
            if qgroup_valid and union_slot_i < union_k_length:
                union_key_row = Int32(mUnionKRowIdx[qgroup_idx, union_slot_i])
            participates = Boolean(False)
            score_partial = Float32(0.0)
            dprob_partial = Float32(0.0)
            if active_row and union_slot_i < union_k_length and union_key_row >= Int32(0):
                row_slot = Int32(mUnionToRowSlot[qgroup_idx, row_idx, union_slot_i])
                participates = row_slot >= Int32(0)
                if participates:
                    if dim0 < mQRows.shape[2]:
                        kval0 = Float32(mKRows[union_key_row, head_idx, dim0])
                        vval0 = Float32(mVRows[union_key_row, head_idx, dim0])
                        score_partial += q0 * kval0
                        dprob_partial += do0 * vval0
                    if dim1 < mQRows.shape[2]:
                        kval1 = Float32(mKRows[union_key_row, head_idx, dim1])
                        vval1 = Float32(mVRows[union_key_row, head_idx, dim1])
                        score_partial += q1 * kval1
                        dprob_partial += do1 * vval1
                    if dim2 < mQRows.shape[2]:
                        kval2 = Float32(mKRows[union_key_row, head_idx, dim2])
                        vval2 = Float32(mVRows[union_key_row, head_idx, dim2])
                        score_partial += q2 * kval2
                        dprob_partial += do2 * vval2
                    if dim3 < mQRows.shape[2]:
                        kval3 = Float32(mKRows[union_key_row, head_idx, dim3])
                        vval3 = Float32(mVRows[union_key_row, head_idx, dim3])
                        score_partial += q3 * kval3
                        dprob_partial += do3 * vval3
            score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
            dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
            prob = Float32(0.0)
            ds_scaled = Float32(0.0)
            if lane16 == Int32(0) and participates:
                prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                ds_scaled = prob * (dprob - dpsum) * softmax_scale
            prob = utils.shuffle_sync(prob, 0, width=16)
            ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)

            dk0 = Float32(0.0)
            dk1 = Float32(0.0)
            dk2 = Float32(0.0)
            dk3 = Float32(0.0)
            dv0 = Float32(0.0)
            dv1 = Float32(0.0)
            dv2 = Float32(0.0)
            dv3 = Float32(0.0)
            if participates:
                if dim0 < mQRows.shape[2]:
                    kval0 = Float32(mKRows[union_key_row, head_idx, dim0])
                    dq0 += ds_scaled * kval0
                    dk0 = ds_scaled * q0
                    dv0 = prob * do0
                if dim1 < mQRows.shape[2]:
                    kval1 = Float32(mKRows[union_key_row, head_idx, dim1])
                    dq1 += ds_scaled * kval1
                    dk1 = ds_scaled * q1
                    dv1 = prob * do1
                if dim2 < mQRows.shape[2]:
                    kval2 = Float32(mKRows[union_key_row, head_idx, dim2])
                    dq2 += ds_scaled * kval2
                    dk2 = ds_scaled * q2
                    dv2 = prob * do2
                if dim3 < mQRows.shape[2]:
                    kval3 = Float32(mKRows[union_key_row, head_idx, dim3])
                    dq3 += ds_scaled * kval3
                    dk3 = ds_scaled * q3
                    dv3 = prob * do3
            partner_lane = lane16 + (Int32(1) - row_idx) * Int32(16)
            partner_dk0 = utils.shuffle_sync(dk0, partner_lane, width=32)
            partner_dk1 = utils.shuffle_sync(dk1, partner_lane, width=32)
            partner_dk2 = utils.shuffle_sync(dk2, partner_lane, width=32)
            partner_dk3 = utils.shuffle_sync(dk3, partner_lane, width=32)
            partner_dv0 = utils.shuffle_sync(dv0, partner_lane, width=32)
            partner_dv1 = utils.shuffle_sync(dv1, partner_lane, width=32)
            partner_dv2 = utils.shuffle_sync(dv2, partner_lane, width=32)
            partner_dv3 = utils.shuffle_sync(dv3, partner_lane, width=32)
            if lane < Int32(16) and union_slot_i < union_k_length and union_key_row >= Int32(0):
                if dim0 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk0 + partner_dk0,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim0)),
                    )
                    utils.atomic_add_fp32(
                        dv0 + partner_dv0,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim0)),
                    )
                if dim1 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk1 + partner_dk1,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim1)),
                    )
                    utils.atomic_add_fp32(
                        dv1 + partner_dv1,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim1)),
                    )
                if dim2 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk2 + partner_dk2,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim2)),
                    )
                    utils.atomic_add_fp32(
                        dv2 + partner_dv2,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim2)),
                    )
                if dim3 < mdKRows.shape[2]:
                    utils.atomic_add_fp32(
                        dk3 + partner_dk3,
                        utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim3)),
                    )
                    utils.atomic_add_fp32(
                        dv3 + partner_dv3,
                        utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim3)),
                    )

        if active_row:
            if dim0 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim0] = dq0
            if dim1 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim1] = dq1
            if dim2 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim2] = dq2
            if dim3 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim3] = dq3


class FlashHSASyntheticDirectRowMicroBwdDQOnlySm100:
    """Row-owned dQ-only backward for row-compact 2x2 synthetic buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = (mQRowIdx.shape[0] + self.qgroups_per_cta - 1) // self.qgroups_per_cta
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mUnionKRowIdx,
            mUnionToRowSlot,
            mQLength,
            mUnionKLength,
            softmax_scale,
            mdQRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_pair_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        row_idx = lane // Int32(16)
        lane16 = lane % Int32(16)
        dim0 = lane16 * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        qgroup_count = Int32(mQRowIdx.shape[0])
        qgroup_idx = qgroup_pair_idx * Int32(self.qgroups_per_cta) + warp_idx
        qgroup_valid = qgroup_idx < qgroup_count

        q_length = Int32(0)
        if qgroup_valid:
            q_length = Int32(mQLength[qgroup_idx])
        global_q_row = Int32(-1)
        active_row = Boolean(False)
        if qgroup_valid and row_idx < q_length:
            global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
            active_row = global_q_row >= Int32(0)

        q0 = Float32(0.0)
        q1 = Float32(0.0)
        q2 = Float32(0.0)
        q3 = Float32(0.0)
        do0 = Float32(0.0)
        do1 = Float32(0.0)
        do2 = Float32(0.0)
        do3 = Float32(0.0)
        out0 = Float32(0.0)
        out1 = Float32(0.0)
        out2 = Float32(0.0)
        out3 = Float32(0.0)
        if active_row:
            if dim0 < mQRows.shape[2]:
                q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                do0 = Float32(mdORows[global_q_row, head_idx, dim0])
            if dim1 < mQRows.shape[2]:
                q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                do1 = Float32(mdORows[global_q_row, head_idx, dim1])
            if dim2 < mQRows.shape[2]:
                q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                do2 = Float32(mdORows[global_q_row, head_idx, dim2])
            if dim3 < mQRows.shape[2]:
                q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                do3 = Float32(mdORows[global_q_row, head_idx, dim3])

        dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
        dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
        lse_log2 = Float32(0.0)
        if active_row:
            lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
        scale_log2 = softmax_scale * Float32(_LOG2_E)
        dq0 = Float32(0.0)
        dq1 = Float32(0.0)
        dq2 = Float32(0.0)
        dq3 = Float32(0.0)
        union_k_length = Int32(0)
        if qgroup_valid:
            union_k_length = Int32(mUnionKLength[qgroup_idx])

        for union_slot in range(mUnionKRowIdx.shape[1]):
            union_slot_i = Int32(union_slot)
            union_key_row = Int32(-1)
            if qgroup_valid and union_slot_i < union_k_length:
                union_key_row = Int32(mUnionKRowIdx[qgroup_idx, union_slot_i])
            participates = Boolean(False)
            score_partial = Float32(0.0)
            dprob_partial = Float32(0.0)
            if active_row and union_slot_i < union_k_length and union_key_row >= Int32(0):
                row_slot = Int32(mUnionToRowSlot[qgroup_idx, row_idx, union_slot_i])
                participates = row_slot >= Int32(0)
                if participates:
                    if dim0 < mQRows.shape[2]:
                        kval0 = Float32(mKRows[union_key_row, head_idx, dim0])
                        vval0 = Float32(mVRows[union_key_row, head_idx, dim0])
                        score_partial += q0 * kval0
                        dprob_partial += do0 * vval0
                    if dim1 < mQRows.shape[2]:
                        kval1 = Float32(mKRows[union_key_row, head_idx, dim1])
                        vval1 = Float32(mVRows[union_key_row, head_idx, dim1])
                        score_partial += q1 * kval1
                        dprob_partial += do1 * vval1
                    if dim2 < mQRows.shape[2]:
                        kval2 = Float32(mKRows[union_key_row, head_idx, dim2])
                        vval2 = Float32(mVRows[union_key_row, head_idx, dim2])
                        score_partial += q2 * kval2
                        dprob_partial += do2 * vval2
                    if dim3 < mQRows.shape[2]:
                        kval3 = Float32(mKRows[union_key_row, head_idx, dim3])
                        vval3 = Float32(mVRows[union_key_row, head_idx, dim3])
                        score_partial += q3 * kval3
                        dprob_partial += do3 * vval3
            score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
            dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
            ds_scaled = Float32(0.0)
            if lane16 == Int32(0) and participates:
                prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                ds_scaled = prob * (dprob - dpsum) * softmax_scale
            ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)
            if participates:
                if dim0 < mQRows.shape[2]:
                    dq0 += ds_scaled * Float32(mKRows[union_key_row, head_idx, dim0])
                if dim1 < mQRows.shape[2]:
                    dq1 += ds_scaled * Float32(mKRows[union_key_row, head_idx, dim1])
                if dim2 < mQRows.shape[2]:
                    dq2 += ds_scaled * Float32(mKRows[union_key_row, head_idx, dim2])
                if dim3 < mQRows.shape[2]:
                    dq3 += ds_scaled * Float32(mKRows[union_key_row, head_idx, dim3])

        if active_row:
            if dim0 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim0] = dq0
            if dim1 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim1] = dq1
            if dim2 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim2] = dq2
            if dim3 < mdQRows.shape[2]:
                mdQRows[global_q_row, head_idx, dim3] = dq3


class FlashHSASyntheticDirectRowMicroBwdKeyOwnedSm100:
    """Key-owned dK/dV backward for row-compact 2x2 synthetic buckets."""

    arch = 100

    def __init__(self):
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mUniqueKeyRowIdx: cute.Tensor,
        mUniqueKeyMemberIdx: cute.Tensor,
        mUniqueKeyUnionIdx: cute.Tensor,
        mUniqueKeyOccurrenceRowPtr: cute.Tensor,
        softmax_scale: Float32,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = (mUniqueKeyRowIdx.shape[0] + 1) // 2
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mUnionToRowSlot,
            mUniqueKeyRowIdx,
            mUniqueKeyMemberIdx,
            mUniqueKeyUnionIdx,
            mUniqueKeyOccurrenceRowPtr,
            softmax_scale,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mUniqueKeyRowIdx: cute.Tensor,
        mUniqueKeyMemberIdx: cute.Tensor,
        mUniqueKeyUnionIdx: cute.Tensor,
        mUniqueKeyOccurrenceRowPtr: cute.Tensor,
        softmax_scale: Float32,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        unique_key_pair_idx, head_idx, _ = cute.arch.block_idx()
        key_in_cta = tidx // Int32(16)
        lane = tidx % Int32(16)
        dim0 = lane * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        key_count = Int32(mUniqueKeyRowIdx.shape[0])
        unique_key_idx = unique_key_pair_idx * Int32(2) + key_in_cta
        active_key = unique_key_idx < key_count
        key_row = Int32(-1)
        occ_start = Int32(0)
        occ_end = Int32(0)
        if active_key:
            key_row = Int32(mUniqueKeyRowIdx[unique_key_idx])
            occ_start = Int32(mUniqueKeyOccurrenceRowPtr[unique_key_idx])
            occ_end = Int32(mUniqueKeyOccurrenceRowPtr[unique_key_idx + Int32(1)])

        kval0 = Float32(0.0)
        kval1 = Float32(0.0)
        kval2 = Float32(0.0)
        kval3 = Float32(0.0)
        vval0 = Float32(0.0)
        vval1 = Float32(0.0)
        vval2 = Float32(0.0)
        vval3 = Float32(0.0)
        if active_key and key_row >= Int32(0):
            if dim0 < mKRows.shape[2]:
                kval0 = Float32(mKRows[key_row, head_idx, dim0])
                vval0 = Float32(mVRows[key_row, head_idx, dim0])
            if dim1 < mKRows.shape[2]:
                kval1 = Float32(mKRows[key_row, head_idx, dim1])
                vval1 = Float32(mVRows[key_row, head_idx, dim1])
            if dim2 < mKRows.shape[2]:
                kval2 = Float32(mKRows[key_row, head_idx, dim2])
                vval2 = Float32(mVRows[key_row, head_idx, dim2])
            if dim3 < mKRows.shape[2]:
                kval3 = Float32(mKRows[key_row, head_idx, dim3])
                vval3 = Float32(mVRows[key_row, head_idx, dim3])

        dk0 = Float32(0.0)
        dk1 = Float32(0.0)
        dk2 = Float32(0.0)
        dk3 = Float32(0.0)
        dv0 = Float32(0.0)
        dv1 = Float32(0.0)
        dv2 = Float32(0.0)
        dv3 = Float32(0.0)
        scale_log2 = softmax_scale * Float32(_LOG2_E)

        for rel_occ in range(8):
            occ_idx = occ_start + Int32(rel_occ)
            occurrence_valid = active_key and occ_idx < occ_end
            member_idx = Int32(-1)
            union_idx = Int32(-1)
            if occurrence_valid:
                member_idx = Int32(mUniqueKeyMemberIdx[occ_idx])
                union_idx = Int32(mUniqueKeyUnionIdx[occ_idx])
            for row_idx in range(2):
                row_i = Int32(row_idx)
                global_q_row = Int32(-1)
                participates = Boolean(False)
                if occurrence_valid:
                    global_q_row = Int32(mQRowIdx[member_idx, row_i])
                    if global_q_row >= Int32(0):
                        row_slot = Int32(mUnionToRowSlot[member_idx, row_i, union_idx])
                        participates = row_slot >= Int32(0)
                q0 = Float32(0.0)
                q1 = Float32(0.0)
                q2 = Float32(0.0)
                q3 = Float32(0.0)
                do0 = Float32(0.0)
                do1 = Float32(0.0)
                do2 = Float32(0.0)
                do3 = Float32(0.0)
                out0 = Float32(0.0)
                out1 = Float32(0.0)
                out2 = Float32(0.0)
                out3 = Float32(0.0)
                lse_log2 = Float32(0.0)
                if participates:
                    if dim0 < mQRows.shape[2]:
                        q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                        out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                        do0 = Float32(mdORows[global_q_row, head_idx, dim0])
                    if dim1 < mQRows.shape[2]:
                        q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                        out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                        do1 = Float32(mdORows[global_q_row, head_idx, dim1])
                    if dim2 < mQRows.shape[2]:
                        q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                        out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                        do2 = Float32(mdORows[global_q_row, head_idx, dim2])
                    if dim3 < mQRows.shape[2]:
                        q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                        out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                        do3 = Float32(mdORows[global_q_row, head_idx, dim3])
                    lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)

                dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
                dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
                score_partial = Float32(0.0)
                dprob_partial = Float32(0.0)
                if participates:
                    score_partial = q0 * kval0 + q1 * kval1 + q2 * kval2 + q3 * kval3
                    dprob_partial = do0 * vval0 + do1 * vval1 + do2 * vval2 + do3 * vval3
                score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
                dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
                prob = Float32(0.0)
                ds_scaled = Float32(0.0)
                if lane == Int32(0) and participates:
                    prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                    ds_scaled = prob * (dprob - dpsum) * softmax_scale
                prob = utils.shuffle_sync(prob, 0, width=16)
                ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)
                if participates:
                    dk0 += ds_scaled * q0
                    dk1 += ds_scaled * q1
                    dk2 += ds_scaled * q2
                    dk3 += ds_scaled * q3
                    dv0 += prob * do0
                    dv1 += prob * do1
                    dv2 += prob * do2
                    dv3 += prob * do3

        if active_key and key_row >= Int32(0):
            if dim0 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk0, utils.elem_pointer(mdKRows, (key_row, head_idx, dim0)))
                utils.atomic_add_fp32(dv0, utils.elem_pointer(mdVRows, (key_row, head_idx, dim0)))
            if dim1 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk1, utils.elem_pointer(mdKRows, (key_row, head_idx, dim1)))
                utils.atomic_add_fp32(dv1, utils.elem_pointer(mdVRows, (key_row, head_idx, dim1)))
            if dim2 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk2, utils.elem_pointer(mdKRows, (key_row, head_idx, dim2)))
                utils.atomic_add_fp32(dv2, utils.elem_pointer(mdVRows, (key_row, head_idx, dim2)))
            if dim3 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk3, utils.elem_pointer(mdKRows, (key_row, head_idx, dim3)))
                utils.atomic_add_fp32(dv3, utils.elem_pointer(mdVRows, (key_row, head_idx, dim3)))


class FlashHSASyntheticDirectRowMicroBwdOneKernelSm100:
    """Key-owned fused backward that computes dQ, dK, and dV in one pass."""

    arch = 100

    def __init__(self):
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mUniqueKeyRowIdx: cute.Tensor,
        mUniqueKeyMemberIdx: cute.Tensor,
        mUniqueKeyUnionIdx: cute.Tensor,
        mUniqueKeyOccurrenceRowPtr: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = (mUniqueKeyRowIdx.shape[0] + 1) // 2
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mUnionToRowSlot,
            mUniqueKeyRowIdx,
            mUniqueKeyMemberIdx,
            mUniqueKeyUnionIdx,
            mUniqueKeyOccurrenceRowPtr,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mUniqueKeyRowIdx: cute.Tensor,
        mUniqueKeyMemberIdx: cute.Tensor,
        mUniqueKeyUnionIdx: cute.Tensor,
        mUniqueKeyOccurrenceRowPtr: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        unique_key_pair_idx, head_idx, _ = cute.arch.block_idx()
        key_in_cta = tidx // Int32(16)
        lane = tidx % Int32(16)
        dim0 = lane * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        key_count = Int32(mUniqueKeyRowIdx.shape[0])
        unique_key_idx = unique_key_pair_idx * Int32(2) + key_in_cta
        active_key = unique_key_idx < key_count
        key_row = Int32(-1)
        occ_start = Int32(0)
        occ_end = Int32(0)
        if active_key:
            key_row = Int32(mUniqueKeyRowIdx[unique_key_idx])
            occ_start = Int32(mUniqueKeyOccurrenceRowPtr[unique_key_idx])
            occ_end = Int32(mUniqueKeyOccurrenceRowPtr[unique_key_idx + Int32(1)])

        kval0 = Float32(0.0)
        kval1 = Float32(0.0)
        kval2 = Float32(0.0)
        kval3 = Float32(0.0)
        vval0 = Float32(0.0)
        vval1 = Float32(0.0)
        vval2 = Float32(0.0)
        vval3 = Float32(0.0)
        if active_key and key_row >= Int32(0):
            if dim0 < mKRows.shape[2]:
                kval0 = Float32(mKRows[key_row, head_idx, dim0])
                vval0 = Float32(mVRows[key_row, head_idx, dim0])
            if dim1 < mKRows.shape[2]:
                kval1 = Float32(mKRows[key_row, head_idx, dim1])
                vval1 = Float32(mVRows[key_row, head_idx, dim1])
            if dim2 < mKRows.shape[2]:
                kval2 = Float32(mKRows[key_row, head_idx, dim2])
                vval2 = Float32(mVRows[key_row, head_idx, dim2])
            if dim3 < mKRows.shape[2]:
                kval3 = Float32(mKRows[key_row, head_idx, dim3])
                vval3 = Float32(mVRows[key_row, head_idx, dim3])

        dk0 = Float32(0.0)
        dk1 = Float32(0.0)
        dk2 = Float32(0.0)
        dk3 = Float32(0.0)
        dv0 = Float32(0.0)
        dv1 = Float32(0.0)
        dv2 = Float32(0.0)
        dv3 = Float32(0.0)
        scale_log2 = softmax_scale * Float32(_LOG2_E)

        for rel_occ in range(8):
            occ_idx = occ_start + Int32(rel_occ)
            occurrence_valid = active_key and occ_idx < occ_end
            member_idx = Int32(-1)
            union_idx = Int32(-1)
            if occurrence_valid:
                member_idx = Int32(mUniqueKeyMemberIdx[occ_idx])
                union_idx = Int32(mUniqueKeyUnionIdx[occ_idx])
            for row_idx in range(2):
                row_i = Int32(row_idx)
                global_q_row = Int32(-1)
                participates = Boolean(False)
                if occurrence_valid:
                    global_q_row = Int32(mQRowIdx[member_idx, row_i])
                    if global_q_row >= Int32(0):
                        row_slot = Int32(mUnionToRowSlot[member_idx, row_i, union_idx])
                        participates = row_slot >= Int32(0)
                q0 = Float32(0.0)
                q1 = Float32(0.0)
                q2 = Float32(0.0)
                q3 = Float32(0.0)
                do0 = Float32(0.0)
                do1 = Float32(0.0)
                do2 = Float32(0.0)
                do3 = Float32(0.0)
                out0 = Float32(0.0)
                out1 = Float32(0.0)
                out2 = Float32(0.0)
                out3 = Float32(0.0)
                lse_log2 = Float32(0.0)
                if participates:
                    if dim0 < mQRows.shape[2]:
                        q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                        out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                        do0 = Float32(mdORows[global_q_row, head_idx, dim0])
                    if dim1 < mQRows.shape[2]:
                        q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                        out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                        do1 = Float32(mdORows[global_q_row, head_idx, dim1])
                    if dim2 < mQRows.shape[2]:
                        q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                        out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                        do2 = Float32(mdORows[global_q_row, head_idx, dim2])
                    if dim3 < mQRows.shape[2]:
                        q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                        out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                        do3 = Float32(mdORows[global_q_row, head_idx, dim3])
                    lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)

                dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
                dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
                score_partial = Float32(0.0)
                dprob_partial = Float32(0.0)
                if participates:
                    score_partial = q0 * kval0 + q1 * kval1 + q2 * kval2 + q3 * kval3
                    dprob_partial = do0 * vval0 + do1 * vval1 + do2 * vval2 + do3 * vval3
                score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
                dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
                prob = Float32(0.0)
                ds_scaled = Float32(0.0)
                if lane == Int32(0) and participates:
                    prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                    ds_scaled = prob * (dprob - dpsum) * softmax_scale
                prob = utils.shuffle_sync(prob, 0, width=16)
                ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)
                if participates:
                    if dim0 < mdQRows.shape[2]:
                        utils.atomic_add_fp32(ds_scaled * kval0, utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim0)))
                        dk0 += ds_scaled * q0
                        dv0 += prob * do0
                    if dim1 < mdQRows.shape[2]:
                        utils.atomic_add_fp32(ds_scaled * kval1, utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim1)))
                        dk1 += ds_scaled * q1
                        dv1 += prob * do1
                    if dim2 < mdQRows.shape[2]:
                        utils.atomic_add_fp32(ds_scaled * kval2, utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim2)))
                        dk2 += ds_scaled * q2
                        dv2 += prob * do2
                    if dim3 < mdQRows.shape[2]:
                        utils.atomic_add_fp32(ds_scaled * kval3, utils.elem_pointer(mdQRows, (global_q_row, head_idx, dim3)))
                        dk3 += ds_scaled * q3
                        dv3 += prob * do3

        if active_key and key_row >= Int32(0):
            if dim0 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk0, utils.elem_pointer(mdKRows, (key_row, head_idx, dim0)))
                utils.atomic_add_fp32(dv0, utils.elem_pointer(mdVRows, (key_row, head_idx, dim0)))
            if dim1 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk1, utils.elem_pointer(mdKRows, (key_row, head_idx, dim1)))
                utils.atomic_add_fp32(dv1, utils.elem_pointer(mdVRows, (key_row, head_idx, dim1)))
            if dim2 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk2, utils.elem_pointer(mdKRows, (key_row, head_idx, dim2)))
                utils.atomic_add_fp32(dv2, utils.elem_pointer(mdVRows, (key_row, head_idx, dim2)))
            if dim3 < mdKRows.shape[2]:
                utils.atomic_add_fp32(dk3, utils.elem_pointer(mdKRows, (key_row, head_idx, dim3)))
                utils.atomic_add_fp32(dv3, utils.elem_pointer(mdVRows, (key_row, head_idx, dim3)))


class FlashHSASyntheticDirectRowMicroBwdSm100:
    """Row-compact direct sparse backward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = (mQRowIdx.shape[0] + self.qgroups_per_cta - 1) // self.qgroups_per_cta
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mRowKRowIdx,
            mUnionKRowIdx,
            mRowKToUnionIdx,
            mUnionToRowSlot,
            mQLength,
            mRowKLength,
            mUnionKLength,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_pair_idx, head_idx, _ = cute.arch.block_idx()
        subwarp_idx = tidx // Int32(16)
        qgroup_in_cta = subwarp_idx // Int32(2)
        row_idx = subwarp_idx % Int32(2)
        lane = tidx % Int32(16)
        qgroup_base = qgroup_pair_idx * Int32(self.qgroups_per_cta)
        qgroup_count = Int32(mQRowIdx.shape[0])
        dim0 = lane * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        smem = cutlass.utils.SmemAllocator()
        sRowDK = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.qgroups_per_cta * 2, 16, 64)), byte_alignment=16
        )
        sRowDV = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.qgroups_per_cta * 2, 16, 64)), byte_alignment=16
        )
        sK = smem.allocate_tensor(
            mKRows.element_type, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        sV = smem.allocate_tensor(
            mVRows.element_type, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        qgroup_idx = qgroup_base + qgroup_in_cta
        qgroup_valid = qgroup_idx < qgroup_count
        row_k_length = Int32(0)
        if qgroup_valid:
            row_k_length = Int32(mRowKLength[qgroup_idx, row_idx])
        for elem_idx in cutlass.range(lane, row_k_length * Int32(64), Int32(16), unroll=1):
            key_slot = elem_idx // Int32(64)
            dim_idx = elem_idx - key_slot * Int32(64)
            sRowDK[subwarp_idx, key_slot, dim_idx] = Float32(0.0)
            sRowDV[subwarp_idx, key_slot, dim_idx] = Float32(0.0)
        for elem_idx in cutlass.range(
            tidx, Int32(self.qgroups_per_cta) * Int32(16) * Int32(64), self.num_threads, unroll=1
        ):
            qgroup_load = elem_idx // (Int32(16) * Int32(64))
            rem = elem_idx - qgroup_load * Int32(16) * Int32(64)
            union_slot = rem // Int32(64)
            dim_idx = rem - union_slot * Int32(64)
            qgroup_load_idx = qgroup_base + qgroup_load
            if qgroup_load_idx < qgroup_count:
                union_k_length = Int32(mUnionKLength[qgroup_load_idx])
                if union_slot < union_k_length:
                    union_key_row = Int32(mUnionKRowIdx[qgroup_load_idx, union_slot])
                    if union_key_row >= Int32(0):
                        sK[qgroup_load, union_slot, dim_idx] = mKRows[union_key_row, head_idx, dim_idx]
                        sV[qgroup_load, union_slot, dim_idx] = mVRows[union_key_row, head_idx, dim_idx]
                    else:
                        sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
                else:
                    sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                    sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
            else:
                sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
        cute.arch.barrier()
        if subwarp_idx < Int32(self.qgroups_per_cta * 2):
            q_length = Int32(0)
            if qgroup_valid:
                q_length = Int32(mQLength[qgroup_idx])
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if qgroup_valid and row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                active_row = global_q_row >= Int32(0)
            q0 = Float32(0.0)
            q1 = Float32(0.0)
            q2 = Float32(0.0)
            q3 = Float32(0.0)
            do0 = Float32(0.0)
            do1 = Float32(0.0)
            do2 = Float32(0.0)
            do3 = Float32(0.0)
            out0 = Float32(0.0)
            out1 = Float32(0.0)
            out2 = Float32(0.0)
            out3 = Float32(0.0)
            if active_row:
                if dim0 < mQRows.shape[2]:
                    q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                    out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                    do0 = Float32(mdORows[global_q_row, head_idx, dim0])
                if dim1 < mQRows.shape[2]:
                    q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                    out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                    do1 = Float32(mdORows[global_q_row, head_idx, dim1])
                if dim2 < mQRows.shape[2]:
                    q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                    out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                    do2 = Float32(mdORows[global_q_row, head_idx, dim2])
                if dim3 < mQRows.shape[2]:
                    q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                    out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                    do3 = Float32(mdORows[global_q_row, head_idx, dim3])
            dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
            dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
            lse_log2 = Float32(0.0)
            if active_row:
                lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
            scale_log2 = softmax_scale * Float32(_LOG2_E)
            dq0 = Float32(0.0)
            dq1 = Float32(0.0)
            dq2 = Float32(0.0)
            dq3 = Float32(0.0)
            for key_slot in range(mRowKRowIdx.shape[2]):
                score_partial = Float32(0.0)
                dprob_partial = Float32(0.0)
                union_slot = Int32(-1)
                if active_row and Int32(key_slot) < row_k_length:
                    union_slot = Int32(mRowKToUnionIdx[qgroup_idx, row_idx, key_slot])
                    if union_slot >= Int32(0):
                        if dim0 < mQRows.shape[2]:
                            kval0 = Float32(sK[qgroup_in_cta, union_slot, dim0])
                            vval0 = Float32(sV[qgroup_in_cta, union_slot, dim0])
                            score_partial += q0 * kval0
                            dprob_partial += do0 * vval0
                        if dim1 < mQRows.shape[2]:
                            kval1 = Float32(sK[qgroup_in_cta, union_slot, dim1])
                            vval1 = Float32(sV[qgroup_in_cta, union_slot, dim1])
                            score_partial += q1 * kval1
                            dprob_partial += do1 * vval1
                        if dim2 < mQRows.shape[2]:
                            kval2 = Float32(sK[qgroup_in_cta, union_slot, dim2])
                            vval2 = Float32(sV[qgroup_in_cta, union_slot, dim2])
                            score_partial += q2 * kval2
                            dprob_partial += do2 * vval2
                        if dim3 < mQRows.shape[2]:
                            kval3 = Float32(sK[qgroup_in_cta, union_slot, dim3])
                            vval3 = Float32(sV[qgroup_in_cta, union_slot, dim3])
                            score_partial += q3 * kval3
                            dprob_partial += do3 * vval3
                score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
                dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
                prob = Float32(0.0)
                ds_scaled = Float32(0.0)
                if lane == Int32(0):
                    if active_row and Int32(key_slot) < row_k_length and union_slot >= Int32(0):
                        prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                        ds_scaled = prob * (dprob - dpsum) * softmax_scale
                prob = utils.shuffle_sync(prob, 0, width=16)
                ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)
                if active_row and Int32(key_slot) < row_k_length and union_slot >= Int32(0):
                    if dim0 < mQRows.shape[2]:
                        kval0 = Float32(sK[qgroup_in_cta, union_slot, dim0])
                        dq0 += ds_scaled * kval0
                        sRowDK[subwarp_idx, key_slot, dim0] = ds_scaled * q0
                        sRowDV[subwarp_idx, key_slot, dim0] = prob * do0
                    if dim1 < mQRows.shape[2]:
                        kval1 = Float32(sK[qgroup_in_cta, union_slot, dim1])
                        dq1 += ds_scaled * kval1
                        sRowDK[subwarp_idx, key_slot, dim1] = ds_scaled * q1
                        sRowDV[subwarp_idx, key_slot, dim1] = prob * do1
                    if dim2 < mQRows.shape[2]:
                        kval2 = Float32(sK[qgroup_in_cta, union_slot, dim2])
                        dq2 += ds_scaled * kval2
                        sRowDK[subwarp_idx, key_slot, dim2] = ds_scaled * q2
                        sRowDV[subwarp_idx, key_slot, dim2] = prob * do2
                    if dim3 < mQRows.shape[2]:
                        kval3 = Float32(sK[qgroup_in_cta, union_slot, dim3])
                        dq3 += ds_scaled * kval3
                        sRowDK[subwarp_idx, key_slot, dim3] = ds_scaled * q3
                        sRowDV[subwarp_idx, key_slot, dim3] = prob * do3
            if active_row:
                if dim0 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim0] = dq0
                if dim1 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim1] = dq1
                if dim2 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim2] = dq2
                if dim3 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim3] = dq3
        cute.arch.barrier()
        if subwarp_idx < Int32(self.qgroups_per_cta * 2) and qgroup_valid:
            union_k_length = Int32(mUnionKLength[qgroup_idx])
            row0_subwarp = qgroup_in_cta * Int32(2)
            row1_subwarp = row0_subwarp + Int32(1)
            for union_slot in range(row_idx, 16, 2):
                union_slot_i = Int32(union_slot)
                if union_slot_i < union_k_length:
                    union_key_row = Int32(mUnionKRowIdx[qgroup_idx, union_slot_i])
                    if union_key_row >= Int32(0):
                        row0_slot = Int32(mUnionToRowSlot[qgroup_idx, 0, union_slot_i])
                        row1_slot = Int32(mUnionToRowSlot[qgroup_idx, 1, union_slot_i])
                        dk0 = Float32(0.0)
                        dk1 = Float32(0.0)
                        dk2 = Float32(0.0)
                        dk3 = Float32(0.0)
                        dv0 = Float32(0.0)
                        dv1 = Float32(0.0)
                        dv2 = Float32(0.0)
                        dv3 = Float32(0.0)
                        if row0_slot >= Int32(0):
                            if dim0 < mdKRows.shape[2]:
                                dk0 += Float32(sRowDK[row0_subwarp, row0_slot, dim0])
                                dv0 += Float32(sRowDV[row0_subwarp, row0_slot, dim0])
                            if dim1 < mdKRows.shape[2]:
                                dk1 += Float32(sRowDK[row0_subwarp, row0_slot, dim1])
                                dv1 += Float32(sRowDV[row0_subwarp, row0_slot, dim1])
                            if dim2 < mdKRows.shape[2]:
                                dk2 += Float32(sRowDK[row0_subwarp, row0_slot, dim2])
                                dv2 += Float32(sRowDV[row0_subwarp, row0_slot, dim2])
                            if dim3 < mdKRows.shape[2]:
                                dk3 += Float32(sRowDK[row0_subwarp, row0_slot, dim3])
                                dv3 += Float32(sRowDV[row0_subwarp, row0_slot, dim3])
                        if row1_slot >= Int32(0):
                            if dim0 < mdKRows.shape[2]:
                                dk0 += Float32(sRowDK[row1_subwarp, row1_slot, dim0])
                                dv0 += Float32(sRowDV[row1_subwarp, row1_slot, dim0])
                            if dim1 < mdKRows.shape[2]:
                                dk1 += Float32(sRowDK[row1_subwarp, row1_slot, dim1])
                                dv1 += Float32(sRowDV[row1_subwarp, row1_slot, dim1])
                            if dim2 < mdKRows.shape[2]:
                                dk2 += Float32(sRowDK[row1_subwarp, row1_slot, dim2])
                                dv2 += Float32(sRowDV[row1_subwarp, row1_slot, dim2])
                            if dim3 < mdKRows.shape[2]:
                                dk3 += Float32(sRowDK[row1_subwarp, row1_slot, dim3])
                                dv3 += Float32(sRowDV[row1_subwarp, row1_slot, dim3])
                        if dim0 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk0, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim0)))
                            utils.atomic_add_fp32(dv0, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim0)))
                        if dim1 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk1, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim1)))
                            utils.atomic_add_fp32(dv1, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim1)))
                        if dim2 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk2, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim2)))
                            utils.atomic_add_fp32(dv2, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim2)))
                        if dim3 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk3, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim3)))
                            utils.atomic_add_fp32(dv3, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim3)))


class FlashHSASyntheticDirectRowMicroBwdUnionLocalSm100:
    """Row-compact direct sparse backward for one-launch synthetic 2xK buckets."""

    arch = 100

    def __init__(self, *, qgroups_per_cta: int = 2):
        self.qgroups_per_cta = qgroups_per_cta
        self.num_threads = 32 * qgroups_per_cta

    @cute.jit
    def __call__(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid_x = (mQRowIdx.shape[0] + self.qgroups_per_cta - 1) // self.qgroups_per_cta
        grid_y = mQRows.shape[1]
        self.kernel(
            mQRows,
            mKRows,
            mVRows,
            mOutRows,
            mdORows,
            mLSERows,
            mQRowIdx,
            mRowKRowIdx,
            mUnionKRowIdx,
            mRowKToUnionIdx,
            mUnionToRowSlot,
            mQLength,
            mRowKLength,
            mUnionKLength,
            softmax_scale,
            mdQRows,
            mdKRows,
            mdVRows,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQRows: cute.Tensor,
        mKRows: cute.Tensor,
        mVRows: cute.Tensor,
        mOutRows: cute.Tensor,
        mdORows: cute.Tensor,
        mLSERows: cute.Tensor,
        mQRowIdx: cute.Tensor,
        mRowKRowIdx: cute.Tensor,
        mUnionKRowIdx: cute.Tensor,
        mRowKToUnionIdx: cute.Tensor,
        mUnionToRowSlot: cute.Tensor,
        mQLength: cute.Tensor,
        mRowKLength: cute.Tensor,
        mUnionKLength: cute.Tensor,
        softmax_scale: Float32,
        mdQRows: cute.Tensor,
        mdKRows: cute.Tensor,
        mdVRows: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qgroup_pair_idx, head_idx, _ = cute.arch.block_idx()
        subwarp_idx = tidx // Int32(16)
        qgroup_in_cta = subwarp_idx // Int32(2)
        row_idx = subwarp_idx % Int32(2)
        lane = tidx % Int32(16)
        qgroup_base = qgroup_pair_idx * Int32(self.qgroups_per_cta)
        qgroup_count = Int32(mQRowIdx.shape[0])
        dim0 = lane * Int32(4)
        dim1 = dim0 + Int32(1)
        dim2 = dim0 + Int32(2)
        dim3 = dim0 + Int32(3)
        smem = cutlass.utils.SmemAllocator()
        sUnionDK = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        sUnionDV = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        sK = smem.allocate_tensor(
            mKRows.element_type, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        sV = smem.allocate_tensor(
            mVRows.element_type, cute.make_layout((self.qgroups_per_cta, 16, 64)), byte_alignment=16
        )
        qgroup_idx = qgroup_base + qgroup_in_cta
        qgroup_valid = qgroup_idx < qgroup_count
        for elem_idx in cutlass.range(
            tidx, Int32(self.qgroups_per_cta) * Int32(16) * Int32(64), self.num_threads, unroll=1
        ):
            qgroup_load = elem_idx // (Int32(16) * Int32(64))
            rem = elem_idx - qgroup_load * Int32(16) * Int32(64)
            union_slot = rem // Int32(64)
            dim_idx = rem - union_slot * Int32(64)
            sUnionDK[qgroup_load, union_slot, dim_idx] = Float32(0.0)
            sUnionDV[qgroup_load, union_slot, dim_idx] = Float32(0.0)
            qgroup_load_idx = qgroup_base + qgroup_load
            if qgroup_load_idx < qgroup_count:
                union_k_length = Int32(mUnionKLength[qgroup_load_idx])
                if union_slot < union_k_length:
                    union_key_row = Int32(mUnionKRowIdx[qgroup_load_idx, union_slot])
                    if union_key_row >= Int32(0):
                        sK[qgroup_load, union_slot, dim_idx] = mKRows[union_key_row, head_idx, dim_idx]
                        sV[qgroup_load, union_slot, dim_idx] = mVRows[union_key_row, head_idx, dim_idx]
                    else:
                        sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                        sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
                else:
                    sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                    sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
            else:
                sK[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sK.element_type)
                sV[qgroup_load, union_slot, dim_idx] = Float32(0.0).to(sV.element_type)
        cute.arch.barrier()
        if subwarp_idx < Int32(self.qgroups_per_cta * 2):
            q_length = Int32(0)
            if qgroup_valid:
                q_length = Int32(mQLength[qgroup_idx])
            row_k_length = Int32(0)
            if qgroup_valid:
                row_k_length = Int32(mRowKLength[qgroup_idx, row_idx])
            global_q_row = Int32(-1)
            active_row = Boolean(False)
            if qgroup_valid and row_idx < q_length:
                global_q_row = Int32(mQRowIdx[qgroup_idx, row_idx])
                active_row = global_q_row >= Int32(0)
            q0 = Float32(0.0)
            q1 = Float32(0.0)
            q2 = Float32(0.0)
            q3 = Float32(0.0)
            do0 = Float32(0.0)
            do1 = Float32(0.0)
            do2 = Float32(0.0)
            do3 = Float32(0.0)
            out0 = Float32(0.0)
            out1 = Float32(0.0)
            out2 = Float32(0.0)
            out3 = Float32(0.0)
            if active_row:
                if dim0 < mQRows.shape[2]:
                    q0 = Float32(mQRows[global_q_row, head_idx, dim0])
                    out0 = Float32(mOutRows[global_q_row, head_idx, dim0])
                    do0 = Float32(mdORows[global_q_row, head_idx, dim0])
                if dim1 < mQRows.shape[2]:
                    q1 = Float32(mQRows[global_q_row, head_idx, dim1])
                    out1 = Float32(mOutRows[global_q_row, head_idx, dim1])
                    do1 = Float32(mdORows[global_q_row, head_idx, dim1])
                if dim2 < mQRows.shape[2]:
                    q2 = Float32(mQRows[global_q_row, head_idx, dim2])
                    out2 = Float32(mOutRows[global_q_row, head_idx, dim2])
                    do2 = Float32(mdORows[global_q_row, head_idx, dim2])
                if dim3 < mQRows.shape[2]:
                    q3 = Float32(mQRows[global_q_row, head_idx, dim3])
                    out3 = Float32(mOutRows[global_q_row, head_idx, dim3])
                    do3 = Float32(mdORows[global_q_row, head_idx, dim3])
            dpsum_partial = out0 * do0 + out1 * do1 + out2 * do2 + out3 * do3
            dpsum = utils.warp_reduce(dpsum_partial, lambda a, b: a + b, width=16)
            lse_log2 = Float32(0.0)
            if active_row:
                lse_log2 = Float32(mLSERows[global_q_row, head_idx]) * Float32(_LOG2_E)
            scale_log2 = softmax_scale * Float32(_LOG2_E)
            dq0 = Float32(0.0)
            dq1 = Float32(0.0)
            dq2 = Float32(0.0)
            dq3 = Float32(0.0)
            for key_slot in range(mRowKRowIdx.shape[2]):
                key_row = Int32(-1)
                score_partial = Float32(0.0)
                dprob_partial = Float32(0.0)
                union_slot = Int32(-1)
                if active_row and Int32(key_slot) < row_k_length:
                    union_slot = Int32(mRowKToUnionIdx[qgroup_idx, row_idx, key_slot])
                    if union_slot >= Int32(0):
                        if dim0 < mQRows.shape[2]:
                            kval0 = Float32(sK[qgroup_in_cta, union_slot, dim0])
                            vval0 = Float32(sV[qgroup_in_cta, union_slot, dim0])
                            score_partial += q0 * kval0
                            dprob_partial += do0 * vval0
                        if dim1 < mQRows.shape[2]:
                            kval1 = Float32(sK[qgroup_in_cta, union_slot, dim1])
                            vval1 = Float32(sV[qgroup_in_cta, union_slot, dim1])
                            score_partial += q1 * kval1
                            dprob_partial += do1 * vval1
                        if dim2 < mQRows.shape[2]:
                            kval2 = Float32(sK[qgroup_in_cta, union_slot, dim2])
                            vval2 = Float32(sV[qgroup_in_cta, union_slot, dim2])
                            score_partial += q2 * kval2
                            dprob_partial += do2 * vval2
                        if dim3 < mQRows.shape[2]:
                            kval3 = Float32(sK[qgroup_in_cta, union_slot, dim3])
                            vval3 = Float32(sV[qgroup_in_cta, union_slot, dim3])
                            score_partial += q3 * kval3
                            dprob_partial += do3 * vval3
                score = utils.warp_reduce(score_partial, lambda a, b: a + b, width=16)
                dprob = utils.warp_reduce(dprob_partial, lambda a, b: a + b, width=16)
                prob = Float32(0.0)
                ds_scaled = Float32(0.0)
                if lane == Int32(0):
                    if active_row and Int32(key_slot) < row_k_length and union_slot >= Int32(0):
                        prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                        ds_scaled = prob * (dprob - dpsum) * softmax_scale
                prob = utils.shuffle_sync(prob, 0, width=16)
                ds_scaled = utils.shuffle_sync(ds_scaled, 0, width=16)
                if active_row and Int32(key_slot) < row_k_length and union_slot >= Int32(0):
                    if dim0 < mQRows.shape[2]:
                        kval0 = Float32(sK[qgroup_in_cta, union_slot, dim0])
                        dq0 += ds_scaled * kval0
                        utils.atomic_add_fp32(
                            ds_scaled * q0,
                            utils.elem_pointer(sUnionDK, (qgroup_in_cta, union_slot, dim0)),
                        )
                        utils.atomic_add_fp32(
                            prob * do0,
                            utils.elem_pointer(sUnionDV, (qgroup_in_cta, union_slot, dim0)),
                        )
                    if dim1 < mQRows.shape[2]:
                        kval1 = Float32(sK[qgroup_in_cta, union_slot, dim1])
                        dq1 += ds_scaled * kval1
                        utils.atomic_add_fp32(
                            ds_scaled * q1,
                            utils.elem_pointer(sUnionDK, (qgroup_in_cta, union_slot, dim1)),
                        )
                        utils.atomic_add_fp32(
                            prob * do1,
                            utils.elem_pointer(sUnionDV, (qgroup_in_cta, union_slot, dim1)),
                        )
                    if dim2 < mQRows.shape[2]:
                        kval2 = Float32(sK[qgroup_in_cta, union_slot, dim2])
                        dq2 += ds_scaled * kval2
                        utils.atomic_add_fp32(
                            ds_scaled * q2,
                            utils.elem_pointer(sUnionDK, (qgroup_in_cta, union_slot, dim2)),
                        )
                        utils.atomic_add_fp32(
                            prob * do2,
                            utils.elem_pointer(sUnionDV, (qgroup_in_cta, union_slot, dim2)),
                        )
                    if dim3 < mQRows.shape[2]:
                        kval3 = Float32(sK[qgroup_in_cta, union_slot, dim3])
                        dq3 += ds_scaled * kval3
                        utils.atomic_add_fp32(
                            ds_scaled * q3,
                            utils.elem_pointer(sUnionDK, (qgroup_in_cta, union_slot, dim3)),
                        )
                        utils.atomic_add_fp32(
                            prob * do3,
                            utils.elem_pointer(sUnionDV, (qgroup_in_cta, union_slot, dim3)),
                        )
            if active_row:
                if dim0 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim0] = dq0
                if dim1 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim1] = dq1
                if dim2 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim2] = dq2
                if dim3 < mdQRows.shape[2]:
                    mdQRows[global_q_row, head_idx, dim3] = dq3
        cute.arch.barrier()
        if subwarp_idx < Int32(self.qgroups_per_cta * 2) and qgroup_valid:
            union_k_length_flush = Int32(mUnionKLength[qgroup_idx])
            for union_slot in range(row_idx, 16, 2):
                union_slot_i = Int32(union_slot)
                if union_slot_i < union_k_length_flush:
                    union_key_row = Int32(mUnionKRowIdx[qgroup_idx, union_slot_i])
                    if union_key_row >= Int32(0):
                        dk0 = Float32(0.0)
                        dk1 = Float32(0.0)
                        dk2 = Float32(0.0)
                        dk3 = Float32(0.0)
                        dv0 = Float32(0.0)
                        dv1 = Float32(0.0)
                        dv2 = Float32(0.0)
                        dv3 = Float32(0.0)
                        if dim0 < mdKRows.shape[2]:
                            dk0 = Float32(sUnionDK[qgroup_in_cta, union_slot_i, dim0])
                            dv0 = Float32(sUnionDV[qgroup_in_cta, union_slot_i, dim0])
                        if dim1 < mdKRows.shape[2]:
                            dk1 = Float32(sUnionDK[qgroup_in_cta, union_slot_i, dim1])
                            dv1 = Float32(sUnionDV[qgroup_in_cta, union_slot_i, dim1])
                        if dim2 < mdKRows.shape[2]:
                            dk2 = Float32(sUnionDK[qgroup_in_cta, union_slot_i, dim2])
                            dv2 = Float32(sUnionDV[qgroup_in_cta, union_slot_i, dim2])
                        if dim3 < mdKRows.shape[2]:
                            dk3 = Float32(sUnionDK[qgroup_in_cta, union_slot_i, dim3])
                            dv3 = Float32(sUnionDV[qgroup_in_cta, union_slot_i, dim3])
                        if dim0 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk0, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim0)))
                            utils.atomic_add_fp32(dv0, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim0)))
                        if dim1 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk1, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim1)))
                            utils.atomic_add_fp32(dv1, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim1)))
                        if dim2 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk2, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim2)))
                            utils.atomic_add_fp32(dv2, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim2)))
                        if dim3 < mdKRows.shape[2]:
                            utils.atomic_add_fp32(dk3, utils.elem_pointer(mdKRows, (union_key_row, head_idx, dim3)))
                            utils.atomic_add_fp32(dv3, utils.elem_pointer(mdVRows, (union_key_row, head_idx, dim3)))


def _run_synthetic_direct_row_micro_fwd_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    row_k_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    row_k_to_union_idx: torch.Tensor,
    q_length: torch.Tensor,
    row_k_length: torch.Tensor,
    union_k_length: torch.Tensor,
    out_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    qgroups_per_cta = _get_synthetic_qgroups_per_cta()
    compile_key = (
        "synthetic_direct_row_micro_fwd_v6",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        q_row_idx.shape[1],
        row_k_row_idx.shape[2],
        union_k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        qgroups_per_cta,
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_fwd_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroFwdSm100(qgroups_per_cta=qgroups_per_cta)
        _run_synthetic_direct_row_micro_fwd_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(row_k_row_idx, assumed_align=4),
            to_cute_tensor(union_k_row_idx, assumed_align=4),
            to_cute_tensor(row_k_to_union_idx, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(row_k_length, assumed_align=4),
            to_cute_tensor(union_k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(out_rows, assumed_align=4),
            to_cute_tensor(lse_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_fwd_kernel.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        q_length,
        row_k_length,
        union_k_length,
        Float32(softmax_scale),
        out_rows,
        lse_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_fwd_kernel.compile_cache = get_jit_cache("hsa_synth_direct_row_micro_fwd")


def _run_synthetic_direct_row_micro_bwd_kernel_short(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    row_k_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    row_k_to_union_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    q_length: torch.Tensor,
    row_k_length: torch.Tensor,
    union_k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    qgroups_per_cta = 2
    compile_key = (
        "synthetic_direct_row_micro_bwd_short_v1",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        union_k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        qgroups_per_cta,
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_short.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdShortSm100(qgroups_per_cta=qgroups_per_cta)
        _run_synthetic_direct_row_micro_bwd_kernel_short.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(row_k_row_idx, assumed_align=4),
            to_cute_tensor(union_k_row_idx, assumed_align=4),
            to_cute_tensor(row_k_to_union_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(row_k_length, assumed_align=4),
            to_cute_tensor(union_k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_short.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_short.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_short"
)


def _run_synthetic_direct_row_micro_bwd_kernel_fused(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    row_k_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    row_k_to_union_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    q_length: torch.Tensor,
    row_k_length: torch.Tensor,
    union_k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    qgroups_per_cta = 1
    compile_key = (
        "synthetic_direct_row_micro_bwd_fused_v1",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        union_k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        qgroups_per_cta,
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_fused.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdFusedSm100()
        _run_synthetic_direct_row_micro_bwd_kernel_fused.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(row_k_row_idx, assumed_align=4),
            to_cute_tensor(union_k_row_idx, assumed_align=4),
            to_cute_tensor(row_k_to_union_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(row_k_length, assumed_align=4),
            to_cute_tensor(union_k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_fused.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_fused.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_fused"
)


def _run_synthetic_direct_row_micro_bwd_kernel_dq_only(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    q_length: torch.Tensor,
    union_k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    qgroups_per_cta = 2
    compile_key = (
        "synthetic_direct_row_micro_bwd_dq_only_v2",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        union_k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        qgroups_per_cta,
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_dq_only.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdDQOnlySm100(qgroups_per_cta=qgroups_per_cta)
        _run_synthetic_direct_row_micro_bwd_kernel_dq_only.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(union_k_row_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(union_k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_dq_only.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_k_row_idx,
        union_to_row_slot,
        q_length,
        union_k_length,
        Float32(softmax_scale),
        dq_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_dq_only.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_dq_only"
)


def _run_synthetic_direct_row_micro_bwd_kernel_key_owned(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    unique_key_row_idx: torch.Tensor,
    unique_key_member_idx: torch.Tensor,
    unique_key_union_idx: torch.Tensor,
    unique_key_occurrence_row_ptr: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_row_micro_bwd_key_owned_v2",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        union_to_row_slot.shape[2],
        unique_key_row_idx.shape[0],
        unique_key_member_idx.shape[0],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_key_owned.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdKeyOwnedSm100()
        _run_synthetic_direct_row_micro_bwd_kernel_key_owned.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(unique_key_row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(unique_key_member_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(unique_key_union_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(unique_key_occurrence_row_ptr, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_key_owned.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        Float32(softmax_scale),
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_key_owned.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_key_owned"
)


def _run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    unique_key_row_idx: torch.Tensor,
    unique_key_member_idx: torch.Tensor,
    unique_key_union_idx: torch.Tensor,
    unique_key_occurrence_row_ptr: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    compile_key = (
        "synthetic_direct_row_micro_bwd_one_kernel_v1",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        union_to_row_slot.shape[2],
        unique_key_row_idx.shape[0],
        unique_key_member_idx.shape[0],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_one_kernel.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdOneKernelSm100()
        _run_synthetic_direct_row_micro_bwd_kernel_one_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(unique_key_row_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(unique_key_member_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(unique_key_union_idx, assumed_align=4, leading_dim=0),
            to_cute_tensor(unique_key_occurrence_row_ptr, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_one_kernel.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        union_to_row_slot,
        unique_key_row_idx,
        unique_key_member_idx,
        unique_key_union_idx,
        unique_key_occurrence_row_ptr,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_one_kernel.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_one_kernel"
)


def _run_synthetic_direct_row_micro_bwd_kernel_row_local(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    row_k_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    row_k_to_union_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    q_length: torch.Tensor,
    row_k_length: torch.Tensor,
    union_k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    qgroups_per_cta = _get_synthetic_qgroups_per_cta_bwd()
    compile_key = (
        "synthetic_direct_row_micro_bwd_row_local_v1",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        row_k_row_idx.shape[2],
        union_k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        qgroups_per_cta,
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_row_local.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdSm100(qgroups_per_cta=qgroups_per_cta)
        _run_synthetic_direct_row_micro_bwd_kernel_row_local.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(row_k_row_idx, assumed_align=4),
            to_cute_tensor(union_k_row_idx, assumed_align=4),
            to_cute_tensor(row_k_to_union_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(row_k_length, assumed_align=4),
            to_cute_tensor(union_k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_row_local.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_row_local.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_row_local"
)


def _run_synthetic_direct_row_micro_bwd_kernel_union_local(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    row_k_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    row_k_to_union_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    q_length: torch.Tensor,
    row_k_length: torch.Tensor,
    union_k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    _require_cute_runtime()
    qgroups_per_cta = _get_synthetic_qgroups_per_cta_bwd()
    compile_key = (
        "synthetic_direct_row_micro_bwd_union_local_v1",
        q_rows.dtype,
        k_rows.dtype,
        v_rows.dtype,
        out_rows.dtype,
        dout_rows.dtype,
        q_row_idx.shape[1],
        row_k_row_idx.shape[2],
        union_k_row_idx.shape[1],
        q_rows.shape[1],
        q_rows.shape[2],
        v_rows.shape[2],
        qgroups_per_cta,
        torch.cuda.get_device_capability(q_rows.device),
    )
    if compile_key not in _run_synthetic_direct_row_micro_bwd_kernel_union_local.compile_cache:
        kernel = FlashHSASyntheticDirectRowMicroBwdUnionLocalSm100(qgroups_per_cta=qgroups_per_cta)
        _run_synthetic_direct_row_micro_bwd_kernel_union_local.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_rows),
            to_cute_tensor(k_rows),
            to_cute_tensor(v_rows),
            to_cute_tensor(out_rows),
            to_cute_tensor(dout_rows),
            to_cute_tensor(lse_rows, assumed_align=4),
            to_cute_tensor(q_row_idx, assumed_align=4),
            to_cute_tensor(row_k_row_idx, assumed_align=4),
            to_cute_tensor(union_k_row_idx, assumed_align=4),
            to_cute_tensor(row_k_to_union_idx, assumed_align=4),
            to_cute_tensor(union_to_row_slot, assumed_align=4),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(row_k_length, assumed_align=4),
            to_cute_tensor(union_k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(dq_rows, assumed_align=4),
            to_cute_tensor(dk_rows, assumed_align=4),
            to_cute_tensor(dv_rows, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_direct_row_micro_bwd_kernel_union_local.compile_cache[compile_key](
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        Float32(softmax_scale),
        dq_rows,
        dk_rows,
        dv_rows,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


_run_synthetic_direct_row_micro_bwd_kernel_union_local.compile_cache = get_jit_cache(
    "hsa_synth_direct_row_micro_bwd_union_local"
)


def _should_use_synthetic_short_bwd(
    q_rows: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    q_row_idx: torch.Tensor,
) -> bool:
    mode = _get_synthetic_short_bwd_mode()
    if mode == "off":
        return False
    if mode == "on":
        return True
    try:
        return (
            q_rows.shape[-1] == 64
            and _get_synthetic_qgroups_per_cta_bwd() == 2
            and q_row_idx.ndim >= 2
            and union_k_row_idx.ndim >= 2
            and q_row_idx.shape[0] <= 2048
            and union_k_row_idx.shape[1] <= 12
        )
    except Exception:
        return False


def _can_use_synthetic_fused_bwd(
    q_rows: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    q_row_idx: torch.Tensor,
) -> bool:
    try:
        return (
            q_rows.shape[-1] == 64
            and q_row_idx.ndim >= 2
            and q_row_idx.shape[1] == 2
            and union_k_row_idx.ndim >= 2
            and union_k_row_idx.shape[1] <= 16
        )
    except Exception:
        return False


def _can_use_synthetic_split_bwd(
    q_rows: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    q_row_idx: torch.Tensor,
    unique_key_row_idx: torch.Tensor,
    max_unique_key_occurrences: int,
) -> bool:
    try:
        return (
            q_rows.shape[-1] == 64
            and q_row_idx.ndim >= 2
            and q_row_idx.shape[1] == 2
            and union_k_row_idx.ndim >= 2
            and union_k_row_idx.shape[1] <= 16
            and unique_key_row_idx.ndim >= 1
            and unique_key_row_idx.shape[0] > 0
            and max_unique_key_occurrences <= 8
        )
    except Exception:
        return False


def _should_use_synthetic_one_kernel_bwd(
    q_rows: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    q_row_idx: torch.Tensor,
    unique_key_row_idx: torch.Tensor,
    max_unique_key_occurrences: int,
) -> bool:
    mode = _get_synthetic_one_kernel_bwd_mode()
    if mode == "off":
        return False
    supported = _can_use_synthetic_split_bwd(
        q_rows,
        union_k_row_idx,
        q_row_idx,
        unique_key_row_idx,
        max_unique_key_occurrences,
    )
    if mode == "on":
        return supported
    try:
        return supported and q_row_idx.shape[0] <= 2048 and union_k_row_idx.shape[1] <= 12
    except Exception:
        return False


def _select_row_compact_synthetic_bwd_mode(
    q_rows: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    q_row_idx: torch.Tensor,
    unique_key_row_idx: torch.Tensor | None,
    max_unique_key_occurrences: int,
) -> str:
    unique_key_tensor = unique_key_row_idx if unique_key_row_idx is not None else torch.empty(0)
    if _should_use_synthetic_one_kernel_bwd(
        q_rows,
        union_k_row_idx,
        q_row_idx,
        unique_key_tensor,
        max_unique_key_occurrences,
    ):
        return "one_kernel"
    if _synthetic_split_bwd_enabled() and _can_use_synthetic_split_bwd(
        q_rows,
        union_k_row_idx,
        q_row_idx,
        unique_key_tensor,
        max_unique_key_occurrences,
    ):
        return "split"
    return "legacy"


def _run_synthetic_direct_row_micro_bwd_kernel(
    q_rows: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    out_rows: torch.Tensor,
    dout_rows: torch.Tensor,
    lse_rows: torch.Tensor,
    q_row_idx: torch.Tensor,
    row_k_row_idx: torch.Tensor,
    union_k_row_idx: torch.Tensor,
    row_k_to_union_idx: torch.Tensor,
    union_to_row_slot: torch.Tensor,
    q_length: torch.Tensor,
    row_k_length: torch.Tensor,
    union_k_length: torch.Tensor,
    dq_rows: torch.Tensor,
    dk_rows: torch.Tensor,
    dv_rows: torch.Tensor,
    *,
    softmax_scale: float,
):
    if _synthetic_fused_bwd_enabled() and _can_use_synthetic_fused_bwd(q_rows, union_k_row_idx, q_row_idx):
        return _run_synthetic_direct_row_micro_bwd_kernel_fused(
            q_rows,
            k_rows,
            v_rows,
            out_rows,
            dout_rows,
            lse_rows,
            q_row_idx,
            row_k_row_idx,
            union_k_row_idx,
            row_k_to_union_idx,
            union_to_row_slot,
            q_length,
            row_k_length,
            union_k_length,
            dq_rows,
            dk_rows,
            dv_rows,
            softmax_scale=softmax_scale,
        )
    if _should_use_synthetic_short_bwd(q_rows, union_k_row_idx, q_row_idx):
        return _run_synthetic_direct_row_micro_bwd_kernel_short(
            q_rows,
            k_rows,
            v_rows,
            out_rows,
            dout_rows,
            lse_rows,
            q_row_idx,
            row_k_row_idx,
            union_k_row_idx,
            row_k_to_union_idx,
            union_to_row_slot,
            q_length,
            row_k_length,
            union_k_length,
            dq_rows,
            dk_rows,
            dv_rows,
            softmax_scale=softmax_scale,
        )
    accum_mode = _get_synthetic_row_bwd_accum_mode()
    if accum_mode == "union_local":
        return _run_synthetic_direct_row_micro_bwd_kernel_union_local(
            q_rows,
            k_rows,
            v_rows,
            out_rows,
            dout_rows,
            lse_rows,
            q_row_idx,
            row_k_row_idx,
            union_k_row_idx,
            row_k_to_union_idx,
            union_to_row_slot,
            q_length,
            row_k_length,
            union_k_length,
            dq_rows,
            dk_rows,
            dv_rows,
            softmax_scale=softmax_scale,
        )
    return _run_synthetic_direct_row_micro_bwd_kernel_row_local(
        q_rows,
        k_rows,
        v_rows,
        out_rows,
        dout_rows,
        lse_rows,
        q_row_idx,
        row_k_row_idx,
        union_k_row_idx,
        row_k_to_union_idx,
        union_to_row_slot,
        q_length,
        row_k_length,
        union_k_length,
        dq_rows,
        dk_rows,
        dv_rows,
        softmax_scale=softmax_scale,
    )


def _run_synthetic_micro_fwd_dense_kernel(
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cute_runtime()
    out = torch.empty(
        (q_buf.shape[0], q_buf.shape[1], q_buf.shape[2], v_buf.shape[3]),
        dtype=torch.float32,
        device=q_buf.device,
    )
    lse = torch.empty((q_buf.shape[0], q_buf.shape[1], q_buf.shape[2]), dtype=torch.float32, device=q_buf.device)
    compile_key = (
        "synthetic_micro_fwd_dense_v2",
        q_buf.dtype,
        k_buf.dtype,
        v_buf.dtype,
        q_buf.shape[1],
        k_buf.shape[1],
        q_buf.shape[2],
        q_buf.shape[3],
        v_buf.shape[3],
        torch.cuda.get_device_capability(q_buf.device),
    )
    if compile_key not in _run_synthetic_micro_fwd_dense_kernel.compile_cache:
        kernel = FlashHSASyntheticMicroFwdDenseSm100()
        _run_synthetic_micro_fwd_dense_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_buf),
            to_cute_tensor(k_buf),
            to_cute_tensor(v_buf),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(k_length, assumed_align=4, leading_dim=0),
            Float32(softmax_scale),
            to_cute_tensor(out, assumed_align=4),
            to_cute_tensor(lse, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_micro_fwd_dense_kernel.compile_cache[compile_key](
        q_buf,
        k_buf,
        v_buf,
        q_length,
        k_length,
        Float32(softmax_scale),
        out,
        lse,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out, lse


_run_synthetic_micro_fwd_dense_kernel.compile_cache = get_jit_cache("hsa_synth_micro_fwd_dense")


def _run_synthetic_micro_fwd_masked_kernel(
    q_buf: torch.Tensor,
    k_buf: torch.Tensor,
    v_buf: torch.Tensor,
    q_length: torch.Tensor,
    k_length: torch.Tensor,
    mask_words: torch.Tensor,
    *,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cute_runtime()
    out = torch.empty(
        (q_buf.shape[0], q_buf.shape[1], q_buf.shape[2], v_buf.shape[3]),
        dtype=torch.float32,
        device=q_buf.device,
    )
    lse = torch.empty((q_buf.shape[0], q_buf.shape[1], q_buf.shape[2]), dtype=torch.float32, device=q_buf.device)
    compile_key = (
        "synthetic_micro_fwd_masked_v2",
        q_buf.dtype,
        k_buf.dtype,
        v_buf.dtype,
        q_buf.shape[1],
        k_buf.shape[1],
        q_buf.shape[2],
        q_buf.shape[3],
        v_buf.shape[3],
        mask_words.shape[2],
        torch.cuda.get_device_capability(q_buf.device),
    )
    if compile_key not in _run_synthetic_micro_fwd_masked_kernel.compile_cache:
        kernel = FlashHSASyntheticMicroFwdMaskedSm100()
        _run_synthetic_micro_fwd_masked_kernel.compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q_buf),
            to_cute_tensor(k_buf),
            to_cute_tensor(v_buf),
            to_cute_tensor(q_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(k_length, assumed_align=4, leading_dim=0),
            to_cute_tensor(mask_words, assumed_align=4, leading_dim=2),
            Float32(softmax_scale),
            to_cute_tensor(out, assumed_align=4),
            to_cute_tensor(lse, assumed_align=4),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
            options="--enable-tvm-ffi",
        )
    _run_synthetic_micro_fwd_masked_kernel.compile_cache[compile_key](
        q_buf,
        k_buf,
        v_buf,
        q_length,
        k_length,
        mask_words,
        Float32(softmax_scale),
        out,
        lse,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out, lse


_run_synthetic_micro_fwd_masked_kernel.compile_cache = get_jit_cache("hsa_synth_micro_fwd_masked")


def _slice_bucket_flat_rows(
    metadata,
    bucket_idx: int,
    *,
    use_q_rows: bool,
    use_source_rows: bool = False,
) -> torch.Tensor:
    ptr = metadata.bucket_q_row_idx_row_ptr if use_q_rows else metadata.bucket_k_row_idx_row_ptr
    if use_q_rows:
        flat = metadata.bucket_q_src_row_idx if use_source_rows else metadata.bucket_q_row_idx
    else:
        flat = metadata.bucket_k_row_idx
    if ptr is None or flat is None:
        raise RuntimeError("Synthetic packed metadata is missing bucket row maps")
    start = int(ptr[bucket_idx].item())
    end = int(ptr[bucket_idx + 1].item())
    return flat[start:end].contiguous()


def _slice_bucket_mask_words(metadata, bucket_idx: int) -> torch.Tensor | None:
    if metadata.bucket_mask_word_row_ptr is None or metadata.bucket_mask_words is None:
        return None
    start = int(metadata.bucket_mask_word_row_ptr[bucket_idx].item())
    end = int(metadata.bucket_mask_word_row_ptr[bucket_idx + 1].item())
    if end <= start:
        return None
    return metadata.bucket_mask_words[start:end].contiguous()


def _slice_bucket_lengths(metadata, bucket_idx: int, *, use_q_lengths: bool) -> torch.Tensor:
    flat = metadata.bucket_q_length if use_q_lengths else metadata.bucket_k_length
    if flat is None:
        raise RuntimeError("Synthetic packed metadata is missing bucket length tables")
    start = int(metadata.bucket_row_ptr[bucket_idx].item())
    end = int(metadata.bucket_row_ptr[bucket_idx + 1].item())
    return flat[start:end].contiguous()


def _slice_qgroup_bucket_rows(metadata, qgroup_bucket_idx: int) -> torch.Tensor:
    ptr = metadata.qgroup_bucket_q_row_idx_row_ptr
    flat = metadata.qgroup_bucket_q_row_idx
    if ptr is None or flat is None:
        raise RuntimeError("Synthetic packed metadata is missing q-group bucket rows")
    start = int(ptr[qgroup_bucket_idx].item())
    end = int(ptr[qgroup_bucket_idx + 1].item())
    return flat[start:end].contiguous()


def _slice_qgroup_bucket_ids(metadata, qgroup_bucket_idx: int) -> torch.Tensor:
    ptr = metadata.qgroup_bucket_row_ptr
    flat = metadata.qgroup_bucket_idx
    if ptr is None or flat is None:
        raise RuntimeError("Synthetic packed metadata is missing q-group bucket ids")
    start = int(ptr[qgroup_bucket_idx].item())
    end = int(ptr[qgroup_bucket_idx + 1].item())
    return flat[start:end].contiguous()


def _slice_qgroup_bucket_split_bucket_ids(metadata, qgroup_bucket_idx: int) -> torch.Tensor:
    ptr = metadata.qgroup_bucket_split_bucket_row_ptr
    flat = metadata.qgroup_bucket_split_bucket_idx
    if ptr is None or flat is None:
        raise RuntimeError("Synthetic packed metadata is missing q-group split bucket ids")
    start = int(ptr[qgroup_bucket_idx].item())
    end = int(ptr[qgroup_bucket_idx + 1].item())
    return flat[start:end].contiguous()


def _run_direct_row_compact_forward(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    total_out: torch.Tensor,
    total_lse: torch.Tensor,
    metadata,
    execution_plan,
    *,
    softmax_scale: float,
) -> bool:
    direct_plan = execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return False
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is None:
        return False

    direct_bucket_packed_q = direct_plan["bucket_packed_q"]
    direct_bucket_packed_k = direct_plan["bucket_packed_k"]
    direct_bucket_q_row_range = direct_plan["bucket_q_row_range"]
    direct_bucket_k_row_range = direct_plan["bucket_k_row_range"]
    direct_bucket_q_length_range = direct_plan["bucket_q_length_range"]
    direct_bucket_k_length_range = direct_plan["bucket_k_length_range"]
    direct_bucket_q_row_idx = direct_plan["bucket_q_row_idx"]
    direct_bucket_k_row_idx = direct_plan["bucket_k_row_idx"]
    direct_bucket_q_length = direct_plan["bucket_q_length"]
    direct_bucket_k_length = direct_plan["bucket_k_length"]
    row_bucket_row_k_cap = row_plan["bucket_row_k_cap"]
    row_bucket_row_k_range = row_plan["bucket_row_k_range"]
    row_bucket_row_k_to_union_range = row_plan["bucket_row_k_to_union_range"]
    row_bucket_row_k_length_range = row_plan["bucket_row_k_length_range"]
    row_bucket_row_k_row_idx = row_plan["bucket_row_k_row_idx"]
    row_bucket_row_k_to_union_idx = row_plan["bucket_row_k_to_union_idx"]
    row_bucket_row_k_length = row_plan["bucket_row_k_length"]

    for direct_bucket_idx, row_k_cap in enumerate(row_bucket_row_k_cap):
        if int(direct_bucket_packed_q[direct_bucket_idx]) != 2 or int(row_k_cap) <= 0:
            continue
        packed_k = int(direct_bucket_packed_k[direct_bucket_idx])
        q_row_start, q_row_end = direct_bucket_q_row_range[direct_bucket_idx]
        k_row_start, k_row_end = direct_bucket_k_row_range[direct_bucket_idx]
        q_length_start, q_length_end = direct_bucket_q_length_range[direct_bucket_idx]
        k_length_start, k_length_end = direct_bucket_k_length_range[direct_bucket_idx]
        row_k_start, row_k_end = row_bucket_row_k_range[direct_bucket_idx]
        row_k_to_union_start, row_k_to_union_end = row_bucket_row_k_to_union_range[direct_bucket_idx]
        row_k_length_start, row_k_length_end = row_bucket_row_k_length_range[direct_bucket_idx]
        bucket_size = q_length_end - q_length_start
        q_row_idx = direct_bucket_q_row_idx[q_row_start:q_row_end].contiguous().view(bucket_size, 2)
        union_k_row_idx = direct_bucket_k_row_idx[k_row_start:k_row_end].contiguous().view(bucket_size, packed_k)
        q_length = direct_bucket_q_length[q_length_start:q_length_end].contiguous()
        union_k_length = direct_bucket_k_length[k_length_start:k_length_end].contiguous()
        row_k_row_idx = row_bucket_row_k_row_idx[row_k_start:row_k_end].contiguous().view(bucket_size, 2, int(row_k_cap))
        row_k_to_union_idx = row_bucket_row_k_to_union_idx[row_k_to_union_start:row_k_to_union_end].contiguous().view(
            bucket_size, 2, int(row_k_cap)
        )
        row_k_length = row_bucket_row_k_length[row_k_length_start:row_k_length_end].contiguous().view(bucket_size, 2)
        _run_synthetic_direct_row_micro_fwd_kernel(
            q_flat,
            k_flat,
            v_flat,
            q_row_idx,
            row_k_row_idx,
            union_k_row_idx,
            row_k_to_union_idx,
            q_length,
            row_k_length,
            union_k_length,
            total_out,
            total_lse,
            softmax_scale=softmax_scale,
        )
    return True


def _run_direct_segmented_forward(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    total_out: torch.Tensor,
    total_lse: torch.Tensor,
    metadata,
    execution_plan,
    *,
    softmax_scale: float,
    runtime,
) -> bool:
    direct_plan = execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return False

    workspace = runtime.synthetic_forward_workspace
    if workspace is None:
        workspace = {}
        runtime.synthetic_forward_workspace = workspace
    direct_workspace = workspace.setdefault("direct_micro", {})

    direct_qgroup_bucket_segment_range = direct_plan["qgroup_bucket_segment_range"]
    direct_qgroup_bucket_segment_idx = direct_plan["qgroup_bucket_segment_idx"]
    direct_bucket_size = direct_plan["bucket_size"]
    direct_bucket_packed_q = direct_plan["bucket_packed_q"]
    direct_bucket_packed_k = direct_plan["bucket_packed_k"]
    direct_bucket_dense = direct_plan["bucket_dense"]
    direct_bucket_words_per_row = direct_plan["bucket_words_per_row"]
    direct_bucket_q_row_range = direct_plan["bucket_q_row_range"]
    direct_bucket_k_row_range = direct_plan["bucket_k_row_range"]
    direct_bucket_q_length_range = direct_plan["bucket_q_length_range"]
    direct_bucket_k_length_range = direct_plan["bucket_k_length_range"]
    direct_bucket_mask_word_range = direct_plan["bucket_mask_word_range"]
    direct_bucket_q_row_idx = direct_plan["bucket_q_row_idx"]
    direct_bucket_k_row_idx = direct_plan["bucket_k_row_idx"]
    direct_bucket_q_length = direct_plan["bucket_q_length"]
    direct_bucket_k_length = direct_plan["bucket_k_length"]
    direct_bucket_mask_words = direct_plan["bucket_mask_words"]

    def _get_temp_buffers():
        key = (
            str(q_flat.device),
            total_out.dtype,
            total_lse.dtype,
            total_out.shape[0],
            total_out.shape[1],
            total_out.shape[2],
        )
        bufs = direct_workspace.get(key)
        if bufs is None:
            bufs = (torch.empty_like(total_out), torch.empty_like(total_lse))
            direct_workspace[key] = bufs
        return bufs

    for qgroup_bucket_idx in range(len(direct_qgroup_bucket_segment_range)):
        segment_start, segment_end = direct_qgroup_bucket_segment_range[qgroup_bucket_idx]
        direct_bucket_ids = direct_qgroup_bucket_segment_idx[segment_start:segment_end]
        if not direct_bucket_ids:
            continue
        if len(direct_bucket_ids) == 1:
            direct_bucket_idx = int(direct_bucket_ids[0])
            bucket_size_value = int(direct_bucket_size[direct_bucket_idx])
            packed_q = int(direct_bucket_packed_q[direct_bucket_idx])
            q_row_start, q_row_end = direct_bucket_q_row_range[direct_bucket_idx]
            packed_k = int(direct_bucket_packed_k[direct_bucket_idx])
            k_row_start, k_row_end = direct_bucket_k_row_range[direct_bucket_idx]
            q_length_start, q_length_end = direct_bucket_q_length_range[direct_bucket_idx]
            k_length_start, k_length_end = direct_bucket_k_length_range[direct_bucket_idx]
            q_row_idx = direct_bucket_q_row_idx[q_row_start:q_row_end].contiguous().view(bucket_size_value, packed_q)
            k_row_idx = direct_bucket_k_row_idx[k_row_start:k_row_end].contiguous().view(bucket_size_value, packed_k)
            q_length = direct_bucket_q_length[q_length_start:q_length_end].contiguous()
            k_length = direct_bucket_k_length[k_length_start:k_length_end].contiguous()
            if bool(direct_bucket_dense[direct_bucket_idx]):
                _run_synthetic_direct_micro_fwd_dense_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    q_row_idx,
                    k_row_idx,
                    q_length,
                    k_length,
                    total_out,
                    total_lse,
                    softmax_scale=softmax_scale,
                )
            else:
                words_per_row = int(direct_bucket_words_per_row[direct_bucket_idx])
                mask_word_start, mask_word_end = direct_bucket_mask_word_range[direct_bucket_idx]
                mask_words = direct_bucket_mask_words[mask_word_start:mask_word_end].contiguous().view(
                    bucket_size_value, packed_q, words_per_row
                )
                _run_synthetic_direct_micro_fwd_masked_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    q_row_idx,
                    k_row_idx,
                    q_length,
                    k_length,
                    mask_words,
                    total_out,
                    total_lse,
                    softmax_scale=softmax_scale,
                )
            continue

        temp_out, temp_lse = _get_temp_buffers()
        for direct_bucket_idx in direct_bucket_ids:
            direct_bucket_idx = int(direct_bucket_idx)
            bucket_size_value = int(direct_bucket_size[direct_bucket_idx])
            packed_q = int(direct_bucket_packed_q[direct_bucket_idx])
            q_row_start, q_row_end = direct_bucket_q_row_range[direct_bucket_idx]
            packed_k = int(direct_bucket_packed_k[direct_bucket_idx])
            k_row_start, k_row_end = direct_bucket_k_row_range[direct_bucket_idx]
            q_length_start, q_length_end = direct_bucket_q_length_range[direct_bucket_idx]
            k_length_start, k_length_end = direct_bucket_k_length_range[direct_bucket_idx]
            q_row_idx = direct_bucket_q_row_idx[q_row_start:q_row_end].contiguous().view(bucket_size_value, packed_q)
            q_row_idx_flat = q_row_idx.reshape(bucket_size_value * packed_q).contiguous()
            k_row_idx = direct_bucket_k_row_idx[k_row_start:k_row_end].contiguous().view(bucket_size_value, packed_k)
            q_length = direct_bucket_q_length[q_length_start:q_length_end].contiguous()
            k_length = direct_bucket_k_length[k_length_start:k_length_end].contiguous()
            if bool(direct_bucket_dense[direct_bucket_idx]):
                _run_synthetic_direct_micro_fwd_dense_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    q_row_idx,
                    k_row_idx,
                    q_length,
                    k_length,
                    temp_out,
                    temp_lse,
                    softmax_scale=softmax_scale,
                )
            else:
                words_per_row = int(direct_bucket_words_per_row[direct_bucket_idx])
                mask_word_start, mask_word_end = direct_bucket_mask_word_range[direct_bucket_idx]
                mask_words = direct_bucket_mask_words[mask_word_start:mask_word_end].contiguous().view(
                    bucket_size_value, packed_q, words_per_row
                )
                _run_synthetic_direct_micro_fwd_masked_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    q_row_idx,
                    k_row_idx,
                    q_length,
                    k_length,
                    mask_words,
                    temp_out,
                    temp_lse,
                    softmax_scale=softmax_scale,
                )
            _run_synthetic_direct_combine_rows_kernel(
                temp_out,
                temp_lse,
                q_row_idx_flat,
                total_out,
                total_lse,
            )
    return True


def _run_bucketed_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule,
    softmax_scale: float,
    metadata,
    *,
    runtime,
):
    hsa_module = _load_hsa_module()
    _flash_attn_fwd = hsa_module._lazy_cute_imports()[5]

    total_rows = schedule.num_rows
    num_q_heads = q.shape[2]
    head_dim_v = v.shape[3]
    q_flat = q.reshape(total_rows, q.shape[2], q.shape[3]).contiguous()
    k_flat = k.reshape(total_rows, k.shape[2], k.shape[3]).contiguous()
    v_flat = v.reshape(total_rows, v.shape[2], v.shape[3]).contiguous()
    total_out = torch.zeros(total_rows, num_q_heads, head_dim_v, dtype=torch.float32, device=q.device)
    total_lse = torch.full((total_rows, num_q_heads), float("-inf"), dtype=torch.float32, device=q.device)

    execution_plan = metadata.forward_execution_plan
    if execution_plan is None:
        raise RuntimeError("Synthetic packed metadata is missing the cached forward execution plan")

    direct_plan = execution_plan.get("direct_execution_plan")
    if (
        _synthetic_micro_fwd_enabled()
        and direct_plan is not None
        and metadata.logical_block_q == 2
        and metadata.logical_block_k == 2
        and _can_use_synthetic_micro_fwd(q_flat, k_flat, v_flat, packed_q=2, packed_k=1)
    ):
        if _can_use_direct_row_compact_runtime(metadata, q_flat, k_flat, v_flat) and _run_direct_row_compact_forward(
            q_flat,
            k_flat,
            v_flat,
            total_out,
            total_lse,
            metadata,
            execution_plan,
            softmax_scale=softmax_scale,
        ):
            out = total_out.to(dtype=q.dtype).view(q.shape[0], q.shape[1], q.shape[2], head_dim_v)
            lse = total_lse.view(q.shape[0], q.shape[1], q.shape[2]).permute(0, 2, 1).contiguous()
            return out, lse
        if _run_direct_segmented_forward(
            q_flat,
            k_flat,
            v_flat,
            total_out,
            total_lse,
            metadata,
            execution_plan,
            softmax_scale=softmax_scale,
            runtime=runtime,
        ):
            out = total_out.to(dtype=q.dtype).view(q.shape[0], q.shape[1], q.shape[2], head_dim_v)
            lse = total_lse.view(q.shape[0], q.shape[1], q.shape[2]).permute(0, 2, 1).contiguous()
            return out, lse

    workspace = runtime.synthetic_forward_workspace
    if workspace is None:
        workspace = {
            "qgroup_q": {},
            "bucket_q": {},
            "bucket_kv": {},
        }
        runtime.synthetic_forward_workspace = workspace

    qgroup_q_workspace = workspace["qgroup_q"]
    q_workspace = workspace["bucket_q"]
    kv_workspace = workspace["bucket_kv"]

    def get_q_workspace(cache: dict, rows: int, packed_q: int) -> torch.Tensor:
        key = (str(q.device), q.dtype, rows, packed_q, q.shape[2], q.shape[3])
        buf = cache.get(key)
        needed = rows * packed_q
        if buf is None or buf.shape[0] < needed:
            buf = torch.empty((needed, q.shape[2], q.shape[3]), dtype=q.dtype, device=q.device)
            cache[key] = buf
        return buf[:needed]

    def get_kv_workspace(rows: int, packed_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(k.device), k.dtype, v.dtype, rows, packed_k, k.shape[2], k.shape[3], v.shape[2], v.shape[3])
        bufs = kv_workspace.get(key)
        needed = rows * packed_k
        if bufs is None or bufs[0].shape[0] < needed:
            bufs = (
                torch.empty((needed, k.shape[2], k.shape[3]), dtype=k.dtype, device=k.device),
                torch.empty((needed, v.shape[2], v.shape[3]), dtype=v.dtype, device=v.device),
            )
            kv_workspace[key] = bufs
        return bufs[0][:needed], bufs[1][:needed]

    qgroup_bucket_packed_q = execution_plan["qgroup_bucket_packed_q"]
    qgroup_bucket_size = execution_plan["qgroup_bucket_size"]
    qgroup_bucket_q_row_range = execution_plan["qgroup_bucket_q_row_range"]
    qgroup_bucket_execution_bucket_range = execution_plan["qgroup_bucket_execution_bucket_range"]
    qgroup_bucket_execution_bucket_idx = execution_plan["qgroup_bucket_execution_bucket_idx"]
    bucket_size_plan = execution_plan["bucket_size"]
    bucket_packed_q = execution_plan["bucket_packed_q"]
    bucket_packed_k = execution_plan["bucket_packed_k"]
    bucket_dense = execution_plan["bucket_dense"]
    bucket_words_per_row = execution_plan["bucket_words_per_row"]
    bucket_q_row_range = execution_plan["bucket_q_row_range"]
    bucket_q_src_row_range = execution_plan["bucket_q_src_row_range"]
    bucket_k_row_range = execution_plan["bucket_k_row_range"]
    bucket_q_length_range = execution_plan["bucket_q_length_range"]
    bucket_k_length_range = execution_plan["bucket_k_length_range"]
    bucket_mask_word_range = execution_plan["bucket_mask_word_range"]
    bucket_use_qgroup_q = execution_plan.get("bucket_use_qgroup_q")
    bucket_scatter_only = execution_plan.get("bucket_scatter_only")

    for qgroup_bucket_idx, packed_q in enumerate(qgroup_bucket_packed_q):
        qgroup_bucket_size_value = qgroup_bucket_size[qgroup_bucket_idx]
        if qgroup_bucket_size_value <= 0:
            continue
        exec_bucket_start, exec_bucket_end = qgroup_bucket_execution_bucket_range[qgroup_bucket_idx]
        split_bucket_ids = qgroup_bucket_execution_bucket_idx[exec_bucket_start:exec_bucket_end]
        if len(split_bucket_ids) == 1:
            split_bucket_idx = int(split_bucket_ids[0])
            bucket_size = bucket_size_plan[split_bucket_idx]
            packed_q_bucket = bucket_packed_q[split_bucket_idx]
            packed_k = bucket_packed_k[split_bucket_idx]
            q_bucket_src_start, q_bucket_src_end = bucket_q_src_row_range[split_bucket_idx]
            k_bucket_start, k_bucket_end = bucket_k_row_range[split_bucket_idx]
            q_length_start, q_length_end = bucket_q_length_range[split_bucket_idx]
            k_length_start, k_length_end = bucket_k_length_range[split_bucket_idx]
            q_bucket_src_row_idx = metadata.bucket_q_src_row_idx[q_bucket_src_start:q_bucket_src_end].contiguous()
            k_bucket_row_idx = metadata.bucket_k_row_idx[k_bucket_start:k_bucket_end].contiguous()
            q_length = metadata.bucket_q_length[q_length_start:q_length_end].contiguous()
            k_length = metadata.bucket_k_length[k_length_start:k_length_end].contiguous()
            dense_bucket = bool(bucket_dense[split_bucket_idx])
            use_qgroup_q = False if bucket_use_qgroup_q is None else bool(bucket_use_qgroup_q[split_bucket_idx])
            scatter_only = False if bucket_scatter_only is None else bool(bucket_scatter_only[split_bucket_idx])
            if (
                scatter_only
                and use_qgroup_q
                and _synthetic_micro_fwd_enabled()
                and _can_use_synthetic_micro_fwd(q_flat, k_flat, v_flat, packed_q=packed_q_bucket, packed_k=packed_k)
            ):
                q_row_idx = q_bucket_src_row_idx.view(bucket_size, packed_q_bucket).contiguous()
                k_row_idx = k_bucket_row_idx.view(bucket_size, packed_k).contiguous()
                if dense_bucket:
                    _run_synthetic_direct_micro_fwd_dense_kernel(
                        q_flat,
                        k_flat,
                        v_flat,
                        q_row_idx,
                        k_row_idx,
                        q_length,
                        k_length,
                        total_out,
                        total_lse,
                        softmax_scale=softmax_scale,
                    )
                else:
                    words_per_row = bucket_words_per_row[split_bucket_idx]
                    mask_word_start, mask_word_end = bucket_mask_word_range[split_bucket_idx]
                    if mask_word_end <= mask_word_start:
                        raise RuntimeError("Synthetic packed bitmap bucket is missing mask words")
                    mask_words = metadata.bucket_mask_words[mask_word_start:mask_word_end].contiguous()
                    mask_words = mask_words.view(bucket_size, packed_q_bucket, words_per_row).contiguous()
                    _run_synthetic_direct_micro_fwd_masked_kernel(
                        q_flat,
                        k_flat,
                        v_flat,
                        q_row_idx,
                        k_row_idx,
                        q_length,
                        k_length,
                        mask_words,
                        total_out,
                        total_lse,
                        softmax_scale=softmax_scale,
                    )
                continue
        qgroup_q_start, qgroup_q_end = qgroup_bucket_q_row_range[qgroup_bucket_idx]
        qgroup_bucket_rows = metadata.qgroup_bucket_q_row_idx[qgroup_q_start:qgroup_q_end].contiguous()
        qgroup_q_buf_flat = get_q_workspace(qgroup_q_workspace, qgroup_bucket_size_value, packed_q)
        _run_synthetic_pack_rows_kernel(q_flat, qgroup_bucket_rows, qgroup_q_buf_flat)
        for split_bucket_idx in split_bucket_ids:
            bucket_size = bucket_size_plan[split_bucket_idx]
            if bucket_size <= 0:
                continue
            packed_q_bucket = bucket_packed_q[split_bucket_idx]
            packed_k = bucket_packed_k[split_bucket_idx]
            q_bucket_start, q_bucket_end = bucket_q_row_range[split_bucket_idx]
            q_bucket_src_start, q_bucket_src_end = bucket_q_src_row_range[split_bucket_idx]
            k_bucket_start, k_bucket_end = bucket_k_row_range[split_bucket_idx]
            q_length_start, q_length_end = bucket_q_length_range[split_bucket_idx]
            k_length_start, k_length_end = bucket_k_length_range[split_bucket_idx]
            q_bucket_src_row_idx = metadata.bucket_q_src_row_idx[q_bucket_src_start:q_bucket_src_end].contiguous()
            k_bucket_row_idx = metadata.bucket_k_row_idx[k_bucket_start:k_bucket_end].contiguous()
            q_length = metadata.bucket_q_length[q_length_start:q_length_end].contiguous()
            k_length = metadata.bucket_k_length[k_length_start:k_length_end].contiguous()

            use_qgroup_q = False if bucket_use_qgroup_q is None else bool(bucket_use_qgroup_q[split_bucket_idx])
            scatter_only = False if bucket_scatter_only is None else bool(bucket_scatter_only[split_bucket_idx])
            if use_qgroup_q:
                q_buf_flat = qgroup_q_buf_flat
            else:
                q_bucket_row_idx = metadata.bucket_q_row_idx[q_bucket_start:q_bucket_end].contiguous()
                q_buf_flat = get_q_workspace(q_workspace, bucket_size, packed_q_bucket)
                _run_synthetic_pack_rows_kernel(qgroup_q_buf_flat, q_bucket_row_idx, q_buf_flat)
            k_buf_flat, v_buf_flat = get_kv_workspace(bucket_size, packed_k)
            _run_synthetic_pack_kv_rows_kernel(k_flat, v_flat, k_bucket_row_idx, k_buf_flat, v_buf_flat)

            q_buf = q_buf_flat.view(bucket_size, packed_q_bucket, q.shape[2], q.shape[3])
            k_buf = k_buf_flat.view(bucket_size, packed_k, k.shape[2], k.shape[3])
            v_buf = v_buf_flat.view(bucket_size, packed_k, v.shape[2], v.shape[3])
            dense_bucket = bool(bucket_dense[split_bucket_idx])
            mask_words = None
            if not dense_bucket:
                words_per_row = bucket_words_per_row[split_bucket_idx]
                mask_word_start, mask_word_end = bucket_mask_word_range[split_bucket_idx]
                if mask_word_end <= mask_word_start:
                    raise RuntimeError("Synthetic packed bitmap bucket is missing mask words")
                mask_words_flat = metadata.bucket_mask_words[mask_word_start:mask_word_end].contiguous()
                mask_words = mask_words_flat.view(bucket_size, packed_q_bucket, words_per_row).contiguous()

            if scatter_only and _synthetic_micro_fwd_enabled() and _can_use_synthetic_micro_fwd(
                q_buf,
                k_buf,
                v_buf,
                packed_q=packed_q_bucket,
                packed_k=packed_k,
            ):
                if dense_bucket:
                    out_bucket, lse_bucket = _run_synthetic_micro_fwd_dense_kernel(
                        q_buf,
                        k_buf,
                        v_buf,
                        q_length,
                        k_length,
                        softmax_scale=softmax_scale,
                    )
                else:
                    out_bucket, lse_bucket = _run_synthetic_micro_fwd_masked_kernel(
                        q_buf,
                        k_buf,
                        v_buf,
                        q_length,
                        k_length,
                        mask_words,
                        softmax_scale=softmax_scale,
                    )
                out_bucket_flat = out_bucket.reshape(bucket_size * packed_q_bucket, num_q_heads, head_dim_v).contiguous()
                lse_bucket_flat = lse_bucket.reshape(bucket_size * packed_q_bucket, num_q_heads).contiguous()
            else:
                q_length = hsa_module._tag_aux_tensor(q_length, leading_dim=0)
                k_length = hsa_module._tag_aux_tensor(k_length, leading_dim=0)

                if dense_bucket:
                    mask_mod = hsa_module.get_hsa_synthetic_packed_dense_mask_mod()
                    aux_tensors = [q_length, k_length]
                else:
                    assert mask_words is not None
                    words_per_row = bucket_words_per_row[split_bucket_idx]
                    mask_words = hsa_module._tag_aux_tensor(mask_words, leading_dim=2)
                    mask_mod = hsa_module.get_hsa_synthetic_packed_bitmap_mask_mod(words_per_row)
                    aux_tensors = [q_length, k_length, mask_words]

                out_bucket, lse_bucket = _flash_attn_fwd(
                    q_buf,
                    k_buf,
                    v_buf,
                    softmax_scale=softmax_scale,
                    causal=False,
                    m_block_size=128,
                    n_block_size=128,
                    pack_gqa=False,
                    mask_mod=mask_mod,
                    aux_tensors=aux_tensors,
                    return_lse=True,
                )
                out_bucket_flat = out_bucket.reshape(bucket_size * packed_q_bucket, num_q_heads, head_dim_v).contiguous()
                lse_bucket_flat = lse_bucket.permute(0, 2, 1).reshape(bucket_size * packed_q_bucket, num_q_heads).contiguous()
            if scatter_only:
                _run_synthetic_scatter_rows_kernel(out_bucket_flat, q_bucket_src_row_idx, total_out)
                _run_synthetic_scatter_lse_kernel(lse_bucket_flat, q_bucket_src_row_idx, total_lse)
            else:
                _run_synthetic_combine_scatter_rows_kernel(
                    out_bucket_flat,
                    lse_bucket_flat,
                    q_bucket_src_row_idx,
                    total_out,
                    total_lse,
                )

    out = total_out.to(dtype=q.dtype).view(q.shape[0], q.shape[1], q.shape[2], head_dim_v)
    lse = total_lse.view(q.shape[0], q.shape[1], q.shape[2]).permute(0, 2, 1).contiguous()
    return out, lse


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
    if not _is_mixed_schedule(schedule) or q.shape[2] != k.shape[2]:
        from flash_attn.cute.flash_hsa_fwd_sm100 import run_hsa_fwd_sm100_blocksparse

        return run_hsa_fwd_sm100_blocksparse(q, k, v, schedule, softmax_scale)

    metadata = runtime.forward_synthetic_grid
    if metadata is None:
        raise RuntimeError("Synthetic packed forward metadata is unavailable")
    out, lse = _run_bucketed_forward(q, k, v, schedule, softmax_scale, metadata, runtime=runtime)
    sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream = _empty_sentence_cache(
        q, k, v
    )
    return out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream


def _can_use_direct_synthetic_micro_runtime(metadata, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if metadata is None or metadata.forward_execution_plan is None:
        return False
    plan = metadata.forward_execution_plan
    direct_plan = plan.get("direct_execution_plan")
    if metadata.logical_block_q != 2 or metadata.logical_block_k != 2:
        return False
    if direct_plan is None or len(direct_plan["bucket_packed_k"]) <= 0:
        return False
    packed_q = 2
    packed_k = max(int(packed_k) for packed_k in direct_plan["bucket_packed_k"])
    return _can_use_synthetic_micro_fwd(q, k, v, packed_q=packed_q, packed_k=packed_k)


def _can_use_direct_row_compact_runtime(metadata, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if metadata is None or metadata.forward_execution_plan is None:
        return False
    plan = metadata.forward_execution_plan
    direct_plan = plan.get("direct_execution_plan")
    if metadata.logical_block_q != 2 or metadata.logical_block_k != 2 or direct_plan is None:
        return False
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is None or len(row_plan["bucket_row_k_cap"]) <= 0:
        return False
    if q.shape[-1] != 64 or v.shape[-1] != 64:
        return False
    max_row_k = max(int(row_k_cap) for row_k_cap in row_plan["bucket_row_k_cap"])
    return max_row_k <= int(row_plan["row_k_cap_limit"]) and _can_use_synthetic_micro_fwd(
        q, k, v, packed_q=2, packed_k=max_row_k
    )


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
    from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_blocksparse

    runtime = _ensure_runtime(schedule, runtime, q, k)
    metadata = runtime.forward_synthetic_grid
    if (
        not _is_mixed_schedule(schedule)
        or q.shape[2] != k.shape[2]
        or not _synthetic_micro_fwd_enabled()
        or not _synthetic_micro_bwd_enabled()
        or not _can_use_direct_synthetic_micro_runtime(metadata, q.reshape(-1, q.shape[2], q.shape[3]), k.reshape(-1, k.shape[2], k.shape[3]), v.reshape(-1, v.shape[2], v.shape[3]))
    ):
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
        )

    plan = metadata.forward_execution_plan
    direct_plan = None if plan is None else plan.get("direct_execution_plan")
    assert metadata is not None and plan is not None and direct_plan is not None

    total_rows = schedule.num_rows
    q_flat = q.reshape(total_rows, q.shape[2], q.shape[3]).contiguous()
    k_flat = k.reshape(total_rows, k.shape[2], k.shape[3]).contiguous()
    v_flat = v.reshape(total_rows, v.shape[2], v.shape[3]).contiguous()
    out_flat = out.reshape(total_rows, out.shape[2], out.shape[3]).contiguous()
    dout_flat = dout.reshape(total_rows, dout.shape[2], dout.shape[3]).contiguous()
    lse_flat = lse.permute(0, 2, 1).reshape(total_rows, q.shape[2]).contiguous()

    workspace = runtime.synthetic_backward_workspace
    if workspace is None:
        workspace = {}
        runtime.synthetic_backward_workspace = workspace
    workspace_key = (
        str(q.device),
        total_rows,
        q.shape[2],
        q.shape[3],
        k.shape[2],
        k.shape[3],
        v.shape[2],
        v.shape[3],
    )
    bufs = workspace.get(workspace_key)
    if bufs is None:
        bufs = {
            "dq": torch.empty((total_rows, q.shape[2], q.shape[3]), dtype=torch.float32, device=q.device),
            "dk": torch.empty((total_rows, k.shape[2], k.shape[3]), dtype=torch.float32, device=q.device),
            "dv": torch.empty((total_rows, v.shape[2], v.shape[3]), dtype=torch.float32, device=q.device),
        }
        workspace[workspace_key] = bufs
    dq_accum = bufs["dq"]
    dk_accum = bufs["dk"]
    dv_accum = bufs["dv"]
    dq_accum.zero_()
    dk_accum.zero_()
    dv_accum.zero_()

    direct_qgroup_bucket_segment_range = direct_plan["qgroup_bucket_segment_range"]
    direct_qgroup_bucket_segment_idx = direct_plan["qgroup_bucket_segment_idx"]
    direct_bucket_size = direct_plan["bucket_size"]
    direct_bucket_packed_q = direct_plan["bucket_packed_q"]
    direct_bucket_packed_k = direct_plan["bucket_packed_k"]
    direct_bucket_dense = direct_plan["bucket_dense"]
    direct_bucket_words_per_row = direct_plan["bucket_words_per_row"]
    direct_bucket_q_row_range = direct_plan["bucket_q_row_range"]
    direct_bucket_k_row_range = direct_plan["bucket_k_row_range"]
    direct_bucket_q_length_range = direct_plan["bucket_q_length_range"]
    direct_bucket_k_length_range = direct_plan["bucket_k_length_range"]
    direct_bucket_mask_word_range = direct_plan["bucket_mask_word_range"]
    direct_bucket_q_row_idx = direct_plan["bucket_q_row_idx"]
    direct_bucket_k_row_idx = direct_plan["bucket_k_row_idx"]
    direct_bucket_q_length = direct_plan["bucket_q_length"]
    direct_bucket_k_length = direct_plan["bucket_k_length"]
    direct_bucket_mask_words = direct_plan["bucket_mask_words"]
    row_plan = direct_plan.get("row_compact_plan")
    row_compact_active = _can_use_direct_row_compact_runtime(metadata, q_flat, k_flat, v_flat) and row_plan is not None
    processed_direct_bucket_ids: set[int] = set()

    if row_compact_active:
        row_bucket_row_k_cap = row_plan["bucket_row_k_cap"]
        row_bucket_max_unique_key_occurrences = row_plan.get(
            "bucket_max_unique_key_occurrences", [0] * len(row_bucket_row_k_cap)
        )
        row_bucket_row_k_range = row_plan["bucket_row_k_range"]
        row_bucket_row_k_to_union_range = row_plan["bucket_row_k_to_union_range"]
        row_bucket_union_to_row_range = row_plan["bucket_union_to_row_range"]
        row_bucket_row_k_length_range = row_plan["bucket_row_k_length_range"]
        row_bucket_unique_key_range = row_plan["bucket_unique_key_range"]
        row_bucket_unique_key_occurrence_range = row_plan["bucket_unique_key_occurrence_range"]
        row_bucket_unique_key_occurrence_ptr_range = row_plan["bucket_unique_key_occurrence_ptr_range"]
        row_bucket_row_k_row_idx = row_plan["bucket_row_k_row_idx"]
        row_bucket_row_k_to_union_idx = row_plan["bucket_row_k_to_union_idx"]
        row_bucket_union_to_row_slot = row_plan["bucket_union_to_row_slot"]
        row_bucket_row_k_length = row_plan["bucket_row_k_length"]
        row_bucket_unique_key_row_idx = row_plan["bucket_unique_key_row_idx"]
        row_bucket_unique_key_member_idx = row_plan["bucket_unique_key_member_idx"]
        row_bucket_unique_key_union_idx = row_plan["bucket_unique_key_union_idx"]
        row_bucket_unique_key_occurrence_row_ptr = row_plan["bucket_unique_key_occurrence_row_ptr"]
        for direct_bucket_idx, row_k_cap in enumerate(row_bucket_row_k_cap):
            packed_q = int(direct_bucket_packed_q[direct_bucket_idx])
            packed_k = int(direct_bucket_packed_k[direct_bucket_idx])
            if packed_q != 2 or int(row_k_cap) <= 0 or packed_k > 16:
                continue
            bucket_size = int(direct_bucket_size[direct_bucket_idx])
            q_row_start, q_row_end = direct_bucket_q_row_range[direct_bucket_idx]
            k_row_start, k_row_end = direct_bucket_k_row_range[direct_bucket_idx]
            q_length_start, q_length_end = direct_bucket_q_length_range[direct_bucket_idx]
            k_length_start, k_length_end = direct_bucket_k_length_range[direct_bucket_idx]
            row_k_start, row_k_end = row_bucket_row_k_range[direct_bucket_idx]
            row_k_to_union_start, row_k_to_union_end = row_bucket_row_k_to_union_range[direct_bucket_idx]
            union_to_row_start, union_to_row_end = row_bucket_union_to_row_range[direct_bucket_idx]
            row_k_length_start, row_k_length_end = row_bucket_row_k_length_range[direct_bucket_idx]
            unique_key_start, unique_key_end = row_bucket_unique_key_range[direct_bucket_idx]
            unique_ptr_start, unique_ptr_end = row_bucket_unique_key_occurrence_ptr_range[direct_bucket_idx]
            q_row_idx = direct_bucket_q_row_idx[q_row_start:q_row_end].contiguous().view(bucket_size, 2)
            q_length = direct_bucket_q_length[q_length_start:q_length_end].contiguous()
            row_k_row_idx = row_bucket_row_k_row_idx[row_k_start:row_k_end].contiguous().view(bucket_size, 2, int(row_k_cap))
            union_k_row_idx = direct_bucket_k_row_idx[k_row_start:k_row_end].contiguous().view(bucket_size, packed_k)
            row_k_to_union_idx = row_bucket_row_k_to_union_idx[row_k_to_union_start:row_k_to_union_end].contiguous().view(
                bucket_size, 2, int(row_k_cap)
            )
            union_to_row_slot = row_bucket_union_to_row_slot[union_to_row_start:union_to_row_end].contiguous().view(
                bucket_size, 2, packed_k
            )
            row_k_length = row_bucket_row_k_length[row_k_length_start:row_k_length_end].contiguous().view(bucket_size, 2)
            union_k_length = direct_bucket_k_length[k_length_start:k_length_end].contiguous()
            row_compact_bwd_mode = "legacy"
            unique_key_row_idx = None
            unique_key_member_idx = None
            unique_key_union_idx = None
            unique_key_occurrence_row_ptr = None
            if unique_key_end > unique_key_start:
                occ_start = row_bucket_unique_key_occurrence_range[unique_key_start][0]
                occ_end = row_bucket_unique_key_occurrence_range[unique_key_end - 1][1]
                unique_key_row_idx = row_bucket_unique_key_row_idx[unique_key_start:unique_key_end].contiguous()
                unique_key_member_idx = row_bucket_unique_key_member_idx[occ_start:occ_end].contiguous()
                unique_key_union_idx = row_bucket_unique_key_union_idx[occ_start:occ_end].contiguous()
                unique_key_occurrence_row_ptr = (
                    row_bucket_unique_key_occurrence_row_ptr[unique_ptr_start:unique_ptr_end] - int(occ_start)
                ).contiguous()
                row_compact_bwd_mode = _select_row_compact_synthetic_bwd_mode(
                    q_flat,
                    union_k_row_idx,
                    q_row_idx,
                    unique_key_row_idx,
                    int(row_bucket_max_unique_key_occurrences[direct_bucket_idx]),
                )
            if row_compact_bwd_mode == "one_kernel":
                _run_synthetic_direct_row_micro_bwd_kernel_one_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    out_flat,
                    dout_flat,
                    lse_flat,
                    q_row_idx,
                    union_to_row_slot,
                    unique_key_row_idx,
                    unique_key_member_idx,
                    unique_key_union_idx,
                    unique_key_occurrence_row_ptr,
                    dq_accum,
                    dk_accum,
                    dv_accum,
                    softmax_scale=softmax_scale,
                )
            elif row_compact_bwd_mode == "split":
                _run_synthetic_direct_row_micro_bwd_kernel_dq_only(
                    q_flat,
                    k_flat,
                    v_flat,
                    out_flat,
                    dout_flat,
                    lse_flat,
                    q_row_idx,
                    union_k_row_idx,
                    union_to_row_slot,
                    q_length,
                    union_k_length,
                    dq_accum,
                    softmax_scale=softmax_scale,
                )
                _run_synthetic_direct_row_micro_bwd_kernel_key_owned(
                    q_flat,
                    k_flat,
                    v_flat,
                    out_flat,
                    dout_flat,
                    lse_flat,
                    q_row_idx,
                    union_to_row_slot,
                    unique_key_row_idx,
                    unique_key_member_idx,
                    unique_key_union_idx,
                    unique_key_occurrence_row_ptr,
                    dk_accum,
                    dv_accum,
                    softmax_scale=softmax_scale,
                )
            else:
                _run_synthetic_direct_row_micro_bwd_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    out_flat,
                    dout_flat,
                    lse_flat,
                    q_row_idx,
                    row_k_row_idx,
                    union_k_row_idx,
                    row_k_to_union_idx,
                    union_to_row_slot,
                    q_length,
                    row_k_length,
                    union_k_length,
                    dq_accum,
                    dk_accum,
                    dv_accum,
                    softmax_scale=softmax_scale,
                )
            processed_direct_bucket_ids.add(direct_bucket_idx)
        if len(processed_direct_bucket_ids) == len(direct_bucket_size):
            dq = dq_accum.to(dtype=q.dtype).view_as(q)
            dk = dk_accum.to(dtype=k.dtype).view_as(k)
            dv = dv_accum.to(dtype=v.dtype).view_as(v)
            return dq, dk, dv

    for qgroup_bucket_idx in range(len(direct_qgroup_bucket_segment_range)):
        segment_start, segment_end = direct_qgroup_bucket_segment_range[qgroup_bucket_idx]
        direct_bucket_ids = direct_qgroup_bucket_segment_idx[segment_start:segment_end]
        for direct_bucket_idx in direct_bucket_ids:
            direct_bucket_idx = int(direct_bucket_idx)
            if row_compact_active and direct_bucket_idx in processed_direct_bucket_ids:
                continue
            bucket_size = int(direct_bucket_size[direct_bucket_idx])
            packed_q = int(direct_bucket_packed_q[direct_bucket_idx])
            q_row_start, q_row_end = direct_bucket_q_row_range[direct_bucket_idx]
            packed_k = int(direct_bucket_packed_k[direct_bucket_idx])
            k_row_start, k_row_end = direct_bucket_k_row_range[direct_bucket_idx]
            q_length_start, q_length_end = direct_bucket_q_length_range[direct_bucket_idx]
            k_length_start, k_length_end = direct_bucket_k_length_range[direct_bucket_idx]
            q_row_idx = direct_bucket_q_row_idx[q_row_start:q_row_end].contiguous().view(bucket_size, packed_q)
            k_row_idx = direct_bucket_k_row_idx[k_row_start:k_row_end].contiguous().view(bucket_size, packed_k)
            q_length = direct_bucket_q_length[q_length_start:q_length_end].contiguous()
            k_length = direct_bucket_k_length[k_length_start:k_length_end].contiguous()

            if bool(direct_bucket_dense[direct_bucket_idx]):
                _run_synthetic_direct_micro_bwd_dense_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    out_flat,
                    dout_flat,
                    lse_flat,
                    q_row_idx,
                    k_row_idx,
                    q_length,
                    k_length,
                    dq_accum,
                    dk_accum,
                    dv_accum,
                    softmax_scale=softmax_scale,
                )
            else:
                words_per_row = int(direct_bucket_words_per_row[direct_bucket_idx])
                mask_word_start, mask_word_end = direct_bucket_mask_word_range[direct_bucket_idx]
                if mask_word_end <= mask_word_start:
                    raise RuntimeError("Synthetic packed bitmap bucket is missing mask words")
                mask_words = direct_bucket_mask_words[mask_word_start:mask_word_end].contiguous().view(
                    bucket_size, packed_q, words_per_row
                )
                _run_synthetic_direct_micro_bwd_masked_kernel(
                    q_flat,
                    k_flat,
                    v_flat,
                    out_flat,
                    dout_flat,
                    lse_flat,
                    q_row_idx,
                    k_row_idx,
                    q_length,
                    k_length,
                    mask_words,
                    dq_accum,
                    dk_accum,
                    dv_accum,
                    softmax_scale=softmax_scale,
                )

    dq = dq_accum.to(dtype=q.dtype).view_as(q)
    dk = dk_accum.to(dtype=k.dtype).view_as(k)
    dv = dv_accum.to(dtype=v.dtype).view_as(v)
    return dq, dk, dv
