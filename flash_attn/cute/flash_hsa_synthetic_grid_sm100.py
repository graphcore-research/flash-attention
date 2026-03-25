from __future__ import annotations

import math

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Float32, Int32

    from flash_attn.cute.cache_utils import get_jit_cache
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor
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
    if reference is None:
        return {
            "logical_block_q": 0,
            "logical_block_k": 0,
            "max_packed_k": 0,
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
        }

    return {
        "logical_block_q": reference.logical_block_q,
        "logical_block_k": reference.logical_block_k,
        "max_packed_k": 0 if reference.max_packed_k is None else reference.max_packed_k,
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
        qgroup_q_start, qgroup_q_end = qgroup_bucket_q_row_range[qgroup_bucket_idx]
        qgroup_bucket_rows = metadata.qgroup_bucket_q_row_idx[qgroup_q_start:qgroup_q_end].contiguous()
        qgroup_q_buf_flat = get_q_workspace(qgroup_q_workspace, qgroup_bucket_size_value, packed_q)
        _run_synthetic_pack_rows_kernel(q_flat, qgroup_bucket_rows, qgroup_q_buf_flat)
        exec_bucket_start, exec_bucket_end = qgroup_bucket_execution_bucket_range[qgroup_bucket_idx]
        split_bucket_ids = qgroup_bucket_execution_bucket_idx[exec_bucket_start:exec_bucket_end]
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
            q_length = hsa_module._tag_aux_tensor(q_length, leading_dim=0)
            k_length = hsa_module._tag_aux_tensor(k_length, leading_dim=0)

            if bool(bucket_dense[split_bucket_idx]):
                mask_mod = hsa_module.get_hsa_synthetic_packed_dense_mask_mod()
                aux_tensors = [q_length, k_length]
            else:
                words_per_row = bucket_words_per_row[split_bucket_idx]
                mask_word_start, mask_word_end = bucket_mask_word_range[split_bucket_idx]
                if mask_word_end <= mask_word_start:
                    raise RuntimeError("Synthetic packed bitmap bucket is missing mask words")
                mask_words_flat = metadata.bucket_mask_words[mask_word_start:mask_word_end].contiguous()
                mask_words = mask_words_flat.view(bucket_size, packed_q_bucket, words_per_row).contiguous()
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
    del runtime
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
    )
