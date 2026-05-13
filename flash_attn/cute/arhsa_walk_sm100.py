from __future__ import annotations

import math

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32
    from cutlass.cute.nvgpu import warp
    from quack import layout_utils

    from flash_attn.cute import ampere_helpers as sm80_utils
    from flash_attn.cute import copy_utils
    from flash_attn.cute import utils as cute_utils
    from flash_attn.cute.cache_utils import get_jit_cache
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    _HAS_CUTE_RUNTIME = True
except Exception:  # pragma: no cover - CPU-only guard
    _HAS_CUTE_RUNTIME = False

    class _FakeCuda:
        class CUstream:
            pass

    class _FakeCute:
        Tensor = object

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def kernel(fn):
            return fn

    cuda = _FakeCuda()
    cute = _FakeCute()
    cutlass = object()
    Float32 = float
    Int32 = int

    class _FakeCuteUtils:
        @staticmethod
        def atomic_add_fp32(*_args, **_kwargs):
            raise NotImplementedError("CuTe runtime is unavailable")

        @staticmethod
        def elem_pointer(*_args, **_kwargs):
            raise NotImplementedError("CuTe runtime is unavailable")

    cute_utils = _FakeCuteUtils()
    copy_utils = _FakeCuteUtils()

    def get_jit_cache(_name):
        return {}

    def to_cute_tensor(*_args, **_kwargs):
        return None


_LOG2_E = math.log2(math.e)
_CUTE_BACKWARD_DTYPES = (torch.float32, torch.bfloat16)


def _require_cute_runtime() -> None:
    if not _HAS_CUTE_RUNTIME:
        raise NotImplementedError("ARHSA walk kernels require CUDA/CuTe runtime")


def build_incoming_edge_csr(dst: torch.Tensor, *, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build destination-major CSR over edge ids for on-the-fly reductions."""
    if dst.ndim != 1:
        raise ValueError(f"dst must be 1D, got {tuple(dst.shape)}")
    dst_cpu = dst.detach().to(device="cpu", dtype=torch.long)
    buckets: list[list[int]] = [[] for _ in range(int(n_nodes))]
    for edge_idx, dst_idx in enumerate(dst_cpu.tolist()):
        if dst_idx < 0 or dst_idx >= n_nodes:
            raise ValueError(f"edge {edge_idx} has destination {dst_idx}, outside [0, {n_nodes})")
        buckets[int(dst_idx)].append(edge_idx)
    row_ptr = [0]
    edge_ids: list[int] = []
    for bucket in buckets:
        edge_ids.extend(bucket)
        row_ptr.append(len(edge_ids))
    device = dst.device
    return (
        torch.tensor(row_ptr, dtype=torch.int32, device=device),
        torch.tensor(edge_ids, dtype=torch.int32, device=device),
    )


def build_outgoing_edge_csr(src: torch.Tensor, *, n_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build source-major CSR over edge ids for outgoing softmax reductions."""
    if src.ndim != 1:
        raise ValueError(f"src must be 1D, got {tuple(src.shape)}")
    src_cpu = src.detach().to(device="cpu", dtype=torch.long)
    buckets: list[list[int]] = [[] for _ in range(int(n_nodes))]
    for edge_idx, src_idx in enumerate(src_cpu.tolist()):
        if src_idx < 0 or src_idx >= n_nodes:
            raise ValueError(f"edge {edge_idx} has source {src_idx}, outside [0, {n_nodes})")
        buckets[int(src_idx)].append(edge_idx)
    row_ptr = [0]
    edge_ids: list[int] = []
    for bucket in buckets:
        edge_ids.extend(bucket)
        row_ptr.append(len(edge_ids))
    device = src.device
    return (
        torch.tensor(row_ptr, dtype=torch.int32, device=device),
        torch.tensor(edge_ids, dtype=torch.int32, device=device),
    )


def build_query_leaf_csr(leaf_query_index: torch.Tensor, *, n_queries: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build query-major CSR over leaf-entry ids for readout reductions."""
    if leaf_query_index.ndim != 1:
        raise ValueError(f"leaf_query_index must be 1D, got {tuple(leaf_query_index.shape)}")
    query_cpu = leaf_query_index.detach().to(device="cpu", dtype=torch.long)
    buckets: list[list[int]] = [[] for _ in range(int(n_queries))]
    for leaf_entry, query_idx in enumerate(query_cpu.tolist()):
        if query_idx < 0 or query_idx >= n_queries:
            raise ValueError(
                f"leaf entry {leaf_entry} has query {query_idx}, outside [0, {n_queries})"
            )
        buckets[int(query_idx)].append(leaf_entry)
    row_ptr = [0]
    leaf_entries: list[int] = []
    for bucket in buckets:
        leaf_entries.extend(bucket)
        row_ptr.append(len(leaf_entries))
    device = leaf_query_index.device
    return (
        torch.tensor(row_ptr, dtype=torch.int32, device=device),
        torch.tensor(leaf_entries, dtype=torch.int32, device=device),
    )


def outgoing_softmax_from_scores(edge_scores: torch.Tensor, src: torch.Tensor, *, n_nodes: int) -> torch.Tensor:
    """Torch reference grouped softmax over outgoing edges for each source node."""
    if edge_scores.ndim != 2:
        raise ValueError(f"edge_scores must have shape [n_edges, n_heads], got {tuple(edge_scores.shape)}")
    src = src.to(device=edge_scores.device, dtype=torch.long).contiguous()
    if src.ndim != 1 or src.shape[0] != edge_scores.shape[0]:
        raise ValueError("src must be 1D with length matching edge_scores")
    n_nodes = int(n_nodes)
    if n_nodes < 0:
        raise ValueError("n_nodes must be >= 0")
    n_heads = int(edge_scores.shape[1])
    src_expanded = src[:, None].expand(-1, n_heads)
    max_per_src = torch.full(
        (n_nodes, n_heads),
        float("-inf"),
        dtype=edge_scores.dtype,
        device=edge_scores.device,
    )
    max_per_src.scatter_reduce_(0, src_expanded, edge_scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(edge_scores - max_per_src[src])
    sum_per_src = torch.zeros(n_nodes, n_heads, dtype=edge_scores.dtype, device=edge_scores.device)
    sum_per_src.scatter_add_(0, src_expanded, exp_scores)
    return exp_scores / sum_per_src[src].clamp(min=1e-8)


class ARHSAMarkovIncomingStepSm100:
    """One exact Markov step as a destination-side incoming-edge reduction."""

    arch = 100

    def __init__(self, *, num_threads: int = 96):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mEdgeProb: cute.Tensor,
        mSrc: cute.Tensor,
        mDstRowPtr: cute.Tensor,
        mDstEdgeIndex: cute.Tensor,
        mNodeIsSink: cute.Tensor,
        mPNext: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mEdgeProb,
            mSrc,
            mDstRowPtr,
            mDstEdgeIndex,
            mNodeIsSink,
            mPNext,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mEdgeProb: cute.Tensor,
        mSrc: cute.Tensor,
        mDstRowPtr: cute.Tensor,
        mDstEdgeIndex: cute.Tensor,
        mNodeIsSink: cute.Tensor,
        mPNext: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            node_idx = task_idx // num_heads
            head_idx = task_idx - node_idx * num_heads
            acc = Float32.zero
            if mNodeIsSink[node_idx]:
                acc = Float32(mP[node_idx, head_idx])
            start = Int32(mDstRowPtr[node_idx])
            end = Int32(mDstRowPtr[node_idx + 1])
            for ptr in cutlass.range(start, end, unroll=1):
                edge_idx = Int32(mDstEdgeIndex[ptr])
                src_idx = Int32(mSrc[edge_idx])
                acc += Float32(mP[src_idx, head_idx]) * Float32(mEdgeProb[edge_idx, head_idx])
            mPNext[node_idx, head_idx] = acc.to(mPNext.element_type)


class ARHSAOutgoingSoftmaxSm100:
    """Grouped softmax over outgoing edges for each source node and head."""

    arch = 100

    def __init__(self, *, num_threads: int = 64):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mEdgeScores: cute.Tensor,
        mSrcRowPtr: cute.Tensor,
        mSrcEdgeIndex: cute.Tensor,
        mEdgeProb: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mEdgeScores,
            mSrcRowPtr,
            mSrcEdgeIndex,
            mEdgeProb,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mEdgeScores: cute.Tensor,
        mSrcRowPtr: cute.Tensor,
        mSrcEdgeIndex: cute.Tensor,
        mEdgeProb: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mEdgeScores.shape[1])
            src_idx = task_idx // num_heads
            head_idx = task_idx - src_idx * num_heads
            start = Int32(mSrcRowPtr[src_idx])
            end = Int32(mSrcRowPtr[src_idx + 1])
            if start < end:
                row_max = Float32(-3.4028234663852886e38)
                for ptr in cutlass.range(start, end, unroll=1):
                    edge_idx = Int32(mSrcEdgeIndex[ptr])
                    score = Float32(mEdgeScores[edge_idx, head_idx])
                    if score > row_max:
                        row_max = score
                row_sum = Float32.zero
                for ptr in cutlass.range(start, end, unroll=1):
                    edge_idx = Int32(mSrcEdgeIndex[ptr])
                    score = Float32(mEdgeScores[edge_idx, head_idx])
                    row_sum += cute.math.exp2((score - row_max) * Float32(_LOG2_E), fastmath=True)
                if row_sum < Float32(1.0e-8):
                    row_sum = Float32(1.0e-8)
                for ptr in cutlass.range(start, end, unroll=1):
                    edge_idx = Int32(mSrcEdgeIndex[ptr])
                    score = Float32(mEdgeScores[edge_idx, head_idx])
                    prob = cute.math.exp2((score - row_max) * Float32(_LOG2_E), fastmath=True) / row_sum
                    mEdgeProb[edge_idx, head_idx] = prob.to(mEdgeProb.element_type)


class ARHSAOutgoingSoftmaxBackwardSm100:
    """Grouped outgoing softmax backward for each source node and head."""

    arch = 100

    def __init__(self, *, num_threads: int = 64):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mEdgeProb: cute.Tensor,
        mGradEdgeProb: cute.Tensor,
        mSrcRowPtr: cute.Tensor,
        mSrcEdgeIndex: cute.Tensor,
        mGradEdgeScores: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mEdgeProb,
            mGradEdgeProb,
            mSrcRowPtr,
            mSrcEdgeIndex,
            mGradEdgeScores,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mEdgeProb: cute.Tensor,
        mGradEdgeProb: cute.Tensor,
        mSrcRowPtr: cute.Tensor,
        mSrcEdgeIndex: cute.Tensor,
        mGradEdgeScores: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mEdgeProb.shape[1])
            src_idx = task_idx // num_heads
            head_idx = task_idx - src_idx * num_heads
            start = Int32(mSrcRowPtr[src_idx])
            end = Int32(mSrcRowPtr[src_idx + 1])
            dot = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                edge_idx = Int32(mSrcEdgeIndex[ptr])
                dot += Float32(mGradEdgeProb[edge_idx, head_idx]) * Float32(mEdgeProb[edge_idx, head_idx])
            for ptr in cutlass.range(start, end, unroll=1):
                edge_idx = Int32(mSrcEdgeIndex[ptr])
                grad_score = Float32(mEdgeProb[edge_idx, head_idx]) * (
                    Float32(mGradEdgeProb[edge_idx, head_idx]) - dot
                )
                mGradEdgeScores[edge_idx, head_idx] = grad_score.to(mGradEdgeScores.element_type)


class ARHSALeafReadoutSm100:
    """Normalize leaf mass and reduce values per query/head/value dimension."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mReadout: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mReadout,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mReadout: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            elems_per_query = num_heads * head_dim_v
            query_idx = task_idx // elems_per_query
            rem = task_idx - query_idx * elems_per_query
            head_idx = rem // head_dim_v
            dim_idx = rem - head_idx * head_dim_v
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            denom = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                denom += Float32(mP[node_idx, head_idx])
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            acc = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                value_idx = Int32(mLeafValueIndex[leaf_entry])
                weight = Float32(mP[node_idx, head_idx]) / denom
                acc += weight * Float32(mValue[value_idx, head_idx, dim_idx])
            mReadout[query_idx, head_idx, dim_idx] = acc.to(mReadout.element_type)


class ARHSALeafReadoutWithDenomSm100:
    """Leaf readout variant that also stores the per-query/head denominator."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mReadout: cute.Tensor,
        mDenom: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mReadout,
            mDenom,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mReadout: cute.Tensor,
        mDenom: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            elems_per_query = num_heads * head_dim_v
            query_idx = task_idx // elems_per_query
            rem = task_idx - query_idx * elems_per_query
            head_idx = rem // head_dim_v
            dim_idx = rem - head_idx * head_dim_v
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            denom = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                denom += Float32(mP[node_idx, head_idx])
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            if dim_idx == Int32(0):
                mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)
            acc = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                value_idx = Int32(mLeafValueIndex[leaf_entry])
                weight = Float32(mP[node_idx, head_idx]) / denom
                acc += weight * Float32(mValue[value_idx, head_idx, dim_idx])
            mReadout[query_idx, head_idx, dim_idx] = acc.to(mReadout.element_type)


class ARHSALeafReadoutQueryWarpSm100:
    """Query/head-owned warp readout for small leaf fanout."""

    arch = 100

    def __init__(
        self,
        *,
        num_threads: int = 64,
        head_dim_is_64: bool = False,
        write_denom: bool = False,
    ):
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32
        self.head_dim_is_64 = head_dim_is_64
        self.write_denom = write_denom

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mReadout: cute.Tensor,
        mDenom: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.head_dim_is_64):
            grid_x = cute.ceil_div(total_tasks, self.warps_per_cta * 2)
        else:
            grid_x = cute.ceil_div(total_tasks, self.warps_per_cta)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mReadout,
            mDenom,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mReadout: cute.Tensor,
        mDenom: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        if cutlass.const_expr(self.head_dim_is_64):
            half_warp = lane // Int32(16)
            lane16 = lane - half_warp * Int32(16)
            task_idx = block_idx * Int32(self.warps_per_cta * 2) + warp_idx * Int32(2) + half_warp
            if task_idx < total_tasks:
                num_heads = Int32(mP.shape[1])
                query_idx = task_idx // num_heads
                head_idx = task_idx - query_idx * num_heads
                start = Int32(mQueryLeafRowPtr[query_idx])
                end = Int32(mQueryLeafRowPtr[query_idx + 1])
                leaf_count = end - start

                denom_partial = Float32.zero
                for leaf_offset in cutlass.range(lane16, leaf_count, Int32(16), unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    denom_partial += Float32(mP[node_idx, head_idx])
                denom = cute_utils.warp_reduce(denom_partial, lambda a, b: a + b, width=16)
                if denom < Float32(1.0e-8):
                    denom = Float32(1.0e-8)
                inv_denom = Float32(1.0) / denom
                if cutlass.const_expr(self.write_denom):
                    if lane16 == Int32(0):
                        mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)

                dim0 = lane16 * Int32(4)
                dim1 = dim0 + Int32(1)
                dim2 = dim0 + Int32(2)
                dim3 = dim0 + Int32(3)
                acc0 = Float32.zero
                acc1 = Float32.zero
                acc2 = Float32.zero
                acc3 = Float32.zero
                for ptr in cutlass.range(start, end, unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    weight = Float32(mP[node_idx, head_idx]) * inv_denom
                    acc0 += weight * Float32(mValue[value_idx, head_idx, dim0])
                    acc1 += weight * Float32(mValue[value_idx, head_idx, dim1])
                    acc2 += weight * Float32(mValue[value_idx, head_idx, dim2])
                    acc3 += weight * Float32(mValue[value_idx, head_idx, dim3])
                mReadout[query_idx, head_idx, dim0] = acc0.to(mReadout.element_type)
                mReadout[query_idx, head_idx, dim1] = acc1.to(mReadout.element_type)
                mReadout[query_idx, head_idx, dim2] = acc2.to(mReadout.element_type)
                mReadout[query_idx, head_idx, dim3] = acc3.to(mReadout.element_type)
        else:
            task_idx = block_idx * Int32(self.warps_per_cta) + warp_idx
            if task_idx < total_tasks:
                num_heads = Int32(mP.shape[1])
                head_dim_v = Int32(mValue.shape[2])
                query_idx = task_idx // num_heads
                head_idx = task_idx - query_idx * num_heads
                start = Int32(mQueryLeafRowPtr[query_idx])
                end = Int32(mQueryLeafRowPtr[query_idx + 1])
                leaf_count = end - start

                denom_partial = Float32.zero
                for leaf_offset in cutlass.range(lane, leaf_count, cute.arch.WARP_SIZE, unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    denom_partial += Float32(mP[node_idx, head_idx])
                denom = cute_utils.warp_reduce(denom_partial, lambda a, b: a + b)
                if denom < Float32(1.0e-8):
                    denom = Float32(1.0e-8)
                inv_denom = Float32(1.0) / denom
                if cutlass.const_expr(self.write_denom):
                    if lane == Int32(0):
                        mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)

                for dim_idx in cutlass.range(lane, head_dim_v, cute.arch.WARP_SIZE, unroll=2):
                    acc = Float32.zero
                    for ptr in cutlass.range(start, end, unroll=1):
                        leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                        node_idx = Int32(mLeafNodeIndex[leaf_entry])
                        value_idx = Int32(mLeafValueIndex[leaf_entry])
                        weight = Float32(mP[node_idx, head_idx]) * inv_denom
                        acc += weight * Float32(mValue[value_idx, head_idx, dim_idx])
                    mReadout[query_idx, head_idx, dim_idx] = acc.to(mReadout.element_type)


class ARHSAPackLeafValuesSm100:
    """Gather leaf value rows into a contiguous leaf-major buffer."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mLeafValueIndex: cute.Tensor,
        mValue: cute.Tensor,
        mPackedValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mLeafValueIndex,
            mValue,
            mPackedValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLeafValueIndex: cute.Tensor,
        mValue: cute.Tensor,
        mPackedValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mValue.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            elems_per_leaf = num_heads * head_dim_v
            leaf_entry = task_idx // elems_per_leaf
            rem = task_idx - leaf_entry * elems_per_leaf
            head_idx = rem // head_dim_v
            dim_idx = rem - head_idx * head_dim_v
            value_idx = Int32(mLeafValueIndex[leaf_entry])
            mPackedValue[leaf_entry, head_idx, dim_idx] = mValue[value_idx, head_idx, dim_idx]


class ARHSALeafReadoutBackwardSm100:
    """Backward for direct leaf readout: produce dP_final and dValue."""

    arch = 100

    def __init__(self, *, num_threads: int = 256, vectorize_dim4: bool = False):
        self.num_threads = num_threads
        self.vectorize_dim4 = vectorize_dim4

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafQueryIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mGradReadout,
            mGradP,
            mGradValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            leaf_entry = task_idx // num_heads
            head_idx = task_idx - leaf_entry * num_heads
            query_idx = Int32(mLeafQueryIndex[leaf_entry])
            node_idx = Int32(mLeafNodeIndex[leaf_entry])
            value_idx = Int32(mLeafValueIndex[leaf_entry])
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])

            denom = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                other_leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                other_node_idx = Int32(mLeafNodeIndex[other_leaf_entry])
                denom += Float32(mP[other_node_idx, head_idx])
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)

            grad_leaf_attn = Float32.zero
            for dim_idx in cutlass.range(head_dim_v, unroll=16):
                grad_leaf_attn += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                    mValue[value_idx, head_idx, dim_idx]
                )

            denom_grad = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                other_leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                other_node_idx = Int32(mLeafNodeIndex[other_leaf_entry])
                other_value_idx = Int32(mLeafValueIndex[other_leaf_entry])
                other_grad_attn = Float32.zero
                for dim_idx in cutlass.range(head_dim_v, unroll=16):
                    other_grad_attn += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                        mValue[other_value_idx, head_idx, dim_idx]
                    )
                denom_grad += (
                    -other_grad_attn
                    * Float32(mP[other_node_idx, head_idx])
                    / (denom * denom)
                )
            grad_mass = grad_leaf_attn / denom + denom_grad
            cute_utils.atomic_add_fp32(
                grad_mass,
                cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
            )

            attn = Float32(mP[node_idx, head_idx]) / denom
            if cutlass.const_expr(self.vectorize_dim4):
                for dim_group in cutlass.range(head_dim_v // Int32(4), unroll=4):
                    dim0 = dim_group * Int32(4)
                    dim1 = dim0 + Int32(1)
                    dim2 = dim0 + Int32(2)
                    dim3 = dim0 + Int32(3)
                    copy_utils.atomic_add_fp32x4(
                        attn * Float32(mGradReadout[query_idx, head_idx, dim0]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim1]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim2]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim3]),
                        cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                    )
            else:
                for dim_idx in cutlass.range(head_dim_v, unroll=16):
                    cute_utils.atomic_add_fp32(
                        attn * Float32(mGradReadout[query_idx, head_idx, dim_idx]),
                        cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim_idx)),
                    )


class ARHSALeafReadoutBackwardStatsSm100:
    """Precompute per-query/head readout backward reductions."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            query_idx = task_idx // num_heads
            head_idx = task_idx - query_idx * num_heads
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])

            denom = Float32.zero
            weighted_grad_sum = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                value_idx = Int32(mLeafValueIndex[leaf_entry])
                mass = Float32(mP[node_idx, head_idx])
                grad_leaf_attn = Float32.zero
                for dim_idx in cutlass.range(head_dim_v, unroll=16):
                    grad_leaf_attn += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                        mValue[value_idx, head_idx, dim_idx]
                    )
                mLeafGradAttn[leaf_entry, head_idx] = grad_leaf_attn.to(mLeafGradAttn.element_type)
                denom += mass
                weighted_grad_sum += mass * grad_leaf_attn
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)
            mWeightedGradSum[query_idx, head_idx] = weighted_grad_sum.to(mWeightedGradSum.element_type)


class ARHSALeafReadoutBackwardStatsLeafMajorSm100:
    """Leaf-major stats pass: parallelize per-leaf dot products and reduce with atomics."""

    arch = 100

    def __init__(self, *, num_threads: int = 256, accumulate_denom: bool = True):
        self.num_threads = num_threads
        self.accumulate_denom = accumulate_denom

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafQueryIndex,
            mLeafValueIndex,
            mValue,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            leaf_entry = task_idx // num_heads
            head_idx = task_idx - leaf_entry * num_heads
            query_idx = Int32(mLeafQueryIndex[leaf_entry])
            node_idx = Int32(mLeafNodeIndex[leaf_entry])
            value_idx = Int32(mLeafValueIndex[leaf_entry])
            mass = Float32(mP[node_idx, head_idx])

            grad_attn = Float32.zero
            for dim_idx in cutlass.range(head_dim_v, unroll=16):
                grad_attn += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                    mValue[value_idx, head_idx, dim_idx]
                )
            mLeafGradAttn[leaf_entry, head_idx] = grad_attn.to(mLeafGradAttn.element_type)
            if cutlass.const_expr(self.accumulate_denom):
                cute_utils.atomic_add_fp32(
                    mass,
                    cute_utils.elem_pointer(mDenom, (query_idx, head_idx)),
                )
            cute_utils.atomic_add_fp32(
                mass * grad_attn,
                cute_utils.elem_pointer(mWeightedGradSum, (query_idx, head_idx)),
            )


class ARHSALeafReadoutBackwardStatsLeafMajorPackedSm100:
    """Leaf-major stats pass over prepacked leaf value rows."""

    arch = 100

    def __init__(self, *, num_threads: int = 256, accumulate_denom: bool = True):
        self.num_threads = num_threads
        self.accumulate_denom = accumulate_denom

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mPackedValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafQueryIndex,
            mPackedValue,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mPackedValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mPackedValue.shape[2])
            leaf_entry = task_idx // num_heads
            head_idx = task_idx - leaf_entry * num_heads
            query_idx = Int32(mLeafQueryIndex[leaf_entry])
            node_idx = Int32(mLeafNodeIndex[leaf_entry])
            mass = Float32(mP[node_idx, head_idx])

            grad_attn = Float32.zero
            for dim_idx in cutlass.range(head_dim_v, unroll=16):
                grad_attn += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                    mPackedValue[leaf_entry, head_idx, dim_idx]
                )
            mLeafGradAttn[leaf_entry, head_idx] = grad_attn.to(mLeafGradAttn.element_type)
            if cutlass.const_expr(self.accumulate_denom):
                cute_utils.atomic_add_fp32(
                    mass,
                    cute_utils.elem_pointer(mDenom, (query_idx, head_idx)),
                )
            cute_utils.atomic_add_fp32(
                mass * grad_attn,
                cute_utils.elem_pointer(mWeightedGradSum, (query_idx, head_idx)),
            )


class ARHSALeafReadoutBackwardStatsQueryWarpSm100:
    """Query/head-owned warp stats pass for small leaf fanout."""

    arch = 100

    def __init__(self, *, num_threads: int = 128, head_dim_is_64: bool = False):
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32
        self.head_dim_is_64 = head_dim_is_64

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.head_dim_is_64):
            grid_x = cute.ceil_div(total_tasks, self.warps_per_cta * 2)
        else:
            grid_x = cute.ceil_div(total_tasks, self.warps_per_cta)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        if cutlass.const_expr(self.head_dim_is_64):
            half_warp = lane // Int32(16)
            lane16 = lane - half_warp * Int32(16)
            task_idx = block_idx * Int32(self.warps_per_cta * 2) + warp_idx * Int32(2) + half_warp
            if task_idx < total_tasks:
                num_heads = Int32(mP.shape[1])
                query_idx = task_idx // num_heads
                head_idx = task_idx - query_idx * num_heads
                start = Int32(mQueryLeafRowPtr[query_idx])
                end = Int32(mQueryLeafRowPtr[query_idx + 1])
                leaf_count = end - start

                denom_partial = Float32.zero
                for leaf_offset in cutlass.range(lane16, leaf_count, Int32(16), unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    denom_partial += Float32(mP[node_idx, head_idx])
                denom = cute_utils.warp_reduce(denom_partial, lambda a, b: a + b, width=16)

                dim0 = lane16 * Int32(4)
                dim1 = dim0 + Int32(1)
                dim2 = dim0 + Int32(2)
                dim3 = dim0 + Int32(3)
                d0 = Float32(mGradReadout[query_idx, head_idx, dim0])
                d1 = Float32(mGradReadout[query_idx, head_idx, dim1])
                d2 = Float32(mGradReadout[query_idx, head_idx, dim2])
                d3 = Float32(mGradReadout[query_idx, head_idx, dim3])
                weighted_grad_sum = Float32.zero

                for ptr in cutlass.range(start, end, unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    partial = d0 * Float32(mValue[value_idx, head_idx, dim0])
                    partial += d1 * Float32(mValue[value_idx, head_idx, dim1])
                    partial += d2 * Float32(mValue[value_idx, head_idx, dim2])
                    partial += d3 * Float32(mValue[value_idx, head_idx, dim3])
                    grad_attn = cute_utils.warp_reduce(partial, lambda a, b: a + b, width=16)
                    if lane16 == Int32(0):
                        node_idx = Int32(mLeafNodeIndex[leaf_entry])
                        mass = Float32(mP[node_idx, head_idx])
                        mLeafGradAttn[leaf_entry, head_idx] = grad_attn.to(mLeafGradAttn.element_type)
                        weighted_grad_sum += mass * grad_attn

                if lane16 == Int32(0):
                    if denom < Float32(1.0e-8):
                        denom = Float32(1.0e-8)
                    mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)
                    mWeightedGradSum[query_idx, head_idx] = weighted_grad_sum.to(
                        mWeightedGradSum.element_type
                    )
        else:
            task_idx = block_idx * Int32(self.warps_per_cta) + warp_idx
            if task_idx < total_tasks:
                num_heads = Int32(mP.shape[1])
                head_dim_v = Int32(mValue.shape[2])
                query_idx = task_idx // num_heads
                head_idx = task_idx - query_idx * num_heads
                start = Int32(mQueryLeafRowPtr[query_idx])
                end = Int32(mQueryLeafRowPtr[query_idx + 1])

                denom = Float32.zero
                weighted_grad_sum = Float32.zero

                for ptr in cutlass.range(start, end, unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    mass = Float32(mP[node_idx, head_idx])
                    partial = Float32.zero
                    for dim_idx in cutlass.range(lane, head_dim_v, cute.arch.WARP_SIZE, unroll=2):
                        partial += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                            mValue[value_idx, head_idx, dim_idx]
                        )
                    grad_attn = cute_utils.warp_reduce(partial, lambda a, b: a + b)
                    if lane == Int32(0):
                        mLeafGradAttn[leaf_entry, head_idx] = grad_attn.to(mLeafGradAttn.element_type)
                    denom += mass
                    weighted_grad_sum += mass * grad_attn

                if lane == Int32(0):
                    if denom < Float32(1.0e-8):
                        denom = Float32(1.0e-8)
                    mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)
                    mWeightedGradSum[query_idx, head_idx] = weighted_grad_sum.to(
                        mWeightedGradSum.element_type
                    )


class ARHSALeafReadoutBackwardStatsTensorCoreD64Sm100:
    """Experimental D=64 stats pass using one warp-level MMA per query/head.

    This pads a single query/head into a 16x8x64 MMA tile and uses only row 0 of
    the M dimension. It is useful for measuring whether tensor cores help the
    per-leaf dO dot V inner products despite the row-specific leaf set.
    """

    arch = 100

    def __init__(self, *, num_threads: int = 128):
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.warps_per_cta)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        task_idx = block_idx * Int32(self.warps_per_cta) + warp_idx
        active = task_idx < total_tasks
        num_heads = Int32(mP.shape[1])
        query_idx = Int32(0)
        head_idx = Int32(0)
        start = Int32(0)
        end = Int32(0)
        leaf_count = Int32(0)
        if active:
            num_heads = Int32(mP.shape[1])
            query_idx = task_idx // num_heads
            head_idx = task_idx - query_idx * num_heads
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            leaf_count = end - start

        smem = cutlass.utils.SmemAllocator()
        sDOAll = smem.allocate_tensor(
            mGradReadout.element_type,
            cute.tile_to_shape(
                sm80_utils.get_smem_layout_atom(mGradReadout.element_type, 64),
                (self.warps_per_cta * 16, 64),
                (0, 1),
            ),
            byte_alignment=16,
        )
        sVAll = smem.allocate_tensor(
            mValue.element_type,
            cute.tile_to_shape(
                sm80_utils.get_smem_layout_atom(mValue.element_type, 64),
                (self.warps_per_cta * 8, 64),
                (0, 1),
            ),
            byte_alignment=16,
        )
        sGradAttnAll = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.warps_per_cta, 8)),
            byte_alignment=16,
        )
        sDO = cute.local_tile(sDOAll, (16, 64), (warp_idx, 0))
        sV = cute.local_tile(sVAll, (8, 64), (warp_idx, 0))
        sGradAttn = sGradAttnAll[warp_idx, None]

        # Only C row 0 is consumed, so rows 1-15 of A do not need initialization.
        for dim_idx in cutlass.range(lane, Int32(64), cute.arch.WARP_SIZE, unroll=1):
            do_val = Float32(0.0).to(sDO.element_type)
            if active:
                do_val = mGradReadout[query_idx, head_idx, dim_idx]
            sDO[Int32(0), dim_idx] = do_val
        for elem_idx in cutlass.range(lane, Int32(8) * Int32(64), cute.arch.WARP_SIZE, unroll=1):
            leaf_slot = elem_idx // Int32(64)
            dim_idx = elem_idx - leaf_slot * Int32(64)
            v_val = Float32(0.0).to(sV.element_type)
            if active:
                if leaf_slot < leaf_count:
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_slot])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    v_val = mValue[value_idx, head_idx, dim_idx]
            sV[leaf_slot, dim_idx] = v_val
        if lane < Int32(8):
            sGradAttn[lane] = Float32(0.0)
        cute.arch.barrier()

        if active:
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(mGradReadout.element_type, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 8, 16),
            )
            thr_mma = tiled_mma.get_slice(lane)
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                mGradReadout.element_type,
            )
            smem_thr_copy_do = cute_utils.make_tiled_copy_A(smem_copy_atom, tiled_mma).get_slice(lane)
            smem_thr_copy_v = cute_utils.make_tiled_copy_B(smem_copy_atom, tiled_mma).get_slice(lane)
            tSrDO = cute_utils.mma_make_fragment_A(sDO, thr_mma)
            tSrV = cute_utils.mma_make_fragment_B(sV, thr_mma)
            tSsDO = smem_thr_copy_do.partition_S(sDO)
            tSsV = smem_thr_copy_v.partition_S(sV)
            acc_shape = thr_mma.partition_shape_C((16, 8))
            c_tile = cute.make_identity_tensor((16, 8))
            tCc = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(c_tile))
            acc = cute.make_fragment(acc_shape, Float32)
            acc.fill(0.0)
            sm80_utils.gemm(
                thr_mma,
                acc,
                tSrDO,
                tSrV,
                tSsDO,
                tSsV,
                smem_thr_copy_do,
                smem_thr_copy_v,
            )
            acc_mn = layout_utils.reshape_acc_to_mn(acc)
            for mi in cutlass.range_constexpr(cute.size(tCc.shape[0])):
                for ni in cutlass.range_constexpr(cute.size(tCc.shape[1])):
                    row_idx = tCc[mi, ni][0]
                    leaf_slot = tCc[mi, ni][1]
                    if row_idx == Int32(0) and leaf_slot < leaf_count:
                        sGradAttn[leaf_slot] = acc_mn[mi, ni]
        cute.arch.barrier()

        if active and lane == Int32(0):
            denom = Float32.zero
            weighted_grad_sum = Float32.zero
            for leaf_slot in cutlass.range(Int32(0), leaf_count, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_slot])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                mass = Float32(mP[node_idx, head_idx])
                grad_attn = Float32(sGradAttn[leaf_slot])
                denom += mass
                weighted_grad_sum += mass * grad_attn
                mLeafGradAttn[leaf_entry, head_idx] = grad_attn.to(mLeafGradAttn.element_type)
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            mDenom[query_idx, head_idx] = denom.to(mDenom.element_type)
            mWeightedGradSum[query_idx, head_idx] = weighted_grad_sum.to(mWeightedGradSum.element_type)


class ARHSALeafReadoutBackwardFusedTensorCoreD64Sm100:
    """Experimental fused D=64 readout backward with tensor-core dO dot V stats."""

    arch = 100

    def __init__(self, *, num_threads: int = 64):
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.warps_per_cta)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mGradReadout,
            mGradP,
            mGradValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        task_idx = block_idx * Int32(self.warps_per_cta) + warp_idx
        active = task_idx < total_tasks
        num_heads = Int32(mP.shape[1])
        query_idx = Int32(0)
        head_idx = Int32(0)
        start = Int32(0)
        end = Int32(0)
        leaf_count = Int32(0)
        if active:
            query_idx = task_idx // num_heads
            head_idx = task_idx - query_idx * num_heads
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            leaf_count = end - start

        smem = cutlass.utils.SmemAllocator()
        sDOAll = smem.allocate_tensor(
            mGradReadout.element_type,
            cute.tile_to_shape(
                sm80_utils.get_smem_layout_atom(mGradReadout.element_type, 64),
                (self.warps_per_cta * 16, 64),
                (0, 1),
            ),
            byte_alignment=16,
        )
        sVAll = smem.allocate_tensor(
            mValue.element_type,
            cute.tile_to_shape(
                sm80_utils.get_smem_layout_atom(mValue.element_type, 64),
                (self.warps_per_cta * 8, 64),
                (0, 1),
            ),
            byte_alignment=16,
        )
        sGradAttnAll = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.warps_per_cta, 8)),
            byte_alignment=16,
        )
        sStatsAll = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.warps_per_cta, 2)),
            byte_alignment=16,
        )
        sDO = cute.local_tile(sDOAll, (16, 64), (warp_idx, 0))
        sV = cute.local_tile(sVAll, (8, 64), (warp_idx, 0))
        sGradAttn = sGradAttnAll[warp_idx, None]
        sStats = sStatsAll[warp_idx, None]

        # Only C row 0 is consumed, so rows 1-15 of A do not need initialization.
        for dim_idx in cutlass.range(lane, Int32(64), cute.arch.WARP_SIZE, unroll=1):
            do_val = Float32(0.0).to(sDO.element_type)
            if active:
                do_val = mGradReadout[query_idx, head_idx, dim_idx]
            sDO[Int32(0), dim_idx] = do_val
        for elem_idx in cutlass.range(lane, Int32(8) * Int32(64), cute.arch.WARP_SIZE, unroll=1):
            leaf_slot = elem_idx // Int32(64)
            dim_idx = elem_idx - leaf_slot * Int32(64)
            v_val = Float32(0.0).to(sV.element_type)
            if active:
                if leaf_slot < leaf_count:
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_slot])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    v_val = mValue[value_idx, head_idx, dim_idx]
            sV[leaf_slot, dim_idx] = v_val
        if lane < Int32(8):
            sGradAttn[lane] = Float32(0.0)
        if lane < Int32(2):
            sStats[lane] = Float32(0.0)
        cute.arch.barrier()

        if active:
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(mGradReadout.element_type, Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 8, 16),
            )
            thr_mma = tiled_mma.get_slice(lane)
            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                mGradReadout.element_type,
            )
            smem_thr_copy_do = cute_utils.make_tiled_copy_A(smem_copy_atom, tiled_mma).get_slice(lane)
            smem_thr_copy_v = cute_utils.make_tiled_copy_B(smem_copy_atom, tiled_mma).get_slice(lane)
            tSrDO = cute_utils.mma_make_fragment_A(sDO, thr_mma)
            tSrV = cute_utils.mma_make_fragment_B(sV, thr_mma)
            tSsDO = smem_thr_copy_do.partition_S(sDO)
            tSsV = smem_thr_copy_v.partition_S(sV)
            acc_shape = thr_mma.partition_shape_C((16, 8))
            c_tile = cute.make_identity_tensor((16, 8))
            tCc = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(c_tile))
            acc = cute.make_fragment(acc_shape, Float32)
            acc.fill(0.0)
            sm80_utils.gemm(
                thr_mma,
                acc,
                tSrDO,
                tSrV,
                tSsDO,
                tSsV,
                smem_thr_copy_do,
                smem_thr_copy_v,
            )
            acc_mn = layout_utils.reshape_acc_to_mn(acc)
            for mi in cutlass.range_constexpr(cute.size(tCc.shape[0])):
                for ni in cutlass.range_constexpr(cute.size(tCc.shape[1])):
                    row_idx = tCc[mi, ni][0]
                    leaf_slot = tCc[mi, ni][1]
                    if row_idx == Int32(0) and leaf_slot < leaf_count:
                        sGradAttn[leaf_slot] = acc_mn[mi, ni]
        cute.arch.barrier()

        if active:
            if lane == Int32(0):
                denom = Float32.zero
                weighted_grad_sum = Float32.zero
                for leaf_slot in cutlass.range(Int32(0), leaf_count, unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_slot])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    mass = Float32(mP[node_idx, head_idx])
                    grad_attn = Float32(sGradAttn[leaf_slot])
                    denom += mass
                    weighted_grad_sum += mass * grad_attn
                if denom < Float32(1.0e-8):
                    denom = Float32(1.0e-8)
                inv_denom = Float32(1.0) / denom
                sStats[0] = inv_denom
                sStats[1] = weighted_grad_sum * inv_denom
        cute.arch.barrier()

        if active:
            inv_denom = Float32(sStats[0])
            weighted_grad_mean = Float32(sStats[1])
            if lane < leaf_count:
                leaf_entry = Int32(mQueryLeafEntryIndex[start + lane])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                grad_attn = Float32(sGradAttn[lane])
                grad_mass = (grad_attn - weighted_grad_mean) * inv_denom
                cute_utils.atomic_add_fp32(
                    grad_mass,
                    cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
                )

            leaf_lane = lane // Int32(16)
            dim_group = lane - leaf_lane * Int32(16)
            for leaf_base in cutlass.range(Int32(0), leaf_count, Int32(2), unroll=1):
                leaf_offset = leaf_base + leaf_lane
                if leaf_offset < leaf_count:
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    mass = Float32(mP[node_idx, head_idx])
                    attn = mass * inv_denom
                    dim0 = dim_group * Int32(4)
                    dim1 = dim0 + Int32(1)
                    dim2 = dim0 + Int32(2)
                    dim3 = dim0 + Int32(3)
                    copy_utils.atomic_add_fp32x4(
                        attn * Float32(mGradReadout[query_idx, head_idx, dim0]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim1]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim2]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim3]),
                        cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                    )


class ARHSALeafReadoutBackwardScatterSm100:
    """Scatter direct leaf readout gradients using cached query/head stats."""

    arch = 100

    def __init__(self, *, num_threads: int = 256, vectorize_dim4: bool = False):
        self.num_threads = num_threads
        self.vectorize_dim4 = vectorize_dim4

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafQueryIndex,
            mLeafValueIndex,
            mValue,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            mGradP,
            mGradValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafQueryIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            leaf_entry = task_idx // num_heads
            head_idx = task_idx - leaf_entry * num_heads
            query_idx = Int32(mLeafQueryIndex[leaf_entry])
            node_idx = Int32(mLeafNodeIndex[leaf_entry])
            value_idx = Int32(mLeafValueIndex[leaf_entry])
            denom = Float32(mDenom[query_idx, head_idx])
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            weighted_grad_sum = Float32(mWeightedGradSum[query_idx, head_idx])
            grad_leaf_attn = Float32(mLeafGradAttn[leaf_entry, head_idx])
            grad_mass = (grad_leaf_attn - weighted_grad_sum / denom) / denom
            cute_utils.atomic_add_fp32(
                grad_mass,
                cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
            )

            attn = Float32(mP[node_idx, head_idx]) / denom
            if cutlass.const_expr(self.vectorize_dim4):
                for dim_group in cutlass.range(head_dim_v // Int32(4), unroll=4):
                    dim0 = dim_group * Int32(4)
                    dim1 = dim0 + Int32(1)
                    dim2 = dim0 + Int32(2)
                    dim3 = dim0 + Int32(3)
                    copy_utils.atomic_add_fp32x4(
                        attn * Float32(mGradReadout[query_idx, head_idx, dim0]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim1]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim2]),
                        attn * Float32(mGradReadout[query_idx, head_idx, dim3]),
                        cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                    )
            else:
                for dim_idx in cutlass.range(head_dim_v, unroll=16):
                    cute_utils.atomic_add_fp32(
                        attn * Float32(mGradReadout[query_idx, head_idx, dim_idx]),
                        cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim_idx)),
                    )


class ARHSALeafReadoutBackwardScatterQueryWarpSm100:
    """Query/head-owned scatter pass that parallelizes value-dim atomics across a warp."""

    arch = 100

    def __init__(
        self,
        *,
        num_threads: int = 128,
        vectorize_dim4: bool = False,
        head_dim_is_64: bool = False,
    ):
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32
        self.vectorize_dim4 = vectorize_dim4
        self.head_dim_is_64 = head_dim_is_64

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.warps_per_cta)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mGradReadout,
            mLeafGradAttn,
            mDenom,
            mWeightedGradSum,
            mGradP,
            mGradValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mGradReadout: cute.Tensor,
        mLeafGradAttn: cute.Tensor,
        mDenom: cute.Tensor,
        mWeightedGradSum: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        task_idx = block_idx * Int32(self.warps_per_cta) + warp_idx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mGradReadout.shape[2])
            query_idx = task_idx // num_heads
            head_idx = task_idx - query_idx * num_heads
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            leaf_count = end - start
            denom = Float32(mDenom[query_idx, head_idx])
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            weighted_grad_sum = Float32(mWeightedGradSum[query_idx, head_idx])

            if cutlass.const_expr(self.head_dim_is_64):
                leaf_lane = lane // Int32(16)
                dim_group = lane - leaf_lane * Int32(16)
                for leaf_base in cutlass.range(Int32(0), leaf_count, Int32(2), unroll=1):
                    leaf_offset = leaf_base + leaf_lane
                    if leaf_offset < leaf_count:
                        leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                        node_idx = Int32(mLeafNodeIndex[leaf_entry])
                        value_idx = Int32(mLeafValueIndex[leaf_entry])
                        mass = Float32(mP[node_idx, head_idx])
                        if dim_group == Int32(0):
                            grad_leaf_attn = Float32(mLeafGradAttn[leaf_entry, head_idx])
                            grad_mass = (grad_leaf_attn - weighted_grad_sum / denom) / denom
                            cute_utils.atomic_add_fp32(
                                grad_mass,
                                cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
                            )

                        attn = mass / denom
                        dim0 = dim_group * Int32(4)
                        dim1 = dim0 + Int32(1)
                        dim2 = dim0 + Int32(2)
                        dim3 = dim0 + Int32(3)
                        copy_utils.atomic_add_fp32x4(
                            attn * Float32(mGradReadout[query_idx, head_idx, dim0]),
                            attn * Float32(mGradReadout[query_idx, head_idx, dim1]),
                            attn * Float32(mGradReadout[query_idx, head_idx, dim2]),
                            attn * Float32(mGradReadout[query_idx, head_idx, dim3]),
                            cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                        )
            else:
                for ptr in cutlass.range(start, end, unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    mass = Float32(mP[node_idx, head_idx])
                    if lane == Int32(0):
                        grad_leaf_attn = Float32(mLeafGradAttn[leaf_entry, head_idx])
                        grad_mass = (grad_leaf_attn - weighted_grad_sum / denom) / denom
                        cute_utils.atomic_add_fp32(
                            grad_mass,
                            cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
                        )

                    attn = mass / denom
                    if cutlass.const_expr(self.vectorize_dim4):
                        for dim_group in cutlass.range(lane, head_dim_v // Int32(4), cute.arch.WARP_SIZE, unroll=1):
                            dim0 = dim_group * Int32(4)
                            dim1 = dim0 + Int32(1)
                            dim2 = dim0 + Int32(2)
                            dim3 = dim0 + Int32(3)
                            copy_utils.atomic_add_fp32x4(
                                attn * Float32(mGradReadout[query_idx, head_idx, dim0]),
                                attn * Float32(mGradReadout[query_idx, head_idx, dim1]),
                                attn * Float32(mGradReadout[query_idx, head_idx, dim2]),
                                attn * Float32(mGradReadout[query_idx, head_idx, dim3]),
                                cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                            )
                    else:
                        for dim_idx in cutlass.range(lane, head_dim_v, cute.arch.WARP_SIZE, unroll=2):
                            cute_utils.atomic_add_fp32(
                                attn * Float32(mGradReadout[query_idx, head_idx, dim_idx]),
                                cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim_idx)),
                            )


class ARHSALeafReadoutBackwardFusedQueryWarpD64Sm100:
    """Fused D=64 query/head readout backward without global stats intermediates."""

    arch = 100

    def __init__(self, *, num_threads: int = 64, denom_precomputed: bool = False):
        self.num_threads = num_threads
        self.warps_per_cta = num_threads // 32
        self.denom_precomputed = denom_precomputed

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mDenom: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.warps_per_cta * 2)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mDenom,
            mValue,
            mGradReadout,
            mGradP,
            mGradValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mDenom: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // cute.arch.WARP_SIZE
        lane = tidx % cute.arch.WARP_SIZE
        half_warp = lane // Int32(16)
        lane16 = lane - half_warp * Int32(16)
        task_idx = block_idx * Int32(self.warps_per_cta * 2) + warp_idx * Int32(2) + half_warp
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            query_idx = task_idx // num_heads
            head_idx = task_idx - query_idx * num_heads
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            leaf_count = end - start

            if cutlass.const_expr(self.denom_precomputed):
                denom = Float32(mDenom[query_idx, head_idx])
            else:
                denom_partial = Float32.zero
                for leaf_offset in cutlass.range(lane16, leaf_count, Int32(16), unroll=1):
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    denom_partial += Float32(mP[node_idx, head_idx])
                denom = cute_utils.warp_reduce(denom_partial, lambda a, b: a + b, width=16)
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)
            inv_denom = Float32(1.0) / denom

            dim0 = lane16 * Int32(4)
            dim1 = dim0 + Int32(1)
            dim2 = dim0 + Int32(2)
            dim3 = dim0 + Int32(3)
            d0 = Float32(mGradReadout[query_idx, head_idx, dim0])
            d1 = Float32(mGradReadout[query_idx, head_idx, dim1])
            d2 = Float32(mGradReadout[query_idx, head_idx, dim2])
            d3 = Float32(mGradReadout[query_idx, head_idx, dim3])

            weighted_grad_sum = Float32.zero
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                value_idx = Int32(mLeafValueIndex[leaf_entry])
                partial = d0 * Float32(mValue[value_idx, head_idx, dim0])
                partial += d1 * Float32(mValue[value_idx, head_idx, dim1])
                partial += d2 * Float32(mValue[value_idx, head_idx, dim2])
                partial += d3 * Float32(mValue[value_idx, head_idx, dim3])
                grad_attn = cute_utils.warp_reduce(partial, lambda a, b: a + b, width=16)
                if lane16 == Int32(0):
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    mass = Float32(mP[node_idx, head_idx])
                    weighted_grad_sum += mass * grad_attn

            weighted_grad_mean = weighted_grad_sum * inv_denom
            for ptr in cutlass.range(start, end, unroll=1):
                leaf_entry = Int32(mQueryLeafEntryIndex[ptr])
                node_idx = Int32(mLeafNodeIndex[leaf_entry])
                value_idx = Int32(mLeafValueIndex[leaf_entry])
                mass = Float32(mP[node_idx, head_idx])
                partial = d0 * Float32(mValue[value_idx, head_idx, dim0])
                partial += d1 * Float32(mValue[value_idx, head_idx, dim1])
                partial += d2 * Float32(mValue[value_idx, head_idx, dim2])
                partial += d3 * Float32(mValue[value_idx, head_idx, dim3])
                grad_attn = cute_utils.warp_reduce(partial, lambda a, b: a + b, width=16)
                if lane16 == Int32(0):
                    grad_mass = (grad_attn - weighted_grad_mean) * inv_denom
                    cute_utils.atomic_add_fp32(
                        grad_mass,
                        cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
                    )

                attn = mass * inv_denom
                copy_utils.atomic_add_fp32x4(
                    attn * d0,
                    attn * d1,
                    attn * d2,
                    attn * d3,
                    cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                )


class ARHSALeafReadoutBackwardFusedSmallSm100:
    """Single-kernel readout backward for small bounded query leaf fanout."""

    arch = 100

    def __init__(
        self,
        *,
        max_leaves_per_query: int,
        num_threads: int = 256,
        vectorize_dim4: bool = False,
    ):
        self.max_leaves_per_query = max_leaves_per_query
        self.num_threads = num_threads
        self.vectorize_dim4 = vectorize_dim4

    @cute.jit
    def __call__(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mP,
            mLeafNodeIndex,
            mLeafValueIndex,
            mQueryLeafRowPtr,
            mQueryLeafEntryIndex,
            mValue,
            mGradReadout,
            mGradP,
            mGradValue,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mP: cute.Tensor,
        mLeafNodeIndex: cute.Tensor,
        mLeafValueIndex: cute.Tensor,
        mQueryLeafRowPtr: cute.Tensor,
        mQueryLeafEntryIndex: cute.Tensor,
        mValue: cute.Tensor,
        mGradReadout: cute.Tensor,
        mGradP: cute.Tensor,
        mGradValue: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mP.shape[1])
            head_dim_v = Int32(mValue.shape[2])
            query_idx = task_idx // num_heads
            head_idx = task_idx - query_idx * num_heads
            start = Int32(mQueryLeafRowPtr[query_idx])
            end = Int32(mQueryLeafRowPtr[query_idx + 1])
            leaf_count = end - start

            denom = Float32.zero
            weighted_grad_sum = Float32.zero
            grad_leaf_attn = [Float32.zero for _ in range(self.max_leaves_per_query)]
            for leaf_offset in cutlass.range_constexpr(self.max_leaves_per_query):
                if leaf_offset < leaf_count:
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    mass = Float32(mP[node_idx, head_idx])
                    grad_attn = Float32.zero
                    for dim_idx in cutlass.range(head_dim_v, unroll=16):
                        grad_attn += Float32(mGradReadout[query_idx, head_idx, dim_idx]) * Float32(
                            mValue[value_idx, head_idx, dim_idx]
                        )
                    grad_leaf_attn[leaf_offset] = grad_attn
                    denom += mass
                    weighted_grad_sum += mass * grad_attn
            if denom < Float32(1.0e-8):
                denom = Float32(1.0e-8)

            for leaf_offset in cutlass.range_constexpr(self.max_leaves_per_query):
                if leaf_offset < leaf_count:
                    leaf_entry = Int32(mQueryLeafEntryIndex[start + leaf_offset])
                    node_idx = Int32(mLeafNodeIndex[leaf_entry])
                    value_idx = Int32(mLeafValueIndex[leaf_entry])
                    grad_mass = (grad_leaf_attn[leaf_offset] - weighted_grad_sum / denom) / denom
                    cute_utils.atomic_add_fp32(
                        grad_mass,
                        cute_utils.elem_pointer(mGradP, (node_idx, head_idx)),
                    )
                    attn = Float32(mP[node_idx, head_idx]) / denom
                    if cutlass.const_expr(self.vectorize_dim4):
                        for dim_group in cutlass.range(head_dim_v // Int32(4), unroll=4):
                            dim0 = dim_group * Int32(4)
                            dim1 = dim0 + Int32(1)
                            dim2 = dim0 + Int32(2)
                            dim3 = dim0 + Int32(3)
                            copy_utils.atomic_add_fp32x4(
                                attn * Float32(mGradReadout[query_idx, head_idx, dim0]),
                                attn * Float32(mGradReadout[query_idx, head_idx, dim1]),
                                attn * Float32(mGradReadout[query_idx, head_idx, dim2]),
                                attn * Float32(mGradReadout[query_idx, head_idx, dim3]),
                                cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim0)),
                            )
                    else:
                        for dim_idx in cutlass.range(head_dim_v, unroll=16):
                            cute_utils.atomic_add_fp32(
                                attn * Float32(mGradReadout[query_idx, head_idx, dim_idx]),
                                cute_utils.elem_pointer(mGradValue, (value_idx, head_idx, dim_idx)),
                            )


class ARHSAMarkovBackwardStepSm100:
    """One reverse Markov step using source-side outgoing-edge reduction."""

    arch = 100

    def __init__(self, *, num_threads: int = 64):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mGradPNext: cute.Tensor,
        mPPrev: cute.Tensor,
        mEdgeProb: cute.Tensor,
        mSrcRowPtr: cute.Tensor,
        mSrcEdgeIndex: cute.Tensor,
        mDst: cute.Tensor,
        mNodeIsSink: cute.Tensor,
        mGradEdgeProb: cute.Tensor,
        mGradPPrev: cute.Tensor,
        total_tasks: Int32,
        stream: cuda.CUstream,
    ):
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mGradPNext,
            mPPrev,
            mEdgeProb,
            mSrcRowPtr,
            mSrcEdgeIndex,
            mDst,
            mNodeIsSink,
            mGradEdgeProb,
            mGradPPrev,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mGradPNext: cute.Tensor,
        mPPrev: cute.Tensor,
        mEdgeProb: cute.Tensor,
        mSrcRowPtr: cute.Tensor,
        mSrcEdgeIndex: cute.Tensor,
        mDst: cute.Tensor,
        mNodeIsSink: cute.Tensor,
        mGradEdgeProb: cute.Tensor,
        mGradPPrev: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mGradPNext.shape[1])
            src_idx = task_idx // num_heads
            head_idx = task_idx - src_idx * num_heads
            acc = Float32.zero
            if mNodeIsSink[src_idx]:
                acc = Float32(mGradPNext[src_idx, head_idx])
            start = Int32(mSrcRowPtr[src_idx])
            end = Int32(mSrcRowPtr[src_idx + 1])
            p_prev = Float32(mPPrev[src_idx, head_idx])
            for ptr in cutlass.range(start, end, unroll=1):
                edge_idx = Int32(mSrcEdgeIndex[ptr])
                dst_idx = Int32(mDst[edge_idx])
                grad_next = Float32(mGradPNext[dst_idx, head_idx])
                acc += grad_next * Float32(mEdgeProb[edge_idx, head_idx])
                mGradEdgeProb[edge_idx, head_idx] = (
                    Float32(mGradEdgeProb[edge_idx, head_idx]) + p_prev * grad_next
                ).to(mGradEdgeProb.element_type)
            mGradPPrev[src_idx, head_idx] = acc.to(mGradPPrev.element_type)


def run_arhsa_markov_incoming_step(
    p: torch.Tensor,
    edge_prob: torch.Tensor,
    src: torch.Tensor,
    dst_row_ptr: torch.Tensor,
    dst_edge_index: torch.Tensor,
    node_is_sink: torch.Tensor,
    p_next: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one Markov step using the CuTe incoming-edge reduction kernel."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if edge_prob.ndim != 2 or edge_prob.shape[1] != p.shape[1]:
        raise ValueError(
            "edge_prob must have shape [n_edges, n_heads] with matching head count, "
            f"got {tuple(edge_prob.shape)} for p={tuple(p.shape)}"
        )
    if p_next is None:
        p_next = torch.empty_like(p)
    if p_next.shape != p.shape:
        raise ValueError(f"p_next shape mismatch: {tuple(p_next.shape)} vs {tuple(p.shape)}")

    src = src.to(device=p.device, dtype=torch.int32).contiguous()
    dst_row_ptr = dst_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    dst_edge_index = dst_edge_index.to(device=p.device, dtype=torch.int32).contiguous()
    node_is_sink = node_is_sink.to(device=p.device, dtype=torch.bool).contiguous()
    edge_prob = edge_prob.contiguous()
    p = p.contiguous()

    total_tasks = int(p.shape[0] * p.shape[1])
    if total_tasks == 0:
        return p_next

    compile_key = (
        "arhsa_markov_incoming_step",
        p.dtype,
        edge_prob.dtype,
        p.shape[1],
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_markov_incoming_step.compile_cache:
        op = ARHSAMarkovIncomingStepSm100()
        run_arhsa_markov_incoming_step.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(edge_prob),
            to_cute_tensor(src, assumed_align=4),
            to_cute_tensor(dst_row_ptr, assumed_align=4),
            to_cute_tensor(dst_edge_index, assumed_align=4),
            to_cute_tensor(node_is_sink),
            to_cute_tensor(p_next),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_markov_incoming_step.compile_cache[compile_key](
        p,
        edge_prob,
        src,
        dst_row_ptr,
        dst_edge_index,
        node_is_sink,
        p_next,
        Int32(total_tasks),
        current_stream,
    )
    return p_next


def run_arhsa_outgoing_softmax(
    edge_scores: torch.Tensor,
    src_row_ptr: torch.Tensor,
    src_edge_index: torch.Tensor,
    *,
    n_nodes: int,
    edge_prob: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run grouped outgoing-edge softmax using the CuTe source-CSR kernel."""
    _require_cute_runtime()
    if edge_scores.device.type != "cuda":
        raise ValueError("edge_scores must be a CUDA tensor")
    if edge_scores.ndim != 2:
        raise ValueError(f"edge_scores must have shape [n_edges, n_heads], got {tuple(edge_scores.shape)}")
    n_nodes = int(n_nodes)
    if n_nodes < 0:
        raise ValueError("n_nodes must be >= 0")
    if edge_prob is None:
        edge_prob = torch.empty_like(edge_scores)
    if edge_prob.shape != edge_scores.shape:
        raise ValueError(f"edge_prob shape mismatch: {tuple(edge_prob.shape)} vs {tuple(edge_scores.shape)}")

    edge_scores = edge_scores.contiguous()
    edge_prob = edge_prob.contiguous()
    src_row_ptr = src_row_ptr.to(device=edge_scores.device, dtype=torch.int32).contiguous()
    src_edge_index = src_edge_index.to(device=edge_scores.device, dtype=torch.int32).contiguous()
    total_tasks = int(n_nodes * edge_scores.shape[1])
    if total_tasks == 0:
        return edge_prob

    compile_key = (
        "arhsa_outgoing_softmax",
        edge_scores.dtype,
        edge_scores.shape[1],
        torch.cuda.get_device_capability(edge_scores.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_outgoing_softmax.compile_cache:
        op = ARHSAOutgoingSoftmaxSm100()
        run_arhsa_outgoing_softmax.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(edge_scores),
            to_cute_tensor(src_row_ptr, assumed_align=4),
            to_cute_tensor(src_edge_index, assumed_align=4),
            to_cute_tensor(edge_prob),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_outgoing_softmax.compile_cache[compile_key](
        edge_scores,
        src_row_ptr,
        src_edge_index,
        edge_prob,
        Int32(total_tasks),
        current_stream,
    )
    return edge_prob


def run_arhsa_outgoing_softmax_backward(
    edge_prob: torch.Tensor,
    grad_edge_prob: torch.Tensor,
    src_row_ptr: torch.Tensor,
    src_edge_index: torch.Tensor,
    *,
    n_nodes: int,
    grad_edge_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run grouped outgoing softmax backward using the CuTe source-CSR kernel."""
    _require_cute_runtime()
    if edge_prob.device.type != "cuda":
        raise ValueError("edge_prob must be a CUDA tensor")
    if edge_prob.ndim != 2:
        raise ValueError(f"edge_prob must have shape [n_edges, n_heads], got {tuple(edge_prob.shape)}")
    if grad_edge_prob.shape != edge_prob.shape:
        raise ValueError(f"grad_edge_prob shape mismatch: {tuple(grad_edge_prob.shape)} vs {tuple(edge_prob.shape)}")
    n_nodes = int(n_nodes)
    if n_nodes < 0:
        raise ValueError("n_nodes must be >= 0")
    if grad_edge_scores is None:
        grad_edge_scores = torch.empty_like(edge_prob)
    if grad_edge_scores.shape != edge_prob.shape:
        raise ValueError(f"grad_edge_scores shape mismatch: {tuple(grad_edge_scores.shape)} vs {tuple(edge_prob.shape)}")

    edge_prob = edge_prob.contiguous()
    grad_edge_prob = grad_edge_prob.contiguous()
    grad_edge_scores = grad_edge_scores.contiguous()
    src_row_ptr = src_row_ptr.to(device=edge_prob.device, dtype=torch.int32).contiguous()
    src_edge_index = src_edge_index.to(device=edge_prob.device, dtype=torch.int32).contiguous()
    total_tasks = int(n_nodes * edge_prob.shape[1])
    if total_tasks == 0:
        return grad_edge_scores

    compile_key = (
        "arhsa_outgoing_softmax_backward",
        edge_prob.dtype,
        grad_edge_prob.dtype,
        grad_edge_scores.dtype,
        edge_prob.shape[1],
        torch.cuda.get_device_capability(edge_prob.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_outgoing_softmax_backward.compile_cache:
        op = ARHSAOutgoingSoftmaxBackwardSm100()
        run_arhsa_outgoing_softmax_backward.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(edge_prob),
            to_cute_tensor(grad_edge_prob),
            to_cute_tensor(src_row_ptr, assumed_align=4),
            to_cute_tensor(src_edge_index, assumed_align=4),
            to_cute_tensor(grad_edge_scores),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_outgoing_softmax_backward.compile_cache[compile_key](
        edge_prob,
        grad_edge_prob,
        src_row_ptr,
        src_edge_index,
        grad_edge_scores,
        Int32(total_tasks),
        current_stream,
    )
    return grad_edge_scores


def run_arhsa_leaf_readout(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
    readout: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    query_warp: bool = False,
) -> torch.Tensor:
    """Run the CuTe direct leaf normalization/readout kernel."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError(
            "value must have shape [n_values, n_heads, head_dim_v] with matching head count, "
            f"got value={tuple(value.shape)} p={tuple(p.shape)}"
        )
    n_queries = int(n_queries)
    if readout is None:
        readout = torch.empty(n_queries, value.shape[1], value.shape[2], dtype=value.dtype, device=value.device)
    if readout.shape != (n_queries, value.shape[1], value.shape[2]):
        raise ValueError(f"readout shape mismatch: got {tuple(readout.shape)}")
    if denom is not None:
        if denom.shape != (n_queries, value.shape[1]):
            raise ValueError(f"denom shape mismatch: got {tuple(denom.shape)}")
        if denom.dtype != torch.float32:
            raise ValueError("denom must be float32 when supplied")
    p = p.contiguous()
    value = value.contiguous()
    if denom is not None:
        denom = denom.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()
    total_tasks = int(n_queries * value.shape[1] * value.shape[2])
    if total_tasks == 0:
        return readout

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if query_warp:
        warp_tasks = int(n_queries * value.shape[1])
        write_denom = denom is not None
        denom_arg = denom if write_denom else readout
        compile_key = (
            "arhsa_leaf_readout_query_warp",
            p.dtype,
            value.dtype,
            value.shape[1],
            value.shape[2],
            value.shape[2] == 64,
            write_denom,
            torch.cuda.get_device_capability(p.device),
        )
        if compile_key not in run_arhsa_leaf_readout.compile_cache:
            op = ARHSALeafReadoutQueryWarpSm100(
                head_dim_is_64=value.shape[2] == 64,
                write_denom=write_denom,
            )
            run_arhsa_leaf_readout.compile_cache[compile_key] = cute.compile(
                op,
                to_cute_tensor(p),
                to_cute_tensor(leaf_node_index, assumed_align=4),
                to_cute_tensor(leaf_value_index, assumed_align=4),
                to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
                to_cute_tensor(query_leaf_entry_index, assumed_align=4),
                to_cute_tensor(value),
                to_cute_tensor(readout),
                to_cute_tensor(denom_arg),
                Int32(warp_tasks),
                current_stream,
                options="--enable-tvm-ffi",
            )
        run_arhsa_leaf_readout.compile_cache[compile_key](
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value,
            readout,
            denom_arg,
            Int32(warp_tasks),
            current_stream,
        )
        return readout

    if denom is not None:
        compile_key = (
            "arhsa_leaf_readout_with_denom",
            p.dtype,
            value.dtype,
            value.shape[1],
            value.shape[2],
            torch.cuda.get_device_capability(p.device),
        )
        if compile_key not in run_arhsa_leaf_readout.compile_cache:
            op = ARHSALeafReadoutWithDenomSm100()
            run_arhsa_leaf_readout.compile_cache[compile_key] = cute.compile(
                op,
                to_cute_tensor(p),
                to_cute_tensor(leaf_node_index, assumed_align=4),
                to_cute_tensor(leaf_value_index, assumed_align=4),
                to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
                to_cute_tensor(query_leaf_entry_index, assumed_align=4),
                to_cute_tensor(value),
                to_cute_tensor(readout),
                to_cute_tensor(denom),
                Int32(total_tasks),
                current_stream,
                options="--enable-tvm-ffi",
            )
        run_arhsa_leaf_readout.compile_cache[compile_key](
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value,
            readout,
            denom,
            Int32(total_tasks),
            current_stream,
        )
        return readout

    compile_key = (
        "arhsa_leaf_readout",
        p.dtype,
        value.dtype,
        value.shape[1],
        value.shape[2],
        torch.cuda.get_device_capability(p.device),
    )
    if compile_key not in run_arhsa_leaf_readout.compile_cache:
        op = ARHSALeafReadoutSm100()
        run_arhsa_leaf_readout.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(readout),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        readout,
        Int32(total_tasks),
        current_stream,
    )
    return readout


def run_arhsa_pack_leaf_values(
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    packed_value: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack ``value[leaf_value_index]`` into a contiguous leaf-major tensor."""
    _require_cute_runtime()
    if value.device.type != "cuda":
        raise ValueError("value must be a CUDA tensor")
    if value.ndim != 3:
        raise ValueError(f"value must have shape [n_values, n_heads, head_dim_v], got {tuple(value.shape)}")
    leaf_entries = int(leaf_value_index.numel())
    if packed_value is None:
        packed_value = torch.empty(
            leaf_entries,
            value.shape[1],
            value.shape[2],
            dtype=value.dtype,
            device=value.device,
        )
    if packed_value.shape != (leaf_entries, value.shape[1], value.shape[2]):
        raise ValueError(f"packed_value shape mismatch: got {tuple(packed_value.shape)}")
    if packed_value.dtype != value.dtype:
        raise ValueError(f"packed_value dtype mismatch: got {packed_value.dtype}, expected {value.dtype}")

    value = value.contiguous()
    packed_value = packed_value.contiguous()
    leaf_value_index = leaf_value_index.to(device=value.device, dtype=torch.int32).contiguous()
    total_tasks = int(leaf_entries * value.shape[1] * value.shape[2])
    if total_tasks == 0:
        return packed_value

    compile_key = (
        "arhsa_pack_leaf_values",
        value.dtype,
        value.shape[1],
        value.shape[2],
        torch.cuda.get_device_capability(value.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_pack_leaf_values.compile_cache:
        op = ARHSAPackLeafValuesSm100()
        run_arhsa_pack_leaf_values.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(packed_value),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_pack_leaf_values.compile_cache[compile_key](
        leaf_value_index,
        value,
        packed_value,
        Int32(total_tasks),
        current_stream,
    )
    return packed_value


def run_arhsa_leaf_readout_backward_stats(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    leaf_grad_attn: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    weighted_grad_sum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache per-query/head readout backward reductions in fp32."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError("value must have shape [n_values, n_heads, head_dim_v] with matching head count")
    if grad_readout.shape != (int(n_queries), value.shape[1], value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")

    n_queries = int(n_queries)
    stats_shape = (n_queries, p.shape[1])
    leaf_grad_shape = (leaf_node_index.numel(), p.shape[1])
    if leaf_grad_attn is None:
        leaf_grad_attn = torch.empty(leaf_grad_shape, dtype=torch.float32, device=p.device)
    if denom is None:
        denom = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if weighted_grad_sum is None:
        weighted_grad_sum = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if leaf_grad_attn.shape != leaf_grad_shape:
        raise ValueError("leaf_grad_attn must have shape [n_leaf_entries, n_heads]")
    if denom.shape != stats_shape or weighted_grad_sum.shape != stats_shape:
        raise ValueError("denom and weighted_grad_sum must have shape [n_queries, n_heads]")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return leaf_grad_attn, denom, weighted_grad_sum

    compile_key = (
        "arhsa_leaf_readout_backward_stats",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        value.shape[1],
        value.shape[2],
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_stats.compile_cache:
        op = ARHSALeafReadoutBackwardStatsSm100()
        run_arhsa_leaf_readout_backward_stats.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_stats.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        Int32(total_tasks),
        current_stream,
    )
    return leaf_grad_attn, denom, weighted_grad_sum


def run_arhsa_leaf_readout_backward_stats_leaf_major(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    leaf_grad_attn: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    weighted_grad_sum: torch.Tensor | None = None,
    denom_precomputed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leaf-major stats pass for higher parallelism before the scatter kernel."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError("value must have shape [n_values, n_heads, head_dim_v] with matching head count")
    if grad_readout.shape != (int(n_queries), value.shape[1], value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")

    n_queries = int(n_queries)
    stats_shape = (n_queries, p.shape[1])
    leaf_grad_shape = (leaf_node_index.numel(), p.shape[1])
    if leaf_grad_attn is None:
        leaf_grad_attn = torch.empty(leaf_grad_shape, dtype=torch.float32, device=p.device)
    if denom is None:
        denom = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if weighted_grad_sum is None:
        weighted_grad_sum = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if leaf_grad_attn.shape != leaf_grad_shape:
        raise ValueError("leaf_grad_attn must have shape [n_leaf_entries, n_heads]")
    if denom.shape != stats_shape or weighted_grad_sum.shape != stats_shape:
        raise ValueError("denom and weighted_grad_sum must have shape [n_queries, n_heads]")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    if not denom_precomputed:
        denom.zero_()
    weighted_grad_sum.zero_()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_query_index = leaf_query_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(leaf_node_index.numel() * p.shape[1])
    if total_tasks == 0:
        return leaf_grad_attn, denom, weighted_grad_sum

    compile_key = (
        "arhsa_leaf_readout_backward_stats_leaf_major",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        value.shape[1],
        value.shape[2],
        bool(denom_precomputed),
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_stats_leaf_major.compile_cache:
        op = ARHSALeafReadoutBackwardStatsLeafMajorSm100(accumulate_denom=not bool(denom_precomputed))
        run_arhsa_leaf_readout_backward_stats_leaf_major.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_query_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_stats_leaf_major.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        Int32(total_tasks),
        current_stream,
    )
    return leaf_grad_attn, denom, weighted_grad_sum


def run_arhsa_leaf_readout_backward_stats_leaf_major_packed(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    packed_value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    leaf_grad_attn: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    weighted_grad_sum: torch.Tensor | None = None,
    denom_precomputed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leaf-major stats pass over contiguous packed leaf value rows."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if packed_value.ndim != 3 or packed_value.shape[1] != p.shape[1]:
        raise ValueError("packed_value must have shape [n_leaf_entries, n_heads, head_dim_v]")
    if grad_readout.shape != (int(n_queries), packed_value.shape[1], packed_value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")

    n_queries = int(n_queries)
    stats_shape = (n_queries, p.shape[1])
    leaf_grad_shape = (leaf_node_index.numel(), p.shape[1])
    if packed_value.shape[0] != leaf_node_index.numel():
        raise ValueError("packed_value first dimension must match leaf_node_index")
    if leaf_grad_attn is None:
        leaf_grad_attn = torch.empty(leaf_grad_shape, dtype=torch.float32, device=p.device)
    if denom is None:
        denom = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if weighted_grad_sum is None:
        weighted_grad_sum = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if leaf_grad_attn.shape != leaf_grad_shape:
        raise ValueError("leaf_grad_attn must have shape [n_leaf_entries, n_heads]")
    if denom.shape != stats_shape or weighted_grad_sum.shape != stats_shape:
        raise ValueError("denom and weighted_grad_sum must have shape [n_queries, n_heads]")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    packed_value = packed_value.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    if not denom_precomputed:
        denom.zero_()
    weighted_grad_sum.zero_()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_query_index = leaf_query_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(leaf_node_index.numel() * p.shape[1])
    if total_tasks == 0:
        return leaf_grad_attn, denom, weighted_grad_sum

    compile_key = (
        "arhsa_leaf_readout_backward_stats_leaf_major_packed",
        p.dtype,
        packed_value.dtype,
        grad_readout.dtype,
        packed_value.shape[1],
        packed_value.shape[2],
        bool(denom_precomputed),
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_stats_leaf_major_packed.compile_cache:
        op = ARHSALeafReadoutBackwardStatsLeafMajorPackedSm100(
            accumulate_denom=not bool(denom_precomputed)
        )
        run_arhsa_leaf_readout_backward_stats_leaf_major_packed.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_query_index, assumed_align=4),
            to_cute_tensor(packed_value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_stats_leaf_major_packed.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_query_index,
        packed_value,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        Int32(total_tasks),
        current_stream,
    )
    return leaf_grad_attn, denom, weighted_grad_sum


def run_arhsa_leaf_readout_backward_stats_query_warp(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    leaf_grad_attn: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    weighted_grad_sum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query/head-owned warp stats pass for small leaf fanout."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError("value must have shape [n_values, n_heads, head_dim_v] with matching head count")
    if grad_readout.shape != (int(n_queries), value.shape[1], value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")

    n_queries = int(n_queries)
    stats_shape = (n_queries, p.shape[1])
    leaf_grad_shape = (leaf_node_index.numel(), p.shape[1])
    if leaf_grad_attn is None:
        leaf_grad_attn = torch.empty(leaf_grad_shape, dtype=torch.float32, device=p.device)
    if denom is None:
        denom = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if weighted_grad_sum is None:
        weighted_grad_sum = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if leaf_grad_attn.shape != leaf_grad_shape:
        raise ValueError("leaf_grad_attn must have shape [n_leaf_entries, n_heads]")
    if denom.shape != stats_shape or weighted_grad_sum.shape != stats_shape:
        raise ValueError("denom and weighted_grad_sum must have shape [n_queries, n_heads]")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return leaf_grad_attn, denom, weighted_grad_sum

    compile_key = (
        "arhsa_leaf_readout_backward_stats_query_warp",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        value.shape[1],
        value.shape[2],
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_stats_query_warp.compile_cache:
        op = ARHSALeafReadoutBackwardStatsQueryWarpSm100(
            head_dim_is_64=(int(value.shape[2]) == 64)
        )
        run_arhsa_leaf_readout_backward_stats_query_warp.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_stats_query_warp.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        Int32(total_tasks),
        current_stream,
    )
    return leaf_grad_attn, denom, weighted_grad_sum


def run_arhsa_leaf_readout_backward_stats_tensor_core_d64(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    leaf_grad_attn: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    weighted_grad_sum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Experimental tensor-core D=64 stats pass for <=8 leaves per query."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if value.dtype != torch.bfloat16 or grad_readout.dtype != torch.bfloat16:
        raise ValueError("tensor-core D64 stats currently requires bfloat16 value and grad_readout")
    if value.ndim != 3 or value.shape[1] != p.shape[1] or int(value.shape[2]) != 64:
        raise ValueError("tensor-core D64 stats requires value shape [n_values, n_heads, 64]")
    if grad_readout.shape != (int(n_queries), value.shape[1], 64):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")

    n_queries = int(n_queries)
    if n_queries > 0:
        max_leaves = int((query_leaf_row_ptr[1:] - query_leaf_row_ptr[:-1]).max().item())
        if max_leaves > 8:
            raise ValueError(f"tensor-core D64 stats currently supports <=8 leaves per query, got {max_leaves}")
    stats_shape = (n_queries, p.shape[1])
    leaf_grad_shape = (leaf_node_index.numel(), p.shape[1])
    if leaf_grad_attn is None:
        leaf_grad_attn = torch.empty(leaf_grad_shape, dtype=torch.float32, device=p.device)
    if denom is None:
        denom = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if weighted_grad_sum is None:
        weighted_grad_sum = torch.empty(stats_shape, dtype=torch.float32, device=p.device)
    if leaf_grad_attn.shape != leaf_grad_shape:
        raise ValueError("leaf_grad_attn must have shape [n_leaf_entries, n_heads]")
    if denom.shape != stats_shape or weighted_grad_sum.shape != stats_shape:
        raise ValueError("denom and weighted_grad_sum must have shape [n_queries, n_heads]")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return leaf_grad_attn, denom, weighted_grad_sum

    compile_key = (
        "arhsa_leaf_readout_backward_stats_tensor_core_d64",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        value.shape[1],
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_stats_tensor_core_d64.compile_cache:
        op = ARHSALeafReadoutBackwardStatsTensorCoreD64Sm100()
        run_arhsa_leaf_readout_backward_stats_tensor_core_d64.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_stats_tensor_core_d64.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        Int32(total_tasks),
        current_stream,
    )
    return leaf_grad_attn, denom, weighted_grad_sum


def run_arhsa_leaf_readout_backward_fused_tensor_core_d64(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    grad_p: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    n_queries: int,
) -> None:
    """Experimental fused tensor-core D=64 readout backward for <=8 leaves/query."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if value.dtype != torch.bfloat16 or grad_readout.dtype != torch.bfloat16:
        raise ValueError("fused tensor-core D64 backward currently requires bfloat16 value and grad_readout")
    if value.ndim != 3 or value.shape[1] != p.shape[1] or int(value.shape[2]) != 64:
        raise ValueError("fused tensor-core D64 backward requires value shape [n_values, n_heads, 64]")
    if grad_readout.shape != (int(n_queries), value.shape[1], 64):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")
    if grad_p.shape != p.shape or grad_p.dtype != torch.float32:
        raise ValueError("grad_p must be a float32 tensor with shape matching p")
    if grad_value.shape != value.shape or grad_value.dtype != torch.float32:
        raise ValueError("grad_value must be a float32 tensor with shape matching value")

    n_queries = int(n_queries)
    if n_queries > 0:
        max_leaves = int((query_leaf_row_ptr[1:] - query_leaf_row_ptr[:-1]).max().item())
        if max_leaves > 8:
            raise ValueError(f"fused tensor-core D64 backward currently supports <=8 leaves per query, got {max_leaves}")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    grad_p = grad_p.contiguous()
    grad_value = grad_value.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return

    compile_key = (
        "arhsa_leaf_readout_backward_fused_tensor_core_d64",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        value.shape[1],
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_fused_tensor_core_d64.compile_cache:
        op = ARHSALeafReadoutBackwardFusedTensorCoreD64Sm100()
        run_arhsa_leaf_readout_backward_fused_tensor_core_d64.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(grad_p),
            to_cute_tensor(grad_value),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_fused_tensor_core_d64.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        grad_readout,
        grad_p,
        grad_value,
        Int32(total_tasks),
        current_stream,
    )


def run_arhsa_leaf_readout_backward_scatter(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    leaf_grad_attn: torch.Tensor,
    denom: torch.Tensor,
    weighted_grad_sum: torch.Tensor,
    grad_p: torch.Tensor,
    grad_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter readout gradients using cached per-query/head stats."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError("value must have shape [n_values, n_heads, head_dim_v] with matching head count")
    if grad_p.shape != p.shape:
        raise ValueError(f"grad_p shape mismatch: got {tuple(grad_p.shape)}")
    if grad_value.shape != value.shape:
        raise ValueError(f"grad_value shape mismatch: got {tuple(grad_value.shape)}")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    grad_p = grad_p.contiguous()
    grad_value = grad_value.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_query_index = leaf_query_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(leaf_node_index.numel() * p.shape[1])
    if total_tasks == 0:
        return grad_p, grad_value

    compile_key = (
        "arhsa_leaf_readout_backward_scatter",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        grad_p.dtype,
        grad_value.dtype,
        value.shape[1],
        value.shape[2],
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_scatter.compile_cache:
        op = ARHSALeafReadoutBackwardScatterSm100(vectorize_dim4=(int(value.shape[2]) % 4 == 0))
        run_arhsa_leaf_readout_backward_scatter.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_query_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            to_cute_tensor(grad_p),
            to_cute_tensor(grad_value),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_scatter.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        grad_p,
        grad_value,
        Int32(total_tasks),
        current_stream,
    )
    return grad_p, grad_value


def run_arhsa_leaf_readout_backward_scatter_query_warp(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    grad_readout: torch.Tensor,
    leaf_grad_attn: torch.Tensor,
    denom: torch.Tensor,
    weighted_grad_sum: torch.Tensor,
    grad_p: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    n_queries: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter readout gradients with one warp per query/head."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if grad_readout.shape != (int(n_queries), p.shape[1], grad_value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")
    if grad_p.shape != p.shape:
        raise ValueError(f"grad_p shape mismatch: got {tuple(grad_p.shape)}")
    if leaf_grad_attn.dtype != torch.float32 or denom.dtype != torch.float32 or weighted_grad_sum.dtype != torch.float32:
        raise ValueError("leaf_grad_attn, denom, and weighted_grad_sum must be float32 tensors")

    p = p.contiguous()
    grad_readout = grad_readout.contiguous()
    leaf_grad_attn = leaf_grad_attn.contiguous()
    denom = denom.contiguous()
    weighted_grad_sum = weighted_grad_sum.contiguous()
    grad_p = grad_p.contiguous()
    grad_value = grad_value.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return grad_p, grad_value

    head_dim_v = int(grad_value.shape[2])
    compile_key = (
        "arhsa_leaf_readout_backward_scatter_query_warp",
        p.dtype,
        grad_readout.dtype,
        grad_p.dtype,
        grad_value.dtype,
        p.shape[1],
        head_dim_v,
        head_dim_v % 4 == 0,
        head_dim_v == 64,
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_scatter_query_warp.compile_cache:
        op = ARHSALeafReadoutBackwardScatterQueryWarpSm100(
            vectorize_dim4=(head_dim_v % 4 == 0),
            head_dim_is_64=(head_dim_v == 64),
        )
        run_arhsa_leaf_readout_backward_scatter_query_warp.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(grad_readout),
            to_cute_tensor(leaf_grad_attn),
            to_cute_tensor(denom),
            to_cute_tensor(weighted_grad_sum),
            to_cute_tensor(grad_p),
            to_cute_tensor(grad_value),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_scatter_query_warp.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        grad_readout,
        leaf_grad_attn,
        denom,
        weighted_grad_sum,
        grad_p,
        grad_value,
        Int32(total_tasks),
        current_stream,
    )
    return grad_p, grad_value


def run_arhsa_leaf_readout_backward_fused_query_warp_d64(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    grad_p: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    n_queries: int,
    denom: torch.Tensor | None = None,
    denom_precomputed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused query-warp readout backward for the common D=64 value path."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1] or value.shape[2] != 64:
        raise ValueError("fused query-warp readout backward currently requires value shape [n_values, n_heads, 64]")
    if grad_readout.shape != (int(n_queries), value.shape[1], 64):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")
    if grad_p.shape != p.shape:
        raise ValueError(f"grad_p shape mismatch: got {tuple(grad_p.shape)}")
    if grad_value.shape != value.shape:
        raise ValueError(f"grad_value shape mismatch: got {tuple(grad_value.shape)}")
    if denom_precomputed:
        if denom is None:
            raise ValueError("denom must be supplied when denom_precomputed=True")
        if denom.shape != (int(n_queries), value.shape[1]):
            raise ValueError(f"denom shape mismatch: got {tuple(denom.shape)}")
        if denom.dtype != torch.float32:
            raise ValueError("denom must be float32 when denom_precomputed=True")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    grad_p = grad_p.contiguous()
    grad_value = grad_value.contiguous()
    denom_arg = denom.contiguous() if denom_precomputed else grad_p
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return grad_p, grad_value

    compile_key = (
        "arhsa_leaf_readout_backward_fused_query_warp_d64",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        grad_p.dtype,
        grad_value.dtype,
        p.shape[1],
        bool(denom_precomputed),
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_fused_query_warp_d64.compile_cache:
        op = ARHSALeafReadoutBackwardFusedQueryWarpD64Sm100(denom_precomputed=bool(denom_precomputed))
        run_arhsa_leaf_readout_backward_fused_query_warp_d64.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(denom_arg),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(grad_p),
            to_cute_tensor(grad_value),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_fused_query_warp_d64.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        denom_arg,
        value,
        grad_readout,
        grad_p,
        grad_value,
        Int32(total_tasks),
        current_stream,
    )
    return grad_p, grad_value


def run_arhsa_leaf_readout_backward_fused_small(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    grad_p: torch.Tensor,
    grad_value: torch.Tensor,
    *,
    n_queries: int,
    max_leaves_per_query: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused readout backward when query leaf fanout is known and small."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError("value must have shape [n_values, n_heads, head_dim_v] with matching head count")
    if grad_readout.shape != (int(n_queries), value.shape[1], value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")
    if grad_p.shape != p.shape:
        raise ValueError(f"grad_p shape mismatch: got {tuple(grad_p.shape)}")
    if grad_value.shape != value.shape:
        raise ValueError(f"grad_value shape mismatch: got {tuple(grad_value.shape)}")
    max_leaves_per_query = int(max_leaves_per_query)
    if max_leaves_per_query <= 0:
        raise ValueError("max_leaves_per_query must be positive")
    if max_leaves_per_query > 16:
        raise ValueError("fused small readout backward currently supports at most 16 leaves/query")

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    grad_p = grad_p.contiguous()
    grad_value = grad_value.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    total_tasks = int(n_queries * p.shape[1])
    if total_tasks == 0:
        return grad_p, grad_value

    compile_key = (
        "arhsa_leaf_readout_backward_fused_small",
        p.dtype,
        value.dtype,
        grad_readout.dtype,
        grad_p.dtype,
        grad_value.dtype,
        value.shape[1],
        value.shape[2],
        max_leaves_per_query,
        torch.cuda.get_device_capability(p.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_leaf_readout_backward_fused_small.compile_cache:
        op = ARHSALeafReadoutBackwardFusedSmallSm100(
            max_leaves_per_query=max_leaves_per_query,
            vectorize_dim4=(int(value.shape[2]) % 4 == 0),
        )
        run_arhsa_leaf_readout_backward_fused_small.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(p),
            to_cute_tensor(leaf_node_index, assumed_align=4),
            to_cute_tensor(leaf_value_index, assumed_align=4),
            to_cute_tensor(query_leaf_row_ptr, assumed_align=4),
            to_cute_tensor(query_leaf_entry_index, assumed_align=4),
            to_cute_tensor(value),
            to_cute_tensor(grad_readout),
            to_cute_tensor(grad_p),
            to_cute_tensor(grad_value),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_leaf_readout_backward_fused_small.compile_cache[compile_key](
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        grad_readout,
        grad_p,
        grad_value,
        Int32(total_tasks),
        current_stream,
    )
    return grad_p, grad_value


def run_arhsa_leaf_readout_backward(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    query_leaf_row_ptr: torch.Tensor,
    query_leaf_entry_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    grad_p: torch.Tensor | None = None,
    grad_value: torch.Tensor | None = None,
    max_leaves_per_query: int | None = None,
    leaf_grad_attn: torch.Tensor | None = None,
    denom: torch.Tensor | None = None,
    weighted_grad_sum: torch.Tensor | None = None,
    grad_p_accum: torch.Tensor | None = None,
    grad_value_accum: torch.Tensor | None = None,
    leaf_major_stats: bool = False,
    denom_precomputed: bool = False,
    packed_value: torch.Tensor | None = None,
    query_warp_stats: bool = False,
    query_warp_scatter: bool = False,
    query_warp_fused: bool = False,
    tensor_core_stats: bool = False,
    tensor_core_fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the CuTe direct leaf readout backward kernel."""
    _require_cute_runtime()
    if p.device.type != "cuda":
        raise ValueError("p must be a CUDA tensor")
    if p.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"p dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {p.dtype}")
    if value.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"value dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {value.dtype}")
    if grad_readout.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"grad_readout dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {grad_readout.dtype}")
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3 or value.shape[1] != p.shape[1]:
        raise ValueError(
            "value must have shape [n_values, n_heads, head_dim_v] with matching head count, "
            f"got value={tuple(value.shape)} p={tuple(p.shape)}"
        )
    if grad_readout.shape != (int(n_queries), value.shape[1], value.shape[2]):
        raise ValueError(f"grad_readout shape mismatch: got {tuple(grad_readout.shape)}")
    grad_p_out = grad_p
    grad_value_out = grad_value
    if grad_p is not None:
        if grad_p.shape != p.shape:
            raise ValueError(f"grad_p shape mismatch: got {tuple(grad_p.shape)}")
    if grad_value is not None:
        if grad_value.shape != value.shape:
            raise ValueError(f"grad_value shape mismatch: got {tuple(grad_value.shape)}")
    if grad_p_accum is not None and grad_p_accum.shape != p.shape:
        raise ValueError(f"grad_p_accum shape mismatch: got {tuple(grad_p_accum.shape)}")
    if grad_value_accum is not None and grad_value_accum.shape != value.shape:
        raise ValueError(f"grad_value_accum shape mismatch: got {tuple(grad_value_accum.shape)}")
    if packed_value is not None:
        if not leaf_major_stats:
            raise ValueError("packed_value currently requires leaf_major_stats=True")
        if packed_value.shape != (leaf_node_index.numel(), value.shape[1], value.shape[2]):
            raise ValueError(f"packed_value shape mismatch: got {tuple(packed_value.shape)}")
        if packed_value.dtype != value.dtype:
            raise ValueError(f"packed_value dtype mismatch: got {packed_value.dtype}, expected {value.dtype}")
    if query_warp_stats and not query_warp_fused:
        if not leaf_major_stats:
            raise ValueError("query_warp_stats=True currently requires leaf_major_stats=True")
        if packed_value is not None:
            raise ValueError("query_warp_stats=True cannot be combined with packed_value")
        if denom_precomputed:
            raise ValueError("query_warp_stats=True cannot reuse a precomputed denom")
    if tensor_core_stats:
        if tensor_core_fused:
            raise ValueError("tensor_core_stats=True cannot be combined with tensor_core_fused=True")
        if not leaf_major_stats:
            raise ValueError("tensor_core_stats=True currently requires leaf_major_stats=True")
        if query_warp_fused:
            raise ValueError("tensor_core_stats=True cannot be combined with query_warp_fused=True")
        if packed_value is not None:
            raise ValueError("tensor_core_stats=True cannot be combined with packed_value")
        if denom_precomputed:
            raise ValueError("tensor_core_stats=True cannot reuse a precomputed denom")
        if value.dtype != torch.bfloat16 or grad_readout.dtype != torch.bfloat16 or int(value.shape[2]) != 64:
            raise ValueError("tensor_core_stats=True currently requires bfloat16 D=64 value and grad_readout")
    if tensor_core_fused:
        if max_leaves_per_query is not None:
            raise ValueError("tensor_core_fused=True cannot be combined with max_leaves_per_query fused-small path")
        if query_warp_fused:
            raise ValueError("tensor_core_fused=True cannot be combined with query_warp_fused=True")
        if packed_value is not None:
            raise ValueError("tensor_core_fused=True cannot be combined with packed_value")
        if denom_precomputed:
            raise ValueError("tensor_core_fused=True cannot reuse a precomputed denom")
        if value.dtype != torch.bfloat16 or grad_readout.dtype != torch.bfloat16 or int(value.shape[2]) != 64:
            raise ValueError("tensor_core_fused=True currently requires bfloat16 D=64 value and grad_readout")
    if query_warp_fused:
        if int(value.shape[2]) != 64:
            raise ValueError("query_warp_fused=True currently requires head_dim_v=64")
        if packed_value is not None:
            raise ValueError("query_warp_fused=True cannot be combined with packed_value")
        if denom_precomputed and denom is None:
            raise ValueError("denom must be supplied when query_warp_fused=True and denom_precomputed=True")
    if denom_precomputed and not leaf_major_stats and not query_warp_fused:
        raise ValueError("denom_precomputed=True currently requires leaf_major_stats=True")
    if denom_precomputed and denom is None:
        raise ValueError("denom must be supplied when denom_precomputed=True")

    if p.dtype == torch.float32:
        if grad_p is None:
            grad_p = torch.zeros_like(p)
        else:
            grad_p.zero_()
    elif grad_p is not None and grad_p.dtype == torch.float32:
        grad_p.zero_()
    else:
        grad_p = grad_p_accum if grad_p_accum is not None else torch.empty_like(p, dtype=torch.float32)
        if grad_p.dtype != torch.float32:
            raise ValueError("grad_p_accum must be float32")
        grad_p.zero_()

    if value.dtype == torch.float32:
        if grad_value is None:
            grad_value = torch.zeros_like(value)
        else:
            grad_value.zero_()
    elif grad_value is not None and grad_value.dtype == torch.float32:
        grad_value.zero_()
    else:
        grad_value = (
            grad_value_accum if grad_value_accum is not None else torch.empty_like(value, dtype=torch.float32)
        )
        if grad_value.dtype != torch.float32:
            raise ValueError("grad_value_accum must be float32")
        grad_value.zero_()

    p = p.contiguous()
    value = value.contiguous()
    grad_readout = grad_readout.contiguous()
    if packed_value is not None:
        packed_value = packed_value.contiguous()
    leaf_node_index = leaf_node_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_query_index = leaf_query_index.to(device=p.device, dtype=torch.int32).contiguous()
    leaf_value_index = leaf_value_index.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_row_ptr = query_leaf_row_ptr.to(device=p.device, dtype=torch.int32).contiguous()
    query_leaf_entry_index = query_leaf_entry_index.to(device=p.device, dtype=torch.int32).contiguous()

    def finalize_outputs() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal grad_p, grad_value
        if grad_p_out is not None and grad_p_out.data_ptr() != grad_p.data_ptr():
            grad_p_out.copy_(grad_p.to(dtype=grad_p_out.dtype))
            grad_p = grad_p_out
        elif grad_p_out is None and grad_p.dtype != p.dtype:
            grad_p = grad_p.to(dtype=p.dtype)

        if grad_value_out is not None and grad_value_out.data_ptr() != grad_value.data_ptr():
            grad_value_out.copy_(grad_value.to(dtype=grad_value_out.dtype))
            grad_value = grad_value_out
        elif grad_value_out is None and grad_value.dtype != value.dtype:
            grad_value = grad_value.to(dtype=value.dtype)

        return grad_p, grad_value

    total_tasks = int(leaf_node_index.numel() * p.shape[1])
    if total_tasks == 0:
        return finalize_outputs()

    if max_leaves_per_query is not None:
        run_arhsa_leaf_readout_backward_fused_small(
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value,
            grad_readout,
            grad_p,
            grad_value,
            n_queries=int(n_queries),
            max_leaves_per_query=int(max_leaves_per_query),
        )
        return finalize_outputs()

    if tensor_core_fused:
        run_arhsa_leaf_readout_backward_fused_tensor_core_d64(
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value,
            grad_readout,
            grad_p,
            grad_value,
            n_queries=int(n_queries),
        )
        return finalize_outputs()

    if query_warp_fused:
        run_arhsa_leaf_readout_backward_fused_query_warp_d64(
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value,
            grad_readout,
            grad_p,
            grad_value,
            n_queries=int(n_queries),
            denom=denom,
            denom_precomputed=denom_precomputed,
        )
        return finalize_outputs()

    if leaf_major_stats:
        if tensor_core_stats:
            leaf_grad_attn, denom, weighted_grad_sum = run_arhsa_leaf_readout_backward_stats_tensor_core_d64(
                p,
                leaf_node_index,
                leaf_value_index,
                query_leaf_row_ptr,
                query_leaf_entry_index,
                value,
                grad_readout,
                n_queries=int(n_queries),
                leaf_grad_attn=leaf_grad_attn,
                denom=denom,
                weighted_grad_sum=weighted_grad_sum,
            )
        elif query_warp_stats:
            leaf_grad_attn, denom, weighted_grad_sum = run_arhsa_leaf_readout_backward_stats_query_warp(
                p,
                leaf_node_index,
                leaf_value_index,
                query_leaf_row_ptr,
                query_leaf_entry_index,
                value,
                grad_readout,
                n_queries=int(n_queries),
                leaf_grad_attn=leaf_grad_attn,
                denom=denom,
                weighted_grad_sum=weighted_grad_sum,
            )
        elif packed_value is not None:
            leaf_grad_attn, denom, weighted_grad_sum = run_arhsa_leaf_readout_backward_stats_leaf_major_packed(
                p,
                leaf_node_index,
                leaf_query_index,
                packed_value,
                grad_readout,
                n_queries=int(n_queries),
                leaf_grad_attn=leaf_grad_attn,
                denom=denom,
                weighted_grad_sum=weighted_grad_sum,
                denom_precomputed=denom_precomputed,
            )
        else:
            leaf_grad_attn, denom, weighted_grad_sum = run_arhsa_leaf_readout_backward_stats_leaf_major(
                p,
                leaf_node_index,
                leaf_query_index,
                leaf_value_index,
                value,
                grad_readout,
                n_queries=int(n_queries),
                leaf_grad_attn=leaf_grad_attn,
                denom=denom,
                weighted_grad_sum=weighted_grad_sum,
                denom_precomputed=denom_precomputed,
            )
    else:
        leaf_grad_attn, denom, weighted_grad_sum = run_arhsa_leaf_readout_backward_stats(
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value,
            grad_readout,
            n_queries=int(n_queries),
            leaf_grad_attn=leaf_grad_attn,
            denom=denom,
            weighted_grad_sum=weighted_grad_sum,
        )
    if query_warp_scatter:
        run_arhsa_leaf_readout_backward_scatter_query_warp(
            p,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            grad_readout,
            leaf_grad_attn,
            denom,
            weighted_grad_sum,
            grad_p,
            grad_value,
            n_queries=int(n_queries),
        )
    else:
        run_arhsa_leaf_readout_backward_scatter(
            p,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            grad_readout,
            leaf_grad_attn,
            denom,
            weighted_grad_sum,
            grad_p,
            grad_value,
        )
    return finalize_outputs()


def run_arhsa_markov_backward_step(
    grad_p_next: torch.Tensor,
    p_prev: torch.Tensor,
    edge_prob: torch.Tensor,
    src_row_ptr: torch.Tensor,
    src_edge_index: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    grad_edge_prob: torch.Tensor,
    grad_p_prev: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one reverse Markov step and accumulate ``grad_edge_prob``."""
    _require_cute_runtime()
    if grad_p_next.device.type != "cuda":
        raise ValueError("grad_p_next must be a CUDA tensor")
    if grad_p_next.ndim != 2:
        raise ValueError(f"grad_p_next must have shape [n_nodes, n_heads], got {tuple(grad_p_next.shape)}")
    if p_prev.shape != grad_p_next.shape:
        raise ValueError(f"p_prev shape mismatch: {tuple(p_prev.shape)} vs {tuple(grad_p_next.shape)}")
    if edge_prob.ndim != 2 or edge_prob.shape[1] != grad_p_next.shape[1]:
        raise ValueError("edge_prob must have shape [n_edges, n_heads] with matching head count")
    if grad_edge_prob.shape != edge_prob.shape:
        raise ValueError(f"grad_edge_prob shape mismatch: {tuple(grad_edge_prob.shape)} vs {tuple(edge_prob.shape)}")
    if grad_p_prev is None:
        grad_p_prev = torch.empty_like(grad_p_next)
    if grad_p_prev.shape != grad_p_next.shape:
        raise ValueError(f"grad_p_prev shape mismatch: {tuple(grad_p_prev.shape)} vs {tuple(grad_p_next.shape)}")

    grad_p_next = grad_p_next.contiguous()
    p_prev = p_prev.contiguous()
    edge_prob = edge_prob.contiguous()
    grad_edge_prob = grad_edge_prob.contiguous()
    src_row_ptr = src_row_ptr.to(device=grad_p_next.device, dtype=torch.int32).contiguous()
    src_edge_index = src_edge_index.to(device=grad_p_next.device, dtype=torch.int32).contiguous()
    dst = dst.to(device=grad_p_next.device, dtype=torch.int32).contiguous()
    node_is_sink = node_is_sink.to(device=grad_p_next.device, dtype=torch.bool).contiguous()
    total_tasks = int(grad_p_next.shape[0] * grad_p_next.shape[1])
    if total_tasks == 0:
        return grad_p_prev

    compile_key = (
        "arhsa_markov_backward_step",
        grad_p_next.dtype,
        p_prev.dtype,
        edge_prob.dtype,
        grad_edge_prob.dtype,
        grad_p_prev.dtype,
        grad_p_next.shape[1],
        torch.cuda.get_device_capability(grad_p_next.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_arhsa_markov_backward_step.compile_cache:
        op = ARHSAMarkovBackwardStepSm100()
        run_arhsa_markov_backward_step.compile_cache[compile_key] = cute.compile(
            op,
            to_cute_tensor(grad_p_next),
            to_cute_tensor(p_prev),
            to_cute_tensor(edge_prob),
            to_cute_tensor(src_row_ptr, assumed_align=4),
            to_cute_tensor(src_edge_index, assumed_align=4),
            to_cute_tensor(dst, assumed_align=4),
            to_cute_tensor(node_is_sink),
            to_cute_tensor(grad_edge_prob),
            to_cute_tensor(grad_p_prev),
            Int32(total_tasks),
            current_stream,
            options="--enable-tvm-ffi",
        )
    run_arhsa_markov_backward_step.compile_cache[compile_key](
        grad_p_next,
        p_prev,
        edge_prob,
        src_row_ptr,
        src_edge_index,
        dst,
        node_is_sink,
        grad_edge_prob,
        grad_p_prev,
        Int32(total_tasks),
        current_stream,
    )
    return grad_p_prev


def run_arhsa_markov_walk_fixed_iters(
    p0: torch.Tensor,
    edge_prob: torch.Tensor,
    src: torch.Tensor,
    dst_row_ptr: torch.Tensor,
    dst_edge_index: torch.Tensor,
    node_is_sink: torch.Tensor,
    *,
    n_iters: int,
    scratch_a: torch.Tensor | None = None,
    scratch_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a fixed-count exact Markov walk using the incoming-reduction step."""
    n_iters = int(n_iters)
    if n_iters < 0:
        raise ValueError("n_iters must be >= 0")
    if n_iters == 0:
        return p0
    if scratch_a is None:
        scratch_a = torch.empty_like(p0)
    if scratch_b is None:
        scratch_b = torch.empty_like(p0)
    if scratch_a.shape != p0.shape or scratch_b.shape != p0.shape:
        raise ValueError("scratch_a and scratch_b must match p0 shape")
    p = p0
    for iter_idx in range(n_iters):
        p_next = scratch_a if iter_idx % 2 == 0 else scratch_b
        run_arhsa_markov_incoming_step(
            p,
            edge_prob,
            src,
            dst_row_ptr,
            dst_edge_index,
            node_is_sink,
            p_next,
        )
        p = p_next
    return p


def _validate_readout_inputs(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if p.ndim != 2:
        raise ValueError(f"p must have shape [n_nodes, n_heads], got {tuple(p.shape)}")
    if value.ndim != 3:
        raise ValueError(f"value must have shape [n_values, n_heads, head_dim_v], got {tuple(value.shape)}")
    if value.shape[1] != p.shape[1]:
        raise ValueError(f"value head count {value.shape[1]} does not match p head count {p.shape[1]}")
    if leaf_node_index.shape != leaf_query_index.shape or leaf_node_index.shape != leaf_value_index.shape:
        raise ValueError("leaf_node_index, leaf_query_index, and leaf_value_index must have matching shapes")
    if leaf_node_index.ndim != 1:
        raise ValueError("leaf readout indices must be 1D")
    if n_queries < 0:
        raise ValueError("n_queries must be >= 0")
    device = p.device
    return (
        leaf_node_index.to(device=device, dtype=torch.long).contiguous(),
        leaf_query_index.to(device=device, dtype=torch.long).contiguous(),
        leaf_value_index.to(device=device, dtype=torch.long).contiguous(),
    )


def readout_arhsa_leaf_attention(
    p: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize leaf mass per query/head and read token values.

    This Torch implementation is kept as a diagnostic/reference path for
    numerical checks and optional ``leaf_attn`` materialization.
    """
    leaf_node_index, leaf_query_index, leaf_value_index = _validate_readout_inputs(
        p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )
    n_heads = int(p.shape[1])
    head_dim_v = int(value.shape[2])
    leaf_mass = p[leaf_node_index]
    denom = torch.zeros(n_queries, n_heads, dtype=p.dtype, device=p.device)
    if leaf_mass.numel() > 0:
        denom.index_add_(0, leaf_query_index, leaf_mass)
    leaf_attn = leaf_mass / denom[leaf_query_index].clamp(min=eps)

    readout = torch.zeros(n_queries, n_heads, head_dim_v, dtype=value.dtype, device=value.device)
    if leaf_attn.numel() > 0:
        contrib = leaf_attn.to(dtype=value.dtype).unsqueeze(-1) * value[leaf_value_index]
        readout.index_add_(0, leaf_query_index, contrib)
    return readout, leaf_attn


def torch_arhsa_markov_walk_fixed_iters(
    p0: torch.Tensor,
    edge_prob: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    *,
    n_iters: int,
) -> torch.Tensor:
    """Torch reference for fixed-iteration absorbing Markov propagation."""
    n_iters = int(n_iters)
    if n_iters < 0:
        raise ValueError("n_iters must be >= 0")
    if p0.ndim != 2:
        raise ValueError(f"p0 must have shape [n_nodes, n_heads], got {tuple(p0.shape)}")
    src = src.to(device=p0.device, dtype=torch.long).contiguous()
    dst = dst.to(device=p0.device, dtype=torch.long).contiguous()
    node_is_sink = node_is_sink.to(device=p0.device, dtype=torch.bool).contiguous()
    edge_prob = edge_prob.to(device=p0.device).contiguous()
    p = p0
    for _ in range(n_iters):
        p_next = torch.where(node_is_sink[:, None], p, torch.zeros_like(p))
        if src.numel() > 0:
            p_next.index_add_(0, dst, p[src] * edge_prob)
        p = p_next
    return p


def torch_arhsa_walk_readout_from_scores_fixed_iters(
    p0: torch.Tensor,
    edge_scores: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
    n_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Torch end-to-end reference for ``edge logits -> random walk -> direct PV``.

    The readout is reduced directly from final leaf mass and values. It does not
    materialize a dense attention matrix. Shadow folding is represented by
    adding an extra leaf entry whose ``leaf_node_index`` is the shadow node and
    whose ``leaf_value_index`` points at the query leaf value row.
    """
    if p0.ndim != 2:
        raise ValueError(f"p0 must have shape [n_nodes, n_heads], got {tuple(p0.shape)}")
    n_nodes = int(p0.shape[0])
    edge_prob = outgoing_softmax_from_scores(edge_scores, src, n_nodes=n_nodes)
    p_final = torch_arhsa_markov_walk_fixed_iters(
        p0,
        edge_prob,
        src,
        dst,
        node_is_sink,
        n_iters=n_iters,
    )
    readout, leaf_attn = readout_arhsa_leaf_attention(
        p_final,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )
    return readout, leaf_attn, p_final, edge_prob


def run_arhsa_walk_readout_fixed_iters(
    p0: torch.Tensor,
    edge_prob: torch.Tensor,
    src: torch.Tensor,
    dst_row_ptr: torch.Tensor,
    dst_edge_index: torch.Tensor,
    node_is_sink: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
    n_iters: int,
    query_leaf_row_ptr: torch.Tensor | None = None,
    query_leaf_entry_index: torch.Tensor | None = None,
    return_leaf_attn: bool = True,
    p_scratch_a: torch.Tensor | None = None,
    p_scratch_b: torch.Tensor | None = None,
    readout: torch.Tensor | None = None,
    query_warp_readout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Exact fixed-iteration ARHSA walk and leaf readout.

    Returns ``(readout, leaf_attn, p_final)``. ``leaf_attn`` is computed by the
    Torch reference only when requested; the hot forward path only needs readout.
    Edge probabilities and initial state are supplied by the caller, matching
    ARHSA's current PyTorch path.
    """
    p = run_arhsa_markov_walk_fixed_iters(
        p0,
        edge_prob,
        src,
        dst_row_ptr,
        dst_edge_index,
        node_is_sink,
        n_iters=n_iters,
        scratch_a=p_scratch_a,
        scratch_b=p_scratch_b,
    )
    if (query_leaf_row_ptr is None) != (query_leaf_entry_index is None):
        raise ValueError("query_leaf_row_ptr and query_leaf_entry_index must be supplied together")
    if query_leaf_row_ptr is None:
        query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
            leaf_query_index,
            n_queries=n_queries,
        )
    readout = run_arhsa_leaf_readout(
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        n_queries=n_queries,
        readout=readout,
        query_warp=query_warp_readout,
    )
    leaf_attn = None
    if return_leaf_attn:
        _, leaf_attn = readout_arhsa_leaf_attention(
            p,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            n_queries=n_queries,
        )
    return readout, leaf_attn, p


def run_arhsa_walk_readout_from_scores_fixed_iters(
    p0: torch.Tensor,
    edge_scores: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
    n_iters: int,
    return_leaf_attn: bool = True,
    use_cute_softmax: bool = True,
    p_scratch_a: torch.Tensor | None = None,
    p_scratch_b: torch.Tensor | None = None,
    readout: torch.Tensor | None = None,
    query_warp_readout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """
    Convenience exact forward prototype from edge logits to readout.

    Returns ``(readout, leaf_attn, p_final, edge_prob)``. The grouped outgoing
    softmax is still Torch; the repeated Markov propagation is CuTe-backed.
    """
    if p0.ndim != 2:
        raise ValueError(f"p0 must have shape [n_nodes, n_heads], got {tuple(p0.shape)}")
    n_nodes = int(p0.shape[0])
    if use_cute_softmax:
        src_row_ptr, src_edge_index = build_outgoing_edge_csr(src, n_nodes=n_nodes)
        edge_prob = run_arhsa_outgoing_softmax(
            edge_scores,
            src_row_ptr,
            src_edge_index,
            n_nodes=n_nodes,
        )
    else:
        edge_prob = outgoing_softmax_from_scores(edge_scores, src, n_nodes=n_nodes)
    dst_row_ptr, dst_edge_index = build_incoming_edge_csr(dst, n_nodes=n_nodes)
    readout, leaf_attn, p_final = run_arhsa_walk_readout_fixed_iters(
        p0,
        edge_prob,
        src,
        dst_row_ptr,
        dst_edge_index,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=n_iters,
        return_leaf_attn=return_leaf_attn,
        p_scratch_a=p_scratch_a,
        p_scratch_b=p_scratch_b,
        readout=readout,
        query_warp_readout=query_warp_readout,
    )
    return readout, leaf_attn, p_final, edge_prob


def torch_arhsa_walk_readout_from_scores_fixed_iters_backward(
    p0: torch.Tensor,
    edge_scores: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    n_iters: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Explicit Torch backward for fixed-iteration ARHSA direct-PV readout.

    Returns gradients for ``(p0, edge_scores, value)``. This is still a Torch
    implementation, but it mirrors the CUDA kernels we need next:

    - reverse direct leaf ``PV`` readout,
    - reverse the absorbing random-walk recurrence,
    - reverse grouped outgoing softmax.
    """
    if p0.ndim != 2:
        raise ValueError(f"p0 must have shape [n_nodes, n_heads], got {tuple(p0.shape)}")
    if grad_readout.ndim != 3:
        raise ValueError(
            f"grad_readout must have shape [n_queries, n_heads, head_dim_v], got {tuple(grad_readout.shape)}"
        )
    n_nodes = int(p0.shape[0])
    n_heads = int(p0.shape[1])
    src = src.to(device=p0.device, dtype=torch.long).contiguous()
    dst = dst.to(device=p0.device, dtype=torch.long).contiguous()
    node_is_sink = node_is_sink.to(device=p0.device, dtype=torch.bool).contiguous()
    leaf_node_index, leaf_query_index, leaf_value_index = _validate_readout_inputs(
        p0,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )

    edge_prob = outgoing_softmax_from_scores(edge_scores, src, n_nodes=n_nodes)
    p_history = [p0]
    p = p0
    for _ in range(int(n_iters)):
        p_next = torch.where(node_is_sink[:, None], p, torch.zeros_like(p))
        if src.numel() > 0:
            p_next.index_add_(0, dst, p[src] * edge_prob)
        p = p_next
        p_history.append(p)

    p_final = p_history[-1]
    leaf_mass = p_final[leaf_node_index]
    denom = torch.zeros(n_queries, n_heads, dtype=p0.dtype, device=p0.device)
    if leaf_mass.numel() > 0:
        denom.index_add_(0, leaf_query_index, leaf_mass)
    denom_safe = denom.clamp(min=eps)
    leaf_attn = leaf_mass / denom_safe[leaf_query_index]

    grad_readout = grad_readout.to(device=p0.device, dtype=value.dtype).contiguous()
    grad_value = torch.zeros_like(value)
    if leaf_attn.numel() > 0:
        grad_value.index_add_(
            0,
            leaf_value_index,
            leaf_attn.to(dtype=value.dtype).unsqueeze(-1) * grad_readout[leaf_query_index],
        )

    grad_leaf_attn = (grad_readout[leaf_query_index] * value[leaf_value_index]).sum(dim=-1)
    grad_leaf_attn = grad_leaf_attn.to(dtype=p0.dtype)
    grad_leaf_mass = grad_leaf_attn / denom_safe[leaf_query_index]
    denom_grad = torch.zeros_like(denom)
    if leaf_mass.numel() > 0:
        denom_grad.index_add_(
            0,
            leaf_query_index,
            -grad_leaf_attn * leaf_mass / denom_safe[leaf_query_index].square(),
        )
    denom_grad = torch.where(denom >= eps, denom_grad, torch.zeros_like(denom_grad))
    grad_leaf_mass = grad_leaf_mass + denom_grad[leaf_query_index]

    grad_p = torch.zeros_like(p_final)
    if grad_leaf_mass.numel() > 0:
        grad_p.index_add_(0, leaf_node_index, grad_leaf_mass)

    grad_edge_prob = torch.zeros_like(edge_prob)
    sink_mask = node_is_sink[:, None]
    for iter_idx in range(int(n_iters) - 1, -1, -1):
        p_prev = p_history[iter_idx]
        grad_p_next = grad_p
        grad_p_prev = torch.where(sink_mask, grad_p_next, torch.zeros_like(grad_p_next))
        if src.numel() > 0:
            grad_p_prev.index_add_(0, src, grad_p_next[dst] * edge_prob)
            grad_edge_prob = grad_edge_prob + p_prev[src] * grad_p_next[dst]
        grad_p = grad_p_prev
    grad_p0 = grad_p

    src_expanded = src[:, None].expand(-1, n_heads)
    softmax_dot = torch.zeros(n_nodes, n_heads, dtype=edge_prob.dtype, device=edge_prob.device)
    if grad_edge_prob.numel() > 0:
        softmax_dot.scatter_add_(0, src_expanded, grad_edge_prob * edge_prob)
    grad_edge_scores = edge_prob * (grad_edge_prob - softmax_dot[src])
    return grad_p0, grad_edge_scores, grad_value


def run_arhsa_walk_readout_from_scores_fixed_iters_backward(
    p0: torch.Tensor,
    edge_scores: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    n_queries: int,
    n_iters: int,
    src_row_ptr: torch.Tensor | None = None,
    src_edge_index: torch.Tensor | None = None,
    dst_row_ptr: torch.Tensor | None = None,
    dst_edge_index: torch.Tensor | None = None,
    query_leaf_row_ptr: torch.Tensor | None = None,
    query_leaf_entry_index: torch.Tensor | None = None,
    max_leaves_per_query: int | None = None,
    leaf_major_stats: bool = False,
    query_warp_stats: bool = False,
    query_warp_scatter: bool = False,
    query_warp_fused: bool = False,
    tensor_core_stats: bool = False,
    tensor_core_fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CuTe-backed backward for fixed-iteration ARHSA direct-PV readout."""
    _require_cute_runtime()
    if p0.device.type != "cuda":
        raise ValueError("p0 must be a CUDA tensor")
    if p0.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"p0 dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {p0.dtype}")
    if edge_scores.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"edge_scores dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {edge_scores.dtype}")
    if value.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"value dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {value.dtype}")
    if grad_readout.dtype not in _CUTE_BACKWARD_DTYPES:
        raise ValueError(f"grad_readout dtype must be one of {_CUTE_BACKWARD_DTYPES}, got {grad_readout.dtype}")
    if p0.ndim != 2:
        raise ValueError(f"p0 must have shape [n_nodes, n_heads], got {tuple(p0.shape)}")
    n_iters = int(n_iters)
    if n_iters < 0:
        raise ValueError("n_iters must be >= 0")
    n_nodes = int(p0.shape[0])

    if (src_row_ptr is None) != (src_edge_index is None):
        raise ValueError("src_row_ptr and src_edge_index must be supplied together")
    if (dst_row_ptr is None) != (dst_edge_index is None):
        raise ValueError("dst_row_ptr and dst_edge_index must be supplied together")
    if (query_leaf_row_ptr is None) != (query_leaf_entry_index is None):
        raise ValueError("query_leaf_row_ptr and query_leaf_entry_index must be supplied together")
    if src_row_ptr is None:
        src_row_ptr, src_edge_index = build_outgoing_edge_csr(src, n_nodes=n_nodes)
    if dst_row_ptr is None:
        dst_row_ptr, dst_edge_index = build_incoming_edge_csr(dst, n_nodes=n_nodes)
    if query_leaf_row_ptr is None:
        query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
            leaf_query_index,
            n_queries=int(n_queries),
        )
    edge_prob = run_arhsa_outgoing_softmax(
        edge_scores,
        src_row_ptr,
        src_edge_index,
        n_nodes=n_nodes,
    )

    p_history = [p0]
    for iter_idx in range(n_iters):
        p_next = torch.empty_like(p0)
        run_arhsa_markov_incoming_step(
            p_history[-1],
            edge_prob,
            src,
            dst_row_ptr,
            dst_edge_index,
            node_is_sink,
            p_next,
        )
        p_history.append(p_next)

    grad_p_final, grad_value = run_arhsa_leaf_readout_backward(
        p_history[-1],
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        grad_readout,
        n_queries=int(n_queries),
        max_leaves_per_query=max_leaves_per_query,
        leaf_major_stats=leaf_major_stats,
        query_warp_stats=query_warp_stats,
        query_warp_scatter=query_warp_scatter,
        query_warp_fused=query_warp_fused,
        tensor_core_stats=tensor_core_stats,
        tensor_core_fused=tensor_core_fused,
    )
    grad_edge_prob = torch.zeros_like(edge_prob)
    grad_next = grad_p_final
    grad_scratch = torch.empty_like(p0)
    dst = dst.to(device=p0.device, dtype=torch.int32).contiguous()
    for iter_idx in range(n_iters - 1, -1, -1):
        run_arhsa_markov_backward_step(
            grad_next,
            p_history[iter_idx],
            edge_prob,
            src_row_ptr,
            src_edge_index,
            dst,
            node_is_sink,
            grad_edge_prob,
            grad_scratch,
        )
        grad_next, grad_scratch = grad_scratch, grad_next
    grad_p0 = grad_next
    grad_edge_scores = run_arhsa_outgoing_softmax_backward(
        edge_prob,
        grad_edge_prob,
        src_row_ptr,
        src_edge_index,
        n_nodes=n_nodes,
    )
    return grad_p0, grad_edge_scores, grad_value


class _ARHSAWalkReadoutFromScoresFixedIters(torch.autograd.Function):
    """CuTe forward/backward with Torch fallback for unsupported dtypes."""

    @staticmethod
    def forward(
        ctx,
        p0: torch.Tensor,
        edge_scores: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_is_sink: torch.Tensor,
        leaf_node_index: torch.Tensor,
        leaf_query_index: torch.Tensor,
        leaf_value_index: torch.Tensor,
        value: torch.Tensor,
        src_row_ptr: torch.Tensor,
        src_edge_index: torch.Tensor,
        dst_row_ptr: torch.Tensor,
        dst_edge_index: torch.Tensor,
        query_leaf_row_ptr: torch.Tensor,
        query_leaf_entry_index: torch.Tensor,
        n_queries: int,
        n_iters: int,
        use_cute_softmax: bool,
        max_leaves_per_query: int | None,
        leaf_major_stats: bool,
        query_warp_stats: bool,
        query_warp_readout: bool,
        query_warp_scatter: bool,
        query_warp_fused: bool,
        tensor_core_stats: bool,
        tensor_core_fused: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            p0,
            edge_scores,
            src,
            dst,
            node_is_sink,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            src_row_ptr,
            src_edge_index,
            dst_row_ptr,
            dst_edge_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
        )
        ctx.n_queries = int(n_queries)
        ctx.n_iters = int(n_iters)
        ctx.max_leaves_per_query = max_leaves_per_query
        ctx.leaf_major_stats = bool(leaf_major_stats)
        ctx.query_warp_stats = bool(query_warp_stats)
        ctx.query_warp_scatter = bool(query_warp_scatter)
        ctx.query_warp_fused = bool(query_warp_fused)
        ctx.tensor_core_stats = bool(tensor_core_stats)
        ctx.tensor_core_fused = bool(tensor_core_fused)
        with torch.no_grad():
            if bool(use_cute_softmax):
                edge_prob = run_arhsa_outgoing_softmax(
                    edge_scores,
                    src_row_ptr,
                    src_edge_index,
                    n_nodes=int(p0.shape[0]),
                )
            else:
                edge_prob = outgoing_softmax_from_scores(edge_scores, src, n_nodes=int(p0.shape[0]))
            readout, _, _ = run_arhsa_walk_readout_fixed_iters(
                p0,
                edge_prob,
                src,
                dst_row_ptr,
                dst_edge_index,
                node_is_sink,
                leaf_node_index,
                leaf_query_index,
                leaf_value_index,
                value,
                n_queries=ctx.n_queries,
                n_iters=ctx.n_iters,
                query_leaf_row_ptr=query_leaf_row_ptr,
                query_leaf_entry_index=query_leaf_entry_index,
                return_leaf_attn=False,
                query_warp_readout=bool(query_warp_readout),
            )
        return readout

    @staticmethod
    def backward(ctx, grad_readout: torch.Tensor):
        (
            p0,
            edge_scores,
            src,
            dst,
            node_is_sink,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            src_row_ptr,
            src_edge_index,
            dst_row_ptr,
            dst_edge_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
        ) = ctx.saved_tensors

        if (
            p0.device.type == "cuda"
            and p0.dtype in _CUTE_BACKWARD_DTYPES
            and edge_scores.dtype in _CUTE_BACKWARD_DTYPES
            and value.dtype in _CUTE_BACKWARD_DTYPES
            and grad_readout.dtype in _CUTE_BACKWARD_DTYPES
        ):
            grad_p0, grad_edge_scores, grad_value = run_arhsa_walk_readout_from_scores_fixed_iters_backward(
                p0,
                edge_scores,
                src,
                dst,
                node_is_sink,
                leaf_node_index,
                leaf_query_index,
                leaf_value_index,
                value,
                grad_readout,
                n_queries=ctx.n_queries,
                n_iters=ctx.n_iters,
                src_row_ptr=src_row_ptr,
                src_edge_index=src_edge_index,
                dst_row_ptr=dst_row_ptr,
                dst_edge_index=dst_edge_index,
                query_leaf_row_ptr=query_leaf_row_ptr,
                query_leaf_entry_index=query_leaf_entry_index,
                max_leaves_per_query=ctx.max_leaves_per_query,
                leaf_major_stats=ctx.leaf_major_stats,
                query_warp_stats=ctx.query_warp_stats,
                query_warp_scatter=ctx.query_warp_scatter,
                query_warp_fused=ctx.query_warp_fused,
                tensor_core_stats=ctx.tensor_core_stats,
                tensor_core_fused=ctx.tensor_core_fused,
            )
        else:
            grad_p0, grad_edge_scores, grad_value = torch_arhsa_walk_readout_from_scores_fixed_iters_backward(
                p0,
                edge_scores,
                src,
                dst,
                node_is_sink,
                leaf_node_index,
                leaf_query_index,
                leaf_value_index,
                value,
                grad_readout,
                n_queries=ctx.n_queries,
                n_iters=ctx.n_iters,
            )
        if not ctx.needs_input_grad[0]:
            grad_p0 = None
        if not ctx.needs_input_grad[1]:
            grad_edge_scores = None
        if not ctx.needs_input_grad[8]:
            grad_value = None
        return (
            grad_p0,
            grad_edge_scores,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_value,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def arhsa_walk_readout_from_scores_fixed_iters_autograd(
    p0: torch.Tensor,
    edge_scores: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    node_is_sink: torch.Tensor,
    leaf_node_index: torch.Tensor,
    leaf_query_index: torch.Tensor,
    leaf_value_index: torch.Tensor,
    value: torch.Tensor,
    *,
    n_queries: int,
    n_iters: int,
    use_cute_softmax: bool = True,
    src_row_ptr: torch.Tensor | None = None,
    src_edge_index: torch.Tensor | None = None,
    dst_row_ptr: torch.Tensor | None = None,
    dst_edge_index: torch.Tensor | None = None,
    query_leaf_row_ptr: torch.Tensor | None = None,
    query_leaf_entry_index: torch.Tensor | None = None,
    max_leaves_per_query: int | None = None,
    leaf_major_stats: bool = False,
    query_warp_stats: bool = False,
    query_warp_readout: bool = False,
    query_warp_scatter: bool = False,
    query_warp_fused: bool = False,
    tensor_core_stats: bool = False,
    tensor_core_fused: bool = False,
) -> torch.Tensor:
    """
    Differentiable ARHSA readout: CuTe forward and CuTe-backed fp32/BF16 backward.

    Gradients are produced for ``p0``, ``edge_scores``, and ``value``; all
    graph/index tensors are treated as non-differentiable routing metadata.
    Supplying the CSR tensors avoids rebuilding routing metadata inside the
    autograd call.
    """
    n_nodes = int(p0.shape[0])
    if (src_row_ptr is None) != (src_edge_index is None):
        raise ValueError("src_row_ptr and src_edge_index must be supplied together")
    if (dst_row_ptr is None) != (dst_edge_index is None):
        raise ValueError("dst_row_ptr and dst_edge_index must be supplied together")
    if (query_leaf_row_ptr is None) != (query_leaf_entry_index is None):
        raise ValueError("query_leaf_row_ptr and query_leaf_entry_index must be supplied together")
    if src_row_ptr is None:
        src_row_ptr, src_edge_index = build_outgoing_edge_csr(src, n_nodes=n_nodes)
    if dst_row_ptr is None:
        dst_row_ptr, dst_edge_index = build_incoming_edge_csr(dst, n_nodes=n_nodes)
    if query_leaf_row_ptr is None:
        query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
            leaf_query_index,
            n_queries=int(n_queries),
        )
    return _ARHSAWalkReadoutFromScoresFixedIters.apply(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        src_row_ptr,
        src_edge_index,
        dst_row_ptr,
        dst_edge_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        int(n_queries),
        int(n_iters),
        bool(use_cute_softmax),
        max_leaves_per_query,
        bool(leaf_major_stats),
        bool(query_warp_stats),
        bool(query_warp_readout),
        bool(query_warp_scatter),
        bool(query_warp_fused),
        bool(tensor_core_stats),
        bool(tensor_core_fused),
    )


run_arhsa_markov_incoming_step.compile_cache = get_jit_cache("arhsa_markov_incoming_step")
run_arhsa_markov_backward_step.compile_cache = get_jit_cache("arhsa_markov_backward_step")
run_arhsa_outgoing_softmax.compile_cache = get_jit_cache("arhsa_outgoing_softmax")
run_arhsa_outgoing_softmax_backward.compile_cache = get_jit_cache("arhsa_outgoing_softmax_backward")
run_arhsa_leaf_readout.compile_cache = get_jit_cache("arhsa_leaf_readout")
run_arhsa_pack_leaf_values.compile_cache = get_jit_cache("arhsa_pack_leaf_values")
run_arhsa_leaf_readout_backward_stats.compile_cache = get_jit_cache("arhsa_leaf_readout_backward_stats")
run_arhsa_leaf_readout_backward_stats_leaf_major.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_stats_leaf_major"
)
run_arhsa_leaf_readout_backward_stats_leaf_major_packed.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_stats_leaf_major_packed"
)
run_arhsa_leaf_readout_backward_stats_query_warp.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_stats_query_warp"
)
run_arhsa_leaf_readout_backward_stats_tensor_core_d64.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_stats_tensor_core_d64"
)
run_arhsa_leaf_readout_backward_fused_tensor_core_d64.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_fused_tensor_core_d64"
)
run_arhsa_leaf_readout_backward_scatter.compile_cache = get_jit_cache("arhsa_leaf_readout_backward_scatter")
run_arhsa_leaf_readout_backward_scatter_query_warp.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_scatter_query_warp"
)
run_arhsa_leaf_readout_backward_fused_query_warp_d64.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_fused_query_warp_d64"
)
run_arhsa_leaf_readout_backward_fused_small.compile_cache = get_jit_cache(
    "arhsa_leaf_readout_backward_fused_small"
)
run_arhsa_leaf_readout_backward.compile_cache = get_jit_cache("arhsa_leaf_readout_backward")
