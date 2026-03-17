# Copyright (c) 2025, Tri Dao.


import cutlass
import cutlass.cute as cute

from quack import layout_utils
import flash_attn.cute.utils as utils


def pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx):
    """Reshape a tensor to fold qhead_per_kvhead into the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        (seqlen_q, headdim, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...)
    For LSE tensors (head_idx=1):
        (seqlen_q, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...)
    """
    head_stride = T.stride[head_idx]
    shape_packed = (
        (qhead_per_kvhead, T.shape[0]),
        *[T.shape[i] for i in range(1, head_idx)],
        nheads_kv,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_packed = (
        (head_stride, T.stride[0]),
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride * qhead_per_kvhead,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(T.iterator, cute.make_layout(shape_packed, stride=stride_packed))


def pack_gqa_layout_seqmajor(T, qhead_per_kvhead, nheads_kv, head_idx):
    """Pack GQA with the sequence submode first inside the folded row mode.

    This keeps the flattened packed row order as `(seqlen_idx, qhead_idx)`, which
    is friendlier to TMA tiling when the head grouping factor does not divide the
    Blackwell FP4 MMA M tile.
    """
    head_stride = T.stride[head_idx]
    shape_packed = (
        (T.shape[0], qhead_per_kvhead),
        *[T.shape[i] for i in range(1, head_idx)],
        nheads_kv,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_packed = (
        (T.stride[0], head_stride),
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride * qhead_per_kvhead,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(T.iterator, cute.make_layout(shape_packed, stride=stride_packed))


def unpack_gqa_layout(T, qhead_per_kvhead, head_idx):
    """Reverse of pack_gqa_layout: unfold qhead_per_kvhead from the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...) -> (seqlen_q, headdim, nheads, batch, ...)
    For LSE tensors (head_idx=1):
        ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...) -> (seqlen_q, nheads, batch, ...)
    """
    seqlen_stride = T.stride[0][1]
    head_stride = T.stride[0][0]
    shape_unpacked = (
        T.shape[0][1],
        *[T.shape[i] for i in range(1, head_idx)],
        T.shape[head_idx] * qhead_per_kvhead,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_unpacked = (
        seqlen_stride,
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(T.iterator, cute.make_layout(shape_unpacked, stride=stride_unpacked))


class PackGQA:
    def __init__(
        self,
        m_block_size: cutlass.Constexpr[int],
        head_dim_padded: cutlass.Constexpr[int],
        check_hdim_oob: cutlass.Constexpr[bool],
        qhead_per_kvhead: cutlass.Constexpr[bool],
        seqmajor_layout: cutlass.Constexpr[bool] = False,
    ):
        self.m_block_size = m_block_size
        self.head_dim_padded = head_dim_padded
        self.check_hdim_oob = check_hdim_oob
        self.qhead_per_kvhead = qhead_per_kvhead
        self.seqmajor_layout = seqmajor_layout

    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
    ):
        num_ptr_per_thread = cute.ceil_div(cute.size(cRows), threads_per_row)
        tPrPtr = cute.make_fragment(num_ptr_per_thread, cutlass.Int64)
        for i in cutlass.range_constexpr(num_ptr_per_thread):
            row = i * num_threads + cRows[tidx % threads_per_row][0]
            idx = block * self.m_block_size + row
            m_idx = idx // self.qhead_per_kvhead
            h_idx = idx - m_idx * self.qhead_per_kvhead
            packed_idx = ((m_idx, h_idx),) if cutlass.const_expr(self.seqmajor_layout) else ((h_idx, m_idx),)
            tPrPtr[i] = utils.elem_pointer(tensor, packed_idx).toint()
        return tPrPtr

    @cute.jit
    def load_tensor(
        self,
        mX: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), cols)
        sX: cute.Tensor,  # (m_block_size, cols) in the destination smem layout
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cX = cute.make_identity_tensor((self.m_block_size, mX.shape[1]))
        tXsX = gmem_thr_copy.partition_D(sX)
        tXcX = gmem_thr_copy.partition_S(cX)
        t0XcX = gmem_thr_copy.get_slice(0).partition_S(cX)
        tXpX = utils.predicate_k(tXcX, limit=mX.shape[1])
        tXcX_row = tXcX[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0]
        if cutlass.const_expr(isinstance(threads_per_row, tuple)):
            threads_per_row = threads_per_row[0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrXPtr = self.compute_ptr(mX[None, 0], tXcX_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            x_gmem_ptr = cute.make_ptr(
                mX.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0XcX[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tXcX_row[0][0]
            ):
                mX_cur = cute.make_tensor(x_gmem_ptr, (mX.shape[1],))
                elems_per_load = cute.size(tXsX.shape[0][0])
                mX_cur_copy = cute.tiled_divide(mX_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tXsX.shape[2])):
                    ki = tXcX[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        mX_cur_copy[None, ki],
                        tXsX[None, m, k],
                        pred=tXpX[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
            # We don't need to clear the destination smem tiles since we'll only consume valid rows.

    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        sQ: cute.Tensor,  # (m_block_size, head_dim_padded)
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        self.load_tensor(mQ, sQ, gmem_tiled_copy, tidx, block, seqlen)

    @cute.jit
    def store_LSE(
        self,
        mLSE: cute.Tensor,  # (qhead_per_kvhead, seqlen_q)
        tLSErLSE: cute.Tensor,  # (m_block_size, head_dim_padded)
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = thr_mma.partition_C(caccO)
        taccOcO_row = layout_utils.reshape_acc_to_mn(taccOcO)[None, 0]
        assert cute.size(tLSErLSE) == cute.size(taccOcO_row)
        threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        assert cute.size(tLSErLSE) <= threads_per_row
        num_threads = tiled_mma.size
        tPrLSEPtr = self.compute_ptr(mLSE, taccOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tLSErLSE)):
            lse_ptr_i64 = utils.shuffle_sync(
                tPrLSEPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            row = block * self.m_block_size + taccOcO_row[m][0]
            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0 and row < seqlen * self.qhead_per_kvhead:
                mLSE_copy = cute.make_tensor(lse_gmem_ptr, (1,))
                mLSE_copy[0] = tLSErLSE[m]

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads according to gmem_tiled_copy
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, "threads_per_row must divide WARP_SIZE"
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(mO[None, 0], tOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = utils.shuffle_sync(
                tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            o_gmem_ptr = cute.make_ptr(
                mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0OcO[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]
            ):
                mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tOrO.shape[0][0])
                mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
