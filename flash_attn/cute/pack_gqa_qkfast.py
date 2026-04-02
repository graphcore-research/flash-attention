# Copyright (c) 2025, Tri Dao.

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from quack import layout_utils
import flash_attn.cute.utils as utils


@dsl_user_op
def copy_gmem_to_smem_u128(gmem_ptr: cute.Pointer, smem_ptr: cute.Pointer, *, loc=None, ip=None):
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [smem_ptr_i32, gmem_ptr_i64],
        "{\n\t"
        ".reg .u32 x0, x1, x2, x3;\n\t"
        "ld.global.v4.u32 {x0, x1, x2, x3}, [$1];\n\t"
        "st.shared.v4.u32 [$0], {x0, x1, x2, x3};\n\t"
        "}\n",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx):
    stride_seqlen, stride_headdim, stride_nheads_kv, *batch_strides = T.stride
    assert cute.size(T.shape[head_idx]) == nheads_kv * qhead_per_kvhead
    shape_unpacked = ((qhead_per_kvhead, T.shape[0]), *T.shape[1:head_idx], nheads_kv, *T.shape[head_idx + 1:])
    stride_unpacked = ((stride_nheads_kv, stride_seqlen), *T.stride[1:head_idx], *T.stride[head_idx:])
    return cute.make_tensor(T.iterator, cute.make_layout(shape_unpacked, stride=stride_unpacked))


def pack_gqa_layout_seqmajor(T, qhead_per_kvhead, nheads_kv, head_idx):
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
    def compute_ptr_unpacked_bytes(
        self,
        tensor: cute.Tensor,
        kv_head_idx: cutlass.Int32,
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
            q_head_idx = kv_head_idx * self.qhead_per_kvhead + h_idx
            tPrPtr[i] = utils.elem_pointer(tensor, (m_idx, 0, q_head_idx)).toint()
        return tPrPtr

    @cute.jit
    def load_tensor(
        self,
        mX: cute.Tensor,
        sX: cute.Tensor,
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
        assert cute.arch.WARP_SIZE % threads_per_row == 0
        num_threads = gmem_tiled_copy.size
        tPrXPtr = self.compute_ptr(mX[None, 0], tXcX_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            x_ptr_i64 = utils.shuffle_sync(
                tPrXPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            x_gmem_ptr = cute.make_ptr(mX.element_type, x_ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
            if t0XcX[0, m, 0][0] < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tXcX_row[0][0]:
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

    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,
        sQ: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        self.load_tensor(mQ, sQ, gmem_tiled_copy, tidx, block, seqlen)

    @cute.jit
    def load_Q_unpacked_bytes(
        self,
        mQ: cute.Tensor,
        kv_head_idx: cutlass.Int32,
        sQ: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        del gmem_tiled_copy
        packed_cols = self.head_dim_padded // 2
        chunk_bytes = 16
        chunks_per_row = packed_cols // chunk_bytes
        rows_per_warp = cute.arch.WARP_SIZE // chunks_per_row
        row_limit = seqlen * self.qhead_per_kvhead
        row_in_group = tidx // chunks_per_row
        chunk_idx = tidx % chunks_per_row
        for row_group in cutlass.range_constexpr(self.m_block_size // rows_per_warp):
            row = row_group * rows_per_warp + row_in_group
            packed_row = block * self.m_block_size + row
            if packed_row < row_limit:
                seqlen_idx = packed_row // self.qhead_per_kvhead
                subhead_idx = packed_row - seqlen_idx * self.qhead_per_kvhead
                q_head = kv_head_idx * self.qhead_per_kvhead + subhead_idx
                byte_col = chunk_idx * chunk_bytes
                copy_gmem_to_smem_u128(
                    utils.elem_pointer(mQ, (seqlen_idx, byte_col, q_head)),
                    utils.elem_pointer(sQ, (row, byte_col)),
                )

    @cute.jit
    def store_LSE(self, mLSE: cute.Tensor, tOrLSE: cute.Tensor, tidx: cutlass.Int32, block: cutlass.Int32, seqlen: cutlass.Int32):
        qhead_idx = tidx // 4
        tidx_mod = tidx % 4
        row = tidx_mod * (cute.arch.WARP_SIZE // 4) + cute.arch.lane_idx()
        if row + block * self.m_block_size < seqlen * self.qhead_per_kvhead:
            m_idx = (row + block * self.m_block_size) // self.qhead_per_kvhead
            h_idx = (row + block * self.m_block_size) - m_idx * self.qhead_per_kvhead
            mLSE[m_idx, h_idx + qhead_idx * self.qhead_per_kvhead] = tOrLSE[row]

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,
        tOrO: cute.Tensor,
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_tiled_copy.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(mO[:, None], tOcO_row, tidx, block, threads_per_row, num_threads)
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = utils.shuffle_sync(
                tPrOPtr[m // threads_per_row], m % threads_per_row, width=threads_per_row
            )
            o_gmem_ptr = cute.make_ptr(mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16)
            if t0OcO[0, m, 0][0] < seqlen * self.qhead_per_kvhead - block * self.m_block_size - tOcO_row[0][0]:
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
