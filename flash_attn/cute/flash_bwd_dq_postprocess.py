# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

import math
from typing import Literal, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from quack import copy_utils

from flash_attn.cute import utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned


class FlashAttentionBackwardDqDensePostprocess:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        arch: Literal[100, 110],
        tile_m: int = 128,
        num_threads: int = 128,
        remap_sigmoid_2cta_rows: bool = False,
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.arch = arch
        self.tile_hdim = int(math.ceil(head_dim / 32) * 32)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.num_threads = num_threads
        self.remap_sigmoid_2cta_rows = remap_sigmoid_2cta_rows
        if remap_sigmoid_2cta_rows:
            assert tile_m == 128, "The SM100 2-CTA row remap is defined for 128-row tiles"

    def _setup_attributes(self):
        num_copy_elems = 128 // Float32.width
        threads_per_row = math.gcd(128, self.tile_hdim) // num_copy_elems
        self.gmem_tiled_copy_dQaccum = copy_utils.tiled_copy_2d(
            Float32,
            threads_per_row,
            self.num_threads,
            num_copy_elems,
        )
        self.gmem_tiled_copy_dQ = copy_utils.tiled_copy_2d(
            self.dtype,
            threads_per_row,
            self.num_threads,
            num_copy_elems,
        )

    @cute.jit
    def __call__(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        if const_expr(mdQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mdQaccum.element_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")

        mdQaccum, mdQ = [assume_tensor_aligned(t) for t in (mdQaccum, mdQ)]
        self._setup_attributes()

        grid_dim = (
            cute.ceil_div(mdQ.shape[1], self.tile_m),
            mdQ.shape[2],
            mdQ.shape[0],
        )

        self.kernel(
            mdQaccum,
            mdQ,
            scale,
            self.gmem_tiled_copy_dQaccum,
            self.gmem_tiled_copy_dQ,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mdQaccum: cute.Tensor,
        mdQ: cute.Tensor,
        scale: cutlass.Float32,
        gmem_tiled_copy_dQaccum: cute.TiledCopy,
        gmem_tiled_copy_dQ: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head_idx, batch_idx = cute.arch.block_idx()

        mdQ_cur = mdQ[batch_idx, None, head_idx, None]
        mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
        gdQaccum_flat = cute.local_tile(
            mdQaccum_cur,
            (self.tile_m * self.tile_hdim,),
            (m_block,),
        )
        if const_expr(self.remap_sigmoid_2cta_rows):
            # The sigmoid 2-CTA reducer writes rows with row-coordinate bits 1
            # and 6 exchanged. Encode the self-inverse transform in the source
            # layout so the output remains ordinary row-major [M, D].
            gdQaccum_layout = cute.make_layout(
                ((2, 2, 16, 2), self.tile_hdim),
                stride=(
                    (
                        self.tile_hdim,
                        64 * self.tile_hdim,
                        4 * self.tile_hdim,
                        2 * self.tile_hdim,
                    ),
                    1,
                ),
            )
        else:
            gdQaccum_layout = cute.make_layout(
                (self.tile_m, self.tile_hdim), stride=(self.tile_hdim, 1)
            )
        gdQaccum = cute.make_tensor(gdQaccum_flat.iterator, gdQaccum_layout)
        gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

        gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_slice(tidx)
        gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_slice(tidx)
        tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum)
        tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ)
        rdQaccum = cute.make_fragment_like(tdQgdQaccum, Float32)
        cute.copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, rdQaccum)

        rdQ = cute.make_fragment_like(tdQgdQ, self.dtype)
        rdQ.store((rdQaccum.load() * scale).to(self.dtype))

        cdQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ)
        tdQpdQ = utils.predicate_k(tdQcdQ, limit=self.head_dim)
        for rest_m in cutlass.range(cute.size(rdQ.shape[1]), unroll_full=True):
            if tdQcdQ[0, rest_m, 0][0] < mdQ.shape[1] - m_block * self.tile_m:
                cute.copy(
                    gmem_tiled_copy_dQ,
                    rdQ[None, rest_m, None],
                    tdQgdQ[None, rest_m, None],
                    pred=tdQpdQ[None, rest_m, None],
                )
