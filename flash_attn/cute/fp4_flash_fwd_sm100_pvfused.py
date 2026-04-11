"""Standalone fused NVFP4 PV forward experiment.

This module is the active home for the narrow PV recurrence rewrite. It starts
from the same dense MHA-only skeleton as the legacy Sage-inspired experiment,
but it owns the fused implementation path and compile-cache identity directly.
"""

import math
import os
from typing import Type, Tuple, Callable, Optional, Literal
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float4E2M1FN, Float8E4M3FN, Float8E8M0FNU, Int32, Int64, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
import cutlass.utils.blockscaled_layout as bs_layout
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL, T, dsl_user_op, if_generate
from cutlass._mlir.dialects import llvm

from quack import copy_utils, layout_utils

from flash_attn.cute.paged_kv import PagedKVManager
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
import flash_attn.cute.pipeline as pipeline_custom
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import SoftmaxSm100, apply_score_mod_inner
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.block_sparse_utils import (
    get_total_block_count,
    produce_block_sparse_loads_sm100,
    softmax_block_sparse_sm100,
    handle_block_sparse_empty_tile_correction_sm100,
)
from flash_attn.cute.pack_gqa import (
    PackGQA,
    copy_gmem_to_smem_u128,
    pack_gqa_layout,
    pack_gqa_layout_seqmajor,
)
from flash_attn.cute import mma_sm100_desc as sm100_desc
from flash_attn.cute import blackwell_helpers as sm100_utils
from flash_attn.cute import copy_utils as fa_copy_utils
from flash_attn.cute import fa_logging
from flash_attn.cute.named_barrier import NamedBarrierFwdSm100
from cutlass.cute import FastDivmodDivisor
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    StaticPersistentTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)


def tile_atom_to_shape_sf_mn(shape, sf_vec_size: int):
    step = tuple([2, 1] + list(range(3, cute.rank(shape) + 1)))
    return cute.tile_to_shape(
        bs_layout.BlockScaledBasicChunk(sf_vec_size, tcgen05.OperandMajorMode.MN).layout,
        shape,
        step,
    )


def tile_atom_to_shape_sfv_vt(shape, sf_vec_size: int):
    """Logical GMEM scale layout for transposed Vt = (D_v, S_k, H_k, B).

    FP4 PV consumes V as operand-B with MN-major block scaling, so the V scales
    must attach to 16-wide output-channel groups in transposed Vt rather than
    following the Q/K K-major scale atom.
    """
    return cute.tile_to_shape(
        bs_layout.BlockScaledBasicChunk(sf_vec_size, tcgen05.OperandMajorMode.MN).layout,
        shape,
        (2, 1, 3, 4),
    )


def make_public_fp4_vt_tensor(mV: cute.Tensor):
    """Interpret packed transposed `Vt` storage `(B, H, D, S)` as logical `(D, S, H, B)`."""
    return cute.make_tensor(
        mV.iterator,
        cute.make_layout(
            (mV.shape[2], mV.shape[3], mV.shape[1], mV.shape[0]),
            stride=(mV.stride[2], mV.stride[3], mV.stride[1], mV.stride[0]),
        ),
    )


def make_public_fp4_v_storage_vt_tensor(mV_storage: cute.Tensor):
    """Interpret packed transposed-byte `Vt` storage `(B, H, D, S_packed)` as `(D, S_packed, H, B)`."""
    return cute.make_tensor(
        mV_storage.iterator,
        cute.make_layout(
            (mV_storage.shape[2], mV_storage.shape[3], mV_storage.shape[1], mV_storage.shape[0]),
            stride=(mV_storage.stride[2], mV_storage.stride[3], mV_storage.stride[1], mV_storage.stride[0]),
        ),
    )


def convert_to_reduction_layout(mma_layout):
    return cute.make_layout(
        (
            (mma_layout.shape[0][1], mma_layout.shape[1]),
            (mma_layout.shape[0][0], mma_layout.shape[2]),
        ),
        stride=(
            (mma_layout.stride[0][1], mma_layout.stride[1]),
            (mma_layout.stride[0][0], mma_layout.stride[2]),
        ),
    )


def convert_to_conversion_layout(mma_layout):
    return cute.make_layout(
        (
            (mma_layout.shape[0][0], (mma_layout.shape[0][1], 2)),
            mma_layout.shape[1],
            mma_layout.shape[2] // 2,
        ),
        stride=(
            (mma_layout.stride[0][0], (mma_layout.stride[0][1], mma_layout.stride[2])),
            mma_layout.stride[1],
            mma_layout.stride[2] * 2,
        ),
    )


class SoftmaxFusedNVFP4:
    """Sage-style row-state wrapper for the narrow fused NVFP4 PV lane.

    This keeps the online softmax state (`row_max`, `row_sum`, `scores_scale`)
    together in registers/fragments even though the surrounding pipeline has not
    been fully rewritten yet. The helper is intentionally minimal: it reuses the
    proven SM100 softmax math for the row update path while giving the fused PV
    lane an explicit home for the recurrence state Sage carries in registers.
    """

    def __init__(self, softmax: SoftmaxSm100, scores_scale: cute.Tensor):
        self._softmax = softmax
        self.scale_log2 = softmax.scale_log2
        self.softmax_scale = softmax.softmax_scale
        self.row_max = softmax.row_max
        self.row_sum = softmax.row_sum
        self.scores_scale = scores_scale

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
    ):
        softmax = SoftmaxSm100.create(
            scale_log2,
            rescale_threshold=rescale_threshold,
            softmax_scale=softmax_scale,
        )
        scores_scale = cute.make_rmem_tensor(softmax.num_rows, Float32)
        return SoftmaxFusedNVFP4(softmax, scores_scale)

    def reset(self) -> None:
        self._softmax.reset()
        self.scores_scale.fill(1.0)

    @cute.jit
    def update_row_max(self, acc_S_row: cute.TensorSSA, is_first: int) -> Tuple[Float32, Float32]:
        row_max, acc_scale = self._softmax.update_row_max(acc_S_row, is_first)
        self.scores_scale[0] = Float32(1.0) if cutlass.const_expr(is_first) else acc_scale
        return row_max, acc_scale

    @cute.jit
    def scale_subtract_rowmax(self, acc_S_row: cute.Tensor, row_max: Float32):
        self._softmax.scale_subtract_rowmax(acc_S_row, row_max)

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
        self._softmax.apply_exp2_convert(
            acc_S_row,
            acc_S_row_converted,
            ex2_emu_freq=ex2_emu_freq,
            ex2_emu_res=ex2_emu_res,
            ex2_emu_start_frg=ex2_emu_start_frg,
        )

    @cute.jit
    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        self._softmax.update_row_sum(acc_S_row_exp, row_scale, is_first=is_first)


@dsl_user_op
def float_to_ue4m3_byte(x: Float32, *, loc=None, ip=None):
    packed_i16 = llvm.inline_asm(
        T.i16(),
        [Float32(x).ir_value(loc=loc, ip=ip), Float32(0.0).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 out;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 out, $2, $1;\n\t"
        "mov.b16 $0, out;\n\t"
        "}\n",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint8(
        llvm.trunc(T.i8(), packed_i16, llvm.IntegerOverflowFlags.none, loc=loc, ip=ip)
    )


@dsl_user_op
def pack_float8_to_e2m1_word(
    f0: Float32,
    f1: Float32,
    f2: Float32,
    f3: Float32,
    f4: Float32,
    f5: Float32,
    f6: Float32,
    f7: Float32,
    *,
    loc=None,
    ip=None,
):
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [
            Float32(f0).ir_value(loc=loc, ip=ip),
            Float32(f1).ir_value(loc=loc, ip=ip),
            Float32(f2).ir_value(loc=loc, ip=ip),
            Float32(f3).ir_value(loc=loc, ip=ip),
            Float32(f4).ir_value(loc=loc, ip=ip),
            Float32(f5).ir_value(loc=loc, ip=ip),
            Float32(f6).ir_value(loc=loc, ip=ip),
            Float32(f7).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b8 byte0;\n\t"
        ".reg .b8 byte1;\n\t"
        ".reg .b8 byte2;\n\t"
        ".reg .b8 byte3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n\t"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n\t"
        "}\n",
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(packed_i32)


class FP4FlashAttentionForwardSm100PVFused:

    def __init__(
        self,
        # dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        is_causal: bool = False,
        is_local: bool = False,
        is_split_kv: bool = False,
        pack_gqa: bool = False,
        q_subtile_factor: int | None = None,
        m_block_size: int = 128,
        n_block_size: int = 128,
        q_stage: cutlass.Constexpr[int] = 2,
        is_persistent: bool = True,
        score_mod: cutlass.Constexpr | None = None,
        mask_mod: cutlass.Constexpr | None = None,
        has_aux_tensors: cutlass.Constexpr = False,
        paged_kv_non_tma: bool = False,
        is_varlen_q: bool = False,
        use_2cta_instrs: bool = False,
        # FP4 QK configuration
        use_fp4_qk: bool = True,
        use_fp4_pv: bool = False,
        fp4_sf_dtype: str = "e4m3",
        fp4_sf_vec_size: int = 16,
        pv_sf_dtype: str = "e4m3",
        pv_sf_vec_size: int = 16,
        pack_gqa_local: bool = False,
        group_qheads_by_kv: bool = False,
        fp4_pv_fp32_online_rescale: bool = False,
    ):
        self.use_tma_KV = not paged_kv_non_tma
        # FP4 QK configuration
        self.use_fp4_qk = use_fp4_qk
        self.use_fp4_pv = use_fp4_pv
        assert self.use_fp4_qk and self.use_fp4_pv, "PVFused only supports the FP4 QK+PV path."
        assert fp4_sf_dtype == "e4m3" and fp4_sf_vec_size == 16, "PVFused expects NVFP4 Q/K."
        assert pv_sf_dtype == "e4m3" and pv_sf_vec_size == 16, "PVFused expects NVFP4 P/V."
        assert head_dim in (64, 128), "PVFused currently only supports head_dim in {64, 128}."
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert head_dim_v == head_dim, "PVFused currently only supports head_dim_v == head_dim."
        assert qhead_per_kvhead >= 1, "PVFused expects a valid qhead_per_kvhead."
        assert not is_local, "PVFused currently does not support local/sliding-window attention."
        assert not is_split_kv, "PVFused currently does not support SplitKV."
        assert not pack_gqa_local and not group_qheads_by_kv, "PVFused currently expects canonical dense MHA/GQA scheduling."
        assert not paged_kv_non_tma and not is_varlen_q, "PVFused currently only supports dense fixed-length inputs."
        self.fp4_pv_direct_loader = os.getenv("FLASH_ATTN_FP4_PV_DIRECT_LOADER", "0") == "1"
        self.fp4_pv_force_cta_direct = os.getenv("FLASH_ATTN_FP4_PV_FORCE_CTA_DIRECT", "0") == "1"
        self.fp4_pv_manual_direct_loader = self.fp4_pv_direct_loader and self.fp4_pv_force_cta_direct
        # Keep the legacy env var name for the optional CTA-local P amax path.
        self.fp4_pv_cta_quant = os.getenv("FLASH_ATTN_FP4_PV_ENABLE_CTA_ENCODE", "0") == "1"
        self.fp4_pv_encode_centric_requested = (
            self.fp4_pv_cta_quant and os.getenv("FLASH_ATTN_FP4_PV_ENCODE_CENTRIC", "0") == "1"
        )
        # Stage 1 keeps the fused row-group quantizer but leaves the broader CTA-local
        # encode-centric experiment opt-in only.
        self.fp4_pv_encode_centric_requested = False
        self.fp4_pv_encode_centric = False
        self.fp4_pv_debug_dump_epi = os.getenv("FLASH_ATTN_FP4_PV_DEBUG_DUMP_EPI", "0") == "1"
        self.fp4_pv_debug_uniform_p = os.getenv("FLASH_ATTN_FP4_PV_DEBUG_UNIFORM_P", "0") == "1"
        self.fp4_pv_debug_dump_pcoords = os.getenv("FLASH_ATTN_FP4_PV_DEBUG_DUMP_PCOORDS", "0") == "1"
        self.fp4_pv_corr_tile_size = int(os.getenv("FLASH_ATTN_FP4_PV_CORR_TILE_SIZE", "16"))
        self.fp4_pv_fp32_online_rescale = use_fp4_pv and fp4_pv_fp32_online_rescale
        self.fp4_sf_dtype = Float8E4M3FN if fp4_sf_dtype == "e4m3" else Float8E8M0FNU
        self.fp4_sf_vec_size = fp4_sf_vec_size
        self.pv_sf_dtype = Float8E4M3FN if pv_sf_dtype == "e4m3" else Float8E8M0FNU
        self.pv_sf_vec_size = pv_sf_vec_size
        # self.dtype = dtype
        # padding head_dim to a multiple of 64 for FP4 (K=64 per MMA instruction)
        # or 16 for BF16
        hdim_multiple_of = 64 if (use_fp4_qk or use_fp4_pv) else 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv = head_dim == head_dim_v
        self.head_dim_v_padded = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.same_hdim_kv_padded = self.head_dim_padded == self.head_dim_v_padded
        assert self.fp4_pv_corr_tile_size in (16, 32, 64), "Supported PV correction tile sizes are 16/32/64."
        assert self.head_dim_v_padded % self.fp4_pv_corr_tile_size == 0
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.q_stage = q_stage
        assert self.q_stage in [1, 2]
        self.use_2cta_instrs = use_2cta_instrs
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        if self.use_fp4_pv:
            # FP4 PV v1 uses a block-scaled P @ V MMA path that does not yet thread the
            # "last split of P is ready" barrier into the UMMA issue. Starting PV early
            # on a partial P tile is therefore unsafe and can wedge the kernel.
            # Keep the correctness-first landing on the simpler full-P handoff.
            self.split_P_arrive = 0
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, "Only SM 10.x and 11.x are supported"

        self.cta_group_size = 2 if self.use_2cta_instrs else 1
        # cta_tiler M includes only 1 CTA, the scheduler will take into account the cluster shape
        self.cta_tiler = (self.q_stage * m_block_size, n_block_size, self.head_dim_padded)
        # With 2CTA, the MMA tiler M covers both CTAs, so it's cta_group_size * m_block_size.
        # Each CTA owns m_block_size rows; the 2CTA MMA instruction spans both.
        self.mma_tiler_qk = (self.cta_group_size * m_block_size, n_block_size, self.head_dim_padded)
        self.mma_tiler_pv = (self.cta_group_size * m_block_size, self.head_dim_v_padded, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = is_local
        self.is_varlen_q = is_varlen_q
        self.use_correction_warps_for_epi = is_varlen_q
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_split_kv = is_split_kv
        self.pack_gqa = pack_gqa
        self.pack_gqa_local = pack_gqa_local
        self.pack_gqa_effective = self.pack_gqa or self.pack_gqa_local
        self.pack_gqa_seqmajor = (
            self.pack_gqa_effective and self.mma_tiler_qk[0] % self.qhead_per_kvhead != 0
        )
        self.group_qheads_by_kv = group_qheads_by_kv
        self.q_subtile_factor = q_subtile_factor
        assert not (self.is_split_kv and self.head_dim_v_padded >= 192), (
            "SplitKV is not supported for hdim >= 192"
        )
        assert not (self.pack_gqa and self.pack_gqa_local), (
            "The explicit and local packed-GQA paths are mutually exclusive"
        )
        assert not (self.group_qheads_by_kv and self.pack_gqa_effective), (
            "KV-grouped GQA scheduling is only supported on the unpacked path"
        )
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        # Does S1 need to wait for S0 to finish
        # self.s0_s1_barrier = self.head_dim_padded in [64, 96] and (not self.is_causal and not self.is_local)
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        # self.enable_ex2_emu = self.head_dim_padded <= 128 and not is_sm103
        self.enable_ex2_emu = (self.head_dim_padded <= 128 or (self.head_dim_padded == 192 and self.use_2cta_instrs and not self.is_causal and not self.is_local)) and not is_sm103
        self.s0_s1_barrier = False
        self.overlap_sO_sQ = (
            (self.head_dim_padded == 192 and self.head_dim_v_padded >= 64) or
            (self.head_dim_v_padded >= 128 and self.is_split_kv)
        )
        if self.overlap_sO_sQ:
            self.is_persistent = False

        assert self.use_tma_KV or not (self.check_hdim_oob or self.check_hdim_v_oob), (
            "Paged KV does not support irregular head dim"
        )

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.use_exact_fp4_pv_lane = self.use_fp4_pv and self.use_tma_KV and self.q_stage == 1 and not self.is_varlen_q
        self.use_exact_fp4_pv_s_ready_handoff = self.use_exact_fp4_pv_lane
        self.use_exact_fp4_pv_p_ready_handoff = self.use_exact_fp4_pv_lane
        self.use_exact_fp4_pv_legacy_stats_pipeline = not self.use_exact_fp4_pv_lane
        if self.use_exact_fp4_pv_lane:
            # The exact fused PV lane only runs q_stage=1 on the must-win row.
            # Keep only the warps that actively participate in this path and
            # compact their IDs so the CTA itself shrinks with the lane instead
            # of carrying the broader FA4 warp map as empty baggage.
            self.correction_warp_ids = (4, 5)
            self.mma_warp_id = 6
            self.epilogue_warp_ids = (7,)
            self.load_warp_ids = (8,)
            self.empty_warp_ids = ()
            self.softmax1_warp_ids = ()
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        if self.q_stage == 1:
            if not self.use_tma_KV:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids
        elif self.is_varlen_q: # fallback
            self.epilogue_warp_ids = (13, 14)

        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded

        # FP4: TMEM offsets for scale factors (placed after S/O regions)
        if self.use_fp4_qk:
            sf_atom_mn = 32
            mma_inst_tile_k = 4
            self.tmem_sfa_cols = (self.mma_tiler_qk[0] // sf_atom_mn) * mma_inst_tile_k
            self.tmem_sfk_cols = (max(self.mma_tiler_qk[1], 128) // sf_atom_mn) * mma_inst_tile_k
            self.tmem_sfa_offset = self.tmem_total
            self.tmem_sfk_offset = self.tmem_sfa_offset + self.tmem_sfa_cols
            self.tmem_total = self.tmem_sfk_offset + self.tmem_sfk_cols
        else:
            self.tmem_sfa_cols = 0
            self.tmem_sfk_cols = 0
            self.tmem_sfa_offset = 0
            self.tmem_sfk_offset = 0

        if self.use_fp4_pv:
            sf_atom_mn = 32
            mma_inst_tile_k = 4
            self.tmem_sfp_cols = (self.mma_tiler_pv[0] // sf_atom_mn) * mma_inst_tile_k
            self.tmem_sfv_cols = (max(self.mma_tiler_pv[1], 128) // sf_atom_mn) * mma_inst_tile_k
            self.tmem_sfp_offset = self.tmem_total
            self.tmem_sfv_offset = self.tmem_sfp_offset + self.tmem_sfp_cols
            self.tmem_total = self.tmem_sfv_offset + self.tmem_sfv_cols
        else:
            self.tmem_sfp_cols = 0
            self.tmem_sfv_cols = 0
            self.tmem_sfp_offset = 0
            self.tmem_sfv_offset = 0

        assert self.tmem_total <= self.tmem_alloc_cols
        p_width = 4 if self.use_fp4_pv else 16
        p_width_ratio = Float32.width // p_width
        self.tmem_s_to_p_offset = self.n_block_size - (self.n_block_size // p_width_ratio)
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        else:
            # self.num_regs_softmax = 192 if self.is_causal or self.is_local else 184
            if not self.enable_ex2_emu:
                self.num_regs_softmax = 192 if not paged_kv_non_tma else 184
            else:
                # self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
                self.num_regs_softmax = 192 if not paged_kv_non_tma else 184
            # self.num_regs_softmax = 176
            # self.num_regs_correction = 96
            # self.num_regs_correction = 64 if self.is_causal or self.is_local else 80
            if not self.enable_ex2_emu:
                self.num_regs_correction = 80 if not paged_kv_non_tma else 64
            else:
                # self.num_regs_correction = 64
                self.num_regs_correction = 80 if not paged_kv_non_tma else 64
            # self.num_regs_other = 32
            # self.num_regs_other = 64
            # self.num_regs_other = 80
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
            # self.num_regs_other = 96 if self.is_causal or self.is_local else 80
            # self.num_regs_other = 64 if self.is_causal or self.is_local else 80

        forced_softmax_regs = os.environ.get("FLASH_ATTN_FP4_FORCE_REGS_SOFTMAX")
        forced_correction_regs = os.environ.get("FLASH_ATTN_FP4_FORCE_REGS_CORRECTION")
        forced_other_regs = os.environ.get("FLASH_ATTN_FP4_FORCE_REGS_OTHER")
        if forced_softmax_regs is not None:
            self.num_regs_softmax = int(forced_softmax_regs)
        if forced_correction_regs is not None:
            self.num_regs_correction = int(forced_correction_regs)
        if forced_other_regs is not None:
            self.num_regs_other = int(forced_other_regs)

        self.buffer_align_bytes = 1024

    @cute.jit
    def q_head_idx(self, head_idx: Int32, split_idx: Int32):
        return (
            head_idx * self.qhead_per_kvhead + split_idx
            if const_expr(self.group_qheads_by_kv)
            else head_idx
        )

    @cute.jit
    def kv_head_idx(self, head_idx: Int32):
        return (
            head_idx
            if const_expr(self.group_qheads_by_kv or self.pack_gqa_effective)
            else head_idx // self.qhead_per_kvhead
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        smem_size_q = self.q_stage * self.m_block_size * self.head_dim_padded * self.q_dtype.width // 8
        smem_size_o = self.q_stage * self.m_block_size * self.head_dim_v_padded * self.o_dtype.width // 8
        smem_size_q_scale = 0
        smem_size_k_scale_per_stage = 0
        smem_size_v_scale_per_stage = 0
        if self.use_fp4_qk:
            smem_size_q_scale = (
                self.q_stage
                * self.m_block_size
                * (self.head_dim_padded // self.fp4_sf_vec_size)
                * self.fp4_sf_dtype.width
                // 8
            )
            smem_size_k_scale_per_stage = (
                self.n_block_size
                * (self.head_dim_padded // self.fp4_sf_vec_size)
                * self.fp4_sf_dtype.width
                // 8
            )
        if self.use_fp4_pv:
            smem_size_v_scale_per_stage = (
                self.n_block_size
                * (self.head_dim_v_padded // self.pv_sf_vec_size)
                * self.pv_sf_dtype.width
                // 8
            )
        smem_size_q_o = (
            smem_size_q + smem_size_o if not self.overlap_sO_sQ else max(smem_size_q, smem_size_o)
        ) + smem_size_q_scale
        smem_size_k_per_stage = self.n_block_size * self.head_dim_padded * self.k_dtype.width // 8
        smem_size_v_per_stage = self.n_block_size * self.head_dim_v_padded * self.v_dtype.width // 8
        smem_size_kv_per_stage = (
            (
                smem_size_k_per_stage
                + smem_size_v_per_stage
                + smem_size_k_scale_per_stage
                + smem_size_v_scale_per_stage
            )
            if self.use_fp4_qk
            else max(smem_size_k_per_stage, smem_size_v_per_stage)
        ) // self.cta_group_size
        kv_stage = (224 * 1024 - smem_size_q_o) // smem_size_kv_per_stage
        if (
            os.getenv("FLASH_ATTN_FP4_CAP_D128_KV_STAGE", "0") == "1"
            and self.use_fp4_qk
            and not self.use_2cta_instrs
            and self.head_dim_padded == 128
            and self.head_dim_v_padded == 128
            and kv_stage > 3
        ):
            # The first PV-era shared-forward merge hard-capped FP4 d128 at 3 KV stages.
            # The pre-PV fast path did not impose that cap, and that older schedule is the
            # performance reference we are restoring by default. Keep the cap as an internal
            # escape hatch while we revalidate the repaired non-PV path.
            kv_stage = 3
        forced_kv_stage = os.environ.get("FLASH_ATTN_FP4_FORCE_KV_STAGE")
        if forced_kv_stage is not None:
            kv_stage = min(kv_stage, int(forced_kv_stage))
        if self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and kv_stage == 2:
            # For hdim 192,128, we can fit 3 stages if we use uneven_kv_smem
             kv_stage = 3
        self.kv_stage = kv_stage
        # print("kv_stage", self.kv_stage)
        self.s_stage = 2
        assert self.s_stage >= self.q_stage
        # For hdim 192,128 1CTA, we don't have enough smem to store all 3 stages of KV:
        # 128 x 192 x 2 bytes x 3 stages = 144KB, and we need 96KB for Q.
        # Instead we store smem as [smem_large, smem_small, smem_large], where smem_large is
        # 128 x 192 and smem_small is 128 x 128. We set the stride between the stages to be
        # 128 * 160, so that indexing the 0th and 2nd stages will get the right address,
        # but for the 1st stage we need to add or subtract (depending on phase) 128 x 64.
        self.uneven_kv_smem = (
            not self.use_fp4_qk and
            self.head_dim_padded == 192 and self.head_dim_v_padded == 128 and self.kv_stage == 3
        )
        self.uneven_kv_smem_offset = (
            self.n_block_size * (self.head_dim_padded - self.head_dim_v_padded) // 2
            if self.uneven_kv_smem
            else 0
        )
        assert self.uneven_kv_smem_offset % 1024 == 0

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        mQ_storage: Optional[cute.Tensor],
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table
        mV_storage: Optional[cute.Tensor],
        mO: cute.Tensor,  # (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,  # (b_k, max_num_pages_per_seq)
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_tensors: Optional[list] = None,
        # FP4: scale factor tensors (same batch/head dims as Q/K, reduced d dim)
        mQ_scale: Optional[cute.Tensor] = None,  # (b, s_q, h, d//sf_vec) fp8
        mK_scale: Optional[cute.Tensor] = None,  # (b_k, s_k, h_k, d//sf_vec) fp8
        mV_scale: Optional[cute.Tensor] = None,  # external FP4 PV scales: (b_k, s_k, h_k, dv//sf_vec)
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.q_storage_dtype = self.q_dtype
        if const_expr(self.pack_gqa_local):
            assert mQ_storage is not None, "pack_gqa_local requires raw packed Q storage"
            self.q_storage_dtype = mQ_storage.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.p_dtype = Float4E2M1FN if const_expr(self.use_fp4_pv) else self.v_dtype
        self.v_storage_dtype = self.v_dtype
        if const_expr(self.use_fp4_pv):
            assert mV_storage is not None, "FP4 PV requires raw packed V storage"
            self.v_storage_dtype = mV_storage.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        if const_expr(mV_storage is not None):
            mV_storage = assume_tensor_aligned(mV_storage)
        if const_expr(self.pack_gqa_local):
            mQ_storage = assume_tensor_aligned(mQ_storage)
        # FP4: align and transpose scale tensors alongside Q/K
        if const_expr(self.use_fp4_qk and mQ_scale is not None):
            mQ_scale = assume_tensor_aligned(mQ_scale)
            mK_scale = assume_tensor_aligned(mK_scale)
        if const_expr(self.use_fp4_pv and mV_scale is not None):
            mV_scale = assume_tensor_aligned(mV_scale)
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        if const_expr(self.pack_gqa_local):
            mQ_storage = cute.make_tensor(
                mQ_storage.iterator, cute.select(mQ_storage.layout, mode=Q_layout_transpose)
            )
        # FP4: transpose scale tensors the same way as Q/K
        if const_expr(self.use_fp4_qk and mQ_scale is not None):
            mQ_scale = cute.make_tensor(mQ_scale.iterator, cute.select(mQ_scale.layout, mode=Q_layout_transpose))
        # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there's cu_seqlens_k or (page_size, d, h_k, num_pages) if there's page_table
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=KV_layout_transpose))
        # FP4: transpose K scale tensor
        if const_expr(self.use_fp4_qk and mK_scale is not None):
            mK_scale = cute.make_tensor(mK_scale.iterator, cute.select(mK_scale.layout, mode=KV_layout_transpose))
        if const_expr(self.is_split_kv):
            O_layout_transpose = [2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0]
            LSE_layout_transpose = [3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0]
            num_splits = mO.shape[0]
        else:
            O_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
            LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            num_splits = Int32(1)
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mLSE = (
            cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
            if const_expr(mLSE is not None)
            else None
        )
        if const_expr(self.use_fp4_pv):
            # FP4 PV consumes pretransposed packed operand-B inputs directly:
            #   mV      : logical Vt = (d, s_k, h_k, b)
            #   mV_scale: compact transposed SFVt groups = (d // sf_vec, s_k, h_k, b)
            assert mCuSeqlensK is None, "FP4 PV currently only supports dense fixed-length inputs"
        else:
            # Preserve the exact pre-PV non-PV BF16 V transform. The old fast path first reused
            # the generic KV transpose and then applied a V-specific transpose to obtain the
            # logical operand-B view consumed by P@V. Re-applying those two layout transforms
            # verbatim avoids subtle stride changes from collapsing them into one direct select.
            mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=KV_layout_transpose))
            V_layout_transpose = [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
            mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=V_layout_transpose))

        if const_expr(self.pack_gqa_effective):
            nheads_kv = mK.shape[2]
            pack_gqa_layout_fn = (
                pack_gqa_layout_seqmajor if const_expr(self.pack_gqa_seqmajor) else pack_gqa_layout
            )
            mQ = pack_gqa_layout_fn(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(self.use_fp4_qk and mQ_scale is not None):
                mQ_scale = pack_gqa_layout_fn(
                    mQ_scale, self.qhead_per_kvhead, nheads_kv, head_idx=2
                )
            if const_expr(not self.pack_gqa_local):
                mO = pack_gqa_layout_fn(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
                if const_expr(mLSE is not None):
                    mLSE = pack_gqa_layout_fn(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        if const_expr(self.use_fp4_qk and mK_scale is not None):
            mQ_scale = cute.make_tensor(
                mQ_scale.iterator,
                bs_layout.tile_atom_to_shape_SF(
                    cute.group_modes(mQ.shape, 2, 4), self.fp4_sf_vec_size
                ),
            )
            mK_scale = cute.make_tensor(
                mK_scale.iterator,
                bs_layout.tile_atom_to_shape_SF(
                    cute.group_modes(mK.shape, 2, 4), self.fp4_sf_vec_size
                ),
            )
        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch Q vs K: {self.q_dtype} != {self.k_dtype}")
        if const_expr(not self.use_fp4_qk):
            # Standard BF16 path: Q/K/V must all match
            if const_expr(self.q_dtype != self.v_dtype):
                raise TypeError(f"Type mismatch Q vs V: {self.q_dtype} != {self.v_dtype}")
        # For FP4 QK: Q/K are FP4 (fp4x2 packed), V stays BF16/FP16
        self._setup_attributes()
        self.use_tma_O = (
            self.arch >= Arch.sm_90
            and mCuSeqlensQ is None
            and mSeqUsedQ is None
            and not self.pack_gqa_local
            and not self.pack_gqa_seqmajor
        )
        # This can be tuned
        # This is currently very ad-hoc, we should tune it systematically
        self.ex2_emu_freq = 0
        # self.ex2_emu_start_frg = 1 if self.is_causal else 0
        self.ex2_emu_start_frg = 1
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = 16
            if const_expr(self.head_dim_padded == 128 and self.use_2cta_instrs):
                self.ex2_emu_freq = 12
            if const_expr(
                self.pack_gqa_effective and self.head_dim_padded > 64 and not self.is_causal and not self.is_local
            ):
                self.ex2_emu_freq = 32 if mCuSeqlensQ is not None or mSeqUsedQ is not None else 10
            if const_expr(self.head_dim_padded > 64 and self.is_causal):
                self.ex2_emu_freq = 10

        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        q_major_mode = tcgen05.OperandMajorMode.K
        k_major_mode = tcgen05.OperandMajorMode.K
        v_major_mode = tcgen05.OperandMajorMode.MN
        self.o_layout = cutlass.utils.LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K
        if const_expr(self.use_fp4_qk):
            # FP4 block-scaled MMA for QK product
            tiled_mma_qk = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float4E2M1FN,
                q_major_mode,
                k_major_mode,
                self.fp4_sf_dtype,
                self.fp4_sf_vec_size,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        else:
            tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
                self.q_dtype,
                q_major_mode,
                k_major_mode,
                self.qk_acc_dtype,
                cta_group,
                self.mma_tiler_qk[:2],
            )
        if const_expr(self.use_fp4_pv):
            tiled_mma_pv = sm100_utils_basic.make_blockscaled_trivial_tiled_mma(
                Float4E2M1FN,
                p_major_mode,
                v_major_mode,
                self.pv_sf_dtype,
                self.pv_sf_vec_size,
                cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )
        else:
            tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
                self.v_dtype,
                p_major_mode,
                v_major_mode,
                self.pv_acc_dtype,
                cta_group,
                self.mma_tiler_pv[:2],
                p_source,
            )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # epi_tile is per-CTA (not full 2CTA) since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.p_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )

        # FP4: Scale factor SMEM layouts for Q and K scales
        if const_expr(self.use_fp4_qk):
            sSFA_layout = bs_layout.make_smem_layout_sfa(
                tiled_mma_qk, self.mma_tiler_qk, self.fp4_sf_vec_size, self.q_stage
            )
            sSFK_layout = bs_layout.make_smem_layout_sfb(
                tiled_mma_qk, self.mma_tiler_qk, self.fp4_sf_vec_size, self.kv_stage
            )
            # TMEM layouts for scale factors
            sSFA_layout_per_stage = cute.select(sSFA_layout, mode=[0, 1, 2])
            sSFK_layout_per_stage = cute.select(sSFK_layout, mode=[0, 1, 2])
            tSFA_layout = bs_layout.make_tmem_layout_sfa(
                tiled_mma_qk, self.mma_tiler_qk, self.fp4_sf_vec_size, sSFA_layout_per_stage
            )
            tSFK_layout = bs_layout.make_tmem_layout_sfb(
                tiled_mma_qk, self.mma_tiler_qk, self.fp4_sf_vec_size, sSFK_layout_per_stage
            )
        else:
            sSFA_layout = None
            sSFK_layout = None
            tSFA_layout = None
            tSFK_layout = None

        if const_expr(self.use_fp4_pv):
            sSFP_layout = bs_layout.make_smem_layout_sfa(
                tiled_mma_pv, self.mma_tiler_pv, self.pv_sf_vec_size, self.q_stage
            )
            sSFV_layout = bs_layout.make_smem_layout_sfb(
                tiled_mma_pv, self.mma_tiler_pv, self.pv_sf_vec_size, self.kv_stage
            )
            sSFP_layout_per_stage = cute.select(sSFP_layout, mode=[0, 1, 2])
            sSFV_layout_per_stage = cute.select(sSFV_layout, mode=[0, 1, 2])
            tSFP_layout = bs_layout.make_tmem_layout_sfa(
                tiled_mma_pv, self.mma_tiler_pv, self.pv_sf_vec_size, sSFP_layout_per_stage
            )
            tSFV_layout = bs_layout.make_tmem_layout_sfb(
                tiled_mma_pv, self.mma_tiler_pv, self.pv_sf_vec_size, sSFV_layout_per_stage
            )
        else:
            sSFP_layout = None
            sSFV_layout = None
            tSFP_layout = None
            tSFV_layout = None

        if const_expr(not self.same_hdim_kv_padded and not self.use_fp4_qk):
            # sK and sV are using the same physical smem so we need to adjust the stride so that they line up
            stride_sK = const_expr(
                max(sK_layout.outer.stride[-1], 0)
            )  # take max to turn tuple to Int32
            stride_sV = const_expr(max(sV_layout.outer.stride[-1], 0))
            stage_stride = const_expr(
                max(stride_sK, stride_sV)
                if not self.uneven_kv_smem
                else (stride_sK + stride_sV) // 2
            )
            sK_layout = cute.make_composed_layout(
                sK_layout.inner,
                0,
                cute.make_layout(
                    (*sK_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sK_layout.outer.stride[:-1], stage_stride),
                ),
            )
            sV_layout = cute.make_composed_layout(
                sV_layout.inner,
                0,
                cute.make_layout(
                    (*sV_layout.outer.shape[:-1], self.kv_stage),
                    stride=(*sV_layout.outer.stride[:-1], stage_stride),
                ),
            )

        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(mQ.element_type, cute.select(sQ_layout, mode=[0, 1, 2])),
            "K": cute.size_in_bytes(mK.element_type, cute.select(sK_layout, mode=[0, 1, 2])),
            "V": cute.size_in_bytes(mV.element_type, cute.select(sV_layout, mode=[0, 1, 2])),
        }
        for name in ("Q", "K", "V"):
            self.tma_copy_bytes[name] *= self.cta_group_size

        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        if const_expr(not self.pack_gqa_local):
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
        else:
            tma_atom_Q = None

        tma_atom_K = None
        tma_atom_V = None
        if const_expr(self.use_tma_KV):
            # TMA load for K
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            # TMA load for V
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        # FP4: Scale factors are small enough to load manually (no TMA needed)
        # Scale tensors are passed directly to the kernel as raw pointers
        # The kernel will load them into SMEM cooperatively

        self.num_epilogue_threads = cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )
            gmem_tiled_copy_O = None
        else:
            tma_atom_O = None
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self.o_dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.o_dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tO_shape_dim_1 = sO_layout.outer.shape[1][0] // async_copy_elems
            tO_layout = cute.make_ordered_layout(
                (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
                order=(1, 0),
            )
            # So that we don't have to check if we overshoot kBlockM when we store O
            assert self.m_block_size % tO_layout.shape[0] == 0
            vO_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

        if const_expr(self.pack_gqa_local):
            q_storage_row_elems = self.head_dim_padded * self.q_dtype.width // self.q_storage_dtype.width
            gmem_tiled_copy_Q = fa_copy_utils.tiled_copy_2d(
                self.q_storage_dtype,
                q_storage_row_elems,
                cute.arch.WARP_SIZE,
                is_async=False,
            )
            gmem_tiled_copy_Q_scale = fa_copy_utils.tiled_copy_2d(
                self.fp4_sf_dtype,
                self.head_dim_padded // self.fp4_sf_vec_size,
                cute.arch.WARP_SIZE,
                is_async=False,
            )
        else:
            gmem_tiled_copy_Q = None
            gmem_tiled_copy_Q_scale = None

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            if const_expr(self.is_causal or self.is_local):
                # The LPT scheduler is a performance heuristic. The FP4 bring-up path keeps
                # causal dense forward on the simpler single-tile scheduler until the runtime
                # path is stable under CuTe's dynamic loop constraints.
                TileScheduler = SingleTileScheduler if const_expr(self.use_fp4_qk) else SingleTileLPTScheduler
            else:
                TileScheduler = (
                    SingleTileScheduler
                    if const_expr(not self.is_persistent)
                    else StaticPersistentTileScheduler
                )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mK.shape[2]) if const_expr(self.group_qheads_by_kv) else cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            self.head_dim_v_padded,
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            num_subhead=self.qhead_per_kvhead if const_expr(self.group_qheads_by_kv) else 1,
            tile_shape_mn=self.cta_tiler[:2],
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
            # The causal LPT scheduler estimates K/V residency in bytes. FP4 Q/K uses a logical
            # 4-bit CuTe dtype, so width // 8 would become 0 and break scheduler compilation.
            # Use packed-byte granularity for the FP4 K estimate in that path.
            element_size=1 if const_expr(self.use_fp4_qk) else self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.is_split_kv,
            cluster_shape_mn=self.cluster_shape_mn,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout) if const_expr(not self.overlap_sO_sQ) else 0
        sQ_size = (
            cute.cosize(sQ_layout) if const_expr(not self.overlap_sO_sQ) else
            cutlass.max(cute.cosize(sQ_layout), cute.cosize(sO_layout) * self.o_dtype.width // self.q_dtype.width)
        )
        sV_size = cute.cosize(sV_layout) if const_expr(self.use_fp4_qk or self.use_fp4_pv) else 0

        # FP4: Compute scale factor SMEM sizes
        if const_expr(self.use_fp4_qk):
            sSFA_size = cute.cosize(sSFA_layout)
            sSFK_size = cute.cosize(sSFK_layout)
        else:
            sSFA_size = 0
            sSFK_size = 0
        if const_expr(self.use_fp4_pv):
            sSFP_size = cute.cosize(sSFP_layout)
            sSFV_size = cute.cosize(sSFV_layout)
        else:
            sSFP_size = 0
            sSFV_size = 0

        if const_expr(self.use_fp4_pv):
            @cute.struct
            class SharedStorage:
                # m_barriers for pipelines
                mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
                mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
                # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
                mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
                # Tmem dealloc cluster barrier
                tmem_dealloc_mbar_ptr: Int64
                # Tmem holding buffer
                tmem_holding_buf: Int32
                # Smem tensors
                # store row_sum / row_max plus a dedicated recurrence-scale buffer
                sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 3]
                sO: cute.struct.Align[
                    cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
                ]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
                ]
                sK: cute.struct.Align[
                    # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                    cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, sV_size],
                    self.buffer_align_bytes,
                ]
                # FP4: Scale factor SMEM for Q and K scales
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.fp4_sf_dtype, sSFA_size],
                    128,  # 128-byte alignment for scale factors
                ]
                sSFK: cute.struct.Align[
                    cute.struct.MemRange[self.fp4_sf_dtype, sSFK_size],
                    128,
                ]
                sSFP: cute.struct.Align[
                    cute.struct.MemRange[self.pv_sf_dtype, sSFP_size],
                    128,
                ]
                sSFV: cute.struct.Align[
                    cute.struct.MemRange[self.pv_sf_dtype, sSFV_size],
                    128,
                ]
        else:
            @cute.struct
            class SharedStorage:
                # m_barriers for pipelines
                mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
                mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_O_full: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 2]
                # mbar_softmax_stats: cute.struct.MemRange[Int64, self.q_stage * 4 * 2]
                mbar_O_epi: cute.struct.MemRange[Int64, self.q_stage * 2]
                mbar_s0_s1_sequence: cute.struct.MemRange[Int64, 2 * 2]
                # Tmem dealloc cluster barrier
                tmem_dealloc_mbar_ptr: Int64
                # Tmem holding buffer
                tmem_holding_buf: Int32
                # Smem tensors
                # store row_sum / row_max plus a dedicated recurrence-scale buffer
                sScale: cute.struct.MemRange[Float32, self.q_stage * self.m_block_size * 3]
                sO: cute.struct.Align[
                    cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
                ]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
                ]
                sK: cute.struct.Align[
                    # cute.cosize(sK_layout) is correct even in the case of self.uneven_kv_smem
                    cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                    self.buffer_align_bytes,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[self.v_dtype, sV_size],
                    self.buffer_align_bytes,
                ]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.fp4_sf_dtype, sSFA_size],
                    128,
                ]
                sSFK: cute.struct.Align[
                    cute.struct.MemRange[self.fp4_sf_dtype, sSFK_size],
                    128,
                ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa_effective, aux_tensors, mPageTable
        )

        head_divmod = None
        if cutlass.const_expr(self.pack_gqa_effective):
            head_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        if cutlass.const_expr(self.use_block_sparsity and mPageTable is not None):
            raise NotImplementedError("Block sparsity + paged KV not supported on SM100")

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mQ_storage,
            mK,
            mV,
            mV_storage,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_Q_scale,
            # FP4: scale factor tensors (raw pointers, no TMA)
            mQ_scale,
            mK_scale,
            mV_scale,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            sSFA_layout,
            sSFK_layout,
            sSFP_layout,
            sSFV_layout,
            tSFA_layout,
            tSFK_layout,
            tSFP_layout,
            tSFV_layout,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            num_splits,
            aux_tensors,
            fastdiv_mods,
            head_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) or (total_q, d, h) if there is cu_seqlens_q
        mQ_storage: Optional[cute.Tensor],  # (s_q, d_packed, h, b) raw packed-byte Q storage
        mK: cute.Tensor,  # (s_k, d, h_k, b_k) or (total_k, d, h_k) if there is cu_seqlens_k or (page_size, d, h_k, num_pages) if there is page_table
        mV: cute.Tensor,  # BF16 path: (d, s_k, h_k, b_k); FP4 PV path: public packed logical V
        mV_storage: Optional[cute.Tensor],  # (b, s_k, h_k, d_packed) raw packed-byte V storage
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_Q_scale: Optional[cute.TiledCopy],
        # FP4: scale factor tensors (raw pointers, no TMA)
        mQ_scale: Optional[cute.Tensor],
        mK_scale: Optional[cute.Tensor],
        mV_scale: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sSFA_layout,  # None for BF16 path, cute.Layout for FP4
        sSFK_layout,  # None for BF16 path, cute.Layout for FP4
        sSFP_layout,  # None for BF16 PV path, cute.Layout for FP4 PV
        sSFV_layout,  # None for BF16 PV path, cute.Layout for FP4 PV
        tSFA_layout,  # None for BF16 path, cute.Layout for FP4 (TMEM)
        tSFK_layout,  # None for BF16 path, cute.Layout for FP4 (TMEM)
        tSFP_layout,  # None for BF16 PV path, cute.Layout for FP4 PV (TMEM)
        tSFV_layout,  # None for BF16 PV path, cute.Layout for FP4 PV (TMEM)
        gmem_tiled_copy_O: Optional[cute.TiledCopy],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        num_splits: Int32,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )
        # Setup cta/thread coordinates
        bidx, _, _ = cute.arch.block_idx()
        if const_expr(cute.size(tiled_mma_qk.thr_id.shape) == 1):
            mma_tile_coord_v = 0
        else:
            mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_retrieve_warp_ids = (
            self.mma_warp_id,
            *self.softmax0_warp_ids,
            *self.softmax1_warp_ids,
            *self.correction_warp_ids,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE * len(tmem_retrieve_warp_ids),
        )
        # Tensor memory dealloc barrier init
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        load_warps = ThreadCooperativeGroup(len(self.load_warp_ids))
        tma_warp = ThreadCooperativeGroup(1)
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.softmax0_warp_ids))
        # softmax_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        # correction_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE)
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(cute.arch.WARP_SIZE * len(self.epilogue_warp_ids))
        # For UMMA-bridging pipelines: the non-MMA side spans both CTAs in the cluster,
        # so the thread count must include warps from both CTAs.
        softmax_warps_cluster = ThreadCooperativeGroup(
            len(self.softmax0_warp_ids) * self.cta_group_size
        )
        softmax_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids) * self.cta_group_size
        )
        correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids) * self.cta_group_size
        )
        softmax_correction_threads_cluster = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids) * self.cta_group_size
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        if const_expr(self.use_tma_KV):
            pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=tma_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        else:
            cpasync_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.load_warp_ids) * cute.arch.WARP_SIZE
            )
            pipeline_kv = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.mbar_load_KV.data_ptr(),
                num_stages=self.kv_stage,
                producer_group=cpasync_producer_group,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_warps_cluster,
            consumer_group=mma_warp,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s0_s1_sequence = None
        if const_expr(self.s0_s1_barrier and self.q_stage > 1):
            # This is not a typical producer-consumer pipeline. We will directly use
            # pipeline_s0_s1_sequence.sync_object_full and will not use
            # pipeline_s0_s1_sequence.sync_object_empty.
            pipeline_s0_s1_sequence = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_s0_s1_sequence.data_ptr(),
                num_stages=2,
                producer_group=softmax_threads,
                consumer_group=softmax_threads,
                defer_sync=True,
            )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = pipeline_custom.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0), num_threads=cute.arch.WARP_SIZE * 2
        )
        pipeline_o_epi = None
        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi = pipeline_custom.PipelineAsync.create(
                barrier_storage=storage.mbar_O_epi.data_ptr(),
                num_stages=self.q_stage,
                producer_group=correction_threads,
                consumer_group=epilogue_threads,
                defer_sync=True,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        if const_expr(self.use_fp4_qk or self.use_fp4_pv):
            # FP4 QK keeps V in BF16/FP16, so K and V can no longer alias the same SMEM buffer.
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            # Strip swizzle info to reuse smem when K and V share dtype/footprint.
            sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
        if const_expr(not self.overlap_sO_sQ):
            sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        else:
            sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, self.o_dtype), sO_layout.outer)

        scale_slots = self.q_stage * self.m_block_size
        sScaleStorage = storage.sScale.get_tensor(cute.make_layout(scale_slots * 3))
        sScale = cute.make_tensor(sScaleStorage.iterator, cute.make_layout(scale_slots * 2))
        sAccScale = cute.make_tensor(
            sScaleStorage.iterator + scale_slots * 2,
            cute.make_layout(scale_slots),
        )

        # FP4: Create SMEM tensors for scale factors
        if const_expr(self.use_fp4_qk):
            sSFA = storage.sSFA.get_tensor(sSFA_layout)
            sSFK = storage.sSFK.get_tensor(sSFK_layout)
        else:
            sSFA = None
            sSFK = None
        if const_expr(self.use_fp4_pv):
            sSFP = storage.sSFP.get_tensor(sSFP_layout)
            sSFV = storage.sSFV.get_tensor(sSFV_layout)
        else:
            sSFP = None
            sSFV = None

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in p_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.p_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (self.tmem_p_offset[1] - self.tmem_p_offset[0]) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(tOrP.layout, cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)))
        )
        if cute.arch.thread_idx()[0] == 0:
            fa_logging.fa_printf(
                2,
                "PVExp tOrP rank=%d kblocks=%d stage_stride=%d\n",
                Int32(cute.rank(tOrP[None, None, None, Int32(0)])),
                Int32(cute.size(tOrP[None, None, None, Int32(0)], mode=[2])),
                Int32(tP_stage_stride),
            )

        block_info = BlockInfo(
            # This is cta_tiler, not mma_tiler_qk, since we move by block by (2 * mma_tiler[0], mma_tiler[1])
            self.cta_tiler[0],
            self.cta_tiler[1],
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=(
                mQ.shape[0]
                if const_expr(not self.pack_gqa_effective)
                else (mQ.shape[0][0] if const_expr(self.pack_gqa_seqmajor) else mQ.shape[0][1])
            ),
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.m_block_size,
            self.n_block_size,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        for i in cutlass.range_constexpr(len(self.empty_warp_ids)):
            if warp_idx == self.empty_warp_ids[i]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mQ_storage,
                mK,
                mV,
                mV_storage,
                mQ_scale,
                mK_scale,
                mV_scale,
                sQ,
                sK,
                sV,
                sSFA,
                sSFK,
                sSFV,
                mPageTable,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_Q_scale,
                pipeline_q,
                pipeline_kv,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            )
        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            if const_expr(self.use_fp4_qk):
                assert tSFA_layout is not None and tSFK_layout is not None
                tCtSFA = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + self.tmem_sfa_offset, dtype=self.fp4_sf_dtype),
                    tSFA_layout,
                )
                tCtSFK = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + self.tmem_sfk_offset, dtype=self.fp4_sf_dtype),
                    tSFK_layout,
                )
            else:
                tCtSFA = None
                tCtSFK = None
            if const_expr(self.use_fp4_pv):
                assert tSFP_layout is not None and tSFV_layout is not None
                tCtSFP = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + self.tmem_sfp_offset, dtype=self.pv_sf_dtype),
                    tSFP_layout,
                )
                tCtSFV = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + self.tmem_sfv_offset, dtype=self.pv_sf_dtype),
                    tSFV_layout,
                )
            else:
                tCtSFP = None
                tCtSFV = None
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                sScale,
                sAccScale,
                sSFA,
                sSFK,
                sSFP,
                sSFV,
                tStS,
                tOtO,
                tOrP,
                tCtSFA,
                tCtSFK,
                tCtSFP,
                tCtSFV,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                pipeline_sm_stats,
                is_leader_cta,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if const_expr(not self.use_correction_warps_for_epi):
            if warp_idx >= self.epilogue_warp_ids[0] and warp_idx <= self.epilogue_warp_ids[-1]:
                cute.arch.setmaxregister_decrease(self.num_regs_other)
                self.epilogue_s2g(
                    mO,
                    sO,
                    gmem_tiled_copy_O,
                    tma_atom_O,
                    pipeline_o_epi,
                    block_info,
                    num_splits,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    mma_tile_coord_v,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or
            (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1])
        ):
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            if const_expr(self.use_fp4_pv) and warp_idx == self.softmax0_warp_ids[0]:
                if const_expr(not self.use_exact_fp4_pv_lane):
                    self.fill_acc_scale_constant(sAccScale, cute.arch.lane_idx(), Float32(1.0))
                assert sSFP is not None
                for q_stage_idx in cutlass.range_constexpr(self.q_stage):
                    self.fill_scale_stage_constant(
                        sSFP[None, None, None, q_stage_idx],
                        cute.arch.lane_idx(),
                        Float32(1.0),
                    )
                if const_expr(not self.use_exact_fp4_pv_lane):
                    assert sSFV is not None
                    for kv_stage_idx in cutlass.range_constexpr(self.kv_stage):
                        self.fill_scale_stage_constant(
                            sSFV[None, None, None, kv_stage_idx],
                            cute.arch.lane_idx(),
                            Float32(1.0),
                        )
                    self.publish_shared_scale_fill()
            tmem.wait_for_alloc()
            tiled_copy_s2t_sfp_exact = None
            tCsSFP_compact_s2t_exact = None
            tCtSFP_compact_s2t_exact = None
            if const_expr(self.use_exact_fp4_pv_lane):
                assert self.use_fp4_pv
                assert sSFP is not None and tSFP_layout is not None
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
                tCtSFP_exact = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + self.tmem_sfp_offset, dtype=self.pv_sf_dtype),
                    tSFP_layout,
                )
                (
                    tiled_copy_s2t_sfp_exact,
                    tCsSFP_compact_s2t_exact,
                    tCtSFP_compact_s2t_exact,
                ) = self.scale_s2t_copy_and_partition(
                    sSFP,
                    tCtSFP_exact,
                    scale_dtype=self.pv_sf_dtype,
                )
            if const_expr(not self.use_exact_fp4_pv_lane):
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                softmax_scale=softmax_scale,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                sAccScale=sAccScale,
                sSFP=sSFP,
                tiled_copy_s2t_sfp_exact=tiled_copy_s2t_sfp_exact,
                tCsSFP_compact_s2t_exact=tCsSFP_compact_s2t_exact,
                tCtSFP_compact_s2t_exact=tCtSFP_compact_s2t_exact,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                learnable_sink=learnable_sink,
                block_info=block_info,
                num_splits=num_splits,
                SeqlenInfoCls=SeqlenInfoCls,
                AttentionMaskCls=AttentionMaskCls,
                TileSchedulerCls=TileSchedulerCls,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                blocksparse_tensors=blocksparse_tensors,
            )

            if const_expr(not self.s0_s1_barrier):
                stage = Int32(0 if const_expr(self.q_stage == 1) or warp_idx < self.softmax1_warp_ids[0] else 1)
                softmax_loop(stage=stage, tStS=tStS)
            else:
                # If there's s0_s1_barrier, it's faster to have 2 WGs having different code
                if warp_idx < self.softmax1_warp_ids[0]:
                    softmax_loop(stage=0, tStS=tStS)
                if warp_idx < self.correction_warp_ids[0] and warp_idx >= self.softmax1_warp_ids[0]:
                    softmax_loop(stage=1, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.wait_for_alloc()
            if const_expr(not self.use_exact_fp4_pv_lane):
                tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        mQ: cute.Tensor,
        mQ_storage: Optional[cute.Tensor],
        mK: cute.Tensor,
        mV: cute.Tensor,
        mV_storage: Optional[cute.Tensor],
        mQ_scale: Optional[cute.Tensor],
        mK_scale: Optional[cute.Tensor],
        mV_scale: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sSFA: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        gmem_tiled_copy_Q: Optional[cute.TiledCopy],
        gmem_tiled_copy_Q_scale: Optional[cute.TiledCopy],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_padded,
            self.check_hdim_oob,
            self.qhead_per_kvhead,
            seqmajor_layout=self.pack_gqa_seqmajor,
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self.kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            mma_tile_coord_v = thr_mma_qk.thr_idx
            q_head_idx = self.q_head_idx(head_idx, split_idx)
            kv_head_idx = self.kv_head_idx(head_idx)
            gV_scale = None
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, q_head_idx]
            if const_expr(self.pack_gqa_local):
                assert mQ_storage is not None
                mQ_storage_cur = seqlen.offset_batch_Q(mQ_storage, batch_idx, dim=3)
            else:
                mQ_storage_cur = None
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128 * 2, 128)
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )  # (128, 128, 2)
            if const_expr(self.use_fp4_qk):
                assert mQ_scale is not None and mK_scale is not None
                assert sSFA is not None and sSFK is not None
                # FP4 QK is dense-only in v1, so the grouped (head, batch) scale mode
                # can be indexed directly without the varlen batch offset helpers.
                mQ_scale_cur = mQ_scale[None, None, (q_head_idx, batch_idx)]
                tiler_gQ_scale = (
                    (self.mma_tiler_qk[0] * self.q_stage) // self.cta_group_size,
                    self.head_dim_padded,
                )
                gQ_scale_block = m_block * self.cta_group_size + mma_tile_coord_v
                gQ_scale = cute.local_tile(mQ_scale_cur, tiler_gQ_scale, (gQ_scale_block, 0))
                gQ_scale = layout_utils.select(
                    cute.flat_divide(gQ_scale, (self.mma_tiler_qk[0] // self.cta_group_size,)),
                    mode=[0, 2, 1],
                )

            if const_expr(mPageTable is None):
                if const_expr(not seqlen.has_cu_seqlens_k):
                    mK_cur = mK[None, None, kv_head_idx, batch_idx]
                    if const_expr(not self.use_fp4_pv or not self.fp4_pv_manual_direct_loader):
                        mV_cur = mV[None, None, kv_head_idx, batch_idx]
                    if const_expr(self.use_fp4_qk):
                        mK_scale_cur = mK_scale[None, None, (kv_head_idx, batch_idx)]
                else:
                    mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, kv_head_idx])
                    if const_expr(not self.use_fp4_pv or not self.fp4_pv_manual_direct_loader):
                        mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, kv_head_idx])
                    if const_expr(self.use_fp4_qk):
                        mK_scale_cur = cute.domain_offset(
                            (seqlen.offset_k, 0), mK_scale[None, None, (kv_head_idx, batch_idx)]
                        )
                gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0))
                if const_expr(not self.use_fp4_pv or not self.fp4_pv_manual_direct_loader):
                    gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None))
                if const_expr(self.use_fp4_qk):
                    gK_scale = cute.local_tile(
                        mK_scale_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0)
                    )
                if const_expr(self.use_fp4_pv and not self.fp4_pv_manual_direct_loader):
                    assert mV_scale is not None and sSFV is not None
            else:
                # Need to keep batch coord None since we'll index into it with page idx
                mK_cur = mK[None, None, kv_head_idx, None]
                gK = cute.local_tile(
                    mK_cur, cute.select(self.mma_tiler_qk, mode=[1, 2]), (None, 0, None)
                )
                if const_expr(not self.use_fp4_pv or not self.fp4_pv_manual_direct_loader):
                    mV_cur = mV[None, None, kv_head_idx, None]
                    gV = cute.local_tile(
                        mV_cur, cute.select(self.mma_tiler_pv, mode=[1, 2]), (0, None, None)
                    )
            tSgQ = thr_mma_qk.partition_A(gQ)
            tSgK = thr_mma_qk.partition_B(gK)
            if const_expr(not self.use_fp4_pv or not self.fp4_pv_manual_direct_loader):
                tOgV = thr_mma_pv.partition_B(gV)
            if const_expr(not self.pack_gqa_local):
                load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
                )
            else:
                load_Q_fn = None

            if const_expr(self.use_tma_KV):
                tKsK, tKgK = cpasync.tma_partition(
                    tma_atom_K,
                    0,  # no multicast
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK, 0, 3),
                )
                if const_expr(self.use_fp4_pv and self.fp4_pv_manual_direct_loader):
                    tVsV, tVgV = None, None
                else:
                    tVsV, tVgV = cpasync.tma_partition(
                        tma_atom_V,
                        0,  # no multicast
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tOgV, 0, 3),
                    )
                paged_kv_manager = None
            else:
                page_size = mK.shape[0]
                paged_kv_manager = PagedKVManager.create(
                    mPageTable,
                    mK,
                    mV,
                    FastDivmodDivisor(page_size),
                    batch_idx,
                    kv_head_idx,
                    tidx,
                    seqlen.seqlen_k,
                    0,  # leftpad_k
                    self.n_block_size,
                    self.head_dim_padded,
                    self.head_dim_v_padded,
                    num_load_threads,
                    mK.element_type,
                )
                tKsK, tKgK = None, None
                tVsV, tVgV = None, None

            load_Q = partial(self.load_Q, load_Q_fn, pipeline_q=pipeline_q, phase=q_producer_phase)
            load_K = partial(
                self.load_KV,
                tma_atom_K,
                tKgK,
                tKsK,
                paged_kv_manager,
                sK,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            if const_expr(self.use_fp4_pv):
                assert mV_scale is not None and sSFV is not None
            if const_expr(self.use_fp4_pv and self.fp4_pv_manual_direct_loader):
                assert mV_storage is not None
                load_V = partial(
                    self.load_v_fp4_pv_stage_public,
                    mV,
                    mV_storage,
                    mV_scale,
                    thr_mma_pv,
                    batch_idx,
                    kv_head_idx,
                    sV,
                    sSFV,
                    seqlen_k=seqlen.seqlen_k,
                    pipeline_kv=pipeline_kv,
                )
            elif const_expr(self.use_fp4_pv):
                load_V = partial(
                    self.load_v_with_scale_stage_public,
                    tma_atom_V,
                    tVgV,
                    tVsV,
                    paged_kv_manager,
                    mV_scale,
                    sV,
                    sSFV,
                    batch_idx,
                    kv_head_idx,
                    seqlen.seqlen_k,
                    pipeline_kv=pipeline_kv,
                )
            else:
                load_V = partial(
                    self.load_KV,
                    tma_atom_V,
                    tVgV,
                    tVsV,
                    paged_kv_manager,
                    sV,
                    pipeline_kv=pipeline_kv,
                    K_or_V="V",
                )

            if const_expr(not self.use_block_sparsity):
                n_block_min, n_block_max = block_info.get_n_block_min_max(
                    seqlen, m_block, split_idx, num_splits
                )
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    n_block_first = n_block_max - 1 if n_block_max > 0 else 0
                    page_idx = (
                        mPageTable[batch_idx, n_block_first]
                        if const_expr(mPageTable is not None and self.use_tma_KV)
                        else None
                    )
                    if const_expr(not self.use_tma_KV):
                        paged_kv_manager.load_page_table(n_block_first)
                    load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # K0
                    if const_expr(self.use_fp4_qk) and warp_idx == self.load_warp_ids[0]:
                        self.load_scale_stage(
                            gK_scale[None, None, n_block_max - 1],
                            sSFK[None, None, None, kv_producer_state.index],
                        )
                    # load_K(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx, extra_tx_count=self.tma_copy_bytes["Q"])  # K0
                    if const_expr(len(self.load_warp_ids) == 1) or warp_idx == self.load_warp_ids[0]:
                        # load_Q(block=0, stage=0)  # Q0
                        if const_expr(not self.pack_gqa_local):
                            pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                            tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                            load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                            if const_expr(self.use_fp4_qk):
                                self.load_scale_stage(
                                    gQ_scale[None, None, 0],
                                    sSFA[None, None, None, 0],
                                )
                        else:
                            assert gmem_tiled_copy_Q is not None
                            assert mQ_storage_cur is not None
                            self.load_Q_packed_local(
                                pack_gqa,
                                mQ_storage_cur,
                                kv_head_idx,
                                sQ[None, None, None, 0],
                                gmem_tiled_copy_Q,
                                pipeline_q,
                                block=0,
                                stage=0,
                                phase=q_producer_phase,
                                seqlen=seqlen.seqlen_q,
                            )
                            if const_expr(self.use_fp4_qk):
                                assert gmem_tiled_copy_Q_scale is not None
                                self.load_scale_packed_local(
                                    pack_gqa,
                                    gQ_scale[None, None, 0],
                                    sSFA[None, None, None, 0],
                                    gmem_tiled_copy_Q_scale,
                                    block=0,
                                    seqlen=seqlen.seqlen_q,
                                )
                            pipeline_q.producer_commit_w_index(0)
                    kv_producer_state.advance()
                    if const_expr(self.q_stage == 2) and (const_expr(len(self.load_warp_ids) == 1) or warp_idx == self.load_warp_ids[0]):
                        if const_expr(not self.pack_gqa_local):
                            pipeline_q.producer_acquire_w_index_phase(1, q_producer_phase)
                            tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(1)
                            load_Q_fn(src_idx=1, dst_idx=1, tma_bar_ptr=tma_bar_ptr)
                            if const_expr(self.use_fp4_qk):
                                self.load_scale_stage(
                                    gQ_scale[None, None, 1],
                                    sSFA[None, None, None, 1],
                                )
                        else:
                            assert gmem_tiled_copy_Q is not None
                            assert mQ_storage_cur is not None
                            self.load_Q_packed_local(
                                pack_gqa,
                                mQ_storage_cur,
                                kv_head_idx,
                                sQ[None, None, None, 1],
                                gmem_tiled_copy_Q,
                                pipeline_q,
                                block=1,
                                stage=1,
                                phase=q_producer_phase,
                                seqlen=seqlen.seqlen_q,
                            )
                            if const_expr(self.use_fp4_qk):
                                assert gmem_tiled_copy_Q_scale is not None
                                self.load_scale_packed_local(
                                    pack_gqa,
                                    gQ_scale[None, None, 1],
                                    sSFA[None, None, None, 1],
                                    gmem_tiled_copy_Q_scale,
                                    block=1,
                                    seqlen=seqlen.seqlen_q,
                                )
                            pipeline_q.producer_commit_w_index(1)
                    q_producer_phase ^= 1
                    load_V(block=n_block_max - 1, producer_state=kv_producer_state, page_idx=page_idx)  # V0
                    kv_producer_state.advance()
                    for i in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                        n_block = n_block_max - 2 - i
                        page_idx = (
                            mPageTable[batch_idx, n_block]
                            if const_expr(mPageTable is not None and self.use_tma_KV)
                            else None
                        )
                        if const_expr(not self.use_tma_KV):
                            paged_kv_manager.load_page_table(n_block)
                    # if cute.arch.thread_idx()[0] % 32 == 0: cute.printf("n_block = {}, page_idx = {}", n_block, page_idx)
                        load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Ki
                        if const_expr(self.use_fp4_qk) and warp_idx == self.load_warp_ids[0]:
                            self.load_scale_stage(
                                gK_scale[None, None, n_block],
                                sSFK[None, None, None, kv_producer_state.index],
                            )
                        kv_producer_state.advance()
                        load_V(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)  # Vi
                        kv_producer_state.advance()

            else:
                kv_producer_state, q_producer_phase = produce_block_sparse_loads_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_producer_state,
                    load_Q,
                    load_K,
                    load_V,
                    pipeline_kv,
                    self.q_stage,
                    q_producer_phase,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail
        if const_expr(not self.pack_gqa_local) and (
            const_expr(len(self.load_warp_ids) == 1) or warp_idx == self.load_warp_ids[0]
        ):
            pipeline_q.producer_acquire_w_index_phase(self.q_stage - 1, q_producer_phase)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.core.ThrMma,
        tiled_mma_pv: cute.core.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sScale: cute.Tensor,
        sAccScale: cute.Tensor,
        sSFA: Optional[cute.Tensor],
        sSFK: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        sSFV: Optional[cute.Tensor],
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        tCtSFA: Optional[cute.Tensor],
        tCtSFK: Optional[cute.Tensor],
        tCtSFP: Optional[cute.Tensor],
        tCtSFV: Optional[cute.Tensor],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        is_leader_cta: Boolean,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors],
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        if const_expr(self.use_fp4_pv):
            # FP4 PV must consume operand-B from the native swizzled Vt shared-memory
            # layout. Use the tiled MMA fragment directly instead of rebuilding a
            # per-thread RMEM alias from byte copies.
            tOrV = tiled_mma_pv.make_fragment_B(sV)
        else:
            tOrV = tiled_mma_pv.make_fragment_B(sV)
        if const_expr(self.use_fp4_qk):
            assert sSFA is not None and sSFK is not None
            assert tCtSFA is not None and tCtSFK is not None
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.scale_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfk,
                tCsSFK_compact_s2t,
                tCtSFK_compact_s2t,
            ) = self.scale_s2t_copy_and_partition(sSFK, tCtSFK)
        else:
            tiled_copy_s2t_sfa = None
            tCsSFA_compact_s2t = None
            tCtSFA_compact_s2t = None
            tiled_copy_s2t_sfk = None
            tCsSFK_compact_s2t = None
            tCtSFK_compact_s2t = None
        if const_expr(self.use_fp4_pv):
            assert sSFP is not None and sSFV is not None
            assert tCtSFP is not None and tCtSFV is not None
            (
                tiled_copy_s2t_sfp,
                tCsSFP_compact_s2t,
                tCtSFP_compact_s2t,
            ) = self.scale_s2t_copy_and_partition(sSFP, tCtSFP, scale_dtype=self.pv_sf_dtype)
            (
                tiled_copy_s2t_sfv,
                tCsSFV_compact_s2t,
                tCtSFV_compact_s2t,
            ) = self.scale_s2t_copy_and_partition(sSFV, tCtSFV, scale_dtype=self.pv_sf_dtype)
        else:
            tiled_copy_s2t_sfp = None
            tCsSFP_compact_s2t = None
            tCtSFP_compact_s2t = None
            tiled_copy_s2t_sfv = None
            tCsSFV_compact_s2t = None
            tCtSFV_compact_s2t = None
        if const_expr(self.q_stage == 2):
            tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
        else:
            tSrQs = (tSrQ[None, None, None, 0],)

        qk_mma_op, pv_mma_op = tiled_mma_qk.op, tiled_mma_pv.op
        if const_expr(self.use_fp4_qk):
            qk_mma_idesc = sm100_desc.mma_op_to_idesc_block_scaled(qk_mma_op)
        else:
            qk_mma_idesc = sm100_desc.mma_op_to_idesc(qk_mma_op)
        q_smem_base = sm100_desc.smem_desc_base_from_tensor(sQ, sm100_desc.Major.K)
        k_smem_base = sm100_desc.smem_desc_base_from_tensor(sK, sm100_desc.Major.K)
        q_smem_start = [sm100_desc.make_smem_desc_start_addr(sQ[None, None, None, stage].iterator) for stage in range(self.q_stage)]
        sm100_utils.declare_ptx_smem_desc(q_smem_start[self.q_stage - 1], q_smem_base, tSrQ[None, None, None, 0].layout, var_name_prefix="fa_fwd_q_smem_desc")
        if const_expr(self.use_fp4_qk):
            sm100_utils.declare_ptx_idesc_block_scaled(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        else:
            sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="fa_fwd_qk_mma_idesc")
        if const_expr(not self.use_fp4_pv):
            sm100_utils.declare_ptx_idesc(pv_mma_op, var_name="fa_fwd_pv_mma_idesc")

        sQ_stage_stride = (sQ.layout.stride[-1] * sQ.element_type.width // 8) >> 4
        if const_expr(self.q_stage == 1):
            sQ_stage_stride = 0
        fp4_scale_vec = "4X" if self.fp4_sf_vec_size == 16 else "2X"

        if const_expr(self.use_fp4_qk):
            # FP4 block-scaled QK GEMM dispatch
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_fp4_block_scaled,
                    self.tmem_s_offset[stage],
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK[None, None, None, 0].layout,
                    smem_var_name_prefix=f"fa_fwd_q_smem_desc",
                    idesc_var_name=f"fa_fwd_qk_mma_idesc",
                    tmem_sa_addr=Int32(self.tmem_sfa_offset),
                    tmem_sb_addr=Int32(self.tmem_sfk_offset),
                    smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                    scale_vec=fp4_scale_vec,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.q_stage)
            ]
        else:
            # Standard BF16 QK GEMM dispatch
            gemm_Si = [
                partial(
                    sm100_utils.gemm_ptx_precomputed_varname,
                    self.tmem_s_offset[stage],
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK[None, None, None, 0].layout,
                    smem_var_name_prefix=f"fa_fwd_q_smem_desc",
                    idesc_var_name=f"fa_fwd_qk_mma_idesc",
                    smem_offset=-sQ_stage_stride if stage == 0 else sQ_stage_stride,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.q_stage)
            ]
        if const_expr(self.use_fp4_pv):
            gemm_Pi = [
                partial(
                    sm100_utils.gemm_ptx_fp4_block_scaled_partial,
                    pv_mma_op,
                    self.tmem_o_offset[stage],
                    tOrP[None, None, None, stage],
                    tmem_sa_addr=Int32(self.tmem_sfp_offset),
                    tmem_sb_addr=Int32(self.tmem_sfv_offset),
                    scale_vec=fp4_scale_vec,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.q_stage)
            ]
        else:
            gemm_Pi = [
                partial(
                    sm100_utils.gemm_ptx_partial,
                    pv_mma_op,
                    self.tmem_o_offset[stage],
                    tOrP[None, None, None, stage],
                    sA=None,
                    split_arrive=self.split_P_arrive if self.split_P_arrive > 0 else None,
                    cta_group=self.cta_group_size,
                )
                for stage in range(self.q_stage)
            ]
        corr_tile_size = self.fp4_pv_corr_tile_size
        tOcO_rescale = tiled_mma_pv.get_slice(Int32(0)).partition_C(
            cute.make_identity_tensor(self.mma_tiler_pv[:2])
        )
        tOcO_rescale_i = cute.composition(
            tOcO_rescale,
            cute.make_layout((self.m_block_size, corr_tile_size)),
        )
        tOtO_rescale_i = [
            cute.composition(
                tOtO[None, None, None, stage],
                cute.make_layout((self.m_block_size, corr_tile_size)),
            )
            for stage in range(self.q_stage)
        ]
        tmem_load_atom_rescale = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom_rescale = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        rescale_frg_count = self.head_dim_v_padded // corr_tile_size
        rescale_num_row_quads = self.m_block_size // cute.arch.WARP_SIZE
        # gemm_Pi = [
        #     partial(
        #         sm100_utils.gemm, tOtO[None, None, None, stage], tCrA=tOrP[None, None, None, stage]
        #     )
        #     for stage in range(self.q_stage)
        # ]

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        P_full_O_rescaled_phase = Int32(0)
        s_ready_producer_phase = Int32(0)
        p_ready_consumer_phase = Int32(0)
        sm_stats_consumer_phase = Int32(0)
        mma_lane_idx = cute.arch.lane_idx()
        thr_tmem_load_rescale = [
            [
                tcgen05.make_tmem_copy(
                    tmem_load_atom_rescale,
                    tOtO_rescale_i[stage],
                ).get_slice(row_quad * cute.arch.WARP_SIZE + mma_lane_idx)
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        thr_tmem_store_rescale = [
            [
                tcgen05.make_tmem_copy(
                    tmem_store_atom_rescale,
                    tOtO_rescale_i[stage],
                ).get_slice(row_quad * cute.arch.WARP_SIZE + mma_lane_idx)
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        tOtO_t2r_rescale = [
            [
                thr_tmem_load_rescale[stage][row_quad].partition_S(tOtO_rescale_i[stage])
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        tOtO_r2t_rescale = [
            [
                thr_tmem_store_rescale[stage][row_quad].partition_D(tOtO_rescale_i[stage])
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        tOtO_t2r_rescale_chunks = [
            [
                [
                    cute.make_tensor(
                        tOtO_t2r_rescale[stage][row_quad].iterator + i * corr_tile_size,
                        tOtO_t2r_rescale[stage][row_quad].layout,
                    )
                    for i in range(rescale_frg_count)
                ]
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        tOtO_r2t_rescale_chunks = [
            [
                [
                    cute.make_tensor(
                        tOtO_r2t_rescale[stage][row_quad].iterator + i * corr_tile_size,
                        tOtO_r2t_rescale[stage][row_quad].layout,
                    )
                    for i in range(rescale_frg_count)
                ]
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        tOrO_t2r_shape_rescale = [
            [
                thr_tmem_load_rescale[stage][row_quad].partition_D(tOcO_rescale_i).shape
                for row_quad in range(rescale_num_row_quads)
            ]
            for stage in range(self.q_stage)
        ]
        exact_stage = Int32(0)
        exact_gemm_P = None
        exact_thr_tmem_load_rescale = None
        exact_thr_tmem_store_rescale = None
        exact_tOtO_t2r_rescale_chunks = None
        exact_tOtO_r2t_rescale_chunks = None
        exact_tOrO_t2r_shape_rescale = None
        if const_expr(self.use_exact_fp4_pv_lane):
            exact_gemm_P = gemm_Pi[0]
            exact_thr_tmem_load_rescale = thr_tmem_load_rescale[0]
            exact_thr_tmem_store_rescale = thr_tmem_store_rescale[0]
            exact_tOtO_t2r_rescale_chunks = tOtO_t2r_rescale_chunks[0]
            exact_tOtO_r2t_rescale_chunks = tOtO_r2t_rescale_chunks[0]
            exact_tOrO_t2r_shape_rescale = tOrO_t2r_shape_rescale[0]
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        kv_head_idx = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            kv_head_idx = self.kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)

            block_iter_count = Int32(0)
            process_tile = False

            if const_expr(self.use_block_sparsity):
                block_iter_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                process_tile = block_iter_count > Int32(0)
            else:
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
                block_iter_count = n_block_max - n_block_min
                if const_expr(not self.is_split_kv):
                    process_tile = True
                else:
                    process_tile = n_block_min < n_block_max

            if process_tile and is_leader_cta:
                for stage in cutlass.range_constexpr(self.q_stage):
                        # GEMM_QK00 (Q0 * K0 -> S0) or GEMM_QK01 (Q1 * K0 -> S1)
                        # 1. wait for Q0 / Q1
                        pipeline_q.consumer_wait_w_index_phase(stage, mma_q_consumer_phase)
                        # 2. wait for K0
                        if const_expr(stage == 0):
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        tSrKi = tSrK[None, None, None, Ki_index]
                        # We don't need to acquire empty S0 / S1.
                        # For the first iteration, we don't need to wait as we're guaranteed S0 / S1
                        # are empty. For subsequent iterations, the wait happened at the end
                        # of the while loop.
                        # 3. gemm
                        # sm100_utils.gemm(tiled_mma_qk, tStS[None, None, None, stage], tSrQ[None, None, None, stage], tSrKi, zero_init=True)
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.use_fp4_qk):
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[(None, None, None, None, stage)],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t[(None, None, None, None, Ki_index)],
                                tCtSFK_compact_s2t,
                            )
                        # gemm_Si[stage](tCrB=tSrKi, sB=sK_cur)
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        # gemm_Si[stage](tCrB=tSrKi)
                        # 4. release S0 / S1
                        pipeline_s_p_o.producer_commit_w_index(stage)
                mma_q_consumer_phase ^= 1
                # 5. release K0
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM (Q1 * K0 -> S1)
                # Note: Q0 & Q1 are still needed in the seqlen_kv loop
                # so we need to release them after the seqlen_kv loop

                # O hasn't been accumulated yet, its first MMA calculation doesn't need to accumulate
                block_loop_count = block_iter_count - 1
                for i in cutlass.range(block_loop_count, unroll=1):
                    # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                    # 1. wait for V0
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    mma_kv_release_state = mma_kv_consumer_state.clone()
                    Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                    tOrVi = tOrV[None, None, None, Vi_index]
                    if const_expr(self.use_fp4_pv):
                        cute.copy(
                            tiled_copy_s2t_sfv,
                            tCsSFV_compact_s2t[(None, None, None, None, Vi_index)],
                            tCtSFV_compact_s2t,
                        )
                    if const_expr(self.use_exact_fp4_pv_lane):
                        self.mma_pv_exact_step(
                            pipeline_p_lastsplit=pipeline_p_lastsplit,
                            pipeline_o_acc=pipeline_o_acc,
                            p_ready_consumer_phase=p_ready_consumer_phase,
                            exact_gemm_P=exact_gemm_P,
                            tOrVi=tOrVi,
                            sV_cur=self.offset_kv_smem(sV[None, None, None, Vi_index], Vi_index, Vi_phase)
                            if const_expr(self.uneven_kv_smem)
                            else sV[None, None, None, Vi_index],
                            sAccScale=sAccScale,
                            lane_idx=mma_lane_idx,
                            frg_count=rescale_frg_count,
                            thr_tmem_load_rescale=exact_thr_tmem_load_rescale,
                            tOtO_t2r_rescale_chunks=exact_tOtO_t2r_rescale_chunks,
                            thr_tmem_store_rescale=exact_thr_tmem_store_rescale,
                            tOtO_r2t_rescale_chunks=exact_tOtO_r2t_rescale_chunks,
                            tOrO_t2r_shape_rescale=exact_tOrO_t2r_shape_rescale,
                            zero_init=i == 0,
                            do_rescale=i > 0,
                            commit_o_ready=False,
                        )
                    else:
                        for stage in cutlass.range_constexpr(self.q_stage):
                            # 2. acquire corrected O0/O1_partial and P0 / P1
                            # For the first iteration in this work tile, waiting for O0/O1_partial
                            # means that the correction warps has finished reading tO during
                            # the last iteration of the previous work tile.
                            pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                            if i > 0:
                                self.mma_rescale_tmem_output(
                                    thr_tmem_load_rescale[stage],
                                    tOtO_t2r_rescale_chunks[stage],
                                    thr_tmem_store_rescale[stage],
                                    tOtO_r2t_rescale_chunks[stage],
                                    tOrO_t2r_shape_rescale[stage],
                                    sAccScale,
                                    stage,
                                    mma_lane_idx,
                                    rescale_frg_count,
                                )
                            # 3. gemm
                            sV_cur = sV[None, None, None, Vi_index]
                            if const_expr(self.uneven_kv_smem):
                                sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                            if const_expr(self.use_fp4_pv):
                                cute.copy(
                                    tiled_copy_s2t_sfp,
                                    tCsSFP_compact_s2t[(None, None, None, None, stage)],
                                    tCtSFP_compact_s2t,
                                )
                                gemm_Pi[stage](
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=i == 0,
                                )
                            else:
                                gemm_Pi[stage](
                                    tCrB=tOrVi,
                                    sB=sV_cur,
                                    zero_init=i == 0,
                                    mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                                    mbar_phase=P_full_O_rescaled_phase,
                                )
                    if const_expr(self.use_exact_fp4_pv_lane):
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        mma_kv_release_state.advance()

                        mma_kv_consumer_state.advance()
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.use_fp4_qk):
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[(None, None, None, None, exact_stage)],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t[(None, None, None, None, Ki_index)],
                                tCtSFK_compact_s2t,
                            )
                        pipeline_s_p_o.producer_acquire_w_index_phase(exact_stage, s_ready_producer_phase)
                        gemm_Si[0](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        pipeline_s_p_o.producer_commit_w_index(exact_stage)
                    else:
                        # 4. release V(i-1)
                        if const_expr(stage == self.q_stage - 1):
                            pipeline_kv.consumer_release(mma_kv_release_state)
                            mma_kv_release_state.advance()

                        # GEMM_QK0i (Q0 * Ki -> S0)
                        if const_expr(stage == 0):
                            mma_kv_consumer_state.advance()
                            pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                        sK_cur = sK[None, None, None, Ki_index]
                        if const_expr(self.uneven_kv_smem):
                            sK_cur = self.offset_kv_smem(sK_cur, Ki_index, Ki_phase)
                        if const_expr(self.use_fp4_qk):
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[(None, None, None, None, stage)],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfk,
                                tCsSFK_compact_s2t[(None, None, None, None, Ki_index)],
                                tCtSFK_compact_s2t,
                            )
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(sK_cur.iterator)
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                    # 4. release Ki
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                    if const_expr(self.use_exact_fp4_pv_lane):
                        p_ready_consumer_phase ^= 1
                        s_ready_producer_phase ^= 1
                    else:
                        P_full_O_rescaled_phase ^= 1
                # End of seqlen_kv loop

                # release Q0 & Q1
                for stage in cutlass.range(self.q_stage):
                    pipeline_q.consumer_release_w_index(stage)

                # GEMM_PV00 (P0 * V0 -> O0_partial), O0 needs to be accumulated in the seqlen_kv loop
                # 1. wait for V0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Vi_index, Vi_phase = mma_kv_consumer_state.index, mma_kv_consumer_state.phase
                tOrVi = tOrV[None, None, None, Vi_index]
                if const_expr(self.use_fp4_pv):
                    cute.copy(
                        tiled_copy_s2t_sfv,
                        tCsSFV_compact_s2t[(None, None, None, None, Vi_index)],
                        tCtSFV_compact_s2t,
                    )
                if const_expr(self.use_exact_fp4_pv_lane):
                    self.mma_pv_exact_step(
                        pipeline_p_lastsplit=pipeline_p_lastsplit,
                        pipeline_o_acc=pipeline_o_acc,
                        p_ready_consumer_phase=p_ready_consumer_phase,
                        exact_gemm_P=exact_gemm_P,
                        tOrVi=tOrVi,
                        sV_cur=self.offset_kv_smem(sV[None, None, None, Vi_index], Vi_index, Vi_phase)
                        if const_expr(self.uneven_kv_smem)
                        else sV[None, None, None, Vi_index],
                        sAccScale=sAccScale,
                        lane_idx=mma_lane_idx,
                        frg_count=rescale_frg_count,
                        thr_tmem_load_rescale=exact_thr_tmem_load_rescale,
                        tOtO_t2r_rescale_chunks=exact_tOtO_t2r_rescale_chunks,
                        thr_tmem_store_rescale=exact_thr_tmem_store_rescale,
                        tOtO_r2t_rescale_chunks=exact_tOtO_r2t_rescale_chunks,
                        tOrO_t2r_shape_rescale=exact_tOrO_t2r_shape_rescale,
                        zero_init=block_loop_count == 0,
                        do_rescale=block_loop_count > 0,
                        commit_o_ready=True,
                    )
                else:
                    for stage in cutlass.range_constexpr(self.q_stage):
                        pipeline_s_p_o.producer_acquire_w_index_phase(stage, P_full_O_rescaled_phase)
                        if block_loop_count > 0:
                            self.mma_rescale_tmem_output(
                                thr_tmem_load_rescale[stage],
                                tOtO_t2r_rescale_chunks[stage],
                                thr_tmem_store_rescale[stage],
                                tOtO_r2t_rescale_chunks[stage],
                                tOrO_t2r_shape_rescale[stage],
                                sAccScale,
                                stage,
                                mma_lane_idx,
                                rescale_frg_count,
                            )
                        sV_cur = sV[None, None, None, Vi_index]
                        if const_expr(self.uneven_kv_smem):
                            sV_cur = self.offset_kv_smem(sV_cur, Vi_index, Vi_phase)
                        if const_expr(self.use_fp4_pv):
                            cute.copy(
                                tiled_copy_s2t_sfp,
                                tCsSFP_compact_s2t[(None, None, None, None, stage)],
                                tCtSFP_compact_s2t,
                            )
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=block_loop_count == 0,
                            )
                        else:
                            gemm_Pi[stage](
                                tCrB=tOrVi,
                                sB=sV_cur,
                                zero_init=block_loop_count == 0,
                                mbar_ptr=pipeline_p_lastsplit.sync_object_full.get_barrier(stage) if self.split_P_arrive > 0 else None,
                                mbar_phase=P_full_O_rescaled_phase,
                            )
                        pipeline_o_acc.producer_commit_w_index(stage)
                if const_expr(self.use_exact_fp4_pv_lane):
                    p_ready_consumer_phase ^= 1
                else:
                    P_full_O_rescaled_phase ^= 1
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()
                # End of GEMM_PV1(i_end) (P1 * Vi_end -> O1)

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # pipeline_s_p_o.producer_acquire_w_index_phase(self.q_stage - 1, P_full_O_rescaled_phase)
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    @cute.jit
    def mma_rescale_tmem_output(
        self,
        thr_tmem_load_rescale: list[cute.core.ThrCopy],
        tOtO_t2r_rescale_chunks: list[list[cute.Tensor]],
        thr_tmem_store_rescale: list[cute.core.ThrCopy],
        tOtO_r2t_rescale_chunks: list[list[cute.Tensor]],
        tOrO_t2r_shape_rescale: list,
        sAccScale: cute.Tensor,
        stage: Int32,
        lane_idx: Int32,
        frg_count: Int32 | int,
    ) -> None:
        """Rescale the live TMEM O tile in-place from the MMA warp.

        The block-scaled PV MMA path currently requires TMEM accumulation, so the
        fused recurrence serializes the old correction-warp rescale across the MMA
        warp by replaying the existing per-row rescale helper over the 4 row quads
        of the M tile.
        """
        row_quad_stride = cute.arch.WARP_SIZE
        num_row_quads = self.m_block_size // row_quad_stride
        base = stage * self.m_block_size
        did_rescale = False
        for row_quad in cutlass.range_constexpr(num_row_quads):
            scale_idx = base + row_quad * row_quad_stride + lane_idx
            scale = sAccScale[scale_idx]
            needs_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
            if needs_rescale and scale < 1.0:
                thr_tmem_load = thr_tmem_load_rescale[row_quad]
                thr_tmem_store = thr_tmem_store_rescale[row_quad]
                tOrO_frg = cute.make_fragment(
                    tOrO_t2r_shape_rescale[row_quad],
                    self.pv_acc_dtype,
                )
                for i in cutlass.range_constexpr(frg_count):
                    tOtO_t2r_i = tOtO_t2r_rescale_chunks[row_quad][i]
                    cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
                    for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                        tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                            (tOrO_frg[j], tOrO_frg[j + 1]),
                            (scale, scale),
                        )
                    tOtO_r2t_i = tOtO_r2t_rescale_chunks[row_quad][i]
                    cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
                did_rescale = True
        if did_rescale:
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def mma_rescale_tmem_output_exact(
        self,
        thr_tmem_load_rescale: list[cute.core.ThrCopy],
        tOtO_t2r_rescale_chunks: list[list[cute.Tensor]],
        thr_tmem_store_rescale: list[cute.core.ThrCopy],
        tOtO_r2t_rescale_chunks: list[list[cute.Tensor]],
        tOrO_t2r_shape_rescale: list,
        sAccScale: cute.Tensor,
        lane_idx: Int32,
        frg_count: Int32 | int,
    ) -> None:
        row_quad_stride = cute.arch.WARP_SIZE
        num_row_quads = self.m_block_size // row_quad_stride
        did_rescale = False
        for row_quad in cutlass.range_constexpr(num_row_quads):
            scale_idx = row_quad * row_quad_stride + lane_idx
            scale = sAccScale[scale_idx]
            needs_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
            if needs_rescale and scale < 1.0:
                thr_tmem_load = thr_tmem_load_rescale[row_quad]
                thr_tmem_store = thr_tmem_store_rescale[row_quad]
                tOrO_frg = cute.make_fragment(
                    tOrO_t2r_shape_rescale[row_quad],
                    self.pv_acc_dtype,
                )
                for i in cutlass.range_constexpr(frg_count):
                    tOtO_t2r_i = tOtO_t2r_rescale_chunks[row_quad][i]
                    cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
                    for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                        tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                            (tOrO_frg[j], tOrO_frg[j + 1]),
                            (scale, scale),
                        )
                    tOtO_r2t_i = tOtO_r2t_rescale_chunks[row_quad][i]
                    cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
                did_rescale = True
        if did_rescale:
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def mma_pv_exact_step(
        self,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        p_ready_consumer_phase: Int32,
        exact_gemm_P,
        tOrVi: cute.Tensor,
        sV_cur: cute.Tensor,
        sAccScale: cute.Tensor,
        lane_idx: Int32,
        frg_count: Int32 | int,
        thr_tmem_load_rescale: list[cute.core.ThrCopy],
        tOtO_t2r_rescale_chunks: list[list[cute.Tensor]],
        thr_tmem_store_rescale: list[cute.core.ThrCopy],
        tOtO_r2t_rescale_chunks: list[list[cute.Tensor]],
        tOrO_t2r_shape_rescale: list,
        zero_init: bool | Boolean,
        do_rescale: bool | Boolean,
        commit_o_ready: bool | Boolean,
    ) -> None:
        exact_stage = Int32(0)
        pipeline_p_lastsplit.consumer_wait_w_index_phase(exact_stage, p_ready_consumer_phase)
        if do_rescale:
            self.mma_rescale_tmem_output_exact(
                thr_tmem_load_rescale,
                tOtO_t2r_rescale_chunks,
                thr_tmem_store_rescale,
                tOtO_r2t_rescale_chunks,
                tOrO_t2r_shape_rescale,
                sAccScale,
                lane_idx,
                frg_count,
            )
        exact_gemm_P(
            tCrB=tOrVi,
            sB=sV_cur,
            zero_init=zero_init,
        )
        pipeline_p_lastsplit.consumer_release_w_index(exact_stage)
        if commit_o_ready:
            pipeline_o_acc.producer_commit_w_index(exact_stage)

    def gemm_pv_block_scaled(
        self,
        tiled_mma_pv: cute.TiledMma,
        acc: cute.Tensor,
        tCrA: cute.Tensor,
        tCtSFP: cute.Tensor,
        tCtSFV: cute.Tensor,
        tCrB: cute.Tensor,
        zero_init: bool | Boolean = False,
    ) -> None:
        mma_atom = cute.make_mma_atom(tiled_mma_pv.op)
        mma_atom.set(tcgen05.Field.SFA, tCtSFP.iterator)
        mma_atom.set(tcgen05.Field.SFB, tCtSFV.iterator)
        num_kblocks = cute.size(tCrA, mode=[2])
        for kblock_idx in cutlass.range_constexpr(num_kblocks):
            mma_atom.set(tcgen05.Field.ACCUMULATE, not zero_init or kblock_idx != 0)
            tCrB_k = (
                tCrB[None, None, None, kblock_idx]
                if const_expr(cute.rank(tCrB) == 4)
                else tCrB[None, None, kblock_idx]
            )
            cute.gemm(mma_atom, acc, tCrA[None, None, kblock_idx], tCrB_k, acc)

    @cute.jit
    def load_scale_stage(self, gScale: cute.Tensor, sScale: cute.Tensor):
        gScale = cute.filter_zeros(gScale)
        sScale = cute.filter_zeros(sScale)
        gScale = cute.group_modes(gScale, 0, cute.rank(gScale))
        sScale = cute.group_modes(sScale, 0, cute.rank(sScale))
        # Dense NVFP4 Q/K uses d64/d128 scales, so one whole warp can copy the
        # stage without the generic partial-fill path introduced during PV work.
        tiled_copy_scale = copy_utils.tiled_copy_1d(
            self.fp4_sf_dtype,
            cute.arch.WARP_SIZE,
            num_copy_elems=1,
            is_async=False,
        )
        thr_copy_scale = tiled_copy_scale.get_slice(cute.arch.lane_idx())
        tAgScale = thr_copy_scale.partition_S(gScale)
        tAsScale = thr_copy_scale.partition_D(sScale)
        cute.copy(tiled_copy_scale, tAgScale, tAsScale)
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_scale_stage_layout(
        self,
        gScale: cute.Tensor,
        sScale: cute.Tensor,
        *,
        fill_rest: cutlass.Constexpr[bool] = True,
    ):
        lane_idx = cute.arch.lane_idx()
        gScale = cute.filter_zeros(gScale)
        sScale = cute.filter_zeros(sScale)
        gScale = cute.group_modes(gScale, 0, cute.rank(gScale))
        sScale = cute.group_modes(sScale, 0, cute.rank(sScale))
        num_src = cute.size(gScale.shape)
        num_dst = cute.size(sScale.shape)
        num_copy = min(num_src, num_dst)
        for idx in cutlass.range(lane_idx, num_copy, cute.arch.WARP_SIZE, unroll=1):
            sScale[idx] = gScale[idx]
        if const_expr(fill_rest and num_dst > num_copy):
            one = self.pv_sf_dtype(1.0)
            for idx in cutlass.range(lane_idx + num_copy, num_dst, cute.arch.WARP_SIZE, unroll=1):
                sScale[idx] = one
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def scale_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
        scale_dtype = None,
    ):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        scale_dtype = self.fp4_sf_dtype if const_expr(scale_dtype is None) else scale_dtype
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(cta_group),
            scale_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def fill_scale_stage_constant(
        self,
        sScale_stage: cute.Tensor,
        thread_idx: Int32,
        value: Float32,
    ):
        flat = cute.group_modes(cute.filter_zeros(sScale_stage), 0, cute.rank(sScale_stage))
        scale_value = self.pv_sf_dtype(value)
        # This helper is called from a single elected softmax warp, so the fill
        # stride must be one warp rather than the whole softmax warpgroup.
        for idx in cutlass.range(thread_idx, cute.size(flat.shape), cute.arch.WARP_SIZE, unroll=1):
            flat[idx] = scale_value

    def publish_shared_scale_fill(self):
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def fill_acc_scale_constant(
        self,
        sAccScale: cute.Tensor,
        lane_idx: Int32,
        value: Float32,
    ):
        flat = cute.group_modes(cute.filter_zeros(sAccScale), 0, cute.rank(sAccScale))
        for idx in cutlass.range(lane_idx, cute.size(flat.shape), cute.arch.WARP_SIZE, unroll=1):
            flat[idx] = value
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def make_fp4_pv_grouped_views(
        self,
        tP_conv: cute.Tensor,
        tPc_conv: cute.Tensor,
    ):
        # Match SageAttention3's conversion grouping after the Python DSL's first
        # grouping has already reduced the rank by one.
        tP_conv_groups = cute.group_modes(
            cute.group_modes(cute.flatten(tP_conv), 0, 2),
            1,
            4,
        )
        tPc_conv_groups = cute.group_modes(
            cute.group_modes(cute.flatten(tPc_conv), 0, 2),
            1,
            4,
        )
        return tP_conv_groups, tPc_conv_groups

    @cute.jit
    def reduce_fp4_pv_group_amax_masked(
        self,
        slot_ptr: cutlass.Int64,
        group_max_log2: Float32,
    ):
        """Reduce a masked FP4 PV block amax over all warp lanes sharing one logical scale slot."""
        reduced = group_max_log2
        has_masked_peer = group_max_log2 == -cutlass.Float32.inf
        for peer_lane in cutlass.range_constexpr(cute.arch.WARP_SIZE):
            peer_slot_ptr = cutlass.Int64(utils.shuffle_sync(slot_ptr, offset=peer_lane))
            peer_max_log2 = utils.shuffle_sync(group_max_log2, offset=peer_lane)
            if peer_slot_ptr == slot_ptr:
                reduced = cute.arch.fmax(reduced, peer_max_log2)
                has_masked_peer = has_masked_peer or peer_max_log2 == -cutlass.Float32.inf
        return reduced, has_masked_peer

    @cute.jit
    def online_softmax_with_quant_pv(
        self,
        softmax: SoftmaxFusedNVFP4,
        tSrS_t2r: cute.Tensor,
        tSrP_r2t_f32: cute.Tensor,
        tScS: cute.Tensor,
        sSFP_stage: cute.Tensor,
        *,
        is_first: cutlass.Constexpr[bool],
        use_masked_exp_emu: cutlass.Constexpr[bool],
    ) -> Tuple[cute.Tensor, Float32]:
        """Fused Sage-style online softmax + NVFP4 P quantization.

        This keeps the hot PV path in one recurrence:
        1. update row max / scores_scale
        2. convert the live softmax fragment into grouped pre-exp values
        3. compute per-group NVFP4 scales and quantized P values
        4. update row_sum from the same pre-exp fragment

        The running O recurrence still lives in the MMA/TMEM path, but the
        quantization work is hidden inside the softmax pass instead of being a
        second post-softmax transform.
        """
        del tScS
        sSFP_logical = cute.make_tensor(
            sSFP_stage.iterator,
            bs_layout.tile_atom_to_shape_SF(
                (self.m_block_size, self.n_block_size, 1),
                self.pv_sf_vec_size,
            ),
        )
        sSFP_logical_u8 = cute.make_tensor(
            cute.recast_ptr(sSFP_logical.iterator, dtype=cutlass.Uint8),
            sSFP_logical.layout,
        )
        sSFP_logical_u8_flat = cute.group_modes(
            cute.filter_zeros(sSFP_logical_u8),
            0,
            cute.rank(sSFP_logical_u8),
        )
        acc_scale = self.online_softmax_with_quant_pv_flatview(
            softmax,
            tSrS_t2r,
            tSrP_r2t_f32,
            sSFP_logical_u8_flat,
            is_first=is_first,
            use_masked_exp_emu=use_masked_exp_emu,
        )
        conv_layout = convert_to_conversion_layout(tSrS_t2r.layout)
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.p_dtype),
            conv_layout,
        )
        return tSrP_r2t, acc_scale

    @cute.jit
    def online_softmax_with_quant_pv_flatview(
        self,
        softmax: SoftmaxFusedNVFP4,
        tSrS_t2r: cute.Tensor,
        tSrP_r2t_f32: cute.Tensor,
        sSFP_logical_u8_flat: cute.Tensor,
        *,
        is_first: cutlass.Constexpr[bool],
        use_masked_exp_emu: cutlass.Constexpr[bool],
    ) -> Float32:
        del use_masked_exp_emu
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        conv_layout = convert_to_conversion_layout(tSrS_t2r.layout)
        tSrP_conv_preexp = cute.make_tensor(tSrS_t2r.iterator, conv_layout)
        tSrP_conv_preexp_flat = cute.flatten(tSrP_conv_preexp)
        num_groups = cute.size(tSrP_conv_preexp_flat.shape) // 8
        tSrP_r2t_words = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=cutlass.Uint32),
            cute.make_layout((num_groups,)),
        )
        one_scale_u8 = float_to_ue4m3_byte(Float32(1.0))
        fp4_max = Float32(6.0)

        row_sum_new = Float32(0.0)
        for group_idx in cutlass.range_constexpr(num_groups):
            base = group_idx * 8
            v0 = tSrP_conv_preexp_flat[base + 0]
            v1 = tSrP_conv_preexp_flat[base + 1]
            v2 = tSrP_conv_preexp_flat[base + 2]
            v3 = tSrP_conv_preexp_flat[base + 3]
            v4 = tSrP_conv_preexp_flat[base + 4]
            v5 = tSrP_conv_preexp_flat[base + 5]
            v6 = tSrP_conv_preexp_flat[base + 6]
            v7 = tSrP_conv_preexp_flat[base + 7]

            e0 = cute.math.exp2(v0, fastmath=True)
            e1 = cute.math.exp2(v1, fastmath=True)
            e2 = cute.math.exp2(v2, fastmath=True)
            e3 = cute.math.exp2(v3, fastmath=True)
            e4 = cute.math.exp2(v4, fastmath=True)
            e5 = cute.math.exp2(v5, fastmath=True)
            e6 = cute.math.exp2(v6, fastmath=True)
            e7 = cute.math.exp2(v7, fastmath=True)
            row_sum_new = (
                (((row_sum_new + e0) + e1) + e2)
                + (((e3 + e4) + e5) + (e6 + e7))
            )

            group_max = cute.arch.fmax(
                cute.arch.fmax(cute.arch.fmax(e0, e1), cute.arch.fmax(e2, e3)),
                cute.arch.fmax(cute.arch.fmax(e4, e5), cute.arch.fmax(e6, e7)),
            )
            group_is_masked = group_max == 0.0
            scale_f32 = Float32(1.0) if group_is_masked else group_max / fp4_max
            scale_u8 = one_scale_u8 if group_is_masked else float_to_ue4m3_byte(scale_f32)
            inv_scale = Float32(0.0) if group_is_masked else cute.arch.rcp_approx(scale_f32)
            p0 = Float32(0.0)
            p1 = Float32(0.0)
            p2 = Float32(0.0)
            p3 = Float32(0.0)
            p4 = Float32(0.0)
            p5 = Float32(0.0)
            p6 = Float32(0.0)
            p7 = Float32(0.0)
            if cutlass.const_expr(group_idx < cute.size(sSFP_logical_u8_flat.shape)):
                sSFP_logical_u8_flat[group_idx] = scale_u8
            if not group_is_masked:
                p0 = e0 * inv_scale
                p1 = e1 * inv_scale
                p2 = e2 * inv_scale
                p3 = e3 * inv_scale
                p4 = e4 * inv_scale
                p5 = e5 * inv_scale
                p6 = e6 * inv_scale
                p7 = e7 * inv_scale
            tSrP_r2t_words[group_idx] = pack_float8_to_e2m1_word(
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
            )

        if cutlass.const_expr(not is_first):
            row_sum_new += softmax.row_sum[0] * acc_scale
        softmax.row_sum[0] = row_sum_new

        return acc_scale

    @cute.jit
    def quantize_p_fragment_to_fp4(
        self,
        softmax: SoftmaxSm100,
        tSrS_t2r: cute.Tensor,
        tSrP_r2t_f32: cute.Tensor,
        tScS: cute.Tensor,
        sSFP_stage: cute.Tensor,
        *,
        use_masked_exp_emu: cutlass.Constexpr[bool],
    ) -> cute.Tensor:
        """Quantize a probability fragment to NVFP4 using Sage-style grouped slots."""
        conv_layout = convert_to_conversion_layout(tSrS_t2r.layout)
        tSrP_conv_quant_f32 = cute.make_tensor(tSrP_r2t_f32.iterator, conv_layout)
        tSrP_conv_quant_f32_flat = cute.flatten(tSrP_conv_quant_f32)
        for idx in cutlass.range_constexpr(cute.size(tSrP_conv_quant_f32_flat.shape)):
            tSrP_conv_quant_f32_flat[idx] = Float32(0.0)
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t_f32,
            ex2_emu_freq=self.ex2_emu_freq if const_expr(not use_masked_exp_emu) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        tScP_conv = cute.composition(tScS, conv_layout)
        sSFP_logical = cute.make_tensor(
            sSFP_stage.iterator,
            bs_layout.tile_atom_to_shape_SF(
                (self.m_block_size, self.n_block_size, 1),
                self.pv_sf_vec_size,
            ),
        )
        sSFP_logical_u8 = cute.make_tensor(
            cute.recast_ptr(sSFP_logical.iterator, dtype=cutlass.Uint8),
            sSFP_logical.layout,
        )
        sSFP_logical_u8_flat = cute.group_modes(
            cute.filter_zeros(sSFP_logical_u8),
            0,
            cute.rank(sSFP_logical_u8),
        )
        num_packed_words = cute.size(tSrP_conv_quant_f32_flat) // 8
        one_scale_u8 = float_to_ue4m3_byte(Float32(1.0))
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.p_dtype),
            conv_layout,
        )
        tSrP_r2t_words = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=cutlass.Uint32),
            cute.make_layout((num_packed_words,)),
        )

        if const_expr(self.fp4_pv_debug_uniform_p):
            for idx in cutlass.range_constexpr(cute.size(sSFP_logical_u8_flat.shape)):
                sSFP_logical_u8_flat[idx] = one_scale_u8
            cute.arch.sync_warp()
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            for idx in cutlass.range_constexpr(cute.size(tSrP_conv_quant_f32_flat)):
                tSrP_conv_quant_f32_flat[idx] = Float32(0.5)
            for j in cutlass.range_constexpr(num_packed_words):
                base = j * 8
                p0 = tSrP_conv_quant_f32_flat[base + 0]
                p1 = tSrP_conv_quant_f32_flat[base + 1]
                p2 = tSrP_conv_quant_f32_flat[base + 2]
                p3 = tSrP_conv_quant_f32_flat[base + 3]
                p4 = tSrP_conv_quant_f32_flat[base + 4]
                p5 = tSrP_conv_quant_f32_flat[base + 5]
                p6 = tSrP_conv_quant_f32_flat[base + 6]
                p7 = tSrP_conv_quant_f32_flat[base + 7]
                tSrP_r2t_words[j] = pack_float8_to_e2m1_word(
                    p0,
                    p1,
                    p2,
                    p3,
                    p4,
                    p5,
                    p6,
                    p7,
                )
            return tSrP_r2t

        tSrP_conv_groups, tScP_conv_groups = self.make_fp4_pv_grouped_views(
            tSrP_conv_quant_f32,
            tScP_conv,
        )
        group_size = cute.size(tSrP_conv_groups, mode=[0])
        num_groups = cute.size(tSrP_conv_groups, mode=[1])
        if cutlass.const_expr(group_size == 8):
            for group_idx in cutlass.range_constexpr(num_groups):
                v0 = tSrP_conv_groups[0, group_idx]
                v1 = tSrP_conv_groups[1, group_idx]
                v2 = tSrP_conv_groups[2, group_idx]
                v3 = tSrP_conv_groups[3, group_idx]
                v4 = tSrP_conv_groups[4, group_idx]
                v5 = tSrP_conv_groups[5, group_idx]
                v6 = tSrP_conv_groups[6, group_idx]
                v7 = tSrP_conv_groups[7, group_idx]
                group_max = cute.arch.fmax(
                    cute.arch.fmax(cute.arch.fmax(v0, v1), cute.arch.fmax(v2, v3)),
                    cute.arch.fmax(cute.arch.fmax(v4, v5), cute.arch.fmax(v6, v7)),
                )
                group_is_zero = group_max == 0.0
                scale_f32 = Float32(1.0) if group_is_zero else group_max / Float32(6.0)
                if cutlass.const_expr(group_idx < cute.size(sSFP_logical_u8_flat.shape)):
                    sSFP_logical_u8_flat[group_idx] = float_to_ue4m3_byte(scale_f32)
                if group_is_zero:
                    tSrP_conv_groups[0, group_idx] = Float32(0.0)
                    tSrP_conv_groups[1, group_idx] = Float32(0.0)
                    tSrP_conv_groups[2, group_idx] = Float32(0.0)
                    tSrP_conv_groups[3, group_idx] = Float32(0.0)
                    tSrP_conv_groups[4, group_idx] = Float32(0.0)
                    tSrP_conv_groups[5, group_idx] = Float32(0.0)
                    tSrP_conv_groups[6, group_idx] = Float32(0.0)
                    tSrP_conv_groups[7, group_idx] = Float32(0.0)
                else:
                    inv_scale = cute.arch.rcp_approx(scale_f32)
                    tSrP_conv_groups[0, group_idx] = v0 * inv_scale
                    tSrP_conv_groups[1, group_idx] = v1 * inv_scale
                    tSrP_conv_groups[2, group_idx] = v2 * inv_scale
                    tSrP_conv_groups[3, group_idx] = v3 * inv_scale
                    tSrP_conv_groups[4, group_idx] = v4 * inv_scale
                    tSrP_conv_groups[5, group_idx] = v5 * inv_scale
                    tSrP_conv_groups[6, group_idx] = v6 * inv_scale
                    tSrP_conv_groups[7, group_idx] = v7 * inv_scale
        else:
            tScP_conv_flat = cute.flatten(tScP_conv)
            flat_count = cute.size(tSrP_conv_quant_f32_flat.shape)
            for idx in cutlass.range_constexpr(flat_count):
                coord = tScP_conv_flat[idx]
                row = Int32(coord[0])
                col = Int32(coord[1])
                group_max = tSrP_conv_quant_f32_flat[idx]
                group_max = cute.arch.fmax(group_max, cute.arch.shuffle_sync_bfly(group_max, offset=1))
                group_max = cute.arch.fmax(group_max, cute.arch.shuffle_sync_bfly(group_max, offset=2))
                group_max = cute.arch.fmax(group_max, cute.arch.shuffle_sync_bfly(group_max, offset=4))
                group_max = cute.arch.fmax(group_max, cute.arch.shuffle_sync_bfly(group_max, offset=8))
                group_max = cute.arch.fmax(group_max, cute.arch.shuffle_sync_bfly(group_max, offset=16))
                scale_f32 = Float32(1.0) if group_max == 0.0 else group_max / Float32(6.0)
                inv_scale = Float32(0.0) if group_max == 0.0 else cute.arch.rcp_approx(scale_f32)
                tSrP_conv_quant_f32_flat[idx] = (
                    Float32(0.0)
                    if group_max == 0.0
                    else tSrP_conv_quant_f32_flat[idx] * inv_scale
                )
                sSFP_logical_u8[row, col, 0] = float_to_ue4m3_byte(scale_f32)
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        for j in cutlass.range_constexpr(num_packed_words):
            base = j * 8
            p0 = tSrP_conv_quant_f32_flat[base + 0]
            p1 = tSrP_conv_quant_f32_flat[base + 1]
            p2 = tSrP_conv_quant_f32_flat[base + 2]
            p3 = tSrP_conv_quant_f32_flat[base + 3]
            p4 = tSrP_conv_quant_f32_flat[base + 4]
            p5 = tSrP_conv_quant_f32_flat[base + 5]
            p6 = tSrP_conv_quant_f32_flat[base + 6]
            p7 = tSrP_conv_quant_f32_flat[base + 7]
            tSrP_r2t_words[j] = pack_float8_to_e2m1_word(
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
            )
        return tSrP_r2t

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        thr_mma_qk: cute.core.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        sAccScale: cute.Tensor,
        sSFP: Optional[cute.Tensor],
        tiled_copy_s2t_sfp_exact: Optional[cute.TiledCopy],
        tCsSFP_compact_s2t_exact: Optional[cute.Tensor],
        tCtSFP_compact_s2t_exact: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        learnable_sink: Optional[cute.Tensor],
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE
            # * (len(self.softmax0_warp_ids) if stage == 0 else len(self.softmax1_warp_ids)
            * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.p_dtype.width
        tScP_shape = (cta_qk_tiler[0], tilePlikeFP32)
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)
        tSrS_t2r_shape_exact = thr_tmem_load.partition_D(tScS).shape
        tSrP_r2t_shape_exact = thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape
        tSrS_t2r_exact = None
        tSrP_r2t_f32_exact = None
        sSFP_logical_u8_flat_exact = None
        if const_expr(self.use_exact_fp4_pv_lane):
            tSrS_t2r_exact = cute.make_fragment(tSrS_t2r_shape_exact, self.qk_acc_dtype)
            tSrP_r2t_f32_exact = cute.make_fragment(tSrP_r2t_shape_exact, Float32)
            assert sSFP is not None
            sSFP_stage_exact = sSFP[None, None, None, stage]
            sSFP_logical_exact = cute.make_tensor(
                sSFP_stage_exact.iterator,
                bs_layout.tile_atom_to_shape_SF(
                    (self.m_block_size, self.n_block_size, 1),
                    self.pv_sf_vec_size,
                ),
            )
            sSFP_logical_u8_exact = cute.make_tensor(
                cute.recast_ptr(sSFP_logical_exact.iterator, dtype=cutlass.Uint8),
                sSFP_logical_exact.layout,
            )
            sSFP_logical_u8_flat_exact = cute.group_modes(
                cute.filter_zeros(sSFP_logical_u8_exact),
                0,
                cute.rank(sSFP_logical_u8_exact),
            )

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)
        s0_s1_sequence_phase = Int32(1 if stage == 0 else 0)

        # self.warp_scheduler_barrier_init()

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            q_head_idx = self.q_head_idx(head_idx, split_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            mask = AttentionMaskCls(seqlen)
            shared_mask_kwargs = dict(
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                thr_mma=thr_mma_qk,
                thr_tmem_load=thr_tmem_load,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                batch_idx=batch_idx,
                head_idx=q_head_idx,
                aux_tensors=aux_tensors,
            )

            # Recompute fastdiv_mods if necessary
            recompute_fastdiv_mods_q = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = cutlass.const_expr(
                aux_tensors is not None and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )

            if cutlass.const_expr(fastdiv_mods is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod
                    if not recompute_fastdiv_mods_q
                    else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod
                    if not recompute_fastdiv_mods_k
                    else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask_mod = self.mask_mod if const_expr(self.mask_mod is not None) else None
            mask_fn = partial(
                mask.apply_mask_sm100,
                mask_mod=mask_mod,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
                **shared_mask_kwargs,
            )
            if const_expr(self.use_block_sparsity):
                #  Full blocks dont need mask_mod
                mask_fn_none = partial(
                    mask.apply_mask_sm100,
                    mask_mod=None,
                    fastdiv_mods=fastdiv_mods,
                    head_divmod=head_divmod,
                    **shared_mask_kwargs,
                )
            else:
                mask_fn_none = None

            if const_expr(self.use_fp4_pv):
                softmax = SoftmaxFusedNVFP4.create(
                    softmax_scale_log2,
                    rescale_threshold=8.0 if const_expr(self.o_dtype.width == 16) else 0.0,
                    softmax_scale=softmax_scale,
                )
            else:
                softmax = SoftmaxSm100.create(
                    softmax_scale_log2,
                    rescale_threshold=8.0 if const_expr(self.o_dtype.width == 16) else 0.0,
                    softmax_scale=softmax_scale,
                )
            softmax.reset()

            if const_expr(self.use_block_sparsity):
                tile_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                has_work = tile_block_count > Int32(0)
            else:
                tile_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or tile_block_count > Int32(0)

            thr_tmem_store_scale = None
            tStScale_r2t = None
            if const_expr(not self.use_exact_fp4_pv_lane):
                tStScale = cute.composition(tSAcc, cute.make_layout((self.m_block_size, 1)))
                tmem_store_scale_atom = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), Float32
                )
                thr_tmem_store_scale = tcgen05.make_tmem_copy(tmem_store_scale_atom, tStScale).get_slice(
                    tidx
                )
                tStScale_r2t = thr_tmem_store_scale.partition_D(tStScale)

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                pipeline_s0_s1_sequence=pipeline_s0_s1_sequence,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                thr_tmem_store_scale=thr_tmem_store_scale,
                tScS_exact=tScS,
                tSrS_t2r_shape_exact=tSrS_t2r_shape_exact,
                tSrP_r2t_shape_exact=tSrP_r2t_shape_exact,
                tSrS_t2r_exact=tSrS_t2r_exact,
                tSrP_r2t_f32_exact=tSrP_r2t_f32_exact,
                tStS_t2r=tStS_t2r,
                tStScale_r2t=tStScale_r2t,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                sAccScale=sAccScale,
                sSFP=sSFP,
                sSFP_logical_u8_flat_exact=sSFP_logical_u8_flat_exact,
                tiled_copy_s2t_sfp_exact=tiled_copy_s2t_sfp_exact,
                tCsSFP_compact_s2t_exact=tCsSFP_compact_s2t_exact,
                tCtSFP_compact_s2t_exact=tCtSFP_compact_s2t_exact,
                stage=stage,
                batch_idx=batch_idx,
                head_idx=q_head_idx,
                m_block=(self.q_stage * m_block + stage) * self.cta_group_size,
                seqlen=seqlen,
                learnable_sink=learnable_sink,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
                head_divmod=head_divmod,
            )

            if const_expr(self.use_block_sparsity) or has_work:
                # See block_sparse_utils.py NOTE [SM100 block-sparse empty tiles: mbarrier contract].
                if const_expr(not self.use_fp4_pv):
                    pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
                    sm_stats_producer_phase ^= 1

            # Block sparse or dense iteration
            if const_expr(self.use_block_sparsity):
                # When aux_tensors exist, Q indices beyond seqlen_q must be wrapped to avoid
                # OOB aux_tensor access. Only edge tiles (where m_tile_end > seqlen_q) need this.
                if const_expr(aux_tensors is not None):
                    m_tile_end = ((self.q_stage * m_block + stage + 1) * self.cta_group_size) * self.m_block_size
                    check_m_boundary = m_tile_end > seqlen.seqlen_q
                else:
                    check_m_boundary = False
                (
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    empty_tile,
                ) = softmax_block_sparse_sm100(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_step,
                    mask_fn,
                    mask_fn_none,
                    mma_si_consumer_phase,
                    sm_stats_producer_phase,
                    s0_s1_sequence_phase,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    self.q_stage,
                    Int32(stage),
                    check_m_boundary,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                if not empty_tile:
                    sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        sScale[
                            tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                        ] = softmax.row_max[0]
                    # if tidx == 0:
                    #     cute.printf("softmax row sum stage %d: %f, row_max = %f\n", stage, softmax.row_sum[0], softmax.row_max[0])
                    # See block_sparse_utils.py NOTE [SM100 block-sparse empty tiles: mbarrier contract].
                    # pipeline_sm_stats.producer_commit_w_index(stage)
                    sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
                    # if tidx == 0: cute.printf("softmax row sum stage %d: %f\n", stage, softmax.row_sum[0])
            else:
                if const_expr(not self.is_split_kv) or tile_block_count > Int32(0):
                    mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        s0_s1_sequence_phase,
                        n_block_max - 1,
                        is_first=True,
                        write_final_stats=tile_block_count == Int32(1),
                        mask_fn=partial(mask_fn, mask_seqlen=True),
                    )
                    n_block_max -= 1
                    # Next couple of iterations with causal masking
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(n_block_max - n_block_min_causal_local_mask, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    sm_stats_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    # The remaining iterations have no masking (but may still need mask_mod)
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    for n_tile in cutlass.range(n_block_max - n_block_min_before_local_mask, unroll=1):
                        n_block = n_block_max - n_tile - 1
                        if const_expr(self.mask_mod is not None):
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                                write_final_stats=n_tile + 1 == n_block_max - n_block_min_before_local_mask,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        else:
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = softmax_step(
                                mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase, n_block,
                                write_final_stats=n_tile + 1 == n_block_max - n_block_min_before_local_mask,
                            )
                    # Separate iterations with local masking on the left
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(0, n_block_max - n_block_min, unroll=1):
                            n_block = n_block_max - 1 - n_tile
                            mma_si_consumer_phase, sm_stats_producer_phase, s0_s1_sequence_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    sm_stats_producer_phase,
                                    s0_s1_sequence_phase,
                                    n_block,
                                    mask_fn=partial(mask_fn, mask_seqlen=False),
                                )
                            )
                            # Now that we no longer already have the 1st iteration, need mask_seqlen=True here

                    if const_expr(not self.use_fp4_pv):
                        # Dense non-PV path publishes final row stats to correction.
                        sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            sScale[
                                tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                            ] = softmax.row_max[0]
                        # pipeline_sm_stats.producer_commit_w_index(stage)
                        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # # Write LSE to gmem
            # if const_expr(mLSE is not None):
            #     acc_O_mn_row_is_zero_or_nan = softmax.row_sum[0] == 0.0 or softmax.row_sum[0] != softmax.row_sum[0]
            #     scale = (
            #         cute.arch.rcp_approx(softmax.row_sum[0] if not acc_O_mn_row_is_zero_or_nan else 1.0)
            #     )
            #     LN2 = math.log(2.0)
            #     lse = (
            #         (softmax.row_max[0] * softmax.scale_log2 + cute.math.log2(softmax.row_sum[0], fastmath=True)) * LN2
            #         if not acc_O_mn_row_is_zero_or_nan else -Float32.inf
            #     )
            #     if const_expr(not seqlen.has_cu_seqlens_q):
            #         mLSE_cur = mLSE[None, head_idx, batch_idx]
            #     else:
            #         mLSE_cur = cute.domain_offset((seqlen.offset_q,), mLSE[None, head_idx])
            #     gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_block * 2 + stage,))
            #     if tidx < seqlen.seqlen_q - (m_block * 2 + stage) * self.m_block_size:
            #         gLSE[tidx] = lse

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        if const_expr(not self.use_fp4_pv):
            pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        # This is equivalent to pipeline_s0_s1.producer_tail
        if const_expr(self.s0_s1_barrier):
            if stage == 0:
                pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)

    @cute.jit
    def softmax_step_non_pv(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        sAccScale: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        sSFP_logical_u8_flat_exact: cute.Tensor,
        tiled_copy_s2t_sfp_exact: Optional[cute.TiledCopy],
        tCsSFP_compact_s2t_exact: Optional[cute.Tensor],
        tCtSFP_compact_s2t_exact: Optional[cute.Tensor],
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        learnable_sink: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        write_final_stats: bool | Boolean = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        del thr_tmem_store_scale, tStScale_r2t, sAccScale, sSFP, learnable_sink
        del tiled_copy_s2t_sfp_exact, tCsSFP_compact_s2t_exact, tCtSFP_compact_s2t_exact
        del write_final_stats

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler
        tScP_shape = (tScS_shape[0], tilePlikeFP32)

        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.v_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1, s0_s1_sequence_phase ^ 1

    @cute.jit
    def softmax_step_exact_pv(
        self,
        mma_si_consumer_phase: Int32,
        p_ready_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tScS_exact: cute.Tensor,
        tSrS_t2r_shape_exact,
        tSrP_r2t_shape_exact,
        tSrS_t2r_exact: cute.Tensor,
        tSrP_r2t_f32_exact: cute.Tensor,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        sAccScale: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        sSFP_logical_u8_flat_exact: Optional[cute.Tensor],
        tiled_copy_s2t_sfp_exact: Optional[cute.TiledCopy],
        tCsSFP_compact_s2t_exact: Optional[cute.Tensor],
        tCtSFP_compact_s2t_exact: Optional[cute.Tensor],
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        learnable_sink: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        write_final_stats: bool | Boolean = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        del learnable_sink
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r_exact)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r_exact,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r_exact, n_block=n_block)
        # The exact lane only needs S until the local fragment is copied, so release
        # the S slot before running the fused softmax + quant path.
        pipeline_s_p_o.consumer_release_w_index(stage)
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        assert sSFP is not None
        pipeline_p_lastsplit.producer_acquire_w_index_phase(stage, p_ready_producer_phase)
        acc_scale = self.online_softmax_with_quant_pv_flatview(
            softmax,
            tSrS_t2r_exact,
            tSrP_r2t_f32_exact,
            sSFP_logical_u8_flat_exact,
            is_first=is_first,
            use_masked_exp_emu=mask_fn is not None,
        )
        thread_idx = thr_tmem_load.thr_idx
        if const_expr(not is_first):
            assert sAccScale is not None
            sAccScale[thread_idx + stage * self.m_block_size] = acc_scale
        if write_final_stats:
            sScale[thread_idx + stage * self.m_block_size] = softmax.row_sum[0]
            sScale[
                thread_idx + stage * self.m_block_size + self.q_stage * self.m_block_size
            ] = softmax.row_max[0]
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        assert tiled_copy_s2t_sfp_exact is not None
        assert tCsSFP_compact_s2t_exact is not None and tCtSFP_compact_s2t_exact is not None
        cute.copy(
            tiled_copy_s2t_sfp_exact,
            tCsSFP_compact_s2t_exact[(None, None, None, None, stage)],
            tCtSFP_compact_s2t_exact,
        )
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32_exact[None, None, i], tStP_r2t[None, None, i])
        cute.arch.fence_view_async_tmem_store()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_p_lastsplit.producer_commit_w_index(stage)
        return (
            mma_si_consumer_phase ^ 1,
            p_ready_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        s0_s1_sequence_phase: Int32,
        n_block: Int32,
        softmax: SoftmaxSm100,
        thr_mma_qk: cute.core.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_s0_s1_sequence: Optional[pipeline.PipelineAsync],
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        thr_tmem_store_scale: cute.CopyAtom,
        tScS_exact: Optional[cute.Tensor],
        tSrS_t2r_shape_exact,
        tSrP_r2t_shape_exact,
        tSrS_t2r_exact: Optional[cute.Tensor],
        tSrP_r2t_f32_exact: Optional[cute.Tensor],
        tStS_t2r: cute.Tensor,
        tStScale_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        sAccScale: Optional[cute.Tensor],
        sSFP: Optional[cute.Tensor],
        sSFP_logical_u8_flat_exact: Optional[cute.Tensor],
        tiled_copy_s2t_sfp_exact: Optional[cute.TiledCopy],
        tCsSFP_compact_s2t_exact: Optional[cute.Tensor],
        tCtSFP_compact_s2t_exact: Optional[cute.Tensor],
        stage: int | Int32,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen,
        learnable_sink: Optional[cute.Tensor] = None,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=(None, None),
        head_divmod=None,
        mask_fn: Optional[Callable] = None,
        is_first: bool = False,
        write_final_stats: bool | Boolean = False,
    ) -> Tuple[cute.Int32, cute.Int32, cute.Int32]:
        if const_expr(self.use_exact_fp4_pv_lane):
            assert tScS_exact is not None
            assert sSFP_logical_u8_flat_exact is not None
            assert tSrS_t2r_exact is not None and tSrP_r2t_f32_exact is not None
            return self.softmax_step_exact_pv(
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                s0_s1_sequence_phase,
                n_block,
                softmax,
                thr_mma_qk,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_s0_s1_sequence,
                thr_tmem_load,
                thr_tmem_store,
                tScS_exact,
                tSrS_t2r_shape_exact,
                tSrP_r2t_shape_exact,
                tSrS_t2r_exact,
                tSrP_r2t_f32_exact,
                tStS_t2r,
                tStP_r2t,
                sScale,
                sAccScale,
                sSFP,
                sSFP_logical_u8_flat_exact,
                tiled_copy_s2t_sfp_exact,
                tCsSFP_compact_s2t_exact,
                tCtSFP_compact_s2t_exact,
                stage,
                batch_idx,
                head_idx,
                m_block,
                seqlen,
                learnable_sink,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
                mask_fn,
                is_first,
                write_final_stats,
            )
        if const_expr(not self.use_fp4_pv):
            return self.softmax_step_non_pv(
                mma_si_consumer_phase,
                sm_stats_producer_phase,
                s0_s1_sequence_phase,
                n_block,
                softmax,
                thr_mma_qk,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_s0_s1_sequence,
                thr_tmem_load,
                thr_tmem_store,
                thr_tmem_store_scale,
                tStS_t2r,
                tStScale_r2t,
                tStP_r2t,
                sScale,
                sAccScale,
                sSFP,
                stage,
                batch_idx,
                head_idx,
                m_block,
                seqlen,
                learnable_sink,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
                mask_fn,
                is_first,
                write_final_stats,
            )
        del learnable_sink
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.p_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape, self.mma_tiler_qk[1])
        tScS_shape = cta_qk_tiler
        tScP_shape = (tScS_shape[0], tilePlikeFP32)

        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_fragment(thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
        if cutlass.const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )

        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)
        tSrP_r2t_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape, Float32
        )
        assert sSFP is not None
        tSrP_r2t, acc_scale = self.online_softmax_with_quant_pv(
            softmax,
            tSrS_t2r,
            tSrP_r2t_f32,
            tScS,
            sSFP[None, None, None, stage],
            is_first=is_first,
            use_masked_exp_emu=mask_fn is not None,
        )
        thread_idx = thr_tmem_load.thr_idx
        if const_expr(not is_first):
            assert sAccScale is not None
            sAccScale[thread_idx + stage * self.m_block_size] = acc_scale
        if write_final_stats:
            sScale[thread_idx + stage * self.m_block_size] = softmax.row_sum[0]
            sScale[
                thread_idx + stage * self.m_block_size + self.q_stage * self.m_block_size
            ] = softmax.row_max[0]
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i])
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = cute.size(tStP_r2t.shape[2]) * self.split_P_arrive // self.n_block_size
                if const_expr(i + 1 == split_P_arrive_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        return (
            mma_si_consumer_phase ^ 1,
            sm_stats_producer_phase,
            s0_s1_sequence_phase ^ 1,
        )

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        if const_expr(self.use_exact_fp4_pv_lane):
            return self.correction_loop_exact_pv(
                thr_mma_qk,
                thr_mma_pv,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_o_acc,
                pipeline_o_epi,
                gmem_tiled_copy_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        if const_expr(not self.use_fp4_pv):
            return self.correction_loop_non_pv(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mO,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                learnable_sink,
                gmem_tiled_copy_O,
                tma_atom_O,
                softmax_scale_log2,
                block_info,
                num_splits,
                SeqlenInfoCls,
                TileSchedulerCls,
                blocksparse_tensors,
            )

        del tma_atom_O

        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            q_head_idx = self.q_head_idx(head_idx, split_idx)
            kv_head_idx = self.kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.pack_gqa_local):
                assert not self.is_split_kv
                assert not seqlen.has_cu_seqlens_q
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)
                gO = None
            elif const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx, split_idx]
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx]
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa_effective):
                        sink_val = Float32(learnable_sink[q_head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                ((m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])

                for stage in cutlass.range_constexpr(self.q_stage):
                    if const_expr(self.use_fp4_pv):
                        pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                        row_sum = sScale[tidx + stage * self.m_block_size]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            row_max = sScale[
                                tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                            ]
                        else:
                            row_max = None
                    else:
                        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                        row_sum = sScale[tidx + stage * self.m_block_size]
                        if const_expr(mLSE is not None or learnable_sink is not None):
                            row_max = sScale[
                                tidx + stage * self.m_block_size + self.q_stage * self.m_block_size
                            ]
                        else:
                            row_max = None
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2, fastmath=True
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO[None, None, stage] if const_expr(not self.pack_gqa_local) else None,
                        gmem_tiled_copy_O,
                        kv_head_idx=kv_head_idx,
                    )
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)

                o_corr_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                if const_expr(self.use_block_sparsity):
                    (
                        _,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa_effective,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtO,
                        sO,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        pipeline_o_epi,
                        Int32(0),
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(self.pack_gqa_local):
                    assert not seqlen.has_cu_seqlens_q
                    assert not self.is_split_kv
                    mLSE_cur = mLSE[None, None, batch_idx]
                elif const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, q_head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, q_head_idx, batch_idx]
                else:
                    offset = seqlen.offset_q
                    if const_expr(self.pack_gqa_effective):
                        offset = (
                            (seqlen.offset_q, 0)
                            if const_expr(self.pack_gqa_seqmajor)
                            else (0, seqlen.offset_q)
                        )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, q_head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if const_expr(self.pack_gqa_local):
                        packed_row = m_tile_idx * self.m_block_size + tidx
                        row_limit = seqlen.seqlen_q * self.qhead_per_kvhead
                        if packed_row < row_limit:
                            seqlen_idx = packed_row // self.qhead_per_kvhead
                            subhead_idx = packed_row - seqlen_idx * self.qhead_per_kvhead
                            lse_head_idx = head_idx * self.qhead_per_kvhead + subhead_idx
                            mLSE_cur[seqlen_idx, lse_head_idx] = lse
                    else:
                        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                        seqlen_q = (
                            seqlen.seqlen_q
                            if const_expr(not self.pack_gqa_effective)
                            else seqlen.seqlen_q * self.qhead_per_kvhead
                        )
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            gLSE[tidx] = lse

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_loop_non_pv(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        learnable_sink: Optional[cute.Tensor],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
    ):
        del tma_atom_O

        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        mma_tile_coord_v = thr_mma_qk.thr_idx

        tStScales_t2r = None
        tSrScale_t2r_shape = None
        if const_expr(not self.use_exact_fp4_pv_lane):
            tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
            tStScale_layout = cute.composition(tStS.layout, cute.make_layout((self.m_block_size, 1)))
            tStScales = tuple(
                cute.make_tensor(tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout)
                for stage in range(self.q_stage)
            )
            tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
            tmem_load_v_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
            )
            thr_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStScales[0]).get_slice(tidx)

            tStScales_t2r = [thr_tmem_load_vec.partition_S(tStScales[stage]) for stage in range(self.q_stage)]
            tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        if const_expr(not self.use_exact_fp4_pv_lane):
            for stage in cutlass.range(self.q_stage):
                pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            q_head_idx = self.q_head_idx(head_idx, split_idx)
            kv_head_idx = self.kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(self.pack_gqa_local):
                assert not self.is_split_kv
                assert not seqlen.has_cu_seqlens_q
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)
                gO = None
            elif const_expr(self.is_split_kv):
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx, split_idx]
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]
            else:
                mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx]
                tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
                gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
                gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

            stats = [(0.0, -Float32.inf if const_expr(mLSE is not None or learnable_sink is not None) else None, True)] * self.q_stage

            if const_expr(self.use_block_sparsity):
                total_block_count = get_total_block_count(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    self.qhead_per_kvhead if const_expr(self.pack_gqa_effective) else 1,
                    self.q_subtile_factor if self.q_subtile_factor is not None else 1,
                )
                has_work = total_block_count > Int32(0)
            else:
                total_block_count = n_block_max - n_block_min
                has_work = const_expr(not self.is_split_kv) or total_block_count > Int32(0)

            if has_work:
                if const_expr(not self.use_fp4_pv):
                    sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                    pipeline_sm_stats.consumer_release_w_index(0)
                    if const_expr(self.q_stage == 2):
                        sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                    sm_stats_consumer_phase ^= 1

                    tSrScale_t2r = cute.make_fragment(tSrScale_t2r_shape, Float32)
                    for _i in cutlass.range(total_block_count - 1, unroll=1):
                        for stage in cutlass.range_constexpr(self.q_stage):
                            sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                            scale = sScale[tidx + stage * self.m_block_size]
                            should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                            if should_rescale:
                                self.correction_rescale(thr_mma_pv, tOtO[None, None, None, stage], tidx, scale)
                            pipeline_s_p_o.consumer_release_w_index(stage)
                            pipeline_sm_stats.consumer_release_w_index(self.q_stage - 1 - stage)
                        sm_stats_consumer_phase ^= 1
                    if const_expr(self.q_stage == 2):
                        pipeline_sm_stats.consumer_release_w_index(1)
                else:
                    # The fused PV path releases the P/O slot directly from the MMA
                    # warp after consuming each P tile, so correction no longer owns
                    # the per-iteration UMMA phase progression.
                    pass

                learnable_sink_val = [None] * self.q_stage
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa_effective):
                        sink_val = Float32(learnable_sink[q_head_idx])
                        learnable_sink_val = [sink_val] * self.q_stage
                    else:
                        for stage in cutlass.range_constexpr(self.q_stage):
                            q_head_idx = (
                                ((m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v) * self.m_block_size + tidx
                            ) % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                            learnable_sink_val[stage] = Float32(learnable_sink[q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    if const_expr(self.use_exact_fp4_pv_lane):
                        # On the exact fused lane, the final row stats are written by softmax
                        # before the last P-ready publish, so waiting for the final O-ready
                        # handoff is enough to make the row_sum / row_max read deterministic.
                        pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    else:
                        sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[tidx + stage * self.m_block_size + self.q_stage * self.m_block_size]
                    else:
                        row_max = None
                    if const_expr(not self.use_fp4_pv):
                        pipeline_sm_stats.consumer_release_w_index(stage)
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if const_expr(not self.is_split_kv) or split_idx == 0:
                            if row_max == -Float32.inf:
                                row_max = sink_val * (LOG2_E / softmax_scale_log2)
                                row_sum = Float32(1.0)
                            else:
                                row_sum += cute.math.exp2(
                                    sink_val * LOG2_E - row_max * softmax_scale_log2, fastmath=True
                                )
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)
                    if const_expr(not self.use_exact_fp4_pv_lane):
                        pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen.seqlen_q,
                        scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO[None, None, stage] if const_expr(not self.pack_gqa_local) else None,
                        gmem_tiled_copy_O,
                        kv_head_idx=kv_head_idx,
                    )
                    if const_expr(not self.use_fp4_pv):
                        pipeline_s_p_o.consumer_release_w_index(stage)
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)

                o_corr_consumer_phase ^= 1
                if const_expr(not self.use_fp4_pv):
                    sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                gmem_tiled_copy_O_for_empty_tile = None
                if const_expr(self.use_correction_warps_for_epi):
                    gmem_tiled_copy_O_for_empty_tile = gmem_tiled_copy_O
                if const_expr(self.use_block_sparsity):
                    (
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                    ) = handle_block_sparse_empty_tile_correction_sm100(
                        tidx,
                        self.q_stage,
                        self.m_block_size,
                        self.qhead_per_kvhead,
                        self.pack_gqa_effective,
                        self.is_split_kv,
                        learnable_sink,
                        mLSE,
                        seqlen,
                        m_block,
                        head_idx,
                        batch_idx,
                        split_idx,
                        sScale,
                        stats,
                        self.correction_epilogue,
                        thr_mma_pv,
                        tOtO,
                        sO,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        pipeline_o_epi,
                        sm_stats_consumer_phase,
                        o_corr_consumer_phase,
                        corr_epi_producer_phase,
                        softmax_scale_log2,
                        mO_cur,
                        gO,
                        gmem_tiled_copy_O_for_empty_tile,
                    )

            if const_expr(mLSE is not None):
                if const_expr(self.pack_gqa_local):
                    assert not seqlen.has_cu_seqlens_q
                    assert not self.is_split_kv
                    mLSE_cur = mLSE[None, None, batch_idx]
                elif const_expr(not seqlen.has_cu_seqlens_q):
                    if const_expr(self.is_split_kv):
                        mLSE_cur = mLSE[None, q_head_idx, batch_idx, split_idx]
                    else:
                        mLSE_cur = mLSE[None, q_head_idx, batch_idx]
                else:
                    offset = seqlen.offset_q
                    if const_expr(self.pack_gqa_effective):
                        offset = (
                            (seqlen.offset_q, 0)
                            if const_expr(self.pack_gqa_seqmajor)
                            else (0, seqlen.offset_q)
                        )
                    if const_expr(self.is_split_kv):
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, q_head_idx, split_idx])
                    else:
                        mLSE_cur = cute.domain_offset((offset,), mLSE[None, q_head_idx])
                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (row_max * softmax_scale_log2 + cute.math.log2(row_sum, fastmath=True)) * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    if const_expr(self.pack_gqa_local):
                        packed_row = m_tile_idx * self.m_block_size + tidx
                        row_limit = seqlen.seqlen_q * self.qhead_per_kvhead
                        if packed_row < row_limit:
                            seqlen_idx = packed_row // self.qhead_per_kvhead
                            subhead_idx = packed_row - seqlen_idx * self.qhead_per_kvhead
                            lse_head_idx = head_idx * self.qhead_per_kvhead + subhead_idx
                            mLSE_cur[seqlen_idx, lse_head_idx] = lse
                    else:
                        gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                        seqlen_q = (
                            seqlen.seqlen_q
                            if const_expr(not self.pack_gqa_effective)
                            else seqlen.seqlen_q * self.qhead_per_kvhead
                        )
                        if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                            gLSE[tidx] = lse

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_loop_exact_pv(
        self,
        thr_mma_qk: cute.core.ThrMma,
        thr_mma_pv: cute.core.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_o_epi: pipeline.PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        softmax_scale_log2: Float32,
        block_info: BlockInfo,
        num_splits: Int32,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.correction_warp_ids))
        mma_tile_coord_v = thr_mma_qk.thr_idx
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)
        softmax_scale_log2_local = softmax_scale_log2

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            q_head_idx = self.q_head_idx(head_idx, split_idx)
            kv_head_idx = self.kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            # The exact fused PV lane is currently scoped to dense fixed-length
            # noncausal MHA on the must-win row. Keep its correction path just
            # as narrow: one stage, no SplitKV/GQA/local/block-sparse handling.
            assert not self.pack_gqa_local
            assert not self.is_split_kv
            assert not self.use_block_sparsity
            assert n_block_min < n_block_max

            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1])
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

            stage = Int32(0)
            pipeline_o_acc.consumer_wait_w_index_phase(stage, o_corr_consumer_phase)
            row_sum = sScale[tidx]
            row_max = sScale[tidx + self.m_block_size] if const_expr(mLSE is not None) else None
            acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
            scale = cute.arch.rcp_approx(row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0)

            if const_expr(not self.use_correction_warps_for_epi):
                pipeline_o_epi.producer_acquire_w_index_phase(stage, corr_epi_producer_phase)
            self.correction_epilogue(
                thr_mma_pv,
                tOtO[None, None, None, stage],
                tidx,
                stage,
                m_block,
                seqlen.seqlen_q,
                scale,
                sO[None, None, stage],
                mO_cur,
                gO[None, None, stage],
                gmem_tiled_copy_O,
                kv_head_idx=kv_head_idx,
            )
            if const_expr(not self.use_correction_warps_for_epi):
                pipeline_o_epi.producer_commit_w_index(stage)

            if const_expr(mLSE is not None):
                mLSE_cur = mLSE[None, q_head_idx, batch_idx]
                m_tile_idx = m_block * self.cta_group_size + mma_tile_coord_v
                LN2 = math.log(2.0)
                lse = (
                    (row_max * softmax_scale_log2_local + cute.math.log2(row_sum, fastmath=True)) * LN2
                    if not acc_O_mn_row_is_zero_or_nan
                    else -Float32.inf
                )
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                if tidx < seqlen.seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse

            o_corr_consumer_phase ^= 1
            corr_epi_producer_phase ^= 1
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        if const_expr(not self.use_correction_warps_for_epi):
            pipeline_o_epi.producer_acquire_w_index_phase(self.q_stage - 1, corr_epi_producer_phase)

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
        do_fence: bool | Boolean = True,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = self.fp4_pv_corr_tile_size
        tOtO_i = cute.composition(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)), self.pv_acc_dtype
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i).get_slice(tidx)
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i).get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
        tOrO_t2r_shape = thr_tmem_load.partition_D(tOcO_i).shape
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)

        frg_count = self.head_dim_v_padded // corr_tile_size
        tOrO_frg = cute.make_fragment((tOrO_t2r_shape, frg_count), self.pv_acc_dtype)
        for i in cutlass.range_constexpr(frg_count):
            tOrO_frg = cute.make_fragment(tOrO_t2r_shape, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(tOtO_t2r.iterator + i * corr_tile_size, tOtO_t2r.layout)
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            tOtO_r2t_i = cute.make_tensor(tOtO_r2t.iterator + i * corr_tile_size, tOtO_r2t.layout)
            cute.copy(thr_tmem_store, tOrO_frg, tOtO_r2t_i)
        if do_fence:
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        stage: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        scale: Float32,
        sO: cute.Tensor,
        mO_cur: Optional[cute.Tensor] = None,
        gO: Optional[cute.Tensor] = None,
        gmem_tiled_copy_O: Optional[cute.TiledCopy] = None,
        kv_head_idx: Optional[Int32] = None,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        This correction_epilogue function handles the final processing step for attention output values.
        It applies a scaling factor to the accumulated attention results and prepares the
        data for efficient transfer back to global memory.

        The method performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion if necessary (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations

        :param thr_mma: Thread MMA operation for the computation
        :type thr_mma: cute.core.ThrMma
        :param tOtO: Tensor containing accumulated attention output
        :type tOtO: cute.Tensor
        :param scale: Final scaling factor to apply to the output
        :type scale: Float32
        :param sO: Shared memory tensor for the final output
        :type sO: cute.Tensor
        """
        corr_tile_size = 8 * 32 // self.o_dtype.width
        # Use CTA 0 mapping for smem partitioning since sO is per-CTA sized
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((self.m_block_size, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((self.m_block_size, corr_tile_size)))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=self.use_2cta_instrs,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_i[(None, None), 0])
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(thr_tmem_load, tOsO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(self.head_dim_v_padded // corr_tile_size, unroll_full=True):
            tOtO_t2r_i = tOtO_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            tOrO_frg = cute.make_fragment(tOcO_t2r[None, 0, 0, i].shape, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load, tOtO_t2r_i, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]), (scale, scale)
                )
            copy_utils.cvt_copy(tiled_smem_store, tOrO_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

        if const_expr(self.use_correction_warps_for_epi):
            assert(not self.use_tma_O)
            assert(gmem_tiled_copy_O is not None)
            cute.arch.barrier(barrier_id=int(NamedBarrierFwdSm100.Epilogue),
                              number_of_threads=len(self.epilogue_warp_ids) * cute.arch.WARP_SIZE)
            mma_tile_coord_v = thr_mma.thr_idx
            m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
            self._store_O_to_gmem(
                sO, gO, mO_cur, gmem_tiled_copy_O, tidx, seqlen_q, m_tile_idx, kv_head_idx=kv_head_idx
            )

    @cute.jit
    def _store_O_to_gmem(
        self,
        sO_stage: cute.Tensor,
        gO: Optional[cute.Tensor],
        mO_cur: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tidx: Int32,
        seqlen_q: Int32,
        m_tile_idx: Int32,
        kv_head_idx: Optional[Int32] = None,
    ):
        """Copy a single stage of O from smem to gmem via registers."""
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO_stage)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_v_padded))
        pack_gqa = PackGQA(
            self.m_block_size,
            self.head_dim_v_padded,
            self.check_hdim_v_oob,
            self.qhead_per_kvhead,
            seqmajor_layout=self.pack_gqa_seqmajor,
        )

        # load acc O from smem to rmem for wider vectorization
        tOrO = cute.make_fragment_like(tOsO, self.o_dtype)
        cute.autovec_copy(tOsO, tOrO)
        # copy acc O from rmem to gmem
        if const_expr(self.pack_gqa_local):
            assert kv_head_idx is not None
            pack_gqa.store_O_unpacked(
                mO_cur, kv_head_idx, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )
        elif const_expr(not self.pack_gqa_effective):
            assert gO is not None
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = copy_utils.predicate_k(tOcO, limit=mO_cur.shape[1])
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0] < seqlen_q - m_tile_idx * self.m_block_size - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_tile_idx, seqlen_q
            )

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_o_epi: pipeline.PipelineAsync,
        block_info: BlockInfo,
        num_splits: int,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mma_tile_coord_v: Int32 = 0,
    ):
        epi_consumer_phase = Int32(0)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            q_head_idx = self.q_head_idx(head_idx, split_idx)
            kv_head_idx = self.kv_head_idx(head_idx)
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)

            if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                if const_expr(self.pack_gqa_local):
                    assert not self.is_split_kv
                    assert not seqlen.has_cu_seqlens_q
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)
                    gO = None
                elif const_expr(self.is_split_kv):
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx, split_idx]
                    tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                    gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2, 128)
                    gO = layout_utils.select(
                        cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                    )  # (128, 128, 2)
                    gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]
                else:
                    mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3)[None, None, q_head_idx]
                    tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
                    gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128 * 2, 128)
                    gO = layout_utils.select(
                        cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
                    )  # (128, 128, 2)
                    gO = cute.flat_divide(gO, (self.mma_tiler_pv[0] // self.cta_group_size,))[None, mma_tile_coord_v, None, None]

                if const_expr(self.use_tma_O):
                    store_O, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_O, 0, cute.make_layout(1), sO, gO
                    )
                    for stage in cutlass.range(self.q_stage, unroll_full=True):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                        # 2. copy O0 / O1 to gmem
                        store_O(src_idx=stage, dst_idx=stage)
                        cute.arch.cp_async_bulk_commit_group()
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Ensure O0 / O1 buffer is ready to be released
                        cute.arch.cp_async_bulk_wait_group(self.q_stage - 1 - stage, read=True)
                        pipeline_o_epi.consumer_release_w_index(stage)
                else:
                    tidx = cute.arch.thread_idx()[0] % (
                        cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
                    )
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # wait from corr, issue tma store on smem
                        # 1. wait for O0 / O1 final
                        pipeline_o_epi.consumer_wait_w_index_phase(stage, epi_consumer_phase)
                        # 2. copy O0 / O1 to gmem
                        m_tile_idx = (m_block * self.q_stage + stage) * self.cta_group_size + mma_tile_coord_v
                        self._store_O_to_gmem(
                            sO[None, None, stage],
                            gO[None, None, stage] if const_expr(not self.pack_gqa_local) else None,
                            mO_cur,
                            gmem_tiled_copy_O,
                            tidx, seqlen.seqlen_q, m_tile_idx, kv_head_idx=kv_head_idx,
                        )
                        pipeline_o_epi.consumer_release_w_index(stage)

                epi_consumer_phase ^= 1

            # Advance to next tile
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    def load_Q(
        self,
        load_Q_fn: Callable,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: int,
        phase: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        load_Q_fn(src_idx=block, dst_idx=stage, tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(stage))

    @cute.jit
    def load_Q_packed_local(
        self,
        pack_gqa: PackGQA,
        mQ_storage_cur: cute.Tensor,
        kv_head_idx: Int32,
        sQ_stage: cute.Tensor,
        gmem_tiled_copy_Q: cute.TiledCopy,
        pipeline_q: pipeline.PipelineAsync,
        block: Int32,
        stage: Int32,
        phase: Int32,
        seqlen: Int32,
    ):
        pipeline_q.producer_acquire_w_index_phase(stage, phase)
        sQ_stage = cute.group_modes(sQ_stage, 0, cute.rank(sQ_stage) - 1)
        sQ_stage_bytes = cute.make_tensor(
            cute.recast_ptr(sQ_stage.iterator, dtype=self.q_storage_dtype),
            cute.recast_layout(self.q_storage_dtype.width, sQ_stage.element_type.width, sQ_stage.layout),
        )
        pack_gqa.load_Q_unpacked_bytes(
            mQ_storage_cur,
            kv_head_idx,
            sQ_stage_bytes,
            gmem_tiled_copy_Q,
            cute.arch.lane_idx(),
            block,
            seqlen,
        )
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_scale_packed_local(
        self,
        pack_gqa: PackGQA,
        mQ_scale_cur: cute.Tensor,
        sSFA_stage: cute.Tensor,
        gmem_tiled_copy_Q_scale: cute.TiledCopy,
        block: Int32,
        seqlen: Int32,
    ):
        sSFA_stage = cute.group_modes(sSFA_stage, 0, cute.rank(sSFA_stage) - 1)
        pack_gqa.load_tensor(
            mQ_scale_cur,
            sSFA_stage,
            gmem_tiled_copy_Q_scale,
            cute.arch.lane_idx(),
            block,
            seqlen,
        )
        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_v_scale_stage_public(
        self,
        mV_scale: cute.Tensor,  # public colwise-SFVt storage: (b, h_k, d, s_k // sf_vec)
        batch_idx: Int32,
        kv_head_idx: Int32,
        sSFV_stage: cute.Tensor,
        block: Int32,
        seqlen_k: Int32,
    ):
        # FP4-PV consumes pretransposed colwise SFVt in the same swizzled GMEM
        # layout used by TK/Sage. `mV_scale` arrives in its public storage
        # shape `(b, h, d, s_k // sf_vec)`; rebuild the logical `(d, s_k, h, b)`
        # block-scaled view from the raw iterator, then tile it exactly like
        # operand-B expects for the PV MMA.
        mSFVt = cute.make_tensor(
            mV_scale.iterator,
            tile_atom_to_shape_sfv_vt(
                (
                    mV_scale.shape[2],
                    seqlen_k,
                    mV_scale.shape[1],
                    mV_scale.shape[0],
                ),
                self.pv_sf_vec_size,
            ),
        )
        mSFVt_cur = mSFVt[None, None, kv_head_idx, batch_idx]
        gSFV = cute.local_tile(
            mSFVt_cur,
            cute.select(self.mma_tiler_pv, mode=[1, 2]),
            (0, block),
        )
        self.load_scale_stage_layout(gSFV, sSFV_stage, fill_rest=False)
    @cute.jit
    def load_v_fp4_pv_stage_public(
        self,
        mV: cute.Tensor,
        mV_storage: cute.Tensor,
        mV_scale: cute.Tensor,
        thr_mma_pv: cute.core.ThrMma,
        batch_idx: Int32,
        kv_head_idx: Int32,
        sV: cute.Tensor,
        sSFV: cute.Tensor,
        block: Int32,
        seqlen_k: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        page_idx: Optional[Int32] = None,
    ):
        del page_idx
        stage, phase = producer_state.index, producer_state.phase
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        pipeline_kv.sync_object_empty.wait(stage, phase)

        sV_stage = sV[None, None, None, stage]
        if const_expr(self.uneven_kv_smem):
            sV_stage = self.offset_kv_smem(sV_stage, stage, phase ^ 1)
        sSFV_stage = sSFV[None, None, None, stage]
        sV_stage_packed = self.as_packed_byte_tensor(
            sV_stage,
            storage_dtype=self.v_storage_dtype,
        )
        sV_stage_packed_flat = cute.group_modes(
            cute.filter_zeros(sV_stage_packed), 0, cute.rank(sV_stage_packed)
        )
        sV_stage_bytes_flat = cute.make_tensor(
            cute.recast_ptr(sV_stage.iterator, dtype=self.v_storage_dtype),
            (self.n_block_size * (self.head_dim_v_padded * self.v_dtype.width // self.v_storage_dtype.width),),
        )
        mV_storage_vt = mV_storage

        zero_v = self.v_storage_dtype(0)
        for idx in cutlass.range(
            tidx, cute.size(sV_stage_bytes_flat.shape), num_load_threads, unroll=1
        ):
            sV_stage_bytes_flat[idx] = zero_v

        valid_source_bytes = self.head_dim_v_padded
        num_row_pairs = self.n_block_size // 2
        block_rowpair_base = block * num_row_pairs
        packed_cols_per_row = cute.size(sV_stage_packed.shape[0][1])
        for row_pair in cutlass.range_constexpr(num_row_pairs):
            seqlen_idx = block * self.n_block_size + row_pair * 2
            if seqlen_idx < seqlen_k:
                for d_idx in cutlass.range(tidx, valid_source_bytes, num_load_threads, unroll=1):
                    d_outer = d_idx // packed_cols_per_row
                    packed_col = d_idx - d_outer * packed_cols_per_row
                    raw_offset = cute.crd2idx(
                        ((packed_col, row_pair), Int32(0), d_outer),
                        sV_stage_packed.layout,
                    )
                    sV_stage_bytes_flat[raw_offset] = mV_storage_vt[
                        d_idx,
                        block_rowpair_base + row_pair,
                        kv_head_idx,
                        batch_idx,
                    ]

        sSFV_logical = cute.make_tensor(
            sSFV_stage.iterator,
            tile_atom_to_shape_sfv_vt(
                (self.head_dim_v_padded, self.n_block_size, 1, 1),
                self.pv_sf_vec_size,
            ),
        )
        sSFV_logical_u8 = cute.make_tensor(
            cute.recast_ptr(utils.elem_pointer(sSFV_logical, (0, 0, 0, 0)), dtype=cutlass.Uint8),
            sSFV_logical.layout,
        )
        sSFV_logical_u8_flat = cute.group_modes(
            cute.filter_zeros(sSFV_logical_u8), 0, cute.rank(sSFV_logical_u8)
        )
        one_scale_u8 = float_to_ue4m3_byte(Float32(1.0))
        for idx in cutlass.range(tidx, cute.size(sSFV_logical_u8_flat.shape), num_load_threads, unroll=1):
            sSFV_logical_u8_flat[idx] = one_scale_u8

        num_d_groups = self.head_dim_v_padded // self.pv_sf_vec_size
        num_seq_groups = mV_scale.shape[3]
        num_heads_kv = mV_scale.shape[1]
        raw_scale_u8 = cute.make_tensor(
            cute.recast_ptr(
                utils.elem_pointer(mV_scale, (0,) * cute.rank(mV_scale)),
                dtype=cutlass.Uint8,
            ),
            (mV_scale.shape[0] * num_heads_kv * self.head_dim_v_padded * num_seq_groups,),
        )
        slice_base = (batch_idx * num_heads_kv + kv_head_idx) * self.head_dim_v_padded * num_seq_groups
        for idx in cutlass.range(tidx, self.n_block_size * num_d_groups, num_load_threads, unroll=1):
            row = idx // num_d_groups
            d_group = idx - row * num_d_groups
            seqlen_idx = block * self.n_block_size + row
            if seqlen_idx < seqlen_k:
                d_idx = d_group * self.pv_sf_vec_size
                seq_group = seqlen_idx // self.pv_sf_vec_size
                tile_m = d_idx // 64
                row_in_tile = d_idx - tile_m * 64
                quad = row_in_tile // 16
                row_mod16 = row_in_tile - quad * 16
                raw_offset = (
                    tile_m * 64 * num_seq_groups
                    + (seq_group // 4) * 256
                    + (seq_group % 4)
                    + quad * 4
                    + row_mod16 * 16
                )
                sSFV_logical_u8[d_idx, row, 0, 0] = raw_scale_u8[slice_base + raw_offset]

        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def load_v_with_scale_stage_public(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tVgV: Optional[cute.Tensor],
        tVsV: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        mV_scale: cute.Tensor,
        sV: cute.Tensor,
        sSFV: cute.Tensor,
        batch_idx: Int32,
        kv_head_idx: Int32,
        seqlen_k: Int32,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        page_idx: Optional[Int32] = None,
    ):
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes["V"] - self.tma_copy_bytes["K"]
        extra_kwargs = {"extra_tx_count": extra_tx_count_kv} if const_expr(self.use_tma_KV) else {}
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)

        self.load_v_scale_stage_public(
            mV_scale,
            batch_idx,
            kv_head_idx,
            sSFV[None, None, None, stage],
            block,
            seqlen_k,
        )

        if const_expr(self.use_tma_KV):
            assert tma_atom is not None and tVgV is not None and tVsV is not None
            tVsV_cur = tVsV[None, stage]
            if const_expr(self.uneven_kv_smem):
                tVsV_cur = self.offset_kv_smem(tVsV_cur, stage, phase ^ 1)
            tVgV_cur = tVgV[None, block] if const_expr(page_idx is None) else tVgV[None, 0, page_idx]
            cute.copy(
                tma_atom,
                tVgV_cur,
                tVsV_cur,
                tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state),
            )
        else:
            assert paged_kv_manager is not None
            paged_kv_manager.load_KV(block, sV[None, None, None, stage], "V")
            cute.arch.cp_async_commit_group()
            pipeline_kv.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def as_packed_byte_tensor(
        self,
        t: cute.Tensor,
        storage_dtype: Optional[Type[cutlass.Numeric]] = None,
    ):
        storage_dtype = self.q_storage_dtype if const_expr(storage_dtype is None) else storage_dtype
        return cute.make_tensor(
            cute.recast_ptr(t.iterator, dtype=storage_dtype),
            cute.recast_layout(storage_dtype.width, t.element_type.width, t.layout),
        )

    @cute.jit
    def copy_partitioned_fp4_bytes(
        self,
        tSrc: cute.Tensor,
        tDst: cute.Tensor,
        tCoord: cute.Tensor,
        valid_n_limit: Int32,
        valid_d_limit: Int32,
    ):
        tSrc_bytes = cute.make_tensor(
            cute.recast_ptr(
                utils.elem_pointer(tSrc, (0,) * cute.rank(tSrc)),
                dtype=self.v_storage_dtype,
            ),
            cute.recast_layout(self.v_storage_dtype.width, tSrc.element_type.width, tSrc.layout),
        )
        tDst_bytes = cute.make_tensor(
            cute.recast_ptr(
                utils.elem_pointer(tDst, (0,) * cute.rank(tDst)),
                dtype=self.v_storage_dtype,
            ),
            cute.recast_layout(self.v_storage_dtype.width, tDst.element_type.width, tDst.layout),
        )
        tSrc_flat = cute.flatten(tSrc_bytes)
        tDst_flat = cute.flatten(tDst_bytes)
        tCoord_flat = cute.flatten(tCoord)
        zero_v = self.v_storage_dtype(0)
        elems_per_storage = self.v_storage_dtype.width // self.v_dtype.width
        for idx in cutlass.range_constexpr(cute.size(tDst_flat.shape)):
            coord = tCoord_flat[idx * elems_per_storage]
            d_idx = coord[0]
            n_idx = coord[1]
            if d_idx < valid_d_limit and n_idx < valid_n_limit:
                tDst_flat[idx] = tSrc_flat[idx]
            else:
                tDst_flat[idx] = zero_v


    @cute.jit
    def load_vt_fp4_pv_stage(
        self,
        mV: cute.Tensor,          # unused: logical internal Vt view
        mV_storage: cute.Tensor,  # public packed-byte Vt storage: (b, h_k, d, s_k // 2)
        mV_scale: cute.Tensor,    # public swizzled SFVt storage: (b, h_k, d, s_k // sf_vec)
        thr_mma_pv: cute.core.ThrMma,
        batch_idx: Int32,
        kv_head_idx: Int32,
        sV: cute.Tensor,
        sSFV: cute.Tensor,
        block: Int32,
        seqlen_k: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        page_idx: Optional[Int32] = None,
    ):
        del mV
        del page_idx
        stage, phase = producer_state.index, producer_state.phase
        pipeline_kv.producer_acquire(producer_state)

        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        sV_stage = sV[None, None, None, stage]
        if const_expr(self.uneven_kv_smem):
            sV_stage = self.offset_kv_smem(sV_stage, stage, phase ^ 1)
        sSFV_stage = sSFV[None, None, None, stage]
        sSFV_logical = cute.make_tensor(
            sSFV_stage.iterator,
            tile_atom_to_shape_sfv_vt(
                (self.head_dim_v_padded, self.n_block_size, 1, 1),
                self.pv_sf_vec_size,
            ),
        )
        sSFV_logical_u8 = cute.make_tensor(
            cute.recast_ptr(utils.elem_pointer(sSFV_logical, (0, 0, 0, 0)), dtype=cutlass.Uint8),
            sSFV_logical.layout,
        )
        sSFV_logical_u8_flat = cute.group_modes(
            cute.filter_zeros(sSFV_logical_u8), 0, cute.rank(sSFV_logical_u8)
        )
        valid_packed_cols = self.head_dim_v_padded * self.v_dtype.width // self.v_storage_dtype.width
        sV_stage_packed = self.as_packed_byte_tensor(
            sV_stage,
            storage_dtype=self.v_storage_dtype,
        )
        zero_v = self.v_storage_dtype(0)
        low_nibble_mask = Int32(0x0F)
        shift_bits = Int32(4)

        one_scale_u8 = float_to_ue4m3_byte(Float32(1.0))
        del thr_mma_pv
        for idx in cutlass.range(
            tidx,
            self.n_block_size * valid_packed_cols,
            num_load_threads,
            unroll=1,
        ):
            packed_col = idx // self.n_block_size
            row = idx - packed_col * self.n_block_size
            seqlen_idx = block * self.n_block_size + row
            row_pair = row // 2
            row_parity = row - row_pair * 2
            if seqlen_idx < seqlen_k:
                src_d0 = packed_col * 2
                src_d1 = src_d0 + 1
                src0_i32 = Int32(mV_storage[batch_idx, kv_head_idx, src_d0, block * (self.n_block_size // 2) + row_pair])
                src1_i32 = Int32(mV_storage[batch_idx, kv_head_idx, src_d1, block * (self.n_block_size // 2) + row_pair])
                nibble_shift = row_parity * shift_bits
                nibble0 = (src0_i32 >> nibble_shift) & low_nibble_mask
                nibble1 = (src1_i32 >> nibble_shift) & low_nibble_mask
                packed_byte = cutlass.Uint8(nibble0 | (nibble1 << shift_bits))
                sV_stage_packed[(row_pair, packed_col), Int32(0), row_parity] = packed_byte
            else:
                sV_stage_packed[(row_pair, packed_col), Int32(0), row_parity] = zero_v

        for idx in cutlass.range(tidx, cute.size(sSFV_logical_u8_flat.shape), num_load_threads, unroll=1):
            sSFV_logical_u8_flat[idx] = one_scale_u8

        num_d_groups = self.head_dim_v_padded // self.pv_sf_vec_size
        num_seq_groups = mV_scale.shape[3]
        num_heads_kv = mV_scale.shape[1]
        raw_scale_u8 = cute.make_tensor(
            cute.recast_ptr(
                utils.elem_pointer(mV_scale, (0,) * cute.rank(mV_scale)),
                dtype=cutlass.Uint8,
            ),
            (mV_scale.shape[0] * num_heads_kv * self.head_dim_v_padded * num_seq_groups,),
        )
        slice_base = (batch_idx * num_heads_kv + kv_head_idx) * self.head_dim_v_padded * num_seq_groups
        for idx in cutlass.range(tidx, self.n_block_size * num_d_groups, num_load_threads, unroll=1):
            row = idx // num_d_groups
            d_group = idx - row * num_d_groups
            seqlen_idx = block * self.n_block_size + row
            if seqlen_idx < seqlen_k:
                d_idx = d_group * self.pv_sf_vec_size
                seq_group = seqlen_idx // self.pv_sf_vec_size
                tile_m = d_idx // 64
                row_in_tile = d_idx - tile_m * 64
                quad = row_in_tile // 16
                row_mod16 = row_in_tile - quad * 16
                raw_offset = (
                    tile_m * 64 * num_seq_groups
                    + (seq_group // 4) * 256
                    + (seq_group % 4)
                    + quad * 4
                    + row_mod16 * 16
                )
                sSFV_logical_u8[d_idx, row, 0, 0] = raw_scale_u8[slice_base + raw_offset]

        cute.arch.sync_warp()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )
        pipeline_kv.producer_commit(producer_state)

    @cute.jit
    def load_KV(
        self,
        tma_atom: Optional[cute.CopyAtom],
        tXgX: Optional[cute.Tensor],
        tXsX: Optional[cute.Tensor],
        paged_kv_manager: Optional[PagedKVManager],
        sX: cute.Tensor,
        block: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
        page_idx: Optional[Int32] = None,
        extra_tx_count: Optional[Int32] = None,
    ):
        assert K_or_V in ("K", "V")
        stage, phase = producer_state.index, producer_state.phase
        extra_tx_count_kv = self.tma_copy_bytes[K_or_V] - self.tma_copy_bytes["K"]
        extra_tx_count = (
            extra_tx_count_kv + (extra_tx_count if extra_tx_count is not None else 0) if const_expr(self.use_tma_KV)
            else None
        )
        extra_kwargs = {"extra_tx_count": extra_tx_count} if const_expr(self.use_tma_KV) else {}
        pipeline_kv.producer_acquire(producer_state, **extra_kwargs)
        if const_expr(K_or_V == "K" and self.uneven_kv_smem):
            # Before this round, the smem location was occupied by V, which is smaller than
            # K. So we need to wait for the stage after that (stage 1) to be empty as well.
            if stage == 0:
                pipeline_kv.sync_object_empty.wait(1, phase)

        if const_expr(self.use_tma_KV):
            assert tXgX is not None and tXsX is not None and tma_atom is not None
            tXsX_cur = tXsX[None, stage]
            if const_expr(self.uneven_kv_smem):
                # Since this is the producer_state, the phase starts at 1, so we have to invert it
                tXsX_cur = self.offset_kv_smem(tXsX_cur, stage, phase ^ 1)
            # Currently we assume that page_size == n_block_size so we index into tXgX with block = 0
            tXgX_cur = tXgX[None, block] if const_expr(page_idx is None) else tXgX[None, 0, page_idx]
            cute.copy(tma_atom, tXgX_cur, tXsX_cur, tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))
        else:
            assert paged_kv_manager is not None
            assert extra_tx_count is None
            paged_kv_manager.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            pipeline_kv.sync_object_full.arrive_cp_async_mbarrier(stage)

    @cute.jit
    def offset_kv_smem(self, sX: cute.Tensor, stage: Int32, phase: Int32):
        if const_expr(self.uneven_kv_smem):
            # smem layout is [smem_large, smem_small, smem_large], and the current stride is
            # (smem_large + smem_small) // 2. So for stage == 1, move right by offset if
            # phase == 0, or left by offset if phase == 1.
            offset = 0 if stage != 1 else self.uneven_kv_smem_offset * (1 - 2 * phase)
            return cute.make_tensor(sX.iterator + offset, sX.layout)
        else:
            return sX

    # @cute.jit
    # def warp_scheduler_barrier_init(self):
    #     warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    #     if warp_group_idx == 0:
    #         cute.arch.barrier_arrive(
    #             barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1), number_of_threads=2 * 128,
    #         )

    # def warp_scheduler_barrier_sync(self):
    #     cute.arch.barrier(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + utils.canonical_warp_group_idx(sync=False),
    #         number_of_threads=2 * 128
    #     )

    # def warp_scheduler_barrier_arrive(self):
    #     cur_wg = utils.canonical_warp_group_idx(sync=False)
    #     next_wg = 1 - cur_wg
    #     cute.arch.barrier_arrive(
    #         barrier_id=int(NamedBarrierFwdSm100.WarpSchedulerWG1) + next_wg, number_of_threads=2 * 128,
    #     )

    @cute.jit
    def apply_score_mod(
        self,
        tSrS_t2r,
        thr_tmem_load,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        n_block,
        softmax,
        seqlen: SeqlenInfoQK,
        aux_tensors=None,
        fastdiv_mods=(None, None),
        head_divmod=None,
    ):
        """Apply score modification for SM100 (constant q_idx)."""
        # Prepare index tensor with extra partition
        cS = cute.make_identity_tensor((self.m_block_size, self.n_block_size))
        cS = cute.domain_offset((m_block * self.m_block_size, n_block * self.n_block_size), cS)
        tScS = thr_mma_qk.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)

        # Shared q_idx for all scores
        q_idx_logical = tScS_t2r[0][0]

        # For Pack-GQA, compute the logical head index for this tile
        if cutlass.const_expr(self.pack_gqa_effective):
            assert head_divmod is not None
            # Building up the logical q_head idx: final_q_head = kv_head * qhead_per_kvhead + (q_physical % qhead_per_kvhead)
            q_physical = q_idx_logical
            q_idx_logical, head_offset = divmod(q_physical, head_divmod)
            head_idx = head_idx * self.qhead_per_kvhead + head_offset

        if cutlass.const_expr(aux_tensors is not None):
            seqlen_q_divmod, _ = fastdiv_mods
            _, q_idx_logical = divmod(q_idx_logical, seqlen_q_divmod)

        apply_score_mod_inner(
            tSrS_t2r,
            tScS_t2r,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax.softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=q_idx_logical,
            qhead_per_kvhead=self.qhead_per_kvhead if cutlass.const_expr(self.pack_gqa_effective) else 1,
        )
