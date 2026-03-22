import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_attn.cute.flash_hsa_fwd_sm100 import _materialize_runtime_state

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32

    from flash_attn.cute import utils
    from flash_attn.cute.cute_dsl_utils import to_cute_tensor

    _HAS_CUTE_RUNTIME = True
except Exception:  # pragma: no cover - import guard for CPU-only schedule tests
    _HAS_CUTE_RUNTIME = False

    class _FakeCuda:
        class CUstream:  # noqa: D401 - simple placeholder
            """CPU placeholder."""

    class _FakeConstexpr:
        def __class_getitem__(cls, item):
            return int

    class _FakeCutlass:
        Constexpr = _FakeConstexpr
        range = range

    class _FakeCuteMath:
        @staticmethod
        def exp2(*args, **kwargs):
            raise NotImplementedError("CuTe runtime is unavailable")

    class _FakeCute:
        Tensor = object
        math = _FakeCuteMath()

        class arch:
            @staticmethod
            def thread_idx():
                return (0, 0, 0)

            @staticmethod
            def block_idx():
                return (0, 0, 0)

            @staticmethod
            def barrier():
                return None

        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def kernel(fn):
            return fn

        @staticmethod
        def compile(*args, **kwargs):
            raise NotImplementedError("CuTe runtime is unavailable")

    class _FakeUtils:
        @staticmethod
        def atomic_add_fp32(*args, **kwargs):
            raise NotImplementedError("CuTe runtime is unavailable")

        @staticmethod
        def elem_pointer(*args, **kwargs):
            raise NotImplementedError("CuTe runtime is unavailable")

    def to_cute_tensor(*args, **kwargs):
        raise NotImplementedError("CuTe runtime is unavailable")

    cuda = _FakeCuda()
    cutlass = _FakeCutlass()
    cute = _FakeCute()
    utils = _FakeUtils()
    Float32 = float
    Int32 = int


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


def _use_hsa_sentence_full_kernel_fastpath() -> bool:
    return os.getenv("FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL", "0") == "1"


def _use_hsa_true_fused_bwd() -> bool:
    return os.getenv("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "0") == "1"


def _use_hsa_sentence_varlen_2cta() -> bool:
    return os.getenv("FLASH_ATTN_HSA_USE_SENTENCE_VARLEN_2CTA", "0") == "1"


def _should_force_hsa_sentence_varlen_2cta(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    use_true_fused: bool,
    monolithic_schedule,
) -> bool:
    if not _use_hsa_sentence_varlen_2cta():
        return False
    if not use_true_fused:
        return False
    if q.shape[-1] != 64 or k.shape[-1] != 64:
        return False
    if monolithic_schedule.anchor_full_q_row_start.numel() != 0:
        return False
    if monolithic_schedule.anchor_tail_q_row_start.numel() != 0:
        return False
    if q.shape[2] != k.shape[2]:
        return False
    return True


def _get_hsa_anchor_kernel_threads(qhead_per_kvhead: int) -> int:
    override = os.getenv("FLASH_ATTN_HSA_ANCHOR_KERNEL_THREADS")
    if override is not None:
        return int(override)
    return 128 if qhead_per_kvhead > 1 else 256


def _monolithic_has_sentence_full_desc(monolithic_schedule) -> bool:
    return monolithic_schedule.sentence_full_q_start.numel() != 0


def _monolithic_has_sentence_tail_desc(monolithic_schedule) -> bool:
    return monolithic_schedule.sentence_tail_q_start.numel() != 0


def _monolithic_has_anchor_full_desc(monolithic_schedule) -> bool:
    return monolithic_schedule.anchor_full_q_row_start.numel() != 0


def _monolithic_has_anchor_tail_desc(monolithic_schedule) -> bool:
    return monolithic_schedule.anchor_tail_q_row_start.numel() != 0


def _monolithic_has_sentence_families(monolithic_schedule) -> bool:
    return _monolithic_has_sentence_full_desc(monolithic_schedule) or _monolithic_has_sentence_tail_desc(
        monolithic_schedule
    )


def _normalize_sentence_lse_override(sentence_lse: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if sentence_lse is None or sentence_lse.numel() == 0:
        return None
    return sentence_lse


@dataclass
class HSAMonolithicSentenceStageResult:
    row_accums: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    done_families: frozenset[str] = frozenset()
    prepared: Optional["HSABwdPreparedTensors"] = None
    runtime_state: Optional[dict] = None


class FlashHSASentencePackOutputsSm100:
    """Pack sentence-stream out / weighted-dout on device."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mdOut: cute.Tensor,
        mTotalLSE: cute.Tensor,
        mSentenceLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mOutPacked: cute.Tensor,
        mdOutPacked: cute.Tensor,
        stream: cuda.CUstream,
    ):
        num_stream_rows = mRowIdx.shape[0]
        num_q_heads = mOut.shape[1]
        head_dim_v = mOut.shape[2]
        total_tasks = num_stream_rows * num_q_heads * head_dim_v
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mOut,
            mdOut,
            mTotalLSE,
            mSentenceLSE,
            mRowIdx,
            mOutPacked,
            mdOutPacked,
            total_tasks,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mdOut: cute.Tensor,
        mTotalLSE: cute.Tensor,
        mSentenceLSE: cute.Tensor,
        mRowIdx: cute.Tensor,
        mOutPacked: cute.Tensor,
        mdOutPacked: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_q_heads = Int32(mOut.shape[1])
            head_dim_v = Int32(mOut.shape[2])
            elems_per_row = num_q_heads * head_dim_v
            stream_row = task_idx // elems_per_row
            rem = task_idx - stream_row * elems_per_row
            q_head = rem // head_dim_v
            dv_idx = rem - q_head * head_dim_v
            global_row = mRowIdx[stream_row]
            weight = cute.math.exp2(
                (Float32(mSentenceLSE[q_head, stream_row]) - Float32(mTotalLSE[global_row, q_head]))
                * Float32(math.log2(math.e)),
                fastmath=True,
            )
            mOutPacked[stream_row, q_head, dv_idx] = mOut[global_row, q_head, dv_idx]
            mdOutPacked[stream_row, q_head, dv_idx] = (
                weight * Float32(mdOut[global_row, q_head, dv_idx])
            ).to(mdOutPacked.element_type)


class FlashHSASentenceScatterRowsSm100:
    """Scatter packed sentence grads into row accumulators on device."""

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
            global_row = mRowIdx[stream_row]
            mDstRows[global_row, head_idx, dim_idx] = Float32(
                mSrcPacked[stream_row, head_idx, dim_idx]
            ).to(mDstRows.element_type)


class FlashHSAPackRowsSm100:
    """Pack selected rows from a flat tensor into a compact stream tensor."""

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
            global_row = mRowIdx[stream_row]
            mDstPacked[stream_row, head_idx, dim_idx] = mSrcRows[
                global_row, head_idx, dim_idx
            ].to(mDstPacked.element_type)


class FlashHSACastRowsSm100:
    """Cast flat row accumulators into output dtype on device."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mSrcRows: cute.Tensor,
        mDstRows: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total_rows = mSrcRows.shape[0]
        num_heads = mSrcRows.shape[1]
        head_dim = mSrcRows.shape[2]
        total_tasks = total_rows * num_heads * head_dim
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mSrcRows,
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
        mSrcRows: cute.Tensor,
        mDstRows: cute.Tensor,
        total_tasks: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_heads = Int32(mSrcRows.shape[1])
            head_dim = Int32(mSrcRows.shape[2])
            elems_per_row = num_heads * head_dim
            row_idx = task_idx // elems_per_row
            rem = task_idx - row_idx * elems_per_row
            head_idx = rem // head_dim
            dim_idx = rem - head_idx * head_dim
            mDstRows[row_idx, head_idx, dim_idx] = Float32(
                mSrcRows[row_idx, head_idx, dim_idx]
            ).to(mDstRows.element_type)


class FlashHSAPackLSESm100:
    """Pack [B, H, T] LSE into flat [B*T, H] rows on device."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mLSE: cute.Tensor,
        mTotalLSE: cute.Tensor,
        stream: cuda.CUstream,
    ):
        batch_size = mLSE.shape[0]
        num_q_heads = mLSE.shape[1]
        seqlen = mLSE.shape[2]
        total_tasks = batch_size * num_q_heads * seqlen
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mLSE,
            mTotalLSE,
            total_tasks,
            seqlen,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mLSE: cute.Tensor,
        mTotalLSE: cute.Tensor,
        total_tasks: Int32,
        seqlen: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_q_heads = Int32(mLSE.shape[1])
            row_idx = task_idx // num_q_heads
            q_head = task_idx - row_idx * num_q_heads
            batch_idx = row_idx // seqlen
            token_idx = row_idx - batch_idx * seqlen
            mTotalLSE[row_idx, q_head] = Float32(mLSE[batch_idx, q_head, token_idx]).to(mTotalLSE.element_type)


class FlashHSAPrepareAuxSm100:
    """Prepare flat total_lse / lse_log2 / dPsum on device."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mdOut: cute.Tensor,
        mLSE: cute.Tensor,
        mTotalLSE: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        stream: cuda.CUstream,
    ):
        batch_size = mLSE.shape[0]
        num_q_heads = mLSE.shape[1]
        seqlen = mLSE.shape[2]
        total_tasks = batch_size * num_q_heads * seqlen
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mOut,
            mdOut,
            mLSE,
            mTotalLSE,
            mLSElog2,
            mdPsum,
            total_tasks,
            seqlen,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mdOut: cute.Tensor,
        mLSE: cute.Tensor,
        mTotalLSE: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        total_tasks: Int32,
        seqlen: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_q_heads = Int32(mLSE.shape[1])
            head_dim_v = Int32(mOut.shape[3])
            row_idx = task_idx // num_q_heads
            q_head = task_idx - row_idx * num_q_heads
            batch_idx = row_idx // seqlen
            token_idx = row_idx - batch_idx * seqlen
            lse_val = Float32(mLSE[batch_idx, q_head, token_idx])
            dpsum = Float32.zero
            for dv_idx in cutlass.range(head_dim_v, unroll=1):
                dpsum += Float32(mOut[batch_idx, token_idx, q_head, dv_idx]) * Float32(
                    mdOut[batch_idx, token_idx, q_head, dv_idx]
                )
            mTotalLSE[row_idx, q_head] = lse_val.to(mTotalLSE.element_type)
            mLSElog2[row_idx, q_head] = (lse_val * Float32(math.log2(math.e))).to(mLSElog2.element_type)
            mdPsum[row_idx, q_head] = dpsum.to(mdPsum.element_type)


class FlashHSAPrepareMonolithicWorkspacesSm100:
    """Prepare monolithic [B, H, T] dPsum / lse_log2 workspaces on device."""

    arch = 100

    def __init__(self, *, num_threads: int = 256):
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mOut: cute.Tensor,
        mdOut: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        stream: cuda.CUstream,
    ):
        batch_size = mLSE.shape[0]
        num_q_heads = mLSE.shape[1]
        seqlen = mLSE.shape[2]
        total_tasks = batch_size * num_q_heads * seqlen
        grid_x = cute.ceil_div(total_tasks, self.num_threads)
        self.kernel(
            mOut,
            mdOut,
            mLSE,
            mdPsum,
            mLSElog2,
            total_tasks,
            seqlen,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mOut: cute.Tensor,
        mdOut: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mLSElog2: cute.Tensor,
        total_tasks: Int32,
        seqlen: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, _, _ = cute.arch.block_idx()
        task_idx = block_idx * self.num_threads + tidx
        if task_idx < total_tasks:
            num_q_heads = Int32(mLSE.shape[1])
            head_dim_v = Int32(mOut.shape[3])
            row_idx = task_idx // num_q_heads
            q_head = task_idx - row_idx * num_q_heads
            batch_idx = row_idx // seqlen
            token_idx = row_idx - batch_idx * seqlen
            lse_val = Float32(mLSE[batch_idx, q_head, token_idx])
            dpsum = Float32.zero
            for dv_idx in cutlass.range(head_dim_v, unroll=1):
                dpsum += Float32(mOut[batch_idx, token_idx, q_head, dv_idx]) * Float32(
                    mdOut[batch_idx, token_idx, q_head, dv_idx]
                )
            mdPsum[batch_idx, q_head, token_idx] = dpsum.to(mdPsum.element_type)
            mLSElog2[batch_idx, q_head, token_idx] = (lse_val * Float32(math.log2(math.e))).to(
                mLSElog2.element_type
            )


@cute.jit
def _accumulate_hsa_bwd_row_masked(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    mask_words: cute.Tensor,
    block_id: Int32,
    q_local: Int32,
    global_q_row: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    key_row_base: Int32,
    k_len: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    qhead_per_kvhead: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    head_dim_v: cutlass.Constexpr[int],
):
    if k_len > 0:
        for qh_offset in cutlass.range(qhead_per_kvhead, unroll=1):
            q_head = qhead_start + qh_offset
            lse_log2 = Float32(mLSElog2[global_q_row, q_head])
            dpsum = Float32(mdPsum[global_q_row, q_head])
            for k_rel in cutlass.range(k_len, unroll=1):
                word_idx = k_rel // 32
                bit_idx = k_rel % 32
                mask_word = cutlass.Uint32(mask_words[block_id, q_local, word_idx])
                bit = utils.shr_u32(mask_word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)
                if bit != cutlass.Uint32(0):
                    key_row = key_row_base + k_rel
                    score = Float32.zero
                    for d in cutlass.range(head_dim, unroll=1):
                        score += Float32(mQ[global_q_row, q_head, d]) * Float32(mK[key_row, kv_head, d])
                    prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                    dprob = Float32.zero
                    for dv_idx in cutlass.range(head_dim_v, unroll=1):
                        dprob += Float32(mdO[global_q_row, q_head, dv_idx]) * Float32(
                            mV[key_row, kv_head, dv_idx]
                        )
                    ds = prob * (dprob - dpsum)
                    ds_scaled = ds * softmax_scale
                    for d in cutlass.range(head_dim, unroll=1):
                        utils.atomic_add_fp32(
                            ds_scaled * Float32(mK[key_row, kv_head, d]),
                            utils.elem_pointer(mdQaccum, (global_q_row, q_head, d)),
                        )
                        mdK[key_row, kv_head, d] = Float32(mdK[key_row, kv_head, d]) + ds_scaled * Float32(
                            mQ[global_q_row, q_head, d]
                        )
                    for dv_idx in cutlass.range(head_dim_v, unroll=1):
                        mdV[key_row, kv_head, dv_idx] = Float32(mdV[key_row, kv_head, dv_idx]) + prob * Float32(
                            mdO[global_q_row, q_head, dv_idx]
                        )


class FlashHSABackwardRowGroupSm100:
    """Internal-only row-group-aware HSA backward kernel on SM100/SM110."""

    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        *,
        qhead_per_kvhead: int,
        q_block_size: int,
        k_block_size: int,
    ):
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.q_block_size = q_block_size
        self.k_block_size = k_block_size
        self.row_group_size = q_block_size // 2
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mask_block_cnt: cute.Tensor,
        mask_block_idx: cute.Tensor,
        full_block_cnt: cute.Tensor,
        full_block_idx: cute.Tensor,
        block_id_table: cute.Tensor,
        mask_words: cute.Tensor,
        row_group_nonempty: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        batch_size = mQ.shape[0] // seqlen
        num_kv_heads = mK.shape[1]
        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            mask_block_cnt,
            mask_block_idx,
            full_block_cnt,
            full_block_idx,
            block_id_table,
            mask_words,
            row_group_nonempty,
            blocks_per_batch,
            seqlen,
            softmax_scale,
        ).launch(
            grid=[blocks_per_batch, num_kv_heads, batch_size],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mask_block_cnt: cute.Tensor,
        mask_block_idx: cute.Tensor,
        full_block_cnt: cute.Tensor,
        full_block_idx: cute.Tensor,
        block_id_table: cute.Tensor,
        mask_words: cute.Tensor,
        row_group_nonempty: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        k_block_idx, kv_head, batch_idx = cute.arch.block_idx()
        if tidx == 0:
            block_flat_start = batch_idx * seqlen + k_block_idx * self.k_block_size
            qhead_start = kv_head * self.qhead_per_kvhead
            scale_log2 = Float32(softmax_scale * math.log2(math.e))
            k_len = min(self.k_block_size, seqlen - k_block_idx * self.k_block_size)

            full_count = full_block_cnt[batch_idx, 0, k_block_idx]
            for idx in cutlass.range(full_count, unroll=1):
                q_block = full_block_idx[batch_idx, 0, k_block_idx, idx]
                q_start = q_block * self.q_block_size
                q_len = min(self.q_block_size, seqlen - q_start)
                for q_local in cutlass.range(q_len, unroll=1):
                    _accumulate_hsa_bwd_row(
                        mQ,
                        mK,
                        mV,
                        mdO,
                        mLSElog2,
                        mdPsum,
                        mdQaccum,
                        mdK,
                        mdV,
                        batch_idx * seqlen + q_start + q_local,
                        kv_head,
                        qhead_start,
                        block_flat_start,
                        k_len,
                        softmax_scale,
                        scale_log2,
                        self.qhead_per_kvhead,
                        self.head_dim,
                        self.head_dim_v,
                    )

            mask_count = mask_block_cnt[batch_idx, 0, k_block_idx]
            for idx in cutlass.range(mask_count, unroll=1):
                q_block = mask_block_idx[batch_idx, 0, k_block_idx, idx]
                q_start = q_block * self.q_block_size
                q_len = min(self.q_block_size, seqlen - q_start)
                block_id = block_id_table[batch_idx, k_block_idx, q_block]
                group_bits = row_group_nonempty[block_id]
                if (group_bits & Int32(1)) != Int32(0):
                    first_group_len = min(self.row_group_size, q_len)
                    for q_local in cutlass.range(first_group_len, unroll=1):
                        _accumulate_hsa_bwd_row_masked(
                            mQ,
                            mK,
                            mV,
                            mdO,
                            mLSElog2,
                            mdPsum,
                            mdQaccum,
                            mdK,
                            mdV,
                            mask_words,
                            block_id,
                            q_local,
                            batch_idx * seqlen + q_start + q_local,
                            kv_head,
                            qhead_start,
                            block_flat_start,
                            k_len,
                            softmax_scale,
                            scale_log2,
                            self.qhead_per_kvhead,
                            self.head_dim,
                            self.head_dim_v,
                        )
                if (group_bits & Int32(2)) != Int32(0):
                    for q_local in cutlass.range(self.row_group_size, q_len, unroll=1):
                        _accumulate_hsa_bwd_row_masked(
                            mQ,
                            mK,
                            mV,
                            mdO,
                            mLSElog2,
                            mdPsum,
                            mdQaccum,
                            mdK,
                            mdV,
                            mask_words,
                            block_id,
                            q_local,
                            batch_idx * seqlen + q_start + q_local,
                            kv_head,
                            qhead_start,
                            block_flat_start,
                            k_len,
                            softmax_scale,
                            scale_log2,
                            self.qhead_per_kvhead,
                            self.head_dim,
                            self.head_dim_v,
                        )


class FlashHSABackwardSm100ScalarReference:
    """Reference-only scalar HSA backward kernel on SM100/SM110."""

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
        self.tile_m = 64
        self.tile_n = k_block_size
        self.num_threads = 32

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sentence_full_kblock_row_ptr: cute.Tensor,
        sentence_full_q_start: cute.Tensor,
        sentence_full_q_len: cute.Tensor,
        sentence_full_q_group_mask: cute.Tensor,
        sentence_full_k_local_start: cute.Tensor,
        sentence_full_k_len: cute.Tensor,
        sentence_tail_kblock_row_ptr: cute.Tensor,
        sentence_tail_q_start: cute.Tensor,
        sentence_tail_q_len: cute.Tensor,
        sentence_tail_q_group_mask: cute.Tensor,
        sentence_tail_k_local_start: cute.Tensor,
        sentence_tail_k_len: cute.Tensor,
        sentence_tail_row0_prefix_len: cute.Tensor,
        anchor_full_kblock_row_ptr: cute.Tensor,
        anchor_full_q_row_start: cute.Tensor,
        anchor_full_q_row_count: cute.Tensor,
        anchor_full_q_group_mask: cute.Tensor,
        anchor_full_k_local_start: cute.Tensor,
        anchor_full_k_len: cute.Tensor,
        anchor_tail_kblock_row_ptr: cute.Tensor,
        anchor_tail_q_row_start: cute.Tensor,
        anchor_tail_q_row_count: cute.Tensor,
        anchor_tail_q_group_mask: cute.Tensor,
        anchor_tail_k_local_start: cute.Tensor,
        anchor_tail_k_len: cute.Tensor,
        anchor_tail_prefix_row_start: cute.Tensor,
        anchor_q_indices: cute.Tensor,
        anchor_prefix_len: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        batch_size = mQ.shape[0] // seqlen
        num_kv_heads = mK.shape[1]
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            sentence_full_kblock_row_ptr,
            sentence_full_q_start,
            sentence_full_q_len,
            sentence_full_q_group_mask,
            sentence_full_k_local_start,
            sentence_full_k_len,
            sentence_tail_kblock_row_ptr,
            sentence_tail_q_start,
            sentence_tail_q_len,
            sentence_tail_q_group_mask,
            sentence_tail_k_local_start,
            sentence_tail_k_len,
            sentence_tail_row0_prefix_len,
            anchor_full_kblock_row_ptr,
            anchor_full_q_row_start,
            anchor_full_q_row_count,
            anchor_full_q_group_mask,
            anchor_full_k_local_start,
            anchor_full_k_len,
            anchor_tail_kblock_row_ptr,
            anchor_tail_q_row_start,
            anchor_tail_q_row_count,
            anchor_tail_q_group_mask,
            anchor_tail_k_local_start,
            anchor_tail_k_len,
            anchor_tail_prefix_row_start,
            anchor_q_indices,
            anchor_prefix_len,
            blocks_per_batch,
            seqlen,
            softmax_scale,
        ).launch(
            grid=[blocks_per_batch, num_kv_heads, batch_size],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sentence_full_kblock_row_ptr: cute.Tensor,
        sentence_full_q_start: cute.Tensor,
        sentence_full_q_len: cute.Tensor,
        sentence_full_q_group_mask: cute.Tensor,
        sentence_full_k_local_start: cute.Tensor,
        sentence_full_k_len: cute.Tensor,
        sentence_tail_kblock_row_ptr: cute.Tensor,
        sentence_tail_q_start: cute.Tensor,
        sentence_tail_q_len: cute.Tensor,
        sentence_tail_q_group_mask: cute.Tensor,
        sentence_tail_k_local_start: cute.Tensor,
        sentence_tail_k_len: cute.Tensor,
        sentence_tail_row0_prefix_len: cute.Tensor,
        anchor_full_kblock_row_ptr: cute.Tensor,
        anchor_full_q_row_start: cute.Tensor,
        anchor_full_q_row_count: cute.Tensor,
        anchor_full_q_group_mask: cute.Tensor,
        anchor_full_k_local_start: cute.Tensor,
        anchor_full_k_len: cute.Tensor,
        anchor_tail_kblock_row_ptr: cute.Tensor,
        anchor_tail_q_row_start: cute.Tensor,
        anchor_tail_q_row_count: cute.Tensor,
        anchor_tail_q_group_mask: cute.Tensor,
        anchor_tail_k_local_start: cute.Tensor,
        anchor_tail_k_len: cute.Tensor,
        anchor_tail_prefix_row_start: cute.Tensor,
        anchor_q_indices: cute.Tensor,
        anchor_prefix_len: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
    ):
        _run_hsa_bwd_monolithic_scalar_kernel_body(
            self,
            mQ,
            mK,
            mV,
            mO,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            sentence_full_kblock_row_ptr,
            sentence_full_q_start,
            sentence_full_q_len,
            sentence_full_q_group_mask,
            sentence_full_k_local_start,
            sentence_full_k_len,
            sentence_tail_kblock_row_ptr,
            sentence_tail_q_start,
            sentence_tail_q_len,
            sentence_tail_q_group_mask,
            sentence_tail_k_local_start,
            sentence_tail_k_len,
            sentence_tail_row0_prefix_len,
            anchor_full_kblock_row_ptr,
            anchor_full_q_row_start,
            anchor_full_q_row_count,
            anchor_full_q_group_mask,
            anchor_full_k_local_start,
            anchor_full_k_len,
            anchor_tail_kblock_row_ptr,
            anchor_tail_q_row_start,
            anchor_tail_q_row_count,
            anchor_tail_q_group_mask,
            anchor_tail_k_local_start,
            anchor_tail_k_len,
            anchor_tail_prefix_row_start,
            anchor_q_indices,
            anchor_prefix_len,
            blocks_per_batch,
            seqlen,
            softmax_scale,
        )


@cute.jit
def _run_hsa_bwd_sentence_full_scalar(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sentence_full_kblock_row_ptr: cute.Tensor,
    sentence_full_q_start: cute.Tensor,
    sentence_full_q_len: cute.Tensor,
    sentence_full_q_group_mask: cute.Tensor,
    sentence_full_k_local_start: cute.Tensor,
    sentence_full_k_len: cute.Tensor,
    global_k_block: Int32,
    batch_idx: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    seqlen: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
):
    sent_full_start = sentence_full_kblock_row_ptr[global_k_block]
    sent_full_end = sentence_full_kblock_row_ptr[global_k_block + 1]
    for desc_idx in cutlass.range(sent_full_start, sent_full_end, unroll=1):
        q_start = sentence_full_q_start[desc_idx]
        q_len = sentence_full_q_len[desc_idx]
        q_group_mask = sentence_full_q_group_mask[desc_idx]
        key_row_base = block_flat_start + sentence_full_k_local_start[desc_idx]
        prefix = sentence_full_k_len[desc_idx]
        if (q_group_mask & Int32(1)) != Int32(0):
            first_group_end = min(Int32(32), q_len)
            for q_off in cutlass.range(first_group_end, unroll=1):
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )
        if (q_group_mask & Int32(2)) != Int32(0):
            for q_off in cutlass.range(Int32(32), q_len, unroll=1):
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _run_hsa_bwd_sentence_full_kernel_slice(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sentence_full_kblock_row_ptr: cute.Tensor,
    sentence_full_q_start: cute.Tensor,
    sentence_full_q_len: cute.Tensor,
    sentence_full_q_group_mask: cute.Tensor,
    sentence_full_k_local_start: cute.Tensor,
    sentence_full_k_len: cute.Tensor,
    global_k_block: Int32,
    batch_idx: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    seqlen: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    # CTA-parallel kernel-owned sentence_full slice. This is still scalar math,
    # but it removes the temporary host-side FA4 helper from the active
    # monolithic path and gives us a direct in-kernel seam to replace with the
    # true FA4/TMA+MMA body next.
    sent_full_start = sentence_full_kblock_row_ptr[global_k_block]
    sent_full_end = sentence_full_kblock_row_ptr[global_k_block + 1]
    for desc_idx in cutlass.range(sent_full_start, sent_full_end, unroll=1):
        q_start = sentence_full_q_start[desc_idx]
        q_len = sentence_full_q_len[desc_idx]
        q_group_mask = sentence_full_q_group_mask[desc_idx]
        key_row_base = block_flat_start + sentence_full_k_local_start[desc_idx]
        prefix = sentence_full_k_len[desc_idx]
        if (q_group_mask & Int32(1)) != Int32(0):
            first_group_end = min(Int32(32), q_len)
            for q_off in cutlass.range(tidx, first_group_end, num_threads, unroll=1):
                _accumulate_hsa_bwd_row_atomic_dkv(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )
        if (q_group_mask & Int32(2)) != Int32(0):
            for q_off in cutlass.range(Int32(32) + tidx, q_len, num_threads, unroll=1):
                _accumulate_hsa_bwd_row_atomic_dkv(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _run_hsa_bwd_sentence_tail_scalar(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sentence_tail_kblock_row_ptr: cute.Tensor,
    sentence_tail_q_start: cute.Tensor,
    sentence_tail_q_len: cute.Tensor,
    sentence_tail_q_group_mask: cute.Tensor,
    sentence_tail_k_local_start: cute.Tensor,
    sentence_tail_k_len: cute.Tensor,
    sentence_tail_row0_prefix_len: cute.Tensor,
    global_k_block: Int32,
    batch_idx: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    seqlen: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
):
    sent_tail_start = sentence_tail_kblock_row_ptr[global_k_block]
    sent_tail_end = sentence_tail_kblock_row_ptr[global_k_block + 1]
    for desc_idx in cutlass.range(sent_tail_start, sent_tail_end, unroll=1):
        q_start = sentence_tail_q_start[desc_idx]
        q_len = sentence_tail_q_len[desc_idx]
        q_group_mask = sentence_tail_q_group_mask[desc_idx]
        key_row_base = block_flat_start + sentence_tail_k_local_start[desc_idx]
        k_len = sentence_tail_k_len[desc_idx]
        row0_prefix_len = sentence_tail_row0_prefix_len[desc_idx]
        if (q_group_mask & Int32(1)) != Int32(0):
            first_group_end = min(Int32(32), q_len)
            for q_off in cutlass.range(first_group_end, unroll=1):
                prefix = min(k_len, row0_prefix_len + q_off)
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )
        if (q_group_mask & Int32(2)) != Int32(0):
            for q_off in cutlass.range(Int32(32), q_len, unroll=1):
                prefix = min(k_len, row0_prefix_len + q_off)
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _run_hsa_bwd_sentence_tail_kernel_slice(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sentence_tail_kblock_row_ptr: cute.Tensor,
    sentence_tail_q_start: cute.Tensor,
    sentence_tail_q_len: cute.Tensor,
    sentence_tail_q_group_mask: cute.Tensor,
    sentence_tail_k_local_start: cute.Tensor,
    sentence_tail_k_len: cute.Tensor,
    sentence_tail_row0_prefix_len: cute.Tensor,
    global_k_block: Int32,
    batch_idx: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    seqlen: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    sent_tail_start = sentence_tail_kblock_row_ptr[global_k_block]
    sent_tail_end = sentence_tail_kblock_row_ptr[global_k_block + 1]
    for desc_idx in cutlass.range(sent_tail_start, sent_tail_end, unroll=1):
        q_start = sentence_tail_q_start[desc_idx]
        q_len = sentence_tail_q_len[desc_idx]
        q_group_mask = sentence_tail_q_group_mask[desc_idx]
        key_row_base = block_flat_start + sentence_tail_k_local_start[desc_idx]
        k_len = sentence_tail_k_len[desc_idx]
        row0_prefix_len = sentence_tail_row0_prefix_len[desc_idx]
        if (q_group_mask & Int32(1)) != Int32(0):
            first_group_end = min(Int32(32), q_len)
            for q_off in cutlass.range(tidx, first_group_end, num_threads, unroll=1):
                prefix = min(k_len, row0_prefix_len + q_off)
                _accumulate_hsa_bwd_row_atomic_dkv(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )
        if (q_group_mask & Int32(2)) != Int32(0):
            for q_off in cutlass.range(Int32(32) + tidx, q_len, num_threads, unroll=1):
                prefix = min(k_len, row0_prefix_len + q_off)
                _accumulate_hsa_bwd_row_atomic_dkv(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    batch_idx * seqlen + q_start + q_off,
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _run_hsa_bwd_anchor_full_scalar(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    anchor_full_kblock_row_ptr: cute.Tensor,
    anchor_full_q_row_start: cute.Tensor,
    anchor_full_q_row_count: cute.Tensor,
    anchor_full_q_group_mask: cute.Tensor,
    anchor_full_k_local_start: cute.Tensor,
    anchor_full_k_len: cute.Tensor,
    anchor_q_indices: cute.Tensor,
    global_k_block: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
):
    anchor_full_start = anchor_full_kblock_row_ptr[global_k_block]
    anchor_full_end = anchor_full_kblock_row_ptr[global_k_block + 1]
    for desc_idx in cutlass.range(anchor_full_start, anchor_full_end, unroll=1):
        q_row_start = anchor_full_q_row_start[desc_idx]
        q_row_count = anchor_full_q_row_count[desc_idx]
        q_group_mask = anchor_full_q_group_mask[desc_idx]
        key_row_base = block_flat_start + anchor_full_k_local_start[desc_idx]
        prefix = anchor_full_k_len[desc_idx]
        if (q_group_mask & Int32(1)) != Int32(0):
            first_group_end = min(Int32(32), q_row_count)
            for q_rel in cutlass.range(first_group_end, unroll=1):
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    anchor_q_indices[q_row_start + q_rel],
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )
        if (q_group_mask & Int32(2)) != Int32(0):
            for q_rel in cutlass.range(Int32(32), q_row_count, unroll=1):
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    anchor_q_indices[q_row_start + q_rel],
                    kv_head,
                    qhead_start,
                    key_row_base,
                    prefix,
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _accumulate_hsa_bwd_qhead_key_atomic(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    global_q_row: Int32,
    kv_head: Int32,
    q_head: Int32,
    key_row: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    head_dim: cutlass.Constexpr[int],
    head_dim_v: cutlass.Constexpr[int],
):
    lse_log2 = Float32(mLSElog2[global_q_row, q_head])
    dpsum = Float32(mdPsum[global_q_row, q_head])
    score = Float32.zero
    for d in cutlass.range(head_dim, unroll=1):
        score += Float32(mQ[global_q_row, q_head, d]) * Float32(mK[key_row, kv_head, d])
    prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
    dprob = Float32.zero
    for dv_idx in cutlass.range(head_dim_v, unroll=1):
        dprob += Float32(mdO[global_q_row, q_head, dv_idx]) * Float32(mV[key_row, kv_head, dv_idx])
    ds = prob * (dprob - dpsum)
    ds_scaled = ds * softmax_scale
    for d in cutlass.range(head_dim, unroll=1):
        utils.atomic_add_fp32(
            ds_scaled * Float32(mK[key_row, kv_head, d]),
            utils.elem_pointer(mdQaccum, (global_q_row, q_head, d)),
        )
        utils.atomic_add_fp32(
            ds_scaled * Float32(mQ[global_q_row, q_head, d]),
            utils.elem_pointer(mdK, (key_row, kv_head, d)),
        )
    for dv_idx in cutlass.range(head_dim_v, unroll=1):
        utils.atomic_add_fp32(
            prob * Float32(mdO[global_q_row, q_head, dv_idx]),
            utils.elem_pointer(mdV, (key_row, kv_head, dv_idx)),
        )


@cute.jit
def _accumulate_hsa_bwd_qhead_key_serial_dkv(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    global_q_row: Int32,
    kv_head: Int32,
    q_head: Int32,
    key_row: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    head_dim: cutlass.Constexpr[int],
    head_dim_v: cutlass.Constexpr[int],
):
    lse_log2 = Float32(mLSElog2[global_q_row, q_head])
    dpsum = Float32(mdPsum[global_q_row, q_head])
    score = Float32.zero
    for d in cutlass.range(head_dim, unroll=1):
        score += Float32(mQ[global_q_row, q_head, d]) * Float32(mK[key_row, kv_head, d])
    prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
    dprob = Float32.zero
    for dv_idx in cutlass.range(head_dim_v, unroll=1):
        dprob += Float32(mdO[global_q_row, q_head, dv_idx]) * Float32(mV[key_row, kv_head, dv_idx])
    ds = prob * (dprob - dpsum)
    ds_scaled = ds * softmax_scale
    for d in cutlass.range(head_dim, unroll=1):
        utils.atomic_add_fp32(
            ds_scaled * Float32(mK[key_row, kv_head, d]),
            utils.elem_pointer(mdQaccum, (global_q_row, q_head, d)),
        )
        mdK[key_row, kv_head, d] = Float32(mdK[key_row, kv_head, d]) + ds_scaled * Float32(
            mQ[global_q_row, q_head, d]
        )
    for dv_idx in cutlass.range(head_dim_v, unroll=1):
        mdV[key_row, kv_head, dv_idx] = Float32(mdV[key_row, kv_head, dv_idx]) + prob * Float32(
            mdO[global_q_row, q_head, dv_idx]
        )


@cute.jit
def _run_hsa_bwd_anchor_descriptor_group_panel(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    q_row_start: Int32,
    q_rel_start: Int32,
    q_rel_end: Int32,
    key_row_base: Int32,
    k_len: Int32,
    prefix_row_start: Int32,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    kv_head: Int32,
    qhead_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
    use_tail_prefix: cutlass.Constexpr[bool],
):
    tasks_per_row = Int32(self.qhead_per_kvhead) * k_len
    total_tasks = (q_rel_end - q_rel_start) * tasks_per_row
    for task_idx in cutlass.range(tidx, total_tasks, num_threads, unroll=1):
        row_task = task_idx // k_len
        k_rel = task_idx - row_task * k_len
        local_row = row_task // Int32(self.qhead_per_kvhead)
        qh_offset = row_task - local_row * Int32(self.qhead_per_kvhead)
        q_rel = q_rel_start + local_row
        prefix = k_len
        if use_tail_prefix:
            prefix = anchor_prefix_len[prefix_row_start + q_rel]
        if k_rel < prefix:
            _accumulate_hsa_bwd_qhead_key_atomic(
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_q_indices[q_row_start + q_rel],
                kv_head,
                qhead_start + qh_offset,
                key_row_base + k_rel,
                softmax_scale,
                scale_log2,
                self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _run_hsa_bwd_anchor_descriptor_group_panel_small_reduced(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    q_row_start: Int32,
    q_rel_start: Int32,
    q_rel_end: Int32,
    key_row_base: Int32,
    k_len: Int32,
    prefix_row_start: Int32,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    kv_head: Int32,
    qhead_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    use_tail_prefix: cutlass.Constexpr[bool],
):
    for k_rel in cutlass.range(k_len, unroll=1):
        key_row = key_row_base + k_rel
        dK_acc = cute.make_fragment(self.head_dim, Float32)
        dV_acc = cute.make_fragment(self.head_dim_v, Float32)
        dK_acc.fill(0.0)
        dV_acc.fill(0.0)
        for q_rel in cutlass.range(q_rel_start, q_rel_end, unroll=1):
            prefix = k_len
            if use_tail_prefix:
                prefix = anchor_prefix_len[prefix_row_start + q_rel]
            if k_rel < prefix:
                global_q_row = anchor_q_indices[q_row_start + q_rel]
                for qh_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
                    q_head = qhead_start + qh_offset
                    lse_log2 = Float32(mLSElog2[global_q_row, q_head])
                    dpsum = Float32(mdPsum[global_q_row, q_head])
                    score = Float32.zero
                    for d in cutlass.range(self.head_dim, unroll=1):
                        score += Float32(mQ[global_q_row, q_head, d]) * Float32(mK[key_row, kv_head, d])
                    prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                    dprob = Float32.zero
                    for dv_idx in cutlass.range(self.head_dim_v, unroll=1):
                        dprob += Float32(mdO[global_q_row, q_head, dv_idx]) * Float32(
                            mV[key_row, kv_head, dv_idx]
                        )
                    ds = prob * (dprob - dpsum)
                    ds_scaled = ds * softmax_scale
                    for d in cutlass.range(self.head_dim, unroll=1):
                        utils.atomic_add_fp32(
                            ds_scaled * Float32(mK[key_row, kv_head, d]),
                            utils.elem_pointer(mdQaccum, (global_q_row, q_head, d)),
                        )
                        dK_acc[d] = Float32(dK_acc[d]) + ds_scaled * Float32(mQ[global_q_row, q_head, d])
                    for dv_idx in cutlass.range(self.head_dim_v, unroll=1):
                        dV_acc[dv_idx] = Float32(dV_acc[dv_idx]) + prob * Float32(mdO[global_q_row, q_head, dv_idx])
        for d in cutlass.range(self.head_dim, unroll=1):
            utils.atomic_add_fp32(
                Float32(dK_acc[d]),
                utils.elem_pointer(mdK, (key_row, kv_head, d)),
            )
        for dv_idx in cutlass.range(self.head_dim_v, unroll=1):
            utils.atomic_add_fp32(
                Float32(dV_acc[dv_idx]),
                utils.elem_pointer(mdV, (key_row, kv_head, dv_idx)),
            )


@cute.jit
def _run_hsa_bwd_anchor_descriptor_panel_serial(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    q_row_start: Int32,
    q_row_count: Int32,
    q_group_mask: Int32,
    key_row_base: Int32,
    k_len: Int32,
    prefix_row_start: Int32,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    kv_head: Int32,
    qhead_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    use_tail_prefix: cutlass.Constexpr[bool],
):
    if (q_group_mask & Int32(1)) != Int32(0):
        first_group_end = min(Int32(32), q_row_count)
        for q_rel in cutlass.range(first_group_end, unroll=1):
            prefix = k_len
            if use_tail_prefix:
                prefix = anchor_prefix_len[prefix_row_start + q_rel]
            global_q_row = anchor_q_indices[q_row_start + q_rel]
            for qh_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
                q_head = qhead_start + qh_offset
                for k_rel in cutlass.range(prefix, unroll=1):
                    _accumulate_hsa_bwd_qhead_key_serial_dkv(
                        mQ,
                        mK,
                        mV,
                        mdO,
                        mLSElog2,
                        mdPsum,
                        mdQaccum,
                        mdK,
                        mdV,
                        global_q_row,
                        kv_head,
                        q_head,
                        key_row_base + k_rel,
                        softmax_scale,
                        scale_log2,
                        self.head_dim,
                        self.head_dim_v,
                    )
    if (q_group_mask & Int32(2)) != Int32(0):
        for q_rel in cutlass.range(Int32(32), q_row_count, unroll=1):
            prefix = k_len
            if use_tail_prefix:
                prefix = anchor_prefix_len[prefix_row_start + q_rel]
            global_q_row = anchor_q_indices[q_row_start + q_rel]
            for qh_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
                q_head = qhead_start + qh_offset
                for k_rel in cutlass.range(prefix, unroll=1):
                    _accumulate_hsa_bwd_qhead_key_serial_dkv(
                        mQ,
                        mK,
                        mV,
                        mdO,
                        mLSElog2,
                        mdPsum,
                        mdQaccum,
                        mdK,
                        mdV,
                        global_q_row,
                        kv_head,
                        q_head,
                        key_row_base + k_rel,
                        softmax_scale,
                        scale_log2,
                        self.head_dim,
                        self.head_dim_v,
                    )


@cute.jit
def _run_hsa_bwd_anchor_descriptor_panel(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    q_row_start: Int32,
    q_row_count: Int32,
    q_group_mask: Int32,
    key_row_base: Int32,
    k_len: Int32,
    prefix_row_start: Int32,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    kv_head: Int32,
    qhead_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
    use_tail_prefix: cutlass.Constexpr[bool],
):
    total_tasks = q_row_count * Int32(self.qhead_per_kvhead) * k_len
    if total_tasks <= Int32(32) and num_threads == 1:
        if (q_group_mask & Int32(1)) != Int32(0):
            _run_hsa_bwd_anchor_descriptor_group_panel_small_reduced(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                q_row_start,
                Int32(0),
                min(Int32(32), q_row_count),
                key_row_base,
                k_len,
                prefix_row_start,
                anchor_q_indices,
                anchor_prefix_len,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                use_tail_prefix,
            )
        if (q_group_mask & Int32(2)) != Int32(0):
            _run_hsa_bwd_anchor_descriptor_group_panel_small_reduced(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                q_row_start,
                Int32(32),
                q_row_count,
                key_row_base,
                k_len,
                prefix_row_start,
                anchor_q_indices,
                anchor_prefix_len,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                use_tail_prefix,
            )
    else:
        if (q_group_mask & Int32(1)) != Int32(0):
            _run_hsa_bwd_anchor_descriptor_group_panel(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                q_row_start,
                Int32(0),
                min(Int32(32), q_row_count),
                key_row_base,
                k_len,
                prefix_row_start,
                anchor_q_indices,
                anchor_prefix_len,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                tidx,
                num_threads,
                use_tail_prefix,
            )
        if (q_group_mask & Int32(2)) != Int32(0):
            _run_hsa_bwd_anchor_descriptor_group_panel(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                q_row_start,
                Int32(32),
                q_row_count,
                key_row_base,
                k_len,
                prefix_row_start,
                anchor_q_indices,
                anchor_prefix_len,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                tidx,
                num_threads,
                use_tail_prefix,
            )


@cute.jit
def _run_hsa_bwd_anchor_full_kernel_slice(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    anchor_full_kblock_row_ptr: cute.Tensor,
    anchor_full_q_row_start: cute.Tensor,
    anchor_full_q_row_count: cute.Tensor,
    anchor_full_q_group_mask: cute.Tensor,
    anchor_full_k_local_start: cute.Tensor,
    anchor_full_k_len: cute.Tensor,
    anchor_q_indices: cute.Tensor,
    global_k_block: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    anchor_full_start = anchor_full_kblock_row_ptr[global_k_block]
    anchor_full_end = anchor_full_kblock_row_ptr[global_k_block + 1]
    desc_count = anchor_full_end - anchor_full_start
    if self.qhead_per_kvhead > 1:
        lane = tidx % Int32(32)
        warp_idx = tidx // Int32(32)
        num_warps = Int32(num_threads // 32)
        for desc_idx in cutlass.range(anchor_full_start + warp_idx, anchor_full_end, num_warps, unroll=1):
            _run_hsa_bwd_anchor_descriptor_panel(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_full_q_row_start[desc_idx],
                anchor_full_q_row_count[desc_idx],
                anchor_full_q_group_mask[desc_idx],
                block_flat_start + anchor_full_k_local_start[desc_idx],
                anchor_full_k_len[desc_idx],
                Int32(0),
                anchor_q_indices,
                anchor_q_indices,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                lane,
                32,
                False,
            )
    elif desc_count <= Int32(6):
        if tidx == 0:
            for desc_idx in cutlass.range(anchor_full_start, anchor_full_end, unroll=1):
                _run_hsa_bwd_anchor_descriptor_panel_serial(
                    self,
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    anchor_full_q_row_start[desc_idx],
                    anchor_full_q_row_count[desc_idx],
                    anchor_full_q_group_mask[desc_idx],
                    block_flat_start + anchor_full_k_local_start[desc_idx],
                    anchor_full_k_len[desc_idx],
                    Int32(0),
                    anchor_q_indices,
                    anchor_q_indices,
                    kv_head,
                    qhead_start,
                    softmax_scale,
                    scale_log2,
                    False,
                )
    else:
        for desc_idx in cutlass.range(anchor_full_start + tidx, anchor_full_end, num_threads, unroll=1):
            _run_hsa_bwd_anchor_descriptor_panel(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_full_q_row_start[desc_idx],
                anchor_full_q_row_count[desc_idx],
                anchor_full_q_group_mask[desc_idx],
                block_flat_start + anchor_full_k_local_start[desc_idx],
                anchor_full_k_len[desc_idx],
                Int32(0),
                anchor_q_indices,
                anchor_q_indices,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                Int32(0),
                1,
                False,
            )


@cute.jit
def _run_hsa_bwd_anchor_tail_scalar(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    anchor_tail_kblock_row_ptr: cute.Tensor,
    anchor_tail_q_row_start: cute.Tensor,
    anchor_tail_q_row_count: cute.Tensor,
    anchor_tail_q_group_mask: cute.Tensor,
    anchor_tail_k_local_start: cute.Tensor,
    anchor_tail_k_len: cute.Tensor,
    anchor_tail_prefix_row_start: cute.Tensor,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    global_k_block: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
):
    anchor_tail_start = anchor_tail_kblock_row_ptr[global_k_block]
    anchor_tail_end = anchor_tail_kblock_row_ptr[global_k_block + 1]
    for desc_idx in cutlass.range(anchor_tail_start, anchor_tail_end, unroll=1):
        q_row_start = anchor_tail_q_row_start[desc_idx]
        q_row_count = anchor_tail_q_row_count[desc_idx]
        q_group_mask = anchor_tail_q_group_mask[desc_idx]
        key_row_base = block_flat_start + anchor_tail_k_local_start[desc_idx]
        prefix_row_start = anchor_tail_prefix_row_start[desc_idx]
        if (q_group_mask & Int32(1)) != Int32(0):
            first_group_end = min(Int32(32), q_row_count)
            for q_rel in cutlass.range(first_group_end, unroll=1):
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    anchor_q_indices[q_row_start + q_rel],
                    kv_head,
                    qhead_start,
                    key_row_base,
                    anchor_prefix_len[prefix_row_start + q_rel],
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )
        if (q_group_mask & Int32(2)) != Int32(0):
            for q_rel in cutlass.range(Int32(32), q_row_count, unroll=1):
                _accumulate_hsa_bwd_row(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    anchor_q_indices[q_row_start + q_rel],
                    kv_head,
                    qhead_start,
                    key_row_base,
                    anchor_prefix_len[prefix_row_start + q_rel],
                    softmax_scale,
                    scale_log2,
                    self.qhead_per_kvhead,
                    self.head_dim,
                    self.head_dim_v,
                )


@cute.jit
def _run_hsa_bwd_anchor_tail_kernel_slice(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    anchor_tail_kblock_row_ptr: cute.Tensor,
    anchor_tail_q_row_start: cute.Tensor,
    anchor_tail_q_row_count: cute.Tensor,
    anchor_tail_q_group_mask: cute.Tensor,
    anchor_tail_k_local_start: cute.Tensor,
    anchor_tail_k_len: cute.Tensor,
    anchor_tail_prefix_row_start: cute.Tensor,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    global_k_block: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    block_flat_start: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
):
    anchor_tail_start = anchor_tail_kblock_row_ptr[global_k_block]
    anchor_tail_end = anchor_tail_kblock_row_ptr[global_k_block + 1]
    desc_count = anchor_tail_end - anchor_tail_start
    if self.qhead_per_kvhead > 1:
        lane = tidx % Int32(32)
        warp_idx = tidx // Int32(32)
        num_warps = Int32(num_threads // 32)
        for desc_idx in cutlass.range(anchor_tail_start + warp_idx, anchor_tail_end, num_warps, unroll=1):
            _run_hsa_bwd_anchor_descriptor_panel(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_tail_q_row_start[desc_idx],
                anchor_tail_q_row_count[desc_idx],
                anchor_tail_q_group_mask[desc_idx],
                block_flat_start + anchor_tail_k_local_start[desc_idx],
                anchor_tail_k_len[desc_idx],
                anchor_tail_prefix_row_start[desc_idx],
                anchor_q_indices,
                anchor_prefix_len,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                lane,
                32,
                True,
            )
    elif desc_count <= Int32(6):
        if tidx == 0:
            for desc_idx in cutlass.range(anchor_tail_start, anchor_tail_end, unroll=1):
                _run_hsa_bwd_anchor_descriptor_panel_serial(
                    self,
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSElog2,
                    mdPsum,
                    mdQaccum,
                    mdK,
                    mdV,
                    anchor_tail_q_row_start[desc_idx],
                    anchor_tail_q_row_count[desc_idx],
                    anchor_tail_q_group_mask[desc_idx],
                    block_flat_start + anchor_tail_k_local_start[desc_idx],
                    anchor_tail_k_len[desc_idx],
                    anchor_tail_prefix_row_start[desc_idx],
                    anchor_q_indices,
                    anchor_prefix_len,
                    kv_head,
                    qhead_start,
                    softmax_scale,
                    scale_log2,
                    True,
                )
    else:
        for desc_idx in cutlass.range(anchor_tail_start + tidx, anchor_tail_end, num_threads, unroll=1):
            _run_hsa_bwd_anchor_descriptor_panel(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_tail_q_row_start[desc_idx],
                anchor_tail_q_row_count[desc_idx],
                anchor_tail_q_group_mask[desc_idx],
                block_flat_start + anchor_tail_k_local_start[desc_idx],
                anchor_tail_k_len[desc_idx],
                anchor_tail_prefix_row_start[desc_idx],
                anchor_q_indices,
                anchor_prefix_len,
                kv_head,
                qhead_start,
                softmax_scale,
                scale_log2,
                Int32(0),
                1,
                True,
            )


@cute.jit
def _run_hsa_bwd_monolithic_scalar_kernel_body(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sentence_full_kblock_row_ptr: cute.Tensor,
    sentence_full_q_start: cute.Tensor,
    sentence_full_q_len: cute.Tensor,
    sentence_full_q_group_mask: cute.Tensor,
    sentence_full_k_local_start: cute.Tensor,
    sentence_full_k_len: cute.Tensor,
    sentence_tail_kblock_row_ptr: cute.Tensor,
    sentence_tail_q_start: cute.Tensor,
    sentence_tail_q_len: cute.Tensor,
    sentence_tail_q_group_mask: cute.Tensor,
    sentence_tail_k_local_start: cute.Tensor,
    sentence_tail_k_len: cute.Tensor,
    sentence_tail_row0_prefix_len: cute.Tensor,
    anchor_full_kblock_row_ptr: cute.Tensor,
    anchor_full_q_row_start: cute.Tensor,
    anchor_full_q_row_count: cute.Tensor,
    anchor_full_q_group_mask: cute.Tensor,
    anchor_full_k_local_start: cute.Tensor,
    anchor_full_k_len: cute.Tensor,
    anchor_tail_kblock_row_ptr: cute.Tensor,
    anchor_tail_q_row_start: cute.Tensor,
    anchor_tail_q_row_count: cute.Tensor,
    anchor_tail_q_group_mask: cute.Tensor,
    anchor_tail_k_local_start: cute.Tensor,
    anchor_tail_k_len: cute.Tensor,
    anchor_tail_prefix_row_start: cute.Tensor,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    blocks_per_batch: Int32,
    seqlen: Int32,
    softmax_scale: Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    k_block_idx, kv_head, batch_idx = cute.arch.block_idx()
    if tidx == 0:
        global_k_block = batch_idx * blocks_per_batch + k_block_idx
        block_flat_start = batch_idx * seqlen + k_block_idx * self.k_block_size
        qhead_start = kv_head * self.qhead_per_kvhead
        scale_log2 = Float32(softmax_scale * math.log2(math.e))
        _run_hsa_bwd_sentence_full_scalar(
            self,
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            sentence_full_kblock_row_ptr,
            sentence_full_q_start,
            sentence_full_q_len,
            sentence_full_q_group_mask,
            sentence_full_k_local_start,
            sentence_full_k_len,
            global_k_block,
            batch_idx,
            kv_head,
            qhead_start,
            block_flat_start,
            seqlen,
            softmax_scale,
            scale_log2,
        )
        _run_hsa_bwd_sentence_tail_scalar(
            self,
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            sentence_tail_kblock_row_ptr,
            sentence_tail_q_start,
            sentence_tail_q_len,
            sentence_tail_q_group_mask,
            sentence_tail_k_local_start,
            sentence_tail_k_len,
            sentence_tail_row0_prefix_len,
            global_k_block,
            batch_idx,
            kv_head,
            qhead_start,
            block_flat_start,
            seqlen,
            softmax_scale,
            scale_log2,
        )


class FlashHSAAnchorBackwardSm100:
    """Anchor-only HSA backward kernel for the mixed monolithic fast path."""

    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        *,
        qhead_per_kvhead: int,
        k_block_size: int,
    ):
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.qhead_per_kvhead = qhead_per_kvhead
        self.k_block_size = k_block_size
        self.num_threads = _get_hsa_anchor_kernel_threads(qhead_per_kvhead)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        anchor_full_kblock_row_ptr: cute.Tensor,
        anchor_full_q_row_start: cute.Tensor,
        anchor_full_q_row_count: cute.Tensor,
        anchor_full_q_group_mask: cute.Tensor,
        anchor_full_k_local_start: cute.Tensor,
        anchor_full_k_len: cute.Tensor,
        anchor_tail_kblock_row_ptr: cute.Tensor,
        anchor_tail_q_row_start: cute.Tensor,
        anchor_tail_q_row_count: cute.Tensor,
        anchor_tail_q_group_mask: cute.Tensor,
        anchor_tail_k_local_start: cute.Tensor,
        anchor_tail_k_len: cute.Tensor,
        anchor_tail_prefix_row_start: cute.Tensor,
        anchor_q_indices: cute.Tensor,
        anchor_prefix_len: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        batch_size = mQ.shape[0] // seqlen
        num_kv_heads = mK.shape[1]
        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            anchor_full_kblock_row_ptr,
            anchor_full_q_row_start,
            anchor_full_q_row_count,
            anchor_full_q_group_mask,
            anchor_full_k_local_start,
            anchor_full_k_len,
            anchor_tail_kblock_row_ptr,
            anchor_tail_q_row_start,
            anchor_tail_q_row_count,
            anchor_tail_q_group_mask,
            anchor_tail_k_local_start,
            anchor_tail_k_len,
            anchor_tail_prefix_row_start,
            anchor_q_indices,
            anchor_prefix_len,
            blocks_per_batch,
            seqlen,
            softmax_scale,
        ).launch(
            grid=[blocks_per_batch, num_kv_heads, batch_size],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        anchor_full_kblock_row_ptr: cute.Tensor,
        anchor_full_q_row_start: cute.Tensor,
        anchor_full_q_row_count: cute.Tensor,
        anchor_full_q_group_mask: cute.Tensor,
        anchor_full_k_local_start: cute.Tensor,
        anchor_full_k_len: cute.Tensor,
        anchor_tail_kblock_row_ptr: cute.Tensor,
        anchor_tail_q_row_start: cute.Tensor,
        anchor_tail_q_row_count: cute.Tensor,
        anchor_tail_q_group_mask: cute.Tensor,
        anchor_tail_k_local_start: cute.Tensor,
        anchor_tail_k_len: cute.Tensor,
        anchor_tail_prefix_row_start: cute.Tensor,
        anchor_q_indices: cute.Tensor,
        anchor_prefix_len: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        k_block_idx, kv_head, batch_idx = cute.arch.block_idx()
        global_k_block = batch_idx * blocks_per_batch + k_block_idx
        block_flat_start = batch_idx * seqlen + k_block_idx * self.k_block_size
        qhead_start = kv_head * self.qhead_per_kvhead
        scale_log2 = Float32(softmax_scale * math.log2(math.e))
        anchor_full_has_work = anchor_full_kblock_row_ptr[global_k_block] != anchor_full_kblock_row_ptr[global_k_block + 1]
        anchor_tail_has_work = anchor_tail_kblock_row_ptr[global_k_block] != anchor_tail_kblock_row_ptr[global_k_block + 1]
        if anchor_full_has_work:
            _run_hsa_bwd_anchor_full_kernel_slice(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_full_kblock_row_ptr,
                anchor_full_q_row_start,
                anchor_full_q_row_count,
                anchor_full_q_group_mask,
                anchor_full_k_local_start,
                anchor_full_k_len,
                anchor_q_indices,
                global_k_block,
                kv_head,
                qhead_start,
                block_flat_start,
                softmax_scale,
                scale_log2,
                tidx,
                self.num_threads,
            )
        if anchor_full_has_work and anchor_tail_has_work:
            cute.arch.barrier()
        if anchor_tail_has_work:
            _run_hsa_bwd_anchor_tail_kernel_slice(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_tail_kblock_row_ptr,
                anchor_tail_q_row_start,
                anchor_tail_q_row_count,
                anchor_tail_q_group_mask,
                anchor_tail_k_local_start,
                anchor_tail_k_len,
                anchor_tail_prefix_row_start,
                anchor_q_indices,
                anchor_prefix_len,
                global_k_block,
                kv_head,
                qhead_start,
                block_flat_start,
                softmax_scale,
                scale_log2,
                tidx,
                self.num_threads,
            )


class FlashHSASentenceBackwardSm100(FlashAttentionBackwardSm100):
    """Sentence-only HSA backward specialization.

    This is the fast sentence-first bring-up path for the true-fused rollout:
    it keeps sentence-family execution on the real FA4 varlen backward kernel,
    with broad 2CTA enabled for sentence-only D64 schedules, while mixed blocks
    continue to use the split sentence+anchor fallback until the mixed fused
    kernel clears the performance gate.
    """

    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        *,
        qhead_per_kvhead: int,
        deterministic: bool,
        use_2cta_instrs: bool,
    ):
        super().__init__(
            head_dim,
            head_dim_v,
            is_causal=True,
            is_local=False,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=64,
            tile_n=128,
            is_persistent=True,
            deterministic=deterministic,
            cluster_size=2 if use_2cta_instrs else 1,
            use_2cta_instrs=use_2cta_instrs,
            score_mod=None,
            score_mod_bwd=None,
            mask_mod=None,
            has_aux_tensors=False,
            subtile_factor=1,
        )
        self.force_2cta_sm100_varlen = bool(self.use_2cta_instrs and self.tile_hdim == 64)

    def run_rows(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        schedule,
        monolithic_schedule,
        softmax_scale: float,
        prepared: Optional["HSABwdPreparedTensors"] = None,
        runtime_state=None,
        sentence_lse_override: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], frozenset[str]]:
        return _run_hsa_bwd_sentence_rows_core(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            self.deterministic,
            prepared=prepared,
            runtime_state=runtime_state,
            sentence_lse_override=sentence_lse_override,
            force_2cta_sm100_varlen=self.force_2cta_sm100_varlen,
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        schedule,
        monolithic_schedule,
        softmax_scale: float,
        prepared: Optional["HSABwdPreparedTensors"] = None,
        runtime_state=None,
        sentence_lse_override: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (dq_accum_rows, dk_accum_rows, dv_accum_rows), _ = self.run_rows(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            prepared=prepared,
            runtime_state=runtime_state,
            sentence_lse_override=sentence_lse_override,
        )
        return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)


class FlashHSABackwardSm100(FlashAttentionBackwardSm100):
    """Port target for the true fused HSA MMA backward kernel on SM100/SM110.

    This class locks the real kernel configuration the HSA path needs and serves
    as the integration surface for the ongoing FA4-style port. The default
    env-gated monolithic path still keeps the split sentence/anchor fallback
    active, while ``FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD=1`` forces execution
    through this kernel for all descriptor families.
    """

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
        super().__init__(
            head_dim,
            head_dim_v,
            is_causal=False,
            is_local=False,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=64,
            tile_n=k_block_size,
            is_persistent=True,
            deterministic=deterministic,
            cluster_size=1,
            use_2cta_instrs=False,
            score_mod=None,
            score_mod_bwd=None,
            mask_mod=None,
            has_aux_tensors=False,
            subtile_factor=1,
        )
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.k_block_size = k_block_size
        self.anchor_row_panel_size = anchor_row_panel_size
        self.tile_m = 64
        self.tile_n = k_block_size
        self.pack_gqa = False
        self.num_threads = getattr(self, "threads_per_cta", 512)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sentence_full_kblock_row_ptr: cute.Tensor,
        sentence_full_q_start: cute.Tensor,
        sentence_full_q_len: cute.Tensor,
        sentence_full_q_group_mask: cute.Tensor,
        sentence_full_k_local_start: cute.Tensor,
        sentence_full_k_len: cute.Tensor,
        sentence_tail_kblock_row_ptr: cute.Tensor,
        sentence_tail_q_start: cute.Tensor,
        sentence_tail_q_len: cute.Tensor,
        sentence_tail_q_group_mask: cute.Tensor,
        sentence_tail_k_local_start: cute.Tensor,
        sentence_tail_k_len: cute.Tensor,
        sentence_tail_row0_prefix_len: cute.Tensor,
        anchor_full_kblock_row_ptr: cute.Tensor,
        anchor_full_q_row_start: cute.Tensor,
        anchor_full_q_row_count: cute.Tensor,
        anchor_full_q_group_mask: cute.Tensor,
        anchor_full_k_local_start: cute.Tensor,
        anchor_full_k_len: cute.Tensor,
        anchor_tail_kblock_row_ptr: cute.Tensor,
        anchor_tail_q_row_start: cute.Tensor,
        anchor_tail_q_row_count: cute.Tensor,
        anchor_tail_q_group_mask: cute.Tensor,
        anchor_tail_k_local_start: cute.Tensor,
        anchor_tail_k_len: cute.Tensor,
        anchor_tail_prefix_row_start: cute.Tensor,
        anchor_q_indices: cute.Tensor,
        anchor_prefix_len: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        # Phase-1 port slice: start exercising the generic SM100 backward setup
        # on the real HSA class before replacing the scalar kernel body.
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSElog2.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype
        self.is_varlen_k = False
        self.is_varlen_q = False
        self.use_tma_store = True
        self.dKV_postprocess = self.qhead_per_kvhead > 1
        self.use_block_sparsity = False
        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()
        batch_size = mQ.shape[0] // seqlen
        num_kv_heads = mK.shape[1]
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mdO,
            mLSElog2,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            sentence_full_kblock_row_ptr,
            sentence_full_q_start,
            sentence_full_q_len,
            sentence_full_q_group_mask,
            sentence_full_k_local_start,
            sentence_full_k_len,
            sentence_tail_kblock_row_ptr,
            sentence_tail_q_start,
            sentence_tail_q_len,
            sentence_tail_q_group_mask,
            sentence_tail_k_local_start,
            sentence_tail_k_len,
            sentence_tail_row0_prefix_len,
            anchor_full_kblock_row_ptr,
            anchor_full_q_row_start,
            anchor_full_q_row_count,
            anchor_full_q_group_mask,
            anchor_full_k_local_start,
            anchor_full_k_len,
            anchor_tail_kblock_row_ptr,
            anchor_tail_q_row_start,
            anchor_tail_q_row_count,
            anchor_tail_q_group_mask,
            anchor_tail_k_local_start,
            anchor_tail_k_len,
            anchor_tail_prefix_row_start,
            anchor_q_indices,
            anchor_prefix_len,
            blocks_per_batch,
            seqlen,
            softmax_scale,
        ).launch(
            grid=[blocks_per_batch, num_kv_heads, batch_size],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mLSElog2: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sentence_full_kblock_row_ptr: cute.Tensor,
        sentence_full_q_start: cute.Tensor,
        sentence_full_q_len: cute.Tensor,
        sentence_full_q_group_mask: cute.Tensor,
        sentence_full_k_local_start: cute.Tensor,
        sentence_full_k_len: cute.Tensor,
        sentence_tail_kblock_row_ptr: cute.Tensor,
        sentence_tail_q_start: cute.Tensor,
        sentence_tail_q_len: cute.Tensor,
        sentence_tail_q_group_mask: cute.Tensor,
        sentence_tail_k_local_start: cute.Tensor,
        sentence_tail_k_len: cute.Tensor,
        sentence_tail_row0_prefix_len: cute.Tensor,
        anchor_full_kblock_row_ptr: cute.Tensor,
        anchor_full_q_row_start: cute.Tensor,
        anchor_full_q_row_count: cute.Tensor,
        anchor_full_q_group_mask: cute.Tensor,
        anchor_full_k_local_start: cute.Tensor,
        anchor_full_k_len: cute.Tensor,
        anchor_tail_kblock_row_ptr: cute.Tensor,
        anchor_tail_q_row_start: cute.Tensor,
        anchor_tail_q_row_count: cute.Tensor,
        anchor_tail_q_group_mask: cute.Tensor,
        anchor_tail_k_local_start: cute.Tensor,
        anchor_tail_k_len: cute.Tensor,
        anchor_tail_prefix_row_start: cute.Tensor,
        anchor_q_indices: cute.Tensor,
        anchor_prefix_len: cute.Tensor,
        blocks_per_batch: Int32,
        seqlen: Int32,
        softmax_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        k_block_idx, kv_head, batch_idx = cute.arch.block_idx()
        global_k_block = batch_idx * blocks_per_batch + k_block_idx
        qhead_start = kv_head * self.qhead_per_kvhead
        block_flat_start = batch_idx * seqlen + k_block_idx * self.k_block_size
        scale_log2 = Float32(softmax_scale * math.log2(math.e))
        sentence_full_has_work = sentence_full_kblock_row_ptr[global_k_block] != sentence_full_kblock_row_ptr[global_k_block + 1]
        sentence_tail_has_work = sentence_tail_kblock_row_ptr[global_k_block] != sentence_tail_kblock_row_ptr[global_k_block + 1]
        anchor_full_has_work = anchor_full_kblock_row_ptr[global_k_block] != anchor_full_kblock_row_ptr[global_k_block + 1]
        anchor_tail_has_work = anchor_tail_kblock_row_ptr[global_k_block] != anchor_tail_kblock_row_ptr[global_k_block + 1]
        if sentence_full_has_work:
            _run_hsa_bwd_sentence_full_kernel_slice(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                sentence_full_kblock_row_ptr,
                sentence_full_q_start,
                sentence_full_q_len,
                sentence_full_q_group_mask,
                sentence_full_k_local_start,
                sentence_full_k_len,
                global_k_block,
                batch_idx,
                kv_head,
                qhead_start,
                block_flat_start,
                seqlen,
                softmax_scale,
                scale_log2,
                tidx,
                self.num_threads,
            )
            cute.arch.barrier()
        if sentence_tail_has_work:
            _run_hsa_bwd_sentence_tail_kernel_slice(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                sentence_tail_kblock_row_ptr,
                sentence_tail_q_start,
                sentence_tail_q_len,
                sentence_tail_q_group_mask,
                sentence_tail_k_local_start,
                sentence_tail_k_len,
                sentence_tail_row0_prefix_len,
                global_k_block,
                batch_idx,
                kv_head,
                qhead_start,
                block_flat_start,
                seqlen,
                softmax_scale,
                scale_log2,
                tidx,
                self.num_threads,
            )
            cute.arch.barrier()
        if anchor_full_has_work:
            _run_hsa_bwd_anchor_full_kernel_slice(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_full_kblock_row_ptr,
                anchor_full_q_row_start,
                anchor_full_q_row_count,
                anchor_full_q_group_mask,
                anchor_full_k_local_start,
                anchor_full_k_len,
                anchor_q_indices,
                global_k_block,
                kv_head,
                qhead_start,
                block_flat_start,
                softmax_scale,
                scale_log2,
                tidx,
                self.num_threads,
            )
        if anchor_full_has_work and anchor_tail_has_work:
            cute.arch.barrier()
        if anchor_tail_has_work:
            _run_hsa_bwd_anchor_tail_kernel_slice(
                self,
                mQ,
                mK,
                mV,
                mdO,
                mLSElog2,
                mdPsum,
                mdQaccum,
                mdK,
                mdV,
                anchor_tail_kblock_row_ptr,
                anchor_tail_q_row_start,
                anchor_tail_q_row_count,
                anchor_tail_q_group_mask,
                anchor_tail_k_local_start,
                anchor_tail_k_len,
                anchor_tail_prefix_row_start,
                anchor_q_indices,
                anchor_prefix_len,
                global_k_block,
                kv_head,
                qhead_start,
                block_flat_start,
                softmax_scale,
                scale_log2,
                tidx,
                self.num_threads,
            )


@cute.jit
def _run_hsa_bwd_monolithic_scalar_kernel_body_without_sentence_full(
    self,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    sentence_full_kblock_row_ptr: cute.Tensor,
    sentence_full_q_start: cute.Tensor,
    sentence_full_q_len: cute.Tensor,
    sentence_full_q_group_mask: cute.Tensor,
    sentence_full_k_local_start: cute.Tensor,
    sentence_full_k_len: cute.Tensor,
    sentence_tail_kblock_row_ptr: cute.Tensor,
    sentence_tail_q_start: cute.Tensor,
    sentence_tail_q_len: cute.Tensor,
    sentence_tail_q_group_mask: cute.Tensor,
    sentence_tail_k_local_start: cute.Tensor,
    sentence_tail_k_len: cute.Tensor,
    sentence_tail_row0_prefix_len: cute.Tensor,
    anchor_full_kblock_row_ptr: cute.Tensor,
    anchor_full_q_row_start: cute.Tensor,
    anchor_full_q_row_count: cute.Tensor,
    anchor_full_q_group_mask: cute.Tensor,
    anchor_full_k_local_start: cute.Tensor,
    anchor_full_k_len: cute.Tensor,
    anchor_tail_kblock_row_ptr: cute.Tensor,
    anchor_tail_q_row_start: cute.Tensor,
    anchor_tail_q_row_count: cute.Tensor,
    anchor_tail_q_group_mask: cute.Tensor,
    anchor_tail_k_local_start: cute.Tensor,
    anchor_tail_k_len: cute.Tensor,
    anchor_tail_prefix_row_start: cute.Tensor,
    anchor_q_indices: cute.Tensor,
    anchor_prefix_len: cute.Tensor,
    blocks_per_batch: Int32,
    seqlen: Int32,
    softmax_scale: Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    k_block_idx, kv_head, batch_idx = cute.arch.block_idx()
    if tidx == 0:
        global_k_block = batch_idx * blocks_per_batch + k_block_idx
        block_flat_start = batch_idx * seqlen + k_block_idx * self.k_block_size
        qhead_start = kv_head * self.qhead_per_kvhead
        scale_log2 = Float32(softmax_scale * math.log2(math.e))


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


@dataclass
class HSABwdPreparedTensors:
    q_flat: torch.Tensor
    k_flat: torch.Tensor
    v_flat: torch.Tensor
    out_flat: torch.Tensor
    dout_flat: torch.Tensor
    total_lse_flat: torch.Tensor
    dout_float_flat: Optional[torch.Tensor] = None
    lse_log2_flat: Optional[torch.Tensor] = None
    dpsum_flat: Optional[torch.Tensor] = None


@dataclass
class HSABwdPrepareBuffers:
    total_lse_flat: torch.Tensor
    lse_log2_flat: torch.Tensor
    dpsum_flat: torch.Tensor


@dataclass
class HSABwdSentenceBuffers:
    q_stream: torch.Tensor
    k_stream: torch.Tensor
    v_stream: torch.Tensor
    out_stream: torch.Tensor
    dout_stream: torch.Tensor
    dq_accum_rows: torch.Tensor
    dk_accum_rows: torch.Tensor
    dv_accum_rows: torch.Tensor


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _zero_tensor_list_(tensors: list[Optional[torch.Tensor]]) -> None:
    live_tensors = [tensor for tensor in tensors if tensor is not None]
    if not live_tensors:
        return
    foreach_zero = getattr(torch, "_foreach_zero_", None)
    if foreach_zero is not None:
        foreach_zero(live_tensors)
    else:
        for tensor in live_tensors:
            tensor.zero_()


def _get_hsa_bwd_sentence_buffers(
    runtime_state,
    prepared: HSABwdPreparedTensors,
) -> HSABwdSentenceBuffers:
    cache = getattr(runtime_state, "_hsa_bwd_sentence_buffer_cache", None)
    if cache is None:
        cache = {}
        setattr(runtime_state, "_hsa_bwd_sentence_buffer_cache", cache)

    q_flat = prepared.q_flat
    k_flat = prepared.k_flat
    v_flat = prepared.v_flat
    sentence_stream = runtime_state["sentence_stream"]
    sentence_q_rows = int(sentence_stream.query_indices.numel())
    sentence_k_rows = int(sentence_stream.key_indices.numel())
    cache_key = (
        str(q_flat.device),
        q_flat.dtype,
        q_flat.shape[0],
        q_flat.shape[1],
        q_flat.shape[2],
        k_flat.shape[1],
        v_flat.shape[2],
        sentence_q_rows,
        sentence_k_rows,
    )
    buffers = cache.get(cache_key)
    if buffers is None:
        total_rows = q_flat.shape[0]
        num_q_heads = q_flat.shape[1]
        head_dim = q_flat.shape[2]
        num_kv_heads = k_flat.shape[1]
        head_dim_v = v_flat.shape[2]
        buffers = HSABwdSentenceBuffers(
            q_stream=torch.empty((sentence_q_rows, num_q_heads, head_dim), dtype=q_flat.dtype, device=q_flat.device),
            k_stream=torch.empty((sentence_k_rows, num_kv_heads, head_dim), dtype=k_flat.dtype, device=k_flat.device),
            v_stream=torch.empty((sentence_k_rows, num_kv_heads, head_dim_v), dtype=v_flat.dtype, device=v_flat.device),
            out_stream=torch.empty((sentence_q_rows, num_q_heads, head_dim_v), dtype=prepared.out_flat.dtype, device=q_flat.device),
            dout_stream=torch.empty((sentence_q_rows, num_q_heads, head_dim_v), dtype=prepared.dout_flat.dtype, device=q_flat.device),
            dq_accum_rows=torch.empty((total_rows, num_q_heads, head_dim), dtype=torch.float32, device=q_flat.device),
            dk_accum_rows=torch.empty((total_rows, num_kv_heads, head_dim), dtype=torch.float32, device=q_flat.device),
            dv_accum_rows=torch.empty((total_rows, num_kv_heads, head_dim_v), dtype=torch.float32, device=q_flat.device),
        )
        cache[cache_key] = buffers
    _zero_tensor_list_([buffers.dq_accum_rows, buffers.dk_accum_rows, buffers.dv_accum_rows])
    return buffers


def _get_hsa_bwd_prepare_buffers(
    q: torch.Tensor,
    v: torch.Tensor,
) -> HSABwdPrepareBuffers:
    cache = run_hsa_bwd_sm100_monolithic.prepare_buffer_cache
    batch_size, seqlen, num_q_heads = q.shape[:3]
    head_dim_v = v.shape[3]
    total_rows = batch_size * seqlen
    cache_key = (
        str(q.device),
        total_rows,
        num_q_heads,
        head_dim_v,
    )
    buffers = cache.get(cache_key)
    if buffers is None:
        buffers = HSABwdPrepareBuffers(
            total_lse_flat=torch.empty((total_rows, num_q_heads), dtype=torch.float32, device=q.device),
            lse_log2_flat=torch.empty((total_rows, num_q_heads), dtype=torch.float32, device=q.device),
            dpsum_flat=torch.empty((total_rows, num_q_heads), dtype=torch.float32, device=q.device),
        )
        cache[cache_key] = buffers
    return buffers


def _get_cached_hsa_bwd_monolithic_workspaces(
    plan: HSAMonolithicBackwardLaunchPlan,
    *,
    device: torch.device | str,
    cache_key,
) -> dict[str, Optional[torch.Tensor]]:
    cache = run_hsa_bwd_sm100_monolithic.workspace_cache
    full_key = (str(device), cache_key)
    workspaces = cache.get(full_key)
    if workspaces is None:
        workspaces = _allocate_hsa_bwd_monolithic_workspaces(plan, device=device)
        cache[full_key] = workspaces
    return workspaces


def _run_hsa_sentence_pack_outputs_kernel(
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    total_lse_flat: torch.Tensor,
    sentence_lse: torch.Tensor,
    sentence_row_idx: torch.Tensor,
    out_stream: torch.Tensor,
    dout_stream: torch.Tensor,
) -> None:
    if sentence_row_idx.numel() == 0:
        return
    compile_key = (
        "sentence_pack_outputs",
        out_flat.dtype,
        out_flat.shape[1],
        out_flat.shape[2],
        torch.cuda.get_device_capability(out_flat.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        pack_outputs = FlashHSASentencePackOutputsSm100()
        compile_args = [
            to_cute_tensor(out_flat),
            to_cute_tensor(dout_flat),
            to_cute_tensor(total_lse_flat, assumed_align=4),
            to_cute_tensor(sentence_lse, assumed_align=4),
            to_cute_tensor(sentence_row_idx, assumed_align=4),
            to_cute_tensor(out_stream),
            to_cute_tensor(dout_stream),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            pack_outputs,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        out_flat,
        dout_flat,
        total_lse_flat,
        sentence_lse,
        sentence_row_idx,
        out_stream,
        dout_stream,
        current_stream,
    )


def _run_hsa_sentence_scatter_rows_kernel(
    packed_src: torch.Tensor,
    row_idx: torch.Tensor,
    dst_rows: torch.Tensor,
) -> None:
    if row_idx.numel() == 0:
        return
    compile_key = (
        "sentence_scatter_rows",
        packed_src.dtype,
        packed_src.shape[1],
        packed_src.shape[2],
        dst_rows.dtype,
        torch.cuda.get_device_capability(packed_src.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        scatter_rows = FlashHSASentenceScatterRowsSm100()
        compile_args = [
            to_cute_tensor(packed_src),
            to_cute_tensor(row_idx, assumed_align=4),
            to_cute_tensor(dst_rows),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            scatter_rows,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        packed_src,
        row_idx,
        dst_rows,
        current_stream,
    )


def _run_hsa_pack_rows_kernel(
    src_rows: torch.Tensor,
    row_idx: torch.Tensor,
    dst_packed: torch.Tensor,
) -> None:
    if row_idx.numel() == 0:
        return
    compile_key = (
        "pack_rows",
        src_rows.dtype,
        src_rows.shape[1],
        src_rows.shape[2],
        dst_packed.dtype,
        torch.cuda.get_device_capability(src_rows.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        pack_rows = FlashHSAPackRowsSm100()
        compile_args = [
            to_cute_tensor(src_rows),
            to_cute_tensor(row_idx, assumed_align=4),
            to_cute_tensor(dst_packed),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            pack_rows,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        src_rows,
        row_idx,
        dst_packed,
        current_stream,
    )


def _run_hsa_pack_lse_kernel(
    lse: torch.Tensor,
    total_lse_flat: torch.Tensor,
) -> None:
    compile_key = (
        "pack_lse",
        lse.dtype,
        lse.shape[1],
        lse.shape[2],
        total_lse_flat.dtype,
        torch.cuda.get_device_capability(lse.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        pack_lse = FlashHSAPackLSESm100()
        compile_args = [
            to_cute_tensor(lse),
            to_cute_tensor(total_lse_flat, assumed_align=4),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            pack_lse,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        lse,
        total_lse_flat,
        current_stream,
    )


def _run_hsa_prepare_aux_kernel(
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    total_lse_flat: torch.Tensor,
    lse_log2_flat: torch.Tensor,
    dpsum_flat: torch.Tensor,
) -> None:
    compile_key = (
        "prepare_aux",
        out.dtype,
        out.shape[2],
        out.shape[3],
        total_lse_flat.dtype,
        torch.cuda.get_device_capability(out.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        prepare_aux = FlashHSAPrepareAuxSm100()
        compile_args = [
            to_cute_tensor(out),
            to_cute_tensor(dout),
            to_cute_tensor(lse),
            to_cute_tensor(total_lse_flat, assumed_align=4),
            to_cute_tensor(lse_log2_flat, assumed_align=4),
            to_cute_tensor(dpsum_flat, assumed_align=4),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            prepare_aux,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        out,
        dout,
        lse,
        total_lse_flat,
        lse_log2_flat,
        dpsum_flat,
        current_stream,
    )


def _run_hsa_prepare_monolithic_workspaces_kernel(
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    dpsum_workspace: torch.Tensor,
    lse_log2_workspace: torch.Tensor,
) -> None:
    compile_key = (
        "prepare_monolithic_workspaces",
        out.dtype,
        out.shape[2],
        out.shape[3],
        dpsum_workspace.dtype,
        torch.cuda.get_device_capability(out.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        prepare_workspaces = FlashHSAPrepareMonolithicWorkspacesSm100()
        compile_args = [
            to_cute_tensor(out),
            to_cute_tensor(dout),
            to_cute_tensor(lse),
            to_cute_tensor(dpsum_workspace, assumed_align=4),
            to_cute_tensor(lse_log2_workspace, assumed_align=4),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            prepare_workspaces,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        out,
        dout,
        lse,
        dpsum_workspace,
        lse_log2_workspace,
        current_stream,
    )


def _run_hsa_cast_rows_kernel(
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
) -> None:
    compile_key = (
        "cast_rows",
        src_rows.dtype,
        src_rows.shape[1],
        src_rows.shape[2],
        dst_rows.dtype,
        torch.cuda.get_device_capability(src_rows.device),
    )
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        cast_rows = FlashHSACastRowsSm100()
        compile_args = [
            to_cute_tensor(src_rows),
            to_cute_tensor(dst_rows),
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            cast_rows,
            *compile_args,
            options="--enable-tvm-ffi",
        )
    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        src_rows,
        dst_rows,
        current_stream,
    )


def _cast_hsa_row_accums_to_outputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _run_hsa_cast_rows_kernel(dq_accum_rows, dq.view_as(q).reshape_as(dq_accum_rows))
    _run_hsa_cast_rows_kernel(dk_accum_rows, dk.view_as(k).reshape_as(dk_accum_rows))
    _run_hsa_cast_rows_kernel(dv_accum_rows, dv.view_as(v).reshape_as(dv_accum_rows))
    return dq, dk, dv


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
    tile_m = 64
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
        workspaces["dk_accum"] = torch.empty(plan.dk_accum_shape, dtype=torch.float32, device=device)
    if plan.dv_accum_shape is not None:
        workspaces["dv_accum"] = torch.empty(plan.dv_accum_shape, dtype=torch.float32, device=device)
    if plan.dQ_semaphore_shape is not None:
        workspaces["dQ_semaphore"] = torch.empty(plan.dQ_semaphore_shape, dtype=torch.int32, device=device)
    if plan.dK_semaphore_shape is not None:
        workspaces["dK_semaphore"] = torch.empty(plan.dK_semaphore_shape, dtype=torch.int32, device=device)
    if plan.dV_semaphore_shape is not None:
        workspaces["dV_semaphore"] = torch.empty(plan.dV_semaphore_shape, dtype=torch.int32, device=device)
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
    _zero_tensor_list_(
        [
            workspaces["dq_accum"],
            workspaces["dk_accum"],
            workspaces["dv_accum"],
        ]
    )
    _zero_tensor_list_(
        [
            workspaces["dQ_semaphore"],
            workspaces["dK_semaphore"],
            workspaces["dV_semaphore"],
        ]
    )

    _run_hsa_prepare_monolithic_workspaces_kernel(
        out,
        dout,
        lse,
        workspaces["dpsum"],
        workspaces["lse_log2"],
    )


@cute.jit
def _accumulate_hsa_bwd_row(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    global_q_row: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    key_row_base: Int32,
    prefix: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    qhead_per_kvhead: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    head_dim_v: cutlass.Constexpr[int],
):
    if prefix > 0:
        for qh_offset in cutlass.range(qhead_per_kvhead, unroll=1):
            q_head = qhead_start + qh_offset
            lse_log2 = Float32(mLSElog2[global_q_row, q_head])
            dpsum = Float32(mdPsum[global_q_row, q_head])
            for k_rel in cutlass.range(prefix, unroll=1):
                key_row = key_row_base + k_rel
                score = Float32.zero
                for d in cutlass.range(head_dim, unroll=1):
                    score += Float32(mQ[global_q_row, q_head, d]) * Float32(mK[key_row, kv_head, d])
                prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                dprob = Float32.zero
                for dv_idx in cutlass.range(head_dim_v, unroll=1):
                    dprob += Float32(mdO[global_q_row, q_head, dv_idx]) * Float32(
                        mV[key_row, kv_head, dv_idx]
                    )
                ds = prob * (dprob - dpsum)
                ds_scaled = ds * softmax_scale
                for d in cutlass.range(head_dim, unroll=1):
                    utils.atomic_add_fp32(
                        ds_scaled * Float32(mK[key_row, kv_head, d]),
                        utils.elem_pointer(mdQaccum, (global_q_row, q_head, d)),
                    )
                    mdK[key_row, kv_head, d] = Float32(mdK[key_row, kv_head, d]) + ds_scaled * Float32(
                        mQ[global_q_row, q_head, d]
                    )
                for dv_idx in cutlass.range(head_dim_v, unroll=1):
                    mdV[key_row, kv_head, dv_idx] = Float32(mdV[key_row, kv_head, dv_idx]) + prob * Float32(
                        mdO[global_q_row, q_head, dv_idx]
                    )


@cute.jit
def _accumulate_hsa_bwd_row_atomic_dkv(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mdO: cute.Tensor,
    mLSElog2: cute.Tensor,
    mdPsum: cute.Tensor,
    mdQaccum: cute.Tensor,
    mdK: cute.Tensor,
    mdV: cute.Tensor,
    global_q_row: Int32,
    kv_head: Int32,
    qhead_start: Int32,
    key_row_base: Int32,
    prefix: Int32,
    softmax_scale: Float32,
    scale_log2: Float32,
    qhead_per_kvhead: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    head_dim_v: cutlass.Constexpr[int],
):
    if prefix > 0:
        for qh_offset in cutlass.range(qhead_per_kvhead, unroll=1):
            q_head = qhead_start + qh_offset
            lse_log2 = Float32(mLSElog2[global_q_row, q_head])
            dpsum = Float32(mdPsum[global_q_row, q_head])
            for k_rel in cutlass.range(prefix, unroll=1):
                key_row = key_row_base + k_rel
                score = Float32.zero
                for d in cutlass.range(head_dim, unroll=1):
                    score += Float32(mQ[global_q_row, q_head, d]) * Float32(mK[key_row, kv_head, d])
                prob = cute.math.exp2(score * scale_log2 - lse_log2, fastmath=True)
                dprob = Float32.zero
                for dv_idx in cutlass.range(head_dim_v, unroll=1):
                    dprob += Float32(mdO[global_q_row, q_head, dv_idx]) * Float32(
                        mV[key_row, kv_head, dv_idx]
                    )
                ds = prob * (dprob - dpsum)
                ds_scaled = ds * softmax_scale
                for d in cutlass.range(head_dim, unroll=1):
                    utils.atomic_add_fp32(
                        ds_scaled * Float32(mK[key_row, kv_head, d]),
                        utils.elem_pointer(mdQaccum, (global_q_row, q_head, d)),
                    )
                    utils.atomic_add_fp32(
                        ds_scaled * Float32(mQ[global_q_row, q_head, d]),
                        utils.elem_pointer(mdK, (key_row, kv_head, d)),
                    )
                for dv_idx in cutlass.range(head_dim_v, unroll=1):
                    utils.atomic_add_fp32(
                        prob * Float32(mdO[global_q_row, q_head, dv_idx]),
                        utils.elem_pointer(mdV, (key_row, kv_head, dv_idx)),
                    )


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
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[1],
        k.shape[2],
        q.shape[-1],
        v.shape[-1],
        qhead_per_kvhead,
        monolithic_schedule.k_block_size,
        monolithic_schedule.anchor_row_panel_size,
        deterministic,
        torch.cuda.get_device_capability(q.device),
        num_q_heads == num_kv_heads,
    )


def _build_hsa_bwd_rowgroup_compile_key(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    runtime,
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
        runtime.backward_block_q,
        runtime.backward_block_k,
        deterministic,
        torch.cuda.get_device_capability(q.device),
        num_q_heads == num_kv_heads,
    )


def _run_hsa_bwd_rowgroup_main_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    runtime,
    softmax_scale: float,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if runtime.backward_block_q != 64 or runtime.backward_block_k not in (64, 128):
        raise NotImplementedError("Row-group HSA backward currently requires exact 64x{64,128} backward runtime tiles")

    total_rows = schedule.num_rows
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[-1]
    head_dim_v = v.shape[-1]
    blocks_per_batch = int(runtime.backward_sparse.mask_block_cnt.shape[2])

    q_flat = q.reshape(total_rows, num_q_heads, head_dim).contiguous()
    k_flat = k.reshape(total_rows, num_kv_heads, head_dim).contiguous()
    v_flat = v.reshape(total_rows, num_kv_heads, head_dim_v).contiguous()
    dout_flat = dout.reshape(total_rows, num_q_heads, head_dim_v).contiguous()
    dpsum_flat = (out.float() * dout.float()).sum(dim=-1).permute(0, 2, 1).contiguous().view(total_rows, num_q_heads)
    lse_log2_flat = (lse.float() * math.log2(math.e)).permute(0, 2, 1).contiguous().view(total_rows, num_q_heads)
    dq_accum_rows = torch.zeros((total_rows, num_q_heads, head_dim), dtype=torch.float32, device=q.device)
    dk_accum_rows = torch.zeros((total_rows, num_kv_heads, head_dim), dtype=torch.float32, device=q.device)
    dv_accum_rows = torch.zeros((total_rows, num_kv_heads, head_dim_v), dtype=torch.float32, device=q.device)

    compile_key = _build_hsa_bwd_rowgroup_compile_key(q, k, v, runtime, deterministic)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        rowgroup_bwd = FlashHSABackwardRowGroupSm100(
            q.shape[-1],
            v.shape[-1],
            qhead_per_kvhead=q.shape[2] // k.shape[2],
            q_block_size=runtime.backward_block_q,
            k_block_size=runtime.backward_block_k,
        )
        compile_args = [
            to_cute_tensor(q_flat),
            to_cute_tensor(k_flat),
            to_cute_tensor(v_flat),
            to_cute_tensor(dout_flat),
            to_cute_tensor(lse_log2_flat, assumed_align=4),
            to_cute_tensor(dpsum_flat, assumed_align=4),
            to_cute_tensor(dq_accum_rows),
            to_cute_tensor(dk_accum_rows),
            to_cute_tensor(dv_accum_rows),
            to_cute_tensor(runtime.backward_sparse.mask_block_cnt, assumed_align=4),
            to_cute_tensor(runtime.backward_sparse.mask_block_idx, assumed_align=4),
            to_cute_tensor(runtime.backward_sparse.full_block_cnt, assumed_align=4),
            to_cute_tensor(runtime.backward_sparse.full_block_idx, assumed_align=4),
            to_cute_tensor(runtime.backward_packed_masks.block_id_table, assumed_align=4),
            to_cute_tensor(runtime.backward_packed_masks.mask_words, assumed_align=4),
            to_cute_tensor(runtime.backward_packed_masks.row_group_nonempty, assumed_align=4),
            blocks_per_batch,
            int(schedule.seqlen),
            softmax_scale,
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            rowgroup_bwd,
            *compile_args,
            options="--enable-tvm-ffi",
        )

    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        q_flat,
        k_flat,
        v_flat,
        dout_flat,
        lse_log2_flat,
        dpsum_flat,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
        runtime.backward_sparse.mask_block_cnt,
        runtime.backward_sparse.mask_block_idx,
        runtime.backward_sparse.full_block_cnt,
        runtime.backward_sparse.full_block_idx,
        runtime.backward_packed_masks.block_id_table,
        runtime.backward_packed_masks.mask_words,
        runtime.backward_packed_masks.row_group_nonempty,
        blocks_per_batch,
        int(schedule.seqlen),
        softmax_scale,
        current_stream,
    )
    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)


def _run_hsa_bwd_monolithic_main_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    schedule,
    monolithic_schedule,
    launch_plan: HSAMonolithicBackwardLaunchPlan,
    softmax_scale: float,
    workspaces: dict[str, Optional[torch.Tensor]],
    compile_key,
    precomputed_row_accums: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    precomputed_done_families: frozenset[str] = frozenset(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = schedule.num_rows
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[-1]
    head_dim_v = v.shape[-1]
    seqlen = q.shape[1]

    q_flat = q.reshape(total_rows, num_q_heads, head_dim).contiguous()
    k_flat = k.reshape(total_rows, num_kv_heads, head_dim).contiguous()
    v_flat = v.reshape(total_rows, num_kv_heads, head_dim_v).contiguous()
    out_flat = out.reshape(total_rows, num_q_heads, head_dim_v).contiguous()
    dout_flat = dout.reshape(total_rows, num_q_heads, head_dim_v).contiguous()
    dpsum_flat = workspaces["dpsum"][:, :, :seqlen].permute(0, 2, 1).contiguous().view(total_rows, num_q_heads)
    lse_log2_flat = (
        workspaces["lse_log2"][:, :, :seqlen].permute(0, 2, 1).contiguous().view(total_rows, num_q_heads)
    )
    use_true_fused = _use_hsa_true_fused_bwd()
    if precomputed_row_accums is None:
        dq_accum_rows = torch.zeros((total_rows, num_q_heads, head_dim), dtype=torch.float32, device=q.device)
        dk_accum_rows = torch.zeros((total_rows, num_kv_heads, head_dim), dtype=torch.float32, device=q.device)
        dv_accum_rows = torch.zeros((total_rows, num_kv_heads, head_dim_v), dtype=torch.float32, device=q.device)
        sentence_full_done = False
        sentence_tail_done = False
        if not use_true_fused:
            panel_batches = _build_hsa_monolithic_panel_batches(schedule, monolithic_schedule)
            if panel_batches["sentence_full"] is not None:
                if _use_hsa_sentence_full_kernel_fastpath():
                    sentence_full_done = _run_hsa_sentence_full_kernel_fa4_slice(
                        q_flat,
                        k_flat,
                        v_flat,
                        out_flat,
                        dout_flat,
                        lse_log2_flat / math.log2(math.e),
                        panel_batches["sentence_full"],
                        softmax_scale,
                        launch_plan.deterministic,
                        dq_accum_rows,
                        dk_accum_rows,
                        dv_accum_rows,
                    )
                else:
                    sentence_full_done = _run_hsa_sentence_full_fa4_fastpath(
                        q_flat,
                        k_flat,
                        v_flat,
                        out_flat,
                        dout_flat,
                        lse_log2_flat / math.log2(math.e),
                        panel_batches["sentence_full"],
                        softmax_scale,
                        launch_plan.deterministic,
                        dq_accum_rows,
                        dk_accum_rows,
                        dv_accum_rows,
                    )
    else:
        dq_accum_rows, dk_accum_rows, dv_accum_rows = precomputed_row_accums
        sentence_full_done = "sentence_full" in precomputed_done_families
        sentence_tail_done = "sentence_tail" in precomputed_done_families
    sentence_full_kblock_row_ptr = monolithic_schedule.sentence_full_kblock_row_ptr
    if sentence_full_done:
        sentence_full_kblock_row_ptr = torch.zeros_like(sentence_full_kblock_row_ptr)
    sentence_tail_kblock_row_ptr = monolithic_schedule.sentence_tail_kblock_row_ptr
    if sentence_tail_done:
        sentence_tail_kblock_row_ptr = torch.zeros_like(sentence_tail_kblock_row_ptr)
    anchor_full_kblock_row_ptr = monolithic_schedule.anchor_full_kblock_row_ptr
    anchor_tail_kblock_row_ptr = monolithic_schedule.anchor_tail_kblock_row_ptr
    sentence_full_active = (not sentence_full_done) and _monolithic_has_sentence_full_desc(monolithic_schedule)
    sentence_tail_active = (not sentence_tail_done) and _monolithic_has_sentence_tail_desc(monolithic_schedule)
    anchor_full_active = _monolithic_has_anchor_full_desc(monolithic_schedule)
    anchor_tail_active = _monolithic_has_anchor_tail_desc(monolithic_schedule)
    if not (sentence_full_active or sentence_tail_active or anchor_full_active or anchor_tail_active):
        return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if not use_true_fused and not sentence_full_active and not sentence_tail_active:
        compile_key = ("anchor_only",) + compile_key
        if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
            anchor_bwd = FlashHSAAnchorBackwardSm100(
                q.shape[-1],
                v.shape[-1],
                qhead_per_kvhead=q.shape[2] // k.shape[2],
                k_block_size=monolithic_schedule.k_block_size,
            )
            compile_args = [
                to_cute_tensor(q_flat),
                to_cute_tensor(k_flat),
                to_cute_tensor(v_flat),
                to_cute_tensor(dout_flat),
                to_cute_tensor(lse_log2_flat, assumed_align=4),
                to_cute_tensor(dpsum_flat, assumed_align=4),
                to_cute_tensor(dq_accum_rows),
                to_cute_tensor(dk_accum_rows),
                to_cute_tensor(dv_accum_rows),
                to_cute_tensor(anchor_full_kblock_row_ptr, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_full_q_row_start, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_full_q_row_count, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_full_q_group_mask, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_full_k_local_start, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_full_k_len, assumed_align=4),
                to_cute_tensor(anchor_tail_kblock_row_ptr, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_tail_q_row_start, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_tail_q_row_count, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_tail_q_group_mask, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_tail_k_local_start, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_tail_k_len, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_tail_prefix_row_start, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_q_indices, assumed_align=4),
                to_cute_tensor(monolithic_schedule.anchor_prefix_len, assumed_align=4),
                int(monolithic_schedule.blocks_per_batch),
                int(schedule.seqlen),
                softmax_scale,
                current_stream,
            ]
            run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
                anchor_bwd,
                *compile_args,
                options="--enable-tvm-ffi",
            )

        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
            q_flat,
            k_flat,
            v_flat,
            dout_flat,
            lse_log2_flat,
            dpsum_flat,
            dq_accum_rows,
            dk_accum_rows,
            dv_accum_rows,
            anchor_full_kblock_row_ptr,
            monolithic_schedule.anchor_full_q_row_start,
            monolithic_schedule.anchor_full_q_row_count,
            monolithic_schedule.anchor_full_q_group_mask,
            monolithic_schedule.anchor_full_k_local_start,
            monolithic_schedule.anchor_full_k_len,
            anchor_tail_kblock_row_ptr,
            monolithic_schedule.anchor_tail_q_row_start,
            monolithic_schedule.anchor_tail_q_row_count,
            monolithic_schedule.anchor_tail_q_group_mask,
            monolithic_schedule.anchor_tail_k_local_start,
            monolithic_schedule.anchor_tail_k_len,
            monolithic_schedule.anchor_tail_prefix_row_start,
            monolithic_schedule.anchor_q_indices,
            monolithic_schedule.anchor_prefix_len,
            int(monolithic_schedule.blocks_per_batch),
            int(schedule.seqlen),
            softmax_scale,
            current_stream,
        )
        return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)

    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        fa_bwd = FlashHSABackwardSm100(
            q.shape[-1],
            v.shape[-1],
            qhead_per_kvhead=q.shape[2] // k.shape[2],
            k_block_size=monolithic_schedule.k_block_size,
            anchor_row_panel_size=monolithic_schedule.anchor_row_panel_size,
            deterministic=launch_plan.deterministic,
        )
        compile_args = [
            to_cute_tensor(q_flat),
            to_cute_tensor(k_flat),
            to_cute_tensor(v_flat),
            to_cute_tensor(out_flat),
            to_cute_tensor(dout_flat),
            to_cute_tensor(lse_log2_flat, assumed_align=4),
            to_cute_tensor(dpsum_flat, assumed_align=4),
            to_cute_tensor(dq_accum_rows),
            to_cute_tensor(dk_accum_rows),
            to_cute_tensor(dv_accum_rows),
            to_cute_tensor(sentence_full_kblock_row_ptr, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_full_q_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_full_q_len, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_full_q_group_mask, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_full_k_local_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_full_k_len, assumed_align=4),
            to_cute_tensor(sentence_tail_kblock_row_ptr, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_tail_q_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_tail_q_len, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_tail_q_group_mask, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_tail_k_local_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_tail_k_len, assumed_align=4),
            to_cute_tensor(monolithic_schedule.sentence_tail_row0_prefix_len, assumed_align=4),
            to_cute_tensor(anchor_full_kblock_row_ptr, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_q_row_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_q_row_count, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_q_group_mask, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_k_local_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_k_len, assumed_align=4),
            to_cute_tensor(anchor_tail_kblock_row_ptr, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_q_row_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_q_row_count, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_q_group_mask, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_k_local_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_k_len, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_prefix_row_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_q_indices, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_prefix_len, assumed_align=4),
            int(monolithic_schedule.blocks_per_batch),
            int(schedule.seqlen),
            softmax_scale,
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            fa_bwd,
            *compile_args,
            options="--enable-tvm-ffi",
        )

    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_log2_flat,
        dpsum_flat,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
        sentence_full_kblock_row_ptr,
        monolithic_schedule.sentence_full_q_start,
        monolithic_schedule.sentence_full_q_len,
        monolithic_schedule.sentence_full_q_group_mask,
        monolithic_schedule.sentence_full_k_local_start,
        monolithic_schedule.sentence_full_k_len,
        sentence_tail_kblock_row_ptr,
        monolithic_schedule.sentence_tail_q_start,
        monolithic_schedule.sentence_tail_q_len,
        monolithic_schedule.sentence_tail_q_group_mask,
        monolithic_schedule.sentence_tail_k_local_start,
        monolithic_schedule.sentence_tail_k_len,
        monolithic_schedule.sentence_tail_row0_prefix_len,
        anchor_full_kblock_row_ptr,
        monolithic_schedule.anchor_full_q_row_start,
        monolithic_schedule.anchor_full_q_row_count,
        monolithic_schedule.anchor_full_q_group_mask,
        monolithic_schedule.anchor_full_k_local_start,
        monolithic_schedule.anchor_full_k_len,
        anchor_tail_kblock_row_ptr,
        monolithic_schedule.anchor_tail_q_row_start,
        monolithic_schedule.anchor_tail_q_row_count,
        monolithic_schedule.anchor_tail_q_group_mask,
        monolithic_schedule.anchor_tail_k_local_start,
        monolithic_schedule.anchor_tail_k_len,
        monolithic_schedule.anchor_tail_prefix_row_start,
        monolithic_schedule.anchor_q_indices,
        monolithic_schedule.anchor_prefix_len,
        int(monolithic_schedule.blocks_per_batch),
        int(schedule.seqlen),
        softmax_scale,
        current_stream,
    )

    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)


def _run_hsa_bwd_anchor_only_main_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    compile_key,
    precomputed_row_accums: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    prepared: Optional[HSABwdPreparedTensors] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prepared = (
        prepared if prepared is not None else _prepare_hsa_bwd_tensors(q, k, v, out, dout, lse, with_anchor_aux=True)
    )
    q_flat = prepared.q_flat
    k_flat = prepared.k_flat
    v_flat = prepared.v_flat
    dout_flat = prepared.dout_flat
    dpsum_flat = prepared.dpsum_flat
    lse_log2_flat = prepared.lse_log2_flat
    assert dpsum_flat is not None
    assert lse_log2_flat is not None
    dq_accum_rows, dk_accum_rows, dv_accum_rows = precomputed_row_accums
    anchor_full_kblock_row_ptr = monolithic_schedule.anchor_full_kblock_row_ptr
    anchor_tail_kblock_row_ptr = monolithic_schedule.anchor_tail_kblock_row_ptr
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compile_key = ("anchor_only", _get_hsa_anchor_kernel_threads(q.shape[2] // k.shape[2])) + compile_key
    if compile_key not in run_hsa_bwd_sm100_monolithic.compile_cache:
        anchor_bwd = FlashHSAAnchorBackwardSm100(
            q.shape[-1],
            v.shape[-1],
            qhead_per_kvhead=q.shape[2] // k.shape[2],
            k_block_size=monolithic_schedule.k_block_size,
        )
        compile_args = [
            to_cute_tensor(q_flat),
            to_cute_tensor(k_flat),
            to_cute_tensor(v_flat),
            to_cute_tensor(dout_flat),
            to_cute_tensor(lse_log2_flat, assumed_align=4),
            to_cute_tensor(dpsum_flat, assumed_align=4),
            to_cute_tensor(dq_accum_rows),
            to_cute_tensor(dk_accum_rows),
            to_cute_tensor(dv_accum_rows),
            to_cute_tensor(anchor_full_kblock_row_ptr, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_q_row_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_q_row_count, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_q_group_mask, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_k_local_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_full_k_len, assumed_align=4),
            to_cute_tensor(anchor_tail_kblock_row_ptr, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_q_row_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_q_row_count, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_q_group_mask, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_k_local_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_k_len, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_tail_prefix_row_start, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_q_indices, assumed_align=4),
            to_cute_tensor(monolithic_schedule.anchor_prefix_len, assumed_align=4),
            int(monolithic_schedule.blocks_per_batch),
            int(schedule.seqlen),
            softmax_scale,
            current_stream,
        ]
        run_hsa_bwd_sm100_monolithic.compile_cache[compile_key] = cute.compile(
            anchor_bwd,
            *compile_args,
            options="--enable-tvm-ffi",
        )

    run_hsa_bwd_sm100_monolithic.compile_cache[compile_key](
        q_flat,
        k_flat,
        v_flat,
        dout_flat,
        lse_log2_flat,
        dpsum_flat,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
        anchor_full_kblock_row_ptr,
        monolithic_schedule.anchor_full_q_row_start,
        monolithic_schedule.anchor_full_q_row_count,
        monolithic_schedule.anchor_full_q_group_mask,
        monolithic_schedule.anchor_full_k_local_start,
        monolithic_schedule.anchor_full_k_len,
        anchor_tail_kblock_row_ptr,
        monolithic_schedule.anchor_tail_q_row_start,
        monolithic_schedule.anchor_tail_q_row_count,
        monolithic_schedule.anchor_tail_q_group_mask,
        monolithic_schedule.anchor_tail_k_local_start,
        monolithic_schedule.anchor_tail_k_len,
        monolithic_schedule.anchor_tail_prefix_row_start,
        monolithic_schedule.anchor_q_indices,
        monolithic_schedule.anchor_prefix_len,
        int(monolithic_schedule.blocks_per_batch),
        int(schedule.seqlen),
        softmax_scale,
        current_stream,
    )

    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)


def _build_hsa_sentence_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    deterministic: bool,
    use_true_fused: bool,
    monolithic_schedule,
) -> FlashHSASentenceBackwardSm100:
    return FlashHSASentenceBackwardSm100(
        q.shape[-1],
        v.shape[-1],
        qhead_per_kvhead=q.shape[2] // k.shape[2],
        deterministic=deterministic,
        use_2cta_instrs=_should_force_hsa_sentence_varlen_2cta(
            q,
            k,
            use_true_fused=use_true_fused,
            monolithic_schedule=monolithic_schedule,
        ),
    )


def _run_hsa_bwd_monolithic_sentence_only(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
    *,
    use_true_fused: bool,
    sentence_lse_override: Optional[torch.Tensor],
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if _monolithic_has_remaining_anchor_families(monolithic_schedule):
        return None
    if use_true_fused:
        sentence_bwd = _build_hsa_sentence_backward(
            q,
            k,
            v,
            deterministic=deterministic,
            use_true_fused=True,
            monolithic_schedule=monolithic_schedule,
        )
        return sentence_bwd.run(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            sentence_lse_override=sentence_lse_override,
        )
    if _use_hsa_sentence_full_kernel_fastpath():
        return _run_hsa_bwd_sentence_families_direct(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            deterministic,
            sentence_lse_override=sentence_lse_override,
        )
    return None


def _precompute_hsa_bwd_monolithic_sentence_stage(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
    *,
    use_true_fused: bool,
    sentence_lse_override: Optional[torch.Tensor],
) -> HSAMonolithicSentenceStageResult:
    if not _monolithic_has_sentence_families(monolithic_schedule):
        return HSAMonolithicSentenceStageResult()
    prepared = _prepare_hsa_bwd_tensors(q, k, v, out, dout, lse, with_anchor_aux=True)
    runtime_state = _materialize_runtime_state(schedule)
    if use_true_fused:
        sentence_bwd = _build_hsa_sentence_backward(
            q,
            k,
            v,
            deterministic=deterministic,
            use_true_fused=True,
            monolithic_schedule=monolithic_schedule,
        )
        row_accums, done_families = sentence_bwd.run_rows(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            prepared=prepared,
            runtime_state=runtime_state,
            sentence_lse_override=sentence_lse_override,
        )
    else:
        row_accums, done_families = _run_hsa_bwd_sentence_families_direct_rows(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            deterministic,
            prepared=prepared,
            runtime_state=runtime_state,
            sentence_lse_override=sentence_lse_override,
        )
    return HSAMonolithicSentenceStageResult(
        row_accums=row_accums,
        done_families=done_families,
        prepared=prepared,
        runtime_state=runtime_state,
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
    sentence_lse: Optional[torch.Tensor] = None,
):
    if not _is_supported_packed_bwd(q, k, v):
        raise NotImplementedError("Monolithic HSA backward scaffold requires CUDA SM100+/fp16 or bf16 tensors")

    hsa_mod = _load_hsa_module()
    monolithic_schedule = hsa_mod._get_hsa_monolithic_backward_schedule(schedule)
    use_true_fused = _use_hsa_true_fused_bwd()
    sentence_lse_override = _normalize_sentence_lse_override(sentence_lse)

    # Stage 1: if sentence descriptors are the whole workload, stay on the
    # active sentence path and return before touching anchor or legacy setup.
    sentence_only_result = _run_hsa_bwd_monolithic_sentence_only(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        softmax_scale,
        deterministic,
        use_true_fused=use_true_fused,
        sentence_lse_override=sentence_lse_override,
    )
    if sentence_only_result is not None:
        return sentence_only_result

    # Stage 2: mixed schedules precompute sentence-family row accumulators once.
    sentence_stage = HSAMonolithicSentenceStageResult()
    if use_true_fused or _use_hsa_sentence_full_kernel_fastpath():
        sentence_stage = _precompute_hsa_bwd_monolithic_sentence_stage(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            deterministic,
            use_true_fused=use_true_fused,
            sentence_lse_override=sentence_lse_override,
        )

    compile_key = _build_hsa_bwd_monolithic_compile_key(q, k, v, monolithic_schedule, deterministic)
    sentence_full_done = (
        not _monolithic_has_sentence_full_desc(monolithic_schedule)
        or "sentence_full" in sentence_stage.done_families
    )
    sentence_tail_done = (
        not _monolithic_has_sentence_tail_desc(monolithic_schedule)
        or "sentence_tail" in sentence_stage.done_families
    )
    # Stage 3: once sentence work is done, keep mixed execution on the narrower
    # anchor-only kernel instead of rebuilding the full monolithic path.
    if (
        sentence_stage.row_accums is not None
        and sentence_full_done
        and sentence_tail_done
        and _monolithic_has_remaining_anchor_families(monolithic_schedule)
    ):
        return _run_hsa_bwd_anchor_only_main_cute(
            q,
            k,
            v,
            out,
            dout,
            lse,
            schedule,
            monolithic_schedule,
            softmax_scale,
            compile_key,
            sentence_stage.row_accums,
            prepared=sentence_stage.prepared,
        )
    # Stage 4: unresolved families fall back to the legacy mixed monolithic kernel.
    if compile_key not in run_hsa_bwd_sm100_monolithic.launch_plan_cache:
        run_hsa_bwd_sm100_monolithic.launch_plan_cache[compile_key] = _build_hsa_bwd_monolithic_launch_plan(
            q,
            k,
            v,
            monolithic_schedule,
            deterministic,
        )
    launch_plan = run_hsa_bwd_sm100_monolithic.launch_plan_cache[compile_key]
    workspaces = _get_cached_hsa_bwd_monolithic_workspaces(
        launch_plan,
        device=q.device,
        cache_key=compile_key,
    )
    _prepare_hsa_bwd_monolithic_workspaces(out, dout, lse, workspaces)
    return _run_hsa_bwd_monolithic_main_cute(
        q,
        k,
        v,
        out,
        dout,
        schedule,
        monolithic_schedule,
        launch_plan,
        softmax_scale,
        workspaces,
        compile_key,
        precomputed_row_accums=precomputed_row_accums,
        precomputed_done_families=precomputed_done_families,
    )


run_hsa_bwd_sm100_monolithic.compile_cache = get_jit_cache("hsa_bwd_monolithic")
run_hsa_bwd_sm100_monolithic.launch_plan_cache = {}
run_hsa_bwd_sm100_monolithic.workspace_cache = {}
run_hsa_bwd_sm100_monolithic.prepare_buffer_cache = {}


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
    lse_sel_base = lse_flat.index_select(0, q_indices.reshape(-1)).view(
        q_indices.shape[0], q_indices.shape[1], lse_flat.shape[1]
    ).permute(0, 2, 1)
    # CuTe requires the leading dimension to have unit stride. For size-1 query
    # panels, a plain permute/contiguous can still leave the trailing stride
    # non-unit, so normalize through an explicit destination tensor.
    lse_sel = torch.empty(lse_sel_base.shape, dtype=lse_sel_base.dtype, device=lse_sel_base.device)
    lse_sel.copy_(lse_sel_base)
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


def _make_hsa_batch(entries, *, device: torch.device | str):
    hsa_mod = _load_hsa_module()
    if not entries:
        return None

    max_q = max(len(q_rows) for q_rows, _, _ in entries)
    max_k = max(len(k_rows) for _, k_rows, _ in entries)
    q_indices = torch.zeros((len(entries), max_q), dtype=torch.int32, device=device)
    k_indices = torch.zeros((len(entries), max_k), dtype=torch.int32, device=device)
    q_length = torch.zeros(len(entries), dtype=torch.int32, device=device)
    k_length = torch.zeros(len(entries), dtype=torch.int32, device=device)
    prefix_len = torch.zeros((len(entries), max_q), dtype=torch.int32, device=device)

    for entry_idx, (q_rows, k_rows, prefixes) in enumerate(entries):
        q_len = len(q_rows)
        k_len = len(k_rows)
        q_length[entry_idx] = q_len
        k_length[entry_idx] = k_len
        q_indices[entry_idx, :q_len] = torch.tensor(q_rows, dtype=torch.int32, device=device)
        k_indices[entry_idx, :k_len] = torch.tensor(k_rows, dtype=torch.int32, device=device)
        prefix_len[entry_idx, :q_len] = torch.tensor(prefixes, dtype=torch.int32, device=device)

    return hsa_mod.HSAHybridBackwardBatch(
        q_indices=q_indices,
        k_indices=k_indices,
        q_length=q_length,
        k_length=k_length,
        prefix_len=prefix_len,
    )


def _build_hsa_monolithic_panel_batches(schedule, monolithic_schedule):
    cache = getattr(schedule, "_hsa_monolithic_panel_batch_cache", None)
    cache_key = (
        str(schedule.sentence_start.device),
        monolithic_schedule.k_block_size,
        monolithic_schedule.anchor_row_panel_size,
        int(monolithic_schedule.sentence_full_q_start.numel()),
        int(monolithic_schedule.sentence_tail_q_start.numel()),
        int(monolithic_schedule.anchor_full_q_row_start.numel()),
        int(monolithic_schedule.anchor_tail_q_row_start.numel()),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    device = schedule.sentence_start.device
    seqlen = schedule.seqlen
    families = {name: [] for name in ("sentence_full", "sentence_tail", "anchor_full", "anchor_tail")}

    for global_k_block in range(monolithic_schedule.num_k_blocks):
        batch_idx = global_k_block // monolithic_schedule.blocks_per_batch
        block_k_start = (global_k_block % monolithic_schedule.blocks_per_batch) * monolithic_schedule.k_block_size
        block_flat_start = batch_idx * seqlen + block_k_start

        start = int(monolithic_schedule.sentence_full_kblock_row_ptr[global_k_block].item())
        end = int(monolithic_schedule.sentence_full_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(start, end):
            q_start = int(monolithic_schedule.sentence_full_q_start[desc_idx].item())
            q_len = int(monolithic_schedule.sentence_full_q_len[desc_idx].item())
            k_local_start = int(monolithic_schedule.sentence_full_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.sentence_full_k_len[desc_idx].item())
            q_rows = list(range(batch_idx * seqlen + q_start, batch_idx * seqlen + q_start + q_len))
            k_rows = list(range(block_flat_start + k_local_start, block_flat_start + k_local_start + k_len))
            families["sentence_full"].append((q_rows, k_rows, [k_len] * q_len))

        start = int(monolithic_schedule.sentence_tail_kblock_row_ptr[global_k_block].item())
        end = int(monolithic_schedule.sentence_tail_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(start, end):
            q_start = int(monolithic_schedule.sentence_tail_q_start[desc_idx].item())
            q_len = int(monolithic_schedule.sentence_tail_q_len[desc_idx].item())
            k_local_start = int(monolithic_schedule.sentence_tail_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.sentence_tail_k_len[desc_idx].item())
            row0_prefix_len = int(monolithic_schedule.sentence_tail_row0_prefix_len[desc_idx].item())
            q_rows = list(range(batch_idx * seqlen + q_start, batch_idx * seqlen + q_start + q_len))
            k_rows = list(range(block_flat_start + k_local_start, block_flat_start + k_local_start + k_len))
            prefixes = [min(k_len, row0_prefix_len + q_off) for q_off in range(q_len)]
            families["sentence_tail"].append((q_rows, k_rows, prefixes))

        start = int(monolithic_schedule.anchor_full_kblock_row_ptr[global_k_block].item())
        end = int(monolithic_schedule.anchor_full_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(start, end):
            q_row_start = int(monolithic_schedule.anchor_full_q_row_start[desc_idx].item())
            q_row_count = int(monolithic_schedule.anchor_full_q_row_count[desc_idx].item())
            k_local_start = int(monolithic_schedule.anchor_full_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.anchor_full_k_len[desc_idx].item())
            q_rows = monolithic_schedule.anchor_q_indices[q_row_start : q_row_start + q_row_count].detach().cpu().tolist()
            k_rows = list(range(block_flat_start + k_local_start, block_flat_start + k_local_start + k_len))
            families["anchor_full"].append((q_rows, k_rows, [k_len] * q_row_count))

        start = int(monolithic_schedule.anchor_tail_kblock_row_ptr[global_k_block].item())
        end = int(monolithic_schedule.anchor_tail_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(start, end):
            q_row_start = int(monolithic_schedule.anchor_tail_q_row_start[desc_idx].item())
            q_row_count = int(monolithic_schedule.anchor_tail_q_row_count[desc_idx].item())
            k_local_start = int(monolithic_schedule.anchor_tail_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.anchor_tail_k_len[desc_idx].item())
            prefix_row_start = int(monolithic_schedule.anchor_tail_prefix_row_start[desc_idx].item())
            q_rows = monolithic_schedule.anchor_q_indices[q_row_start : q_row_start + q_row_count].detach().cpu().tolist()
            prefixes = monolithic_schedule.anchor_prefix_len[
                prefix_row_start : prefix_row_start + q_row_count
            ].detach().cpu().tolist()
            k_rows = list(range(block_flat_start + k_local_start, block_flat_start + k_local_start + k_len))
            families["anchor_tail"].append((q_rows, k_rows, prefixes))

    batches = {name: _make_hsa_batch(entries, device=device) for name, entries in families.items()}
    all_entries = []
    for name in ("sentence_full", "sentence_tail", "anchor_full", "anchor_tail"):
        all_entries.extend(families[name])
    sentence_family_entries = []
    for name in ("sentence_full", "sentence_tail"):
        sentence_family_entries.extend(families[name])
    remaining_entries = []
    for name in ("sentence_tail", "anchor_full", "anchor_tail"):
        remaining_entries.extend(families[name])
    batches["all"] = _make_hsa_batch(all_entries, device=device)
    batches["sentence_families"] = _make_hsa_batch(sentence_family_entries, device=device)
    batches["remaining"] = _make_hsa_batch(remaining_entries, device=device)
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_monolithic_panel_batch_cache", cache)
    cache[cache_key] = batches
    return batches


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


def _run_hsa_descriptor_mma_batches(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    if batch is None or batch.q_indices.numel() == 0:
        return False

    hsa_mod = _load_hsa_module()
    _, _, _, _, _, _, flash_attn_bwd = _lazy_cute_imports()
    q_indices, k_indices, q_sel, k_sel, v_sel, out_sel, dout_sel, lse_sel = _gather_batch_tensors(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
    )
    prefix_len = batch.prefix_len.contiguous()
    q_length_table = batch.q_length.unsqueeze(1).expand(-1, prefix_len.shape[1]).contiguous()
    k_length_table = batch.k_length.unsqueeze(1).expand(-1, prefix_len.shape[1]).contiguous()
    prefix_len.__leading_dim__ = 1
    q_length_table.__leading_dim__ = 1
    k_length_table.__leading_dim__ = 1

    dq, dk, dv = flash_attn_bwd(
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
        aux_tensors=[prefix_len, q_length_table, k_length_table],
    )
    q_valid = torch.arange(q_indices.shape[1], device=q_flat.device).view(1, -1) < batch.q_length.unsqueeze(1)
    k_valid = torch.arange(k_indices.shape[1], device=k_flat.device).view(1, -1) < batch.k_length.unsqueeze(1)
    dq_accum_rows.index_add_(0, q_indices[q_valid].long(), dq[q_valid].float())
    dk_accum_rows.index_add_(0, k_indices[k_valid].long(), dk[k_valid].float())
    dv_accum_rows.index_add_(0, k_indices[k_valid].long(), dv[k_valid].float())
    return True


def _run_hsa_monolithic_mma_batches(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
    )


def _run_hsa_sentence_full_kernel_fa4_slice(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    # Internal rollout bridge for FLASH_ATTN_HSA_USE_KERNEL_SENTENCE_FULL=1.
    # This keeps the opt-in sentence_full path on the real FA4 backward kernel
    # while the mixed-family monolithic kernel still owns sentence_tail and the
    # anchor families.
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
    )


def _monolithic_has_remaining_non_sentence_full(monolithic_schedule) -> bool:
    return (
        _monolithic_has_sentence_tail_desc(monolithic_schedule)
        or _monolithic_has_anchor_full_desc(monolithic_schedule)
        or _monolithic_has_anchor_tail_desc(monolithic_schedule)
    )


def _monolithic_has_remaining_anchor_families(monolithic_schedule) -> bool:
    return _monolithic_has_anchor_full_desc(monolithic_schedule) or _monolithic_has_anchor_tail_desc(
        monolithic_schedule
    )


def _prepare_hsa_bwd_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    *,
    with_anchor_aux: bool = False,
) -> HSABwdPreparedTensors:
    total_rows = q.shape[0] * q.shape[1]
    num_q_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[-1]
    head_dim_v = v.shape[-1]
    q_flat = q.reshape(total_rows, num_q_heads, head_dim).contiguous()
    k_flat = k.reshape(total_rows, num_kv_heads, head_dim).contiguous()
    v_flat = v.reshape(total_rows, num_kv_heads, head_dim_v).contiguous()
    out_flat = out.reshape(total_rows, num_q_heads, head_dim_v).contiguous()
    dout_flat = dout.reshape(total_rows, num_q_heads, head_dim_v).contiguous()
    aux_buffers = _get_hsa_bwd_prepare_buffers(q, v)
    total_lse_flat = aux_buffers.total_lse_flat
    lse_log2_flat = None
    dpsum_flat = None
    if with_anchor_aux:
        lse_log2_flat = aux_buffers.lse_log2_flat
        dpsum_flat = aux_buffers.dpsum_flat
        _run_hsa_prepare_aux_kernel(
            out,
            dout,
            lse,
            total_lse_flat,
            lse_log2_flat,
            dpsum_flat,
        )
    else:
        _run_hsa_pack_lse_kernel(
            lse,
            total_lse_flat,
        )
    return HSABwdPreparedTensors(
        q_flat=q_flat,
        k_flat=k_flat,
        v_flat=v_flat,
        out_flat=out_flat,
        dout_flat=dout_flat,
        dout_float_flat=None,
        total_lse_flat=total_lse_flat,
        lse_log2_flat=lse_log2_flat,
        dpsum_flat=dpsum_flat,
    )


def _run_hsa_bwd_sentence_rows_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
    prepared: Optional[HSABwdPreparedTensors] = None,
    runtime_state=None,
    sentence_lse_override: Optional[torch.Tensor] = None,
    *,
    force_2cta_sm100_varlen: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], frozenset[str]]:
    hsa_mod = _load_hsa_module()
    _, _, _, _, _, flash_attn_fwd, flash_attn_bwd = _lazy_cute_imports()

    prepared = prepared if prepared is not None else _prepare_hsa_bwd_tensors(q, k, v, out, dout, lse)
    q_flat = prepared.q_flat
    k_flat = prepared.k_flat
    v_flat = prepared.v_flat
    out_flat = prepared.out_flat
    dout_flat = prepared.dout_flat
    total_lse_flat = prepared.total_lse_flat

    runtime_state = runtime_state if runtime_state is not None else _materialize_runtime_state(schedule)
    buffers = _get_hsa_bwd_sentence_buffers(runtime_state, prepared)
    dq_accum_rows = buffers.dq_accum_rows
    dk_accum_rows = buffers.dk_accum_rows
    dv_accum_rows = buffers.dv_accum_rows
    done_families = set()
    sentence_stream = runtime_state["sentence_stream"]
    if not sentence_stream.is_empty:
        sentence_q_idx = runtime_state.get("sentence_stream_indices_long")
        if sentence_q_idx is None:
            sentence_q_idx = sentence_stream.query_indices.long()
        sentence_k_idx = runtime_state.get("sentence_stream_key_indices_long")
        if sentence_k_idx is None:
            sentence_k_idx = sentence_stream.key_indices.long()
        sentence_row_idx = runtime_state.get("sentence_stream_row_indices_long")
        if sentence_row_idx is None:
            sentence_row_idx = sentence_stream.row_indices.long()
        q_stream = buffers.q_stream
        k_stream = buffers.k_stream
        v_stream = buffers.v_stream
        _run_hsa_pack_rows_kernel(q_flat, sentence_q_idx, q_stream)
        _run_hsa_pack_rows_kernel(k_flat, sentence_k_idx, k_stream)
        _run_hsa_pack_rows_kernel(v_flat, sentence_k_idx, v_stream)
        sentence_lse = sentence_lse_override
        if sentence_lse is None or sentence_lse.numel() == 0:
            _, sentence_lse = flash_attn_fwd(
                q_stream,
                k_stream,
                v_stream,
                cu_seqlens_q=sentence_stream.cu_seqlens_q,
                cu_seqlens_k=sentence_stream.cu_seqlens_k,
                max_seqlen_q=sentence_stream.max_seqlen_q,
                max_seqlen_k=sentence_stream.max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                return_lse=True,
            )
        if sentence_row_idx.numel() > 0:
            out_stream = buffers.out_stream
            dout_stream = buffers.dout_stream
            _run_hsa_sentence_pack_outputs_kernel(
                out_flat,
                dout_flat,
                total_lse_flat,
                sentence_lse,
                sentence_stream.row_indices,
                out_stream,
                dout_stream,
            )
            dq, dk, dv = flash_attn_bwd(
                q_stream,
                k_stream,
                v_stream,
                out_stream,
                dout_stream,
                sentence_lse,
                softmax_scale=softmax_scale,
                causal=True,
                cu_seqlens_q=sentence_stream.cu_seqlens_q,
                cu_seqlens_k=sentence_stream.cu_seqlens_k,
                max_seqlen_q=sentence_stream.max_seqlen_q,
                max_seqlen_k=sentence_stream.max_seqlen_k,
                deterministic=deterministic,
                force_2cta_sm100_varlen=force_2cta_sm100_varlen,
            )
            _run_hsa_sentence_scatter_rows_kernel(dq, sentence_q_idx, dq_accum_rows)
            _run_hsa_sentence_scatter_rows_kernel(dk, sentence_k_idx, dk_accum_rows)
            _run_hsa_sentence_scatter_rows_kernel(dv, sentence_k_idx, dv_accum_rows)
            done_families.add("sentence_full")
            done_families.add("sentence_tail")
    return (dq_accum_rows, dk_accum_rows, dv_accum_rows), frozenset(done_families)


def _run_hsa_bwd_sentence_families_direct_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
    prepared: Optional[HSABwdPreparedTensors] = None,
    runtime_state=None,
    sentence_lse_override: Optional[torch.Tensor] = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], frozenset[str]]:
    return _run_hsa_bwd_sentence_rows_core(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        softmax_scale,
        deterministic,
        prepared=prepared,
        runtime_state=runtime_state,
        sentence_lse_override=sentence_lse_override,
        force_2cta_sm100_varlen=_use_hsa_sentence_varlen_2cta(),
    )


def _run_hsa_bwd_sentence_full_direct_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_accums, _ = _run_hsa_bwd_sentence_families_direct_rows(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        softmax_scale,
        deterministic,
    )
    return row_accums


def _run_hsa_bwd_sentence_families_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
    sentence_lse_override: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    (dq_accum_rows, dk_accum_rows, dv_accum_rows), _ = _run_hsa_bwd_sentence_families_direct_rows(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        softmax_scale,
        deterministic,
        sentence_lse_override=sentence_lse_override,
    )
    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)


def _run_hsa_bwd_sentence_full_direct(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule,
    monolithic_schedule,
    softmax_scale: float,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq_accum_rows, dk_accum_rows, dv_accum_rows = _run_hsa_bwd_sentence_full_direct_rows(
        q,
        k,
        v,
        out,
        dout,
        lse,
        schedule,
        monolithic_schedule,
        softmax_scale,
        deterministic,
    )
    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_accum_rows, dk_accum_rows, dv_accum_rows)


def _run_hsa_sentence_full_fa4_fastpath(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
    )


def _run_hsa_sentence_full_mma_batches(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
    )


def _run_hsa_sentence_tail_mma_batches(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
    )


def _run_hsa_anchor_full_mma_batches(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
    )


def _run_hsa_anchor_tail_mma_batches(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    batch,
    softmax_scale: float,
    deterministic: bool,
    dq_accum_rows: torch.Tensor,
    dk_accum_rows: torch.Tensor,
    dv_accum_rows: torch.Tensor,
) -> bool:
    return _run_hsa_descriptor_mma_batches(
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        dout_flat,
        lse_flat,
        batch,
        softmax_scale,
        deterministic,
        dq_accum_rows,
        dk_accum_rows,
        dv_accum_rows,
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

    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_acc, dk_acc, dv_acc)


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

    return _cast_hsa_row_accums_to_outputs(q, k, v, dq_acc, dk_acc, dv_acc)


def run_hsa_bwd_sm100_blocksparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    sentence_lse: Optional[torch.Tensor],
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
        sentence_lse,
        schedule,
        softmax_scale,
        deterministic,
        keep_ids,
        hash_ids,
    )
