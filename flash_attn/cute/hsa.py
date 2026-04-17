import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import torch


def _lazy_cute_imports():
    import cutlass
    import cutlass.cute as cute

    from flash_attn.cute import utils
    from flash_attn.cute.block_sparsity import fast_sampling
    from flash_attn.cute.interface import (
        _flash_attn_bwd,
        _flash_attn_fwd,
        flash_attn_func,
    )

    return cutlass, cute, utils, fast_sampling, flash_attn_func, _flash_attn_fwd, _flash_attn_bwd


def _move_nested_tensors(value, *, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_nested_tensors(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_nested_tensors(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device=device) for item in value)
    return value


@dataclass
class HSAStreamPack:
    """Dense FA4 substream metadata for one exact HSA component."""

    query_indices: torch.Tensor
    key_indices: torch.Tensor
    row_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int

    def to(self, device: torch.device | str):
        return HSAStreamPack(
            query_indices=self.query_indices.to(device=device),
            key_indices=self.key_indices.to(device=device),
            row_indices=self.row_indices.to(device=device),
            cu_seqlens_q=self.cu_seqlens_q.to(device=device),
            cu_seqlens_k=self.cu_seqlens_k.to(device=device),
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
        )

    @property
    def is_empty(self) -> bool:
        return self.row_indices.numel() == 0


_DESC_SENTENCE = 0
_DESC_SECTION = 1
_DESC_DOCUMENT = 2

_HSA_FWD_TILE_NONE = 0
_HSA_FWD_TILE_AFFINE_PREFIX = 1
_HSA_FWD_TILE_ROW_PREFIX = 2
_HSA_FWD_TILE_BITMAP = 3


@dataclass
class HSABlockDescriptors:
    """Exact block-grouped descriptors for future monolithic HSA kernels."""

    block_size: int
    blocks_per_batch: int
    block_row_ptr: torch.Tensor
    kind: torch.Tensor
    segment_id: torch.Tensor
    row_start: torch.Tensor
    row_end: torch.Tensor
    offset_start: torch.Tensor
    offset_end: torch.Tensor
    row_mask: torch.Tensor

    def to(self, device: torch.device | str):
        return HSABlockDescriptors(
            block_size=self.block_size,
            blocks_per_batch=self.blocks_per_batch,
            block_row_ptr=self.block_row_ptr.to(device=device),
            kind=self.kind.to(device=device),
            segment_id=self.segment_id.to(device=device),
            row_start=self.row_start.to(device=device),
            row_end=self.row_end.to(device=device),
            offset_start=self.offset_start.to(device=device),
            offset_end=self.offset_end.to(device=device),
            row_mask=self.row_mask.to(device=device),
        )

    @property
    def num_blocks(self) -> int:
        return self.block_row_ptr.shape[0] - 1


@dataclass
class HSABlockSparseTensors:
    """Torch-native block sparsity metadata used by the HSA fast path."""

    mask_block_cnt: torch.Tensor
    mask_block_idx: torch.Tensor
    full_block_cnt: Optional[torch.Tensor]
    full_block_idx: Optional[torch.Tensor]
    block_size: tuple[int, int]

    def to(self, device: torch.device | str):
        return HSABlockSparseTensors(
            mask_block_cnt=self.mask_block_cnt.to(device=device),
            mask_block_idx=self.mask_block_idx.to(device=device),
            full_block_cnt=None if self.full_block_cnt is None else self.full_block_cnt.to(device=device),
            full_block_idx=None if self.full_block_idx is None else self.full_block_idx.to(device=device),
            block_size=self.block_size,
        )


@dataclass
class HSAForwardTileMasks:
    """Packed exact metadata for HSA forward partial tiles."""

    block_id_table: torch.Tensor
    tile_kind: torch.Tensor
    affine_base: torch.Tensor
    row_prefix_row_ptr: torch.Tensor
    row_prefix_len: torch.Tensor
    bitmap_word_row_ptr: torch.Tensor
    bitmap_words: torch.Tensor
    q_block_size: int
    k_block_size: int
    words_per_row: int

    def to(self, device: torch.device | str):
        return HSAForwardTileMasks(
            block_id_table=self.block_id_table.to(device=device),
            tile_kind=self.tile_kind.to(device=device),
            affine_base=self.affine_base.to(device=device),
            row_prefix_row_ptr=self.row_prefix_row_ptr.to(device=device),
            row_prefix_len=self.row_prefix_len.to(device=device),
            bitmap_word_row_ptr=self.bitmap_word_row_ptr.to(device=device),
            bitmap_words=self.bitmap_words.to(device=device),
            q_block_size=self.q_block_size,
            k_block_size=self.k_block_size,
            words_per_row=self.words_per_row,
        )


@dataclass
class HSASyntheticGridMetadata:
    """Logical occupancy/compaction metadata layered over physical sparse tiles."""

    logical_block_q: int
    logical_block_k: int
    physical_block_q: int
    physical_block_k: int
    tile_batch_idx: torch.Tensor
    tile_q_block_idx: torch.Tensor
    tile_k_block_idx: torch.Tensor
    tile_q_subgroup_idx: torch.Tensor
    tile_q_row_ptr: torch.Tensor
    tile_q_rows: torch.Tensor
    tile_k_row_ptr: torch.Tensor
    tile_k_rows: torch.Tensor
    tile_logical_pair_row_ptr: torch.Tensor
    tile_logical_pairs: torch.Tensor
    compact_mask_row_ptr: torch.Tensor
    compact_mask_col_idx: torch.Tensor
    tile_allowed_pairs: torch.Tensor
    tile_packed_q: torch.Tensor
    tile_packed_k: torch.Tensor
    tile_dense: torch.Tensor
    bucket_row_ptr: torch.Tensor
    bucket_tile_idx: torch.Tensor
    bucket_packed_q: torch.Tensor
    bucket_packed_k: torch.Tensor
    bucket_dense: torch.Tensor
    bucket_allowed_pairs: Optional[torch.Tensor] = None
    bucket_fill: Optional[torch.Tensor] = None
    max_packed_k: Optional[int] = None
    max_direct_segments: Optional[int] = None
    sparse_parse_fwd: Optional[bool] = None
    tile_fill: Optional[torch.Tensor] = None
    tile_q_length: Optional[torch.Tensor] = None
    tile_k_length: Optional[torch.Tensor] = None
    bucket_q_row_idx_row_ptr: Optional[torch.Tensor] = None
    bucket_q_row_idx: Optional[torch.Tensor] = None
    bucket_q_src_row_idx: Optional[torch.Tensor] = None
    bucket_k_row_idx_row_ptr: Optional[torch.Tensor] = None
    bucket_k_row_idx: Optional[torch.Tensor] = None
    bucket_q_length: Optional[torch.Tensor] = None
    bucket_k_length: Optional[torch.Tensor] = None
    bucket_split_slot: Optional[torch.Tensor] = None
    bucket_qgroup_bucket_idx: Optional[torch.Tensor] = None
    bucket_mask_word_row_ptr: Optional[torch.Tensor] = None
    bucket_mask_words: Optional[torch.Tensor] = None
    bucket_words_per_row: Optional[torch.Tensor] = None
    qgroup_row_ptr: Optional[torch.Tensor] = None
    qgroup_rows: Optional[torch.Tensor] = None
    qgroup_length: Optional[torch.Tensor] = None
    qgroup_packed_q: Optional[torch.Tensor] = None
    qgroup_num_splits: Optional[torch.Tensor] = None
    qgroup_bucket_row_ptr: Optional[torch.Tensor] = None
    qgroup_bucket_idx: Optional[torch.Tensor] = None
    qgroup_bucket_packed_q: Optional[torch.Tensor] = None
    qgroup_bucket_q_row_idx_row_ptr: Optional[torch.Tensor] = None
    qgroup_bucket_q_row_idx: Optional[torch.Tensor] = None
    qgroup_bucket_split_bucket_row_ptr: Optional[torch.Tensor] = None
    qgroup_bucket_split_bucket_idx: Optional[torch.Tensor] = None
    forward_execution_plan: Optional[dict] = None
    host_index_view: Optional[dict] = None

    def to(self, device: torch.device | str):
        return HSASyntheticGridMetadata(
            logical_block_q=self.logical_block_q,
            logical_block_k=self.logical_block_k,
            physical_block_q=self.physical_block_q,
            physical_block_k=self.physical_block_k,
            tile_batch_idx=self.tile_batch_idx.to(device=device),
            tile_q_block_idx=self.tile_q_block_idx.to(device=device),
            tile_k_block_idx=self.tile_k_block_idx.to(device=device),
            tile_q_subgroup_idx=self.tile_q_subgroup_idx.to(device=device),
            tile_q_row_ptr=self.tile_q_row_ptr.to(device=device),
            tile_q_rows=self.tile_q_rows.to(device=device),
            tile_k_row_ptr=self.tile_k_row_ptr.to(device=device),
            tile_k_rows=self.tile_k_rows.to(device=device),
            tile_logical_pair_row_ptr=self.tile_logical_pair_row_ptr.to(device=device),
            tile_logical_pairs=self.tile_logical_pairs.to(device=device),
            compact_mask_row_ptr=self.compact_mask_row_ptr.to(device=device),
            compact_mask_col_idx=self.compact_mask_col_idx.to(device=device),
            tile_allowed_pairs=self.tile_allowed_pairs.to(device=device),
            tile_packed_q=self.tile_packed_q.to(device=device),
            tile_packed_k=self.tile_packed_k.to(device=device),
            tile_dense=self.tile_dense.to(device=device),
            tile_fill=None if self.tile_fill is None else self.tile_fill.to(device=device),
            bucket_row_ptr=self.bucket_row_ptr.to(device=device),
            bucket_tile_idx=self.bucket_tile_idx.to(device=device),
            bucket_packed_q=self.bucket_packed_q.to(device=device),
            bucket_packed_k=self.bucket_packed_k.to(device=device),
            bucket_dense=self.bucket_dense.to(device=device),
            bucket_allowed_pairs=(
                None if self.bucket_allowed_pairs is None else self.bucket_allowed_pairs.to(device=device)
            ),
            bucket_fill=None if self.bucket_fill is None else self.bucket_fill.to(device=device),
            max_packed_k=self.max_packed_k,
            max_direct_segments=self.max_direct_segments,
            sparse_parse_fwd=self.sparse_parse_fwd,
            tile_q_length=None if self.tile_q_length is None else self.tile_q_length.to(device=device),
            tile_k_length=None if self.tile_k_length is None else self.tile_k_length.to(device=device),
            bucket_q_row_idx_row_ptr=(
                None if self.bucket_q_row_idx_row_ptr is None else self.bucket_q_row_idx_row_ptr.to(device=device)
            ),
            bucket_q_row_idx=None if self.bucket_q_row_idx is None else self.bucket_q_row_idx.to(device=device),
            bucket_q_src_row_idx=(
                None if self.bucket_q_src_row_idx is None else self.bucket_q_src_row_idx.to(device=device)
            ),
            bucket_k_row_idx_row_ptr=(
                None if self.bucket_k_row_idx_row_ptr is None else self.bucket_k_row_idx_row_ptr.to(device=device)
            ),
            bucket_k_row_idx=None if self.bucket_k_row_idx is None else self.bucket_k_row_idx.to(device=device),
            bucket_q_length=None if self.bucket_q_length is None else self.bucket_q_length.to(device=device),
            bucket_k_length=None if self.bucket_k_length is None else self.bucket_k_length.to(device=device),
            bucket_split_slot=None if self.bucket_split_slot is None else self.bucket_split_slot.to(device=device),
            bucket_qgroup_bucket_idx=(
                None if self.bucket_qgroup_bucket_idx is None else self.bucket_qgroup_bucket_idx.to(device=device)
            ),
            bucket_mask_word_row_ptr=(
                None if self.bucket_mask_word_row_ptr is None else self.bucket_mask_word_row_ptr.to(device=device)
            ),
            bucket_mask_words=None if self.bucket_mask_words is None else self.bucket_mask_words.to(device=device),
            bucket_words_per_row=(
                None if self.bucket_words_per_row is None else self.bucket_words_per_row.to(device=device)
            ),
            qgroup_row_ptr=None if self.qgroup_row_ptr is None else self.qgroup_row_ptr.to(device=device),
            qgroup_rows=None if self.qgroup_rows is None else self.qgroup_rows.to(device=device),
            qgroup_length=None if self.qgroup_length is None else self.qgroup_length.to(device=device),
            qgroup_packed_q=None if self.qgroup_packed_q is None else self.qgroup_packed_q.to(device=device),
            qgroup_num_splits=None if self.qgroup_num_splits is None else self.qgroup_num_splits.to(device=device),
            qgroup_bucket_row_ptr=(
                None if self.qgroup_bucket_row_ptr is None else self.qgroup_bucket_row_ptr.to(device=device)
            ),
            qgroup_bucket_idx=None if self.qgroup_bucket_idx is None else self.qgroup_bucket_idx.to(device=device),
            qgroup_bucket_packed_q=(
                None if self.qgroup_bucket_packed_q is None else self.qgroup_bucket_packed_q.to(device=device)
            ),
            qgroup_bucket_q_row_idx_row_ptr=(
                None
                if self.qgroup_bucket_q_row_idx_row_ptr is None
                else self.qgroup_bucket_q_row_idx_row_ptr.to(device=device)
            ),
            qgroup_bucket_q_row_idx=(
                None if self.qgroup_bucket_q_row_idx is None else self.qgroup_bucket_q_row_idx.to(device=device)
            ),
            qgroup_bucket_split_bucket_row_ptr=(
                None
                if self.qgroup_bucket_split_bucket_row_ptr is None
                else self.qgroup_bucket_split_bucket_row_ptr.to(device=device)
            ),
            qgroup_bucket_split_bucket_idx=(
                None
                if self.qgroup_bucket_split_bucket_idx is None
                else self.qgroup_bucket_split_bucket_idx.to(device=device)
            ),
            forward_execution_plan=self.forward_execution_plan,
            host_index_view=self.host_index_view,
        )

    @property
    def num_tiles(self) -> int:
        return int(self.tile_batch_idx.numel())


@dataclass
class HSABlockSparseRuntime:
    """Cached runtime metadata for the single-call HSA block-sparse path."""

    forward_sparse: HSABlockSparseTensors
    forward_tile_masks: HSAForwardTileMasks
    backward_sparse: Optional[HSABlockSparseTensors]
    backward_packed_masks: Optional["HSABwdPackedMasks"]
    forward_aux_tensors: list[torch.Tensor]
    backward_aux_tensors: list[torch.Tensor]
    forward_block_q: int
    forward_block_k: int
    backward_block_q: int
    backward_block_k: int
    backward_subtile_factor: int
    forward_sparse_torch: Optional[Any] = None
    backward_sparse_torch: Optional[Any] = None
    forward_synthetic_grid: Optional[HSASyntheticGridMetadata] = None
    backward_synthetic_grid: Optional[HSASyntheticGridMetadata] = None
    synthetic_grid: Optional[HSASyntheticGridMetadata] = None
    synthetic_forward_workspace: Optional[dict] = None
    synthetic_backward_workspace: Optional[dict] = None
    synthetic_forward_prob_token: int = 0

    def to(self, device: torch.device | str):
        def _move_aux(tensor: torch.Tensor) -> torch.Tensor:
            moved = tensor.to(device=device)
            if hasattr(tensor, "__assumed_align__"):
                setattr(moved, "__assumed_align__", getattr(tensor, "__assumed_align__"))
            if hasattr(tensor, "__leading_dim__"):
                setattr(moved, "__leading_dim__", getattr(tensor, "__leading_dim__"))
            return moved

        forward_sparse = self.forward_sparse.to(device=device)
        backward_sparse = None if self.backward_sparse is None else self.backward_sparse.to(device=device)
        backward_packed_masks = (
            None if self.backward_packed_masks is None else self.backward_packed_masks.to(device=device)
        )
        return HSABlockSparseRuntime(
            forward_sparse=forward_sparse,
            forward_tile_masks=self.forward_tile_masks.to(device=device),
            backward_sparse=backward_sparse,
            backward_packed_masks=backward_packed_masks,
            forward_aux_tensors=[_move_aux(tensor) for tensor in self.forward_aux_tensors],
            backward_aux_tensors=[_move_aux(tensor) for tensor in self.backward_aux_tensors],
            forward_sparse_torch=_to_block_sparse_tensors_torch(forward_sparse),
            backward_sparse_torch=None if backward_sparse is None else _to_block_sparse_tensors_torch(backward_sparse),
            forward_block_q=self.forward_block_q,
            forward_block_k=self.forward_block_k,
            backward_block_q=self.backward_block_q,
            backward_block_k=self.backward_block_k,
            backward_subtile_factor=self.backward_subtile_factor,
            forward_synthetic_grid=(
                None if self.forward_synthetic_grid is None else self.forward_synthetic_grid.to(device=device)
            ),
            backward_synthetic_grid=(
                None if self.backward_synthetic_grid is None else self.backward_synthetic_grid.to(device=device)
            ),
            synthetic_grid=None if self.synthetic_grid is None else self.synthetic_grid.to(device=device),
            synthetic_forward_workspace=None,
            synthetic_backward_workspace=None,
            synthetic_forward_prob_token=0,
        )


@dataclass
class HSABwdPackedMasks:
    """Packed exact masks for HSA backward partial blocks."""

    block_id_table: torch.Tensor
    mask_words: torch.Tensor
    row_group_nonempty: torch.Tensor
    q_block_size: int
    k_block_size: int
    words_per_row: int

    def to(self, device: torch.device | str):
        return HSABwdPackedMasks(
            block_id_table=self.block_id_table.to(device=device),
            mask_words=self.mask_words.to(device=device),
            row_group_nonempty=self.row_group_nonempty.to(device=device),
            q_block_size=self.q_block_size,
            k_block_size=self.k_block_size,
            words_per_row=self.words_per_row,
        )


@dataclass
class HSAHybridBackwardSchedule:
    """Packed KV-local anchor-panel metadata for the HSA backward path."""

    k_block_size: int
    anchor_row_panel_size: int
    blocks_per_batch: int
    sentence_kblock_row_ptr: torch.Tensor
    sentence_segment_id: torch.Tensor
    sentence_q_start: torch.Tensor
    sentence_q_len: torch.Tensor
    sentence_k_start: torch.Tensor
    sentence_k_len: torch.Tensor
    anchor_kblock_row_ptr: torch.Tensor
    anchor_kind: torch.Tensor
    anchor_segment_id: torch.Tensor
    anchor_q_row_ptr: torch.Tensor
    anchor_q_indices: torch.Tensor
    anchor_k_row_ptr: torch.Tensor
    anchor_k_indices: torch.Tensor
    anchor_prefix_row_ptr: torch.Tensor
    anchor_prefix_len: torch.Tensor

    def to(self, device: torch.device | str):
        return HSAHybridBackwardSchedule(
            k_block_size=self.k_block_size,
            anchor_row_panel_size=self.anchor_row_panel_size,
            blocks_per_batch=self.blocks_per_batch,
            sentence_kblock_row_ptr=self.sentence_kblock_row_ptr.to(device=device),
            sentence_segment_id=self.sentence_segment_id.to(device=device),
            sentence_q_start=self.sentence_q_start.to(device=device),
            sentence_q_len=self.sentence_q_len.to(device=device),
            sentence_k_start=self.sentence_k_start.to(device=device),
            sentence_k_len=self.sentence_k_len.to(device=device),
            anchor_kblock_row_ptr=self.anchor_kblock_row_ptr.to(device=device),
            anchor_kind=self.anchor_kind.to(device=device),
            anchor_segment_id=self.anchor_segment_id.to(device=device),
            anchor_q_row_ptr=self.anchor_q_row_ptr.to(device=device),
            anchor_q_indices=self.anchor_q_indices.to(device=device),
            anchor_k_row_ptr=self.anchor_k_row_ptr.to(device=device),
            anchor_k_indices=self.anchor_k_indices.to(device=device),
            anchor_prefix_row_ptr=self.anchor_prefix_row_ptr.to(device=device),
            anchor_prefix_len=self.anchor_prefix_len.to(device=device),
        )

    @property
    def num_k_blocks(self) -> int:
        return self.sentence_kblock_row_ptr.shape[0] - 1


@dataclass
class HSAHybridBackwardBatch:
    """Stacked gather/scatter batch for one exact HSA backward panel shape."""

    q_indices: torch.Tensor
    k_indices: torch.Tensor
    q_length: torch.Tensor
    k_length: torch.Tensor
    prefix_len: Optional[torch.Tensor]

    def to(self, device: torch.device | str):
        return HSAHybridBackwardBatch(
            q_indices=self.q_indices.to(device=device),
            k_indices=self.k_indices.to(device=device),
            q_length=self.q_length.to(device=device),
            k_length=self.k_length.to(device=device),
            prefix_len=None if self.prefix_len is None else self.prefix_len.to(device=device),
        )


@dataclass
class HSAMonolithicBackwardSchedule:
    """Kernel-facing packed backward descriptors grouped by key block."""

    k_block_size: int
    anchor_row_panel_size: int
    blocks_per_batch: int
    sentence_full_kblock_row_ptr: torch.Tensor
    sentence_full_q_start: torch.Tensor
    sentence_full_q_len: torch.Tensor
    sentence_full_q_group_mask: torch.Tensor
    sentence_full_k_local_start: torch.Tensor
    sentence_full_k_len: torch.Tensor
    sentence_tail_kblock_row_ptr: torch.Tensor
    sentence_tail_q_start: torch.Tensor
    sentence_tail_q_len: torch.Tensor
    sentence_tail_q_group_mask: torch.Tensor
    sentence_tail_k_local_start: torch.Tensor
    sentence_tail_k_len: torch.Tensor
    sentence_tail_row0_prefix_len: torch.Tensor
    anchor_full_kblock_row_ptr: torch.Tensor
    anchor_full_q_row_start: torch.Tensor
    anchor_full_q_row_count: torch.Tensor
    anchor_full_q_group_mask: torch.Tensor
    anchor_full_k_local_start: torch.Tensor
    anchor_full_k_len: torch.Tensor
    anchor_tail_kblock_row_ptr: torch.Tensor
    anchor_tail_q_row_start: torch.Tensor
    anchor_tail_q_row_count: torch.Tensor
    anchor_tail_q_group_mask: torch.Tensor
    anchor_tail_k_local_start: torch.Tensor
    anchor_tail_k_len: torch.Tensor
    anchor_tail_prefix_row_start: torch.Tensor
    anchor_q_indices: torch.Tensor
    anchor_prefix_len: torch.Tensor

    def to(self, device: torch.device | str):
        return HSAMonolithicBackwardSchedule(
            k_block_size=self.k_block_size,
            anchor_row_panel_size=self.anchor_row_panel_size,
            blocks_per_batch=self.blocks_per_batch,
            sentence_full_kblock_row_ptr=self.sentence_full_kblock_row_ptr.to(device=device),
            sentence_full_q_start=self.sentence_full_q_start.to(device=device),
            sentence_full_q_len=self.sentence_full_q_len.to(device=device),
            sentence_full_q_group_mask=self.sentence_full_q_group_mask.to(device=device),
            sentence_full_k_local_start=self.sentence_full_k_local_start.to(device=device),
            sentence_full_k_len=self.sentence_full_k_len.to(device=device),
            sentence_tail_kblock_row_ptr=self.sentence_tail_kblock_row_ptr.to(device=device),
            sentence_tail_q_start=self.sentence_tail_q_start.to(device=device),
            sentence_tail_q_len=self.sentence_tail_q_len.to(device=device),
            sentence_tail_q_group_mask=self.sentence_tail_q_group_mask.to(device=device),
            sentence_tail_k_local_start=self.sentence_tail_k_local_start.to(device=device),
            sentence_tail_k_len=self.sentence_tail_k_len.to(device=device),
            sentence_tail_row0_prefix_len=self.sentence_tail_row0_prefix_len.to(device=device),
            anchor_full_kblock_row_ptr=self.anchor_full_kblock_row_ptr.to(device=device),
            anchor_full_q_row_start=self.anchor_full_q_row_start.to(device=device),
            anchor_full_q_row_count=self.anchor_full_q_row_count.to(device=device),
            anchor_full_q_group_mask=self.anchor_full_q_group_mask.to(device=device),
            anchor_full_k_local_start=self.anchor_full_k_local_start.to(device=device),
            anchor_full_k_len=self.anchor_full_k_len.to(device=device),
            anchor_tail_kblock_row_ptr=self.anchor_tail_kblock_row_ptr.to(device=device),
            anchor_tail_q_row_start=self.anchor_tail_q_row_start.to(device=device),
            anchor_tail_q_row_count=self.anchor_tail_q_row_count.to(device=device),
            anchor_tail_q_group_mask=self.anchor_tail_q_group_mask.to(device=device),
            anchor_tail_k_local_start=self.anchor_tail_k_local_start.to(device=device),
            anchor_tail_k_len=self.anchor_tail_k_len.to(device=device),
            anchor_tail_prefix_row_start=self.anchor_tail_prefix_row_start.to(device=device),
            anchor_q_indices=self.anchor_q_indices.to(device=device),
            anchor_prefix_len=self.anchor_prefix_len.to(device=device),
        )

    @property
    def num_k_blocks(self) -> int:
        return self.sentence_full_kblock_row_ptr.shape[0] - 1


@dataclass
class HSAFusedForwardSchedule:
    """Packed query-block descriptors for the internal HSA fused-forward path."""

    q_block_size: int
    k_block_size: int
    blocks_per_batch: int
    sentence_qblock_row_ptr: torch.Tensor
    sentence_segment_id: torch.Tensor
    sentence_q_offset_start: torch.Tensor
    sentence_q_offset_end: torch.Tensor
    sentence_k_offset_start: torch.Tensor
    sentence_k_offset_end: torch.Tensor
    anchor_qblock_row_ptr: torch.Tensor
    anchor_kind: torch.Tensor
    anchor_segment_id: torch.Tensor
    anchor_q_offset_start: torch.Tensor
    anchor_q_offset_end: torch.Tensor
    anchor_k_offset_start: torch.Tensor
    anchor_k_offset_end: torch.Tensor
    anchor_prefix_row_ptr: torch.Tensor
    anchor_prefix_len: torch.Tensor
    sentence_full_qblock_row_ptr: torch.Tensor
    sentence_full_segment_id: torch.Tensor
    sentence_full_q_offset_start: torch.Tensor
    sentence_full_q_offset_end: torch.Tensor
    sentence_full_k_offset_start: torch.Tensor
    sentence_full_k_offset_end: torch.Tensor
    sentence_tail_qblock_row_ptr: torch.Tensor
    sentence_tail_segment_id: torch.Tensor
    sentence_tail_q_offset_start: torch.Tensor
    sentence_tail_q_offset_end: torch.Tensor
    sentence_tail_k_offset_start: torch.Tensor
    sentence_tail_k_offset_end: torch.Tensor
    sentence_tail_prefix_row_ptr: torch.Tensor
    sentence_tail_prefix_len: torch.Tensor
    anchor_full_qblock_row_ptr: torch.Tensor
    anchor_full_kind: torch.Tensor
    anchor_full_segment_id: torch.Tensor
    anchor_full_q_offset_start: torch.Tensor
    anchor_full_q_offset_end: torch.Tensor
    anchor_full_k_row_ptr: torch.Tensor
    anchor_full_k_indices: torch.Tensor
    anchor_tail_qblock_row_ptr: torch.Tensor
    anchor_tail_kind: torch.Tensor
    anchor_tail_segment_id: torch.Tensor
    anchor_tail_q_offset_start: torch.Tensor
    anchor_tail_q_offset_end: torch.Tensor
    anchor_tail_k_row_ptr: torch.Tensor
    anchor_tail_k_indices: torch.Tensor
    anchor_tail_prefix_row_ptr: torch.Tensor
    anchor_tail_prefix_len: torch.Tensor

    def to(self, device: torch.device | str):
        return HSAFusedForwardSchedule(
            q_block_size=self.q_block_size,
            k_block_size=self.k_block_size,
            blocks_per_batch=self.blocks_per_batch,
            sentence_qblock_row_ptr=self.sentence_qblock_row_ptr.to(device=device),
            sentence_segment_id=self.sentence_segment_id.to(device=device),
            sentence_q_offset_start=self.sentence_q_offset_start.to(device=device),
            sentence_q_offset_end=self.sentence_q_offset_end.to(device=device),
            sentence_k_offset_start=self.sentence_k_offset_start.to(device=device),
            sentence_k_offset_end=self.sentence_k_offset_end.to(device=device),
            anchor_qblock_row_ptr=self.anchor_qblock_row_ptr.to(device=device),
            anchor_kind=self.anchor_kind.to(device=device),
            anchor_segment_id=self.anchor_segment_id.to(device=device),
            anchor_q_offset_start=self.anchor_q_offset_start.to(device=device),
            anchor_q_offset_end=self.anchor_q_offset_end.to(device=device),
            anchor_k_offset_start=self.anchor_k_offset_start.to(device=device),
            anchor_k_offset_end=self.anchor_k_offset_end.to(device=device),
            anchor_prefix_row_ptr=self.anchor_prefix_row_ptr.to(device=device),
            anchor_prefix_len=self.anchor_prefix_len.to(device=device),
            sentence_full_qblock_row_ptr=self.sentence_full_qblock_row_ptr.to(device=device),
            sentence_full_segment_id=self.sentence_full_segment_id.to(device=device),
            sentence_full_q_offset_start=self.sentence_full_q_offset_start.to(device=device),
            sentence_full_q_offset_end=self.sentence_full_q_offset_end.to(device=device),
            sentence_full_k_offset_start=self.sentence_full_k_offset_start.to(device=device),
            sentence_full_k_offset_end=self.sentence_full_k_offset_end.to(device=device),
            sentence_tail_qblock_row_ptr=self.sentence_tail_qblock_row_ptr.to(device=device),
            sentence_tail_segment_id=self.sentence_tail_segment_id.to(device=device),
            sentence_tail_q_offset_start=self.sentence_tail_q_offset_start.to(device=device),
            sentence_tail_q_offset_end=self.sentence_tail_q_offset_end.to(device=device),
            sentence_tail_k_offset_start=self.sentence_tail_k_offset_start.to(device=device),
            sentence_tail_k_offset_end=self.sentence_tail_k_offset_end.to(device=device),
            sentence_tail_prefix_row_ptr=self.sentence_tail_prefix_row_ptr.to(device=device),
            sentence_tail_prefix_len=self.sentence_tail_prefix_len.to(device=device),
            anchor_full_qblock_row_ptr=self.anchor_full_qblock_row_ptr.to(device=device),
            anchor_full_kind=self.anchor_full_kind.to(device=device),
            anchor_full_segment_id=self.anchor_full_segment_id.to(device=device),
            anchor_full_q_offset_start=self.anchor_full_q_offset_start.to(device=device),
            anchor_full_q_offset_end=self.anchor_full_q_offset_end.to(device=device),
            anchor_full_k_row_ptr=self.anchor_full_k_row_ptr.to(device=device),
            anchor_full_k_indices=self.anchor_full_k_indices.to(device=device),
            anchor_tail_qblock_row_ptr=self.anchor_tail_qblock_row_ptr.to(device=device),
            anchor_tail_kind=self.anchor_tail_kind.to(device=device),
            anchor_tail_segment_id=self.anchor_tail_segment_id.to(device=device),
            anchor_tail_q_offset_start=self.anchor_tail_q_offset_start.to(device=device),
            anchor_tail_q_offset_end=self.anchor_tail_q_offset_end.to(device=device),
            anchor_tail_k_row_ptr=self.anchor_tail_k_row_ptr.to(device=device),
            anchor_tail_k_indices=self.anchor_tail_k_indices.to(device=device),
            anchor_tail_prefix_row_ptr=self.anchor_tail_prefix_row_ptr.to(device=device),
            anchor_tail_prefix_len=self.anchor_tail_prefix_len.to(device=device),
        )

    @property
    def num_q_blocks(self) -> int:
        return self.sentence_qblock_row_ptr.shape[0] - 1


@dataclass
class HSAFusedForwardBatch:
    """Stacked gather-only forward batch built from fused forward descriptors."""

    q_indices: torch.Tensor
    k_indices: torch.Tensor
    q_length: torch.Tensor
    k_length: torch.Tensor
    q_offset_start: torch.Tensor
    prefix_len: Optional[torch.Tensor]

    def to(self, device: torch.device | str):
        return HSAFusedForwardBatch(
            q_indices=self.q_indices.to(device=device),
            k_indices=self.k_indices.to(device=device),
            q_length=self.q_length.to(device=device),
            k_length=self.k_length.to(device=device),
            q_offset_start=self.q_offset_start.to(device=device),
            prefix_len=None if self.prefix_len is None else self.prefix_len.to(device=device),
        )


@dataclass
class HSASchedule:
    """Exact HSA schedule with canonical CSR/segment views plus derived block descriptors."""

    batch_size_value: int
    seqlen_value: int
    block_size_value: int
    sentence_start: torch.Tensor
    sentence_len: torch.Tensor
    section_row_ptr: torch.Tensor
    section_col_idx: torch.Tensor
    document_row_ptr: torch.Tensor
    document_col_idx: torch.Tensor
    sentence_q_start: torch.Tensor
    sentence_q_len: torch.Tensor
    section_t_row_ptr: torch.Tensor
    section_t_col_idx: torch.Tensor
    document_t_row_ptr: torch.Tensor
    document_t_col_idx: torch.Tensor
    sentence_segment_ptr: torch.Tensor
    sentence_segment_pos: torch.Tensor
    sentence_segment_id: torch.Tensor
    sentence_segment_offset: torch.Tensor
    section_segment_ptr: torch.Tensor
    section_segment_pos: torch.Tensor
    section_segment_id: torch.Tensor
    section_segment_offset: torch.Tensor
    section_self_allowed: torch.Tensor
    document_segment_ptr: torch.Tensor
    document_segment_pos: torch.Tensor
    document_segment_id: torch.Tensor
    document_segment_offset: torch.Tensor
    document_self_allowed: torch.Tensor
    forward_descriptors: HSABlockDescriptors
    backward_descriptors: HSABlockDescriptors
    sentence_stream: HSAStreamPack
    section_prefix_stream: HSAStreamPack
    document_prefix_stream: HSAStreamPack
    section_self_indices: torch.Tensor
    document_self_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.batch_size_value

    @property
    def seqlen(self) -> int:
        return self.seqlen_value

    @property
    def block_size(self) -> int:
        return self.block_size_value

    @property
    def num_rows(self) -> int:
        return self.batch_size_value * self.seqlen_value

    def to(self, device: torch.device | str):
        return HSASchedule(
            batch_size_value=self.batch_size_value,
            seqlen_value=self.seqlen_value,
            block_size_value=self.block_size_value,
            sentence_start=self.sentence_start.to(device=device),
            sentence_len=self.sentence_len.to(device=device),
            section_row_ptr=self.section_row_ptr.to(device=device),
            section_col_idx=self.section_col_idx.to(device=device),
            document_row_ptr=self.document_row_ptr.to(device=device),
            document_col_idx=self.document_col_idx.to(device=device),
            sentence_q_start=self.sentence_q_start.to(device=device),
            sentence_q_len=self.sentence_q_len.to(device=device),
            section_t_row_ptr=self.section_t_row_ptr.to(device=device),
            section_t_col_idx=self.section_t_col_idx.to(device=device),
            document_t_row_ptr=self.document_t_row_ptr.to(device=device),
            document_t_col_idx=self.document_t_col_idx.to(device=device),
            sentence_segment_ptr=self.sentence_segment_ptr.to(device=device),
            sentence_segment_pos=self.sentence_segment_pos.to(device=device),
            sentence_segment_id=self.sentence_segment_id.to(device=device),
            sentence_segment_offset=self.sentence_segment_offset.to(device=device),
            section_segment_ptr=self.section_segment_ptr.to(device=device),
            section_segment_pos=self.section_segment_pos.to(device=device),
            section_segment_id=self.section_segment_id.to(device=device),
            section_segment_offset=self.section_segment_offset.to(device=device),
            section_self_allowed=self.section_self_allowed.to(device=device),
            document_segment_ptr=self.document_segment_ptr.to(device=device),
            document_segment_pos=self.document_segment_pos.to(device=device),
            document_segment_id=self.document_segment_id.to(device=device),
            document_segment_offset=self.document_segment_offset.to(device=device),
            document_self_allowed=self.document_self_allowed.to(device=device),
            forward_descriptors=self.forward_descriptors.to(device=device),
            backward_descriptors=self.backward_descriptors.to(device=device),
            sentence_stream=self.sentence_stream.to(device=device),
            section_prefix_stream=self.section_prefix_stream.to(device=device),
            document_prefix_stream=self.document_prefix_stream.to(device=device),
            section_self_indices=self.section_self_indices.to(device=device),
            document_self_indices=self.document_self_indices.to(device=device),
        )


def _ensure_int32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.int32 else x.to(dtype=torch.int32)


def _empty_int32(device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int32, device=device)


def _tag_aux_tensor(
    t: torch.Tensor,
    *,
    assumed_align: Optional[int] = 4,
    leading_dim: int = -1,
) -> torch.Tensor:
    if assumed_align is not None:
        setattr(t, "__assumed_align__", assumed_align)
    setattr(t, "__leading_dim__", t.ndim - 1 if leading_dim == -1 else leading_dim)
    return t


def _make_stream_pack(
    query_indices: list[int],
    key_indices: list[int],
    row_indices: list[int],
    cu_seqlens_q: list[int],
    cu_seqlens_k: list[int],
    max_seqlen_q: int,
    max_seqlen_k: int,
    device,
) -> HSAStreamPack:
    return HSAStreamPack(
        query_indices=torch.tensor(query_indices, dtype=torch.int32, device=device)
        if query_indices
        else _empty_int32(device),
        key_indices=torch.tensor(key_indices, dtype=torch.int32, device=device)
        if key_indices
        else _empty_int32(device),
        row_indices=torch.tensor(row_indices, dtype=torch.int32, device=device)
        if row_indices
        else _empty_int32(device),
        cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, device=device),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def _rows_to_csr(rows: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    row_ptr = [0]
    col_idx: list[int] = []
    for row in rows:
        col_idx.extend(row)
        row_ptr.append(len(col_idx))
    row_ptr_tensor = torch.tensor(row_ptr, dtype=torch.int32, device=device)
    col_idx_tensor = (
        torch.tensor(col_idx, dtype=torch.int32, device=device) if col_idx else _empty_int32(device)
    )
    return row_ptr_tensor, col_idx_tensor


def _build_segment_metadata(
    segments_flat: list[list[int]],
    total_rows: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    segment_ptr = [0]
    segment_pos: list[int] = []
    row_segment_id = [-1] * total_rows
    row_segment_offset = [-1] * total_rows

    for segment_id, segment in enumerate(segments_flat):
        segment_pos.extend(segment)
        segment_ptr.append(len(segment_pos))
        for offset, flat_row in enumerate(segment):
            row_segment_id[flat_row] = segment_id
            row_segment_offset[flat_row] = offset

    return (
        torch.tensor(segment_ptr, dtype=torch.int32, device=device),
        torch.tensor(segment_pos, dtype=torch.int32, device=device)
        if segment_pos
        else _empty_int32(device),
        torch.tensor(row_segment_id, dtype=torch.int32, device=device),
        torch.tensor(row_segment_offset, dtype=torch.int32, device=device),
    )


def _row_to_block_id(flat_row: int, seqlen: int, block_size: int, blocks_per_batch: int) -> int:
    batch_idx, token_idx = divmod(flat_row, seqlen)
    return batch_idx * blocks_per_batch + (token_idx // block_size)


def _append_block_descriptors(
    block_descriptors: list[list[tuple[int, int, int, int, int, int]]],
    *,
    kind: int,
    segments_flat: list[list[int]],
    seqlen: int,
    block_size: int,
    blocks_per_batch: int,
):
    for segment_id, segment in enumerate(segments_flat):
        offset_start = 0
        while offset_start < len(segment):
            first_flat_row = segment[offset_start]
            block_id = _row_to_block_id(first_flat_row, seqlen, block_size, blocks_per_batch)
            block_start = (first_flat_row % seqlen // block_size) * block_size
            row_start = (first_flat_row % seqlen) - block_start

            offset_end = offset_start + 1
            while offset_end < len(segment):
                next_flat_row = segment[offset_end]
                if _row_to_block_id(next_flat_row, seqlen, block_size, blocks_per_batch) != block_id:
                    break
                offset_end += 1

            last_flat_row = segment[offset_end - 1]
            row_end = (last_flat_row % seqlen) - block_start + 1
            block_descriptors[block_id].append(
                (kind, segment_id, row_start, row_end, offset_start, offset_end)
            )
            offset_start = offset_end


def _build_block_descriptors(
    *,
    batch_size: int,
    seqlen: int,
    block_size: int,
    sentence_segments_flat: list[list[int]],
    section_segments_flat: list[list[int]],
    document_segments_flat: list[list[int]],
    device,
) -> HSABlockDescriptors:
    blocks_per_batch = (seqlen + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_batch
    block_descriptors: list[list[tuple[int, int, int, int, int, int]]] = [[] for _ in range(num_blocks)]

    _append_block_descriptors(
        block_descriptors,
        kind=_DESC_SENTENCE,
        segments_flat=sentence_segments_flat,
        seqlen=seqlen,
        block_size=block_size,
        blocks_per_batch=blocks_per_batch,
    )
    _append_block_descriptors(
        block_descriptors,
        kind=_DESC_SECTION,
        segments_flat=section_segments_flat,
        seqlen=seqlen,
        block_size=block_size,
        blocks_per_batch=blocks_per_batch,
    )
    _append_block_descriptors(
        block_descriptors,
        kind=_DESC_DOCUMENT,
        segments_flat=document_segments_flat,
        seqlen=seqlen,
        block_size=block_size,
        blocks_per_batch=blocks_per_batch,
    )

    block_row_ptr = [0]
    kind: list[int] = []
    segment_id: list[int] = []
    row_start: list[int] = []
    row_end: list[int] = []
    offset_start: list[int] = []
    offset_end: list[int] = []
    row_mask: list[torch.Tensor] = []

    for descriptors in block_descriptors:
        descriptors.sort(key=lambda desc: (desc[0], desc[2], desc[4], desc[1]))
        for desc_kind, desc_segment_id, desc_row_start, desc_row_end, desc_offset_start, desc_offset_end in descriptors:
            kind.append(desc_kind)
            segment_id.append(desc_segment_id)
            row_start.append(desc_row_start)
            row_end.append(desc_row_end)
            offset_start.append(desc_offset_start)
            offset_end.append(desc_offset_end)

            mask = torch.zeros(block_size, dtype=torch.bool, device=device)
            mask[desc_row_start:desc_row_end] = True
            row_mask.append(mask)
        block_row_ptr.append(len(kind))

    row_mask_tensor = (
        torch.stack(row_mask, dim=0)
        if row_mask
        else torch.empty((0, block_size), dtype=torch.bool, device=device)
    )
    return HSABlockDescriptors(
        block_size=block_size,
        blocks_per_batch=blocks_per_batch,
        block_row_ptr=torch.tensor(block_row_ptr, dtype=torch.int32, device=device),
        kind=torch.tensor(kind, dtype=torch.int32, device=device) if kind else _empty_int32(device),
        segment_id=torch.tensor(segment_id, dtype=torch.int32, device=device)
        if segment_id
        else _empty_int32(device),
        row_start=torch.tensor(row_start, dtype=torch.int32, device=device) if row_start else _empty_int32(device),
        row_end=torch.tensor(row_end, dtype=torch.int32, device=device) if row_end else _empty_int32(device),
        offset_start=torch.tensor(offset_start, dtype=torch.int32, device=device)
        if offset_start
        else _empty_int32(device),
        offset_end=torch.tensor(offset_end, dtype=torch.int32, device=device)
        if offset_end
        else _empty_int32(device),
        row_mask=row_mask_tensor,
    )


def _build_segments(valid: list[bool], ids: list[int]) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    current_id: Optional[int] = None
    for pos, is_valid in enumerate(valid):
        if not is_valid:
            continue
        seg_id = ids[pos]
        if current and seg_id != current_id:
            segments.append(current)
            current = []
        current.append(pos)
        current_id = seg_id
    if current:
        segments.append(current)
    return segments


def _segment_fields_for_kind(
    schedule: HSASchedule,
    kind: int,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if kind == _DESC_SENTENCE:
        return schedule.sentence_segment_ptr, schedule.sentence_segment_pos, None
    if kind == _DESC_SECTION:
        return schedule.section_segment_ptr, schedule.section_segment_pos, schedule.section_self_allowed
    if kind == _DESC_DOCUMENT:
        return schedule.document_segment_ptr, schedule.document_segment_pos, schedule.document_self_allowed
    raise ValueError(f"Unsupported descriptor kind: {kind}")


def build_hsa_schedule(keep_ids: torch.Tensor, hash_ids: torch.Tensor) -> HSASchedule:
    """
    Build an exact HSA schedule with:
      - query-major forward CSR,
      - key-major transposed CSR,
      - dense FA4 stream packs for the sentence / section-prefix / document-prefix paths.
    """
    keep_ids = _ensure_int32(keep_ids)
    hash_ids = _ensure_int32(hash_ids)
    bsz, _, seqlen = keep_ids.shape
    device = keep_ids.device
    total_rows = bsz * seqlen

    sentence_start = [0] * total_rows
    sentence_len = [0] * total_rows
    sentence_q_start = [0] * total_rows
    sentence_q_len = [0] * total_rows
    section_rows: list[list[int]] = [[] for _ in range(total_rows)]
    section_t_rows: list[list[int]] = [[] for _ in range(total_rows)]
    document_rows: list[list[int]] = [[] for _ in range(total_rows)]
    document_t_rows: list[list[int]] = [[] for _ in range(total_rows)]

    sentence_segments_flat: list[list[int]] = []
    section_segments_flat: list[list[int]] = []
    document_segments_flat: list[list[int]] = []
    section_self_indices: list[int] = []
    document_self_indices: list[int] = []

    keep_cpu = keep_ids.detach().cpu()
    hash_cpu = hash_ids.detach().cpu()

    for batch_idx in range(bsz):
        ki0 = [bool(x) for x in keep_cpu[batch_idx, 0].tolist()]
        ki1 = [bool(x) for x in keep_cpu[batch_idx, 1].tolist()]
        ki2 = [bool(x) for x in keep_cpu[batch_idx, 2].tolist()]
        h0 = [int(x) for x in hash_cpu[batch_idx, 0].tolist()]
        h1 = [int(x) for x in hash_cpu[batch_idx, 1].tolist()]
        h2 = [int(x) for x in hash_cpu[batch_idx, 2].tolist()]
        row_base = batch_idx * seqlen
        sentence_anchor_by_section_sent = {
            (h1[pos], h0[pos]): pos
            for pos in range(seqlen)
            if ki0[pos] and ki1[pos] and not ki2[pos]
        }
        section_anchor_by_doc_sec = {
            (h2[pos], h1[pos]): pos
            for pos in range(seqlen)
            if (not ki0[pos]) and ki1[pos] and ki2[pos]
        }

        sentence_segments = _build_segments(ki0, h0)
        for seg in sentence_segments:
            flat_seg = [row_base + pos for pos in seg]
            sentence_segments_flat.append(flat_seg)
            seg_start = seg[0]
            seg_len = len(seg)
            for idx, pos in enumerate(seg):
                flat_row = row_base + pos
                sentence_start[flat_row] = seg_start
                sentence_len[flat_row] = idx + 1
                sentence_q_start[flat_row] = pos
                sentence_q_len[flat_row] = seg_len - idx

        section_segments = _build_segments(ki1, h1)
        for seg in section_segments:
            flat_seg = [row_base + pos for pos in seg]
            section_segments_flat.append(flat_seg)
            seg_len = len(seg)
            for idx, pos in enumerate(seg):
                flat_row = row_base + pos
                prefix_end = idx if ki0[pos] else idx + 1
                if prefix_end > 0:
                    section_rows[flat_row].extend(seg[:prefix_end])
                if ki1[pos] and not ki0[pos]:
                    section_self_indices.append(flat_row)
            for idx, pos in enumerate(seg):
                flat_row = row_base + pos
                query_start = idx + 1 if ki0[pos] else idx
                if query_start < seg_len:
                    section_t_rows[flat_row].extend(seg[query_start:])

        document_segments = _build_segments(ki2, h2)
        for seg in document_segments:
            flat_seg = [row_base + pos for pos in seg]
            document_segments_flat.append(flat_seg)
            seg_len = len(seg)
            for idx, pos in enumerate(seg):
                flat_row = row_base + pos
                exclude_self = ki0[pos] or ki1[pos]
                prefix_end = idx if exclude_self else idx + 1
                if prefix_end > 0:
                    document_rows[flat_row].extend(seg[:prefix_end])
                if ki2[pos] and not ki1[pos] and not ki0[pos]:
                    document_self_indices.append(flat_row)
            for idx, pos in enumerate(seg):
                flat_row = row_base + pos
                query_start = idx + 1 if (ki0[pos] or ki1[pos]) else idx
                if query_start < seg_len:
                    document_t_rows[flat_row].extend(seg[query_start:])

        for pos in range(seqlen):
            flat_row = row_base + pos
            is_body = ki0[pos] and not ki1[pos]
            is_sentence_anchor = ki0[pos] and ki1[pos] and not ki2[pos]

            if is_body:
                prev_sentence_anchor = sentence_anchor_by_section_sent.get((h1[pos], h0[pos] - 1))
                if prev_sentence_anchor is not None:
                    section_rows[flat_row].append(prev_sentence_anchor)
                    section_t_rows[row_base + prev_sentence_anchor].append(pos)

            if is_body or is_sentence_anchor:
                prev_section_anchor = section_anchor_by_doc_sec.get((h2[pos], h1[pos] - 1))
                if prev_section_anchor is not None:
                    document_rows[flat_row].append(prev_section_anchor)
                    document_t_rows[row_base + prev_section_anchor].append(pos)

        for flat_row in range(row_base, row_base + seqlen):
            section_rows[flat_row] = sorted(set(section_rows[flat_row]))
            section_t_rows[flat_row] = sorted(set(section_t_rows[flat_row]))
            document_rows[flat_row] = sorted(set(document_rows[flat_row]))
            document_t_rows[flat_row] = sorted(set(document_t_rows[flat_row]))

    section_self_indices_tensor = (
        torch.tensor(section_self_indices, dtype=torch.int32, device=device) if section_self_indices else _empty_int32(device)
    )
    document_self_indices_tensor = (
        torch.tensor(document_self_indices, dtype=torch.int32, device=device)
        if document_self_indices
        else _empty_int32(device)
    )
    section_self_allowed_tensor = torch.zeros(total_rows, dtype=torch.bool, device=device)
    if section_self_indices:
        section_self_allowed_tensor[section_self_indices_tensor.to(dtype=torch.long)] = True
    document_self_allowed_tensor = torch.zeros(total_rows, dtype=torch.bool, device=device)
    if document_self_indices:
        document_self_allowed_tensor[document_self_indices_tensor.to(dtype=torch.long)] = True

    sentence_query_indices: list[int] = []
    sentence_cu = [0]
    max_sentence = 0
    for seg in sentence_segments_flat:
        sentence_query_indices.extend(seg)
        sentence_cu.append(sentence_cu[-1] + len(seg))
        max_sentence = max(max_sentence, len(seg))
    sentence_stream = _make_stream_pack(
        query_indices=sentence_query_indices,
        key_indices=sentence_query_indices,
        row_indices=sentence_query_indices,
        cu_seqlens_q=sentence_cu,
        cu_seqlens_k=sentence_cu,
        max_seqlen_q=max_sentence,
        max_seqlen_k=max_sentence,
        device=device,
    )

    section_prefix_q: list[int] = []
    section_prefix_k: list[int] = []
    section_prefix_cu_q = [0]
    section_prefix_cu_k = [0]
    max_section_prefix = 0
    for seg in section_segments_flat:
        if len(seg) <= 1:
            continue
        q_chunk = seg[1:]
        k_chunk = seg[:-1]
        section_prefix_q.extend(q_chunk)
        section_prefix_k.extend(k_chunk)
        section_prefix_cu_q.append(section_prefix_cu_q[-1] + len(q_chunk))
        section_prefix_cu_k.append(section_prefix_cu_k[-1] + len(k_chunk))
        max_section_prefix = max(max_section_prefix, len(q_chunk))
    section_prefix_stream = _make_stream_pack(
        query_indices=section_prefix_q,
        key_indices=section_prefix_k,
        row_indices=section_prefix_q,
        cu_seqlens_q=section_prefix_cu_q,
        cu_seqlens_k=section_prefix_cu_k,
        max_seqlen_q=max_section_prefix,
        max_seqlen_k=max_section_prefix,
        device=device,
    )

    document_prefix_q: list[int] = []
    document_prefix_k: list[int] = []
    document_prefix_cu_q = [0]
    document_prefix_cu_k = [0]
    max_document_prefix = 0
    for seg in document_segments_flat:
        if len(seg) <= 1:
            continue
        q_chunk = seg[1:]
        k_chunk = seg[:-1]
        document_prefix_q.extend(q_chunk)
        document_prefix_k.extend(k_chunk)
        document_prefix_cu_q.append(document_prefix_cu_q[-1] + len(q_chunk))
        document_prefix_cu_k.append(document_prefix_cu_k[-1] + len(k_chunk))
        max_document_prefix = max(max_document_prefix, len(q_chunk))
    document_prefix_stream = _make_stream_pack(
        query_indices=document_prefix_q,
        key_indices=document_prefix_k,
        row_indices=document_prefix_q,
        cu_seqlens_q=document_prefix_cu_q,
        cu_seqlens_k=document_prefix_cu_k,
        max_seqlen_q=max_document_prefix,
        max_seqlen_k=max_document_prefix,
        device=device,
    )

    section_row_ptr, section_col_idx = _rows_to_csr(section_rows, device)
    section_t_row_ptr, section_t_col_idx = _rows_to_csr(section_t_rows, device)
    document_row_ptr, document_col_idx = _rows_to_csr(document_rows, device)
    document_t_row_ptr, document_t_col_idx = _rows_to_csr(document_t_rows, device)
    sentence_segment_ptr, sentence_segment_pos, sentence_segment_id, sentence_segment_offset = _build_segment_metadata(
        sentence_segments_flat, total_rows, device
    )
    section_segment_ptr, section_segment_pos, section_segment_id, section_segment_offset = _build_segment_metadata(
        section_segments_flat, total_rows, device
    )
    document_segment_ptr, document_segment_pos, document_segment_id, document_segment_offset = _build_segment_metadata(
        document_segments_flat, total_rows, device
    )
    block_size = 128
    forward_descriptors = _build_block_descriptors(
        batch_size=bsz,
        seqlen=seqlen,
        block_size=block_size,
        sentence_segments_flat=sentence_segments_flat,
        section_segments_flat=section_segments_flat,
        document_segments_flat=document_segments_flat,
        device=device,
    )
    if _use_hsa_runtime_forward_only():
        backward_descriptors = forward_descriptors
    else:
        backward_descriptors = _build_block_descriptors(
            batch_size=bsz,
            seqlen=seqlen,
            block_size=block_size,
            sentence_segments_flat=sentence_segments_flat,
            section_segments_flat=section_segments_flat,
            document_segments_flat=document_segments_flat,
            device=device,
        )

    return HSASchedule(
        batch_size_value=bsz,
        seqlen_value=seqlen,
        block_size_value=block_size,
        sentence_start=torch.tensor(sentence_start, dtype=torch.int32, device=device),
        sentence_len=torch.tensor(sentence_len, dtype=torch.int32, device=device),
        section_row_ptr=section_row_ptr,
        section_col_idx=section_col_idx,
        document_row_ptr=document_row_ptr,
        document_col_idx=document_col_idx,
        sentence_q_start=torch.tensor(sentence_q_start, dtype=torch.int32, device=device),
        sentence_q_len=torch.tensor(sentence_q_len, dtype=torch.int32, device=device),
        section_t_row_ptr=section_t_row_ptr,
        section_t_col_idx=section_t_col_idx,
        document_t_row_ptr=document_t_row_ptr,
        document_t_col_idx=document_t_col_idx,
        sentence_segment_ptr=sentence_segment_ptr,
        sentence_segment_pos=sentence_segment_pos,
        sentence_segment_id=sentence_segment_id,
        sentence_segment_offset=sentence_segment_offset,
        section_segment_ptr=section_segment_ptr,
        section_segment_pos=section_segment_pos,
        section_segment_id=section_segment_id,
        section_segment_offset=section_segment_offset,
        section_self_allowed=section_self_allowed_tensor,
        document_segment_ptr=document_segment_ptr,
        document_segment_pos=document_segment_pos,
        document_segment_id=document_segment_id,
        document_segment_offset=document_segment_offset,
        document_self_allowed=document_self_allowed_tensor,
        forward_descriptors=forward_descriptors,
        backward_descriptors=backward_descriptors,
        sentence_stream=sentence_stream,
        section_prefix_stream=section_prefix_stream,
        document_prefix_stream=document_prefix_stream,
        section_self_indices=section_self_indices_tensor,
        document_self_indices=document_self_indices_tensor,
    )


def schedule_to_attend_mask(schedule: HSASchedule) -> torch.Tensor:
    """Expand an `HSASchedule` back into the exact dense bool attention mask."""
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    attend = torch.zeros(bsz, seqlen, seqlen, dtype=torch.bool, device=schedule.sentence_start.device)
    for flat_row in range(schedule.num_rows):
        batch_idx, q_pos = divmod(flat_row, seqlen)

        sent_len = int(schedule.sentence_len[flat_row].item())
        if sent_len > 0:
            sent_start = int(schedule.sentence_start[flat_row].item())
            attend[batch_idx, q_pos, sent_start : sent_start + sent_len] = True

        sec_start = int(schedule.section_row_ptr[flat_row].item())
        sec_end = int(schedule.section_row_ptr[flat_row + 1].item())
        if sec_end > sec_start:
            sec_cols = schedule.section_col_idx[sec_start:sec_end].long()
            attend[batch_idx, q_pos, sec_cols] = True

        doc_start = int(schedule.document_row_ptr[flat_row].item())
        doc_end = int(schedule.document_row_ptr[flat_row + 1].item())
        if doc_end > doc_start:
            doc_cols = schedule.document_col_idx[doc_start:doc_end].long()
            attend[batch_idx, q_pos, doc_cols] = True
    return attend


def forward_descriptors_to_attend_mask(schedule: HSASchedule) -> torch.Tensor:
    """Expand forward block descriptors back into the exact dense bool attention mask."""
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    attend = torch.zeros(bsz, seqlen, seqlen, dtype=torch.bool, device=schedule.sentence_start.device)
    descriptors = schedule.forward_descriptors

    for block_id in range(descriptors.num_blocks):
        desc_start = int(descriptors.block_row_ptr[block_id].item())
        desc_end = int(descriptors.block_row_ptr[block_id + 1].item())
        batch_idx = block_id // descriptors.blocks_per_batch

        for desc_idx in range(desc_start, desc_end):
            kind = int(descriptors.kind[desc_idx].item())
            seg_ptr, seg_pos, self_allowed = _segment_fields_for_kind(schedule, kind)
            segment_id = int(descriptors.segment_id[desc_idx].item())
            segment_start = int(seg_ptr[segment_id].item())
            offset_start = int(descriptors.offset_start[desc_idx].item())
            offset_end = int(descriptors.offset_end[desc_idx].item())
            rows = seg_pos[segment_start + offset_start : segment_start + offset_end]
            row_mask = descriptors.row_mask[desc_idx]

            for row_delta, flat_row_tensor in enumerate(rows):
                local_row = int(descriptors.row_start[desc_idx].item()) + row_delta
                if local_row >= descriptors.block_size or not bool(row_mask[local_row].item()):
                    continue
                flat_row = int(flat_row_tensor.item())
                _, q_pos = divmod(flat_row, seqlen)
                row_offset = offset_start + row_delta

                if kind == _DESC_SENTENCE:
                    key_offset_end = row_offset + 1
                else:
                    allow_self = bool(self_allowed[flat_row].item())
                    key_offset_end = row_offset + (1 if allow_self else 0)

                if key_offset_end <= 0:
                    continue

                key_rows = seg_pos[segment_start : segment_start + key_offset_end].long() % seqlen
                attend[batch_idx, q_pos, key_rows] = True

    return attend


def backward_descriptors_to_attend_mask(schedule: HSASchedule) -> torch.Tensor:
    """Expand backward block descriptors back into the transpose of the exact dense bool mask."""
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    attend_t = torch.zeros(bsz, seqlen, seqlen, dtype=torch.bool, device=schedule.sentence_start.device)
    descriptors = schedule.backward_descriptors

    for block_id in range(descriptors.num_blocks):
        desc_start = int(descriptors.block_row_ptr[block_id].item())
        desc_end = int(descriptors.block_row_ptr[block_id + 1].item())
        batch_idx = block_id // descriptors.blocks_per_batch

        for desc_idx in range(desc_start, desc_end):
            kind = int(descriptors.kind[desc_idx].item())
            seg_ptr, seg_pos, self_allowed = _segment_fields_for_kind(schedule, kind)
            segment_id = int(descriptors.segment_id[desc_idx].item())
            segment_start = int(seg_ptr[segment_id].item())
            segment_end = int(seg_ptr[segment_id + 1].item())
            offset_start = int(descriptors.offset_start[desc_idx].item())
            offset_end = int(descriptors.offset_end[desc_idx].item())
            rows = seg_pos[segment_start + offset_start : segment_start + offset_end]
            row_mask = descriptors.row_mask[desc_idx]

            for row_delta, flat_key_tensor in enumerate(rows):
                local_row = int(descriptors.row_start[desc_idx].item()) + row_delta
                if local_row >= descriptors.block_size or not bool(row_mask[local_row].item()):
                    continue
                flat_key = int(flat_key_tensor.item())
                _, key_pos = divmod(flat_key, seqlen)
                key_offset = offset_start + row_delta

                if kind == _DESC_SENTENCE:
                    query_offset_start = key_offset
                else:
                    allow_self = bool(self_allowed[flat_key].item())
                    query_offset_start = key_offset if allow_self else key_offset + 1

                if segment_start + query_offset_start >= segment_end:
                    continue

                query_rows = seg_pos[segment_start + query_offset_start : segment_end].long() % seqlen
                attend_t[batch_idx, query_rows, key_pos] = True

    return attend_t


def _build_hsa_schedule_aux_tensors(schedule: HSASchedule) -> list[torch.Tensor]:
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    device = schedule.sentence_start.device
    return [
        _tag_aux_tensor(schedule.sentence_segment_id.view(bsz, seqlen).contiguous()),
        _tag_aux_tensor(schedule.sentence_segment_offset.view(bsz, seqlen).contiguous()),
        _tag_aux_tensor(schedule.section_segment_id.view(bsz, seqlen).contiguous()),
        _tag_aux_tensor(schedule.section_segment_offset.view(bsz, seqlen).contiguous()),
        _tag_aux_tensor(schedule.section_self_allowed.view(bsz, seqlen).to(dtype=torch.int32, device=device).contiguous()),
        _tag_aux_tensor(schedule.document_segment_id.view(bsz, seqlen).contiguous()),
        _tag_aux_tensor(schedule.document_segment_offset.view(bsz, seqlen).contiguous()),
        _tag_aux_tensor(schedule.document_self_allowed.view(bsz, seqlen).to(dtype=torch.int32, device=device).contiguous()),
    ]


def schedule_aux_to_attend_mask(schedule: HSASchedule) -> torch.Tensor:
    """
    Expand the schedule into the exact dense bool attention mask.

    The compressed aux encoding does not currently represent all cross-level HSA
    edges, so the exact row-CSR schedule is the source of truth here.
    """
    return schedule_to_attend_mask(schedule)


def _build_hsa_forward_tile_mask_aux_tensors(tile_masks: HSAForwardTileMasks) -> list[torch.Tensor]:
    return [
        _tag_aux_tensor(tile_masks.block_id_table),
        _tag_aux_tensor(tile_masks.tile_kind),
        _tag_aux_tensor(tile_masks.affine_base),
        _tag_aux_tensor(tile_masks.row_prefix_row_ptr),
        _tag_aux_tensor(tile_masks.row_prefix_len),
        _tag_aux_tensor(tile_masks.bitmap_word_row_ptr),
        _tag_aux_tensor(tile_masks.bitmap_words),
    ]


def _tile_prefix_len_for_row(
    tile_masks: HSAForwardTileMasks,
    *,
    block_id: int,
    q_local: int,
    k_len: int,
) -> int:
    kind = int(tile_masks.tile_kind[block_id].item())
    if kind == _HSA_FWD_TILE_AFFINE_PREFIX:
        return max(0, min(k_len, int(tile_masks.affine_base[block_id].item()) + q_local))
    if kind == _HSA_FWD_TILE_ROW_PREFIX:
        start = int(tile_masks.row_prefix_row_ptr[block_id].item())
        return max(0, min(k_len, int(tile_masks.row_prefix_len[start + q_local].item())))
    raise ValueError(f"tile kind {kind} does not encode row prefixes")


def forward_tile_masks_to_attend_mask(
    schedule: HSASchedule,
    sparse_tensors: HSABlockSparseTensors,
    tile_masks: HSAForwardTileMasks,
) -> torch.Tensor:
    attend = torch.zeros(
        schedule.batch_size,
        schedule.seqlen,
        schedule.seqlen,
        dtype=torch.bool,
        device=schedule.sentence_start.device,
    )
    q_block_size, k_block_size = sparse_tensors.block_size
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = sparse_tensors.full_block_idx.detach().cpu()
    block_id_table = tile_masks.block_id_table.detach().cpu()
    tile_kind = tile_masks.tile_kind.detach().cpu()
    affine_base = tile_masks.affine_base.detach().cpu()
    row_prefix_row_ptr = tile_masks.row_prefix_row_ptr.detach().cpu()
    row_prefix_len = tile_masks.row_prefix_len.detach().cpu()
    bitmap_word_row_ptr = tile_masks.bitmap_word_row_ptr.detach().cpu()
    bitmap_words = tile_masks.bitmap_words.detach().cpu()

    for batch_idx in range(bsz):
        num_q_blocks = mask_block_cnt.shape[2]
        for q_block in range(num_q_blocks):
            q_start = q_block * q_block_size
            q_end = min(seqlen, q_start + q_block_size)
            q_len = q_end - q_start
            full_count = int(full_block_cnt[batch_idx, 0, q_block].item())
            for idx in range(full_count):
                k_block = int(full_block_idx[batch_idx, 0, q_block, idx].item())
                k_start = k_block * k_block_size
                k_end = min(seqlen, k_start + k_block_size)
                attend[batch_idx, q_start:q_end, k_start:k_end] = True

            mask_count = int(mask_block_cnt[batch_idx, 0, q_block].item())
            for idx in range(mask_count):
                k_block = int(mask_block_idx[batch_idx, 0, q_block, idx].item())
                k_start = k_block * k_block_size
                k_end = min(seqlen, k_start + k_block_size)
                k_len = k_end - k_start
                block_id = int(block_id_table[batch_idx, q_block, k_block].item())
                kind = int(tile_kind[block_id].item())
                if kind in {_HSA_FWD_TILE_AFFINE_PREFIX, _HSA_FWD_TILE_ROW_PREFIX}:
                    for q_local in range(q_len):
                        if kind == _HSA_FWD_TILE_AFFINE_PREFIX:
                            prefix_len = max(0, min(k_len, int(affine_base[block_id].item()) + q_local))
                        else:
                            prefix_start = int(row_prefix_row_ptr[block_id].item())
                            prefix_len = max(
                                0,
                                min(k_len, int(row_prefix_len[prefix_start + q_local].item())),
                            )
                        if prefix_len > 0:
                            attend[batch_idx, q_start + q_local, k_start : k_start + prefix_len] = True
                    continue

                word_start = int(bitmap_word_row_ptr[block_id].item())
                for q_local in range(q_len):
                    for k_local in range(k_len):
                        word_idx = k_local // 32
                        bit_idx = k_local % 32
                        word = int(
                            bitmap_words[word_start + q_local * tile_masks.words_per_row + word_idx].item()
                        ) & 0xFFFFFFFF
                        if (word >> bit_idx) & 1:
                            attend[batch_idx, q_start + q_local, k_start + k_local] = True

    return attend


def _accumulate_interval_counts(
    counts: list[list[list[int]]],
    batch_idx: int,
    outer_block: int,
    start: int,
    end: int,
    inner_block_size: int,
):
    cursor = start
    while cursor < end:
        inner_block = cursor // inner_block_size
        block_end = min(end, (inner_block + 1) * inner_block_size)
        counts[batch_idx][outer_block][inner_block] += block_end - cursor
        cursor = block_end


def _counts_to_block_sparse_tensors(
    counts: list[list[list[int]]],
    valid_counts: list[list[int]],
    *,
    block_size: tuple[int, int],
    device,
) -> HSABlockSparseTensors:
    bsz = len(counts)
    outer_blocks = len(counts[0]) if bsz > 0 else 0
    inner_blocks = len(counts[0][0]) if bsz > 0 and outer_blocks > 0 else 0

    mask_block_cnt = torch.zeros((bsz, 1, outer_blocks), dtype=torch.int32, device=device)
    mask_block_idx = torch.zeros((bsz, 1, outer_blocks, inner_blocks), dtype=torch.int32, device=device)
    full_block_cnt = torch.zeros((bsz, 1, outer_blocks), dtype=torch.int32, device=device)
    full_block_idx = torch.zeros((bsz, 1, outer_blocks, inner_blocks), dtype=torch.int32, device=device)

    for batch_idx in range(bsz):
        for outer_block in range(outer_blocks):
            mask_indices: list[int] = []
            full_indices: list[int] = []
            for inner_block in range(inner_blocks):
                allowed_count = counts[batch_idx][outer_block][inner_block]
                if allowed_count == 0:
                    continue
                if allowed_count == valid_counts[outer_block][inner_block]:
                    full_indices.append(inner_block)
                else:
                    mask_indices.append(inner_block)

            mask_block_cnt[batch_idx, 0, outer_block] = len(mask_indices)
            full_block_cnt[batch_idx, 0, outer_block] = len(full_indices)
            if mask_indices:
                mask_block_idx[batch_idx, 0, outer_block, : len(mask_indices)] = torch.tensor(
                    mask_indices, dtype=torch.int32, device=device
                )
            if full_indices:
                full_block_idx[batch_idx, 0, outer_block, : len(full_indices)] = torch.tensor(
                    full_indices, dtype=torch.int32, device=device
                )

    return HSABlockSparseTensors(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=block_size,
    )


def _build_forward_hsa_block_sparse_tensors(
    schedule: HSASchedule,
    *,
    q_block_size: int,
    k_block_size: int,
) -> HSABlockSparseTensors:
    sparse_tensors, _ = _build_forward_hsa_tile_masks(
        schedule,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )
    return sparse_tensors


def _set_interval_block_mask_bits(
    block_masks: dict[tuple[int, int, int], list[list[int]]],
    *,
    batch_idx: int,
    q_idx: int,
    start: int,
    end: int,
    q_block_size: int,
    k_block_size: int,
    words_per_row: int,
):
    q_block = q_idx // q_block_size
    q_local = q_idx - q_block * q_block_size
    cursor = start
    while cursor < end:
        k_block = cursor // k_block_size
        block_words = block_masks.setdefault(
            (batch_idx, q_block, k_block),
            [[0 for _ in range(words_per_row)] for _ in range(q_block_size)],
        )
        block_end = min(end, (k_block + 1) * k_block_size)
        for k_idx in range(cursor, block_end):
            _set_block_mask_bit(
                block_words,
                q_local=q_local,
                k_local=k_idx - k_block * k_block_size,
            )
        cursor = block_end


def _get_block_mask_bit(
    block_words: list[list[int]],
    *,
    row_idx: int,
    bit_idx: int,
) -> int:
    word_idx = bit_idx // 32
    return (block_words[row_idx][word_idx] >> (bit_idx % 32)) & 1


def _row_prefix_len_if_contiguous(
    block_words: list[list[int]],
    *,
    row_idx: int,
    k_len: int,
) -> Optional[int]:
    prefix_len = 0
    while prefix_len < k_len and _get_block_mask_bit(block_words, row_idx=row_idx, bit_idx=prefix_len):
        prefix_len += 1
    for bit_idx in range(prefix_len, k_len):
        if _get_block_mask_bit(block_words, row_idx=row_idx, bit_idx=bit_idx):
            return None
    return prefix_len


def _find_affine_prefix_base(prefix_lengths: list[int], *, k_len: int) -> Optional[int]:
    lower = -10**9
    upper = 10**9
    for row_idx, prefix_len in enumerate(prefix_lengths):
        if prefix_len <= 0:
            upper = min(upper, -row_idx)
        elif prefix_len >= k_len:
            lower = max(lower, k_len - row_idx)
        else:
            affine_base = prefix_len - row_idx
            lower = max(lower, affine_base)
            upper = min(upper, affine_base)
    return lower if lower <= upper else None


def _classify_forward_partial_block(
    block_words: list[list[int]],
    *,
    q_len: int,
    k_len: int,
    words_per_row: int,
) -> tuple[int, int, list[int], list[int]]:
    prefix_lengths: list[int] = []
    for row_idx in range(q_len):
        prefix_len = _row_prefix_len_if_contiguous(block_words, row_idx=row_idx, k_len=k_len)
        if prefix_len is None:
            bitmap_words = [
                _as_signed_int32(block_words[row_idx][word_idx])
                for row_idx in range(q_len)
                for word_idx in range(words_per_row)
            ]
            return _HSA_FWD_TILE_BITMAP, 0, [], bitmap_words
        prefix_lengths.append(prefix_len)

    affine_base = _find_affine_prefix_base(prefix_lengths, k_len=k_len)
    if affine_base is not None:
        return _HSA_FWD_TILE_AFFINE_PREFIX, affine_base, [], []
    return _HSA_FWD_TILE_ROW_PREFIX, 0, prefix_lengths, []


def _classify_forward_partial_row_masks(
    row_masks: list[int],
    *,
    q_len: int,
    k_len: int,
    words_per_row: int,
) -> tuple[int, int, list[int], list[int]]:
    valid_mask = (1 << k_len) - 1 if k_len > 0 else 0
    prefix_lengths: list[int] = []
    for row_idx in range(q_len):
        row_mask = row_masks[row_idx] & valid_mask
        if row_mask & (row_mask + 1):
            bitmap_words = [
                _as_signed_int32(((row_masks[bitmap_row] & valid_mask) >> (32 * word_idx)) & 0xFFFFFFFF)
                for bitmap_row in range(q_len)
                for word_idx in range(words_per_row)
            ]
            return _HSA_FWD_TILE_BITMAP, 0, [], bitmap_words
        prefix_lengths.append(row_mask.bit_length())

    affine = _find_affine_prefix_base(prefix_lengths, k_len=k_len)
    if affine is not None:
        return _HSA_FWD_TILE_AFFINE_PREFIX, affine, [], []
    return _HSA_FWD_TILE_ROW_PREFIX, 0, prefix_lengths, []


def _build_forward_hsa_tile_masks(
    schedule: HSASchedule,
    *,
    q_block_size: int,
    k_block_size: int,
) -> tuple[HSABlockSparseTensors, HSAForwardTileMasks]:
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    num_q_blocks = (seqlen + q_block_size - 1) // q_block_size
    num_k_blocks = (seqlen + k_block_size - 1) // k_block_size
    words_per_row = (k_block_size + 31) // 32

    sentence_start = schedule.sentence_start.detach().cpu().tolist()
    sentence_len = schedule.sentence_len.detach().cpu().tolist()
    section_row_ptr = schedule.section_row_ptr.detach().cpu().tolist()
    section_col_idx = schedule.section_col_idx.detach().cpu().tolist()
    document_row_ptr = schedule.document_row_ptr.detach().cpu().tolist()
    document_col_idx = schedule.document_col_idx.detach().cpu().tolist()

    mask_block_cnt = torch.zeros((bsz, 1, num_q_blocks), dtype=torch.int32, device=schedule.sentence_start.device)
    mask_block_idx = torch.zeros(
        (bsz, 1, num_q_blocks, num_k_blocks),
        dtype=torch.int32,
        device=schedule.sentence_start.device,
    )
    full_block_cnt = torch.zeros((bsz, 1, num_q_blocks), dtype=torch.int32, device=schedule.sentence_start.device)
    full_block_idx = torch.zeros(
        (bsz, 1, num_q_blocks, num_k_blocks),
        dtype=torch.int32,
        device=schedule.sentence_start.device,
    )

    block_id_table = torch.zeros(
        (bsz, num_q_blocks, num_k_blocks),
        dtype=torch.int32,
        device=schedule.sentence_start.device,
    )
    tile_kind = [_HSA_FWD_TILE_NONE]
    affine_base = [0]
    row_prefix_row_ptr = [0]
    row_prefix_len: list[int] = [0 for _ in range(q_block_size)]
    bitmap_word_row_ptr = [0]
    bitmap_words: list[int] = [0 for _ in range(q_block_size * words_per_row)]

    for batch_idx in range(bsz):
        batch_row_base = batch_idx * seqlen
        for q_block in range(num_q_blocks):
            mask_indices: list[int] = []
            full_indices: list[int] = []
            q_len = min(q_block_size, seqlen - q_block * q_block_size)
            q_row_start = batch_row_base + q_block * q_block_size
            local_block_masks: dict[int, list[int]] = {}

            for q_local in range(q_len):
                flat_row = q_row_start + q_local
                sent_start = sentence_start[flat_row]
                sent_end = sent_start + sentence_len[flat_row]
                cursor = sent_start
                while cursor < sent_end:
                    k_block = cursor // k_block_size
                    row_masks = local_block_masks.get(k_block)
                    if row_masks is None:
                        row_masks = [0] * q_len
                        local_block_masks[k_block] = row_masks
                    block_end = min(sent_end, (k_block + 1) * k_block_size)
                    start_local = cursor - k_block * k_block_size
                    row_masks[q_local] |= ((1 << (block_end - cursor)) - 1) << start_local
                    cursor = block_end

                for offset in range(section_row_ptr[flat_row], section_row_ptr[flat_row + 1]):
                    key_idx = section_col_idx[offset]
                    k_block = key_idx // k_block_size
                    row_masks = local_block_masks.get(k_block)
                    if row_masks is None:
                        row_masks = [0] * q_len
                        local_block_masks[k_block] = row_masks
                    row_masks[q_local] |= 1 << (key_idx - k_block * k_block_size)

                for offset in range(document_row_ptr[flat_row], document_row_ptr[flat_row + 1]):
                    key_idx = document_col_idx[offset]
                    k_block = key_idx // k_block_size
                    row_masks = local_block_masks.get(k_block)
                    if row_masks is None:
                        row_masks = [0] * q_len
                        local_block_masks[k_block] = row_masks
                    row_masks[q_local] |= 1 << (key_idx - k_block * k_block_size)

            for k_block in sorted(local_block_masks):
                row_masks = local_block_masks[k_block]
                k_len = min(k_block_size, seqlen - k_block * k_block_size)
                valid_count = q_len * k_len
                valid_mask = (1 << k_len) - 1 if k_len > 0 else 0
                allowed_count = sum((row_mask & valid_mask).bit_count() for row_mask in row_masks)

                if allowed_count == valid_count:
                    full_indices.append(k_block)
                    continue

                mask_indices.append(k_block)
                kind, affine, prefix_vals, bitmap_vals = _classify_forward_partial_row_masks(
                    row_masks,
                    q_len=q_len,
                    k_len=k_len,
                    words_per_row=words_per_row,
                )
                block_id = len(tile_kind)
                block_id_table[batch_idx, q_block, k_block] = block_id
                tile_kind.append(kind)
                affine_base.append(affine)
                row_prefix_row_ptr.append(len(row_prefix_len))
                bitmap_word_row_ptr.append(len(bitmap_words))
                prefix_slot = [0 for _ in range(q_block_size)]
                bitmap_slot = [0 for _ in range(q_block_size * words_per_row)]
                if kind == _HSA_FWD_TILE_ROW_PREFIX:
                    prefix_slot[: len(prefix_vals)] = prefix_vals
                elif kind == _HSA_FWD_TILE_BITMAP:
                    bitmap_slot[: len(bitmap_vals)] = bitmap_vals
                row_prefix_len.extend(prefix_slot)
                bitmap_words.extend(bitmap_slot)

            mask_block_cnt[batch_idx, 0, q_block] = len(mask_indices)
            full_block_cnt[batch_idx, 0, q_block] = len(full_indices)
            if mask_indices:
                mask_block_idx[batch_idx, 0, q_block, : len(mask_indices)] = torch.tensor(
                    mask_indices,
                    dtype=torch.int32,
                    device=schedule.sentence_start.device,
                )
            if full_indices:
                full_block_idx[batch_idx, 0, q_block, : len(full_indices)] = torch.tensor(
                    full_indices,
                    dtype=torch.int32,
                    device=schedule.sentence_start.device,
                )

    row_prefix_row_ptr.append(len(row_prefix_len))
    bitmap_word_row_ptr.append(len(bitmap_words))

    sparse_tensors = HSABlockSparseTensors(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(q_block_size, k_block_size),
    )
    tile_masks = HSAForwardTileMasks(
        block_id_table=block_id_table,
        tile_kind=torch.tensor(tile_kind, dtype=torch.int32, device=schedule.sentence_start.device),
        affine_base=torch.tensor(affine_base, dtype=torch.int32, device=schedule.sentence_start.device),
        row_prefix_row_ptr=torch.tensor(row_prefix_row_ptr, dtype=torch.int32, device=schedule.sentence_start.device),
        row_prefix_len=torch.tensor(row_prefix_len, dtype=torch.int32, device=schedule.sentence_start.device)
        if row_prefix_len
        else _empty_int32(schedule.sentence_start.device),
        bitmap_word_row_ptr=torch.tensor(
            bitmap_word_row_ptr,
            dtype=torch.int32,
            device=schedule.sentence_start.device,
        ),
        bitmap_words=torch.tensor(bitmap_words, dtype=torch.int32, device=schedule.sentence_start.device)
        if bitmap_words
        else _empty_int32(schedule.sentence_start.device),
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        words_per_row=words_per_row,
    )
    return sparse_tensors, tile_masks


def _set_block_mask_bit(
    block_words: list[list[int]],
    *,
    q_local: int,
    k_local: int,
):
    word_idx = k_local // 32
    bit_idx = k_local % 32
    block_words[q_local][word_idx] |= 1 << bit_idx


def _as_signed_int32(word: int) -> int:
    return word if word < (1 << 31) else word - (1 << 32)


def _build_backward_hsa_packed_masks(
    schedule: HSASchedule,
    *,
    q_block_size: int,
    k_block_size: int,
) -> tuple[HSABlockSparseTensors, HSABwdPackedMasks]:
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    num_q_blocks = (seqlen + q_block_size - 1) // q_block_size
    num_k_blocks = (seqlen + k_block_size - 1) // k_block_size
    words_per_row = (k_block_size + 31) // 32
    block_masks: dict[tuple[int, int, int], list[list[int]]] = {}

    sentence_q_start = schedule.sentence_q_start.detach().cpu().tolist()
    sentence_q_len = schedule.sentence_q_len.detach().cpu().tolist()
    section_t_row_ptr = schedule.section_t_row_ptr.detach().cpu().tolist()
    section_t_col_idx = schedule.section_t_col_idx.detach().cpu().tolist()
    document_t_row_ptr = schedule.document_t_row_ptr.detach().cpu().tolist()
    document_t_col_idx = schedule.document_t_col_idx.detach().cpu().tolist()

    def add_pair(batch_idx: int, key_idx: int, query_idx: int):
        k_block = key_idx // k_block_size
        q_block = query_idx // q_block_size
        q_local = query_idx - q_block * q_block_size
        k_local = key_idx - k_block * k_block_size
        block_words = block_masks.setdefault(
            (batch_idx, k_block, q_block),
            [[0 for _ in range(words_per_row)] for _ in range(q_block_size)],
        )
        _set_block_mask_bit(block_words, q_local=q_local, k_local=k_local)

    for flat_key in range(schedule.num_rows):
        batch_idx, key_idx = divmod(flat_key, seqlen)

        sent_q_len = sentence_q_len[flat_key]
        if sent_q_len > 0:
            sent_q_start = sentence_q_start[flat_key]
            for query_idx in range(sent_q_start, sent_q_start + sent_q_len):
                add_pair(batch_idx, key_idx, query_idx)

        for offset in range(section_t_row_ptr[flat_key], section_t_row_ptr[flat_key + 1]):
            add_pair(batch_idx, key_idx, section_t_col_idx[offset])
        for offset in range(document_t_row_ptr[flat_key], document_t_row_ptr[flat_key + 1]):
            add_pair(batch_idx, key_idx, document_t_col_idx[offset])

    mask_block_cnt = torch.zeros((bsz, 1, num_k_blocks), dtype=torch.int32, device=schedule.sentence_start.device)
    mask_block_idx = torch.zeros(
        (bsz, 1, num_k_blocks, num_q_blocks),
        dtype=torch.int32,
        device=schedule.sentence_start.device,
    )
    full_block_cnt = torch.zeros((bsz, 1, num_k_blocks), dtype=torch.int32, device=schedule.sentence_start.device)
    full_block_idx = torch.zeros(
        (bsz, 1, num_k_blocks, num_q_blocks),
        dtype=torch.int32,
        device=schedule.sentence_start.device,
    )
    block_id_table = torch.zeros(
        (bsz, num_k_blocks, num_q_blocks),
        dtype=torch.int32,
        device=schedule.sentence_start.device,
    )

    partial_mask_words: list[list[list[int]]] = [
        [[0 for _ in range(words_per_row)] for _ in range(q_block_size)]
    ]
    partial_row_group_nonempty: list[int] = [0]

    for batch_idx in range(bsz):
        for k_block in range(num_k_blocks):
            mask_indices: list[int] = []
            full_indices: list[int] = []
            q_blocks = sorted(
                q_block
                for (batch, key_block, q_block) in block_masks.keys()
                if batch == batch_idx and key_block == k_block
            )

            k_len = min(k_block_size, seqlen - k_block * k_block_size)
            tail_mask = (1 << (k_len % 32)) - 1 if k_len % 32 != 0 else None

            for q_block in q_blocks:
                block_words = block_masks[(batch_idx, k_block, q_block)]
                q_len = min(q_block_size, seqlen - q_block * q_block_size)
                valid_count = q_len * k_len
                allowed_count = 0
                for row_idx in range(q_len):
                    for word_idx, word in enumerate(block_words[row_idx]):
                        clamped_word = word
                        if tail_mask is not None and word_idx == words_per_row - 1:
                            clamped_word &= tail_mask
                        allowed_count += int(clamped_word).bit_count()

                if allowed_count == valid_count:
                    full_indices.append(q_block)
                    continue

                mask_indices.append(q_block)
                block_id_table[batch_idx, k_block, q_block] = len(partial_mask_words)
                partial_mask_words.append(block_words)
                row_group_bits = 0
                half_rows = q_block_size // 2
                for q_local in range(min(q_len, half_rows)):
                    if any(block_words[q_local]):
                        row_group_bits |= 1
                        break
                for q_local in range(half_rows, q_len):
                    if any(block_words[q_local]):
                        row_group_bits |= 2
                        break
                partial_row_group_nonempty.append(row_group_bits)

            mask_block_cnt[batch_idx, 0, k_block] = len(mask_indices)
            full_block_cnt[batch_idx, 0, k_block] = len(full_indices)
            if mask_indices:
                mask_block_idx[batch_idx, 0, k_block, : len(mask_indices)] = torch.tensor(
                    mask_indices,
                    dtype=torch.int32,
                    device=schedule.sentence_start.device,
                )
            if full_indices:
                full_block_idx[batch_idx, 0, k_block, : len(full_indices)] = torch.tensor(
                    full_indices,
                    dtype=torch.int32,
                    device=schedule.sentence_start.device,
                )

    sparse_tensors = HSABlockSparseTensors(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(q_block_size, k_block_size),
    )
    packed_masks = HSABwdPackedMasks(
        block_id_table=block_id_table,
        mask_words=torch.tensor(
            [
                [[_as_signed_int32(word) for word in row] for row in block_words]
                for block_words in partial_mask_words
            ],
            dtype=torch.int32,
            device=schedule.sentence_start.device,
        ),
        row_group_nonempty=torch.tensor(
            partial_row_group_nonempty,
            dtype=torch.int32,
            device=schedule.sentence_start.device,
        ),
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        words_per_row=words_per_row,
    )
    return sparse_tensors, packed_masks


def _build_backward_hsa_block_sparse_tensors(
    schedule: HSASchedule,
    *,
    q_block_size: int,
    k_block_size: int,
) -> HSABlockSparseTensors:
    sparse_tensors, _ = _build_backward_hsa_packed_masks(
        schedule,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
    )
    return sparse_tensors


def forward_block_sparse_to_attend_mask(
    schedule: HSASchedule,
    sparse_tensors: HSABlockSparseTensors,
) -> torch.Tensor:
    aux_mask = schedule_aux_to_attend_mask(schedule)
    attend = torch.zeros_like(aux_mask)
    q_block_size, k_block_size = sparse_tensors.block_size
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = None if sparse_tensors.full_block_cnt is None else sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = None if sparse_tensors.full_block_idx is None else sparse_tensors.full_block_idx.detach().cpu()

    for batch_idx in range(bsz):
        num_q_blocks = mask_block_cnt.shape[2]
        for q_block in range(num_q_blocks):
            q_start = q_block * q_block_size
            q_end = min(seqlen, q_start + q_block_size)
            mask_count = int(mask_block_cnt[batch_idx, 0, q_block].item())
            for idx in range(mask_count):
                k_block = int(mask_block_idx[batch_idx, 0, q_block, idx].item())
                k_start = k_block * k_block_size
                k_end = min(seqlen, k_start + k_block_size)
                attend[batch_idx, q_start:q_end, k_start:k_end] = aux_mask[batch_idx, q_start:q_end, k_start:k_end]
            if full_block_cnt is not None and full_block_idx is not None:
                full_count = int(full_block_cnt[batch_idx, 0, q_block].item())
                for idx in range(full_count):
                    k_block = int(full_block_idx[batch_idx, 0, q_block, idx].item())
                    k_start = k_block * k_block_size
                    k_end = min(seqlen, k_start + k_block_size)
                    attend[batch_idx, q_start:q_end, k_start:k_end] = True

    return attend


def backward_block_sparse_to_attend_mask(
    schedule: HSASchedule,
    sparse_tensors: HSABlockSparseTensors,
) -> torch.Tensor:
    aux_mask = schedule_aux_to_attend_mask(schedule)
    attend = torch.zeros_like(aux_mask)
    q_block_size, k_block_size = sparse_tensors.block_size
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = None if sparse_tensors.full_block_cnt is None else sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = None if sparse_tensors.full_block_idx is None else sparse_tensors.full_block_idx.detach().cpu()

    for batch_idx in range(bsz):
        num_k_blocks = mask_block_cnt.shape[2]
        for k_block in range(num_k_blocks):
            k_start = k_block * k_block_size
            k_end = min(seqlen, k_start + k_block_size)
            mask_count = int(mask_block_cnt[batch_idx, 0, k_block].item())
            for idx in range(mask_count):
                q_block = int(mask_block_idx[batch_idx, 0, k_block, idx].item())
                q_start = q_block * q_block_size
                q_end = min(seqlen, q_start + q_block_size)
                attend[batch_idx, q_start:q_end, k_start:k_end] = aux_mask[batch_idx, q_start:q_end, k_start:k_end]
            if full_block_cnt is not None and full_block_idx is not None:
                full_count = int(full_block_cnt[batch_idx, 0, k_block].item())
                for idx in range(full_count):
                    q_block = int(full_block_idx[batch_idx, 0, k_block, idx].item())
                    q_start = q_block * q_block_size
                    q_end = min(seqlen, q_start + q_block_size)
                    attend[batch_idx, q_start:q_end, k_start:k_end] = True

    return attend


def backward_packed_masks_to_attend_mask(
    schedule: HSASchedule,
    sparse_tensors: HSABlockSparseTensors,
    packed_masks: HSABwdPackedMasks,
) -> torch.Tensor:
    attend = torch.zeros(
        schedule.batch_size,
        schedule.seqlen,
        schedule.seqlen,
        dtype=torch.bool,
        device=schedule.sentence_start.device,
    )
    q_block_size, k_block_size = sparse_tensors.block_size
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = sparse_tensors.full_block_idx.detach().cpu()
    block_id_table = packed_masks.block_id_table.detach().cpu()
    mask_words = packed_masks.mask_words.detach().cpu()

    for batch_idx in range(bsz):
        num_k_blocks = mask_block_cnt.shape[2]
        for k_block in range(num_k_blocks):
            k_start = k_block * k_block_size
            k_end = min(seqlen, k_start + k_block_size)
            full_count = int(full_block_cnt[batch_idx, 0, k_block].item())
            for idx in range(full_count):
                q_block = int(full_block_idx[batch_idx, 0, k_block, idx].item())
                q_start = q_block * q_block_size
                q_end = min(seqlen, q_start + q_block_size)
                attend[batch_idx, q_start:q_end, k_start:k_end] = True

            mask_count = int(mask_block_cnt[batch_idx, 0, k_block].item())
            for idx in range(mask_count):
                q_block = int(mask_block_idx[batch_idx, 0, k_block, idx].item())
                q_start = q_block * q_block_size
                q_end = min(seqlen, q_start + q_block_size)
                block_id = int(block_id_table[batch_idx, k_block, q_block].item())
                for q_local in range(q_end - q_start):
                    for k_local in range(k_end - k_start):
                        word_idx = k_local // 32
                        bit_idx = k_local % 32
                        word = int(mask_words[block_id, q_local, word_idx].item()) & 0xFFFFFFFF
                        if (word >> bit_idx) & 1:
                            attend[batch_idx, q_start + q_local, k_start + k_local] = True

    return attend


def _build_hsa_hybrid_backward_schedule(
    schedule: HSASchedule,
    *,
    k_block_size: int = 128,
    anchor_row_panel_size: int = 64,
) -> HSAHybridBackwardSchedule:
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    blocks_per_batch = (seqlen + k_block_size - 1) // k_block_size
    total_k_blocks = bsz * blocks_per_batch
    device = schedule.sentence_start.device

    sentence_segment_ptr = schedule.sentence_segment_ptr.detach().cpu().tolist()
    sentence_segment_pos = schedule.sentence_segment_pos.detach().cpu().tolist()
    section_segment_ptr = schedule.section_segment_ptr.detach().cpu().tolist()
    section_segment_pos = schedule.section_segment_pos.detach().cpu().tolist()
    section_self_allowed = schedule.section_self_allowed.detach().cpu().tolist()
    document_segment_ptr = schedule.document_segment_ptr.detach().cpu().tolist()
    document_segment_pos = schedule.document_segment_pos.detach().cpu().tolist()
    document_self_allowed = schedule.document_self_allowed.detach().cpu().tolist()

    sentence_buckets: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(total_k_blocks)]
    anchor_buckets: list[list[tuple[int, int, list[int], list[int], list[int]]]] = [[] for _ in range(total_k_blocks)]

    def append_sentence_descriptors(segment_ptr: list[int], segment_pos: list[int]):
        num_segments = len(segment_ptr) - 1
        for segment_id in range(num_segments):
            seg_start = segment_ptr[segment_id]
            seg_end = segment_ptr[segment_id + 1]
            if seg_end <= seg_start:
                continue
            segment_rows = segment_pos[seg_start:seg_end]
            seg_len = len(segment_rows)
            offset_start = 0
            while offset_start < seg_len:
                flat_row = segment_rows[offset_start]
                batch_idx, token_idx = divmod(flat_row, seqlen)
                k_block = token_idx // k_block_size
                offset_end = offset_start + 1
                while offset_end < seg_len:
                    next_row = segment_rows[offset_end]
                    next_batch_idx, next_token_idx = divmod(next_row, seqlen)
                    if next_batch_idx != batch_idx or next_token_idx // k_block_size != k_block:
                        break
                    offset_end += 1
                global_k_block = batch_idx * blocks_per_batch + k_block
                sentence_buckets[global_k_block].append(
                    (
                        segment_id,
                        token_idx,
                        seg_len - offset_start,
                        token_idx,
                        offset_end - offset_start,
                    )
                )
                offset_start = offset_end

    def append_anchor_descriptors(
        kind: int,
        segment_ptr: list[int],
        segment_pos: list[int],
        self_allowed: list[bool],
    ):
        num_segments = len(segment_ptr) - 1
        for segment_id in range(num_segments):
            seg_start = segment_ptr[segment_id]
            seg_end = segment_ptr[segment_id + 1]
            segment_rows = segment_pos[seg_start:seg_end]
            seg_len = len(segment_rows)
            if seg_len <= 0:
                continue
            offset_start = 0
            while offset_start < seg_len:
                flat_key = segment_rows[offset_start]
                batch_idx, token_idx = divmod(flat_key, seqlen)
                k_block = token_idx // k_block_size
                offset_end = offset_start + 1
                while offset_end < seg_len:
                    next_key = segment_rows[offset_end]
                    next_batch_idx, next_token_idx = divmod(next_key, seqlen)
                    if next_batch_idx != batch_idx or next_token_idx // k_block_size != k_block:
                        break
                    offset_end += 1

                q_rows: list[int] = []
                prefix_len: list[int] = []
                chunk_len = offset_end - offset_start
                for q_offset in range(seg_len):
                    flat_query = segment_rows[q_offset]
                    allow_self = 1 if self_allowed[flat_query] else 0
                    local_prefix = min(chunk_len, max(0, q_offset + allow_self - offset_start))
                    if local_prefix <= 0:
                        continue
                    q_rows.append(flat_query)
                    prefix_len.append(local_prefix)

                if q_rows:
                    global_k_block = batch_idx * blocks_per_batch + k_block
                    key_rows = segment_rows[offset_start:offset_end]
                    for panel_start in range(0, len(q_rows), anchor_row_panel_size):
                        panel_end = min(len(q_rows), panel_start + anchor_row_panel_size)
                        anchor_buckets[global_k_block].append(
                            (
                                kind,
                                segment_id,
                                q_rows[panel_start:panel_end],
                                key_rows,
                                prefix_len[panel_start:panel_end],
                            )
                        )

                offset_start = offset_end

    append_sentence_descriptors(sentence_segment_ptr, sentence_segment_pos)
    append_anchor_descriptors(_DESC_SECTION, section_segment_ptr, section_segment_pos, section_self_allowed)
    append_anchor_descriptors(_DESC_DOCUMENT, document_segment_ptr, document_segment_pos, document_self_allowed)

    sentence_kblock_row_ptr = [0]
    sentence_segment_id: list[int] = []
    sentence_q_start: list[int] = []
    sentence_q_len: list[int] = []
    sentence_k_start: list[int] = []
    sentence_k_len: list[int] = []
    anchor_kblock_row_ptr = [0]
    anchor_kind: list[int] = []
    anchor_segment_id: list[int] = []
    anchor_q_row_ptr = [0]
    anchor_q_indices: list[int] = []
    anchor_k_row_ptr = [0]
    anchor_k_indices: list[int] = []
    anchor_prefix_row_ptr = [0]
    anchor_prefix_len: list[int] = []

    for k_block_id in range(total_k_blocks):
        sentence_descs = sorted(sentence_buckets[k_block_id], key=lambda desc: (desc[3], desc[1], desc[0]))
        for segment_id, q_start, q_len, k_start, k_len in sentence_descs:
            sentence_segment_id.append(segment_id)
            sentence_q_start.append(q_start)
            sentence_q_len.append(q_len)
            sentence_k_start.append(k_start)
            sentence_k_len.append(k_len)
        sentence_kblock_row_ptr.append(len(sentence_segment_id))

        anchor_descs = sorted(
            anchor_buckets[k_block_id],
            key=lambda desc: (desc[0], desc[3][0] if desc[3] else -1, desc[2][0] if desc[2] else -1, desc[1]),
        )
        for kind, segment_id, q_rows, key_rows, prefix_len in anchor_descs:
            anchor_kind.append(kind)
            anchor_segment_id.append(segment_id)
            anchor_q_indices.extend(q_rows)
            anchor_k_indices.extend(key_rows)
            anchor_prefix_len.extend(prefix_len)
            anchor_q_row_ptr.append(len(anchor_q_indices))
            anchor_k_row_ptr.append(len(anchor_k_indices))
            anchor_prefix_row_ptr.append(len(anchor_prefix_len))
        anchor_kblock_row_ptr.append(len(anchor_kind))

    return HSAHybridBackwardSchedule(
        k_block_size=k_block_size,
        anchor_row_panel_size=anchor_row_panel_size,
        blocks_per_batch=blocks_per_batch,
        sentence_kblock_row_ptr=torch.tensor(sentence_kblock_row_ptr, dtype=torch.int32, device=device),
        sentence_segment_id=torch.tensor(sentence_segment_id, dtype=torch.int32, device=device)
        if sentence_segment_id
        else _empty_int32(device),
        sentence_q_start=torch.tensor(sentence_q_start, dtype=torch.int32, device=device)
        if sentence_q_start
        else _empty_int32(device),
        sentence_q_len=torch.tensor(sentence_q_len, dtype=torch.int32, device=device)
        if sentence_q_len
        else _empty_int32(device),
        sentence_k_start=torch.tensor(sentence_k_start, dtype=torch.int32, device=device)
        if sentence_k_start
        else _empty_int32(device),
        sentence_k_len=torch.tensor(sentence_k_len, dtype=torch.int32, device=device)
        if sentence_k_len
        else _empty_int32(device),
        anchor_kblock_row_ptr=torch.tensor(anchor_kblock_row_ptr, dtype=torch.int32, device=device),
        anchor_kind=torch.tensor(anchor_kind, dtype=torch.int32, device=device) if anchor_kind else _empty_int32(device),
        anchor_segment_id=torch.tensor(anchor_segment_id, dtype=torch.int32, device=device)
        if anchor_segment_id
        else _empty_int32(device),
        anchor_q_row_ptr=torch.tensor(anchor_q_row_ptr, dtype=torch.int32, device=device),
        anchor_q_indices=torch.tensor(anchor_q_indices, dtype=torch.int32, device=device)
        if anchor_q_indices
        else _empty_int32(device),
        anchor_k_row_ptr=torch.tensor(anchor_k_row_ptr, dtype=torch.int32, device=device),
        anchor_k_indices=torch.tensor(anchor_k_indices, dtype=torch.int32, device=device)
        if anchor_k_indices
        else _empty_int32(device),
        anchor_prefix_row_ptr=torch.tensor(anchor_prefix_row_ptr, dtype=torch.int32, device=device),
        anchor_prefix_len=torch.tensor(anchor_prefix_len, dtype=torch.int32, device=device)
        if anchor_prefix_len
        else _empty_int32(device),
    )


def hybrid_backward_to_attend_mask(
    schedule: HSASchedule,
    hybrid_schedule: HSAHybridBackwardSchedule,
) -> torch.Tensor:
    attend = torch.zeros(
        schedule.batch_size,
        schedule.seqlen,
        schedule.seqlen,
        dtype=torch.bool,
        device=schedule.sentence_start.device,
    )
    seqlen = schedule.seqlen

    for global_k_block in range(hybrid_schedule.num_k_blocks):
        batch_idx = global_k_block // hybrid_schedule.blocks_per_batch

        sent_desc_start = int(hybrid_schedule.sentence_kblock_row_ptr[global_k_block].item())
        sent_desc_end = int(hybrid_schedule.sentence_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_desc_start, sent_desc_end):
            q_start = int(hybrid_schedule.sentence_q_start[desc_idx].item())
            q_len = int(hybrid_schedule.sentence_q_len[desc_idx].item())
            k_start = int(hybrid_schedule.sentence_k_start[desc_idx].item())
            k_len = int(hybrid_schedule.sentence_k_len[desc_idx].item())
            for k_offset in range(k_len):
                key_pos = k_start + k_offset
                attend[batch_idx, q_start + k_offset : q_start + q_len, key_pos] = True

        anchor_desc_start = int(hybrid_schedule.anchor_kblock_row_ptr[global_k_block].item())
        anchor_desc_end = int(hybrid_schedule.anchor_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_desc_start, anchor_desc_end):
            q_row_start = int(hybrid_schedule.anchor_q_row_ptr[desc_idx].item())
            q_row_end = int(hybrid_schedule.anchor_q_row_ptr[desc_idx + 1].item())
            query_rows = hybrid_schedule.anchor_q_indices[q_row_start:q_row_end].long() % seqlen
            k_row_start = int(hybrid_schedule.anchor_k_row_ptr[desc_idx].item())
            k_row_end = int(hybrid_schedule.anchor_k_row_ptr[desc_idx + 1].item())
            prefix_row_start = int(hybrid_schedule.anchor_prefix_row_ptr[desc_idx].item())
            prefix_row_end = int(hybrid_schedule.anchor_prefix_row_ptr[desc_idx + 1].item())
            key_rows = hybrid_schedule.anchor_k_indices[k_row_start:k_row_end].long() % seqlen
            prefix_len = hybrid_schedule.anchor_prefix_len[prefix_row_start:prefix_row_end].long()
            for query_pos, prefix in zip(query_rows.tolist(), prefix_len.tolist()):
                if prefix > 0:
                    attend[batch_idx, query_pos, key_rows[:prefix].long()] = True

    return attend


def _build_hsa_monolithic_backward_schedule(
    hybrid_schedule: HSAHybridBackwardSchedule,
    *,
    seqlen: int,
    device: torch.device | str,
) -> HSAMonolithicBackwardSchedule:
    k_block_size = hybrid_schedule.k_block_size
    num_k_blocks = hybrid_schedule.num_k_blocks
    q_desc_rows = hybrid_schedule.anchor_row_panel_size

    def _q_group_mask(row_count: int) -> int:
        mask = 0
        if row_count > 0:
            mask |= 0b01
        if row_count > 32:
            mask |= 0b10
        return mask

    sentence_full_kblock_row_ptr = [0]
    sentence_full_q_start: list[int] = []
    sentence_full_q_len: list[int] = []
    sentence_full_q_group_mask: list[int] = []
    sentence_full_k_local_start: list[int] = []
    sentence_full_k_len: list[int] = []

    sentence_tail_kblock_row_ptr = [0]
    sentence_tail_q_start: list[int] = []
    sentence_tail_q_len: list[int] = []
    sentence_tail_q_group_mask: list[int] = []
    sentence_tail_k_local_start: list[int] = []
    sentence_tail_k_len: list[int] = []
    sentence_tail_row0_prefix_len: list[int] = []

    anchor_full_kblock_row_ptr = [0]
    anchor_full_q_row_start: list[int] = []
    anchor_full_q_row_count: list[int] = []
    anchor_full_q_group_mask: list[int] = []
    anchor_full_k_local_start: list[int] = []
    anchor_full_k_len: list[int] = []

    anchor_tail_kblock_row_ptr = [0]
    anchor_tail_q_row_start: list[int] = []
    anchor_tail_q_row_count: list[int] = []
    anchor_tail_q_group_mask: list[int] = []
    anchor_tail_k_local_start: list[int] = []
    anchor_tail_k_len: list[int] = []
    anchor_tail_prefix_row_start: list[int] = []
    anchor_prefix_len: list[int] = []

    for global_k_block in range(num_k_blocks):
        batch_idx = global_k_block // hybrid_schedule.blocks_per_batch
        k_block = global_k_block % hybrid_schedule.blocks_per_batch
        block_flat_start = batch_idx * seqlen + k_block * k_block_size
        sent_desc_start = int(hybrid_schedule.sentence_kblock_row_ptr[global_k_block].item())
        sent_desc_end = int(hybrid_schedule.sentence_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_desc_start, sent_desc_end):
            q_start = int(hybrid_schedule.sentence_q_start[desc_idx].item())
            q_len = int(hybrid_schedule.sentence_q_len[desc_idx].item())
            k_start = int(hybrid_schedule.sentence_k_start[desc_idx].item())
            k_len = int(hybrid_schedule.sentence_k_len[desc_idx].item())
            if q_len <= 0 or k_len <= 0:
                continue
            k_local_start = k_start % k_block_size
            row0_prefix_len = max(0, min(k_len, q_start - k_start + 1))
            tail_rows = min(q_len, max(0, k_len - row0_prefix_len))
            if tail_rows > 0:
                chunk_start = 0
                while chunk_start < tail_rows:
                    chunk_len = min(q_desc_rows, tail_rows - chunk_start)
                    sentence_tail_q_start.append(q_start + chunk_start)
                    sentence_tail_q_len.append(chunk_len)
                    sentence_tail_q_group_mask.append(_q_group_mask(chunk_len))
                    sentence_tail_k_local_start.append(k_local_start)
                    sentence_tail_k_len.append(k_len)
                    sentence_tail_row0_prefix_len.append(row0_prefix_len + chunk_start)
                    chunk_start += chunk_len
            if q_len > tail_rows:
                full_q_start = q_start + tail_rows
                full_q_len = q_len - tail_rows
                chunk_start = 0
                while chunk_start < full_q_len:
                    chunk_len = min(q_desc_rows, full_q_len - chunk_start)
                    sentence_full_q_start.append(full_q_start + chunk_start)
                    sentence_full_q_len.append(chunk_len)
                    sentence_full_q_group_mask.append(_q_group_mask(chunk_len))
                    sentence_full_k_local_start.append(k_local_start)
                    sentence_full_k_len.append(k_len)
                    chunk_start += chunk_len
        sentence_full_kblock_row_ptr.append(len(sentence_full_q_start))
        sentence_tail_kblock_row_ptr.append(len(sentence_tail_q_start))

        anchor_desc_start = int(hybrid_schedule.anchor_kblock_row_ptr[global_k_block].item())
        anchor_desc_end = int(hybrid_schedule.anchor_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_desc_start, anchor_desc_end):
            q_row_start = int(hybrid_schedule.anchor_q_row_ptr[desc_idx].item())
            q_row_end = int(hybrid_schedule.anchor_q_row_ptr[desc_idx + 1].item())
            q_row_count = q_row_end - q_row_start
            if q_row_count <= 0:
                continue
            k_row_start = int(hybrid_schedule.anchor_k_row_ptr[desc_idx].item())
            k_row_end = int(hybrid_schedule.anchor_k_row_ptr[desc_idx + 1].item())
            key_rows = hybrid_schedule.anchor_k_indices[k_row_start:k_row_end].detach().cpu().tolist()
            if not key_rows:
                continue
            local_key_rows = [flat_row - block_flat_start for flat_row in key_rows]
            if any(local_row < 0 or local_row >= k_block_size for local_row in local_key_rows):
                raise ValueError("Anchor key rows must stay within the owning key block")
            if any(curr <= prev for prev, curr in zip(local_key_rows, local_key_rows[1:])):
                raise ValueError("Anchor key rows must remain strictly increasing within a key block")

            prefix_row_start = int(hybrid_schedule.anchor_prefix_row_ptr[desc_idx].item())
            prefix_row_end = int(hybrid_schedule.anchor_prefix_row_ptr[desc_idx + 1].item())
            prefix_rows = hybrid_schedule.anchor_prefix_len[prefix_row_start:prefix_row_end].detach().cpu().tolist()
            if len(prefix_rows) != q_row_count:
                raise ValueError("Anchor prefix metadata must align with anchor query rows")
            if any(curr < prev for prev, curr in zip(prefix_rows, prefix_rows[1:])):
                raise ValueError("Anchor prefix rows must be nondecreasing within a packed descriptor")

            run_start = 0
            while run_start < len(local_key_rows):
                run_end = run_start + 1
                while run_end < len(local_key_rows) and local_key_rows[run_end] == local_key_rows[run_end - 1] + 1:
                    run_end += 1

                k_local_start = local_key_rows[run_start]
                k_len = run_end - run_start
                local_prefix_rows = [min(max(prefix - run_start, 0), k_len) for prefix in prefix_rows]

                positive_start = 0
                while positive_start < q_row_count and local_prefix_rows[positive_start] == 0:
                    positive_start += 1
                if positive_start < q_row_count:
                    full_start = positive_start
                    while full_start < q_row_count and local_prefix_rows[full_start] < k_len:
                        full_start += 1
                    if any(prefix != k_len for prefix in local_prefix_rows[full_start:]):
                        raise ValueError("Anchor local prefix rows must end in a full-visibility suffix")

                    tail_prefix_rows = local_prefix_rows[positive_start:full_start]
                    if tail_prefix_rows:
                        chunk_start = 0
                        while chunk_start < len(tail_prefix_rows):
                            chunk_rows = tail_prefix_rows[chunk_start : chunk_start + q_desc_rows]
                            anchor_tail_q_row_start.append(q_row_start + positive_start + chunk_start)
                            anchor_tail_q_row_count.append(len(chunk_rows))
                            anchor_tail_q_group_mask.append(_q_group_mask(len(chunk_rows)))
                            anchor_tail_k_local_start.append(k_local_start)
                            anchor_tail_k_len.append(k_len)
                            anchor_tail_prefix_row_start.append(len(anchor_prefix_len))
                            anchor_prefix_len.extend(chunk_rows)
                            chunk_start += len(chunk_rows)

                    full_count = q_row_count - full_start
                    if full_count > 0:
                        chunk_start = 0
                        while chunk_start < full_count:
                            chunk_len = min(q_desc_rows, full_count - chunk_start)
                            anchor_full_q_row_start.append(q_row_start + full_start + chunk_start)
                            anchor_full_q_row_count.append(chunk_len)
                            anchor_full_q_group_mask.append(_q_group_mask(chunk_len))
                            anchor_full_k_local_start.append(k_local_start)
                            anchor_full_k_len.append(k_len)
                            chunk_start += chunk_len

                run_start = run_end
        anchor_full_kblock_row_ptr.append(len(anchor_full_q_row_start))
        anchor_tail_kblock_row_ptr.append(len(anchor_tail_q_row_start))

    def _tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device) if values else _empty_int32(device)

    return HSAMonolithicBackwardSchedule(
        k_block_size=k_block_size,
        anchor_row_panel_size=hybrid_schedule.anchor_row_panel_size,
        blocks_per_batch=hybrid_schedule.blocks_per_batch,
        sentence_full_kblock_row_ptr=torch.tensor(sentence_full_kblock_row_ptr, dtype=torch.int32, device=device),
        sentence_full_q_start=_tensor(sentence_full_q_start),
        sentence_full_q_len=_tensor(sentence_full_q_len),
        sentence_full_q_group_mask=_tensor(sentence_full_q_group_mask),
        sentence_full_k_local_start=_tensor(sentence_full_k_local_start),
        sentence_full_k_len=_tensor(sentence_full_k_len),
        sentence_tail_kblock_row_ptr=torch.tensor(sentence_tail_kblock_row_ptr, dtype=torch.int32, device=device),
        sentence_tail_q_start=_tensor(sentence_tail_q_start),
        sentence_tail_q_len=_tensor(sentence_tail_q_len),
        sentence_tail_q_group_mask=_tensor(sentence_tail_q_group_mask),
        sentence_tail_k_local_start=_tensor(sentence_tail_k_local_start),
        sentence_tail_k_len=_tensor(sentence_tail_k_len),
        sentence_tail_row0_prefix_len=_tensor(sentence_tail_row0_prefix_len),
        anchor_full_kblock_row_ptr=torch.tensor(anchor_full_kblock_row_ptr, dtype=torch.int32, device=device),
        anchor_full_q_row_start=_tensor(anchor_full_q_row_start),
        anchor_full_q_row_count=_tensor(anchor_full_q_row_count),
        anchor_full_q_group_mask=_tensor(anchor_full_q_group_mask),
        anchor_full_k_local_start=_tensor(anchor_full_k_local_start),
        anchor_full_k_len=_tensor(anchor_full_k_len),
        anchor_tail_kblock_row_ptr=torch.tensor(anchor_tail_kblock_row_ptr, dtype=torch.int32, device=device),
        anchor_tail_q_row_start=_tensor(anchor_tail_q_row_start),
        anchor_tail_q_row_count=_tensor(anchor_tail_q_row_count),
        anchor_tail_q_group_mask=_tensor(anchor_tail_q_group_mask),
        anchor_tail_k_local_start=_tensor(anchor_tail_k_local_start),
        anchor_tail_k_len=_tensor(anchor_tail_k_len),
        anchor_tail_prefix_row_start=_tensor(anchor_tail_prefix_row_start),
        anchor_q_indices=hybrid_schedule.anchor_q_indices.clone(),
        anchor_prefix_len=_tensor(anchor_prefix_len),
    )


def monolithic_backward_to_attend_mask(
    schedule: HSASchedule,
    monolithic_schedule: HSAMonolithicBackwardSchedule,
) -> torch.Tensor:
    attend = torch.zeros(
        schedule.batch_size,
        schedule.seqlen,
        schedule.seqlen,
        dtype=torch.bool,
        device=schedule.sentence_start.device,
    )
    seqlen = schedule.seqlen
    k_block_size = monolithic_schedule.k_block_size

    for global_k_block in range(monolithic_schedule.num_k_blocks):
        batch_idx = global_k_block // monolithic_schedule.blocks_per_batch
        block_k_start = (global_k_block % monolithic_schedule.blocks_per_batch) * k_block_size

        sent_full_start = int(monolithic_schedule.sentence_full_kblock_row_ptr[global_k_block].item())
        sent_full_end = int(monolithic_schedule.sentence_full_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_full_start, sent_full_end):
            q_start = int(monolithic_schedule.sentence_full_q_start[desc_idx].item())
            q_len = int(monolithic_schedule.sentence_full_q_len[desc_idx].item())
            k_local_start = int(monolithic_schedule.sentence_full_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.sentence_full_k_len[desc_idx].item())
            key_slice = slice(block_k_start + k_local_start, block_k_start + k_local_start + k_len)
            attend[batch_idx, q_start : q_start + q_len, key_slice] = True

        sent_tail_start = int(monolithic_schedule.sentence_tail_kblock_row_ptr[global_k_block].item())
        sent_tail_end = int(monolithic_schedule.sentence_tail_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_tail_start, sent_tail_end):
            q_start = int(monolithic_schedule.sentence_tail_q_start[desc_idx].item())
            q_len = int(monolithic_schedule.sentence_tail_q_len[desc_idx].item())
            k_local_start = int(monolithic_schedule.sentence_tail_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.sentence_tail_k_len[desc_idx].item())
            row0_prefix_len = int(monolithic_schedule.sentence_tail_row0_prefix_len[desc_idx].item())
            for q_offset in range(q_len):
                prefix = min(k_len, row0_prefix_len + q_offset)
                if prefix > 0:
                    attend[
                        batch_idx,
                        q_start + q_offset,
                        block_k_start + k_local_start : block_k_start + k_local_start + prefix,
                    ] = True

        anchor_full_start = int(monolithic_schedule.anchor_full_kblock_row_ptr[global_k_block].item())
        anchor_full_end = int(monolithic_schedule.anchor_full_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_full_start, anchor_full_end):
            q_row_start = int(monolithic_schedule.anchor_full_q_row_start[desc_idx].item())
            q_row_count = int(monolithic_schedule.anchor_full_q_row_count[desc_idx].item())
            k_local_start = int(monolithic_schedule.anchor_full_k_local_start[desc_idx].item())
            k_len = int(monolithic_schedule.anchor_full_k_len[desc_idx].item())
            query_rows = monolithic_schedule.anchor_q_indices[q_row_start : q_row_start + q_row_count].long() % seqlen
            attend[
                batch_idx,
                query_rows,
                block_k_start + k_local_start : block_k_start + k_local_start + k_len,
            ] = True

        anchor_tail_start = int(monolithic_schedule.anchor_tail_kblock_row_ptr[global_k_block].item())
        anchor_tail_end = int(monolithic_schedule.anchor_tail_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_tail_start, anchor_tail_end):
            q_row_start = int(monolithic_schedule.anchor_tail_q_row_start[desc_idx].item())
            q_row_count = int(monolithic_schedule.anchor_tail_q_row_count[desc_idx].item())
            k_local_start = int(monolithic_schedule.anchor_tail_k_local_start[desc_idx].item())
            prefix_row_start = int(monolithic_schedule.anchor_tail_prefix_row_start[desc_idx].item())
            query_rows = monolithic_schedule.anchor_q_indices[q_row_start : q_row_start + q_row_count].long() % seqlen
            prefix_rows = monolithic_schedule.anchor_prefix_len[prefix_row_start : prefix_row_start + q_row_count].long()
            for query_pos, prefix in zip(query_rows.tolist(), prefix_rows.tolist()):
                if prefix > 0:
                    attend[
                        batch_idx,
                        query_pos,
                        block_k_start + k_local_start : block_k_start + k_local_start + prefix,
                    ] = True

    return attend


def _get_hsa_monolithic_backward_schedule(
    schedule: HSASchedule,
    *,
    k_block_size: int = 128,
    anchor_row_panel_size: int = 64,
) -> HSAMonolithicBackwardSchedule:
    cache = getattr(schedule, "_hsa_monolithic_backward_cache", None)
    cache_key = (str(schedule.sentence_start.device), k_block_size, anchor_row_panel_size)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    hybrid_schedule = _get_hsa_hybrid_backward_schedule(
        schedule,
        k_block_size=k_block_size,
        anchor_row_panel_size=anchor_row_panel_size,
    )
    monolithic_schedule = _build_hsa_monolithic_backward_schedule(
        hybrid_schedule,
        seqlen=schedule.seqlen,
        device=schedule.sentence_start.device,
    )
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_monolithic_backward_cache", cache)
    cache[cache_key] = monolithic_schedule
    return monolithic_schedule


def _build_hsa_fused_forward_schedule(
    schedule: HSASchedule,
    *,
    q_block_size: int,
    k_block_size: int = 128,
) -> HSAFusedForwardSchedule:
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    blocks_per_batch = (seqlen + q_block_size - 1) // q_block_size
    total_q_blocks = bsz * blocks_per_batch
    device = schedule.sentence_start.device

    sentence_segment_ptr = schedule.sentence_segment_ptr.detach().cpu().tolist()
    sentence_segment_pos = schedule.sentence_segment_pos.detach().cpu().tolist()
    section_segment_ptr = schedule.section_segment_ptr.detach().cpu().tolist()
    section_segment_pos = schedule.section_segment_pos.detach().cpu().tolist()
    section_self_allowed = schedule.section_self_allowed.detach().cpu().tolist()
    document_segment_ptr = schedule.document_segment_ptr.detach().cpu().tolist()
    document_segment_pos = schedule.document_segment_pos.detach().cpu().tolist()
    document_self_allowed = schedule.document_self_allowed.detach().cpu().tolist()

    sentence_buckets: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(total_q_blocks)]
    anchor_buckets: list[list[tuple[int, int, int, int, int, int, list[int]]]] = [[] for _ in range(total_q_blocks)]
    sentence_full_buckets: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(total_q_blocks)]
    sentence_tail_buckets: list[list[tuple[int, int, int, int, int, list[int]]]] = [[] for _ in range(total_q_blocks)]
    anchor_full_buckets: list[list[tuple[int, int, int, int, list[int]]]] = [[] for _ in range(total_q_blocks)]
    anchor_tail_buckets: list[list[tuple[int, int, int, int, list[int], list[int]]]] = [[] for _ in range(total_q_blocks)]

    def append_sentence_descriptors(segment_ptr: list[int], segment_pos: list[int]):
        num_segments = len(segment_ptr) - 1
        for segment_id in range(num_segments):
            seg_start = segment_ptr[segment_id]
            seg_end = segment_ptr[segment_id + 1]
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            q_offset_start = 0
            while q_offset_start < seg_len:
                flat_query = segment_pos[seg_start + q_offset_start]
                batch_idx, token_idx = divmod(flat_query, seqlen)
                q_block = token_idx // q_block_size
                q_offset_end = q_offset_start + 1
                while q_offset_end < seg_len:
                    next_query = segment_pos[seg_start + q_offset_end]
                    next_batch_idx, next_token_idx = divmod(next_query, seqlen)
                    if next_batch_idx != batch_idx or next_token_idx // q_block_size != q_block:
                        break
                    q_offset_end += 1
                global_q_block = batch_idx * blocks_per_batch + q_block
                sentence_buckets[global_q_block].append(
                    (segment_id, q_offset_start, q_offset_end, 0, q_offset_end)
                )
                prefix_min = q_offset_start + 1
                prefix_max = q_offset_end
                for k_offset_start in range(0, prefix_max, k_block_size):
                    k_offset_end = min(k_offset_start + k_block_size, prefix_max)
                    if k_offset_end <= prefix_min:
                        sentence_full_buckets[global_q_block].append(
                            (segment_id, q_offset_start, q_offset_end, k_offset_start, k_offset_end)
                        )
                    else:
                        prefix_rows = [
                            max(0, min(k_offset_end - k_offset_start, q_offset + 1 - k_offset_start))
                            for q_offset in range(q_offset_start, q_offset_end)
                        ]
                        sentence_tail_buckets[global_q_block].append(
                            (
                                segment_id,
                                q_offset_start,
                                q_offset_end,
                                k_offset_start,
                                k_offset_end,
                                prefix_rows,
                            )
                        )
                q_offset_start = q_offset_end

    def append_anchor_descriptors(
        kind: int,
        segment_ptr: list[int],
        segment_pos: list[int],
        self_allowed: list[bool],
    ):
        num_segments = len(segment_ptr) - 1
        for segment_id in range(num_segments):
            seg_start = segment_ptr[segment_id]
            seg_end = segment_ptr[segment_id + 1]
            seg_len = seg_end - seg_start
            if seg_len <= 0:
                continue
            q_offset_start = 0
            while q_offset_start < seg_len:
                flat_query = segment_pos[seg_start + q_offset_start]
                batch_idx, token_idx = divmod(flat_query, seqlen)
                q_block = token_idx // q_block_size
                q_offset_end = q_offset_start + 1
                while q_offset_end < seg_len:
                    next_query = segment_pos[seg_start + q_offset_end]
                    next_batch_idx, next_token_idx = divmod(next_query, seqlen)
                    if next_batch_idx != batch_idx or next_token_idx // q_block_size != q_block:
                        break
                    q_offset_end += 1

                prefix_len = [
                    q_offset + (1 if self_allowed[segment_pos[seg_start + q_offset]] else 0)
                    for q_offset in range(q_offset_start, q_offset_end)
                ]
                k_offset_end = max(prefix_len) if prefix_len else 0
                if k_offset_end > 0:
                    global_q_block = batch_idx * blocks_per_batch + q_block
                    anchor_buckets[global_q_block].append(
                        (kind, segment_id, q_offset_start, q_offset_end, 0, k_offset_end, prefix_len)
                    )
                    min_prefix = prefix_len[0]
                    segment_rows = segment_pos[seg_start:seg_end]
                    for k_offset_start in range(0, k_offset_end, k_block_size):
                        k_offset_end_tile = min(k_offset_start + k_block_size, k_offset_end)
                        key_rows = segment_rows[k_offset_start:k_offset_end_tile]
                        if k_offset_end_tile <= min_prefix:
                            anchor_full_buckets[global_q_block].append(
                                (kind, segment_id, q_offset_start, q_offset_end, key_rows)
                            )
                        else:
                            prefix_rows = [
                                max(0, min(k_offset_end_tile - k_offset_start, prefix - k_offset_start))
                                for prefix in prefix_len
                            ]
                            anchor_tail_buckets[global_q_block].append(
                                (
                                    kind,
                                    segment_id,
                                    q_offset_start,
                                    q_offset_end,
                                    key_rows,
                                    prefix_rows,
                                )
                            )
                q_offset_start = q_offset_end

    append_sentence_descriptors(sentence_segment_ptr, sentence_segment_pos)
    append_anchor_descriptors(_DESC_SECTION, section_segment_ptr, section_segment_pos, section_self_allowed)
    append_anchor_descriptors(_DESC_DOCUMENT, document_segment_ptr, document_segment_pos, document_self_allowed)

    sentence_qblock_row_ptr = [0]
    sentence_segment_id: list[int] = []
    sentence_q_offset_start: list[int] = []
    sentence_q_offset_end: list[int] = []
    sentence_k_offset_start: list[int] = []
    sentence_k_offset_end: list[int] = []
    anchor_qblock_row_ptr = [0]
    anchor_kind: list[int] = []
    anchor_segment_id: list[int] = []
    anchor_q_offset_start: list[int] = []
    anchor_q_offset_end: list[int] = []
    anchor_k_offset_start: list[int] = []
    anchor_k_offset_end: list[int] = []
    anchor_prefix_row_ptr = [0]
    anchor_prefix_len: list[int] = []
    sentence_full_qblock_row_ptr = [0]
    sentence_full_segment_id: list[int] = []
    sentence_full_q_offset_start: list[int] = []
    sentence_full_q_offset_end: list[int] = []
    sentence_full_k_offset_start: list[int] = []
    sentence_full_k_offset_end: list[int] = []
    sentence_tail_qblock_row_ptr = [0]
    sentence_tail_segment_id: list[int] = []
    sentence_tail_q_offset_start: list[int] = []
    sentence_tail_q_offset_end: list[int] = []
    sentence_tail_k_offset_start: list[int] = []
    sentence_tail_k_offset_end: list[int] = []
    sentence_tail_prefix_row_ptr = [0]
    sentence_tail_prefix_len: list[int] = []
    anchor_full_qblock_row_ptr = [0]
    anchor_full_kind: list[int] = []
    anchor_full_segment_id: list[int] = []
    anchor_full_q_offset_start: list[int] = []
    anchor_full_q_offset_end: list[int] = []
    anchor_full_k_row_ptr = [0]
    anchor_full_k_indices: list[int] = []
    anchor_tail_qblock_row_ptr = [0]
    anchor_tail_kind: list[int] = []
    anchor_tail_segment_id: list[int] = []
    anchor_tail_q_offset_start: list[int] = []
    anchor_tail_q_offset_end: list[int] = []
    anchor_tail_k_row_ptr = [0]
    anchor_tail_k_indices: list[int] = []
    anchor_tail_prefix_row_ptr = [0]
    anchor_tail_prefix_len: list[int] = []

    for q_block_id in range(total_q_blocks):
        sentence_descs = sorted(sentence_buckets[q_block_id], key=lambda desc: (desc[1], desc[2], desc[0]))
        for segment_id, q_start, q_end, k_start, k_end in sentence_descs:
            sentence_segment_id.append(segment_id)
            sentence_q_offset_start.append(q_start)
            sentence_q_offset_end.append(q_end)
            sentence_k_offset_start.append(k_start)
            sentence_k_offset_end.append(k_end)
        sentence_qblock_row_ptr.append(len(sentence_segment_id))

        anchor_descs = sorted(anchor_buckets[q_block_id], key=lambda desc: (desc[0], desc[2], desc[3], desc[1]))
        for kind, segment_id, q_start, q_end, k_start, k_end, prefix_rows in anchor_descs:
            anchor_kind.append(kind)
            anchor_segment_id.append(segment_id)
            anchor_q_offset_start.append(q_start)
            anchor_q_offset_end.append(q_end)
            anchor_k_offset_start.append(k_start)
            anchor_k_offset_end.append(k_end)
            anchor_prefix_len.extend(prefix_rows)
            anchor_prefix_row_ptr.append(len(anchor_prefix_len))
        anchor_qblock_row_ptr.append(len(anchor_kind))

        sentence_full_descs = sorted(
            sentence_full_buckets[q_block_id],
            key=lambda desc: (desc[3], desc[1], desc[2], desc[0]),
        )
        for segment_id, q_start, q_end, k_start, k_end in sentence_full_descs:
            sentence_full_segment_id.append(segment_id)
            sentence_full_q_offset_start.append(q_start)
            sentence_full_q_offset_end.append(q_end)
            sentence_full_k_offset_start.append(k_start)
            sentence_full_k_offset_end.append(k_end)
        sentence_full_qblock_row_ptr.append(len(sentence_full_segment_id))

        sentence_tail_descs = sorted(
            sentence_tail_buckets[q_block_id],
            key=lambda desc: (desc[3], desc[1], desc[2], desc[0]),
        )
        for segment_id, q_start, q_end, k_start, k_end, prefix_rows in sentence_tail_descs:
            sentence_tail_segment_id.append(segment_id)
            sentence_tail_q_offset_start.append(q_start)
            sentence_tail_q_offset_end.append(q_end)
            sentence_tail_k_offset_start.append(k_start)
            sentence_tail_k_offset_end.append(k_end)
            sentence_tail_prefix_len.extend(prefix_rows)
            sentence_tail_prefix_row_ptr.append(len(sentence_tail_prefix_len))
        sentence_tail_qblock_row_ptr.append(len(sentence_tail_segment_id))

        anchor_full_descs = sorted(
            anchor_full_buckets[q_block_id],
            key=lambda desc: (desc[0], len(desc[4]), desc[2], desc[3], desc[1]),
        )
        for kind, segment_id, q_start, q_end, key_rows in anchor_full_descs:
            anchor_full_kind.append(kind)
            anchor_full_segment_id.append(segment_id)
            anchor_full_q_offset_start.append(q_start)
            anchor_full_q_offset_end.append(q_end)
            anchor_full_k_indices.extend(key_rows)
            anchor_full_k_row_ptr.append(len(anchor_full_k_indices))
        anchor_full_qblock_row_ptr.append(len(anchor_full_kind))

        anchor_tail_descs = sorted(
            anchor_tail_buckets[q_block_id],
            key=lambda desc: (desc[0], len(desc[4]), desc[2], desc[3], desc[1]),
        )
        for kind, segment_id, q_start, q_end, key_rows, prefix_rows in anchor_tail_descs:
            anchor_tail_kind.append(kind)
            anchor_tail_segment_id.append(segment_id)
            anchor_tail_q_offset_start.append(q_start)
            anchor_tail_q_offset_end.append(q_end)
            anchor_tail_k_indices.extend(key_rows)
            anchor_tail_k_row_ptr.append(len(anchor_tail_k_indices))
            anchor_tail_prefix_len.extend(prefix_rows)
            anchor_tail_prefix_row_ptr.append(len(anchor_tail_prefix_len))
        anchor_tail_qblock_row_ptr.append(len(anchor_tail_kind))

    def _tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device=device) if values else _empty_int32(device)

    return HSAFusedForwardSchedule(
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        blocks_per_batch=blocks_per_batch,
        sentence_qblock_row_ptr=torch.tensor(sentence_qblock_row_ptr, dtype=torch.int32, device=device),
        sentence_segment_id=torch.tensor(sentence_segment_id, dtype=torch.int32, device=device)
        if sentence_segment_id
        else _empty_int32(device),
        sentence_q_offset_start=torch.tensor(sentence_q_offset_start, dtype=torch.int32, device=device)
        if sentence_q_offset_start
        else _empty_int32(device),
        sentence_q_offset_end=torch.tensor(sentence_q_offset_end, dtype=torch.int32, device=device)
        if sentence_q_offset_end
        else _empty_int32(device),
        sentence_k_offset_start=torch.tensor(sentence_k_offset_start, dtype=torch.int32, device=device)
        if sentence_k_offset_start
        else _empty_int32(device),
        sentence_k_offset_end=torch.tensor(sentence_k_offset_end, dtype=torch.int32, device=device)
        if sentence_k_offset_end
        else _empty_int32(device),
        anchor_qblock_row_ptr=torch.tensor(anchor_qblock_row_ptr, dtype=torch.int32, device=device),
        anchor_kind=torch.tensor(anchor_kind, dtype=torch.int32, device=device) if anchor_kind else _empty_int32(device),
        anchor_segment_id=torch.tensor(anchor_segment_id, dtype=torch.int32, device=device)
        if anchor_segment_id
        else _empty_int32(device),
        anchor_q_offset_start=torch.tensor(anchor_q_offset_start, dtype=torch.int32, device=device)
        if anchor_q_offset_start
        else _empty_int32(device),
        anchor_q_offset_end=torch.tensor(anchor_q_offset_end, dtype=torch.int32, device=device)
        if anchor_q_offset_end
        else _empty_int32(device),
        anchor_k_offset_start=torch.tensor(anchor_k_offset_start, dtype=torch.int32, device=device)
        if anchor_k_offset_start
        else _empty_int32(device),
        anchor_k_offset_end=torch.tensor(anchor_k_offset_end, dtype=torch.int32, device=device)
        if anchor_k_offset_end
        else _empty_int32(device),
        anchor_prefix_row_ptr=torch.tensor(anchor_prefix_row_ptr, dtype=torch.int32, device=device),
        anchor_prefix_len=_tensor(anchor_prefix_len),
        sentence_full_qblock_row_ptr=torch.tensor(sentence_full_qblock_row_ptr, dtype=torch.int32, device=device),
        sentence_full_segment_id=_tensor(sentence_full_segment_id),
        sentence_full_q_offset_start=_tensor(sentence_full_q_offset_start),
        sentence_full_q_offset_end=_tensor(sentence_full_q_offset_end),
        sentence_full_k_offset_start=_tensor(sentence_full_k_offset_start),
        sentence_full_k_offset_end=_tensor(sentence_full_k_offset_end),
        sentence_tail_qblock_row_ptr=torch.tensor(sentence_tail_qblock_row_ptr, dtype=torch.int32, device=device),
        sentence_tail_segment_id=_tensor(sentence_tail_segment_id),
        sentence_tail_q_offset_start=_tensor(sentence_tail_q_offset_start),
        sentence_tail_q_offset_end=_tensor(sentence_tail_q_offset_end),
        sentence_tail_k_offset_start=_tensor(sentence_tail_k_offset_start),
        sentence_tail_k_offset_end=_tensor(sentence_tail_k_offset_end),
        sentence_tail_prefix_row_ptr=torch.tensor(sentence_tail_prefix_row_ptr, dtype=torch.int32, device=device),
        sentence_tail_prefix_len=_tensor(sentence_tail_prefix_len),
        anchor_full_qblock_row_ptr=torch.tensor(anchor_full_qblock_row_ptr, dtype=torch.int32, device=device),
        anchor_full_kind=_tensor(anchor_full_kind),
        anchor_full_segment_id=_tensor(anchor_full_segment_id),
        anchor_full_q_offset_start=_tensor(anchor_full_q_offset_start),
        anchor_full_q_offset_end=_tensor(anchor_full_q_offset_end),
        anchor_full_k_row_ptr=torch.tensor(anchor_full_k_row_ptr, dtype=torch.int32, device=device),
        anchor_full_k_indices=_tensor(anchor_full_k_indices),
        anchor_tail_qblock_row_ptr=torch.tensor(anchor_tail_qblock_row_ptr, dtype=torch.int32, device=device),
        anchor_tail_kind=_tensor(anchor_tail_kind),
        anchor_tail_segment_id=_tensor(anchor_tail_segment_id),
        anchor_tail_q_offset_start=_tensor(anchor_tail_q_offset_start),
        anchor_tail_q_offset_end=_tensor(anchor_tail_q_offset_end),
        anchor_tail_k_row_ptr=torch.tensor(anchor_tail_k_row_ptr, dtype=torch.int32, device=device),
        anchor_tail_k_indices=_tensor(anchor_tail_k_indices),
        anchor_tail_prefix_row_ptr=torch.tensor(anchor_tail_prefix_row_ptr, dtype=torch.int32, device=device),
        anchor_tail_prefix_len=_tensor(anchor_tail_prefix_len),
    )


def fused_forward_to_attend_mask(
    schedule: HSASchedule,
    fused_schedule: HSAFusedForwardSchedule,
) -> torch.Tensor:
    attend = torch.zeros(
        schedule.batch_size,
        schedule.seqlen,
        schedule.seqlen,
        dtype=torch.bool,
        device=schedule.sentence_start.device,
    )
    seqlen = schedule.seqlen

    for global_q_block in range(fused_schedule.num_q_blocks):
        batch_idx = global_q_block // fused_schedule.blocks_per_batch

        sent_full_start = int(fused_schedule.sentence_full_qblock_row_ptr[global_q_block].item())
        sent_full_end = int(fused_schedule.sentence_full_qblock_row_ptr[global_q_block + 1].item())
        for desc_idx in range(sent_full_start, sent_full_end):
            segment_id = int(fused_schedule.sentence_full_segment_id[desc_idx].item())
            q_offset_start = int(fused_schedule.sentence_full_q_offset_start[desc_idx].item())
            q_offset_end = int(fused_schedule.sentence_full_q_offset_end[desc_idx].item())
            k_offset_start = int(fused_schedule.sentence_full_k_offset_start[desc_idx].item())
            k_offset_end = int(fused_schedule.sentence_full_k_offset_end[desc_idx].item())
            seg_ptr_start = int(schedule.sentence_segment_ptr[segment_id].item())
            seg_ptr_end = int(schedule.sentence_segment_ptr[segment_id + 1].item())
            segment_rows = schedule.sentence_segment_pos[seg_ptr_start:seg_ptr_end].long() % seqlen
            key_rows = segment_rows[k_offset_start:k_offset_end]
            query_rows = segment_rows[q_offset_start:q_offset_end]
            for query_pos in query_rows.tolist():
                attend[batch_idx, query_pos, key_rows.long()] = True

        sent_tail_start = int(fused_schedule.sentence_tail_qblock_row_ptr[global_q_block].item())
        sent_tail_end = int(fused_schedule.sentence_tail_qblock_row_ptr[global_q_block + 1].item())
        for desc_idx in range(sent_tail_start, sent_tail_end):
            segment_id = int(fused_schedule.sentence_tail_segment_id[desc_idx].item())
            q_offset_start = int(fused_schedule.sentence_tail_q_offset_start[desc_idx].item())
            q_offset_end = int(fused_schedule.sentence_tail_q_offset_end[desc_idx].item())
            k_offset_start = int(fused_schedule.sentence_tail_k_offset_start[desc_idx].item())
            k_offset_end = int(fused_schedule.sentence_tail_k_offset_end[desc_idx].item())
            prefix_start = int(fused_schedule.sentence_tail_prefix_row_ptr[desc_idx].item())
            prefix_end = int(fused_schedule.sentence_tail_prefix_row_ptr[desc_idx + 1].item())
            seg_ptr_start = int(schedule.sentence_segment_ptr[segment_id].item())
            seg_ptr_end = int(schedule.sentence_segment_ptr[segment_id + 1].item())
            segment_rows = schedule.sentence_segment_pos[seg_ptr_start:seg_ptr_end].long() % seqlen
            key_rows = segment_rows[k_offset_start:k_offset_end]
            query_rows = segment_rows[q_offset_start:q_offset_end]
            prefix_len = fused_schedule.sentence_tail_prefix_len[prefix_start:prefix_end].long()
            for query_pos, prefix in zip(query_rows.tolist(), prefix_len.tolist()):
                if prefix > 0:
                    attend[batch_idx, query_pos, key_rows[:prefix].long()] = True

        anchor_full_start = int(fused_schedule.anchor_full_qblock_row_ptr[global_q_block].item())
        anchor_full_end = int(fused_schedule.anchor_full_qblock_row_ptr[global_q_block + 1].item())
        for desc_idx in range(anchor_full_start, anchor_full_end):
            kind = int(fused_schedule.anchor_full_kind[desc_idx].item())
            segment_id = int(fused_schedule.anchor_full_segment_id[desc_idx].item())
            q_offset_start = int(fused_schedule.anchor_full_q_offset_start[desc_idx].item())
            q_offset_end = int(fused_schedule.anchor_full_q_offset_end[desc_idx].item())
            k_row_start = int(fused_schedule.anchor_full_k_row_ptr[desc_idx].item())
            k_row_end = int(fused_schedule.anchor_full_k_row_ptr[desc_idx + 1].item())
            if kind == _DESC_SECTION:
                seg_ptr = schedule.section_segment_ptr
                seg_pos = schedule.section_segment_pos
            else:
                seg_ptr = schedule.document_segment_ptr
                seg_pos = schedule.document_segment_pos
            seg_ptr_start = int(seg_ptr[segment_id].item())
            seg_ptr_end = int(seg_ptr[segment_id + 1].item())
            segment_rows = seg_pos[seg_ptr_start:seg_ptr_end].long() % seqlen
            query_rows = segment_rows[q_offset_start:q_offset_end]
            key_rows = fused_schedule.anchor_full_k_indices[k_row_start:k_row_end].long() % seqlen
            for query_pos in query_rows.tolist():
                attend[batch_idx, query_pos, key_rows.long()] = True

        anchor_tail_start = int(fused_schedule.anchor_tail_qblock_row_ptr[global_q_block].item())
        anchor_tail_end = int(fused_schedule.anchor_tail_qblock_row_ptr[global_q_block + 1].item())
        for desc_idx in range(anchor_tail_start, anchor_tail_end):
            kind = int(fused_schedule.anchor_tail_kind[desc_idx].item())
            segment_id = int(fused_schedule.anchor_tail_segment_id[desc_idx].item())
            q_offset_start = int(fused_schedule.anchor_tail_q_offset_start[desc_idx].item())
            q_offset_end = int(fused_schedule.anchor_tail_q_offset_end[desc_idx].item())
            k_row_start = int(fused_schedule.anchor_tail_k_row_ptr[desc_idx].item())
            k_row_end = int(fused_schedule.anchor_tail_k_row_ptr[desc_idx + 1].item())
            prefix_start = int(fused_schedule.anchor_tail_prefix_row_ptr[desc_idx].item())
            prefix_end = int(fused_schedule.anchor_tail_prefix_row_ptr[desc_idx + 1].item())
            if kind == _DESC_SECTION:
                seg_ptr = schedule.section_segment_ptr
                seg_pos = schedule.section_segment_pos
            else:
                seg_ptr = schedule.document_segment_ptr
                seg_pos = schedule.document_segment_pos
            seg_ptr_start = int(seg_ptr[segment_id].item())
            seg_ptr_end = int(seg_ptr[segment_id + 1].item())
            segment_rows = seg_pos[seg_ptr_start:seg_ptr_end].long() % seqlen
            query_rows = segment_rows[q_offset_start:q_offset_end]
            key_rows = fused_schedule.anchor_tail_k_indices[k_row_start:k_row_end].long() % seqlen
            prefix_len = fused_schedule.anchor_tail_prefix_len[prefix_start:prefix_end].long()
            for query_pos, prefix in zip(query_rows.tolist(), prefix_len.tolist()):
                if prefix > 0:
                    attend[batch_idx, query_pos, key_rows[:prefix].long()] = True

    return attend


def _get_hsa_forward_q_block_size(q: torch.Tensor, k: torch.Tensor) -> int:
    num_q_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    arch_major = torch.cuda.get_device_capability(q.device)[0]
    q_stage = 2 if arch_major == 10 and q.shape[1] * qhead_per_kvhead > 128 else 1
    return q_stage * 128


def _get_hsa_hybrid_backward_schedule(
    schedule: HSASchedule,
    *,
    k_block_size: int = 128,
    anchor_row_panel_size: int = 64,
) -> HSAHybridBackwardSchedule:
    cache = getattr(schedule, "_hsa_hybrid_backward_cache", None)
    cache_key = (str(schedule.sentence_start.device), k_block_size, anchor_row_panel_size)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    hybrid_schedule = _build_hsa_hybrid_backward_schedule(
        schedule,
        k_block_size=k_block_size,
        anchor_row_panel_size=anchor_row_panel_size,
    )
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_hybrid_backward_cache", cache)
    cache[cache_key] = hybrid_schedule
    return hybrid_schedule


def _get_hsa_hybrid_backward_batches(
    schedule: HSASchedule,
    hybrid_schedule: HSAHybridBackwardSchedule,
) -> tuple[list[HSAHybridBackwardBatch], list[HSAHybridBackwardBatch]]:
    cache = getattr(schedule, "_hsa_hybrid_backward_batch_cache", None)
    cache_key = (str(schedule.sentence_start.device), hybrid_schedule.k_block_size, hybrid_schedule.anchor_row_panel_size)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    device = schedule.sentence_start.device
    seqlen = schedule.seqlen
    sentence_batches: list[HSAHybridBackwardBatch] = []

    anchor_batches: list[HSAHybridBackwardBatch] = []

    for global_k_block in range(hybrid_schedule.num_k_blocks):
        batch_idx = global_k_block // hybrid_schedule.blocks_per_batch
        batch_base = batch_idx * seqlen

        sentence_entries: list[tuple[list[int], list[int]]] = []
        sent_desc_start = int(hybrid_schedule.sentence_kblock_row_ptr[global_k_block].item())
        sent_desc_end = int(hybrid_schedule.sentence_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(sent_desc_start, sent_desc_end):
            q_start = int(hybrid_schedule.sentence_q_start[desc_idx].item())
            q_len = int(hybrid_schedule.sentence_q_len[desc_idx].item())
            k_start = int(hybrid_schedule.sentence_k_start[desc_idx].item())
            k_len = int(hybrid_schedule.sentence_k_len[desc_idx].item())
            q_rows = list(range(batch_base + q_start, batch_base + q_start + q_len))
            k_rows = list(range(batch_base + k_start, batch_base + k_start + k_len))
            sentence_entries.append((q_rows, k_rows))
        if sentence_entries:
            max_q = max(len(q_rows) for q_rows, _ in sentence_entries)
            max_k = max(len(k_rows) for _, k_rows in sentence_entries)
            q_indices = torch.zeros((len(sentence_entries), max_q), dtype=torch.int32, device=device)
            k_indices = torch.zeros((len(sentence_entries), max_k), dtype=torch.int32, device=device)
            q_length = torch.zeros(len(sentence_entries), dtype=torch.int32, device=device)
            k_length = torch.zeros(len(sentence_entries), dtype=torch.int32, device=device)
            for entry_idx, (q_rows, k_rows) in enumerate(sentence_entries):
                q_len = len(q_rows)
                k_len = len(k_rows)
                q_length[entry_idx] = q_len
                k_length[entry_idx] = k_len
                q_indices[entry_idx, :q_len] = torch.tensor(q_rows, dtype=torch.int32, device=device)
                k_indices[entry_idx, :k_len] = torch.tensor(k_rows, dtype=torch.int32, device=device)
            sentence_batches.append(
                HSAHybridBackwardBatch(
                    q_indices=q_indices,
                    k_indices=k_indices,
                    q_length=q_length,
                    k_length=k_length,
                    prefix_len=None,
                )
            )

        anchor_entries: list[tuple[list[int], list[int], list[int]]] = []
        anchor_desc_start = int(hybrid_schedule.anchor_kblock_row_ptr[global_k_block].item())
        anchor_desc_end = int(hybrid_schedule.anchor_kblock_row_ptr[global_k_block + 1].item())
        for desc_idx in range(anchor_desc_start, anchor_desc_end):
            q_row_start = int(hybrid_schedule.anchor_q_row_ptr[desc_idx].item())
            q_row_end = int(hybrid_schedule.anchor_q_row_ptr[desc_idx + 1].item())
            k_row_start = int(hybrid_schedule.anchor_k_row_ptr[desc_idx].item())
            k_row_end = int(hybrid_schedule.anchor_k_row_ptr[desc_idx + 1].item())
            prefix_row_start = int(hybrid_schedule.anchor_prefix_row_ptr[desc_idx].item())
            prefix_row_end = int(hybrid_schedule.anchor_prefix_row_ptr[desc_idx + 1].item())
            q_rows = hybrid_schedule.anchor_q_indices[q_row_start:q_row_end].detach().cpu().tolist()
            k_rows = hybrid_schedule.anchor_k_indices[k_row_start:k_row_end].detach().cpu().tolist()
            prefix_rows = hybrid_schedule.anchor_prefix_len[prefix_row_start:prefix_row_end].detach().cpu().tolist()
            anchor_entries.append((q_rows, k_rows, prefix_rows))
        if anchor_entries:
            max_q = max(len(q_rows) for q_rows, _, _ in anchor_entries)
            max_k = max(len(k_rows) for _, k_rows, _ in anchor_entries)
            q_indices = torch.zeros((len(anchor_entries), max_q), dtype=torch.int32, device=device)
            k_indices = torch.zeros((len(anchor_entries), max_k), dtype=torch.int32, device=device)
            q_length = torch.zeros(len(anchor_entries), dtype=torch.int32, device=device)
            k_length = torch.zeros(len(anchor_entries), dtype=torch.int32, device=device)
            prefix_len = torch.zeros((len(anchor_entries), max_q), dtype=torch.int32, device=device)
            for entry_idx, (q_rows, k_rows, prefix_rows) in enumerate(anchor_entries):
                q_len = len(q_rows)
                k_len = len(k_rows)
                q_length[entry_idx] = q_len
                k_length[entry_idx] = k_len
                q_indices[entry_idx, :q_len] = torch.tensor(q_rows, dtype=torch.int32, device=device)
                k_indices[entry_idx, :k_len] = torch.tensor(k_rows, dtype=torch.int32, device=device)
                prefix_len[entry_idx, :q_len] = torch.tensor(prefix_rows, dtype=torch.int32, device=device)
            anchor_batches.append(
                HSAHybridBackwardBatch(
                    q_indices=q_indices,
                    k_indices=k_indices,
                    q_length=q_length,
                    k_length=k_length,
                    prefix_len=prefix_len,
                )
            )

    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_hybrid_backward_batch_cache", cache)
    cache[cache_key] = (sentence_batches, anchor_batches)
    return sentence_batches, anchor_batches


def _get_hsa_fused_forward_schedule(schedule: HSASchedule, q: torch.Tensor, k: torch.Tensor) -> HSAFusedForwardSchedule:
    cache = getattr(schedule, "_hsa_fused_forward_cache", None)
    q_block_size = _get_hsa_forward_q_block_size(q, k)
    cache_key = (str(schedule.sentence_start.device), q_block_size, 128)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    fused_schedule = _build_hsa_fused_forward_schedule(
        schedule,
        q_block_size=q_block_size,
        k_block_size=128,
    )
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_fused_forward_cache", cache)
    cache[cache_key] = fused_schedule
    return fused_schedule


def _get_hsa_fused_forward_batches(
    schedule: HSASchedule,
    fused_schedule: HSAFusedForwardSchedule,
) -> tuple[list[HSAFusedForwardBatch], list[HSAFusedForwardBatch]]:
    cache = getattr(schedule, "_hsa_fused_forward_batch_cache", None)
    cache_key = (str(schedule.sentence_start.device), fused_schedule.q_block_size, fused_schedule.k_block_size)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    device = schedule.sentence_start.device
    sentence_segment_ptr = schedule.sentence_segment_ptr.detach().cpu().tolist()
    sentence_segment_pos = schedule.sentence_segment_pos.detach().cpu().tolist()
    section_segment_ptr = schedule.section_segment_ptr.detach().cpu().tolist()
    section_segment_pos = schedule.section_segment_pos.detach().cpu().tolist()
    document_segment_ptr = schedule.document_segment_ptr.detach().cpu().tolist()
    document_segment_pos = schedule.document_segment_pos.detach().cpu().tolist()

    sentence_entries: list[tuple[list[int], list[int], int]] = []
    for desc_idx in range(fused_schedule.sentence_segment_id.numel()):
        segment_id = int(fused_schedule.sentence_segment_id[desc_idx].item())
        q_offset_start = int(fused_schedule.sentence_q_offset_start[desc_idx].item())
        q_offset_end = int(fused_schedule.sentence_q_offset_end[desc_idx].item())
        k_offset_end = int(fused_schedule.sentence_k_offset_end[desc_idx].item())
        seg_start = sentence_segment_ptr[segment_id]
        seg_end = sentence_segment_ptr[segment_id + 1]
        segment_rows = sentence_segment_pos[seg_start:seg_end]
        sentence_entries.append((segment_rows[q_offset_start:q_offset_end], segment_rows[:k_offset_end], q_offset_start))

    sentence_batches: list[HSAFusedForwardBatch] = []
    if sentence_entries:
        max_q = max(len(q_rows) for q_rows, _, _ in sentence_entries)
        max_k = max(len(k_rows) for _, k_rows, _ in sentence_entries)
        q_indices = torch.zeros((len(sentence_entries), max_q), dtype=torch.int32, device=device)
        k_indices = torch.zeros((len(sentence_entries), max_k), dtype=torch.int32, device=device)
        q_length = torch.zeros(len(sentence_entries), dtype=torch.int32, device=device)
        k_length = torch.zeros(len(sentence_entries), dtype=torch.int32, device=device)
        q_offset_start = torch.zeros(len(sentence_entries), dtype=torch.int32, device=device)
        for entry_idx, (q_rows, k_rows, q_start) in enumerate(sentence_entries):
            q_len = len(q_rows)
            k_len = len(k_rows)
            q_length[entry_idx] = q_len
            k_length[entry_idx] = k_len
            q_offset_start[entry_idx] = q_start
            q_indices[entry_idx, :q_len] = torch.tensor(q_rows, dtype=torch.int32, device=device)
            k_indices[entry_idx, :k_len] = torch.tensor(k_rows, dtype=torch.int32, device=device)
        sentence_batches.append(
            HSAFusedForwardBatch(
                q_indices=q_indices,
                k_indices=k_indices,
                q_length=q_length,
                k_length=k_length,
                q_offset_start=q_offset_start,
                prefix_len=None,
            )
        )

    anchor_entries: list[tuple[list[int], list[int], list[int]]] = []
    for desc_idx in range(fused_schedule.anchor_kind.numel()):
        kind = int(fused_schedule.anchor_kind[desc_idx].item())
        segment_id = int(fused_schedule.anchor_segment_id[desc_idx].item())
        q_offset_start = int(fused_schedule.anchor_q_offset_start[desc_idx].item())
        q_offset_end = int(fused_schedule.anchor_q_offset_end[desc_idx].item())
        k_offset_end = int(fused_schedule.anchor_k_offset_end[desc_idx].item())
        prefix_start = int(fused_schedule.anchor_prefix_row_ptr[desc_idx].item())
        prefix_end = int(fused_schedule.anchor_prefix_row_ptr[desc_idx + 1].item())
        if kind == _DESC_SECTION:
            seg_ptr = section_segment_ptr
            seg_pos = section_segment_pos
        else:
            seg_ptr = document_segment_ptr
            seg_pos = document_segment_pos
        seg_start = seg_ptr[segment_id]
        seg_end = seg_ptr[segment_id + 1]
        segment_rows = seg_pos[seg_start:seg_end]
        anchor_entries.append(
            (
                segment_rows[q_offset_start:q_offset_end],
                segment_rows[:k_offset_end],
                fused_schedule.anchor_prefix_len[prefix_start:prefix_end].detach().cpu().tolist(),
            )
        )

    anchor_batches: list[HSAFusedForwardBatch] = []
    if anchor_entries:
        max_q = max(len(q_rows) for q_rows, _, _ in anchor_entries)
        max_k = max(len(k_rows) for _, k_rows, _ in anchor_entries)
        q_indices = torch.zeros((len(anchor_entries), max_q), dtype=torch.int32, device=device)
        k_indices = torch.zeros((len(anchor_entries), max_k), dtype=torch.int32, device=device)
        q_length = torch.zeros(len(anchor_entries), dtype=torch.int32, device=device)
        k_length = torch.zeros(len(anchor_entries), dtype=torch.int32, device=device)
        q_offset_start = torch.zeros(len(anchor_entries), dtype=torch.int32, device=device)
        prefix_len = torch.zeros((len(anchor_entries), max_q), dtype=torch.int32, device=device)
        for entry_idx, (q_rows, k_rows, prefix_rows) in enumerate(anchor_entries):
            q_len = len(q_rows)
            k_len = len(k_rows)
            q_length[entry_idx] = q_len
            k_length[entry_idx] = k_len
            q_indices[entry_idx, :q_len] = torch.tensor(q_rows, dtype=torch.int32, device=device)
            k_indices[entry_idx, :k_len] = torch.tensor(k_rows, dtype=torch.int32, device=device)
            prefix_len[entry_idx, :q_len] = torch.tensor(prefix_rows, dtype=torch.int32, device=device)
        anchor_batches.append(
            HSAFusedForwardBatch(
                q_indices=q_indices,
                k_indices=k_indices,
                q_length=q_length,
                k_length=k_length,
                q_offset_start=q_offset_start,
                prefix_len=prefix_len,
            )
        )

    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_fused_forward_batch_cache", cache)
    cache[cache_key] = (sentence_batches, anchor_batches)
    return sentence_batches, anchor_batches


def _get_hsa_block_sparse_runtime(
    schedule: HSASchedule,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    require_backward: Optional[bool] = None,
) -> HSABlockSparseRuntime:
    cache = getattr(schedule, "_hsa_block_sparse_runtime_cache", None)
    require_backward = not _use_hsa_runtime_forward_only() if require_backward is None else require_backward
    forward_block_q = _get_hsa_forward_q_block_size(q, k)
    backward_block_q = _get_hsa_backward_block_q()
    forward_block_k = 128
    backward_block_k = _get_hsa_backward_block_k()
    backward_subtile_factor = _get_hsa_backward_subtile_factor()
    # FA4 backward expects the sparse Q block to be subtile_factor * tile_m.
    backward_sparse_block_q = backward_block_q * backward_subtile_factor
    cache_key = (
        str(q.device),
        forward_block_q,
        forward_block_k,
        backward_block_q,
        backward_block_k,
        backward_subtile_factor,
        require_backward,
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    forward_sparse, forward_tile_masks = _build_forward_hsa_tile_masks(
        schedule,
        q_block_size=forward_block_q,
        k_block_size=forward_block_k,
    )

    if require_backward:
        backward_sparse, backward_packed_masks = _build_backward_hsa_packed_masks(
            schedule,
            q_block_size=backward_sparse_block_q,
            k_block_size=backward_block_k,
        )
        backward_aux_tensors = [
            _tag_aux_tensor(backward_packed_masks.block_id_table),
            _tag_aux_tensor(backward_packed_masks.mask_words),
            _tag_aux_tensor(backward_packed_masks.row_group_nonempty),
        ]
        backward_sparse_torch = _to_block_sparse_tensors_torch(backward_sparse)
    else:
        backward_sparse = None
        backward_packed_masks = None
        backward_aux_tensors = []
        backward_sparse_torch = None

    runtime = HSABlockSparseRuntime(
        forward_sparse=forward_sparse,
        forward_tile_masks=forward_tile_masks,
        backward_sparse=backward_sparse,
        backward_packed_masks=backward_packed_masks,
        forward_aux_tensors=_build_hsa_forward_tile_mask_aux_tensors(forward_tile_masks),
        backward_aux_tensors=backward_aux_tensors,
        forward_sparse_torch=_to_block_sparse_tensors_torch(forward_sparse),
        backward_sparse_torch=backward_sparse_torch,
        forward_block_q=forward_block_q,
        forward_block_k=forward_block_k,
        backward_block_q=backward_block_q,
        backward_block_k=backward_block_k,
        backward_subtile_factor=backward_subtile_factor,
    )
    if _use_hsa_synthetic_grid():
        runtime.forward_synthetic_grid = _ensure_hsa_synthetic_grid_metadata(
            schedule,
            runtime,
            require_full_forward_plan=False,
            require_backward=False,
        )
        runtime.synthetic_grid = runtime.forward_synthetic_grid
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_block_sparse_runtime_cache", cache)
    cache[cache_key] = runtime
    return runtime


def _to_block_sparse_tensors_torch(sparse_tensors: HSABlockSparseTensors):
    try:
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    except ModuleNotFoundError:
        return None

    return BlockSparseTensorsTorch(
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        block_size=sparse_tensors.block_size,
    )


def _use_hsa_synthetic_grid() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1"


def _get_hsa_synthetic_logical_block_size(axis: str) -> int:
    axis = axis.lower()
    if axis not in ("q", "k"):
        raise ValueError(f"Unknown synthetic logical block axis: {axis}")
    specific_key = f"FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_{axis.upper()}"
    value = os.environ.get(specific_key, os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK", "32"))
    logical_block = int(value)
    if logical_block not in (2, 4, 8, 16, 32, 64, 128):
        raise ValueError(
            f"{specific_key} / FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK must be one of 2, 4, 8, 16, 32, 64, or 128; got {logical_block}"
        )
    return logical_block


def _get_hsa_synthetic_max_packed_k(logical_block_k: int) -> int:
    max_packed_k = int(os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K", "128"))
    if max_packed_k <= 0 or max_packed_k > 128:
        raise ValueError("FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K must be between 1 and 128")
    if max_packed_k % logical_block_k != 0:
        raise ValueError(
            "FLASH_ATTN_HSA_SYNTHETIC_MAX_PACKED_K must be a multiple of FLASH_ATTN_HSA_SYNTHETIC_LOGICAL_BLOCK_K"
        )
    return max_packed_k


def _get_hsa_synthetic_max_direct_segments() -> int:
    max_direct_segments = int(os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS", "1"))
    if max_direct_segments not in (1, 2, 3, 4):
        raise ValueError("FLASH_ATTN_HSA_SYNTHETIC_MAX_DIRECT_SEGMENTS must be 1, 2, 3, or 4")
    return max_direct_segments


def _get_hsa_synthetic_max_runtime_qgroup_segments() -> int:
    max_runtime_qgroup_segments = int(os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MAX_RUNTIME_QGROUP_SEGMENTS", "0"))
    if max_runtime_qgroup_segments < 0:
        raise ValueError("FLASH_ATTN_HSA_SYNTHETIC_MAX_RUNTIME_QGROUP_SEGMENTS must be >= 0")
    return max_runtime_qgroup_segments


def _get_hsa_synthetic_max_runtime_qgroup_segments_p90() -> int:
    max_runtime_qgroup_segments_p90 = int(
        os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MAX_RUNTIME_QGROUP_SEGMENTS_P90", "0")
    )
    if max_runtime_qgroup_segments_p90 < 0:
        raise ValueError("FLASH_ATTN_HSA_SYNTHETIC_MAX_RUNTIME_QGROUP_SEGMENTS_P90 must be >= 0")
    return max_runtime_qgroup_segments_p90


def _get_hsa_synthetic_hybrid_max_qgroup_segments() -> int:
    hybrid_max_qgroup_segments = int(os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_HYBRID_MAX_QGROUP_SEGMENTS", "0"))
    if hybrid_max_qgroup_segments < 0:
        raise ValueError("FLASH_ATTN_HSA_SYNTHETIC_HYBRID_MAX_QGROUP_SEGMENTS must be >= 0")
    return hybrid_max_qgroup_segments


def _segment_count_quantile(segment_counts: list[int], quantile: float) -> int:
    if not segment_counts:
        return 0
    sorted_counts = sorted(int(count) for count in segment_counts)
    rank = max(0, min(len(sorted_counts) - 1, math.ceil(len(sorted_counts) * quantile) - 1))
    return sorted_counts[rank]


def _use_hsa_synthetic_sparse_parse_fwd() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_PARSE_SPARSE_FWD", "0") == "1"


def _use_hsa_synthetic_fast_partition() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_FAST_PARTITION", "0") == "1"


def _use_hsa_synthetic_forward_plan_only() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_FORWARD_PLAN_ONLY", "0") == "1"


def _forward_execution_plan_has_bucket_execution(plan: Optional[dict]) -> bool:
    if not isinstance(plan, dict):
        return False
    required_keys = (
        "qgroup_bucket_packed_q",
        "qgroup_bucket_execution_bucket_range",
        "qgroup_bucket_execution_bucket_idx",
        "bucket_size",
        "bucket_packed_q",
        "bucket_packed_k",
        "bucket_q_row_range",
        "bucket_q_src_row_range",
        "bucket_k_row_range",
    )
    return all(key in plan for key in required_keys)


def _get_precomputed_forward_direct_plan(
    schedule: HSASchedule,
    *,
    forward_block_q: int,
    logical_block_q: int,
    logical_block_k: int,
    max_packed_k: int,
    max_direct_segments: int,
    device: Optional[torch.device | str] = None,
) -> Optional[dict]:
    payload = getattr(schedule, "_precomputed_forward_direct_plan_payload", None)
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("forward_block_q", -1)) != int(forward_block_q):
            continue
        if int(entry.get("logical_block_q", -1)) != int(logical_block_q):
            continue
        if int(entry.get("logical_block_k", -1)) != int(logical_block_k):
            continue
        if int(entry.get("max_packed_k", -1)) != int(max_packed_k):
            continue
        if int(entry.get("max_direct_segments", -1)) != int(max_direct_segments):
            continue
        direct_execution_plan = entry.get("direct_execution_plan")
        if isinstance(direct_execution_plan, dict):
            if device is None:
                return direct_execution_plan
            cache = getattr(schedule, "_precomputed_forward_direct_plan_device_cache", None)
            cache_key = (
                int(forward_block_q),
                int(logical_block_q),
                int(logical_block_k),
                int(max_packed_k),
                int(max_direct_segments),
                str(torch.device(device)),
            )
            if cache is not None and cache_key in cache:
                return cache[cache_key]
            moved_direct_execution_plan = _move_nested_tensors(direct_execution_plan, device=device)
            if cache is None:
                cache = {}
                setattr(schedule, "_precomputed_forward_direct_plan_device_cache", cache)
            cache[cache_key] = moved_direct_execution_plan
            return moved_direct_execution_plan
    return None


def _get_precomputed_cached_generalized_forward_payload(
    schedule: HSASchedule,
    *,
    forward_block_q: int,
    logical_block_q: int,
    logical_block_k: int,
    max_packed_k: int,
    max_direct_segments: int,
    device: Optional[torch.device | str] = None,
    allow_same_forward_block_fallback: bool = False,
) -> Optional[dict]:
    def _move_cached_generalized_payload(entry: dict[str, Any]) -> Optional[dict]:
        cached_payload = entry.get("cached_generalized_forward_payload")
        if not isinstance(cached_payload, dict):
            return None
        if device is None:
            return cached_payload
        cache = getattr(schedule, "_precomputed_cached_generalized_forward_payload_device_cache", None)
        cache_key = (
            int(entry.get("forward_block_q", -1)),
            int(entry.get("logical_block_q", -1)),
            int(entry.get("logical_block_k", -1)),
            int(entry.get("max_packed_k", -1)),
            int(entry.get("max_direct_segments", -1)),
            str(torch.device(device)),
        )
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        moved_payload = _move_nested_tensors(cached_payload, device=device)
        if cache is None:
            cache = {}
            setattr(schedule, "_precomputed_cached_generalized_forward_payload_device_cache", cache)
        cache[cache_key] = moved_payload
        return moved_payload

    payload = getattr(schedule, "_precomputed_forward_direct_plan_payload", None)
    if not isinstance(payload, dict):
        return None
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return None
    fallback_entry = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("forward_block_q", -1)) != int(forward_block_q):
            continue
        cached_payload = entry.get("cached_generalized_forward_payload")
        if not isinstance(cached_payload, dict):
            continue
        if int(entry.get("logical_block_q", -1)) != int(logical_block_q):
            if allow_same_forward_block_fallback and fallback_entry is None:
                fallback_entry = entry
            continue
        if int(entry.get("logical_block_k", -1)) != int(logical_block_k):
            if allow_same_forward_block_fallback and fallback_entry is None:
                fallback_entry = entry
            continue
        if int(entry.get("max_packed_k", -1)) != int(max_packed_k):
            if allow_same_forward_block_fallback and fallback_entry is None:
                fallback_entry = entry
            continue
        if int(entry.get("max_direct_segments", -1)) != int(max_direct_segments):
            if allow_same_forward_block_fallback and fallback_entry is None:
                fallback_entry = entry
            continue
        return _move_cached_generalized_payload(entry)
    if allow_same_forward_block_fallback and fallback_entry is not None:
        return _move_cached_generalized_payload(fallback_entry)
    return None


def _resolve_precomputed_cached_generalized_forward_payload(
    schedule: HSASchedule,
    q: torch.Tensor,
    k: torch.Tensor,
) -> Optional[dict]:
    if not _use_hsa_synthetic_grid():
        return None
    if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "0") != "1":
        return None
    if _get_hsa_blocksparse_backward_mode(schedule) != "sparse_mask":
        return None
    logical_block_q = _get_hsa_synthetic_logical_block_size("q")
    logical_block_k = _get_hsa_synthetic_logical_block_size("k")
    max_packed_k = _get_hsa_synthetic_max_packed_k(logical_block_k)
    max_direct_segments = _get_hsa_synthetic_max_direct_segments()
    forward_block_q = _get_hsa_forward_q_block_size(q, k)
    allow_forward_block_fallback = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0") == "1"
    return _get_precomputed_cached_generalized_forward_payload(
        schedule,
        forward_block_q=forward_block_q,
        logical_block_q=logical_block_q,
        logical_block_k=logical_block_k,
        max_packed_k=max_packed_k,
        max_direct_segments=max_direct_segments,
        device=q.device,
        allow_same_forward_block_fallback=allow_forward_block_fallback,
    )


def _can_use_cached_generalized_synthetic_micro_bwd(
    schedule: HSASchedule,
    runtime: HSABlockSparseRuntime,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    if os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0") != "1":
        return False
    if not _can_use_hsa_synthetic_grid_for_inputs(schedule, q, k, runtime=runtime):
        return False
    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _can_use_direct_synthetic_micro_runtime

    metadata = _ensure_hsa_synthetic_grid_metadata(schedule, runtime, require_backward=True)
    q_flat = q.reshape(-1, q.shape[2], q.shape[3]).contiguous()
    k_flat = k.reshape(-1, k.shape[2], k.shape[3]).contiguous()
    v_flat = v.reshape(-1, v.shape[2], v.shape[3]).contiguous()
    return _can_use_direct_synthetic_micro_runtime(metadata, q_flat, k_flat, v_flat)


def _build_precomputed_direct_only_synthetic_grid_metadata(
    *,
    device,
    logical_block_q: int,
    logical_block_k: int,
    physical_block_q: int,
    physical_block_k: int,
    max_packed_k: int,
    max_direct_segments: int,
    sparse_parse_fwd: bool,
    direct_execution_plan: dict,
) -> HSASyntheticGridMetadata:
    empty_i32 = _empty_int32(device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)
    empty_bool = torch.empty(0, dtype=torch.bool, device=device)
    zero_ptr = torch.tensor([0], dtype=torch.int32, device=device)
    forward_execution_plan = {
        "direct_execution_plan": direct_execution_plan,
        "hybrid_direct_execution_plan": None,
        "hybrid_regular_execution_plan": None,
        "hybrid_selected_qgroup_idx": [],
    }
    return HSASyntheticGridMetadata(
        logical_block_q=logical_block_q,
        logical_block_k=logical_block_k,
        physical_block_q=physical_block_q,
        physical_block_k=physical_block_k,
        tile_batch_idx=empty_i32,
        tile_q_block_idx=empty_i32,
        tile_k_block_idx=empty_i32,
        tile_q_subgroup_idx=empty_i32,
        tile_q_row_ptr=zero_ptr,
        tile_q_rows=empty_i32,
        tile_k_row_ptr=zero_ptr,
        tile_k_rows=empty_i32,
        tile_logical_pair_row_ptr=zero_ptr,
        tile_logical_pairs=empty_i32,
        compact_mask_row_ptr=zero_ptr,
        compact_mask_col_idx=empty_i32,
        tile_allowed_pairs=empty_i32,
        tile_packed_q=empty_i32,
        tile_packed_k=empty_i32,
        tile_dense=empty_bool,
        bucket_row_ptr=zero_ptr,
        bucket_tile_idx=empty_i32,
        bucket_packed_q=empty_i32,
        bucket_packed_k=empty_i32,
        bucket_dense=empty_bool,
        bucket_allowed_pairs=empty_i32,
        bucket_fill=empty_f32,
        max_packed_k=max_packed_k,
        max_direct_segments=max_direct_segments,
        sparse_parse_fwd=sparse_parse_fwd,
        tile_fill=empty_f32,
        tile_q_length=empty_i32,
        tile_k_length=empty_i32,
        bucket_q_row_idx_row_ptr=zero_ptr,
        bucket_q_row_idx=empty_i32,
        bucket_q_src_row_idx=empty_i32,
        bucket_k_row_idx_row_ptr=zero_ptr,
        bucket_k_row_idx=empty_i32,
        bucket_q_length=empty_i32,
        bucket_k_length=empty_i32,
        bucket_split_slot=empty_i32,
        bucket_qgroup_bucket_idx=empty_i32,
        bucket_mask_word_row_ptr=zero_ptr,
        bucket_mask_words=empty_i32,
        bucket_words_per_row=empty_i32,
        qgroup_row_ptr=zero_ptr,
        qgroup_rows=empty_i32,
        qgroup_length=empty_i32,
        qgroup_packed_q=empty_i32,
        qgroup_num_splits=empty_i32,
        qgroup_bucket_row_ptr=zero_ptr,
        qgroup_bucket_idx=empty_i32,
        qgroup_bucket_packed_q=empty_i32,
        qgroup_bucket_q_row_idx_row_ptr=zero_ptr,
        qgroup_bucket_q_row_idx=empty_i32,
        qgroup_bucket_split_bucket_row_ptr=zero_ptr,
        qgroup_bucket_split_bucket_idx=empty_i32,
        forward_execution_plan=forward_execution_plan,
        host_index_view=None,
    )


def _use_hsa_unpacked_direct_fwd() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_UNPACKED_DIRECT_FWD", "0") == "1"


def _use_hsa_runtime_forward_only() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_RUNTIME_FORWARD_ONLY", "0") == "1"


def _require_hsa_synthetic_row_compact() -> bool:
    return os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_REQUIRE_ROW_COMPACT", "0") == "1"


def _hsa_synthetic_forward_has_row_compact_plan(metadata: Optional[HSASyntheticGridMetadata]) -> bool:
    if metadata is None or metadata.forward_execution_plan is None:
        return False
    direct_plan = metadata.forward_execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return False
    row_plan = direct_plan.get("row_compact_plan")
    if row_plan is None:
        return False
    row_k_caps = row_plan.get("bucket_row_k_cap")
    if row_k_caps is None or len(row_k_caps) <= 0:
        return False
    row_k_cap_limit = int(row_plan.get("row_k_cap_limit", 0))
    return row_k_cap_limit > 0 and max(int(row_k_cap) for row_k_cap in row_k_caps) <= row_k_cap_limit


def _hsa_synthetic_forward_segments_within_runtime_limit(metadata: Optional[HSASyntheticGridMetadata]) -> bool:
    max_runtime_qgroup_segments = _get_hsa_synthetic_max_runtime_qgroup_segments()
    max_runtime_qgroup_segments_p90 = _get_hsa_synthetic_max_runtime_qgroup_segments_p90()
    if max_runtime_qgroup_segments <= 0 and max_runtime_qgroup_segments_p90 <= 0:
        return True
    if metadata is None or metadata.forward_execution_plan is None:
        return False
    direct_plan = metadata.forward_execution_plan.get("direct_execution_plan")
    if direct_plan is None:
        return False
    qgroup_num_segments = direct_plan.get("qgroup_num_segments")
    if qgroup_num_segments is None or len(qgroup_num_segments) <= 0:
        return False
    qgroup_num_segments = [int(num_segments) for num_segments in qgroup_num_segments]
    if max_runtime_qgroup_segments > 0 and max(qgroup_num_segments) > max_runtime_qgroup_segments:
        return False
    if (
        max_runtime_qgroup_segments_p90 > 0
        and _segment_count_quantile(qgroup_num_segments, 0.9) > max_runtime_qgroup_segments_p90
    ):
        return False
    return True


def _can_use_hsa_synthetic_grid_for_inputs(
    schedule: HSASchedule,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    runtime=None,
) -> bool:
    if (
        not _use_hsa_synthetic_grid()
        or _use_hsa_unpacked_direct_fwd()
        or _schedule_has_only_sentence_backward_families(schedule)
        or q.shape[2] != k.shape[2]
    ):
        return False
    if _require_hsa_synthetic_row_compact():
        runtime = _get_hsa_block_sparse_runtime(schedule, q, k) if runtime is None else runtime
        _ensure_hsa_synthetic_grid_metadata(schedule, runtime)
        if not _hsa_synthetic_forward_has_row_compact_plan(runtime.forward_synthetic_grid):
            return False
        if not _hsa_synthetic_forward_segments_within_runtime_limit(runtime.forward_synthetic_grid):
            return False
    return True


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _wrap_u32_to_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


def _build_hsa_qgroup_union_entry(
    qgroup_idx: int,
    *,
    qgroup_packed_q: list[int] | torch.Tensor,
    qgroup_length: list[int] | torch.Tensor,
    split_entries: list[dict[str, object]],
    split_indices_by_qgroup: dict[int, list[int]],
) -> dict[str, object]:
    split_ids = sorted(split_indices_by_qgroup.get(qgroup_idx, []), key=lambda idx: int(split_entries[idx]["slot"]))
    packed_q = int(qgroup_packed_q[qgroup_idx])
    union_k_length = sum(int(split_entries[split_idx]["k_length"]) for split_idx in split_ids)
    union_words_per_row = (union_k_length + 31) // 32
    union_mask_words = [0] * (packed_q * union_words_per_row)
    union_padded_k_rows: list[int] = []
    union_allowed_pairs = 0
    col_offset = 0

    for split_idx in split_ids:
        split_entry = split_entries[split_idx]
        split_k_length = int(split_entry["k_length"])
        split_mask_words = split_entry["mask_words"]
        split_words_per_row = 0 if packed_q <= 0 else len(split_mask_words) // packed_q
        union_padded_k_rows.extend(int(row_idx) for row_idx in split_entry["padded_k_rows"][:split_k_length])
        for q_slot in range(packed_q):
            row_base = q_slot * split_words_per_row
            for word_idx in range(split_words_per_row):
                word = int(split_mask_words[row_base + word_idx]) & 0xFFFFFFFF
                while word:
                    bit = word & -word
                    bit_idx = bit.bit_length() - 1
                    local_k = word_idx * 32 + bit_idx
                    if local_k < split_k_length:
                        global_k = col_offset + local_k
                        union_word_idx = global_k // 32
                        union_bit_idx = global_k % 32
                        union_mask_words[q_slot * union_words_per_row + union_word_idx] |= 1 << union_bit_idx
                        union_allowed_pairs += 1
                    word ^= bit
        col_offset += split_k_length

    q_count = int(qgroup_length[qgroup_idx])
    dense = union_allowed_pairs == q_count * union_k_length
    return {
        "k_length": union_k_length,
        "words_per_row": union_words_per_row,
        "mask_words": union_mask_words,
        "padded_k_rows": union_padded_k_rows,
        "allowed_pairs": union_allowed_pairs,
        "dense": dense,
    }


def _build_hsa_qgroup_compact_union_entry(
    qgroup_idx: int,
    *,
    qgroup_packed_q: list[int] | torch.Tensor,
    qgroup_length: list[int] | torch.Tensor,
    split_entries: list[dict[str, object]],
    split_indices_by_qgroup: dict[int, list[int]],
) -> dict[str, object]:
    split_ids = sorted(split_indices_by_qgroup.get(qgroup_idx, []), key=lambda idx: int(split_entries[idx]["slot"]))
    packed_q = int(qgroup_packed_q[qgroup_idx])
    union_rows: list[int] = []
    row_cols: list[list[int]] = [[] for _ in range(packed_q)]
    allowed_pairs = 0
    col_offset = 0

    for split_idx in split_ids:
        split_entry = split_entries[split_idx]
        split_k_length = int(split_entry["k_length"])
        split_rows = [int(row_idx) for row_idx in split_entry["padded_k_rows"][:split_k_length]]
        split_mask_words = split_entry["mask_words"]
        split_words_per_row = 0 if packed_q <= 0 else len(split_mask_words) // packed_q

        split_active_rows: list[int] = []
        split_slot_to_compact: dict[int, int] = {}
        for slot_idx, row_idx in enumerate(split_rows):
            if row_idx < 0:
                continue
            split_slot_to_compact[slot_idx] = len(split_active_rows)
            split_active_rows.append(row_idx)

        for q_slot in range(packed_q):
            row_base = q_slot * split_words_per_row
            for word_idx in range(split_words_per_row):
                word = int(split_mask_words[row_base + word_idx]) & 0xFFFFFFFF
                while word:
                    bit = word & -word
                    bit_idx = bit.bit_length() - 1
                    local_k = word_idx * 32 + bit_idx
                    if local_k < split_k_length:
                        compact_local_k = split_slot_to_compact.get(local_k)
                        if compact_local_k is not None:
                            row_cols[q_slot].append(col_offset + compact_local_k)
                            allowed_pairs += 1
                    word ^= bit
        union_rows.extend(split_active_rows)
        col_offset += len(split_active_rows)

    union_k_length = len(union_rows)
    words_per_row = (union_k_length + 31) // 32
    mask_words = [0] * (packed_q * words_per_row)
    for q_slot, cols in enumerate(row_cols):
        for col_idx in cols:
            word_idx = col_idx // 32
            bit_idx = col_idx % 32
            mask_words[q_slot * words_per_row + word_idx] |= 1 << bit_idx

    q_count = int(qgroup_length[qgroup_idx])
    dense = union_k_length > 0 and allowed_pairs == q_count * union_k_length
    return {
        "k_length": union_k_length,
        "words_per_row": words_per_row,
        "mask_words": mask_words,
        "union_rows": union_rows,
        "allowed_pairs": allowed_pairs,
        "dense": dense,
    }


def _build_hsa_qgroup_union_support_plan(
    *,
    device,
    qgroup_row_ptr: list[int],
    qgroup_length: list[int],
    qgroup_packed_q: list[int],
    split_entries: list[dict[str, object]],
    split_indices_by_qgroup: dict[int, list[int]],
) -> dict[str, object]:
    qgroup_row_range = [
        (qgroup_row_ptr[qgroup_idx], qgroup_row_ptr[qgroup_idx + 1])
        for qgroup_idx in range(len(qgroup_length))
    ]
    qgroup_union_k_row_range: list[tuple[int, int]] = []
    qgroup_mask_word_range: list[tuple[int, int]] = []
    qgroup_union_k_row_idx: list[int] = []
    qgroup_mask_words: list[int] = []
    qgroup_union_k_length: list[int] = []
    qgroup_words_per_row: list[int] = []
    qgroup_allowed_pairs: list[int] = []
    qgroup_fill: list[float] = []

    for qgroup_idx in range(len(qgroup_length)):
        union_entry = _build_hsa_qgroup_compact_union_entry(
            qgroup_idx,
            qgroup_packed_q=qgroup_packed_q,
            qgroup_length=qgroup_length,
            split_entries=split_entries,
            split_indices_by_qgroup=split_indices_by_qgroup,
        )
        union_start = len(qgroup_union_k_row_idx)
        qgroup_union_k_row_idx.extend(int(row_idx) for row_idx in union_entry["union_rows"])
        qgroup_union_k_row_range.append((union_start, len(qgroup_union_k_row_idx)))

        mask_start = len(qgroup_mask_words)
        qgroup_mask_words.extend(_wrap_u32_to_i32(int(word)) for word in union_entry["mask_words"])
        qgroup_mask_word_range.append((mask_start, len(qgroup_mask_words)))

        union_k_length = int(union_entry["k_length"])
        qgroup_union_k_length.append(union_k_length)
        qgroup_words_per_row.append(int(union_entry["words_per_row"]))
        allowed_pairs = int(union_entry["allowed_pairs"])
        qgroup_allowed_pairs.append(allowed_pairs)
        packed_q = int(qgroup_packed_q[qgroup_idx])
        qgroup_fill.append(allowed_pairs / max(packed_q * union_k_length, 1) if union_k_length > 0 else 0.0)

    return {
        "qgroup_row_range": qgroup_row_range,
        "qgroup_union_k_row_range": qgroup_union_k_row_range,
        "qgroup_mask_word_range": qgroup_mask_word_range,
        "qgroup_union_k_row_idx": torch.tensor(qgroup_union_k_row_idx, dtype=torch.int32, device=device),
        "qgroup_union_k_length": torch.tensor(qgroup_union_k_length, dtype=torch.int32, device=device),
        "qgroup_mask_words": torch.tensor(qgroup_mask_words, dtype=torch.int32, device=device),
        "qgroup_words_per_row": torch.tensor(qgroup_words_per_row, dtype=torch.int32, device=device),
        "qgroup_allowed_pairs": torch.tensor(qgroup_allowed_pairs, dtype=torch.int32, device=device),
        "qgroup_fill": torch.tensor(qgroup_fill, dtype=torch.float32, device=device),
    }


def _finalize_hsa_synthetic_grid_metadata(
    *,
    device,
    logical_block_q: int,
    logical_block_k: int,
    physical_block_q: int,
    physical_block_k: int,
    tile_batch_idx: list[int],
    tile_q_block_idx: list[int],
    tile_k_block_idx: list[int],
    tile_q_row_ptr: list[int],
    tile_q_rows: list[int],
    tile_k_row_ptr: list[int],
    tile_k_rows: list[int],
    tile_logical_pair_row_ptr: list[int],
    tile_logical_pairs: list[list[int]],
    compact_mask_row_ptr: list[int],
    compact_mask_col_idx: list[int],
) -> HSASyntheticGridMetadata:
    num_tiles = len(tile_batch_idx)
    tile_allowed_pairs: list[int] = []
    tile_packed_q: list[int] = []
    tile_packed_k: list[int] = []
    tile_dense: list[bool] = []
    tile_fill: list[float] = []
    tile_q_length: list[int] = []
    tile_k_length: list[int] = []
    bucket_map: dict[tuple[int, int, bool], list[int]] = {}

    for tile_idx in range(num_tiles):
        q_count = tile_q_row_ptr[tile_idx + 1] - tile_q_row_ptr[tile_idx]
        k_count = tile_k_row_ptr[tile_idx + 1] - tile_k_row_ptr[tile_idx]
        allowed_pairs = compact_mask_row_ptr[tile_q_row_ptr[tile_idx + 1]] - compact_mask_row_ptr[tile_q_row_ptr[tile_idx]]
        packed_q = _align_up(q_count, logical_block_q)
        packed_k = _align_up(k_count, logical_block_k)
        is_dense = q_count > 0 and k_count > 0 and allowed_pairs == q_count * k_count
        tile_allowed_pairs.append(allowed_pairs)
        tile_packed_q.append(packed_q)
        tile_packed_k.append(packed_k)
        tile_dense.append(is_dense)
        tile_fill.append(allowed_pairs / max(packed_q * packed_k, 1))
        tile_q_length.append(q_count)
        tile_k_length.append(k_count)
        bucket_map.setdefault((packed_q, packed_k, is_dense), []).append(tile_idx)

    bucket_row_ptr = [0]
    bucket_tile_idx: list[int] = []
    bucket_packed_q: list[int] = []
    bucket_packed_k: list[int] = []
    bucket_dense: list[int] = []
    for (packed_q, packed_k, is_dense), tile_ids in sorted(bucket_map.items()):
        bucket_packed_q.append(packed_q)
        bucket_packed_k.append(packed_k)
        bucket_dense.append(1 if is_dense else 0)
        bucket_tile_idx.extend(tile_ids)
        bucket_row_ptr.append(len(bucket_tile_idx))

    return HSASyntheticGridMetadata(
        logical_block_q=logical_block_q,
        logical_block_k=logical_block_k,
        physical_block_q=physical_block_q,
        physical_block_k=physical_block_k,
        tile_batch_idx=torch.tensor(tile_batch_idx, dtype=torch.int32, device=device),
        tile_q_block_idx=torch.tensor(tile_q_block_idx, dtype=torch.int32, device=device),
        tile_k_block_idx=torch.tensor(tile_k_block_idx, dtype=torch.int32, device=device),
        tile_q_subgroup_idx=torch.zeros(num_tiles, dtype=torch.int32, device=device),
        tile_q_row_ptr=torch.tensor(tile_q_row_ptr, dtype=torch.int32, device=device),
        tile_q_rows=torch.tensor(tile_q_rows, dtype=torch.int32, device=device),
        tile_k_row_ptr=torch.tensor(tile_k_row_ptr, dtype=torch.int32, device=device),
        tile_k_rows=torch.tensor(tile_k_rows, dtype=torch.int32, device=device),
        tile_logical_pair_row_ptr=torch.tensor(tile_logical_pair_row_ptr, dtype=torch.int32, device=device),
        tile_logical_pairs=torch.tensor(tile_logical_pairs, dtype=torch.int32, device=device),
        compact_mask_row_ptr=torch.tensor(compact_mask_row_ptr, dtype=torch.int32, device=device),
        compact_mask_col_idx=torch.tensor(compact_mask_col_idx, dtype=torch.int32, device=device),
        tile_allowed_pairs=torch.tensor(tile_allowed_pairs, dtype=torch.int32, device=device),
        tile_packed_q=torch.tensor(tile_packed_q, dtype=torch.int32, device=device),
        tile_packed_k=torch.tensor(tile_packed_k, dtype=torch.int32, device=device),
        tile_dense=torch.tensor(tile_dense, dtype=torch.bool, device=device),
        tile_fill=torch.tensor(tile_fill, dtype=torch.float32, device=device),
        bucket_row_ptr=torch.tensor(bucket_row_ptr, dtype=torch.int32, device=device),
        bucket_tile_idx=torch.tensor(bucket_tile_idx, dtype=torch.int32, device=device),
        bucket_packed_q=torch.tensor(bucket_packed_q, dtype=torch.int32, device=device),
        bucket_packed_k=torch.tensor(bucket_packed_k, dtype=torch.int32, device=device),
        bucket_dense=torch.tensor(bucket_dense, dtype=torch.bool, device=device),
        max_packed_k=None,
        tile_q_length=torch.tensor(tile_q_length, dtype=torch.int32, device=device),
        tile_k_length=torch.tensor(tile_k_length, dtype=torch.int32, device=device),
        bucket_q_src_row_idx=None,
    )


def _build_hsa_forward_synthetic_grid_metadata(
    schedule: HSASchedule,
    runtime: HSABlockSparseRuntime,
    *,
    logical_block_q: int,
    logical_block_k: int,
    max_packed_k: int,
) -> HSASyntheticGridMetadata:
    sparse_tensors = runtime.forward_sparse
    tile_masks = runtime.forward_tile_masks
    q_block_size, k_block_size = sparse_tensors.block_size
    if q_block_size % logical_block_q != 0 or k_block_size % logical_block_k != 0:
        raise ValueError("Synthetic grid requires logical block sizes that evenly divide the physical sparse blocks")
    if max_packed_k <= 0 or max_packed_k > k_block_size:
        raise ValueError("Synthetic packed K width must be between 1 and the physical K block size")
    if max_packed_k % logical_block_k != 0:
        raise ValueError("Synthetic packed K width must be a multiple of the synthetic logical K block size")

    device = sparse_tensors.mask_block_cnt.device
    seqlen = schedule.seqlen
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = None if sparse_tensors.full_block_cnt is None else sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = None if sparse_tensors.full_block_idx is None else sparse_tensors.full_block_idx.detach().cpu()
    block_id_table = tile_masks.block_id_table.detach().cpu()
    tile_kind = tile_masks.tile_kind.detach().cpu()
    affine_base = tile_masks.affine_base.detach().cpu()
    row_prefix_row_ptr = tile_masks.row_prefix_row_ptr.detach().cpu()
    row_prefix_len = tile_masks.row_prefix_len.detach().cpu()
    bitmap_word_row_ptr = tile_masks.bitmap_word_row_ptr.detach().cpu()
    bitmap_words = tile_masks.bitmap_words.detach().cpu()
    words_per_row = tile_masks.words_per_row
    num_batches = int(mask_block_cnt.shape[0])
    num_q_blocks = int(mask_block_cnt.shape[2])

    tile_batch_idx: list[int] = []
    tile_q_block_idx: list[int] = []
    tile_q_subgroup_idx: list[int] = []
    tile_k_block_idx: list[int] = []
    tile_q_row_ptr = [0]
    tile_k_row_ptr = [0]
    tile_logical_pair_row_ptr = [0]
    tile_q_rows: list[int] = []
    tile_k_rows: list[int] = []
    tile_logical_pairs: list[list[int]] = []
    compact_mask_row_ptr = [0]
    compact_mask_col_idx: list[int] = []
    tile_allowed_pairs: list[int] = []
    tile_packed_q: list[int] = []
    tile_packed_k: list[int] = []
    tile_dense: list[bool] = []
    tile_fill: list[float] = []
    tile_q_length: list[int] = []
    tile_k_length: list[int] = []

    qgroup_row_ptr = [0]
    qgroup_rows: list[int] = []
    qgroup_length: list[int] = []
    qgroup_packed_q: list[int] = []
    qgroup_num_splits: list[int] = []

    split_entries: list[dict[str, object]] = []
    split_indices_by_qgroup: dict[int, list[int]] = {}
    max_blocks_per_split = max(1, max_packed_k // logical_block_k)
    max_direct_segments = _get_hsa_synthetic_max_direct_segments()
    sparse_parse_fwd = _use_hsa_synthetic_sparse_parse_fwd()
    forward_plan_only = _use_hsa_synthetic_forward_plan_only()
    pack_k_bin = os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_PACKED_K_BIN", "0") == "1"
    bitmap_block_support_cache: dict[int, list[list[int]]] = {}

    def _bin_packed_k(packed_k_native: int) -> int:
        if not pack_k_bin or logical_block_q != 2 or logical_block_k != 2:
            return packed_k_native
        if packed_k_native <= 2:
            return 2
        if packed_k_native <= 4:
            return 4
        if packed_k_native <= 8:
            return 8
        if packed_k_native <= 16:
            return 16
        return min(max_packed_k, _align_up(packed_k_native, 16))

    def _fill_score(blocks: list[dict[str, object]], packed_q: int) -> float:
        if not blocks or packed_q <= 0:
            return 0.0
        allowed_pairs = sum(int(block["allowed_pairs"]) for block in blocks)
        packed_k = logical_block_k * len(blocks)
        return allowed_pairs / max(packed_q * packed_k, 1)

    def _candidate_pack_key(
        blocks: list[dict[str, object]],
        packed_q: int,
    ) -> tuple[float, int, int, int]:
        allowed_pairs = sum(int(block["allowed_pairs"]) for block in blocks)
        packed_k = logical_block_k * len(blocks)
        min_logical_k_idx = min(int(block["logical_k_idx"]) for block in blocks)
        return (
            allowed_pairs / max(packed_q * packed_k, 1),
            allowed_pairs,
            -packed_k,
            -min_logical_k_idx,
        )

    def _greedy_partition_logical_blocks(
        logical_blocks: dict[int, dict[str, object]],
        packed_q: int,
    ) -> list[list[dict[str, object]]]:
        def _pack_key(
            *,
            allowed_pairs: int,
            num_blocks: int,
            min_logical_k_idx: int,
        ) -> tuple[float, int, int, int]:
            packed_k = logical_block_k * num_blocks
            return (
                allowed_pairs / max(packed_q * packed_k, 1),
                allowed_pairs,
                -packed_k,
                -min_logical_k_idx,
            )

        remaining = sorted(logical_blocks.values(), key=lambda block: int(block["logical_k_idx"]))
        if _use_hsa_synthetic_fast_partition():
            ranked = sorted(
                remaining,
                key=lambda block: (
                    int(block["allowed_pairs"]),
                    -int(block["logical_k_idx"]),
                ),
                reverse=True,
            )
            splits: list[list[dict[str, object]]] = []
            for split_start in range(0, len(ranked), max_blocks_per_split):
                current = ranked[split_start : split_start + max_blocks_per_split]
                current.sort(key=lambda block: int(block["logical_k_idx"]))
                splits.append(current)
            return splits
        splits: list[list[dict[str, object]]] = []
        while remaining:
            seed_idx = max(
                range(len(remaining)),
                key=lambda idx: _pack_key(
                    allowed_pairs=int(remaining[idx]["allowed_pairs"]),
                    num_blocks=1,
                    min_logical_k_idx=int(remaining[idx]["logical_k_idx"]),
                ),
            )
            current = [remaining.pop(seed_idx)]
            current_allowed_pairs = int(current[0]["allowed_pairs"])
            current_num_blocks = 1
            current_min_logical_k_idx = int(current[0]["logical_k_idx"])
            current_key = _pack_key(
                allowed_pairs=current_allowed_pairs,
                num_blocks=current_num_blocks,
                min_logical_k_idx=current_min_logical_k_idx,
            )
            while remaining and len(current) < max_blocks_per_split:
                best_add_idx: int | None = None
                best_add_key = current_key
                for candidate_idx, candidate in enumerate(remaining):
                    candidate_key = _pack_key(
                        allowed_pairs=current_allowed_pairs + int(candidate["allowed_pairs"]),
                        num_blocks=current_num_blocks + 1,
                        min_logical_k_idx=min(
                            current_min_logical_k_idx,
                            int(candidate["logical_k_idx"]),
                        ),
                    )
                    if candidate_key > best_add_key:
                        best_add_idx = candidate_idx
                        best_add_key = candidate_key
                if best_add_idx is None:
                    break
                next_block = remaining.pop(best_add_idx)
                current.append(next_block)
                current_allowed_pairs += int(next_block["allowed_pairs"])
                current_num_blocks += 1
                current_min_logical_k_idx = min(current_min_logical_k_idx, int(next_block["logical_k_idx"]))
                current_key = best_add_key
            current.sort(key=lambda block: int(block["logical_k_idx"]))
            splits.append(current)
        return splits

    def _accumulate_forward_qgroup_logical_block(
        logical_blocks: dict[int, dict[str, object]],
        *,
        q_local: int,
        global_k_row: int,
    ) -> None:
        logical_k_idx = global_k_row // logical_block_k
        local_k = global_k_row % logical_block_k
        block = logical_blocks.setdefault(
            logical_k_idx,
            {
                "logical_k_idx": logical_k_idx,
                "allowed_pairs": 0,
                "active_local_cols": set(),
                "cols_by_q_local": {},
            },
        )
        block["allowed_pairs"] = int(block["allowed_pairs"]) + 1
        active_local_cols = block["active_local_cols"]
        assert isinstance(active_local_cols, set)
        active_local_cols.add(local_k)
        cols_by_q_local = block["cols_by_q_local"]
        assert isinstance(cols_by_q_local, dict)
        cols_for_q = cols_by_q_local.setdefault(q_local, set())
        assert isinstance(cols_for_q, set)
        cols_for_q.add(local_k)

    def _append_forward_qgroup(
        batch_idx: int,
        q_block_idx: int,
        q_subgroup_idx: int,
        q_rows_local: list[int],
        logical_blocks: dict[int, dict[str, object]],
    ) -> None:
        if not q_rows_local or not logical_blocks:
            return

        q_count = len(q_rows_local)
        packed_q = _align_up(q_count, logical_block_q)
        q_slot_for_local = {q_local: q_slot for q_slot, q_local in enumerate(q_rows_local)}
        for block in logical_blocks.values():
            cols_by_q_local = block.pop("cols_by_q_local")
            assert isinstance(cols_by_q_local, dict)
            cols_by_qslot = {}
            for q_local, cols in cols_by_q_local.items():
                q_slot = q_slot_for_local[int(q_local)]
                cols_by_qslot[q_slot] = cols
            block["cols_by_qslot"] = cols_by_qslot

        qgroup_idx = len(qgroup_length)
        batch_base = batch_idx * seqlen
        q_base = q_block_idx * q_block_size
        qgroup_rows.extend(batch_base + q_base + q_local for q_local in q_rows_local)
        qgroup_row_ptr.append(len(qgroup_rows))
        qgroup_length.append(q_count)
        qgroup_packed_q.append(packed_q)

        splits = _greedy_partition_logical_blocks(logical_blocks, packed_q)
        qgroup_num_splits.append(len(splits))

        for split_slot, split_blocks in enumerate(splits):
            packed_k_native = logical_block_k * len(split_blocks)
            packed_k = _bin_packed_k(packed_k_native)
            split_packed_cols = [[] for _ in range(q_count)]
            padded_k_rows: list[int] = []
            active_rows_dense = True
            active_k_rows_sorted: list[int] = []
            split_global_rows = [[] for _ in range(q_count)] if not forward_plan_only else None
            logical_pairs_local: list[tuple[int, int]] = [] if not forward_plan_only else []

            for block_pos, block in enumerate(split_blocks):
                logical_k_idx = int(block["logical_k_idx"])
                cols_by_qslot = block["cols_by_qslot"]
                assert isinstance(cols_by_qslot, dict)
                active_local_cols = block["active_local_cols"]
                assert isinstance(active_local_cols, set)
                block_base = logical_k_idx * logical_block_k
                packed_col_base = block_pos * logical_block_k
                for local_k in range(logical_block_k):
                    global_row = block_base + local_k
                    if global_row < seqlen and local_k in active_local_cols:
                        padded_k_rows.append(batch_base + global_row)
                        if not forward_plan_only:
                            active_k_rows_sorted.append(global_row)
                    else:
                        padded_k_rows.append(-1)
                        active_rows_dense = False
                for q_slot in sorted(cols_by_qslot):
                    cols = cols_by_qslot[q_slot]
                    assert isinstance(cols, set)
                    if cols and not forward_plan_only:
                        logical_pairs_local.append((q_slot // logical_block_q, block_pos))
                    for local_k in sorted(cols):
                        split_packed_cols[q_slot].append(packed_col_base + local_k)
                        if split_global_rows is not None:
                            split_global_rows[q_slot].append(block_base + local_k)

            allowed_pairs = sum(len(cols) for cols in split_packed_cols)
            split_fill = allowed_pairs / max(packed_q * packed_k, 1)
            dense = allowed_pairs == q_count * packed_k and active_rows_dense
            words_per_row_split = (packed_k + 31) // 32
            mask_words = [0] * (packed_q * words_per_row_split)
            for q_slot, cols in enumerate(split_packed_cols):
                for k_slot in cols:
                    word_idx = k_slot // 32
                    bit_idx = k_slot % 32
                    mask_words[q_slot * words_per_row_split + word_idx] |= 1 << bit_idx

            if packed_k > packed_k_native:
                padded_k_rows.extend([-1] * (packed_k - packed_k_native))
                active_rows_dense = False

            if not forward_plan_only:
                k_compact = {k_row: idx for idx, k_row in enumerate(active_k_rows_sorted)}
                compact_cols_per_row: list[list[int]] = []
                assert split_global_rows is not None
                for q_slot in range(q_count):
                    compact_cols_per_row.append([k_compact[k_row] for k_row in split_global_rows[q_slot]])

                tile_batch_idx.append(batch_idx)
                tile_q_block_idx.append(q_block_idx)
                tile_q_subgroup_idx.append(q_subgroup_idx)
                tile_k_block_idx.append(-1)
                tile_q_rows.extend(batch_base + q_base + q_local for q_local in q_rows_local)
                tile_k_rows.extend(batch_base + k_row for k_row in active_k_rows_sorted)
                tile_logical_pairs.extend([[q_sub, k_sub] for q_sub, k_sub in logical_pairs_local])
                tile_q_row_ptr.append(len(tile_q_rows))
                tile_k_row_ptr.append(len(tile_k_rows))
                tile_logical_pair_row_ptr.append(len(tile_logical_pairs))
                for compact_cols in compact_cols_per_row:
                    compact_mask_col_idx.extend(compact_cols)
                    compact_mask_row_ptr.append(len(compact_mask_col_idx))
                tile_allowed_pairs.append(allowed_pairs)
                tile_packed_q.append(packed_q)
                tile_packed_k.append(packed_k)
                tile_dense.append(dense)
                tile_fill.append(split_fill)
                tile_q_length.append(q_count)
                tile_k_length.append(len(active_k_rows_sorted))

            split_entry = {
                "qgroup_idx": qgroup_idx,
                "slot": split_slot,
                "packed_k": packed_k,
                "dense": dense,
                "padded_k_rows": padded_k_rows,
                "q_length": q_count,
                "k_length": packed_k_native,
                "mask_words": mask_words,
                "split_fill": split_fill,
            }
            split_idx = len(split_entries)
            split_entries.append(split_entry)
            split_indices_by_qgroup.setdefault(qgroup_idx, []).append(split_idx)

    for batch_idx in range(num_batches):
        for q_block_idx in range(num_q_blocks):
            q_start = q_block_idx * q_block_size
            q_len = min(q_block_size, seqlen - q_start)
            if q_len <= 0:
                continue
            num_q_subgroups = (q_len + logical_block_q - 1) // logical_block_q
            for q_subgroup_idx in range(num_q_subgroups):
                subgroup_q_start = q_subgroup_idx * logical_block_q
                subgroup_q_end = min(q_len, subgroup_q_start + logical_block_q)
                q_rows_local: set[int] = set()
                logical_blocks: dict[int, dict[str, object]] = {}

                full_cnt = 0 if full_block_cnt is None else int(full_block_cnt[batch_idx, 0, q_block_idx].item())
                for offset in range(full_cnt):
                    k_block_idx = int(full_block_idx[batch_idx, 0, q_block_idx, offset].item())
                    k_start = k_block_idx * k_block_size
                    k_len = min(k_block_size, seqlen - k_start)
                    if k_len <= 0:
                        continue
                    for q_local in range(subgroup_q_start, subgroup_q_end):
                        q_rows_local.add(q_local)
                        for global_k_row in range(k_start, k_start + k_len):
                            _accumulate_forward_qgroup_logical_block(
                                logical_blocks,
                                q_local=q_local,
                                global_k_row=global_k_row,
                            )

                partial_cnt = int(mask_block_cnt[batch_idx, 0, q_block_idx].item())
                for offset in range(partial_cnt):
                    k_block_idx = int(mask_block_idx[batch_idx, 0, q_block_idx, offset].item())
                    k_start = k_block_idx * k_block_size
                    k_len = min(k_block_size, seqlen - k_start)
                    if k_len <= 0:
                        continue
                    block_id = int(block_id_table[batch_idx, q_block_idx, k_block_idx].item())
                    kind = int(tile_kind[block_id].item())

                    if kind in (_HSA_FWD_TILE_AFFINE_PREFIX, _HSA_FWD_TILE_ROW_PREFIX):
                        for q_local in range(subgroup_q_start, subgroup_q_end):
                            if kind == _HSA_FWD_TILE_AFFINE_PREFIX:
                                prefix = max(0, min(k_len, int(affine_base[block_id].item()) + q_local))
                            else:
                                prefix_start = int(row_prefix_row_ptr[block_id].item())
                                prefix = max(0, min(k_len, int(row_prefix_len[prefix_start + q_local].item())))
                            if prefix <= 0:
                                continue
                            q_rows_local.add(q_local)
                            for global_k_row in range(k_start, k_start + prefix):
                                _accumulate_forward_qgroup_logical_block(
                                    logical_blocks,
                                    q_local=q_local,
                                    global_k_row=global_k_row,
                                )
                    else:
                        block_support_rows = bitmap_block_support_cache.get(block_id)
                        if block_support_rows is None:
                            word_start = int(bitmap_word_row_ptr[block_id].item())
                            block_support_rows = [[] for _ in range(q_block_size)]
                            for bitmap_q_local in range(q_block_size):
                                row_support_rows: list[int] = []
                                for word_idx in range(words_per_row):
                                    word = (
                                        int(bitmap_words[word_start + bitmap_q_local * words_per_row + word_idx].item())
                                        & 0xFFFFFFFF
                                    )
                                    while word:
                                        bit = word & -word
                                        bit_idx = bit.bit_length() - 1
                                        k_local = word_idx * 32 + bit_idx
                                        if k_local < k_len:
                                            row_support_rows.append(k_start + k_local)
                                        word ^= bit
                                block_support_rows[bitmap_q_local] = row_support_rows
                            bitmap_block_support_cache[block_id] = block_support_rows
                        for q_local in range(subgroup_q_start, subgroup_q_end):
                            support_rows = block_support_rows[q_local]
                            if not support_rows:
                                continue
                            q_rows_local.add(q_local)
                            for global_k_row in support_rows:
                                _accumulate_forward_qgroup_logical_block(
                                    logical_blocks,
                                    q_local=q_local,
                                    global_k_row=global_k_row,
                                )
                _append_forward_qgroup(
                    batch_idx,
                    q_block_idx,
                    q_subgroup_idx,
                    sorted(q_rows_local),
                    logical_blocks,
                )

    qgroup_union_support_plan = _build_hsa_qgroup_union_support_plan(
        device=device,
        qgroup_row_ptr=qgroup_row_ptr,
        qgroup_length=qgroup_length,
        qgroup_packed_q=qgroup_packed_q,
        split_entries=split_entries,
        split_indices_by_qgroup=split_indices_by_qgroup,
    )

    row_compact_cap = 16

    def _build_qgroup_merged_entry(
        qgroup_idx: int,
        packed_q: int,
        q_count: int,
        split_ids: list[int],
        *,
        split_entries_ref: list[dict[str, object]],
    ) -> dict[str, object]:
        split_ids = sorted(split_ids, key=lambda idx: int(split_entries_ref[idx]["slot"]))
        merged_k_length = sum(int(split_entries_ref[split_idx]["k_length"]) for split_idx in split_ids)
        merged_words_per_row = (merged_k_length + 31) // 32
        merged_mask_words = [0] * (packed_q * merged_words_per_row)
        merged_padded_k_rows: list[int] = []
        merged_allowed_pairs = 0
        merged_dense = True
        col_offset = 0

        for split_idx in split_ids:
            split_entry = split_entries_ref[split_idx]
            split_k_length = int(split_entry["k_length"])
            split_mask_words = split_entry["mask_words"]
            split_words_per_row = 0 if packed_q <= 0 else len(split_mask_words) // packed_q
            merged_padded_k_rows.extend(int(row_idx) for row_idx in split_entry["padded_k_rows"][:split_k_length])
            merged_dense = merged_dense and bool(split_entry["dense"])
            for q_slot in range(packed_q):
                row_base = q_slot * split_words_per_row
                for word_idx in range(split_words_per_row):
                    word = int(split_mask_words[row_base + word_idx]) & 0xFFFFFFFF
                    while word:
                        bit = word & -word
                        bit_idx = bit.bit_length() - 1
                        local_k = word_idx * 32 + bit_idx
                        if local_k < split_k_length:
                            global_k = col_offset + local_k
                            merged_word_idx = global_k // 32
                            merged_bit_idx = global_k % 32
                            merged_mask_words[q_slot * merged_words_per_row + merged_word_idx] |= 1 << merged_bit_idx
                            merged_allowed_pairs += 1
                        word ^= bit
            col_offset += split_k_length

        merged_dense = merged_dense and merged_allowed_pairs == q_count * merged_k_length
        return {
            "qgroup_idx": qgroup_idx,
            "split_ids": list(split_ids),
            "k_length": merged_k_length,
            "words_per_row": merged_words_per_row,
            "mask_words": merged_mask_words,
            "padded_k_rows": merged_padded_k_rows,
            "allowed_pairs": merged_allowed_pairs,
            "dense": merged_dense,
            "split_fill": merged_allowed_pairs / max(packed_q * merged_k_length, 1),
            "slot": min((int(split_entries_ref[split_idx]["slot"]) for split_idx in split_ids), default=0),
        }

    def _merge_qgroup_split_groups(
        qgroup_idx: int,
        packed_q: int,
        q_count: int,
        split_groups: list[list[int]],
        *,
        split_entries_ref: list[dict[str, object]],
        target_max_segments: int | None,
        merge_k_cap: int,
    ) -> list[list[int]] | None:
        if target_max_segments is not None and len(split_groups) <= target_max_segments:
            return [sorted(group, key=lambda idx: int(split_entries_ref[idx]["slot"])) for group in split_groups]
        working_groups = [list(group) for group in split_groups]
        while True:
            if target_max_segments is not None and len(working_groups) <= target_max_segments:
                break
            best_pair: tuple[int, int] | None = None
            best_key: tuple[float, int, int, int] | None = None
            for left_idx in range(len(working_groups)):
                for right_idx in range(left_idx + 1, len(working_groups)):
                    merged_ids = working_groups[left_idx] + working_groups[right_idx]
                    merged_entry = _build_qgroup_merged_entry(
                        qgroup_idx,
                        packed_q,
                        q_count,
                        merged_ids,
                        split_entries_ref=split_entries_ref,
                    )
                    merged_k_length = int(merged_entry["k_length"])
                    if merged_k_length > merge_k_cap:
                        continue
                    min_slot = min(int(split_entries_ref[split_idx]["slot"]) for split_idx in merged_ids)
                    candidate_key = (
                        float(merged_entry["split_fill"]),
                        int(merged_entry["allowed_pairs"]),
                        -merged_k_length,
                        -min_slot,
                    )
                    if best_key is None or candidate_key > best_key:
                        best_key = candidate_key
                        best_pair = (left_idx, right_idx)
            if best_pair is None:
                break
            left_idx, right_idx = best_pair
            merged_group = working_groups[left_idx] + working_groups[right_idx]
            working_groups[left_idx] = sorted(merged_group, key=lambda idx: int(split_entries_ref[idx]["slot"]))
            del working_groups[right_idx]
            working_groups.sort(key=lambda group: min(int(split_entries_ref[split_idx]["slot"]) for split_idx in group))
        if target_max_segments is not None and len(working_groups) > target_max_segments:
            return None
        return working_groups

    def _build_direct_sharded_split_entries(
        shard_k_cap: int,
    ) -> tuple[list[dict[str, object]], dict[int, list[int]]]:
        def _build_compact_direct_split_entry(
            split_entry: dict[str, object],
            *,
            qgroup_idx: int,
            packed_q: int,
            q_count: int,
            slot: int,
        ) -> dict[str, object] | None:
            split_k_length = int(split_entry["k_length"])
            split_rows = [int(row_idx) for row_idx in split_entry["padded_k_rows"][:split_k_length]]
            split_mask_words = split_entry["mask_words"]
            split_words_per_row = 0 if packed_q <= 0 else len(split_mask_words) // packed_q
            compact_rows: list[int] = []
            split_slot_to_compact: dict[int, int] = {}
            compact_cols_by_qslot: list[list[int]] = [[] for _ in range(packed_q)]
            compact_allowed_pairs = 0

            for split_slot_idx, row_idx in enumerate(split_rows):
                if row_idx < 0:
                    continue
                split_slot_to_compact[split_slot_idx] = len(compact_rows)
                compact_rows.append(row_idx)

            if not compact_rows:
                return None

            for q_slot in range(packed_q):
                row_base = q_slot * split_words_per_row
                for word_idx in range(split_words_per_row):
                    word = int(split_mask_words[row_base + word_idx]) & 0xFFFFFFFF
                    while word:
                        bit = word & -word
                        bit_idx = bit.bit_length() - 1
                        split_local_k = word_idx * 32 + bit_idx
                        if split_local_k < split_k_length:
                            compact_local_k = split_slot_to_compact.get(split_local_k)
                            if compact_local_k is not None:
                                compact_cols_by_qslot[q_slot].append(compact_local_k)
                                compact_allowed_pairs += 1
                        word ^= bit

            compact_k_length = len(compact_rows)
            compact_words_per_row = (compact_k_length + 31) // 32
            compact_mask_words = [0] * (packed_q * compact_words_per_row)
            for q_slot, compact_cols in enumerate(compact_cols_by_qslot):
                for compact_local_k in compact_cols:
                    compact_word_idx = compact_local_k // 32
                    compact_bit_idx = compact_local_k % 32
                    compact_mask_words[q_slot * compact_words_per_row + compact_word_idx] |= 1 << compact_bit_idx

            return {
                "qgroup_idx": qgroup_idx,
                "slot": slot,
                "packed_k": compact_k_length,
                "dense": compact_allowed_pairs == q_count * compact_k_length,
                "padded_k_rows": compact_rows,
                "q_length": q_count,
                "k_length": compact_k_length,
                "mask_words": compact_mask_words,
                "split_fill": compact_allowed_pairs / max(packed_q * compact_k_length, 1),
                "allowed_pairs": compact_allowed_pairs,
            }

        direct_split_entries: list[dict[str, object]] = []
        direct_split_indices_by_qgroup: dict[int, list[int]] = {}

        for qgroup_idx in range(len(qgroup_length)):
            packed_q = int(qgroup_packed_q[qgroup_idx])
            q_count = int(qgroup_length[qgroup_idx])
            next_slot = 0
            split_ids = sorted(split_indices_by_qgroup.get(qgroup_idx, []), key=lambda idx: int(split_entries[idx]["slot"]))
            for split_idx in split_ids:
                compact_split_entry = _build_compact_direct_split_entry(
                    split_entries[split_idx],
                    qgroup_idx=qgroup_idx,
                    packed_q=packed_q,
                    q_count=q_count,
                    slot=next_slot,
                )
                if compact_split_entry is None:
                    continue

                split_k_length = int(compact_split_entry["k_length"])
                split_mask_words = compact_split_entry["mask_words"]
                split_words_per_row = 0 if packed_q <= 0 else len(split_mask_words) // packed_q
                split_padded_k_rows = compact_split_entry["padded_k_rows"]
                for shard_start in range(0, split_k_length, shard_k_cap):
                    shard_end = min(split_k_length, shard_start + shard_k_cap)
                    shard_k_length = shard_end - shard_start
                    shard_words_per_row = (shard_k_length + 31) // 32
                    shard_mask_words = [0] * (packed_q * shard_words_per_row)
                    shard_padded_k_rows = [int(row_idx) for row_idx in split_padded_k_rows[shard_start:shard_end]]
                    shard_allowed_pairs = 0
                    for q_slot in range(packed_q):
                        row_base = q_slot * split_words_per_row
                        for local_k in range(shard_k_length):
                            split_col = shard_start + local_k
                            word_idx = split_col // 32
                            bit_idx = split_col % 32
                            split_word = int(split_mask_words[row_base + word_idx]) & 0xFFFFFFFF
                            if ((split_word >> bit_idx) & 1) == 0:
                                continue
                            shard_word_idx = local_k // 32
                            shard_bit_idx = local_k % 32
                            shard_mask_words[q_slot * shard_words_per_row + shard_word_idx] |= 1 << shard_bit_idx
                            shard_allowed_pairs += 1
                    shard_dense = shard_allowed_pairs == q_count * shard_k_length
                    direct_split_entries.append(
                        {
                            "qgroup_idx": qgroup_idx,
                            "slot": next_slot,
                            "packed_k": shard_k_length,
                            "dense": shard_dense,
                            "padded_k_rows": shard_padded_k_rows,
                            "q_length": q_count,
                            "k_length": shard_k_length,
                            "mask_words": shard_mask_words,
                            "split_fill": shard_allowed_pairs / max(packed_q * shard_k_length, 1),
                            "allowed_pairs": shard_allowed_pairs,
                        }
                    )
                    direct_split_indices_by_qgroup.setdefault(qgroup_idx, []).append(len(direct_split_entries) - 1)
                    next_slot += 1

        return direct_split_entries, direct_split_indices_by_qgroup

    def _build_row_compact_plan_from_bucket_lists(
        *,
        bucket_packed_q: list[int],
        bucket_packed_k: list[int],
        bucket_dense: list[bool],
        bucket_k_row_range: list[tuple[int, int]],
        bucket_data_range: list[tuple[int, int]],
        bucket_mask_word_range: list[tuple[int, int]],
        bucket_q_length: list[int],
        bucket_k_length: list[int],
        bucket_k_row_idx: list[int],
        bucket_mask_words: list[int],
    ) -> dict[str, object] | None:
        row_bucket_row_k_range: list[tuple[int, int]] = []
        row_bucket_row_k_to_union_range: list[tuple[int, int]] = []
        row_bucket_union_to_row_range: list[tuple[int, int]] = []
        row_bucket_row_k_length_range: list[tuple[int, int]] = []
        row_bucket_unique_key_range: list[tuple[int, int]] = []
        row_bucket_unique_key_occurrence_range: list[tuple[int, int]] = []
        row_bucket_unique_key_occurrence_ptr_range: list[tuple[int, int]] = []
        row_bucket_row_k_cap: list[int] = []
        row_bucket_max_unique_key_occurrences: list[int] = []
        row_bucket_row_k_row_idx: list[int] = []
        row_bucket_row_k_to_union_idx: list[int] = []
        row_bucket_union_to_row_slot: list[int] = []
        row_bucket_row_k_length: list[int] = []
        row_bucket_unique_key_row_idx: list[int] = []
        row_bucket_unique_key_member_idx: list[int] = []
        row_bucket_unique_key_union_idx: list[int] = []
        row_bucket_unique_key_occurrence_row_ptr: list[int] = []
        row_bucket_avg_row_k: list[float] = []
        row_bucket_max_row_k: list[int] = []

        for bucket_idx in range(len(bucket_packed_q)):
            packed_q = int(bucket_packed_q[bucket_idx])
            packed_k = int(bucket_packed_k[bucket_idx])
            if packed_q != 2 or packed_k > row_compact_cap:
                return None

            bucket_size_value = bucket_data_range[bucket_idx][1] - bucket_data_range[bucket_idx][0]
            words_per_row = (packed_k + 31) // 32
            q_length_start, q_length_end = bucket_data_range[bucket_idx]
            k_length_start, k_length_end = bucket_data_range[bucket_idx]
            q_lengths_bucket = bucket_q_length[q_length_start:q_length_end]
            k_lengths_bucket = bucket_k_length[k_length_start:k_length_end]
            k_row_start, _ = bucket_k_row_range[bucket_idx]
            mask_word_start, _ = bucket_mask_word_range[bucket_idx]
            dense_bucket = bool(bucket_dense[bucket_idx])

            per_bucket_rows: list[list[int]] = []
            per_bucket_to_union: list[list[int]] = []
            per_bucket_union_rows: list[list[int]] = []
            per_bucket_lengths: list[int] = []
            bucket_max_row_k = 0

            for member_idx in range(bucket_size_value):
                q_count = int(q_lengths_bucket[member_idx])
                k_length = int(k_lengths_bucket[member_idx])
                k_member_start = k_row_start + member_idx * packed_k
                k_rows_member = [int(row_idx) for row_idx in bucket_k_row_idx[k_member_start:k_member_start + k_length]]
                per_bucket_union_rows.append(k_rows_member)
                mask_member_start = mask_word_start + member_idx * packed_q * words_per_row

                for q_slot in range(packed_q):
                    row_k_rows: list[int] = []
                    row_k_to_union: list[int] = []
                    if q_slot < q_count:
                        if dense_bucket:
                            row_k_rows = list(k_rows_member)
                            row_k_to_union = list(range(k_length))
                        else:
                            row_word_start = mask_member_start + q_slot * words_per_row
                            for word_idx in range(words_per_row):
                                word = int(bucket_mask_words[row_word_start + word_idx]) & 0xFFFFFFFF
                                while word:
                                    bit = word & -word
                                    bit_idx = bit.bit_length() - 1
                                    local_k = word_idx * 32 + bit_idx
                                    if local_k < k_length:
                                        row_k_rows.append(k_rows_member[local_k])
                                        row_k_to_union.append(local_k)
                                    word ^= bit
                    per_bucket_rows.append(row_k_rows)
                    per_bucket_to_union.append(row_k_to_union)
                    per_bucket_lengths.append(len(row_k_rows))
                    bucket_max_row_k = max(bucket_max_row_k, len(row_k_rows))

            if bucket_max_row_k > row_compact_cap:
                return None

            row_k_start = len(row_bucket_row_k_row_idx)
            row_k_to_union_start = len(row_bucket_row_k_to_union_idx)
            union_to_row_start = len(row_bucket_union_to_row_slot)
            row_k_length_start = len(row_bucket_row_k_length)
            unique_key_start = len(row_bucket_unique_key_row_idx)
            unique_ptr_start = len(row_bucket_unique_key_occurrence_row_ptr)
            row_bucket_unique_key_occurrence_row_ptr.append(len(row_bucket_unique_key_member_idx))
            row_bucket_row_k_cap.append(bucket_max_row_k)
            row_bucket_avg_row_k.append((sum(per_bucket_lengths) / len(per_bucket_lengths)) if per_bucket_lengths else 0.0)
            row_bucket_max_row_k.append(bucket_max_row_k)
            for row_k_rows, row_k_to_union in zip(per_bucket_rows, per_bucket_to_union):
                union_to_row = [-1] * packed_k
                for row_slot_idx, union_idx in enumerate(row_k_to_union):
                    if 0 <= union_idx < packed_k:
                        union_to_row[union_idx] = row_slot_idx
                row_bucket_row_k_length.append(len(row_k_rows))
                row_bucket_row_k_row_idx.extend(row_k_rows)
                row_bucket_row_k_row_idx.extend([-1] * (bucket_max_row_k - len(row_k_rows)))
                row_bucket_row_k_to_union_idx.extend(row_k_to_union)
                row_bucket_row_k_to_union_idx.extend([-1] * (bucket_max_row_k - len(row_k_to_union)))
                row_bucket_union_to_row_slot.extend(union_to_row)

            key_occurrence_map: dict[int, list[tuple[int, int]]] = {}
            for member_idx, union_rows in enumerate(per_bucket_union_rows):
                for union_idx, key_row in enumerate(union_rows):
                    if key_row >= 0:
                        key_occurrence_map.setdefault(int(key_row), []).append((member_idx, union_idx))
            bucket_max_unique_key_occurrences = 0
            for key_row in sorted(key_occurrence_map):
                bucket_max_unique_key_occurrences = max(
                    bucket_max_unique_key_occurrences, len(key_occurrence_map[key_row])
                )
                occ_start = len(row_bucket_unique_key_member_idx)
                for member_idx, union_idx in key_occurrence_map[key_row]:
                    row_bucket_unique_key_member_idx.append(int(member_idx))
                    row_bucket_unique_key_union_idx.append(int(union_idx))
                row_bucket_unique_key_row_idx.append(int(key_row))
                row_bucket_unique_key_occurrence_range.append((occ_start, len(row_bucket_unique_key_member_idx)))
                row_bucket_unique_key_occurrence_row_ptr.append(len(row_bucket_unique_key_member_idx))

            row_bucket_row_k_range.append((row_k_start, len(row_bucket_row_k_row_idx)))
            row_bucket_row_k_to_union_range.append((row_k_to_union_start, len(row_bucket_row_k_to_union_idx)))
            row_bucket_union_to_row_range.append((union_to_row_start, len(row_bucket_union_to_row_slot)))
            row_bucket_row_k_length_range.append((row_k_length_start, len(row_bucket_row_k_length)))
            row_bucket_unique_key_range.append((unique_key_start, len(row_bucket_unique_key_row_idx)))
            row_bucket_unique_key_occurrence_ptr_range.append(
                (unique_ptr_start, len(row_bucket_unique_key_occurrence_row_ptr))
            )
            row_bucket_max_unique_key_occurrences.append(bucket_max_unique_key_occurrences)

        return {
            "row_k_cap_limit": row_compact_cap,
            "bucket_row_k_cap": row_bucket_row_k_cap,
            "bucket_max_unique_key_occurrences": row_bucket_max_unique_key_occurrences,
            "bucket_row_k_range": row_bucket_row_k_range,
            "bucket_row_k_to_union_range": row_bucket_row_k_to_union_range,
            "bucket_union_to_row_range": row_bucket_union_to_row_range,
            "bucket_row_k_length_range": row_bucket_row_k_length_range,
            "bucket_unique_key_range": row_bucket_unique_key_range,
            "bucket_unique_key_occurrence_range": row_bucket_unique_key_occurrence_range,
            "bucket_unique_key_occurrence_ptr_range": row_bucket_unique_key_occurrence_ptr_range,
            "bucket_avg_row_k": row_bucket_avg_row_k,
            "bucket_max_row_k": row_bucket_max_row_k,
            "bucket_row_k_row_idx": torch.tensor(row_bucket_row_k_row_idx, dtype=torch.int32, device=device),
            "bucket_row_k_to_union_idx": torch.tensor(row_bucket_row_k_to_union_idx, dtype=torch.int32, device=device),
            "bucket_union_to_row_slot": torch.tensor(row_bucket_union_to_row_slot, dtype=torch.int32, device=device),
            "bucket_row_k_length": torch.tensor(row_bucket_row_k_length, dtype=torch.int32, device=device),
            "bucket_unique_key_row_idx": torch.tensor(row_bucket_unique_key_row_idx, dtype=torch.int32, device=device),
            "bucket_unique_key_member_idx": torch.tensor(
                row_bucket_unique_key_member_idx, dtype=torch.int32, device=device
            ),
            "bucket_unique_key_union_idx": torch.tensor(row_bucket_unique_key_union_idx, dtype=torch.int32, device=device),
            "bucket_unique_key_occurrence_row_ptr": torch.tensor(
                row_bucket_unique_key_occurrence_row_ptr, dtype=torch.int32, device=device
            ),
        }

    qgroup_bucket_map: dict[int, list[int]] = {}
    for qgroup_idx, packed_q in enumerate(qgroup_packed_q):
        qgroup_bucket_map.setdefault(packed_q, []).append(qgroup_idx)

    qgroup_bucket_row_ptr = [0]
    qgroup_bucket_idx: list[int] = []
    qgroup_bucket_packed_q: list[int] = []
    qgroup_bucket_q_row_idx_row_ptr = [0]
    qgroup_bucket_q_row_idx: list[int] = []
    qgroup_bucket_local_pos: dict[int, tuple[int, int]] = {}
    for packed_q, qgroup_ids in sorted(qgroup_bucket_map.items()):
        qgroup_bucket_id = len(qgroup_bucket_packed_q)
        qgroup_bucket_packed_q.append(packed_q)
        qgroup_bucket_idx.extend(qgroup_ids)
        qgroup_bucket_row_ptr.append(len(qgroup_bucket_idx))
        for local_pos, qgroup_idx in enumerate(qgroup_ids):
            qgroup_bucket_local_pos[qgroup_idx] = (qgroup_bucket_id, local_pos)
            q_start_idx = qgroup_row_ptr[qgroup_idx]
            q_end_idx = qgroup_row_ptr[qgroup_idx + 1]
            q_rows = qgroup_rows[q_start_idx:q_end_idx]
            qgroup_bucket_q_row_idx.extend(q_rows)
            qgroup_bucket_q_row_idx.extend([-1] * (packed_q - len(q_rows)))
        qgroup_bucket_q_row_idx_row_ptr.append(len(qgroup_bucket_q_row_idx))

    def _build_direct_execution_plan(
        *,
        split_entries_ref: list[dict[str, object]],
        split_indices_by_qgroup_ref: dict[int, list[int]],
        target_max_segments: int | None,
        merge_k_cap: int,
        allowed_qgroups: set[int] | None = None,
    ) -> dict[str, object] | None:
        direct_bucket_row_ptr = [0]
        direct_bucket_qgroup_bucket_idx: list[int] = []
        direct_bucket_segment_slot: list[int] = []
        direct_bucket_packed_q: list[int] = []
        direct_bucket_packed_k: list[int] = []
        direct_bucket_dense: list[bool] = []
        direct_bucket_q_row_idx_row_ptr = [0]
        direct_bucket_q_row_idx: list[int] = []
        direct_bucket_k_row_idx_row_ptr = [0]
        direct_bucket_k_row_idx: list[int] = []
        direct_bucket_q_length: list[int] = []
        direct_bucket_k_length: list[int] = []
        direct_bucket_mask_word_row_ptr = [0]
        direct_bucket_mask_words: list[int] = []
        direct_bucket_words_per_row: list[int] = []
        direct_bucket_allowed_pairs: list[int] = []
        direct_bucket_fill: list[float] = []
        direct_qgroup_num_segments = [0] * len(qgroup_length)
        direct_qgroup_bucket_segment_row_ptr = [0]
        direct_qgroup_bucket_segment_idx: list[int] = []
        direct_qgroup_bucket_num_segments: list[int] = []
        direct_available = True

        for qgroup_bucket_id in range(len(qgroup_bucket_packed_q)):
            qgroup_start = qgroup_bucket_row_ptr[qgroup_bucket_id]
            qgroup_end = qgroup_bucket_row_ptr[qgroup_bucket_id + 1]
            qgroup_ids = [
                int(qgroup_idx)
                for qgroup_idx in qgroup_bucket_idx[qgroup_start:qgroup_end]
                if allowed_qgroups is None or int(qgroup_idx) in allowed_qgroups
            ]
            packed_q = int(qgroup_bucket_packed_q[qgroup_bucket_id])
            qgroup_segments: dict[int, list[dict[str, object]]] = {}
            bucket_num_segments = 0

            for qgroup_idx in qgroup_ids:
                q_count = int(qgroup_length[qgroup_idx])
                split_ids = sorted(
                    split_indices_by_qgroup_ref.get(qgroup_idx, []),
                    key=lambda idx: int(split_entries_ref[idx]["slot"]),
                )
                split_groups = [[split_idx] for split_idx in split_ids]
                merged_groups = _merge_qgroup_split_groups(
                    qgroup_idx,
                    packed_q,
                    q_count,
                    split_groups,
                    split_entries_ref=split_entries_ref,
                    target_max_segments=target_max_segments,
                    merge_k_cap=merge_k_cap,
                )
                if merged_groups is None:
                    direct_available = False
                    break
                merged_entries = [
                    _build_qgroup_merged_entry(
                        qgroup_idx,
                        packed_q,
                        q_count,
                        merged_group,
                        split_entries_ref=split_entries_ref,
                    )
                    for merged_group in merged_groups
                ]
                qgroup_segments[qgroup_idx] = merged_entries
                direct_qgroup_num_segments[qgroup_idx] = len(merged_entries)
                bucket_num_segments = max(bucket_num_segments, len(merged_entries))
            if not direct_available:
                break

            direct_qgroup_bucket_num_segments.append(bucket_num_segments)
            for segment_slot in range(bucket_num_segments):
                active_members = [
                    (qgroup_idx, qgroup_segments[qgroup_idx][segment_slot])
                    for qgroup_idx in qgroup_ids
                    if segment_slot < len(qgroup_segments[qgroup_idx])
                ]
                bucket_size_value = len(active_members)
                packed_k = max((int(entry["k_length"]) for _, entry in active_members), default=0)
                if packed_k > merge_k_cap:
                    direct_available = False
                    break
                words_per_row = (packed_k + 31) // 32
                bucket_dense = all(bool(entry["dense"]) for _, entry in active_members)
                bucket_allowed_pairs_value = 0
                bucket_idx = len(direct_bucket_packed_q)
                direct_qgroup_bucket_segment_idx.append(bucket_idx)
                direct_bucket_qgroup_bucket_idx.append(qgroup_bucket_id)
                direct_bucket_segment_slot.append(segment_slot)
                direct_bucket_packed_q.append(packed_q)
                direct_bucket_packed_k.append(packed_k)
                direct_bucket_dense.append(bucket_dense)
                direct_bucket_row_ptr.append(direct_bucket_row_ptr[-1] + bucket_size_value)

                for qgroup_idx, entry in active_members:
                    qgroup_idx = int(qgroup_idx)
                    q_count = int(qgroup_length[qgroup_idx])
                    q_start_idx = qgroup_row_ptr[qgroup_idx]
                    q_end_idx = qgroup_row_ptr[qgroup_idx + 1]
                    q_rows = qgroup_rows[q_start_idx:q_end_idx]
                    direct_bucket_q_row_idx.extend(q_rows)
                    direct_bucket_q_row_idx.extend([-1] * (packed_q - len(q_rows)))
                    k_length = int(entry["k_length"])
                    direct_bucket_q_length.append(q_count)
                    direct_bucket_k_length.append(k_length)
                    bucket_allowed_pairs_value += int(entry["allowed_pairs"])
                    direct_bucket_k_row_idx.extend(int(row_idx) for row_idx in entry["padded_k_rows"][:k_length])
                    direct_bucket_k_row_idx.extend([-1] * (packed_k - k_length))
                    if not bucket_dense:
                        entry_words_per_row = int(entry["words_per_row"])
                        entry_mask_words = entry["mask_words"]
                        for q_slot in range(packed_q):
                            row_base = q_slot * entry_words_per_row
                            direct_bucket_mask_words.extend(
                                _wrap_u32_to_i32(int(word))
                                for word in entry_mask_words[row_base:row_base + entry_words_per_row]
                            )
                            direct_bucket_mask_words.extend([0] * (words_per_row - entry_words_per_row))

                direct_bucket_q_row_idx_row_ptr.append(len(direct_bucket_q_row_idx))
                direct_bucket_k_row_idx_row_ptr.append(len(direct_bucket_k_row_idx))
                direct_bucket_mask_word_row_ptr.append(len(direct_bucket_mask_words))
                direct_bucket_words_per_row.append(words_per_row)
                direct_bucket_allowed_pairs.append(bucket_allowed_pairs_value)
                direct_bucket_fill.append(
                    bucket_allowed_pairs_value / max(bucket_size_value * packed_q * max(packed_k, 1), 1)
                    if packed_k > 0
                    else 0.0
                )
            if not direct_available:
                break
            direct_qgroup_bucket_segment_row_ptr.append(len(direct_qgroup_bucket_segment_idx))

        if direct_available:
            direct_bucket_count = len(direct_bucket_packed_q)
            direct_bucket_q_row_range = [
                (direct_bucket_q_row_idx_row_ptr[idx], direct_bucket_q_row_idx_row_ptr[idx + 1])
                for idx in range(direct_bucket_count)
            ]
            direct_bucket_k_row_range = [
                (direct_bucket_k_row_idx_row_ptr[idx], direct_bucket_k_row_idx_row_ptr[idx + 1])
                for idx in range(direct_bucket_count)
            ]
            direct_bucket_data_range = [
                (direct_bucket_row_ptr[idx], direct_bucket_row_ptr[idx + 1])
                for idx in range(direct_bucket_count)
            ]
            direct_bucket_mask_word_range = [
                (direct_bucket_mask_word_row_ptr[idx], direct_bucket_mask_word_row_ptr[idx + 1])
                for idx in range(direct_bucket_count)
            ]
            row_compact_plan = _build_row_compact_plan_from_bucket_lists(
                bucket_packed_q=direct_bucket_packed_q,
                bucket_packed_k=direct_bucket_packed_k,
                bucket_dense=direct_bucket_dense,
                bucket_k_row_range=direct_bucket_k_row_range,
                bucket_data_range=direct_bucket_data_range,
                bucket_mask_word_range=direct_bucket_mask_word_range,
                bucket_q_length=direct_bucket_q_length,
                bucket_k_length=direct_bucket_k_length,
                bucket_k_row_idx=direct_bucket_k_row_idx,
                bucket_mask_words=direct_bucket_mask_words,
            )
            union_row_compact_plan = None
            if sparse_parse_fwd and logical_block_k > 2:
                union_bucket_row_ptr = [0]
                union_bucket_qgroup_bucket_idx: list[int] = []
                union_bucket_packed_q: list[int] = []
                union_bucket_packed_k: list[int] = []
                union_bucket_dense: list[bool] = []
                union_bucket_q_row_idx_row_ptr = [0]
                union_bucket_q_row_idx: list[int] = []
                union_bucket_k_row_idx_row_ptr = [0]
                union_bucket_k_row_idx: list[int] = []
                union_bucket_q_length: list[int] = []
                union_bucket_k_length: list[int] = []
                union_bucket_mask_word_row_ptr = [0]
                union_bucket_mask_words: list[int] = []
                union_bucket_words_per_row: list[int] = []
                union_bucket_allowed_pairs: list[int] = []
                union_bucket_fill: list[float] = []
                union_available = True

                for qgroup_bucket_id in range(len(qgroup_bucket_packed_q)):
                    qgroup_start = qgroup_bucket_row_ptr[qgroup_bucket_id]
                    qgroup_end = qgroup_bucket_row_ptr[qgroup_bucket_id + 1]
                    qgroup_ids = [int(qgroup_idx) for qgroup_idx in qgroup_bucket_idx[qgroup_start:qgroup_end]]
                    packed_q = int(qgroup_bucket_packed_q[qgroup_bucket_id])
                    union_entries = {
                        qgroup_idx: _build_hsa_qgroup_union_entry(
                            qgroup_idx,
                            qgroup_packed_q=qgroup_packed_q,
                            qgroup_length=qgroup_length,
                            split_entries=split_entries_ref,
                            split_indices_by_qgroup=split_indices_by_qgroup_ref,
                        )
                        for qgroup_idx in qgroup_ids
                    }
                    packed_k = max((int(entry["k_length"]) for entry in union_entries.values()), default=0)
                    if packed_q != 2 or packed_k <= 0 or packed_k > merge_k_cap:
                        union_available = False
                        break

                    words_per_row = (packed_k + 31) // 32
                    union_bucket_qgroup_bucket_idx.append(qgroup_bucket_id)
                    union_bucket_packed_q.append(packed_q)
                    union_bucket_packed_k.append(packed_k)
                    union_bucket_row_ptr.append(union_bucket_row_ptr[-1] + len(qgroup_ids))
                    all_dense = all(
                        bool(union_entries[qgroup_idx]["dense"]) and int(union_entries[qgroup_idx]["k_length"]) == packed_k
                        for qgroup_idx in qgroup_ids
                    )
                    union_bucket_dense.append(all_dense)
                    bucket_allowed_pairs_value = 0

                    for qgroup_idx in qgroup_ids:
                        q_start_idx = qgroup_row_ptr[qgroup_idx]
                        q_end_idx = qgroup_row_ptr[qgroup_idx + 1]
                        q_rows = qgroup_rows[q_start_idx:q_end_idx]
                        q_count = int(qgroup_length[qgroup_idx])
                        union_entry = union_entries[qgroup_idx]
                        union_k_length = int(union_entry["k_length"])
                        union_words_per_row = int(union_entry["words_per_row"])
                        union_bucket_q_row_idx.extend(q_rows)
                        union_bucket_q_row_idx.extend([-1] * (packed_q - len(q_rows)))
                        union_bucket_k_row_idx.extend(int(row_idx) for row_idx in union_entry["padded_k_rows"])
                        union_bucket_k_row_idx.extend([-1] * (packed_k - union_k_length))
                        union_bucket_q_length.append(q_count)
                        union_bucket_k_length.append(union_k_length)
                        bucket_allowed_pairs_value += int(union_entry["allowed_pairs"])
                        if not all_dense:
                            union_mask_words = union_entry["mask_words"]
                            for q_slot in range(packed_q):
                                row_base = q_slot * union_words_per_row
                                union_bucket_mask_words.extend(
                                    _wrap_u32_to_i32(int(word))
                                    for word in union_mask_words[row_base:row_base + union_words_per_row]
                                )
                                union_bucket_mask_words.extend([0] * (words_per_row - union_words_per_row))

                    union_bucket_q_row_idx_row_ptr.append(len(union_bucket_q_row_idx))
                    union_bucket_k_row_idx_row_ptr.append(len(union_bucket_k_row_idx))
                    union_bucket_mask_word_row_ptr.append(len(union_bucket_mask_words))
                    union_bucket_words_per_row.append(words_per_row)
                    union_bucket_allowed_pairs.append(bucket_allowed_pairs_value)
                    union_bucket_fill.append(bucket_allowed_pairs_value / max(len(qgroup_ids) * packed_q * packed_k, 1))

                if union_available:
                    union_bucket_q_row_range = [
                        (union_bucket_q_row_idx_row_ptr[idx], union_bucket_q_row_idx_row_ptr[idx + 1])
                        for idx in range(len(union_bucket_packed_q))
                    ]
                    union_bucket_k_row_range = [
                        (union_bucket_k_row_idx_row_ptr[idx], union_bucket_k_row_idx_row_ptr[idx + 1])
                        for idx in range(len(union_bucket_packed_q))
                    ]
                    union_bucket_data_range = [
                        (union_bucket_row_ptr[idx], union_bucket_row_ptr[idx + 1]) for idx in range(len(union_bucket_packed_q))
                    ]
                    union_bucket_mask_word_range = [
                        (union_bucket_mask_word_row_ptr[idx], union_bucket_mask_word_row_ptr[idx + 1])
                        for idx in range(len(union_bucket_packed_q))
                    ]
                    union_bucket_row_plan = _build_row_compact_plan_from_bucket_lists(
                        bucket_packed_q=union_bucket_packed_q,
                        bucket_packed_k=union_bucket_packed_k,
                        bucket_dense=union_bucket_dense,
                        bucket_k_row_range=union_bucket_k_row_range,
                        bucket_data_range=union_bucket_data_range,
                        bucket_mask_word_range=union_bucket_mask_word_range,
                        bucket_q_length=union_bucket_q_length,
                        bucket_k_length=union_bucket_k_length,
                        bucket_k_row_idx=union_bucket_k_row_idx,
                        bucket_mask_words=union_bucket_mask_words,
                    )
                    if union_bucket_row_plan is not None:
                        union_row_compact_plan = {
                            "bucket_qgroup_bucket_idx": union_bucket_qgroup_bucket_idx,
                            "bucket_size": [end - start for start, end in union_bucket_data_range],
                            "bucket_packed_q": union_bucket_packed_q,
                            "bucket_packed_k": union_bucket_packed_k,
                            "bucket_dense": union_bucket_dense,
                            "bucket_words_per_row": union_bucket_words_per_row,
                            "bucket_q_row_range": union_bucket_q_row_range,
                            "bucket_k_row_range": union_bucket_k_row_range,
                            "bucket_q_length_range": union_bucket_data_range,
                            "bucket_k_length_range": union_bucket_data_range,
                            "bucket_mask_word_range": union_bucket_mask_word_range,
                            "bucket_allowed_pairs": union_bucket_allowed_pairs,
                            "bucket_fill": union_bucket_fill,
                            "bucket_q_row_idx": torch.tensor(union_bucket_q_row_idx, dtype=torch.int32, device=device),
                            "bucket_k_row_idx": torch.tensor(union_bucket_k_row_idx, dtype=torch.int32, device=device),
                            "bucket_q_length": torch.tensor(union_bucket_q_length, dtype=torch.int32, device=device),
                            "bucket_k_length": torch.tensor(union_bucket_k_length, dtype=torch.int32, device=device),
                            "bucket_mask_words": torch.tensor(union_bucket_mask_words, dtype=torch.int32, device=device),
                            "row_compact_plan": union_bucket_row_plan,
                        }
            effective_max_direct_segments = max(
                max((int(num_segments) for num_segments in direct_qgroup_num_segments), default=0),
                0 if target_max_segments is None else int(target_max_segments),
            )
            return {
                "max_direct_segments": effective_max_direct_segments,
                "qgroup_num_segments": direct_qgroup_num_segments,
                "qgroup_bucket_num_segments": direct_qgroup_bucket_num_segments,
                "qgroup_bucket_segment_range": [
                    (direct_qgroup_bucket_segment_row_ptr[idx], direct_qgroup_bucket_segment_row_ptr[idx + 1])
                    for idx in range(len(direct_qgroup_bucket_num_segments))
                ],
                "qgroup_bucket_segment_idx": direct_qgroup_bucket_segment_idx,
                "bucket_qgroup_bucket_idx": direct_bucket_qgroup_bucket_idx,
                "bucket_segment_slot": direct_bucket_segment_slot,
                "bucket_size": [end - start for start, end in direct_bucket_data_range],
                "bucket_packed_q": direct_bucket_packed_q,
                "bucket_packed_k": direct_bucket_packed_k,
                "bucket_dense": direct_bucket_dense,
                "bucket_words_per_row": direct_bucket_words_per_row,
                "bucket_q_row_range": direct_bucket_q_row_range,
                "bucket_k_row_range": direct_bucket_k_row_range,
                "bucket_q_length_range": direct_bucket_data_range,
                "bucket_k_length_range": direct_bucket_data_range,
                "bucket_mask_word_range": direct_bucket_mask_word_range,
                "bucket_allowed_pairs": direct_bucket_allowed_pairs,
                "bucket_fill": direct_bucket_fill,
                "bucket_q_row_idx": torch.tensor(direct_bucket_q_row_idx, dtype=torch.int32, device=device),
                "bucket_k_row_idx": torch.tensor(direct_bucket_k_row_idx, dtype=torch.int32, device=device),
                "bucket_q_length": torch.tensor(direct_bucket_q_length, dtype=torch.int32, device=device),
                "bucket_k_length": torch.tensor(direct_bucket_k_length, dtype=torch.int32, device=device),
                "bucket_mask_words": torch.tensor(direct_bucket_mask_words, dtype=torch.int32, device=device),
                "row_compact_plan": row_compact_plan,
                "union_row_compact_plan": union_row_compact_plan,
            }
        return None

    def _select_hybrid_direct_qgroups(
        direct_plan: dict[str, object] | None,
        *,
        max_qgroup_segments: int,
    ) -> set[int]:
        if direct_plan is None or max_qgroup_segments <= 0:
            return set()
        row_plan = direct_plan.get("row_compact_plan")
        qgroup_num_segments = direct_plan.get("qgroup_num_segments")
        if row_plan is None or qgroup_num_segments is None:
            return set()
        selected_qgroups = {
            qgroup_idx
            for qgroup_idx, num_segments in enumerate(qgroup_num_segments)
            if 0 < int(num_segments) <= max_qgroup_segments
        }
        if not selected_qgroups:
            return set()
        total_direct_qgroups = sum(1 for num_segments in qgroup_num_segments if int(num_segments) > 0)
        if total_direct_qgroups <= 0 or len(selected_qgroups) >= total_direct_qgroups:
            return set()
        return selected_qgroups

    row_compact_split_entries: list[dict[str, object]] | None = None
    row_compact_split_indices_by_qgroup: dict[int, list[int]] | None = None
    direct_execution_plan = None
    if logical_block_q == 2:
        if logical_block_k <= row_compact_cap:
            row_compact_split_entries, row_compact_split_indices_by_qgroup = _build_direct_sharded_split_entries(
                row_compact_cap
            )
            direct_execution_plan = _build_direct_execution_plan(
                split_entries_ref=row_compact_split_entries,
                split_indices_by_qgroup_ref=row_compact_split_indices_by_qgroup,
                target_max_segments=None,
                merge_k_cap=row_compact_cap,
            )
        if direct_execution_plan is None or direct_execution_plan.get("row_compact_plan") is None:
            direct_execution_plan = _build_direct_execution_plan(
                split_entries_ref=split_entries,
                split_indices_by_qgroup_ref=split_indices_by_qgroup,
                target_max_segments=max_direct_segments,
                merge_k_cap=max_packed_k,
            )
    def _build_forward_execution_plan(
        *,
        excluded_qgroups: set[int] | None = None,
        direct_execution_plan_value: dict[str, object] | None = None,
    ) -> dict[str, object]:
        execution_bucket_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for split_idx, split_entry in enumerate(split_entries):
            qgroup_idx = int(split_entry["qgroup_idx"])
            if excluded_qgroups is not None and qgroup_idx in excluded_qgroups:
                continue
            qgroup_bucket_id, qgroup_local_idx = qgroup_bucket_local_pos[qgroup_idx]
            key = (
                qgroup_bucket_id,
                int(split_entry["packed_k"]),
            )
            execution_bucket_map.setdefault(key, []).append((split_idx, qgroup_local_idx))

        one_launch_bucket_data: dict[int, dict[str, object]] = {}
        if logical_block_q == 2 and logical_block_k == 2:
            for qgroup_bucket_id in range(len(qgroup_bucket_packed_q)):
                qgroup_start = qgroup_bucket_row_ptr[qgroup_bucket_id]
                qgroup_end = qgroup_bucket_row_ptr[qgroup_bucket_id + 1]
                qgroup_ids = [
                    int(qgroup_idx)
                    for qgroup_idx in qgroup_bucket_idx[qgroup_start:qgroup_end]
                    if excluded_qgroups is None or int(qgroup_idx) not in excluded_qgroups
                ]
                if not qgroup_ids:
                    continue
                union_entries = {
                    qgroup_idx: _build_hsa_qgroup_union_entry(
                        qgroup_idx,
                        qgroup_packed_q=qgroup_packed_q,
                        qgroup_length=qgroup_length,
                        split_entries=split_entries,
                        split_indices_by_qgroup=split_indices_by_qgroup,
                    )
                    for qgroup_idx in qgroup_ids
                }
                bucket_packed_k_value = max((int(entry["k_length"]) for entry in union_entries.values()), default=0)
                if bucket_packed_k_value <= max_packed_k:
                    one_launch_bucket_data[qgroup_bucket_id] = {
                        "qgroup_ids": qgroup_ids,
                        "union_entries": union_entries,
                        "packed_k": bucket_packed_k_value,
                    }

        bucket_row_ptr = [0]
        bucket_tile_idx: list[int] = []
        bucket_packed_q: list[int] = []
        bucket_packed_k: list[int] = []
        bucket_dense: list[int] = []
        bucket_q_row_idx_row_ptr = [0]
        bucket_q_row_idx: list[int] = []
        bucket_q_src_row_idx: list[int] = []
        bucket_k_row_idx_row_ptr = [0]
        bucket_k_row_idx: list[int] = []
        bucket_q_length: list[int] = []
        bucket_k_length: list[int] = []
        bucket_split_slot: list[int] = []
        bucket_qgroup_bucket_idx: list[int] = []
        bucket_mask_word_row_ptr = [0]
        bucket_mask_words: list[int] = []
        bucket_words_per_row: list[int] = []
        bucket_allowed_pairs: list[int] = []
        bucket_fill: list[float] = []
        bucket_use_qgroup_q: list[bool] = []
        bucket_scatter_only: list[bool] = []
        qgroup_bucket_to_split_buckets: dict[int, list[int]] = {
            bucket_id: [] for bucket_id in range(len(qgroup_bucket_packed_q))
        }

        for qgroup_bucket_id in range(len(qgroup_bucket_packed_q)):
            one_launch_bucket = one_launch_bucket_data.get(qgroup_bucket_id)
            if one_launch_bucket is None:
                continue
            packed_q = qgroup_bucket_packed_q[qgroup_bucket_id]
            packed_k = int(one_launch_bucket["packed_k"])
            words_per_row_bucket = (packed_k + 31) // 32
            qgroup_ids = one_launch_bucket["qgroup_ids"]
            union_entries = one_launch_bucket["union_entries"]
            bucket_idx = len(bucket_packed_q)
            qgroup_bucket_to_split_buckets[qgroup_bucket_id].append(bucket_idx)
            bucket_packed_q.append(packed_q)
            bucket_packed_k.append(packed_k)
            bucket_qgroup_bucket_idx.append(qgroup_bucket_id)
            bucket_split_slot.append(0)
            bucket_use_qgroup_q.append(True)
            bucket_scatter_only.append(True)
            bucket_tile_idx.extend(int(qgroup_idx) for qgroup_idx in qgroup_ids)
            bucket_row_ptr.append(len(bucket_tile_idx))

            all_dense = all(
                bool(union_entries[int(qgroup_idx)]["dense"]) and int(union_entries[int(qgroup_idx)]["k_length"]) == packed_k
                for qgroup_idx in qgroup_ids
            )
            bucket_dense.append(1 if all_dense else 0)
            bucket_allowed_pairs_value = 0

            for qgroup_local_idx, qgroup_idx in enumerate(qgroup_ids):
                qgroup_idx = int(qgroup_idx)
                q_start_idx = qgroup_row_ptr[qgroup_idx]
                q_end_idx = qgroup_row_ptr[qgroup_idx + 1]
                q_rows = qgroup_rows[q_start_idx:q_end_idx]
                q_count = int(qgroup_length[qgroup_idx])
                union_entry = union_entries[qgroup_idx]
                union_k_length = int(union_entry["k_length"])
                union_words_per_row = int(union_entry["words_per_row"])
                base_row = qgroup_local_idx * packed_q
                bucket_q_row_idx.extend(base_row + row for row in range(packed_q))
                bucket_q_src_row_idx.extend(q_rows)
                bucket_q_src_row_idx.extend([-1] * (packed_q - len(q_rows)))
                bucket_k_row_idx.extend(int(row_idx) for row_idx in union_entry["padded_k_rows"])
                bucket_k_row_idx.extend([-1] * (packed_k - union_k_length))
                bucket_q_length.append(q_count)
                bucket_k_length.append(union_k_length)
                bucket_allowed_pairs_value += int(union_entry["allowed_pairs"])
                if not all_dense:
                    union_mask_words = union_entry["mask_words"]
                    for q_slot in range(packed_q):
                        row_base = q_slot * union_words_per_row
                        bucket_mask_words.extend(
                            _wrap_u32_to_i32(int(word))
                            for word in union_mask_words[row_base:row_base + union_words_per_row]
                        )
                        bucket_mask_words.extend([0] * (words_per_row_bucket - union_words_per_row))

            bucket_q_row_idx_row_ptr.append(len(bucket_q_row_idx))
            bucket_k_row_idx_row_ptr.append(len(bucket_k_row_idx))
            bucket_mask_word_row_ptr.append(len(bucket_mask_words))
            bucket_words_per_row.append(words_per_row_bucket)
            bucket_allowed_pairs.append(bucket_allowed_pairs_value)
            bucket_fill.append(bucket_allowed_pairs_value / max(len(qgroup_ids) * packed_q * packed_k, 1))

        for (qgroup_bucket_id, packed_k), split_members in sorted(execution_bucket_map.items()):
            if qgroup_bucket_id in one_launch_bucket_data:
                continue
            packed_q = qgroup_bucket_packed_q[qgroup_bucket_id]
            words_per_row_bucket = (packed_k + 31) // 32
            split_members.sort(key=lambda item: (item[1], item[0]))
            all_dense = all(bool(split_entries[split_idx]["dense"]) for split_idx, _ in split_members)
            bucket_idx = len(bucket_packed_q)
            qgroup_bucket_to_split_buckets[qgroup_bucket_id].append(bucket_idx)
            bucket_packed_q.append(packed_q)
            bucket_packed_k.append(packed_k)
            bucket_dense.append(1 if all_dense else 0)
            bucket_split_slot.append(min(int(split_entries[split_idx]["slot"]) for split_idx, _ in split_members))
            bucket_qgroup_bucket_idx.append(qgroup_bucket_id)
            bucket_use_qgroup_q.append(False)
            bucket_scatter_only.append(False)
            bucket_tile_idx.extend(split_idx for split_idx, _ in split_members)
            bucket_row_ptr.append(len(bucket_tile_idx))
            bucket_allowed_pairs_value = 0

            for split_idx, qgroup_local_idx in split_members:
                split_entry = split_entries[split_idx]
                base_row = qgroup_local_idx * packed_q
                bucket_q_row_idx.extend(base_row + row for row in range(packed_q))
                qgroup_idx = int(split_entry["qgroup_idx"])
                q_start_idx = qgroup_row_ptr[qgroup_idx]
                q_end_idx = qgroup_row_ptr[qgroup_idx + 1]
                q_rows = qgroup_rows[q_start_idx:q_end_idx]
                bucket_q_src_row_idx.extend(q_rows)
                bucket_q_src_row_idx.extend([-1] * (packed_q - len(q_rows)))
                bucket_k_row_idx.extend(int(row_idx) for row_idx in split_entry["padded_k_rows"])
                bucket_q_length.append(int(split_entry["q_length"]))
                bucket_k_length.append(int(split_entry["k_length"]))
                if bool(split_entry["dense"]):
                    bucket_allowed_pairs_value += int(split_entry["q_length"]) * int(split_entry["k_length"])
                else:
                    bucket_allowed_pairs_value += sum(
                        (int(word) & 0xFFFFFFFF).bit_count() for word in split_entry["mask_words"]
                    )
                if not all_dense:
                    bucket_mask_words.extend(_wrap_u32_to_i32(int(word)) for word in split_entry["mask_words"])

            bucket_q_row_idx_row_ptr.append(len(bucket_q_row_idx))
            bucket_k_row_idx_row_ptr.append(len(bucket_k_row_idx))
            bucket_mask_word_row_ptr.append(len(bucket_mask_words))
            bucket_words_per_row.append(words_per_row_bucket)
            bucket_allowed_pairs.append(bucket_allowed_pairs_value)
            bucket_fill.append(bucket_allowed_pairs_value / max(len(split_members) * packed_q * packed_k, 1))

        qgroup_bucket_split_bucket_row_ptr = [0]
        qgroup_bucket_split_bucket_idx: list[int] = []
        for qgroup_bucket_id in range(len(qgroup_bucket_packed_q)):
            split_bucket_ids = qgroup_bucket_to_split_buckets.get(qgroup_bucket_id, [])
            split_bucket_ids.sort(
                key=lambda bucket_idx: (
                    bucket_packed_k[bucket_idx],
                    bucket_dense[bucket_idx],
                    bucket_idx,
                )
            )
            qgroup_bucket_split_bucket_idx.extend(split_bucket_ids)
            qgroup_bucket_split_bucket_row_ptr.append(len(qgroup_bucket_split_bucket_idx))

        bucket_size = [bucket_row_ptr[idx + 1] - bucket_row_ptr[idx] for idx in range(len(bucket_packed_q))]
        bucket_q_row_range = [
            (bucket_q_row_idx_row_ptr[idx], bucket_q_row_idx_row_ptr[idx + 1]) for idx in range(len(bucket_packed_q))
        ]
        bucket_k_row_range = [
            (bucket_k_row_idx_row_ptr[idx], bucket_k_row_idx_row_ptr[idx + 1]) for idx in range(len(bucket_packed_q))
        ]
        bucket_data_range = [(bucket_row_ptr[idx], bucket_row_ptr[idx + 1]) for idx in range(len(bucket_packed_q))]
        bucket_mask_word_range = [
            (bucket_mask_word_row_ptr[idx], bucket_mask_word_row_ptr[idx + 1]) for idx in range(len(bucket_packed_q))
        ]
        qgroup_bucket_range = [
            (qgroup_bucket_row_ptr[idx], qgroup_bucket_row_ptr[idx + 1]) for idx in range(len(qgroup_bucket_packed_q))
        ]
        qgroup_bucket_q_row_range = [
            (qgroup_bucket_q_row_idx_row_ptr[idx], qgroup_bucket_q_row_idx_row_ptr[idx + 1])
            for idx in range(len(qgroup_bucket_packed_q))
        ]
        qgroup_bucket_execution_bucket_range = [
            (qgroup_bucket_split_bucket_row_ptr[idx], qgroup_bucket_split_bucket_row_ptr[idx + 1])
            for idx in range(len(qgroup_bucket_packed_q))
        ]
        return {
            "qgroup_bucket_packed_q": qgroup_bucket_packed_q,
            "qgroup_bucket_size": [end - start for start, end in qgroup_bucket_range],
            "qgroup_bucket_range": qgroup_bucket_range,
            "qgroup_bucket_q_row_range": qgroup_bucket_q_row_range,
            "qgroup_bucket_execution_bucket_range": qgroup_bucket_execution_bucket_range,
            "qgroup_bucket_execution_bucket_idx": qgroup_bucket_split_bucket_idx,
            "bucket_row_ptr": bucket_row_ptr,
            "bucket_tile_idx": bucket_tile_idx,
            "bucket_qgroup_bucket_idx": bucket_qgroup_bucket_idx,
            "bucket_size": bucket_size,
            "bucket_packed_q": bucket_packed_q,
            "bucket_packed_k": bucket_packed_k,
            "bucket_dense": [bool(value) for value in bucket_dense],
            "bucket_allowed_pairs": bucket_allowed_pairs,
            "bucket_fill": bucket_fill,
            "bucket_words_per_row": bucket_words_per_row,
            "bucket_q_row_idx_row_ptr": bucket_q_row_idx_row_ptr,
            "bucket_q_row_idx": bucket_q_row_idx,
            "bucket_q_src_row_idx": bucket_q_src_row_idx,
            "bucket_q_row_range": bucket_q_row_range,
            "bucket_q_src_row_range": bucket_q_row_range,
            "bucket_k_row_idx_row_ptr": bucket_k_row_idx_row_ptr,
            "bucket_k_row_idx": bucket_k_row_idx,
            "bucket_k_row_range": bucket_k_row_range,
            "bucket_q_length": bucket_q_length,
            "bucket_k_length": bucket_k_length,
            "bucket_q_length_range": bucket_data_range,
            "bucket_k_length_range": bucket_data_range,
            "bucket_split_slot": bucket_split_slot,
            "bucket_mask_word_row_ptr": bucket_mask_word_row_ptr,
            "bucket_mask_words": bucket_mask_words,
            "bucket_mask_word_range": bucket_mask_word_range,
            "bucket_use_qgroup_q": bucket_use_qgroup_q,
            "bucket_scatter_only": bucket_scatter_only,
            "qgroup_bucket_split_bucket_row_ptr": qgroup_bucket_split_bucket_row_ptr,
            "qgroup_bucket_split_bucket_idx": qgroup_bucket_split_bucket_idx,
            "qgroup_union_support_plan": qgroup_union_support_plan,
            "direct_execution_plan": direct_execution_plan_value,
        }

    forward_execution_plan = _build_forward_execution_plan(direct_execution_plan_value=direct_execution_plan)
    hybrid_selected_qgroups = _select_hybrid_direct_qgroups(
        direct_execution_plan,
        max_qgroup_segments=_get_hsa_synthetic_hybrid_max_qgroup_segments(),
    )
    hybrid_direct_execution_plan = None
    hybrid_regular_execution_plan = None
    if hybrid_selected_qgroups and row_compact_split_entries is not None and row_compact_split_indices_by_qgroup is not None:
        hybrid_direct_execution_plan = _build_direct_execution_plan(
            split_entries_ref=row_compact_split_entries,
            split_indices_by_qgroup_ref=row_compact_split_indices_by_qgroup,
            target_max_segments=None,
            merge_k_cap=row_compact_cap,
            allowed_qgroups=hybrid_selected_qgroups,
        )
        if hybrid_direct_execution_plan is None or hybrid_direct_execution_plan.get("row_compact_plan") is None:
            hybrid_direct_execution_plan = None
        else:
            hybrid_regular_execution_plan = _build_forward_execution_plan(
                excluded_qgroups=hybrid_selected_qgroups,
                direct_execution_plan_value=direct_execution_plan,
            )
    forward_execution_plan["hybrid_direct_execution_plan"] = hybrid_direct_execution_plan
    forward_execution_plan["hybrid_regular_execution_plan"] = hybrid_regular_execution_plan
    forward_execution_plan["hybrid_selected_qgroup_idx"] = sorted(hybrid_selected_qgroups)

    return HSASyntheticGridMetadata(
        logical_block_q=logical_block_q,
        logical_block_k=logical_block_k,
        physical_block_q=q_block_size,
        physical_block_k=k_block_size,
        tile_batch_idx=torch.tensor(tile_batch_idx, dtype=torch.int32, device=device),
        tile_q_block_idx=torch.tensor(tile_q_block_idx, dtype=torch.int32, device=device),
        tile_q_subgroup_idx=torch.tensor(tile_q_subgroup_idx, dtype=torch.int32, device=device),
        tile_k_block_idx=torch.tensor(tile_k_block_idx, dtype=torch.int32, device=device),
        tile_q_row_ptr=torch.tensor(tile_q_row_ptr, dtype=torch.int32, device=device),
        tile_q_rows=torch.tensor(tile_q_rows, dtype=torch.int32, device=device),
        tile_k_row_ptr=torch.tensor(tile_k_row_ptr, dtype=torch.int32, device=device),
        tile_k_rows=torch.tensor(tile_k_rows, dtype=torch.int32, device=device),
        tile_logical_pair_row_ptr=torch.tensor(tile_logical_pair_row_ptr, dtype=torch.int32, device=device),
        tile_logical_pairs=torch.tensor(tile_logical_pairs, dtype=torch.int32, device=device),
        compact_mask_row_ptr=torch.tensor(compact_mask_row_ptr, dtype=torch.int32, device=device),
        compact_mask_col_idx=torch.tensor(compact_mask_col_idx, dtype=torch.int32, device=device),
        tile_allowed_pairs=torch.tensor(tile_allowed_pairs, dtype=torch.int32, device=device),
        tile_packed_q=torch.tensor(tile_packed_q, dtype=torch.int32, device=device),
        tile_packed_k=torch.tensor(tile_packed_k, dtype=torch.int32, device=device),
        tile_dense=torch.tensor(tile_dense, dtype=torch.bool, device=device),
        tile_fill=torch.tensor(tile_fill, dtype=torch.float32, device=device),
        bucket_row_ptr=torch.tensor(forward_execution_plan["bucket_row_ptr"], dtype=torch.int32, device=device),
        bucket_tile_idx=torch.tensor(forward_execution_plan["bucket_tile_idx"], dtype=torch.int32, device=device),
        bucket_packed_q=torch.tensor(forward_execution_plan["bucket_packed_q"], dtype=torch.int32, device=device),
        bucket_packed_k=torch.tensor(forward_execution_plan["bucket_packed_k"], dtype=torch.int32, device=device),
        bucket_dense=torch.tensor(forward_execution_plan["bucket_dense"], dtype=torch.bool, device=device),
        bucket_allowed_pairs=torch.tensor(forward_execution_plan["bucket_allowed_pairs"], dtype=torch.int32, device=device),
        bucket_fill=torch.tensor(forward_execution_plan["bucket_fill"], dtype=torch.float32, device=device),
        max_packed_k=max_packed_k,
        max_direct_segments=max_direct_segments,
        sparse_parse_fwd=sparse_parse_fwd,
        tile_q_length=torch.tensor(tile_q_length, dtype=torch.int32, device=device),
        tile_k_length=torch.tensor(tile_k_length, dtype=torch.int32, device=device),
        bucket_q_row_idx_row_ptr=torch.tensor(
            forward_execution_plan["bucket_q_row_idx_row_ptr"], dtype=torch.int32, device=device
        ),
        bucket_q_row_idx=torch.tensor(forward_execution_plan["bucket_q_row_idx"], dtype=torch.int32, device=device),
        bucket_q_src_row_idx=torch.tensor(forward_execution_plan["bucket_q_src_row_idx"], dtype=torch.int32, device=device),
        bucket_k_row_idx_row_ptr=torch.tensor(
            forward_execution_plan["bucket_k_row_idx_row_ptr"], dtype=torch.int32, device=device
        ),
        bucket_k_row_idx=torch.tensor(forward_execution_plan["bucket_k_row_idx"], dtype=torch.int32, device=device),
        bucket_q_length=torch.tensor(forward_execution_plan["bucket_q_length"], dtype=torch.int32, device=device),
        bucket_k_length=torch.tensor(forward_execution_plan["bucket_k_length"], dtype=torch.int32, device=device),
        bucket_split_slot=torch.tensor(forward_execution_plan["bucket_split_slot"], dtype=torch.int32, device=device),
        bucket_qgroup_bucket_idx=torch.tensor(
            forward_execution_plan["bucket_qgroup_bucket_idx"], dtype=torch.int32, device=device
        ),
        bucket_mask_word_row_ptr=torch.tensor(
            forward_execution_plan["bucket_mask_word_row_ptr"], dtype=torch.int32, device=device
        ),
        bucket_mask_words=torch.tensor(forward_execution_plan["bucket_mask_words"], dtype=torch.int32, device=device),
        bucket_words_per_row=torch.tensor(forward_execution_plan["bucket_words_per_row"], dtype=torch.int32, device=device),
        qgroup_row_ptr=torch.tensor(qgroup_row_ptr, dtype=torch.int32, device=device),
        qgroup_rows=torch.tensor(qgroup_rows, dtype=torch.int32, device=device),
        qgroup_length=torch.tensor(qgroup_length, dtype=torch.int32, device=device),
        qgroup_packed_q=torch.tensor(qgroup_packed_q, dtype=torch.int32, device=device),
        qgroup_num_splits=torch.tensor(qgroup_num_splits, dtype=torch.int32, device=device),
        qgroup_bucket_row_ptr=torch.tensor(qgroup_bucket_row_ptr, dtype=torch.int32, device=device),
        qgroup_bucket_idx=torch.tensor(qgroup_bucket_idx, dtype=torch.int32, device=device),
        qgroup_bucket_packed_q=torch.tensor(qgroup_bucket_packed_q, dtype=torch.int32, device=device),
        qgroup_bucket_q_row_idx_row_ptr=torch.tensor(
            qgroup_bucket_q_row_idx_row_ptr, dtype=torch.int32, device=device
        ),
        qgroup_bucket_q_row_idx=torch.tensor(qgroup_bucket_q_row_idx, dtype=torch.int32, device=device),
        qgroup_bucket_split_bucket_row_ptr=torch.tensor(
            forward_execution_plan["qgroup_bucket_split_bucket_row_ptr"], dtype=torch.int32, device=device
        ),
        qgroup_bucket_split_bucket_idx=torch.tensor(
            forward_execution_plan["qgroup_bucket_split_bucket_idx"], dtype=torch.int32, device=device
        ),
        forward_execution_plan=forward_execution_plan,
    )


def _build_hsa_backward_synthetic_grid_metadata(
    schedule: HSASchedule,
    runtime: HSABlockSparseRuntime,
    *,
    logical_block_q: int = 32,
    logical_block_k: int = 32,
) -> HSASyntheticGridMetadata:
    sparse_tensors = runtime.backward_sparse
    packed_masks = runtime.backward_packed_masks
    q_block_size, k_block_size = sparse_tensors.block_size
    if q_block_size % logical_block_q != 0 or k_block_size % logical_block_k != 0:
        raise ValueError("Synthetic grid requires logical block sizes that evenly divide the physical sparse blocks")

    device = sparse_tensors.mask_block_cnt.device
    seqlen = schedule.seqlen
    mask_block_cnt = sparse_tensors.mask_block_cnt.detach().cpu()
    mask_block_idx = sparse_tensors.mask_block_idx.detach().cpu()
    full_block_cnt = None if sparse_tensors.full_block_cnt is None else sparse_tensors.full_block_cnt.detach().cpu()
    full_block_idx = None if sparse_tensors.full_block_idx is None else sparse_tensors.full_block_idx.detach().cpu()
    block_id_table = packed_masks.block_id_table.detach().cpu()
    mask_words = packed_masks.mask_words.detach().cpu()
    words_per_row = packed_masks.words_per_row
    num_batches = int(mask_block_cnt.shape[0])
    num_k_blocks = int(mask_block_cnt.shape[2])

    tile_batch_idx: list[int] = []
    tile_q_block_idx: list[int] = []
    tile_k_block_idx: list[int] = []
    tile_q_row_ptr = [0]
    tile_k_row_ptr = [0]
    tile_logical_pair_row_ptr = [0]
    tile_q_rows: list[int] = []
    tile_k_rows: list[int] = []
    tile_logical_pairs: list[list[int]] = []
    compact_mask_row_ptr = [0]
    compact_mask_col_idx: list[int] = []

    def _append_tile(
        batch_idx: int,
        q_block_idx: int,
        k_block_idx: int,
        q_rows_local: list[int],
        k_rows_local: list[int],
        row_cols_local: list[list[int]],
        logical_pairs_local: set[tuple[int, int]],
    ) -> None:
        if not q_rows_local or not k_rows_local:
            return
        k_compact = {k_local: idx for idx, k_local in enumerate(k_rows_local)}
        tile_batch_idx.append(batch_idx)
        tile_q_block_idx.append(q_block_idx)
        tile_k_block_idx.append(k_block_idx)
        tile_q_rows.extend(q_rows_local)
        tile_k_rows.extend(k_rows_local)
        tile_logical_pairs.extend([[q_sub, k_sub] for q_sub, k_sub in sorted(logical_pairs_local)])
        tile_q_row_ptr.append(len(tile_q_rows))
        tile_k_row_ptr.append(len(tile_k_rows))
        tile_logical_pair_row_ptr.append(len(tile_logical_pairs))
        for cols in row_cols_local:
            compact_cols = [k_compact[k_local] for k_local in cols]
            compact_mask_col_idx.extend(compact_cols)
            compact_mask_row_ptr.append(len(compact_mask_col_idx))

    for batch_idx in range(num_batches):
        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * k_block_size
            k_len = min(k_block_size, seqlen - k_start)
            if k_len <= 0:
                continue
            tail_mask = (1 << (k_len % 32)) - 1 if k_len % 32 != 0 else None

            full_cnt = 0 if full_block_cnt is None else int(full_block_cnt[batch_idx, 0, k_block_idx].item())
            for offset in range(full_cnt):
                q_block_idx = int(full_block_idx[batch_idx, 0, k_block_idx, offset].item())
                q_start = q_block_idx * q_block_size
                q_len = min(q_block_size, seqlen - q_start)
                if q_len <= 0:
                    continue
                q_rows_local = list(range(q_len))
                k_rows_local = list(range(k_len))
                row_cols_local = [list(range(k_len)) for _ in range(q_len)]
                logical_pairs_local = {
                    (q_sub, k_sub)
                    for q_sub in range((q_len + logical_block_q - 1) // logical_block_q)
                    for k_sub in range((k_len + logical_block_k - 1) // logical_block_k)
                }
                _append_tile(
                    batch_idx,
                    q_block_idx,
                    k_block_idx,
                    q_rows_local,
                    k_rows_local,
                    row_cols_local,
                    logical_pairs_local,
                )

            partial_cnt = int(mask_block_cnt[batch_idx, 0, k_block_idx].item())
            for offset in range(partial_cnt):
                q_block_idx = int(mask_block_idx[batch_idx, 0, k_block_idx, offset].item())
                q_start = q_block_idx * q_block_size
                q_len = min(q_block_size, seqlen - q_start)
                if q_len <= 0:
                    continue
                block_id = int(block_id_table[batch_idx, k_block_idx, q_block_idx].item())
                row_cols_map: dict[int, list[int]] = {}
                active_k_rows: set[int] = set()
                logical_pairs_local: set[tuple[int, int]] = set()
                for q_local in range(q_len):
                    cols: list[int] = []
                    for word_idx in range(words_per_row):
                        word = int(mask_words[block_id, q_local, word_idx].item()) & 0xFFFFFFFF
                        if tail_mask is not None and word_idx == words_per_row - 1:
                            word &= tail_mask
                        while word:
                            bit = word & -word
                            bit_idx = bit.bit_length() - 1
                            k_local = word_idx * 32 + bit_idx
                            if k_local < k_len:
                                cols.append(k_local)
                                active_k_rows.add(k_local)
                                logical_pairs_local.add((q_local // logical_block_q, k_local // logical_block_k))
                            word ^= bit
                    if cols:
                        row_cols_map[q_local] = cols

                q_rows_local = sorted(row_cols_map.keys())
                k_rows_local = sorted(active_k_rows)
                row_cols_local = [row_cols_map[q_local] for q_local in q_rows_local]
                _append_tile(
                    batch_idx,
                    q_block_idx,
                    k_block_idx,
                    q_rows_local,
                    k_rows_local,
                    row_cols_local,
                    logical_pairs_local,
                )

    return _finalize_hsa_synthetic_grid_metadata(
        device=device,
        logical_block_q=logical_block_q,
        logical_block_k=logical_block_k,
        physical_block_q=q_block_size,
        physical_block_k=k_block_size,
        tile_batch_idx=tile_batch_idx,
        tile_q_block_idx=tile_q_block_idx,
        tile_k_block_idx=tile_k_block_idx,
        tile_q_row_ptr=tile_q_row_ptr,
        tile_q_rows=tile_q_rows,
        tile_k_row_ptr=tile_k_row_ptr,
        tile_k_rows=tile_k_rows,
        tile_logical_pair_row_ptr=tile_logical_pair_row_ptr,
        tile_logical_pairs=tile_logical_pairs,
        compact_mask_row_ptr=compact_mask_row_ptr,
        compact_mask_col_idx=compact_mask_col_idx,
    )


def _ensure_hsa_backward_synthetic_grid_metadata(
    schedule: HSASchedule,
    runtime: HSABlockSparseRuntime,
) -> HSASyntheticGridMetadata:
    if runtime.backward_synthetic_grid is None:
        runtime.backward_synthetic_grid = _build_hsa_backward_synthetic_grid_metadata(schedule, runtime)
    return runtime.backward_synthetic_grid


def _ensure_hsa_synthetic_grid_metadata(
    schedule: HSASchedule,
    runtime: HSABlockSparseRuntime,
    *,
    require_full_forward_plan: bool = False,
    require_backward: bool = False,
) -> HSASyntheticGridMetadata:
    forward_logical_block_q = _get_hsa_synthetic_logical_block_size("q")
    forward_logical_block_k = _get_hsa_synthetic_logical_block_size("k")
    forward_max_packed_k = _get_hsa_synthetic_max_packed_k(forward_logical_block_k)
    forward_max_direct_segments = _get_hsa_synthetic_max_direct_segments()
    forward_sparse_parse_fwd = _use_hsa_synthetic_sparse_parse_fwd()
    needs_forward_rebuild = (
        runtime.forward_synthetic_grid is None
        or runtime.forward_synthetic_grid.logical_block_q != forward_logical_block_q
        or runtime.forward_synthetic_grid.logical_block_k != forward_logical_block_k
        or runtime.forward_synthetic_grid.max_packed_k != forward_max_packed_k
        or runtime.forward_synthetic_grid.max_direct_segments != forward_max_direct_segments
        or bool(runtime.forward_synthetic_grid.sparse_parse_fwd) != forward_sparse_parse_fwd
    )
    if needs_forward_rebuild:
        precomputed_direct_plan = None if require_full_forward_plan else _get_precomputed_forward_direct_plan(
            schedule,
            forward_block_q=runtime.forward_block_q,
            logical_block_q=forward_logical_block_q,
            logical_block_k=forward_logical_block_k,
            max_packed_k=forward_max_packed_k,
            max_direct_segments=forward_max_direct_segments,
            device=runtime.forward_sparse.mask_block_cnt.device,
        )
        if precomputed_direct_plan is not None:
            runtime.forward_synthetic_grid = _build_precomputed_direct_only_synthetic_grid_metadata(
                device=runtime.forward_sparse.mask_block_cnt.device,
                logical_block_q=forward_logical_block_q,
                logical_block_k=forward_logical_block_k,
                physical_block_q=runtime.forward_block_q,
                physical_block_k=runtime.forward_block_k,
                max_packed_k=forward_max_packed_k,
                max_direct_segments=forward_max_direct_segments,
                sparse_parse_fwd=forward_sparse_parse_fwd,
                direct_execution_plan=precomputed_direct_plan,
            )
        else:
            runtime.forward_synthetic_grid = _build_hsa_forward_synthetic_grid_metadata(
                schedule,
                runtime,
                logical_block_q=forward_logical_block_q,
                logical_block_k=forward_logical_block_k,
                max_packed_k=forward_max_packed_k,
            )
    elif require_full_forward_plan and not _forward_execution_plan_has_bucket_execution(
        runtime.forward_synthetic_grid.forward_execution_plan
    ):
        runtime.forward_synthetic_grid = _build_hsa_forward_synthetic_grid_metadata(
            schedule,
            runtime,
            logical_block_q=forward_logical_block_q,
            logical_block_k=forward_logical_block_k,
            max_packed_k=forward_max_packed_k,
        )
    if require_backward:
        _ensure_hsa_backward_synthetic_grid_metadata(schedule, runtime)
    runtime.synthetic_grid = runtime.forward_synthetic_grid
    return runtime.synthetic_grid


def _get_hsa_backward_subtile_factor() -> int:
    subtile_factor = int(os.environ.get("FLASH_ATTN_HSA_BACKWARD_SUBTILE_FACTOR", "1"))
    if subtile_factor not in (1, 2):
        raise ValueError("FLASH_ATTN_HSA_BACKWARD_SUBTILE_FACTOR must be 1 or 2")
    return subtile_factor


def _get_hsa_backward_block_q() -> int:
    block_q = int(os.environ.get("FLASH_ATTN_HSA_BACKWARD_BLOCK_Q", "64"))
    if block_q not in (32, 64, 128):
        raise ValueError("FLASH_ATTN_HSA_BACKWARD_BLOCK_Q must be 32, 64, or 128")
    return block_q


def _get_hsa_backward_block_k() -> int:
    block_k = int(os.environ.get("FLASH_ATTN_HSA_BACKWARD_BLOCK_K", "128"))
    if block_k not in (32, 64, 128):
        raise ValueError("FLASH_ATTN_HSA_BACKWARD_BLOCK_K must be 32, 64, or 128")
    return block_k


@lru_cache(maxsize=1)
def get_hsa_schedule_mask_mod():
    """Return the CuTe mask_mod for exact HSA using canonical schedule aux tensors."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()

    @fast_sampling
    @cute.jit
    def _hsa_schedule_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        sentence_segment_id = aux_tensors[0]
        sentence_segment_offset = aux_tensors[1]
        section_segment_id = aux_tensors[2]
        section_segment_offset = aux_tensors[3]
        section_self_allowed = aux_tensors[4]
        document_segment_id = aux_tensors[5]
        document_segment_offset = aux_tensors[6]
        document_self_allowed = aux_tensors[7]

        b = batch[0]
        q_idx = m_idx[0]
        kv_idx = n_idx[0]
        safe_q_idx = q_idx % seqlen_info.seqlen_q
        safe_kv_idx = kv_idx % seqlen_info.seqlen_k
        in_bounds = (q_idx < seqlen_info.seqlen_q) & (kv_idx < seqlen_info.seqlen_k)

        q_sent_id = utils.scalar_to_ssa(sentence_segment_id[b, safe_q_idx], cutlass.Int32)
        q_sent_offset = utils.scalar_to_ssa(sentence_segment_offset[b, safe_q_idx], cutlass.Int32)
        k_sent_id = utils.scalar_to_ssa(sentence_segment_id[b, safe_kv_idx], cutlass.Int32)
        k_sent_offset = utils.scalar_to_ssa(sentence_segment_offset[b, safe_kv_idx], cutlass.Int32)

        q_sec_id = utils.scalar_to_ssa(section_segment_id[b, safe_q_idx], cutlass.Int32)
        q_sec_offset = utils.scalar_to_ssa(section_segment_offset[b, safe_q_idx], cutlass.Int32)
        q_sec_self = utils.scalar_to_ssa(section_self_allowed[b, safe_q_idx], cutlass.Int32)
        k_sec_id = utils.scalar_to_ssa(section_segment_id[b, safe_kv_idx], cutlass.Int32)
        k_sec_offset = utils.scalar_to_ssa(section_segment_offset[b, safe_kv_idx], cutlass.Int32)

        q_doc_id = utils.scalar_to_ssa(document_segment_id[b, safe_q_idx], cutlass.Int32)
        q_doc_offset = utils.scalar_to_ssa(document_segment_offset[b, safe_q_idx], cutlass.Int32)
        q_doc_self = utils.scalar_to_ssa(document_self_allowed[b, safe_q_idx], cutlass.Int32)
        k_doc_id = utils.scalar_to_ssa(document_segment_id[b, safe_kv_idx], cutlass.Int32)
        k_doc_offset = utils.scalar_to_ssa(document_segment_offset[b, safe_kv_idx], cutlass.Int32)

        same_sentence = (q_sent_id >= 0) & (q_sent_id == k_sent_id) & (k_sent_offset <= q_sent_offset)
        same_section = (q_sec_id >= 0) & (q_sec_id == k_sec_id) & (k_sec_offset < (q_sec_offset + q_sec_self))
        same_document = (q_doc_id >= 0) & (q_doc_id == k_doc_id) & (k_doc_offset < (q_doc_offset + q_doc_self))
        return in_bounds & (same_sentence | same_section | same_document)

    return _hsa_schedule_mask


@lru_cache(maxsize=None)
def get_hsa_forward_tile_mask_mod(q_block_size: int, k_block_size: int):
    """Return the CuTe mask_mod for exact HSA forward partial tiles."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()
    words_per_row = (k_block_size + 31) // 32

    @fast_sampling
    @cute.jit
    def _hsa_forward_tile_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        block_id_table = aux_tensors[0]
        tile_kind = aux_tensors[1]
        affine_base = aux_tensors[2]
        row_prefix_row_ptr = aux_tensors[3]
        row_prefix_len = aux_tensors[4]
        bitmap_word_row_ptr = aux_tensors[5]
        bitmap_words = aux_tensors[6]

        b = utils.ssa_to_scalar(batch)
        q_idx = utils.ssa_to_scalar(m_idx)
        kv_idx = utils.ssa_to_scalar(n_idx)
        safe_q_idx = q_idx % seqlen_info.seqlen_q
        safe_kv_idx = kv_idx % seqlen_info.seqlen_k
        in_bounds = (q_idx < seqlen_info.seqlen_q) & (kv_idx < seqlen_info.seqlen_k)

        q_block = safe_q_idx // q_block_size
        k_block = safe_kv_idx // k_block_size
        q_local = safe_q_idx % q_block_size
        k_local = safe_kv_idx % k_block_size
        word_idx = k_local // 32
        bit_idx = k_local % 32

        block_id = block_id_table[b, q_block, k_block]
        kind = utils.scalar_to_ssa(tile_kind[block_id], cutlass.Int32)

        affine_prefix = utils.scalar_to_ssa(affine_base[block_id], cutlass.Int32) + q_local
        prefix_offset = row_prefix_row_ptr[block_id] + q_local
        row_prefix = utils.scalar_to_ssa(row_prefix_len[prefix_offset], cutlass.Int32)
        word_offset = bitmap_word_row_ptr[block_id] + q_local * words_per_row + word_idx
        word = cutlass.Uint32(bitmap_words[word_offset])
        bit = utils.shr_u32(word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)

        affine_allowed = (kind == _HSA_FWD_TILE_AFFINE_PREFIX) & (k_local < affine_prefix)
        row_prefix_allowed = (kind == _HSA_FWD_TILE_ROW_PREFIX) & (k_local < row_prefix)
        bitmap_allowed = (kind == _HSA_FWD_TILE_BITMAP) & (bit != cutlass.Uint32(0))

        return in_bounds & (affine_allowed | row_prefix_allowed | bitmap_allowed)

    return _hsa_forward_tile_mask


@lru_cache(maxsize=None)
def get_hsa_backward_packed_mask_mod(q_block_size: int, k_block_size: int):
    """Return the CuTe mask_mod for packed backward HSA partial blocks."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()

    @fast_sampling
    @cute.jit
    def _hsa_backward_packed_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        block_id_table = aux_tensors[0]
        mask_words = aux_tensors[1]

        b = utils.ssa_to_scalar(batch)
        q_idx = utils.ssa_to_scalar(m_idx)
        kv_idx = utils.ssa_to_scalar(n_idx)
        safe_q_idx = q_idx % seqlen_info.seqlen_q
        safe_kv_idx = kv_idx % seqlen_info.seqlen_k
        in_bounds = (q_idx < seqlen_info.seqlen_q) & (kv_idx < seqlen_info.seqlen_k)

        q_block = safe_q_idx // q_block_size
        k_block = safe_kv_idx // k_block_size
        q_local = safe_q_idx % q_block_size
        k_local = safe_kv_idx % k_block_size
        word_idx = k_local // 32
        bit_idx = k_local % 32

        block_id = block_id_table[b, k_block, q_block]
        mask_word = cutlass.Uint32(mask_words[block_id, q_local, word_idx])
        bit = utils.shr_u32(mask_word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)
        allowed = cutlass.Boolean(in_bounds & (bit != cutlass.Uint32(0)))
        return utils.scalar_to_ssa(allowed, cutlass.Boolean)

    return _hsa_backward_packed_mask


@lru_cache(maxsize=1)
def get_hsa_synthetic_packed_dense_mask_mod():
    """Return a CuTe mask_mod for synthetic packed dense buckets."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()

    @fast_sampling
    @cute.jit
    def _hsa_synthetic_packed_dense_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        q_length = aux_tensors[0]
        k_length = aux_tensors[1]

        b = utils.ssa_to_scalar(batch)
        q_idx = utils.ssa_to_scalar(m_idx)
        kv_idx = utils.ssa_to_scalar(n_idx)
        used_q = utils.scalar_to_ssa(q_length[b], cutlass.Int32)
        used_k = utils.scalar_to_ssa(k_length[b], cutlass.Int32)
        return (q_idx < used_q) & (kv_idx < used_k)

    return _hsa_synthetic_packed_dense_mask


@lru_cache(maxsize=None)
def get_hsa_synthetic_packed_bitmap_mask_mod(words_per_row: int):
    """Return a CuTe mask_mod for synthetic packed bitmap buckets."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()

    @fast_sampling
    @cute.jit
    def _hsa_synthetic_packed_bitmap_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        q_length = aux_tensors[0]
        k_length = aux_tensors[1]
        mask_words = aux_tensors[2]

        b = utils.ssa_to_scalar(batch)
        q_idx = utils.ssa_to_scalar(m_idx)
        kv_idx = utils.ssa_to_scalar(n_idx)
        used_q = utils.scalar_to_ssa(q_length[b], cutlass.Int32)
        used_k = utils.scalar_to_ssa(k_length[b], cutlass.Int32)
        in_bounds = (q_idx < used_q) & (kv_idx < used_k)
        safe_q_idx = q_idx % seqlen_info.seqlen_q
        safe_k_idx = kv_idx % seqlen_info.seqlen_k
        word_idx = safe_k_idx // 32
        bit_idx = safe_k_idx % 32
        word = cutlass.Uint32(mask_words[b, safe_q_idx, word_idx])
        bit = utils.shr_u32(word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)
        return in_bounds & (bit != cutlass.Uint32(0))

    return _hsa_synthetic_packed_bitmap_mask


@lru_cache(maxsize=1)
def get_hsa_panel_prefix_mask_mod():
    """Return a CuTe mask_mod for packed panel batches with per-row prefix lengths."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()

    @fast_sampling
    @cute.jit
    def _hsa_panel_prefix_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        prefix_len = aux_tensors[0]
        q_length = aux_tensors[1]
        k_length = aux_tensors[2]

        b = batch[0]
        q_idx = m_idx[0]
        kv_idx = n_idx[0]
        safe_q_idx = q_idx % seqlen_info.seqlen_q

        used_q = utils.scalar_to_ssa(q_length[b, safe_q_idx], cutlass.Int32)
        used_k = utils.scalar_to_ssa(k_length[b, safe_q_idx], cutlass.Int32)
        row_prefix = utils.scalar_to_ssa(prefix_len[b, safe_q_idx], cutlass.Int32)
        return (q_idx < used_q) & (kv_idx < used_k) & (kv_idx < row_prefix)

    return _hsa_panel_prefix_mask


@lru_cache(maxsize=1)
def get_hsa_mask_mod():
    """Return the CuTe mask_mod implementing nanochat's decoder-HDT attention."""
    cutlass, cute, utils, fast_sampling, _, _, _ = _lazy_cute_imports()

    @fast_sampling
    @cute.jit
    def _hsa_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        keep_ids = aux_tensors[0]
        hash_ids = aux_tensors[1]

        b = batch[0]
        q_idx = m_idx[0]
        kv_idx = n_idx[0]
        safe_q_idx = q_idx % seqlen_info.seqlen_q
        safe_kv_idx = kv_idx % seqlen_info.seqlen_k
        in_bounds = (q_idx < seqlen_info.seqlen_q) & (kv_idx < seqlen_info.seqlen_k)

        q_ki0 = utils.scalar_to_ssa(keep_ids[b, 0, safe_q_idx], cutlass.Int32)
        q_ki1 = utils.scalar_to_ssa(keep_ids[b, 1, safe_q_idx], cutlass.Int32)
        q_ki2 = utils.scalar_to_ssa(keep_ids[b, 2, safe_q_idx], cutlass.Int32)
        k_ki0 = utils.scalar_to_ssa(keep_ids[b, 0, safe_kv_idx], cutlass.Int32)
        k_ki1 = utils.scalar_to_ssa(keep_ids[b, 1, safe_kv_idx], cutlass.Int32)
        k_ki2 = utils.scalar_to_ssa(keep_ids[b, 2, safe_kv_idx], cutlass.Int32)

        q_h0 = utils.scalar_to_ssa(hash_ids[b, 0, safe_q_idx], cutlass.Int32)
        q_h1 = utils.scalar_to_ssa(hash_ids[b, 1, safe_q_idx], cutlass.Int32)
        q_h2 = utils.scalar_to_ssa(hash_ids[b, 2, safe_q_idx], cutlass.Int32)
        k_h0 = utils.scalar_to_ssa(hash_ids[b, 0, safe_kv_idx], cutlass.Int32)
        k_h1 = utils.scalar_to_ssa(hash_ids[b, 1, safe_kv_idx], cutlass.Int32)
        k_h2 = utils.scalar_to_ssa(hash_ids[b, 2, safe_kv_idx], cutlass.Int32)

        same_sent = q_h0 == k_h0
        same_sec = q_h1 == k_h1
        same_doc = q_h2 == k_h2
        prev_sent = q_h0 == (k_h0 + 1)
        prev_sec = q_h1 == (k_h1 + 1)

        is_body_q = (q_ki0 != 0) & (q_ki1 == 0)
        is_sen_q = (q_ki0 != 0) & (q_ki1 != 0) & (q_ki2 == 0)
        is_sen_k = (k_ki0 != 0) & (k_ki1 != 0) & (k_ki2 == 0)
        is_sec_k = (k_ki0 == 0) & (k_ki1 != 0) & (k_ki2 != 0)

        level0 = (
            (q_ki0 != 0)
            & (k_ki0 != 0)
            & same_sent
            & ((q_ki1 == 0) | (k_ki1 == 0))
        )
        level1 = (q_ki1 != 0) & (k_ki1 != 0) & same_sec
        level2 = (q_ki2 != 0) & (k_ki2 != 0) & same_doc
        cross_body_sen = is_body_q & is_sen_k & prev_sent & same_sec
        cross_body_sec = is_body_q & is_sec_k & prev_sec & same_doc
        cross_sen_sec = is_sen_q & is_sec_k & prev_sec & same_doc
        causal = kv_idx <= q_idx
        return in_bounds & causal & (level0 | level1 | level2 | cross_body_sen | cross_body_sec | cross_sen_sec)

    return _hsa_mask


def compute_hsa_mask(keep_ids: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
    """Return the additive float mask used by decoder-HDT attention."""
    keep_ids = _ensure_int32(keep_ids)
    hash_ids = _ensure_int32(hash_ids)
    bsz, _, seqlen = keep_ids.shape
    device = keep_ids.device

    ki0 = keep_ids[:, 0].bool()
    ki1 = keep_ids[:, 1].bool()
    ki2 = keep_ids[:, 2].bool()

    h0 = hash_ids[:, 0]
    h1 = hash_ids[:, 1]
    h2 = hash_ids[:, 2]

    pos = torch.arange(seqlen, device=device)
    causal = pos[None, :, None] >= pos[None, None, :]

    same_sent = h0.unsqueeze(2) == h0.unsqueeze(1)
    same_sec = h1.unsqueeze(2) == h1.unsqueeze(1)
    same_doc = h2.unsqueeze(2) == h2.unsqueeze(1)

    cls_mask = (
        ki0.unsqueeze(2)
        & ki1.unsqueeze(2)
        & ki0.unsqueeze(1)
        & ki1.unsqueeze(1)
    )

    level0 = ki0.unsqueeze(2) & ki0.unsqueeze(1) & same_sent & ~cls_mask
    level1 = ki1.unsqueeze(2) & ki1.unsqueeze(1) & same_sec
    level2 = ki2.unsqueeze(2) & ki2.unsqueeze(1) & same_doc

    is_body_q = ki0.unsqueeze(2) & ~ki1.unsqueeze(2)
    is_sen_q = ki0.unsqueeze(2) & ki1.unsqueeze(2) & ~ki2.unsqueeze(2)
    is_sen_k = ki0.unsqueeze(1) & ki1.unsqueeze(1) & ~ki2.unsqueeze(1)
    is_sec_k = ~ki0.unsqueeze(1) & ki1.unsqueeze(1) & ki2.unsqueeze(1)
    prev_sent = h0.unsqueeze(2) == (h0.unsqueeze(1) + 1)
    prev_sec = h1.unsqueeze(2) == (h1.unsqueeze(1) + 1)
    cross_body_sen = is_body_q & is_sen_k & prev_sent & same_sec
    cross_body_sec = is_body_q & is_sec_k & prev_sec & same_doc
    cross_sen_sec = is_sen_q & is_sec_k & prev_sec & same_doc

    attend = causal & (level0 | level1 | level2 | cross_body_sen | cross_body_sec | cross_sen_sec)

    mask = torch.zeros(bsz, seqlen, seqlen, device=device, dtype=torch.float32)
    mask.masked_fill_(~attend, float("-inf"))
    return mask


def hsa_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Explicit masked-softmax reference for HSA attention."""
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    if q_ref.size(1) != k_ref.size(1):
        repeat_factor = q_ref.size(1) // k_ref.size(1)
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=1)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=1)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    scores = scores + compute_hsa_mask(keep_ids, hash_ids).unsqueeze(1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v_ref)
    return out.transpose(1, 2).to(dtype=q.dtype)


def hsa_sparse_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule: HSASchedule,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
):
    """Reference attention using the exact schedule-expanded mask."""
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    if q_ref.size(1) != k_ref.size(1):
        repeat_factor = q_ref.size(1) // k_ref.size(1)
        k_ref = k_ref.repeat_interleave(repeat_factor, dim=1)
        v_ref = v_ref.repeat_interleave(repeat_factor, dim=1)

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q_ref, k_ref.transpose(-2, -1)) * scale
    attend = schedule_to_attend_mask(schedule)
    additive_mask = torch.zeros_like(attend, dtype=torch.float32)
    additive_mask.masked_fill_(~attend, float("-inf"))
    scores = scores + additive_mask.unsqueeze(1)
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, v_ref).transpose(1, 2).to(dtype=q.dtype)
    if not return_lse:
        return out
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _expand_kv_to_q_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    if x.shape[1] == num_q_heads:
        return x
    repeat_factor = num_q_heads // x.shape[1]
    return x.repeat_interleave(repeat_factor, dim=1)


def _collapse_q_to_kv_heads(x: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    if x.shape[1] == num_kv_heads:
        return x
    repeat_factor = x.shape[1] // num_kv_heads
    return x.view(x.shape[0], num_kv_heads, repeat_factor, x.shape[-1]).sum(dim=2)


def _run_varlen_fa4_stream(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    stream: HSAStreamPack,
    softmax_scale: float,
):
    if stream.is_empty:
        num_q_heads = q_flat.shape[1]
        head_dim_v = v_flat.shape[-1]
        empty_out = torch.empty(0, num_q_heads, head_dim_v, dtype=q_flat.dtype, device=q_flat.device)
        empty_lse = torch.empty(num_q_heads, 0, dtype=torch.float32, device=q_flat.device)
        return empty_out, empty_lse

    _, _, _, _, _, flash_attn_fwd, _ = _lazy_cute_imports()
    q_stream = q_flat.index_select(0, stream.query_indices.long())
    k_stream = k_flat.index_select(0, stream.key_indices.long())
    v_stream = v_flat.index_select(0, stream.key_indices.long())
    return flash_attn_fwd(
        q_stream,
        k_stream,
        v_stream,
        cu_seqlens_q=stream.cu_seqlens_q,
        cu_seqlens_k=stream.cu_seqlens_k,
        max_seqlen_q=stream.max_seqlen_q,
        max_seqlen_k=stream.max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=True,
        return_lse=True,
    )


def _combine_stream(
    total_out: torch.Tensor,
    total_lse: torch.Tensor,
    row_indices: torch.Tensor,
    stream_out: torch.Tensor,
    stream_lse: torch.Tensor,
):
    if row_indices.numel() == 0:
        return
    row_idx = row_indices.long()
    stream_lse_t = stream_lse.transpose(0, 1).contiguous()
    prev_out = total_out.index_select(0, row_idx)
    prev_lse = total_lse.index_select(0, row_idx)
    next_lse = torch.logaddexp(prev_lse, stream_lse_t)
    prev_weight = torch.exp(prev_lse - next_lse)
    stream_weight = torch.exp(stream_lse_t - next_lse)
    combined_out = prev_out * prev_weight.unsqueeze(-1) + stream_out.float() * stream_weight.unsqueeze(-1)
    total_out.index_copy_(0, row_idx, combined_out)
    total_lse.index_copy_(0, row_idx, next_lse)


def _combine_rows(
    total_out: torch.Tensor,
    total_lse: torch.Tensor,
    row_indices: torch.Tensor,
    row_out: torch.Tensor,
    row_lse: torch.Tensor,
):
    if row_indices.numel() == 0:
        return
    row_idx = row_indices.long()
    if torch.unique(row_idx).numel() == row_idx.numel():
        prev_out = total_out.index_select(0, row_idx)
        prev_lse = total_lse.index_select(0, row_idx)
        next_lse = torch.logaddexp(prev_lse, row_lse)
        prev_weight = torch.exp(prev_lse - next_lse)
        row_weight = torch.exp(row_lse - next_lse)
        combined_out = prev_out * prev_weight.unsqueeze(-1) + row_out.float() * row_weight.unsqueeze(-1)
        total_out.index_copy_(0, row_idx, combined_out)
        total_lse.index_copy_(0, row_idx, next_lse)
        return

    unique_rows, inverse = torch.unique(row_idx, sorted=False, return_inverse=True)
    combined_out = total_out.index_select(0, unique_rows)
    combined_lse = total_lse.index_select(0, unique_rows)
    row_out_float = row_out.float()
    for contrib_idx in range(row_idx.numel()):
        slot = int(inverse[contrib_idx].item())
        next_lse = torch.logaddexp(combined_lse[slot], row_lse[contrib_idx])
        prev_weight = torch.exp(combined_lse[slot] - next_lse)
        row_weight = torch.exp(row_lse[contrib_idx] - next_lse)
        combined_out[slot] = (
            combined_out[slot] * prev_weight.unsqueeze(-1)
            + row_out_float[contrib_idx] * row_weight.unsqueeze(-1)
        )
        combined_lse[slot] = next_lse
    total_out.index_copy_(0, unique_rows, combined_out)
    total_lse.index_copy_(0, unique_rows, combined_lse)


def _run_hsa_forward_batch(
    q_sel: torch.Tensor,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count, q_len = q_sel.shape[0], q_sel.shape[1]
    k_len = k_sel.shape[1]
    q_float = q_sel.float().contiguous()
    k_expanded = _expand_kv_to_q_heads(k_sel.reshape(-1, k_sel.shape[2], k_sel.shape[3]).float(), q_sel.shape[2])
    k_expanded = k_expanded.view(batch_count, k_len, q_sel.shape[2], k_sel.shape[3]).contiguous()
    v_expanded = _expand_kv_to_q_heads(v_sel.reshape(-1, v_sel.shape[2], v_sel.shape[3]).float(), q_sel.shape[2])
    v_expanded = v_expanded.view(batch_count, k_len, q_sel.shape[2], v_sel.shape[3]).contiguous()

    q_hqd = q_float.permute(0, 2, 1, 3).reshape(batch_count * q_sel.shape[2], q_len, q_sel.shape[3]).contiguous()
    k_hkd = k_expanded.permute(0, 2, 1, 3).reshape(batch_count * q_sel.shape[2], k_len, k_sel.shape[3]).contiguous()
    v_hkd = v_expanded.permute(0, 2, 1, 3).reshape(batch_count * q_sel.shape[2], k_len, v_sel.shape[3]).contiguous()

    scores = torch.bmm(q_hqd, k_hkd.transpose(1, 2)) * softmax_scale
    valid_expanded = mask.unsqueeze(1).expand(batch_count, q_sel.shape[2], q_len, k_len).reshape(
        batch_count * q_sel.shape[2], q_len, k_len
    )
    scores = scores.masked_fill(~valid_expanded, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.bmm(probs, v_hkd)
    lse = torch.logsumexp(scores, dim=-1)
    out = out.view(batch_count, q_sel.shape[2], q_len, v_sel.shape[3]).permute(0, 2, 1, 3).contiguous()
    lse = lse.view(batch_count, q_sel.shape[2], q_len).permute(0, 2, 1).contiguous()
    return out, lse


def _self_stream_forward(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    row_indices: torch.Tensor,
    softmax_scale: float,
):
    if row_indices.numel() == 0:
        num_q_heads = q_flat.shape[1]
        head_dim_v = v_flat.shape[-1]
        empty_out = torch.empty(0, num_q_heads, head_dim_v, dtype=q_flat.dtype, device=q_flat.device)
        empty_lse = torch.empty(num_q_heads, 0, dtype=torch.float32, device=q_flat.device)
        return empty_out, empty_lse
    row_idx = row_indices.long()
    q_self = q_flat.index_select(0, row_idx).float()
    k_self = _expand_kv_to_q_heads(k_flat.index_select(0, row_idx).float(), q_self.shape[1])
    v_self = _expand_kv_to_q_heads(v_flat.index_select(0, row_idx), q_self.shape[1])
    lse = (q_self * k_self).sum(dim=-1) * softmax_scale
    return v_self.contiguous(), lse.transpose(0, 1).contiguous()


def _run_hsa_fa4_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule: HSASchedule,
    softmax_scale: float,
):
    bsz, seqlen, num_q_heads, _ = q.shape
    head_dim_v = v.shape[-1]
    total_rows = bsz * seqlen

    q_flat = q.reshape(total_rows, num_q_heads, q.shape[-1])
    k_flat = k.reshape(total_rows, k.shape[-2], k.shape[-1])
    v_flat = v.reshape(total_rows, v.shape[-2], head_dim_v)

    total_out = torch.zeros(total_rows, num_q_heads, head_dim_v, dtype=torch.float32, device=q.device)
    total_lse = torch.full((total_rows, num_q_heads), float("-inf"), dtype=torch.float32, device=q.device)

    sentence_out, sentence_lse = _run_varlen_fa4_stream(
        q_flat,
        k_flat,
        v_flat,
        schedule.sentence_stream,
        softmax_scale,
    )
    _combine_stream(total_out, total_lse, schedule.sentence_stream.row_indices, sentence_out, sentence_lse)

    section_prefix_out, section_prefix_lse = _run_varlen_fa4_stream(
        q_flat,
        k_flat,
        v_flat,
        schedule.section_prefix_stream,
        softmax_scale,
    )
    _combine_stream(
        total_out,
        total_lse,
        schedule.section_prefix_stream.row_indices,
        section_prefix_out,
        section_prefix_lse,
    )

    section_self_out, section_self_lse = _self_stream_forward(
        q_flat,
        k_flat,
        v_flat,
        schedule.section_self_indices,
        softmax_scale,
    )
    _combine_stream(total_out, total_lse, schedule.section_self_indices, section_self_out, section_self_lse)

    document_prefix_out, document_prefix_lse = _run_varlen_fa4_stream(
        q_flat,
        k_flat,
        v_flat,
        schedule.document_prefix_stream,
        softmax_scale,
    )
    _combine_stream(
        total_out,
        total_lse,
        schedule.document_prefix_stream.row_indices,
        document_prefix_out,
        document_prefix_lse,
    )

    document_self_out, document_self_lse = _self_stream_forward(
        q_flat,
        k_flat,
        v_flat,
        schedule.document_self_indices,
        softmax_scale,
    )
    _combine_stream(total_out, total_lse, schedule.document_self_indices, document_self_out, document_self_lse)

    out = total_out.to(dtype=q.dtype).view(bsz, seqlen, num_q_heads, head_dim_v)
    lse = total_lse.view(bsz, seqlen, num_q_heads).permute(0, 2, 1).contiguous()
    return out, lse, sentence_lse, section_prefix_lse, document_prefix_lse


def _run_hsa_fused_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule: HSASchedule,
    softmax_scale: float,
):
    fused_schedule = _get_hsa_fused_forward_schedule(schedule, q, k)
    sentence_batches, anchor_batches = _get_hsa_fused_forward_batches(schedule, fused_schedule)
    bsz, seqlen, num_q_heads, _ = q.shape
    head_dim_v = v.shape[-1]
    total_rows = schedule.num_rows

    q_flat = q.reshape(total_rows, num_q_heads, q.shape[-1])
    k_flat = k.reshape(total_rows, k.shape[-2], k.shape[-1])
    v_flat = v.reshape(total_rows, v.shape[-2], head_dim_v)

    total_out = torch.zeros(total_rows, num_q_heads, head_dim_v, dtype=torch.float32, device=q.device)
    total_lse = torch.full((total_rows, num_q_heads), float("-inf"), dtype=torch.float32, device=q.device)

    for batch in sentence_batches:
        q_indices = batch.q_indices.long()
        k_indices = batch.k_indices.long()
        q_sel = q_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], q.shape[2], q.shape[3]
        )
        k_sel = k_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], k.shape[2], k.shape[3]
        )
        v_sel = v_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], v.shape[2], v.shape[3]
        )
        q_offsets = torch.arange(q_indices.shape[1], device=q.device, dtype=torch.int32).view(1, -1, 1)
        k_offsets = torch.arange(k_indices.shape[1], device=q.device, dtype=torch.int32).view(1, 1, -1)
        q_valid = q_offsets.squeeze(-1) < batch.q_length.unsqueeze(1)
        k_valid = k_offsets.squeeze(1) < batch.k_length.unsqueeze(1)
        mask = q_valid.unsqueeze(-1) & k_valid.unsqueeze(1) & (
            k_offsets <= (q_offsets + batch.q_offset_start.view(-1, 1, 1))
        )
        out_batch, lse_batch = _run_hsa_forward_batch(q_sel, k_sel, v_sel, mask, softmax_scale)
        _combine_rows(total_out, total_lse, q_indices[q_valid], out_batch[q_valid], lse_batch[q_valid])

    for batch in anchor_batches:
        q_indices = batch.q_indices.long()
        k_indices = batch.k_indices.long()
        q_sel = q_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], q.shape[2], q.shape[3]
        )
        k_sel = k_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], k.shape[2], k.shape[3]
        )
        v_sel = v_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], v.shape[2], v.shape[3]
        )
        q_offsets = torch.arange(q_indices.shape[1], device=q.device, dtype=torch.int32).view(1, -1, 1)
        k_offsets = torch.arange(k_indices.shape[1], device=q.device, dtype=torch.int32).view(1, 1, -1)
        q_valid = q_offsets.squeeze(-1) < batch.q_length.unsqueeze(1)
        k_valid = k_offsets.squeeze(1) < batch.k_length.unsqueeze(1)
        mask = q_valid.unsqueeze(-1) & k_valid.unsqueeze(1) & (
            k_offsets < batch.prefix_len.unsqueeze(-1)
        )
        out_batch, lse_batch = _run_hsa_forward_batch(q_sel, k_sel, v_sel, mask, softmax_scale)
        _combine_rows(total_out, total_lse, q_indices[q_valid], out_batch[q_valid], lse_batch[q_valid])

    out = total_out.to(dtype=q.dtype).view(bsz, seqlen, num_q_heads, head_dim_v)
    lse = total_lse.view(bsz, seqlen, num_q_heads).permute(0, 2, 1).contiguous()
    return out, lse


def _stream_weights(total_lse_flat: torch.Tensor, row_indices: torch.Tensor, stream_lse: torch.Tensor) -> torch.Tensor:
    if row_indices.numel() == 0:
        return torch.empty(0, total_lse_flat.shape[-1], dtype=torch.float32, device=total_lse_flat.device)
    row_idx = row_indices.long()
    stream_lse_t = stream_lse.transpose(0, 1).contiguous()
    return torch.exp(stream_lse_t - total_lse_flat.index_select(0, row_idx))


def _run_varlen_fa4_bwd(
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_total_flat: torch.Tensor,
    dout_weighted_flat: torch.Tensor,
    stream_lse: torch.Tensor,
    stream: HSAStreamPack,
    softmax_scale: float,
    deterministic: bool,
):
    if stream.is_empty:
        empty_q = torch.empty(0, q_flat.shape[1], q_flat.shape[-1], dtype=q_flat.dtype, device=q_flat.device)
        empty_k = torch.empty(0, k_flat.shape[1], k_flat.shape[-1], dtype=k_flat.dtype, device=k_flat.device)
        empty_v = torch.empty(0, v_flat.shape[1], v_flat.shape[-1], dtype=v_flat.dtype, device=v_flat.device)
        return empty_q, empty_k, empty_v

    _, _, _, _, _, _, flash_attn_bwd = _lazy_cute_imports()
    q_stream = q_flat.index_select(0, stream.query_indices.long())
    k_stream = k_flat.index_select(0, stream.key_indices.long())
    v_stream = v_flat.index_select(0, stream.key_indices.long())
    out_stream = out_total_flat.index_select(0, stream.row_indices.long()).to(dtype=q_flat.dtype)
    dout_stream = dout_weighted_flat.index_select(0, stream.row_indices.long()).to(dtype=q_flat.dtype)
    return flash_attn_bwd(
        q_stream,
        k_stream,
        v_stream,
        out_stream,
        dout_stream,
        stream_lse,
        softmax_scale=softmax_scale,
        causal=True,
        cu_seqlens_q=stream.cu_seqlens_q,
        cu_seqlens_k=stream.cu_seqlens_k,
        max_seqlen_q=stream.max_seqlen_q,
        max_seqlen_k=stream.max_seqlen_k,
        deterministic=deterministic,
    )


def _accumulate_self_stream_grads(
    dq_acc: torch.Tensor,
    dk_acc: torch.Tensor,
    dv_acc: torch.Tensor,
    q_flat: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    out_total_flat: torch.Tensor,
    dout_flat: torch.Tensor,
    total_lse_flat: torch.Tensor,
    row_indices: torch.Tensor,
    softmax_scale: float,
):
    if row_indices.numel() == 0:
        return
    row_idx = row_indices.long()
    q_sel = q_flat.index_select(0, row_idx).float()
    k_sel = k_flat.index_select(0, row_idx).float()
    v_sel = v_flat.index_select(0, row_idx)
    expanded_k = _expand_kv_to_q_heads(k_sel, q_sel.shape[1])
    expanded_v = _expand_kv_to_q_heads(v_sel.float(), q_sel.shape[1])
    self_lse = (q_sel * expanded_k).sum(dim=-1) * softmax_scale
    weights = torch.exp(self_lse - total_lse_flat.index_select(0, row_idx))
    dout_weighted = weights.unsqueeze(-1) * dout_flat.index_select(0, row_idx)
    out_sel = out_total_flat.index_select(0, row_idx)

    ds = (dout_weighted * (expanded_v - out_sel)).sum(dim=-1)
    dq_part = softmax_scale * ds.unsqueeze(-1) * expanded_k
    dk_part = softmax_scale * ds.unsqueeze(-1) * q_sel
    dv_part = dout_weighted

    dq_acc.index_add_(0, row_idx, dq_part)
    dk_acc.index_add_(0, row_idx, _collapse_q_to_kv_heads(dk_part, k_sel.shape[1]))
    dv_acc.index_add_(0, row_idx, _collapse_q_to_kv_heads(dv_part, v_sel.shape[1]))


def _run_hsa_backward_panel(
    q_sel: torch.Tensor,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    out_sel: torch.Tensor,
    dout_sel: torch.Tensor,
    lse_sel: torch.Tensor,
    softmax_scale: float,
    *,
    prefix_len: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = _run_hsa_backward_panel_batched(
        q_sel.unsqueeze(0),
        k_sel.unsqueeze(0),
        v_sel.unsqueeze(0),
        out_sel.unsqueeze(0),
        dout_sel.unsqueeze(0),
        lse_sel.unsqueeze(0),
        softmax_scale,
        prefix_len=None if prefix_len is None else prefix_len.unsqueeze(0),
        mask=None if mask is None else mask.unsqueeze(0),
    )
    return dq[0], dk[0], dv[0]


def _run_hsa_backward_panel_batched(
    q_sel: torch.Tensor,
    k_sel: torch.Tensor,
    v_sel: torch.Tensor,
    out_sel: torch.Tensor,
    dout_sel: torch.Tensor,
    lse_sel: torch.Tensor,
    softmax_scale: float,
    *,
    prefix_len: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_count, q_len = q_sel.shape[0], q_sel.shape[1]
    k_len = k_sel.shape[1]
    q_float = q_sel.float().contiguous()
    k_expanded = _expand_kv_to_q_heads(k_sel.reshape(-1, k_sel.shape[2], k_sel.shape[3]).float(), q_sel.shape[2])
    k_expanded = k_expanded.view(batch_count, k_len, q_sel.shape[2], k_sel.shape[3]).contiguous()
    v_expanded = _expand_kv_to_q_heads(v_sel.reshape(-1, v_sel.shape[2], v_sel.shape[3]).float(), q_sel.shape[2])
    v_expanded = v_expanded.view(batch_count, k_len, q_sel.shape[2], v_sel.shape[3]).contiguous()
    out_float = out_sel.float().contiguous()
    dout_float = dout_sel.float().contiguous()

    num_q_heads = q_sel.shape[2]
    q_hqd = q_float.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, q_len, q_sel.shape[3]).contiguous()
    k_hkd = k_expanded.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, k_len, k_sel.shape[3]).contiguous()
    v_hkd = v_expanded.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, k_len, v_sel.shape[3]).contiguous()

    scores = torch.bmm(q_hqd, k_hkd.transpose(1, 2)) * softmax_scale
    if prefix_len is not None:
        valid = torch.arange(k_len, device=q_sel.device).view(1, 1, k_len) < prefix_len.unsqueeze(-1)
    else:
        assert mask is not None
        valid = mask
    valid_expanded = valid.unsqueeze(1).expand(batch_count, num_q_heads, q_len, k_len).reshape(
        batch_count * num_q_heads, q_len, k_len
    )
    scores = scores.masked_fill(~valid_expanded, float("-inf"))

    lse_expanded = lse_sel.permute(0, 2, 1).reshape(batch_count * num_q_heads, q_len, 1)
    probs = torch.exp(scores - lse_expanded)
    probs = probs * valid_expanded

    dout_hqd = dout_float.permute(0, 2, 1, 3).reshape(batch_count * num_q_heads, q_len, dout_sel.shape[3]).contiguous()
    dprob = torch.bmm(dout_hqd, v_hkd.transpose(1, 2))
    delta = (out_float * dout_float).sum(dim=-1).permute(0, 2, 1).reshape(batch_count * num_q_heads, q_len, 1)
    dscores = probs * (dprob - delta)

    dq = torch.bmm(dscores, k_hkd).view(batch_count, num_q_heads, q_len, q_sel.shape[3]).permute(0, 2, 1, 3)
    dq = dq.contiguous() * softmax_scale
    dk_expanded = torch.bmm(dscores.transpose(1, 2), q_hqd)
    dk_expanded = dk_expanded.view(batch_count, num_q_heads, k_len, k_sel.shape[3]).permute(0, 2, 1, 3).contiguous()
    dk_expanded = dk_expanded * softmax_scale
    dv_expanded = torch.bmm(probs.transpose(1, 2), dout_hqd)
    dv_expanded = dv_expanded.view(batch_count, num_q_heads, k_len, v_sel.shape[3]).permute(0, 2, 1, 3).contiguous()
    dk = _collapse_q_to_kv_heads(
        dk_expanded.view(batch_count * k_len, num_q_heads, k_sel.shape[3]),
        k_sel.shape[2],
    ).view(batch_count, k_len, k_sel.shape[2], k_sel.shape[3])
    dv = _collapse_q_to_kv_heads(
        dv_expanded.view(batch_count * k_len, num_q_heads, v_sel.shape[3]),
        v_sel.shape[2],
    ).view(batch_count, k_len, v_sel.shape[2], v_sel.shape[3])
    return dq, dk, dv


def _run_hsa_hybrid_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule: HSASchedule,
    softmax_scale: float,
):
    hybrid_schedule = _get_hsa_hybrid_backward_schedule(schedule)
    sentence_batches, anchor_batches = _get_hsa_hybrid_backward_batches(schedule, hybrid_schedule)
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    total_rows = schedule.num_rows
    head_dim = q.shape[-1]
    head_dim_v = v.shape[-1]

    q_flat = q.reshape(total_rows, q.shape[2], head_dim)
    k_flat = k.reshape(total_rows, k.shape[2], head_dim)
    v_flat = v.reshape(total_rows, v.shape[2], head_dim_v)
    out_flat = out.reshape(total_rows, out.shape[2], out.shape[3]).float()
    dout_flat = dout.reshape(total_rows, dout.shape[2], dout.shape[3]).float()
    lse_flat = lse.permute(0, 2, 1).contiguous().view(total_rows, q.shape[2]).float()

    dq_acc = torch.zeros_like(q_flat, dtype=torch.float32)
    dk_acc = torch.zeros_like(k_flat, dtype=torch.float32)
    dv_acc = torch.zeros_like(v_flat, dtype=torch.float32)

    for batch in sentence_batches:
        q_indices = batch.q_indices.long()
        k_indices = batch.k_indices.long()
        q_sel = q_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], q.shape[2], head_dim
        )
        k_sel = k_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], k.shape[2], head_dim
        )
        v_sel = v_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], v.shape[2], head_dim_v
        )
        out_sel = out_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], out.shape[2], out.shape[3]
        )
        dout_sel = dout_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], dout.shape[2], dout.shape[3]
        )
        lse_sel = lse_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], q.shape[2]
        )
        q_offsets = torch.arange(q_indices.shape[1], device=q.device, dtype=torch.int32).view(1, -1, 1)
        k_offsets = torch.arange(k_indices.shape[1], device=q.device, dtype=torch.int32).view(1, 1, -1)
        q_valid = q_offsets.squeeze(-1) < batch.q_length.unsqueeze(1)
        k_valid = k_offsets.squeeze(1) < batch.k_length.unsqueeze(1)
        mask = (q_offsets >= k_offsets) & q_valid.unsqueeze(-1) & k_valid.unsqueeze(1)
        dq, dk, dv = _run_hsa_backward_panel_batched(
            q_sel,
            k_sel,
            v_sel,
            out_sel,
            dout_sel,
            lse_sel,
            softmax_scale,
            mask=mask,
        )
        dq_acc.index_add_(0, q_indices[q_valid].long(), dq[q_valid])
        dk_acc.index_add_(0, k_indices[k_valid].long(), dk[k_valid])
        dv_acc.index_add_(0, k_indices[k_valid].long(), dv[k_valid])

    for batch in anchor_batches:
        q_indices = batch.q_indices.long()
        k_indices = batch.k_indices.long()
        q_sel = q_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], q.shape[2], head_dim
        )
        k_sel = k_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], k.shape[2], head_dim
        )
        v_sel = v_flat.index_select(0, k_indices.reshape(-1)).view(
            k_indices.shape[0], k_indices.shape[1], v.shape[2], head_dim_v
        )
        out_sel = out_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], out.shape[2], out.shape[3]
        )
        dout_sel = dout_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], dout.shape[2], dout.shape[3]
        )
        lse_sel = lse_flat.index_select(0, q_indices.reshape(-1)).view(
            q_indices.shape[0], q_indices.shape[1], q.shape[2]
        )
        q_offsets = torch.arange(q_indices.shape[1], device=q.device, dtype=torch.int32).view(1, -1, 1)
        k_offsets = torch.arange(k_indices.shape[1], device=q.device, dtype=torch.int32).view(1, 1, -1)
        q_valid = q_offsets.squeeze(-1) < batch.q_length.unsqueeze(1)
        k_valid = k_offsets.squeeze(1) < batch.k_length.unsqueeze(1)
        mask = q_valid.unsqueeze(-1) & k_valid.unsqueeze(1) & (
            k_offsets < batch.prefix_len.unsqueeze(-1)
        )
        dq, dk, dv = _run_hsa_backward_panel_batched(
            q_sel,
            k_sel,
            v_sel,
            out_sel,
            dout_sel,
            lse_sel,
            softmax_scale,
            mask=mask,
        )
        dq_acc.index_add_(0, q_indices[q_valid].long(), dq[q_valid])
        dk_acc.index_add_(0, k_indices[k_valid].long(), dk[k_valid])
        dv_acc.index_add_(0, k_indices[k_valid].long(), dv[k_valid])

    return (
        dq_acc.view_as(q).to(dtype=q.dtype),
        dk_acc.view_as(k).to(dtype=k.dtype),
        dv_acc.view_as(v).to(dtype=v.dtype),
    )


def _run_hsa_blocksparse_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    schedule: HSASchedule,
    softmax_scale: float,
):
    _, _, _, _, _, flash_attn_fwd, _ = _lazy_cute_imports()
    runtime = _get_hsa_block_sparse_runtime(schedule, q, k)
    out, lse = flash_attn_fwd(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        pack_gqa=False,
        mask_mod=get_hsa_forward_tile_mask_mod(runtime.forward_tile_masks.q_block_size, runtime.forward_tile_masks.k_block_size),
        aux_tensors=runtime.forward_aux_tensors,
        block_sparse_tensors=runtime.forward_sparse_torch,
        return_lse=True,
    )
    return out, lse


def _run_hsa_packed_mask_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    schedule: HSASchedule,
    softmax_scale: float,
    deterministic: bool,
    keep_ids: Optional[torch.Tensor] = None,
    hash_ids: Optional[torch.Tensor] = None,
    runtime: Optional[HSABlockSparseRuntime] = None,
):
    _, _, _, _, _, _, flash_attn_bwd = _lazy_cute_imports()
    runtime = runtime if runtime is not None else _get_hsa_block_sparse_runtime(schedule, q, k, require_backward=True)
    if runtime.backward_sparse is None or runtime.backward_packed_masks is None or runtime.backward_sparse_torch is None:
        runtime = _get_hsa_block_sparse_runtime(schedule, q, k, require_backward=True)
    block_sparse_tensors = runtime.backward_sparse_torch
    if _should_disable_hsa_block_sparse_backward(schedule, runtime):
        block_sparse_tensors = None
    return flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale=softmax_scale,
        causal=False,
        pack_gqa=False,
        deterministic=deterministic,
        m_block_size=runtime.backward_block_q,
        n_block_size=runtime.backward_block_k,
        mask_mod=get_hsa_backward_packed_mask_mod(
            runtime.backward_packed_masks.q_block_size,
            runtime.backward_packed_masks.k_block_size,
        ),
        aux_tensors=runtime.backward_aux_tensors,
        block_sparse_tensors=block_sparse_tensors,
        subtile_factor_override=runtime.backward_subtile_factor,
    )


def _zero_hsa_grads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)


def _has_zero_upstream_grad(dout: torch.Tensor) -> bool:
    # The sparse kernels are not needed when the caller provides an exact zero
    # upstream gradient. Returning exact zeros here avoids launching the sparse
    # backward on a case where the true result is known.
    return not bool(torch.any(dout).item())


def _should_disable_hsa_block_sparse_backward(
    schedule: HSASchedule,
    runtime: HSABlockSparseRuntime,
) -> bool:
    """Dense traversal is more reliable when the backward work collapses to one tile.

    In the single-tile case block sparsity prunes nothing, but the live SM100
    sparse traversal path shows a dQ mismatch relative to the dense mask-mod
    traversal. Prefer the dense traversal there until the generic sparse kernel
    path is fixed.
    """
    sparse = runtime.backward_sparse
    if sparse is None:
        return True
    q_block_size, k_block_size = sparse.block_size
    num_q_blocks = (schedule.seqlen + q_block_size - 1) // q_block_size
    num_k_blocks = (schedule.seqlen + k_block_size - 1) // k_block_size
    return num_q_blocks <= 1 and num_k_blocks <= 1


def _schedule_has_only_sentence_backward_families(schedule: HSASchedule) -> bool:
    monolithic_schedule = _get_hsa_monolithic_backward_schedule(schedule)
    return monolithic_schedule.anchor_full_q_row_start.numel() == 0 and monolithic_schedule.anchor_tail_q_row_start.numel() == 0


def _get_hsa_blocksparse_backward_mode(schedule: HSASchedule) -> str:
    if _use_hsa_runtime_forward_only():
        return "sparse_mask"
    if (
        os.environ.get("FLASH_ATTN_HSA_USE_MONOLITHIC_BWD", "0") == "1"
        and _schedule_has_only_sentence_backward_families(schedule)
    ):
        return "monolithic_sentence"
    if (
        os.environ.get("FLASH_ATTN_HSA_USE_PACKED_BWD", "0") == "1"
        or os.environ.get("FLASH_ATTN_HSA_USE_HYBRID_BWD", "0") == "1"
    ):
        return "legacy_packed"
    return "sparse_mask"


def _allow_hsa_true_fused_bwd_for_call(*, use_synthetic_grid: bool) -> bool:
    if os.environ.get("FLASH_ATTN_HSA_USE_TRUE_FUSED_BWD", "0") != "1":
        return False
    direct_micro_preset_active = (
        os.environ.get("FLASH_ATTN_HSA_USE_SYNTHETIC_GRID", "0") == "1"
        and os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_FWD", "0") == "1"
        and os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0") != "1"
    )
    if direct_micro_preset_active and not use_synthetic_grid:
        return False
    return True


def _run_hsa_blocksparse_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    sentence_lse: Optional[torch.Tensor],
    sentence_q_stream: Optional[torch.Tensor],
    sentence_k_stream: Optional[torch.Tensor],
    sentence_v_stream: Optional[torch.Tensor],
    sentence_out_stream: Optional[torch.Tensor],
    schedule: HSASchedule,
    softmax_scale: float,
    deterministic: bool,
    keep_ids: Optional[torch.Tensor] = None,
    hash_ids: Optional[torch.Tensor] = None,
    runtime: Optional[HSABlockSparseRuntime] = None,
):
    backward_mode = _get_hsa_blocksparse_backward_mode(schedule)
    if backward_mode == "monolithic_sentence":
        del keep_ids, hash_ids
        from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_monolithic

        try:
            return run_hsa_bwd_sm100_monolithic(
                q,
                k,
                v,
                out,
                dout,
                lse,
                schedule,
                softmax_scale,
                deterministic,
                sentence_lse=sentence_lse,
                sentence_q_stream=sentence_q_stream,
                sentence_k_stream=sentence_k_stream,
                sentence_v_stream=sentence_v_stream,
                sentence_out_stream=sentence_out_stream,
            )
        except NotImplementedError:
            pass
    if backward_mode == "legacy_packed":
        del keep_ids, hash_ids
        from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_packed

        try:
            return run_hsa_bwd_sm100_packed(
                q,
                k,
                v,
                out,
                dout,
                lse,
                schedule,
                softmax_scale,
                deterministic,
            )
        except NotImplementedError:
            return _run_hsa_hybrid_backward(q, k, v, out, dout, lse, schedule, softmax_scale)
    return _run_hsa_packed_mask_backward(
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
        runtime=runtime,
        )


class _FlashAttnHSABlockSparseFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep_ids: Optional[torch.Tensor],
        hash_ids: Optional[torch.Tensor],
        schedule: HSASchedule,
        softmax_scale: float,
        deterministic: bool,
        return_lse: bool,
    ):
        ctx.block_sparse_runtime = _get_hsa_block_sparse_runtime(schedule, q, k)
        ctx.use_synthetic_grid = _can_use_hsa_synthetic_grid_for_inputs(
            schedule,
            q,
            k,
            runtime=ctx.block_sparse_runtime,
        )
        if ctx.use_synthetic_grid:
            from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import run_hsa_fwd_sm100_synthetic_grid

            out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream = (
                run_hsa_fwd_sm100_synthetic_grid(
                    q,
                    k,
                    v,
                    schedule,
                    softmax_scale,
                    runtime=ctx.block_sparse_runtime,
                )
            )
        else:
            from flash_attn.cute.flash_hsa_fwd_sm100 import run_hsa_fwd_sm100_blocksparse

            out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream = (
                run_hsa_fwd_sm100_blocksparse(
                    q,
                    k,
                    v,
                    schedule,
                    softmax_scale,
                    allow_true_fused_bwd=_allow_hsa_true_fused_bwd_for_call(use_synthetic_grid=ctx.use_synthetic_grid),
                )
            )
        ctx.hsa_backward_mode = _get_hsa_blocksparse_backward_mode(schedule)
        ctx.schedule = schedule
        ctx.keep_ids = keep_ids
        ctx.hash_ids = hash_ids
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.synthetic_forward_prob_token = getattr(ctx.block_sparse_runtime, "synthetic_forward_prob_token", 0)
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            sentence_lse,
            sentence_q_stream,
            sentence_k_stream,
            sentence_v_stream,
            sentence_out_stream,
        )
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, sentence_lse, sentence_q_stream, sentence_k_stream, sentence_v_stream, sentence_out_stream = (
            ctx.saved_tensors
        )
        if _has_zero_upstream_grad(dout):
            dq, dk, dv = _zero_hsa_grads(q, k, v)
            return dq, dk, dv, None, None, None, None, None, None
        if ctx.block_sparse_runtime.backward_sparse is None:
            ctx.block_sparse_runtime = _get_hsa_block_sparse_runtime(
                ctx.schedule,
                q,
                k,
                require_backward=True,
            )
        sentence_lse = sentence_lse if sentence_lse.numel() > 0 else None
        sentence_q_stream = sentence_q_stream if sentence_q_stream.numel() > 0 else None
        sentence_k_stream = sentence_k_stream if sentence_k_stream.numel() > 0 else None
        sentence_v_stream = sentence_v_stream if sentence_v_stream.numel() > 0 else None
        sentence_out_stream = sentence_out_stream if sentence_out_stream.numel() > 0 else None

        if ctx.hsa_backward_mode == "sparse_mask":
            if ctx.use_synthetic_grid and os.environ.get("FLASH_ATTN_HSA_SYNTHETIC_MICRO_BWD", "0") == "1":
                from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import run_hsa_bwd_sm100_synthetic_grid

                dq, dk, dv = run_hsa_bwd_sm100_synthetic_grid(
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
                    ctx.schedule,
                    ctx.softmax_scale,
                    ctx.deterministic,
                    ctx.keep_ids,
                    ctx.hash_ids,
                    forward_prob_token=ctx.synthetic_forward_prob_token,
                    runtime=ctx.block_sparse_runtime,
                )
            else:
                dq, dk, dv = _run_hsa_packed_mask_backward(
                    q,
                    k,
                    v,
                    out,
                    dout,
                    lse,
                    ctx.schedule,
                    ctx.softmax_scale,
                    ctx.deterministic,
                    ctx.keep_ids,
                    ctx.hash_ids,
                    runtime=ctx.block_sparse_runtime,
                )
        elif ctx.hsa_backward_mode == "monolithic_sentence":
            from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_monolithic

            dq, dk, dv = run_hsa_bwd_sm100_monolithic(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.schedule,
                ctx.softmax_scale,
                ctx.deterministic,
                sentence_lse=sentence_lse,
                sentence_q_stream=sentence_q_stream,
                sentence_k_stream=sentence_k_stream,
                sentence_v_stream=sentence_v_stream,
                sentence_out_stream=sentence_out_stream,
                allow_true_fused=_allow_hsa_true_fused_bwd_for_call(use_synthetic_grid=ctx.use_synthetic_grid),
            )
        else:
            from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_blocksparse

            dq, dk, dv = run_hsa_bwd_sm100_blocksparse(
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
                ctx.schedule,
                ctx.softmax_scale,
                ctx.deterministic,
                ctx.keep_ids,
                ctx.hash_ids,
                runtime=ctx.block_sparse_runtime,
            )
        return dq, dk, dv, None, None, None, None, None, None


class _FlashAttnHSACachedGeneralizedForwardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        keep_ids: Optional[torch.Tensor],
        hash_ids: Optional[torch.Tensor],
        schedule: HSASchedule,
        cached_forward_payload: dict[str, Any],
        softmax_scale: float,
        deterministic: bool,
        return_lse: bool,
    ):
        from flash_attn.cute.hsa_cached_2d_forward_analysis import (
            can_use_cached_generalized_fused_backward,
            run_cached_generalized_packed_forward,
        )

        ctx.block_sparse_runtime = _get_hsa_block_sparse_runtime(schedule, q, k)
        ctx.use_synthetic_grid = _can_use_hsa_synthetic_grid_for_inputs(
            schedule,
            q,
            k,
            runtime=ctx.block_sparse_runtime,
        )
        ctx.hsa_backward_mode = _get_hsa_blocksparse_backward_mode(schedule)
        ctx.synthetic_forward_prob_token = getattr(ctx.block_sparse_runtime, "synthetic_forward_prob_token", 0)
        ctx.cached_forward_payload = cached_forward_payload
        ctx.use_cached_generalized_fused_bwd = (
            ctx.hsa_backward_mode == "sparse_mask"
            and os.environ.get("FLASH_ATTN_HSA_CACHED_GENERALIZED_FUSED_BWD", "0") == "1"
            and can_use_cached_generalized_fused_backward(
                cached_forward_payload,
                q,
                k,
                v,
                deterministic=deterministic,
            )
        )
        ctx.use_synthetic_micro_bwd = (
            not ctx.use_cached_generalized_fused_bwd
            and
            ctx.hsa_backward_mode == "sparse_mask"
            and _can_use_cached_generalized_synthetic_micro_bwd(schedule, ctx.block_sparse_runtime, q, k, v)
        )
        setattr(schedule, "_last_cached_generalized_fused_bwd_used", False)
        out, lse = run_cached_generalized_packed_forward(
            cached_forward_payload,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            return_lse=True,
        )
        ctx.schedule = schedule
        ctx.keep_ids = keep_ids
        ctx.hash_ids = hash_ids
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.save_for_backward(q, k, v, out, lse)
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        if _has_zero_upstream_grad(dout):
            dq, dk, dv = _zero_hsa_grads(q, k, v)
            setattr(ctx.schedule, "_last_cached_generalized_fused_bwd_used", False)
            return dq, dk, dv, None, None, None, None, None, None, None
        if getattr(ctx.block_sparse_runtime, "backward_sparse", None) is None:
            try:
                ctx.block_sparse_runtime = _get_hsa_block_sparse_runtime(
                    ctx.schedule,
                    q,
                    k,
                    require_backward=True,
                )
            except TypeError:
                ctx.block_sparse_runtime = _get_hsa_block_sparse_runtime(
                    ctx.schedule,
                    q,
                    k,
                )
        if ctx.use_cached_generalized_fused_bwd:
            from flash_attn.cute.hsa_cached_2d_forward_analysis import run_cached_generalized_packed_backward

            dq, dk, dv = run_cached_generalized_packed_backward(
                ctx.cached_forward_payload,
                q,
                k,
                v,
                out,
                dout,
                lse,
                softmax_scale=ctx.softmax_scale,
                deterministic=ctx.deterministic,
            )
            setattr(ctx.schedule, "_last_cached_generalized_fused_bwd_used", True)
        elif ctx.use_synthetic_micro_bwd:
            from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import run_hsa_bwd_sm100_synthetic_grid

            dq, dk, dv = run_hsa_bwd_sm100_synthetic_grid(
                q,
                k,
                v,
                out,
                dout,
                lse,
                None,
                None,
                None,
                None,
                None,
                ctx.schedule,
                ctx.softmax_scale,
                ctx.deterministic,
                ctx.keep_ids,
                ctx.hash_ids,
                forward_prob_token=ctx.synthetic_forward_prob_token,
                runtime=ctx.block_sparse_runtime,
            )
            setattr(ctx.schedule, "_last_cached_generalized_fused_bwd_used", False)
        else:
            dq, dk, dv = _run_hsa_packed_mask_backward(
                q,
                k,
                v,
                out,
                dout,
                lse,
                ctx.schedule,
                ctx.softmax_scale,
                ctx.deterministic,
                ctx.keep_ids,
                ctx.hash_ids,
                runtime=ctx.block_sparse_runtime,
            )
            setattr(ctx.schedule, "_last_cached_generalized_fused_bwd_used", False)
        return dq, dk, dv, None, None, None, None, None, None, None


class _FlashAttnHSASparseExactFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        schedule: HSASchedule,
        softmax_scale: float,
        deterministic: bool,
        return_lse: bool,
    ):
        from flash_attn.cute.flash_hsa_fwd_sm100 import run_hsa_fwd_sm100_exact

        out, lse, sentence_lse, section_prefix_lse, document_prefix_lse = run_hsa_fwd_sm100_exact(
            q,
            k,
            v,
            schedule,
            softmax_scale,
        )
        ctx.schedule = schedule
        ctx.softmax_scale = softmax_scale
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        total_lse_flat = lse.permute(0, 2, 1).contiguous().view(schedule.num_rows, q.shape[2])
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            total_lse_flat,
            sentence_lse,
            section_prefix_lse,
            document_prefix_lse,
        )
        if return_lse:
            ctx.mark_non_differentiable(lse)
            return out, lse
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_exact

        q, k, v, out, total_lse_flat, sentence_lse, section_prefix_lse, document_prefix_lse = ctx.saved_tensors
        if _has_zero_upstream_grad(dout):
            dq, dk, dv = _zero_hsa_grads(q, k, v)
            return dq, dk, dv, None, None, None, None
        schedule = ctx.schedule
        dq, dk, dv = run_hsa_bwd_sm100_exact(
            q,
            k,
            v,
            out,
            dout,
            total_lse_flat,
            sentence_lse,
            section_prefix_lse,
            document_prefix_lse,
            schedule,
            ctx.softmax_scale,
            ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None


def flash_attn_hsa_sparse_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ids: Optional[torch.Tensor] = None,
    hash_ids: Optional[torch.Tensor] = None,
    hsa_schedule: Optional[HSASchedule] = None,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """
    Exact sparse HSA entrypoint using dense FA4 varlen substreams plus exact combine logic.
    Falls back to the schedule reference path when CUDA/FA4 is unavailable.
    """
    if hsa_schedule is None:
        assert keep_ids is not None and hash_ids is not None, (
            "keep_ids and hash_ids are required when hsa_schedule is not provided"
        )
        hsa_schedule = build_hsa_schedule(keep_ids, hash_ids)
    elif hsa_schedule.sentence_start.device != q.device:
        hsa_schedule = hsa_schedule.to(q.device)

    if q.device.type != "cuda" or q.dtype not in {torch.float16, torch.bfloat16}:
        return hsa_sparse_reference_attention(
            q,
            k,
            v,
            hsa_schedule,
            softmax_scale=softmax_scale,
            return_lse=return_lse,
        )

    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    normalized_keep_ids = None
    normalized_hash_ids = None
    if keep_ids is not None and hash_ids is not None:
        normalized_keep_ids = _ensure_int32(keep_ids)
        normalized_hash_ids = _ensure_int32(hash_ids)
        if normalized_keep_ids.device != q.device:
            normalized_keep_ids = normalized_keep_ids.to(device=q.device)
        if normalized_hash_ids.device != q.device:
            normalized_hash_ids = normalized_hash_ids.to(device=q.device)
        if not normalized_keep_ids.is_contiguous():
            normalized_keep_ids = normalized_keep_ids.contiguous()
        if not normalized_hash_ids.is_contiguous():
            normalized_hash_ids = normalized_hash_ids.contiguous()

    if torch.cuda.get_device_capability(q.device)[0] >= 10:
        cached_forward_payload = _resolve_precomputed_cached_generalized_forward_payload(hsa_schedule, q, k)
        if isinstance(cached_forward_payload, dict):
            return _FlashAttnHSACachedGeneralizedForwardFunc.apply(
                q,
                k,
                v,
                normalized_keep_ids,
                normalized_hash_ids,
                hsa_schedule,
                cached_forward_payload,
                scale,
                deterministic,
                return_lse,
            )
        return _FlashAttnHSABlockSparseFunc.apply(
            q,
            k,
            v,
            normalized_keep_ids,
            normalized_hash_ids,
            hsa_schedule,
            scale,
            deterministic,
            return_lse,
        )
    return _FlashAttnHSASparseExactFunc.apply(q, k, v, hsa_schedule, scale, deterministic, return_lse)


def flash_attn_hsa_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_ids: torch.Tensor,
    hash_ids: torch.Tensor,
    softmax_scale: Optional[float] = None,
    deterministic: bool = False,
    return_lse: bool = False,
):
    """Public fixed-length HSA training entrypoint built on top of flash_attn_func."""
    _, _, _, _, flash_attn_func, _, _ = _lazy_cute_imports()
    keep_ids = _ensure_int32(keep_ids)
    hash_ids = _ensure_int32(hash_ids)
    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=softmax_scale,
        causal=False,
        deterministic=deterministic,
        mask_mod=get_hsa_mask_mod(),
        aux_tensors=[keep_ids.contiguous(), hash_ids.contiguous()],
        return_lse=return_lse,
    )
