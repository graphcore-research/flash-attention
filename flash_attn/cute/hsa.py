import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

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
class HSABlockSparseRuntime:
    """Cached runtime metadata for the single-call HSA block-sparse path."""

    forward_sparse: HSABlockSparseTensors
    backward_sparse: HSABlockSparseTensors
    backward_packed_masks: "HSABwdPackedMasks"
    forward_aux_tensors: list[torch.Tensor]
    backward_aux_tensors: list[torch.Tensor]
    forward_block_q: int
    backward_block_q: int
    block_k: int
    backward_subtile_factor: int

    def to(self, device: torch.device | str):
        def _move_aux(tensor: torch.Tensor) -> torch.Tensor:
            moved = tensor.to(device=device)
            if hasattr(tensor, "__assumed_align__"):
                setattr(moved, "__assumed_align__", getattr(tensor, "__assumed_align__"))
            if hasattr(tensor, "__leading_dim__"):
                setattr(moved, "__leading_dim__", getattr(tensor, "__leading_dim__"))
            return moved

        return HSABlockSparseRuntime(
            forward_sparse=self.forward_sparse.to(device=device),
            backward_sparse=self.backward_sparse.to(device=device),
            backward_packed_masks=self.backward_packed_masks.to(device=device),
            forward_aux_tensors=[_move_aux(tensor) for tensor in self.forward_aux_tensors],
            backward_aux_tensors=[_move_aux(tensor) for tensor in self.backward_aux_tensors],
            forward_block_q=self.forward_block_q,
            backward_block_q=self.backward_block_q,
            block_k=self.block_k,
            backward_subtile_factor=self.backward_subtile_factor,
        )


@dataclass
class HSABwdPackedMasks:
    """Packed exact masks for HSA backward partial blocks."""

    block_id_table: torch.Tensor
    mask_words: torch.Tensor
    q_block_size: int
    k_block_size: int
    words_per_row: int

    def to(self, device: torch.device | str):
        return HSABwdPackedMasks(
            block_id_table=self.block_id_table.to(device=device),
            mask_words=self.mask_words.to(device=device),
            q_block_size=self.q_block_size,
            k_block_size=self.k_block_size,
            words_per_row=self.words_per_row,
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
    section_self_allowed = [False] * total_rows
    document_self_allowed = [False] * total_rows

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
                    section_self_allowed[flat_row] = True
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
                    document_self_allowed[flat_row] = True
            for idx, pos in enumerate(seg):
                flat_row = row_base + pos
                query_start = idx + 1 if (ki0[pos] or ki1[pos]) else idx
                if query_start < seg_len:
                    document_t_rows[flat_row].extend(seg[query_start:])

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
        section_self_allowed=torch.tensor(section_self_allowed, dtype=torch.bool, device=device),
        document_segment_ptr=document_segment_ptr,
        document_segment_pos=document_segment_pos,
        document_segment_id=document_segment_id,
        document_segment_offset=document_segment_offset,
        document_self_allowed=torch.tensor(document_self_allowed, dtype=torch.bool, device=device),
        forward_descriptors=forward_descriptors,
        backward_descriptors=backward_descriptors,
        sentence_stream=sentence_stream,
        section_prefix_stream=section_prefix_stream,
        document_prefix_stream=document_prefix_stream,
        section_self_indices=(
            torch.tensor(section_self_indices, dtype=torch.int32, device=device)
            if section_self_indices
            else _empty_int32(device)
        ),
        document_self_indices=(
            torch.tensor(document_self_indices, dtype=torch.int32, device=device)
            if document_self_indices
            else _empty_int32(device)
        ),
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
    """Expand the schedule aux-tensor encoding into the exact dense bool attention mask."""
    (
        sentence_segment_id,
        sentence_segment_offset,
        section_segment_id,
        section_segment_offset,
        section_self_allowed,
        document_segment_id,
        document_segment_offset,
        document_self_allowed,
    ) = [tensor.detach().cpu() for tensor in _build_hsa_schedule_aux_tensors(schedule)]
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    attend = torch.zeros(bsz, seqlen, seqlen, dtype=torch.bool)

    for batch_idx in range(bsz):
        for q_idx in range(seqlen):
            q_sentence_segment = int(sentence_segment_id[batch_idx, q_idx].item())
            q_sentence_offset = int(sentence_segment_offset[batch_idx, q_idx].item())
            q_section_segment = int(section_segment_id[batch_idx, q_idx].item())
            q_section_offset = int(section_segment_offset[batch_idx, q_idx].item())
            q_section_self = int(section_self_allowed[batch_idx, q_idx].item())
            q_document_segment = int(document_segment_id[batch_idx, q_idx].item())
            q_document_offset = int(document_segment_offset[batch_idx, q_idx].item())
            q_document_self = int(document_self_allowed[batch_idx, q_idx].item())

            for k_idx in range(seqlen):
                allowed = False
                k_sentence_segment = int(sentence_segment_id[batch_idx, k_idx].item())
                if q_sentence_segment >= 0 and q_sentence_segment == k_sentence_segment:
                    k_sentence_offset = int(sentence_segment_offset[batch_idx, k_idx].item())
                    allowed = k_sentence_offset <= q_sentence_offset

                if not allowed:
                    k_section_segment = int(section_segment_id[batch_idx, k_idx].item())
                    if q_section_segment >= 0 and q_section_segment == k_section_segment:
                        k_section_offset = int(section_segment_offset[batch_idx, k_idx].item())
                        allowed = k_section_offset < q_section_offset + q_section_self

                if not allowed:
                    k_document_segment = int(document_segment_id[batch_idx, k_idx].item())
                    if q_document_segment >= 0 and q_document_segment == k_document_segment:
                        k_document_offset = int(document_segment_offset[batch_idx, k_idx].item())
                        allowed = k_document_offset < q_document_offset + q_document_self

                attend[batch_idx, q_idx, k_idx] = allowed

    return attend.to(device=schedule.sentence_start.device)


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
    bsz, seqlen = schedule.batch_size, schedule.seqlen
    num_q_blocks = (seqlen + q_block_size - 1) // q_block_size
    num_k_blocks = (seqlen + k_block_size - 1) // k_block_size
    counts = [[[0 for _ in range(num_k_blocks)] for _ in range(num_q_blocks)] for _ in range(bsz)]

    sentence_start = schedule.sentence_start.detach().cpu().tolist()
    sentence_len = schedule.sentence_len.detach().cpu().tolist()
    section_row_ptr = schedule.section_row_ptr.detach().cpu().tolist()
    section_col_idx = schedule.section_col_idx.detach().cpu().tolist()
    document_row_ptr = schedule.document_row_ptr.detach().cpu().tolist()
    document_col_idx = schedule.document_col_idx.detach().cpu().tolist()

    for flat_row in range(schedule.num_rows):
        batch_idx, q_idx = divmod(flat_row, seqlen)
        q_block = q_idx // q_block_size

        sent_len = sentence_len[flat_row]
        if sent_len > 0:
            sent_start = sentence_start[flat_row]
            _accumulate_interval_counts(
                counts,
                batch_idx,
                q_block,
                sent_start,
                sent_start + sent_len,
                k_block_size,
            )

        for offset in range(section_row_ptr[flat_row], section_row_ptr[flat_row + 1]):
            counts[batch_idx][q_block][section_col_idx[offset] // k_block_size] += 1
        for offset in range(document_row_ptr[flat_row], document_row_ptr[flat_row + 1]):
            counts[batch_idx][q_block][document_col_idx[offset] // k_block_size] += 1

    valid_counts = [
        [
            min(q_block_size, seqlen - q_block * q_block_size)
            * min(k_block_size, seqlen - k_block * k_block_size)
            for k_block in range(num_k_blocks)
        ]
        for q_block in range(num_q_blocks)
    ]
    return _counts_to_block_sparse_tensors(
        counts,
        valid_counts,
        block_size=(q_block_size, k_block_size),
        device=schedule.sentence_start.device,
    )


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


def _get_hsa_forward_q_block_size(q: torch.Tensor, k: torch.Tensor) -> int:
    num_q_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]
    qhead_per_kvhead = num_q_heads // num_kv_heads
    arch_major = torch.cuda.get_device_capability(q.device)[0]
    q_stage = 2 if arch_major == 10 and q.shape[1] * qhead_per_kvhead > 128 else 1
    return q_stage * 128


def _get_hsa_block_sparse_runtime(schedule: HSASchedule, q: torch.Tensor, k: torch.Tensor) -> HSABlockSparseRuntime:
    cache = getattr(schedule, "_hsa_block_sparse_runtime_cache", None)
    forward_block_q = _get_hsa_forward_q_block_size(q, k)
    backward_block_q = 128
    block_k = 128
    cache_key = (str(q.device), forward_block_q, backward_block_q, block_k)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    backward_sparse, backward_packed_masks = _build_backward_hsa_packed_masks(
        schedule,
        q_block_size=backward_block_q,
        k_block_size=block_k,
    )

    runtime = HSABlockSparseRuntime(
        forward_sparse=_build_forward_hsa_block_sparse_tensors(
            schedule,
            q_block_size=forward_block_q,
            k_block_size=block_k,
        ),
        backward_sparse=backward_sparse,
        backward_packed_masks=backward_packed_masks,
        forward_aux_tensors=_build_hsa_schedule_aux_tensors(schedule),
        backward_aux_tensors=[
            _tag_aux_tensor(backward_packed_masks.block_id_table),
            _tag_aux_tensor(backward_packed_masks.mask_words),
        ],
        forward_block_q=forward_block_q,
        backward_block_q=backward_block_q,
        block_k=block_k,
        backward_subtile_factor=1,
    )
    if cache is None:
        cache = {}
        setattr(schedule, "_hsa_block_sparse_runtime_cache", cache)
    cache[cache_key] = runtime
    return runtime


def _to_block_sparse_tensors_torch(sparse_tensors: HSABlockSparseTensors):
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    return BlockSparseTensorsTorch(
        mask_block_cnt=sparse_tensors.mask_block_cnt,
        mask_block_idx=sparse_tensors.mask_block_idx,
        full_block_cnt=sparse_tensors.full_block_cnt,
        full_block_idx=sparse_tensors.full_block_idx,
        block_size=sparse_tensors.block_size,
    )


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


@lru_cache(maxsize=1)
def get_hsa_backward_packed_mask_mod():
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

        q_block = safe_q_idx // 128
        k_block = safe_kv_idx // 128
        q_local = safe_q_idx % 128
        k_local = safe_kv_idx % 128
        word_idx = k_local // 32
        bit_idx = k_local % 32

        block_id = block_id_table[b, k_block, q_block]
        mask_word = cutlass.Uint32(mask_words[block_id, q_local, word_idx])
        bit = utils.shr_u32(mask_word, cutlass.Uint32(bit_idx)) & cutlass.Uint32(1)
        allowed = cutlass.Boolean(in_bounds & (bit != cutlass.Uint32(0)))
        return utils.scalar_to_ssa(allowed, cutlass.Boolean)

    return _hsa_backward_packed_mask


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

        level0 = (
            (q_ki0 != 0)
            & (k_ki0 != 0)
            & same_sent
            & ((q_ki1 == 0) | (k_ki1 == 0))
        )
        level1 = (q_ki1 != 0) & (k_ki1 != 0) & same_sec
        level2 = (q_ki2 != 0) & (k_ki2 != 0) & same_doc
        causal = kv_idx <= q_idx
        return in_bounds & causal & (level0 | level1 | level2)

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
    attend = causal & (level0 | level1 | level2)

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
        mask_mod=get_hsa_schedule_mask_mod(),
        aux_tensors=runtime.forward_aux_tensors,
        block_sparse_tensors=_to_block_sparse_tensors_torch(runtime.forward_sparse),
        return_lse=True,
    )
    return out, lse


def _run_hsa_blocksparse_backward(
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
):
    _, _, _, _, _, _, flash_attn_bwd = _lazy_cute_imports()
    runtime = _get_hsa_block_sparse_runtime(schedule, q, k)
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
        mask_mod=get_hsa_backward_packed_mask_mod(),
        aux_tensors=runtime.backward_aux_tensors,
        block_sparse_tensors=_to_block_sparse_tensors_torch(runtime.backward_sparse),
        subtile_factor_override=runtime.backward_subtile_factor,
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
        from flash_attn.cute.flash_hsa_fwd_sm100 import run_hsa_fwd_sm100_blocksparse

        out, lse = run_hsa_fwd_sm100_blocksparse(q, k, v, schedule, softmax_scale)
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
        from flash_attn.cute.flash_hsa_bwd_sm100 import run_hsa_bwd_sm100_blocksparse

        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = run_hsa_bwd_sm100_blocksparse(
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
        )
        return dq, dk, dv, None, None, None, None, None, None


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
