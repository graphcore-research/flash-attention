import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flash_attn.cute.hsa_cached_2d_forward_analysis as cached_2d


def test_cached_2d_range_annotation_materializes_kernel_descriptors(monkeypatch):
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    range_execution = [
        {"group_start": 0, "group_end": 2, "scatter_only": True, "family": "union_2d"},
        {"group_start": 2, "group_end": 3, "scatter_only": True, "family": "k_window"},
        {"group_start": 3, "group_end": 5, "scatter_only": False, "family": "union_2d"},
    ]

    monkeypatch.setattr(cached_2d, "_can_use_synthetic_2d_masked_fwd", lambda *args, **kwargs: True)

    annotated = cached_2d._annotate_range_execution_kernels(
        range_execution,
        rows_per_group=16,
        max_union_k=64,
        tile_k=32,
        union_kernel="tc16x32",
        q_flat=q,
        k_flat=q,
        v_flat=q,
    )
    descriptors = cached_2d._materialize_range_execution_tensors(annotated, device=q.device)

    assert [entry["kernel_kind"] for entry in annotated] == ["tc_scatter", "scatter", "packed"]
    assert descriptors["range_group_start"].tolist() == [0, 2, 3]
    assert descriptors["range_group_end"].tolist() == [2, 3, 5]
    assert descriptors["range_scatter_only"].tolist() == [1, 1, 0]
    assert descriptors["range_kernel_kind"].tolist() == [2, 1, 0]


def test_cached_2d_flat_row_idx_uses_precomputed_payload_slice():
    payload = {
        "q_row_idx": torch.tensor([[0, 1], [2, -1], [3, 4]], dtype=torch.int32),
        "q_row_idx_flat": torch.tensor([0, 1, 2, -1, 3, 4], dtype=torch.int32),
    }

    row_idx = cached_2d._slice_cached_flat_row_idx(
        payload,
        flat_key="q_row_idx_flat",
        matrix_key="q_row_idx",
        group_start=1,
        group_end=3,
        width=2,
    )

    assert row_idx.tolist() == [2, -1, 3, 4]
