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


def test_cached_2d_monolithic_forward_requires_complete_fused_tail_payload():
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    payload = {
        "total_rows": 4,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.empty((1, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "fused_output_row_count": 3,
        "geometry": {"fused_total_coverage_frac": 1.0},
    }

    assert cached_2d._cached_monolithic_forward_support_reason(payload, q, q, q) == "requires_cuda"

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (4, 2, 64)

    assert (
        cached_2d._cached_monolithic_forward_support_reason(payload, FakeCudaTensor(), FakeCudaTensor(), FakeCudaTensor())
        == "incomplete_fused_output_row_coverage"
    )

    payload["fused_output_row_count"] = 4
    assert cached_2d._cached_monolithic_forward_support_reason(payload, FakeCudaTensor(), FakeCudaTensor(), FakeCudaTensor()) is None
