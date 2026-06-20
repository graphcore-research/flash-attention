import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flash_attn.cute.hsa_cached_2d_forward_analysis as cached_2d
import flash_attn.cute.hsa_explicit_2d_sparse_analysis as explicit_2d


def test_explicit_2d_average_pairwise_row_jaccard_matches_manual():
    mask_rows = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    expected = []
    for left_idx in range(mask_rows.shape[0]):
        for right_idx in range(left_idx + 1, mask_rows.shape[0]):
            left = mask_rows[left_idx]
            right = mask_rows[right_idx]
            intersection = int(torch.logical_and(left, right).sum().item())
            union = int(torch.logical_or(left, right).sum().item())
            expected.append(0.0 if union <= 0 else intersection / union)

    assert explicit_2d._average_pairwise_row_jaccard(mask_rows) == pytest.approx(
        sum(expected) / len(expected)
    )


def test_explicit_2d_report_splits_payload_and_geometry_timing():
    report = explicit_2d.analyze_explicit_2d_sparse_forward(
        case_family="disjoint_confetti",
        seqlen=16,
        heads=1,
        head_dim=8,
        packed_q=4,
        support_k=16,
        islands_per_row=2,
        island_width=2,
        row_shift=3,
        warmup_iters=0,
        benchmark_iters=1,
        variants=("dense",),
        device="cpu",
        dtype=torch.float32,
        seed=0,
        check_correctness=True,
    )

    total = report["payload_build_seconds"]
    assert report["payload_build_total_seconds"] == total
    assert report["diagnostic_geometry_seconds"] >= 0.0
    assert report["payload_build_excluding_diagnostic_geometry_seconds"] >= 0.0
    assert (
        report["payload_build_excluding_diagnostic_geometry_seconds"]
        + report["diagnostic_geometry_seconds"]
    ) == pytest.approx(total, abs=0.01)


def test_runtime_payload_cache_reuses_runtime_dict():
    class Runtime:
        pass

    runtime = Runtime()
    cache = cached_2d._runtime_payload_cache(runtime, "_payload_cache")
    assert isinstance(cache, dict)
    cache["key"] = {"status": "ready"}
    assert cached_2d._runtime_payload_cache(runtime, "_payload_cache") is cache

    runtime._payload_cache = []
    assert cached_2d._runtime_payload_cache(runtime, "_payload_cache") is None


def test_cached_payload_shape_key_separates_shape_policy_layout_and_plan():
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    k = torch.empty((5, 2, 64), dtype=torch.bfloat16)
    v = torch.empty((5, 2, 64), dtype=torch.bfloat16)
    direct_plan = {}
    row_plan = {}
    policy = (("max_rows_per_group", 16),)

    base = cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )

    assert base == cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=torch.empty((6, 2, 64), dtype=torch.bfloat16),
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=(("max_rows_per_group", 8),),
        include_mask_bool=False,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan=direct_plan,
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=True,
    )
    assert base != cached_2d._cached_payload_shape_key(
        q_flat=q,
        k_flat=k,
        v_flat=v,
        direct_plan={},
        row_plan=row_plan,
        policy_key=policy,
        include_mask_bool=False,
    )


def test_cached_lse_public_to_flat_triton_gate_rejects_cpu():
    q = torch.empty((2, 16, 8, 64), dtype=torch.bfloat16)
    q_flat = q.reshape(-1, 8, 64)
    lse = torch.empty((2, 8, 16), dtype=torch.float32)

    assert not cached_2d._can_use_triton_lse_public_to_flat(q, q_flat, lse)


def test_cached_lse_public_to_flat_triton_matches_pytorch_and_reuses_buffer(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton public-LSE-to-flat test")
    if not cached_2d._HAS_TRITON_LSE_PUBLIC_TO_FLAT:
        pytest.skip("Triton unavailable")

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_BWD_TRITON_LSE_PUBLIC_TO_FLAT", raising=False)
    q = torch.empty((2, 128, 8, 64), dtype=torch.bfloat16, device="cuda")
    q_flat = q.reshape(-1, 8, 64)
    lse = torch.randn((2, 8, 128), dtype=torch.float32, device="cuda")
    ref = lse.permute(0, 2, 1).contiguous().view(-1, 8).float()

    payload = {}
    lse_flat = cached_2d._get_cached_lse_flat_for_backward_buffer(payload, q_flat, 8)
    lse_flat_again = cached_2d._get_cached_lse_flat_for_backward_buffer(payload, q_flat, 8)
    assert lse_flat_again.data_ptr() == lse_flat.data_ptr()

    got = cached_2d._flatten_cached_lse_for_backward(q, q_flat, lse, out=lse_flat)
    torch.cuda.synchronize()

    assert got.data_ptr() == lse_flat.data_ptr()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_cached_packing_policy_allows_wide_2d_union_but_not_wide_direct():
    assert cached_2d.CachedPackingPolicy().max_union_k_2d == 1024

    policy = cached_2d._coerce_cached_packing_policy(
        cached_2d.CachedPackingPolicy(max_union_k_direct=128, max_union_k_2d=1024)
    )
    assert policy.max_union_k_2d == 1024

    try:
        cached_2d._coerce_cached_packing_policy(
            cached_2d.CachedPackingPolicy(max_union_k_direct=256, max_union_k_2d=512)
        )
    except ValueError as exc:
        assert "max_union_k_direct" in str(exc)
    else:
        raise AssertionError("expected max_union_k_direct >128 to be rejected")


def test_synthetic_2d_masked_fwd_gate_allows_k2048_d64_and_caps_d128(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for direct 2D gate test")

    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _can_use_synthetic_2d_masked_fwd

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_MAX_PACKED_K", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_D128_MAX_PACKED_K", raising=False)
    q64 = torch.empty((1, 8, 64), dtype=torch.bfloat16, device="cuda")
    assert _can_use_synthetic_2d_masked_fwd(q64, q64, q64, packed_q=16, packed_k=2048)
    assert not _can_use_synthetic_2d_masked_fwd(q64, q64, q64, packed_q=16, packed_k=2049)

    q128 = torch.empty((1, 8, 128), dtype=torch.bfloat16, device="cuda")
    assert _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=128)
    assert not _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=129)

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_GATHER_D128_MAX_PACKED_K", "256")
    assert _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=256)
    assert not _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=257)

    payload = {
        "union_kernel": "tc16x32",
        "packed_q": 16,
        "tile_k": 32,
        "support_rows": 128,
        "q_length": torch.full((1,), 16, dtype=torch.int32, device="cuda"),
    }
    range_entry = {"family": "union_2d", "scatter_only": True}
    assert cached_2d._can_use_cached_union_tc(
        payload,
        q128,
        q128,
        q128,
        group_start=0,
        group_end=1,
        range_entry=range_entry,
    )
    payload["support_rows"] = 129
    assert not cached_2d._can_use_cached_union_tc(
        payload,
        q128,
        q128,
        q128,
        group_start=0,
        group_end=1,
        range_entry=range_entry,
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", "0")
    assert not _can_use_synthetic_2d_masked_fwd(q128, q128, q128, packed_q=16, packed_k=128)


def test_synthetic_2d_masked_gather_scatter_tc_d128_matches_scalar(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for D128 TC direct 2D correctness test")

    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
        _run_synthetic_2d_masked_gather_scatter_fwd_kernel,
        _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel,
    )

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", raising=False)
    torch.manual_seed(124)
    groups, packed_q, packed_k, num_heads, head_dim = 1, 16, 32, 2, 128
    softmax_scale = head_dim**-0.5
    q_rows = torch.randn(groups * packed_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_rows = torch.randn(groups * packed_k, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_rows = torch.randn(groups * packed_k, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q_row_idx = torch.arange(groups * packed_q, dtype=torch.int32, device="cuda").view(groups, packed_q)
    k_row_idx = torch.arange(groups * packed_k, dtype=torch.int32, device="cuda").view(groups, packed_k)
    q_length = torch.full((groups,), packed_q, dtype=torch.int32, device="cuda")
    k_length = torch.full((groups,), packed_k, dtype=torch.int32, device="cuda")
    mask_words_cpu = torch.zeros((groups, packed_q, (packed_k + 31) // 32), dtype=torch.int32)
    for q_idx in range(packed_q):
        for k_idx in range(packed_k):
            if (k_idx + 3 * q_idx) % 5 != 0:
                word_idx, bit_idx = divmod(k_idx, 32)
                mask_words_cpu[0, q_idx, word_idx] |= 1 << bit_idx
    mask_words = mask_words_cpu.to("cuda")
    out_scalar = torch.empty((groups * packed_q, num_heads, head_dim), dtype=torch.float32, device="cuda")
    lse_scalar = torch.empty((groups * packed_q, num_heads), dtype=torch.float32, device="cuda")
    out_tc = torch.empty_like(out_scalar)
    lse_tc = torch.empty_like(lse_scalar)

    _run_synthetic_2d_masked_gather_scatter_fwd_kernel(
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        out_scalar,
        lse_scalar,
        softmax_scale=softmax_scale,
    )
    _run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel(
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        out_tc,
        lse_tc,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_tc, out_scalar, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(lse_tc, lse_scalar, rtol=1e-5, atol=2e-5)


def test_synthetic_2d_masked_gather_d128_matches_dense_partial_mask(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for D128 direct 2D correctness test")

    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import _run_synthetic_2d_masked_gather_fwd_kernel

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_D128", raising=False)
    torch.manual_seed(123)
    groups, packed_q, packed_k, num_heads, head_dim = 1, 4, 65, 2, 128
    softmax_scale = head_dim**-0.5
    q_rows = torch.randn(groups * packed_q, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_rows = torch.randn(groups * packed_k, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_rows = torch.randn(groups * packed_k, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q_row_idx = torch.arange(groups * packed_q, dtype=torch.int32, device="cuda").view(groups, packed_q)
    k_row_idx = torch.arange(groups * packed_k, dtype=torch.int32, device="cuda").view(groups, packed_k)
    q_length = torch.full((groups,), packed_q, dtype=torch.int32, device="cuda")
    k_length = torch.full((groups,), packed_k, dtype=torch.int32, device="cuda")
    mask_words_cpu = torch.zeros((groups, packed_q, (packed_k + 31) // 32), dtype=torch.int32)
    for q_idx in range(packed_q):
        for k_idx in range(packed_k):
            if (k_idx + 3 * q_idx) % 5 != 0 and (k_idx < 60 or k_idx % 2 == 1):
                word_idx, bit_idx = divmod(k_idx, 32)
                mask_words_cpu[0, q_idx, word_idx] |= 1 << bit_idx
    mask_words = mask_words_cpu.to("cuda")

    out, lse = _run_synthetic_2d_masked_gather_fwd_kernel(
        q_rows,
        k_rows,
        v_rows,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()

    ref_out = torch.zeros_like(out)
    ref_lse = torch.empty_like(lse)
    for q_idx in range(packed_q):
        active_keys = []
        for k_idx in range(packed_k):
            word_idx, bit_idx = divmod(k_idx, 32)
            if int(mask_words_cpu[0, q_idx, word_idx]) & (1 << bit_idx):
                active_keys.append(k_idx)
        for head_idx in range(num_heads):
            scores = (
                q_rows[q_idx, head_idx].float()
                @ k_rows[torch.tensor(active_keys, device="cuda"), head_idx].float().T
            ) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            ref_out[0, q_idx, head_idx] = probs @ v_rows[torch.tensor(active_keys, device="cuda"), head_idx].float()
            ref_lse[0, q_idx, head_idx] = torch.logsumexp(scores, dim=-1)

    torch.testing.assert_close(out, ref_out, rtol=0, atol=2e-5)
    torch.testing.assert_close(lse, ref_lse, rtol=0, atol=2e-5)


def test_explicit_2d_payloads_preserve_long_noncontiguous_q_rows():
    import flash_attn.cute.hsa_explicit_2d_sparse_analysis as explicit_2d

    packed_out = torch.arange(2 * 4 * 1 * 2, dtype=torch.float32).view(2, 4, 1, 2)
    q_length = torch.tensor([3, 2], dtype=torch.int32)
    q_row_idx = torch.tensor(
        [
            [65536, 3, 8, -1],
            [70000, 2, -1, -1],
        ],
        dtype=torch.int32,
    )

    scattered = explicit_2d.scatter_explicit_packed_rows(
        packed_out,
        q_row_idx,
        q_length,
        total_rows=70001,
    )

    assert scattered.shape == (70001, 1, 2)
    torch.testing.assert_close(scattered[65536], packed_out[0, 0])
    torch.testing.assert_close(scattered[3], packed_out[0, 1])
    torch.testing.assert_close(scattered[8], packed_out[0, 2])
    torch.testing.assert_close(scattered[70000], packed_out[1, 0])
    torch.testing.assert_close(scattered[2], packed_out[1, 1])

    q_buf = torch.randn((2, 4, 1, 64), dtype=torch.float32)
    k_buf = torch.randn((2, 8, 1, 64), dtype=torch.float32)
    v_buf = torch.randn((2, 8, 1, 64), dtype=torch.float32)
    mask_bool = torch.zeros((2, 4, 8), dtype=torch.bool)
    mask_bool[0, 0, [0, 2, 4]] = True
    mask_bool[0, 1, [1, 3]] = True
    mask_bool[0, 2, [2, 5]] = True
    mask_bool[1, 0, [0, 7]] = True
    mask_bool[1, 1, [3, 6]] = True

    direct_bucket, _ = explicit_2d._build_direct_2d_bucket(
        q_buf=q_buf,
        k_buf=k_buf,
        v_buf=v_buf,
        mask_bool=mask_bool,
        q_length=q_length,
        q_row_idx=q_row_idx,
    )
    compact_payload, _ = explicit_2d._build_direct_2d_compact_payload(
        q_buf=q_buf,
        k_buf=k_buf,
        v_buf=v_buf,
        mask_bool=mask_bool,
        q_length=q_length,
        q_row_idx=q_row_idx,
    )

    assert direct_bucket["total_rows"] == 70001
    assert compact_payload["total_rows"] == 70001
    torch.testing.assert_close(compact_payload["q_row_idx"], q_row_idx)


def test_explicit_2d_compact_payload_handles_irregular_union_lengths():
    q_buf = torch.randn((3, 4, 1, 2), dtype=torch.float32)
    k_buf = torch.arange(3 * 16 * 1 * 2, dtype=torch.float32).view(3, 16, 1, 2)
    v_buf = -k_buf
    q_length = torch.tensor([3, 4, 2], dtype=torch.int32)
    q_row_idx = torch.tensor(
        [
            [0, 1, 2, -1],
            [10, 11, 12, 13],
            [20, 21, -1, -1],
        ],
        dtype=torch.int32,
    )
    mask_bool = torch.zeros((3, 4, 16), dtype=torch.bool)
    mask_bool[0, 0, [0, 2]] = True
    mask_bool[0, 1, [4]] = True
    mask_bool[0, 3, [15]] = True  # Invalid row; must not expand bucket union.
    mask_bool[1, 0, [1, 3]] = True
    mask_bool[1, 1, [5]] = True
    mask_bool[1, 2, [7]] = True
    mask_bool[1, 3, [9]] = True
    mask_bool[2, 0, [0, 1, 2]] = True
    mask_bool[2, 1, [3, 4, 5, 6]] = True

    compact_payload, geometry = explicit_2d._build_direct_2d_compact_payload(
        q_buf=q_buf,
        k_buf=k_buf,
        v_buf=v_buf,
        mask_bool=mask_bool,
        q_length=q_length,
        q_row_idx=q_row_idx,
        tile_k=8,
        total_rows=22,
    )

    assert geometry["direct_2d_compact_buckets_compacted"] == 3
    assert geometry["direct_2d_compact_buckets_passthrough"] == 0
    assert len(compact_payload["groups"]) == 1
    group = compact_payload["groups"][0]
    torch.testing.assert_close(group["bucket_indices"], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(group["custom_k_length"], torch.tensor([3, 5, 7], dtype=torch.int32))
    assert group["support_rows"] == 7

    for local_idx, bucket_idx in enumerate([0, 1, 2]):
        valid_rows = int(q_length[bucket_idx].item())
        union_cols = torch.nonzero(mask_bool[bucket_idx, :valid_rows].any(dim=0), as_tuple=False).flatten()
        union_k = int(union_cols.numel())
        torch.testing.assert_close(group["custom_k_buf"][local_idx, :union_k], k_buf[bucket_idx, union_cols])
        torch.testing.assert_close(group["custom_v_buf"][local_idx, :union_k], v_buf[bucket_idx, union_cols])
        torch.testing.assert_close(group["custom_mask_bool"][local_idx, :, :union_k], mask_bool[bucket_idx, :, union_cols])
        assert not bool(group["custom_mask_bool"][local_idx, :, union_k:].any().item())
        assert bool((group["custom_k_buf"][local_idx, union_k:] == 0).all().item())
        assert bool((group["custom_v_buf"][local_idx, union_k:] == 0).all().item())


def test_explicit_2d_compact_irregular_union_forward_matches_dense():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("CUDA SM100+ required for explicit 2D direct kernel")

    torch.manual_seed(0)
    num_buckets = 8
    packed_q = 16
    support_k = 128
    heads = 2
    head_dim = 64
    q_buf = torch.randn((num_buckets, packed_q, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k_buf = torch.randn((num_buckets, support_k, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v_buf = torch.randn((num_buckets, support_k, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    q_length = torch.full((num_buckets,), packed_q, dtype=torch.int32, device="cuda")
    q_row_idx = torch.arange(num_buckets * packed_q, dtype=torch.int32, device="cuda").view(num_buckets, packed_q)
    mask_bool = torch.zeros((num_buckets, packed_q, support_k), dtype=torch.bool, device="cuda")
    for bucket_idx in range(num_buckets):
        union_k = 17 + (bucket_idx * 7) % 24
        for q_slot in range(packed_q):
            row_live = 1 + (q_slot % 4)
            start = (q_slot * 3) % max(1, union_k - row_live + 1)
            mask_bool[bucket_idx, q_slot, start : start + row_live] = True

    mask_words = explicit_2d._encode_mask_rows_to_words(mask_bool.reshape(num_buckets * packed_q, support_k)).view(
        num_buckets,
        packed_q,
        -1,
    )
    full_bucket = {
        "packed_q": packed_q,
        "support_rows": support_k,
        "custom_q_buf": q_buf.contiguous(),
        "custom_k_buf": k_buf.contiguous(),
        "custom_v_buf": v_buf.contiguous(),
        "custom_mask_bool": mask_bool.contiguous(),
        "custom_mask_words": mask_words.contiguous(),
        "custom_q_length": q_length.contiguous(),
        "custom_k_length": torch.full((num_buckets,), support_k, dtype=torch.int32, device="cuda"),
        "q_row_idx": q_row_idx.contiguous(),
        "total_rows": num_buckets * packed_q,
    }
    compact_payload, geometry = explicit_2d._build_direct_2d_compact_payload(
        q_buf=q_buf,
        k_buf=k_buf,
        v_buf=v_buf,
        mask_bool=mask_bool,
        q_length=q_length,
        q_row_idx=q_row_idx,
        tile_k=64,
        total_rows=num_buckets * packed_q,
        contiguous_q_rows=True,
    )
    assert geometry["direct_2d_compact_buckets_compacted"] == num_buckets
    assert geometry["direct_2d_compact_buckets_passthrough"] == 0

    softmax_scale = head_dim ** (-0.5)
    dense = explicit_2d._run_dense_explicit_bucket_forward(full_bucket, softmax_scale=softmax_scale)
    compact = explicit_2d._run_direct_2d_compact_forward(
        {"full_bucket": full_bucket, "direct_2d_compact_payload": compact_payload},
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(compact, dense, rtol=0, atol=5e-2)


def test_synthetic_2d_masked_long_noncontiguous_rows_match_dense(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for long 2D row-index correctness test")

    from flash_attn.cute.flash_hsa_synthetic_grid_sm100 import (
        _run_synthetic_2d_masked_gather_combine_fwd_kernel,
        _run_synthetic_2d_masked_gather_fwd_kernel,
        _run_synthetic_2d_masked_gather_scatter_fwd_kernel,
    )

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GATHER_MAX_PACKED_K", raising=False)
    torch.manual_seed(0)
    rows = 70032
    heads = 2
    head_dim = 64
    groups = 2
    packed_q = 8
    packed_k = 32
    q = torch.randn((rows, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((rows, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((rows, heads, head_dim), dtype=torch.bfloat16, device="cuda")
    q_row_idx = torch.tensor(
        [
            [65536, 65537, 65539, 42, 69999, -1, -1, -1],
            [131, 65540, 66000, 70031, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    k_row_idx = torch.tensor(
        [
            [65535, 65536, 65537, 65538, 100, 101, 102, 103] + [-1] * 24,
            [0, 31, 65536, 66001, 70030, 70031, 60000, 40000] + [-1] * 24,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    q_length = torch.tensor([5, 4], dtype=torch.int32, device="cuda")
    k_length = torch.tensor([8, 8], dtype=torch.int32, device="cuda")
    mask_words_cpu = torch.zeros((groups, packed_q, (packed_k + 31) // 32), dtype=torch.int32)
    for group_idx in range(groups):
        for row_idx in range(int(q_length[group_idx])):
            for col_idx in range(8):
                if (col_idx + row_idx + group_idx) % 3 != 1:
                    mask_words_cpu[group_idx, row_idx, 0] |= 1 << col_idx
    mask_words = mask_words_cpu.to(device="cuda")
    scale = 1.0 / math.sqrt(float(head_dim))

    packed_out, packed_lse = _run_synthetic_2d_masked_gather_fwd_kernel(
        q,
        k,
        v,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        softmax_scale=scale,
        tile_k=32,
    )

    ref_out = torch.zeros_like(packed_out)
    ref_lse = torch.full_like(packed_lse, float("-inf"))
    for group_idx in range(groups):
        for row_idx in range(int(q_length[group_idx])):
            q_global = int(q_row_idx[group_idx, row_idx].item())
            key_rows = [
                int(k_row_idx[group_idx, col_idx].item())
                for col_idx in range(int(k_length[group_idx]))
                if int(mask_words_cpu[group_idx, row_idx, 0]) & (1 << col_idx)
            ]
            scores = torch.einsum("hd,khd->hk", q[q_global].float(), k[key_rows].float()) * scale
            probs = torch.softmax(scores, dim=-1)
            ref_out[group_idx, row_idx] = torch.einsum("hk,khd->hd", probs, v[key_rows].float())
            ref_lse[group_idx, row_idx] = torch.logsumexp(scores, dim=-1)

    torch.cuda.synchronize()
    for group_idx in range(groups):
        for row_idx in range(int(q_length[group_idx])):
            torch.testing.assert_close(
                packed_out[group_idx, row_idx].float(),
                ref_out[group_idx, row_idx].float(),
                atol=2e-3,
                rtol=2e-3,
            )
            torch.testing.assert_close(
                packed_lse[group_idx, row_idx].float(),
                ref_lse[group_idx, row_idx].float(),
                atol=2e-3,
                rtol=2e-3,
            )

    scattered_out = torch.full((rows, heads, head_dim), -123.0, dtype=torch.float32, device="cuda")
    scattered_lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device="cuda")
    _run_synthetic_2d_masked_gather_scatter_fwd_kernel(
        q,
        k,
        v,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        scattered_out,
        scattered_lse,
        softmax_scale=scale,
        tile_k=32,
    )
    combine_out = torch.zeros((rows, heads, head_dim), dtype=torch.float32, device="cuda")
    combine_lse = torch.full((rows, heads), float("-inf"), dtype=torch.float32, device="cuda")
    _run_synthetic_2d_masked_gather_combine_fwd_kernel(
        q,
        k,
        v,
        q_row_idx,
        k_row_idx,
        q_length,
        k_length,
        mask_words,
        combine_out,
        combine_lse,
        softmax_scale=scale,
        tile_k=32,
    )
    torch.cuda.synchronize()

    for group_idx in range(groups):
        for row_idx in range(int(q_length[group_idx])):
            q_global = int(q_row_idx[group_idx, row_idx].item())
            torch.testing.assert_close(scattered_out[q_global], ref_out[group_idx, row_idx], atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(scattered_lse[q_global], ref_lse[group_idx, row_idx], atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(combine_out[q_global], ref_out[group_idx, row_idx], atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(combine_lse[q_global], ref_lse[group_idx, row_idx], atol=2e-3, rtol=2e-3)


def test_cached_fused_grad_helper_auto_gate_uses_row_threshold(monkeypatch):
    name = "FLASH_ATTN_HSA_CACHED_FUSED_GRAD_ZERO"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(f"{name}_MAX_ROWS", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_FUSED_GRAD_HELPER_MAX_ROWS", raising=False)

    assert cached_2d._use_cached_fused_grad_helper(name, torch.arange(4096, dtype=torch.int32))
    assert not cached_2d._use_cached_fused_grad_helper(name, torch.arange(4097, dtype=torch.int32))

    monkeypatch.setenv(f"{name}_MAX_ROWS", "8")
    assert cached_2d._use_cached_fused_grad_helper(name, torch.arange(8, dtype=torch.int32))
    assert not cached_2d._use_cached_fused_grad_helper(name, torch.arange(9, dtype=torch.int32))

    monkeypatch.setenv(name, "1")
    assert cached_2d._use_cached_fused_grad_helper(name, torch.arange(9, dtype=torch.int32))

    monkeypatch.setenv(name, "0")
    assert not cached_2d._use_cached_fused_grad_helper(name, torch.arange(8, dtype=torch.int32))


def test_cached_torch_contiguous_grad_helper_env_gate(monkeypatch):
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_TORCH_CONTIG_GRAD_HELPERS", raising=False)
    assert cached_2d._use_torch_contiguous_grad_helpers()

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_TORCH_CONTIG_GRAD_HELPERS", "0")
    assert not cached_2d._use_torch_contiguous_grad_helpers()

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_TORCH_CONTIG_GRAD_HELPERS", "1")
    assert cached_2d._use_torch_contiguous_grad_helpers()


def test_cached_direct_2d_payload_cache_stats_record_hit_and_miss():
    class SyntheticGrid:
        def __init__(self, direct_plan):
            self.forward_execution_plan = {"direct_execution_plan": direct_plan}

    class Runtime:
        def __init__(self, direct_plan):
            self.forward_synthetic_grid = SyntheticGrid(direct_plan)

    direct_plan = {
        "bucket_size": [1],
        "bucket_packed_q": [2],
        "bucket_packed_k": [4],
        "bucket_q_row_range": [(0, 2)],
        "bucket_q_row_idx": torch.tensor([0, 1], dtype=torch.int32),
        "row_compact_plan": {
            "bucket_row_k_range": [(0, 8)],
            "bucket_row_k_length_range": [(0, 2)],
            "bucket_row_k_cap": [4],
            "bucket_row_k_row_idx": torch.tensor([0, 1, 2, 3, 1, 2, 3, 4], dtype=torch.int32),
            "bucket_row_k_length": torch.tensor([4, 4], dtype=torch.int32),
        },
    }
    runtime = Runtime(direct_plan)
    q = torch.empty((1, 2, 2, 64), dtype=torch.bfloat16)
    k = torch.empty((1, 5, 2, 64), dtype=torch.bfloat16)
    v = torch.empty((1, 5, 2, 64), dtype=torch.bfloat16)

    cached_2d.reset_cached_direct_2d_forward_payload_cache_stats(runtime)
    first = cached_2d.build_cached_direct_2d_forward_payload(runtime, q, k, v)
    second = cached_2d.build_cached_direct_2d_forward_payload(runtime, q, k, v)
    third = cached_2d.build_cached_direct_2d_forward_payload(runtime, q, k, v)
    stats = cached_2d.get_cached_direct_2d_forward_payload_cache_stats(runtime)

    assert first["status"] == "ready"
    assert second is first
    assert third is first
    assert stats["calls"] == 3
    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["cache_size"] == 1
    assert stats["last_event"] == "hit"
    assert stats["last_payload_status"] == "ready"
    assert stats["build_seconds_total"] >= 0.0
    assert stats["last_geometry"]["cached_direct_2d_groups"] == 1


def test_cached_backward_key_owned_dkdv_gate_requires_occurrence_payload():
    assert not cached_2d._can_use_cached_backward_key_owned_dkdv(None)

    payload = {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "owned_occurrence_kind": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_range_idx": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.tensor([0], dtype=torch.int32),
        "owned_occurrence_col_idx": torch.tensor([0], dtype=torch.int32),
    }

    assert cached_2d._can_use_cached_backward_key_owned_dkdv(payload)
    payload["owned_k_row_idx"] = torch.empty((0,), dtype=torch.int32)
    assert not cached_2d._can_use_cached_backward_key_owned_dkdv(payload)


def test_cached_backward_key_owned_overwrite_gate_requires_all_kv_rows():
    payload = {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_ptr": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "owned_occurrence_kind": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_range_idx": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_col_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
    }
    k_flat = torch.empty((3, 2, 64), dtype=torch.bfloat16)
    assert cached_2d._cached_backward_key_owned_overwrites_all_kv_rows(payload, k_flat)

    k_flat = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    assert not cached_2d._cached_backward_key_owned_overwrites_all_kv_rows(payload, k_flat)


def test_cached_backward_key_owned_auto_gate_is_small_all_owned_only(monkeypatch):
    payload = {
        "status": "ready",
        "backward_kernel_family": "cached_tc8x8_fused",
        "owned_k_row_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_ptr": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        "owned_occurrence_kind": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_range_idx": torch.tensor([0, 0, 1], dtype=torch.int32),
        "owned_occurrence_tile_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
        "owned_occurrence_col_idx": torch.tensor([0, 1, 2], dtype=torch.int32),
    }
    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED_MAX_ROWS", "3")
    assert cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((3, 2, 64), dtype=torch.bfloat16),
    )
    assert not cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((4, 2, 64), dtype=torch.bfloat16),
    )

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_GENERALIZED_BWD_KEY_OWNED_MAX_ROWS", raising=False)
    payload["owned_k_row_idx"] = torch.arange(128, dtype=torch.int32)
    payload["owned_occurrence_ptr"] = torch.arange(129, dtype=torch.int32)
    payload["owned_occurrence_kind"] = torch.zeros((128,), dtype=torch.int32)
    payload["owned_occurrence_range_idx"] = torch.zeros((128,), dtype=torch.int32)
    payload["owned_occurrence_tile_idx"] = torch.zeros((128,), dtype=torch.int32)
    payload["owned_occurrence_col_idx"] = torch.zeros((128,), dtype=torch.int32)
    assert cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((128, 2, 64), dtype=torch.bfloat16),
    )
    assert not cached_2d._auto_use_cached_backward_key_owned_dkdv(
        payload,
        torch.empty((129, 2, 64), dtype=torch.bfloat16),
    )


def test_cached_backward_direct_dq_auto_gate_uses_row_threshold(monkeypatch):
    payload = {
        "total_rows": 4096,
        "residual_mode": "fused_tail",
        "fused_output_row_count": 4096,
        "fused_q_row_idx": torch.empty((512, 8), dtype=torch.int32),
        "fused_q_length": torch.full((512,), 8, dtype=torch.int32),
    }

    class FakeCudaTensor:
        is_cuda = True
        shape = (4096, 8, 64)

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", raising=False)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_TINY_ROWS", raising=False)
    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_MIN_ROWS", "4096")
    assert cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_MIN_ROWS", "4097")
    assert not cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", "1")
    assert cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", "0")
    assert not cached_2d._use_cached_backward_direct_dq(payload, FakeCudaTensor())

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD", raising=False)
    payload["total_rows"] = 65
    payload["fused_output_row_count"] = 65
    payload["fused_q_row_idx"] = torch.arange(65, dtype=torch.int32).view(13, 5)
    payload["fused_q_length"] = torch.full((13,), 5, dtype=torch.int32)

    class FakeTinyCudaTensor:
        is_cuda = True
        shape = (65, 8, 64)

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_TINY_ROWS", "128")
    assert cached_2d._use_cached_backward_direct_dq(payload, FakeTinyCudaTensor())

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_DQ_BWD_TINY_ROWS", "64")
    assert not cached_2d._use_cached_backward_direct_dq(payload, FakeTinyCudaTensor())


def test_cached_backward_payload_empty_exact_local_k_is_rank2():
    payload = {
        "status": "ready",
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "fused_q_length": torch.tensor([2], dtype=torch.int32),
        "fused_exact_tile_ptr": torch.tensor([0, 0], dtype=torch.int32),
        "fused_exact_k_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "fused_tail_tile_ptr": torch.tensor([0, 1], dtype=torch.int32),
        "fused_tail_k_row_idx": torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "fused_tail_mask_words": torch.tensor([[[0b111], [0b011]]], dtype=torch.int32),
    }

    backward_payload = cached_2d.build_cached_generalized_backward_payload(payload)

    assert backward_payload is not None
    assert tuple(backward_payload["exact_tile_local_k_idx"].shape) == (0, 8)
    assert tuple(backward_payload["tail_tile_local_k_idx"].shape) == (1, 8)


def test_cached_masked_payload_force_combine_for_grouped_scatter(monkeypatch):
    calls = []

    def combine(*args, **kwargs):
        calls.append("combine")

    def scatter_tc(*args, **kwargs):
        calls.append("scatter_tc")

    def scatter(*args, **kwargs):
        calls.append("scatter")

    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_combine_fwd_kernel", combine)
    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_scatter_tc_fwd_kernel", scatter_tc)
    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_scatter_fwd_kernel", scatter)

    payload = {
        "packed_q": 2,
        "support_rows": 4,
        "tile_k": 32,
        "q_row_idx": torch.empty((2, 2), dtype=torch.int32),
        "range_tc_scatter_q_row_idx": torch.tensor([[0, 1]], dtype=torch.int32),
        "range_tc_scatter_k_row_idx": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "range_tc_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_tc_scatter_k_length": torch.tensor([4], dtype=torch.int32),
        "range_tc_scatter_mask_words": torch.tensor([[[0b1111], [0b1111]]], dtype=torch.int32),
        "range_tc_scatter_group_count": 1,
        "range_tc_scatter_row_count": 2,
        "range_scatter_q_row_idx": torch.tensor([[2, 3]], dtype=torch.int32),
        "range_scatter_k_row_idx": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_scatter_k_length": torch.tensor([4], dtype=torch.int32),
        "range_scatter_mask_words": torch.tensor([[[0b1111], [0b1111]]], dtype=torch.int32),
        "range_scatter_union_group_count": 1,
        "range_scatter_union_row_count": 2,
        "range_packed_q_row_idx": torch.empty((0, 2), dtype=torch.int32),
        "range_packed_k_row_idx": torch.empty((0, 4), dtype=torch.int32),
        "range_packed_q_length": torch.empty((0,), dtype=torch.int32),
        "range_packed_k_length": torch.empty((0,), dtype=torch.int32),
        "range_packed_mask_words": torch.empty((0, 2, 1), dtype=torch.int32),
        "range_packed_q_row_idx_flat": torch.empty((0,), dtype=torch.int32),
        "range_packed_k_row_idx_flat": torch.empty((0,), dtype=torch.int32),
    }
    q = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    out = torch.empty((4, 1, 64), dtype=torch.float32)
    lse = torch.empty((4, 1), dtype=torch.float32)

    cached_2d._run_cached_masked_payload_forward(
        payload,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=1.0,
        force_combine_scatter=False,
    )
    assert calls == ["scatter_tc", "scatter"]

    calls.clear()
    cached_2d._run_cached_masked_payload_forward(
        payload,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=1.0,
        force_combine_scatter=True,
    )
    assert calls == ["combine", "combine"]


def test_cached_masked_payload_serializes_duplicate_scatter_groups(monkeypatch):
    calls = []

    def combine(*args, **kwargs):
        q_row_idx = args[3]
        calls.append(tuple(q_row_idx.reshape(-1).tolist()))

    monkeypatch.setattr(cached_2d, "_run_synthetic_2d_masked_gather_combine_fwd_kernel", combine)

    payload = {
        "total_rows": 2,
        "packed_q": 2,
        "support_rows": 4,
        "tile_k": 32,
        "q_row_idx": torch.empty((3, 2), dtype=torch.int32),
        "range_tc_scatter_q_row_idx": torch.empty((0, 2), dtype=torch.int32),
        "range_tc_scatter_k_row_idx": torch.empty((0, 4), dtype=torch.int32),
        "range_tc_scatter_q_length": torch.empty((0,), dtype=torch.int32),
        "range_tc_scatter_k_length": torch.empty((0,), dtype=torch.int32),
        "range_tc_scatter_mask_words": torch.empty((0, 2, 1), dtype=torch.int32),
        "range_tc_scatter_group_count": 0,
        "range_tc_scatter_row_count": 0,
        "range_scatter_q_row_idx": torch.tensor([[0, -1], [0, -1], [1, -1]], dtype=torch.int32),
        "range_scatter_k_row_idx": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([1, 1, 1], dtype=torch.int32),
        "range_scatter_k_length": torch.tensor([4, 4, 4], dtype=torch.int32),
        "range_scatter_mask_words": torch.tensor(
            [[[0b1111], [0]], [[0b1111], [0]], [[0b1111], [0]]],
            dtype=torch.int32,
        ),
        "range_scatter_union_group_count": 3,
        "range_scatter_union_row_count": 3,
        "range_packed_q_row_idx": torch.empty((0, 2), dtype=torch.int32),
        "range_packed_k_row_idx": torch.empty((0, 4), dtype=torch.int32),
        "range_packed_q_length": torch.empty((0,), dtype=torch.int32),
        "range_packed_k_length": torch.empty((0,), dtype=torch.int32),
        "range_packed_mask_words": torch.empty((0, 2, 1), dtype=torch.int32),
        "range_packed_q_row_idx_flat": torch.empty((0,), dtype=torch.int32),
        "range_packed_k_row_idx_flat": torch.empty((0,), dtype=torch.int32),
    }
    q = torch.empty((2, 1, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 1, 64), dtype=torch.bfloat16)
    out = torch.empty((2, 1, 64), dtype=torch.float32)
    lse = torch.empty((2, 1), dtype=torch.float32)

    cached_2d._run_cached_masked_payload_forward(
        payload,
        q,
        k,
        v,
        out,
        lse,
        softmax_scale=1.0,
        force_combine_scatter=True,
    )

    assert calls == [(0, -1), (0, -1, 1, -1)]


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


def test_attach_precomputed_cached_payload_preserves_container_and_clears_caches():
    class Schedule:
        pass

    schedule = Schedule()
    schedule._precomputed_forward_direct_plan_payload = {
        "version": 7,
        "entries": [
            {
                "forward_block_q": 128,
                "logical_block_q": 64,
                "logical_block_k": 64,
                "max_packed_k": 128,
                "max_direct_segments": 4,
                "direct_execution_plan": {"kept": True},
            }
        ],
    }
    schedule._precomputed_cached_generalized_forward_payload_device_cache = {"stale": object()}
    schedule._resolved_cached_generalized_forward_payload_fast_cache = {"stale": object()}
    schedule._resolved_cached_generalized_forward_payload_cache = {"stale": object()}
    cached_payload = {"status": "ready", "reason": "unit"}

    returned = cached_2d.attach_precomputed_cached_generalized_forward_payload(
        schedule,
        cached_payload,
        forward_block_q=128,
        logical_block_q=64,
        logical_block_k=64,
        max_packed_k=128,
        max_direct_segments=4,
    )

    assert returned is schedule
    container = schedule._precomputed_forward_direct_plan_payload
    assert container["version"] == 7
    assert len(container["entries"]) == 1
    assert container["entries"][0]["direct_execution_plan"] == {"kept": True}
    assert container["entries"][0]["cached_generalized_forward_payload"] is cached_payload
    assert not hasattr(schedule, "_precomputed_cached_generalized_forward_payload_device_cache")
    assert not hasattr(schedule, "_resolved_cached_generalized_forward_payload_fast_cache")
    assert not hasattr(schedule, "_resolved_cached_generalized_forward_payload_cache")


def test_cached_2d_monolithic_forward_requires_complete_fused_tail_payload():
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    payload = {
        "total_rows": 4,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32),
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


def test_cached_2d_forward_prefers_monolithic_clean_fused_tail(monkeypatch):
    payload = {
        "status": "ready",
        "packed_q": 16,
        "support_rows": 32,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "exact_dense_q_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((0, 16), dtype=torch.int32),
    }
    q = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    k = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    v = torch.empty((4, 2, 64), dtype=torch.bfloat16)
    calls = []

    def monolithic(*args, **kwargs):
        calls.append("monolithic")
        return "monolithic"

    def direct_final(*args, **kwargs):
        raise AssertionError("direct-final residual path should not run for clean fused-tail payloads")

    monkeypatch.setattr(cached_2d, "_run_cached_monolithic_fused_tail_forward", monolithic)
    monkeypatch.setattr(cached_2d, "_run_cached_direct_final_residual_forward", direct_final)

    assert cached_2d.run_cached_direct_2d_forward(payload, q, k, v) == "monolithic"
    assert calls == ["monolithic"]


def test_cached_2d_direct_final_residual_allows_scatter_only_full_coverage():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 1,
        "range_scatter_row_count": 1,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_residual_allows_exact_dense_base_coverage():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 0,
        "exact_dense_output_row_count": 4,
        "range_tc_scatter_row_count": 1,
        "range_scatter_row_count": 1,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.tensor(
            [[4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.int32,
        ),
        "range_scatter_q_row_idx": torch.tensor(
            [[5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.int32,
        ),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.0},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_residual_gates_packed_overlap_without_fp32_combine(monkeypatch):
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.empty((1, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 6,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 0,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", raising=False)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", "0")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_online_combine_requires_fp32_accum"
    )

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", "1")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", raising=False)

    payload["fused_output_row_count"] = 4
    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32)
    payload["range_packed_q_row_idx"] = torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )

    payload["fused_output_row_count"] = 6
    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
    payload["range_scatter_row_count"] = 1
    payload["range_scatter_q_row_idx"] = torch.tensor([[4, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_ONLINE_COMBINE", "0")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_online_combine_requires_fp32_accum"
    )

    payload["fused_output_row_count"] = 4
    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32)
    payload["range_packed_q_row_idx"] = torch.tensor([[3, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "mixed_residual_incomplete_direct_final_row_coverage"
    )


def test_cached_2d_direct_final_residual_initializes_missing_packed_rows():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 0,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]

    payload["fused_q_row_idx"] = torch.tensor([[0, 1, 2, 3, 4, 5, -1, -1]], dtype=torch.int32)
    payload.pop("_workspace", None)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]


def test_cached_2d_direct_final_base_rows_trim_in_payload_order():
    payload = {
        "total_rows": 8,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[4, 5, 6, 7, 0, 1, 2, 3]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 4,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([4], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (8, 2, 64)

    base_rows = cached_2d._get_direct_final_base_output_row_idx(payload, torch.device("cpu"), base_source="fused")
    assert base_rows.tolist() == [4, 5, 6, 7]
    assert cached_2d._direct_final_base_residual_union_row_count(payload, torch.device("cpu")) == 8
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_residual_initializes_missing_online_scatter_rows():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 3,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[3, 4, 5, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([3], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_requires_online_combine(payload, torch.device("cpu"), base_source="fused")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]


def test_cached_2d_direct_final_allows_serial_mixed_residual_overlap():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[3, 4, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[4, 5, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_packed_q_length": torch.tensor([2], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_requires_online_combine(payload, torch.device("cpu"), base_source="fused")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]


def test_cached_2d_direct_final_mixed_disjoint_residual_keeps_scatter_direct(monkeypatch):
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 1,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[4, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([1], dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[5, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_packed_q_length": torch.tensor([1], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }
    q = torch.empty((6, 2, 64), dtype=torch.bfloat16)
    k = torch.empty((6, 2, 64), dtype=torch.bfloat16)
    v = torch.empty((6, 2, 64), dtype=torch.bfloat16)
    work = torch.zeros((6, 2, 64), dtype=torch.float32)
    lse = torch.zeros((6, 2), dtype=torch.float32)
    calls = []

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_MONOLITHIC_FWD", "1")
    monkeypatch.setattr(cached_2d, "_cached_direct_final_residual_support_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_get_cached_direct_2d_final_buffers", lambda *args, **kwargs: (work, lse))
    monkeypatch.setattr(cached_2d, "_run_cached_fused_exact_tail_ranges", lambda *args, **kwargs: (1, 4))

    def masked_payload(*args, **kwargs):
        calls.append(bool(kwargs["force_combine_scatter"]))
        return 0, 0, 1, 2

    monkeypatch.setattr(cached_2d, "_run_cached_masked_payload_forward", masked_payload)
    monkeypatch.setattr(cached_2d, "_record_union_runtime_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_record_exact_dense_runtime_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_record_fused_runtime_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_record_cached_forward_path", lambda *args, **kwargs: None)

    out = cached_2d._run_cached_direct_final_residual_forward(
        payload,
        q,
        q,
        k,
        v,
        softmax_scale=1.0,
        return_lse=False,
        lse_layout="flat",
    )

    assert out is work
    assert calls == [False]


def test_cached_2d_direct_final_online_combine_cast_mode_gate(monkeypatch):
    payload = {
        "total_rows": 2,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "exact_dense_q_row_idx": torch.empty((0, 8), dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "q_length": torch.tensor([2], dtype=torch.int32),
        "fused_output_row_count": 0,
        "exact_dense_output_row_count": 0,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[0, 1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.0},
    }
    q = torch.empty((2, 1, 64), dtype=torch.bfloat16)
    k = torch.empty((2, 1, 64), dtype=torch.bfloat16)
    v = torch.empty((2, 1, 64), dtype=torch.bfloat16)
    work = torch.ones((2, 1, 64), dtype=torch.float32)
    lse = torch.empty((2, 1), dtype=torch.float32)
    calls = []

    monkeypatch.setattr(cached_2d, "_cached_direct_final_residual_support_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_direct_final_requires_online_combine", lambda *args, **kwargs: True)
    monkeypatch.setattr(cached_2d, "_get_cached_direct_2d_output_buffers", lambda *args, **kwargs: (work, lse))
    monkeypatch.setattr(cached_2d, "_direct_final_base_residual_union_row_count", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        cached_2d,
        "_get_direct_final_missing_init_row_idx",
        lambda *args, **kwargs: torch.empty(0, dtype=torch.int32),
    )
    monkeypatch.setattr(cached_2d, "_run_cached_masked_payload_forward", lambda *args, **kwargs: (0, 0, 1, 2))
    monkeypatch.setattr(cached_2d, "_record_union_runtime_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_record_exact_dense_runtime_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_record_fused_runtime_geometry", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_record_cached_forward_path", lambda *args, **kwargs: None)
    monkeypatch.setattr(cached_2d, "_get_cached_all_row_idx", lambda *args, **kwargs: torch.arange(2, dtype=torch.int32))

    def cast_all(src, dst):
        calls.append(("all", src.shape, dst.shape))

    def cast_indexed(src, row_idx, dst):
        calls.append(("indexed", row_idx.tolist(), src.shape, dst.shape))

    monkeypatch.setattr(cached_2d, "_run_cached_cast_all_rows_kernel", cast_all)
    monkeypatch.setattr(cached_2d, "_run_cached_cast_rows_kernel", cast_indexed)
    monkeypatch.delenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_CONTIG_CAST", raising=False)
    out = cached_2d._run_cached_direct_final_residual_forward(
        payload,
        q,
        q,
        k,
        v,
        softmax_scale=1.0,
        return_lse=False,
        lse_layout="flat",
    )
    assert calls == []
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), torch.ones_like(out, dtype=torch.float32))

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_CONTIG_CAST", "cute")
    cached_2d._run_cached_direct_final_residual_forward(
        payload,
        q,
        q,
        k,
        v,
        softmax_scale=1.0,
        return_lse=False,
        lse_layout="flat",
    )
    assert calls[-1][0] == "all"

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_CONTIG_CAST", "0")
    cached_2d._run_cached_direct_final_residual_forward(
        payload,
        q,
        q,
        k,
        v,
        softmax_scale=1.0,
        return_lse=False,
        lse_layout="flat",
    )
    assert calls[-1][0] == "indexed"


def test_cached_2d_direct_final_blocks_duplicate_rows_inside_residual_kernel():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 5, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((1, 16), dtype=torch.int32),
        "fused_output_row_count": 5,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor([[4, 4, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_length": torch.tensor([2], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_has_duplicate_residual_rows_within_kernel(payload, torch.device("cpu"))
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_duplicate_residual_rows_within_kernel"
    )


def test_cached_2d_direct_final_allows_serial_duplicate_residual_groups():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 5, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 5,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor(
            [
                [4, -1, -1, -1, -1, -1, -1, -1],
                [4, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
        "range_scatter_q_length": torch.tensor([1, 1], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert cached_2d._direct_final_has_duplicate_residual_rows_within_kernel(payload, torch.device("cpu"))
    assert cached_2d._direct_final_can_serialize_duplicate_residual_rows(payload, torch.device("cpu"))
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )


def test_cached_2d_direct_final_gates_too_many_duplicate_residual_ranges(monkeypatch):
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, 5, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 5,
        "range_tc_scatter_row_count": 0,
        "range_scatter_row_count": 2,
        "range_packed_group_count": 0,
        "range_tc_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_scatter_q_row_idx": torch.tensor(
            [
                [4, -1, -1, -1, -1, -1, -1, -1],
                [4, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        ),
        "range_scatter_q_length": torch.tensor([1, 1], dtype=torch.int32),
        "range_packed_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    monkeypatch.setenv("FLASH_ATTN_HSA_CACHED_DIRECT_FINAL_DUP_SERIAL_MAX_RANGES", "1")
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "direct_final_duplicate_residual_rows_within_kernel"
    )


def test_cached_2d_direct_final_residual_initializes_missing_mixed_rows():
    payload = {
        "total_rows": 6,
        "residual_mode": "fused_tail",
        "exact_kernel_family": "tc8x8",
        "exact_dense_rows_per_range": 8,
        "exact_dense_keys_per_tile": 8,
        "fused_q_row_idx": torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
        "q_row_idx": torch.empty((2, 16), dtype=torch.int32),
        "fused_output_row_count": 4,
        "range_tc_scatter_row_count": 1,
        "range_scatter_row_count": 0,
        "range_packed_group_count": 1,
        "range_tc_scatter_q_row_idx": torch.tensor([[4, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "range_scatter_q_row_idx": torch.empty((0, 16), dtype=torch.int32),
        "range_packed_q_row_idx": torch.tensor([[5, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32),
        "geometry": {"fused_total_coverage_frac": 0.8},
    }

    class FakeCudaTensor:
        is_cuda = True
        dtype = torch.bfloat16
        shape = (6, 2, 64)

    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        is None
    )
    missing = cached_2d._get_direct_final_missing_init_row_idx(payload, torch.device("cpu"))
    assert missing.tolist() == [4, 5]

    payload["range_packed_q_row_idx"] = torch.tensor([[3, -1, -1, -1, -1, -1, -1, -1]], dtype=torch.int32)
    assert (
        cached_2d._cached_direct_final_residual_support_reason(
            payload,
            FakeCudaTensor(),
            FakeCudaTensor(),
            FakeCudaTensor(),
        )
        == "mixed_residual_incomplete_direct_final_row_coverage"
    )
