import pytest
import torch


def _reference_step(p, edge_prob, src, dst, node_is_sink):
    p_next = torch.where(node_is_sink[:, None], p, torch.zeros_like(p))
    if src.numel() > 0:
        p_next.index_add_(0, dst, p[src] * edge_prob)
    return p_next


def _reference_outgoing_softmax(edge_scores, src, *, n_nodes):
    out = torch.empty_like(edge_scores)
    for node_idx in range(n_nodes):
        mask = src == node_idx
        if bool(mask.any().item()):
            out[mask] = torch.softmax(edge_scores[mask], dim=0)
    return out


def test_build_incoming_edge_csr_groups_edges_by_destination():
    from flash_attn.cute.arhsa_walk_sm100 import build_incoming_edge_csr

    dst = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int32)
    row_ptr, edge_idx = build_incoming_edge_csr(dst, n_nodes=4)

    assert row_ptr.cpu().tolist() == [0, 2, 3, 5, 5]
    assert edge_idx.cpu().tolist() == [1, 4, 3, 0, 2]


def test_build_outgoing_edge_csr_groups_edges_by_source():
    from flash_attn.cute.arhsa_walk_sm100 import build_outgoing_edge_csr

    src = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int32)
    row_ptr, edge_idx = build_outgoing_edge_csr(src, n_nodes=4)

    assert row_ptr.cpu().tolist() == [0, 2, 3, 5, 5]
    assert edge_idx.cpu().tolist() == [1, 4, 3, 0, 2]


def test_build_query_leaf_csr_groups_leaf_entries_by_query():
    from flash_attn.cute.arhsa_walk_sm100 import build_query_leaf_csr

    leaf_query_index = torch.tensor([2, 0, 2, 1, 0], dtype=torch.int32)
    row_ptr, leaf_entries = build_query_leaf_csr(leaf_query_index, n_queries=4)

    assert row_ptr.cpu().tolist() == [0, 2, 3, 5, 5]
    assert leaf_entries.cpu().tolist() == [1, 4, 3, 0, 2]


def test_outgoing_softmax_from_scores_matches_grouped_reference():
    from flash_attn.cute.arhsa_walk_sm100 import outgoing_softmax_from_scores

    src = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64)
    edge_scores = torch.randn(6, 3)

    got = outgoing_softmax_from_scores(edge_scores, src, n_nodes=4)
    expected = _reference_outgoing_softmax(edge_scores, src, n_nodes=4)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_outgoing_softmax_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_outgoing_edge_csr,
        run_arhsa_outgoing_softmax,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64, device=device)
    edge_scores = torch.randn(6, 3, device=device, dtype=torch.float32)
    row_ptr, edge_idx = build_outgoing_edge_csr(src, n_nodes=4)

    got = run_arhsa_outgoing_softmax(edge_scores, row_ptr, edge_idx, n_nodes=4)
    expected = _reference_outgoing_softmax(edge_scores, src, n_nodes=4)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_outgoing_softmax_backward_matches_torch_autograd():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_outgoing_edge_csr,
        run_arhsa_outgoing_softmax,
        run_arhsa_outgoing_softmax_backward,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int64, device=device)
    edge_scores_data = torch.randn(6, 3, device=device, dtype=torch.float32)
    grad_prob = torch.randn(6, 3, device=device, dtype=torch.float32)
    row_ptr, edge_idx = build_outgoing_edge_csr(src, n_nodes=4)

    edge_prob = run_arhsa_outgoing_softmax(edge_scores_data, row_ptr, edge_idx, n_nodes=4)
    got = run_arhsa_outgoing_softmax_backward(edge_prob, grad_prob, row_ptr, edge_idx, n_nodes=4)

    edge_scores_ref = edge_scores_data.clone().requires_grad_()
    expected_prob = _reference_outgoing_softmax(edge_scores_ref, src, n_nodes=4)
    expected_prob.backward(grad_prob)

    torch.testing.assert_close(got, edge_scores_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_markov_incoming_step_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_incoming_edge_csr,
        run_arhsa_markov_incoming_step,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 3
    p = torch.randn(n_nodes, n_heads, device=device, dtype=torch.float32)
    edge_prob = torch.rand(src.numel(), n_heads, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, False, True], device=device)
    row_ptr, edge_idx = build_incoming_edge_csr(dst, n_nodes=n_nodes)

    got = run_arhsa_markov_incoming_step(
        p,
        edge_prob,
        src,
        row_ptr,
        edge_idx,
        node_is_sink,
    )
    expected = _reference_step(p, edge_prob, src.long(), dst.long(), node_is_sink)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_markov_backward_step_matches_torch_autograd():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_outgoing_edge_csr,
        run_arhsa_markov_backward_step,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 3
    p_prev_data = torch.randn(n_nodes, n_heads, device=device, dtype=torch.float32).abs()
    edge_prob_data = torch.rand(src.numel(), n_heads, device=device, dtype=torch.float32)
    grad_next = torch.randn(n_nodes, n_heads, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, False, True], device=device)
    row_ptr, edge_idx = build_outgoing_edge_csr(src, n_nodes=n_nodes)
    grad_edge_prob = torch.zeros_like(edge_prob_data)

    got_p = run_arhsa_markov_backward_step(
        grad_next,
        p_prev_data,
        edge_prob_data,
        row_ptr,
        edge_idx,
        dst,
        node_is_sink,
        grad_edge_prob,
    )

    p_prev_ref = p_prev_data.clone().requires_grad_()
    edge_prob_ref = edge_prob_data.clone().requires_grad_()
    p_next = _reference_step(p_prev_ref, edge_prob_ref, src.long(), dst.long(), node_is_sink)
    p_next.backward(grad_next)

    torch.testing.assert_close(got_p, p_prev_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(grad_edge_prob, edge_prob_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_markov_walk_fixed_iters_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_incoming_edge_csr,
        run_arhsa_markov_walk_fixed_iters,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 3
    p = torch.randn(n_nodes, n_heads, device=device, dtype=torch.float32).abs()
    edge_prob = torch.rand(src.numel(), n_heads, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, False, True], device=device)
    row_ptr, edge_idx = build_incoming_edge_csr(dst, n_nodes=n_nodes)

    got = run_arhsa_markov_walk_fixed_iters(
        p,
        edge_prob,
        src,
        row_ptr,
        edge_idx,
        node_is_sink,
        n_iters=4,
    )
    expected = p
    for _ in range(4):
        expected = _reference_step(expected, edge_prob, src.long(), dst.long(), node_is_sink)

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


def _reference_readout(p, leaf_node_index, leaf_query_index, leaf_value_index, value, *, n_queries):
    leaf_mass = p[leaf_node_index]
    denom = torch.zeros(n_queries, p.shape[1], dtype=p.dtype, device=p.device)
    denom.index_add_(0, leaf_query_index, leaf_mass)
    leaf_attn = leaf_mass / denom[leaf_query_index].clamp(min=1e-8)
    readout = torch.zeros(
        n_queries,
        value.shape[1],
        value.shape[2],
        dtype=value.dtype,
        device=value.device,
    )
    readout.index_add_(0, leaf_query_index, leaf_attn.unsqueeze(-1).to(value.dtype) * value[leaf_value_index])
    return readout, leaf_attn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        run_arhsa_leaf_readout,
    )

    device = torch.device("cuda")
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p = torch.rand(n_nodes, n_heads, device=device, dtype=torch.float32)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.float32)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )

    got = run_arhsa_leaf_readout(
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        n_queries=n_queries,
    )
    expected, _ = _reference_readout(
        p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("head_dim_v", [5, 64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_query_warp_matches_torch_reference(head_dim_v):
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        run_arhsa_leaf_readout,
    )

    device = torch.device("cuda")
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    p = torch.rand(n_nodes, n_heads, device=device, dtype=torch.float32)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.float32)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )
    denom = torch.empty(n_queries, n_heads, device=device, dtype=torch.float32)

    got = run_arhsa_leaf_readout(
        p,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value,
        n_queries=n_queries,
        denom=denom,
        query_warp=True,
    )
    expected, _ = _reference_readout(
        p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )
    expected_denom = torch.zeros_like(denom)
    expected_denom.index_add_(0, leaf_query_index, p[leaf_node_index])

    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(denom, expected_denom.clamp(min=1e-8), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_pack_leaf_values_matches_gather():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import run_arhsa_pack_leaf_values

    device = torch.device("cuda")
    torch.manual_seed(43)
    value = torch.randn(7, 2, 5, device=device, dtype=torch.bfloat16)
    leaf_value_index = torch.tensor([0, 3, 1, 3, 6, 2], dtype=torch.int64, device=device)

    got = run_arhsa_pack_leaf_values(leaf_value_index, value)

    torch.testing.assert_close(got, value[leaf_value_index], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_writes_reusable_denominator():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout,
        run_arhsa_leaf_readout_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(41)
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.float32)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value_data = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.float32)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.float32)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )
    denom = torch.empty(n_queries, n_heads, device=device, dtype=torch.float32)

    got_readout = run_arhsa_leaf_readout(
        p_data,
        leaf_node_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        n_queries=n_queries,
        denom=denom,
    )
    expected_readout, _ = readout_arhsa_leaf_attention(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_data,
        n_queries=n_queries,
    )
    expected_denom = torch.zeros_like(denom)
    expected_denom.index_add_(0, leaf_query_index, p_data[leaf_node_index])

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        denom=denom,
        leaf_major_stats=True,
        denom_precomputed=True,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_readout, expected_readout, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(denom, expected_denom.clamp(min=1e-8), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_p, p_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_value, value_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_backward_matches_torch_autograd():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(17)
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.float32)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value_data = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.float32)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.float32)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p, p_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_value, value_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_backward_matches_torch_autograd_bfloat16():
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(23)
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value_data = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    assert got_p.dtype is torch.bfloat16
    assert got_value.dtype is torch.bfloat16
    torch.testing.assert_close(got_p, p_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_value, value_ref.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_backward_fused_small_matches_torch_autograd_bfloat16():
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(31)
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value_data = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        max_leaves_per_query=3,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p, p_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_value, value_ref.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_backward_leaf_major_stats_matches_torch_autograd_bfloat16():
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(37)
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value_data = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        leaf_major_stats=True,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p, p_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_value, value_ref.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_backward_packed_stats_matches_torch_autograd_bfloat16():
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout_backward,
        run_arhsa_pack_leaf_values,
    )

    device = torch.device("cuda")
    torch.manual_seed(47)
    n_queries = 3
    n_nodes = 7
    n_heads = 2
    head_dim_v = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 0], dtype=torch.int64, device=device)
    value_data = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )
    packed_value = run_arhsa_pack_leaf_values(leaf_value_index, value_data)

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        leaf_major_stats=True,
        packed_value=packed_value,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p, p_ref.grad, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_value, value_ref.grad, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("query_warp_scatter", [False, True])
@pytest.mark.parametrize("query_warp_fused", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_leaf_readout_backward_query_warp_stats_matches_torch_autograd_bfloat16(
    query_warp_scatter,
    query_warp_fused,
):
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_query_leaf_csr,
        readout_arhsa_leaf_attention,
        run_arhsa_leaf_readout,
        run_arhsa_leaf_readout_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(53)
    n_queries = 4
    n_nodes = 11
    n_heads = 3
    head_dim_v = 64
    leaves_per_query = 5
    p_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    leaf_query_index = torch.arange(n_queries, dtype=torch.int64, device=device).repeat_interleave(
        leaves_per_query
    )
    leaf_node_index = torch.randint(n_nodes, (n_queries * leaves_per_query,), dtype=torch.int64, device=device)
    leaf_value_index = torch.randint(n_nodes, (n_queries * leaves_per_query,), dtype=torch.int64, device=device)
    value_data = torch.randn(n_nodes, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )
    denom = None
    denom_precomputed = False
    if query_warp_fused:
        denom = torch.empty(n_queries, n_heads, device=device, dtype=torch.float32)
        run_arhsa_leaf_readout(
            p_data,
            leaf_node_index,
            leaf_value_index,
            query_leaf_row_ptr,
            query_leaf_entry_index,
            value_data,
            n_queries=n_queries,
            denom=denom,
            query_warp=True,
        )
        denom_precomputed = True

    got_p, got_value = run_arhsa_leaf_readout_backward(
        p_data,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        query_leaf_row_ptr,
        query_leaf_entry_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        denom=denom,
        denom_precomputed=denom_precomputed,
        leaf_major_stats=True,
        query_warp_stats=True,
        query_warp_scatter=query_warp_scatter,
        query_warp_fused=query_warp_fused,
    )

    p_ref = p_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _ = readout_arhsa_leaf_attention(
        p_ref,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p, p_ref.grad, atol=7e-2, rtol=7e-2)
    torch.testing.assert_close(got_value, value_ref.grad, atol=7e-2, rtol=7e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_fixed_iters_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        build_incoming_edge_csr,
        build_query_leaf_csr,
        run_arhsa_walk_readout_fixed_iters,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    head_dim_v = 5
    p0 = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.float32)
    p0[0] = torch.tensor([1.0, 0.7], device=device)
    p0[2] = torch.tensor([0.0, 0.3], device=device)
    edge_prob = torch.rand(src.numel(), n_heads, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)
    value = torch.randn(3, n_heads, head_dim_v, device=device, dtype=torch.float32)
    row_ptr, edge_idx = build_incoming_edge_csr(dst, n_nodes=n_nodes)
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=n_queries,
    )

    readout, leaf_attn, p_final = run_arhsa_walk_readout_fixed_iters(
        p0,
        edge_prob,
        src,
        row_ptr,
        edge_idx,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=4,
        query_leaf_row_ptr=query_leaf_row_ptr,
        query_leaf_entry_index=query_leaf_entry_index,
    )
    fast_readout, fast_leaf_attn, fast_p_final = run_arhsa_walk_readout_fixed_iters(
        p0,
        edge_prob,
        src,
        row_ptr,
        edge_idx,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=4,
        query_leaf_row_ptr=query_leaf_row_ptr,
        query_leaf_entry_index=query_leaf_entry_index,
        return_leaf_attn=False,
    )

    expected_p = p0
    for _ in range(4):
        expected_p = _reference_step(expected_p, edge_prob, src.long(), dst.long(), node_is_sink)
    expected_readout, expected_leaf_attn = _reference_readout(
        expected_p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )

    torch.testing.assert_close(p_final, expected_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(leaf_attn, expected_leaf_attn, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(readout, expected_readout, atol=1e-6, rtol=1e-6)
    assert fast_leaf_attn is None
    torch.testing.assert_close(fast_p_final, expected_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_readout, expected_readout, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_from_scores_fixed_iters_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        run_arhsa_walk_readout_from_scores_fixed_iters,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    p0 = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.float32)
    p0[0] = torch.tensor([1.0, 0.7], device=device)
    p0[2] = torch.tensor([0.0, 0.3], device=device)
    edge_scores = torch.randn(src.numel(), n_heads, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)
    value = torch.randn(3, n_heads, 5, device=device, dtype=torch.float32)

    readout, leaf_attn, p_final, edge_prob = run_arhsa_walk_readout_from_scores_fixed_iters(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=4,
    )
    fast_readout, fast_leaf_attn, fast_p_final, fast_edge_prob = (
        run_arhsa_walk_readout_from_scores_fixed_iters(
            p0,
            edge_scores,
            src,
            dst,
            node_is_sink,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            n_queries=n_queries,
            n_iters=4,
            return_leaf_attn=False,
        )
    )

    expected_readout, expected_leaf_attn, expected_p, expected_edge_prob = (
        torch_arhsa_walk_readout_from_scores_fixed_iters(
            p0,
            edge_scores,
            src,
            dst,
            node_is_sink,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            n_queries=n_queries,
            n_iters=4,
        )
    )
    manual_expected_readout, manual_expected_leaf_attn = _reference_readout(
        expected_p,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
    )
    torch.testing.assert_close(manual_expected_leaf_attn, expected_leaf_attn, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(manual_expected_readout, expected_readout, atol=1e-6, rtol=1e-6)

    torch.testing.assert_close(edge_prob, expected_edge_prob, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(p_final, expected_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(leaf_attn, expected_leaf_attn, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(readout, expected_readout, atol=1e-6, rtol=1e-6)
    assert fast_leaf_attn is None
    torch.testing.assert_close(fast_edge_prob, expected_edge_prob, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_p_final, expected_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(fast_readout, expected_readout, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_from_scores_matches_torch_reference_with_shadow_entry():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        run_arhsa_walk_readout_from_scores_fixed_iters,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    # Node 4 is an absorbing shadow node. Readout folds its final mass into
    # query 0 by giving it the same value row as the current query leaf.
    src = torch.tensor([0, 0, 1, 2, 3], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4], dtype=torch.int32, device=device)
    n_nodes = 5
    n_heads = 2
    n_queries = 1
    p0 = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.float32)
    p0[0] = torch.tensor([1.0, 0.8], device=device)
    p0[2] = torch.tensor([0.0, 0.2], device=device)
    edge_scores = torch.randn(src.numel(), n_heads, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 0], dtype=torch.int64, device=device)
    value = torch.randn(1, n_heads, 5, device=device, dtype=torch.float32)

    readout, leaf_attn, p_final, edge_prob = run_arhsa_walk_readout_from_scores_fixed_iters(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=5,
    )
    expected_readout, expected_leaf_attn, expected_p, expected_edge_prob = (
        torch_arhsa_walk_readout_from_scores_fixed_iters(
            p0,
            edge_scores,
            src,
            dst,
            node_is_sink,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            n_queries=n_queries,
            n_iters=5,
        )
    )

    torch.testing.assert_close(edge_prob, expected_edge_prob, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(p_final, expected_p, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(leaf_attn, expected_leaf_attn, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(readout, expected_readout, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ],
)
def test_arhsa_walk_readout_from_scores_matches_torch_reference_low_precision(dtype, atol, rtol):
    pytest.importorskip("cutlass")
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        run_arhsa_walk_readout_from_scores_fixed_iters,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    torch.manual_seed(3)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    p0 = torch.zeros(n_nodes, n_heads, device=device, dtype=dtype)
    p0[0] = torch.tensor([1.0, 0.7], device=device, dtype=dtype)
    p0[2] = torch.tensor([0.0, 0.3], device=device, dtype=dtype)
    edge_scores = torch.randn(src.numel(), n_heads, device=device, dtype=dtype)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)
    value = torch.randn(3, n_heads, 5, device=device, dtype=dtype)

    readout, leaf_attn, p_final, edge_prob = run_arhsa_walk_readout_from_scores_fixed_iters(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=4,
        return_leaf_attn=False,
    )
    expected_readout, _, expected_p, expected_edge_prob = (
        torch_arhsa_walk_readout_from_scores_fixed_iters(
            p0,
            edge_scores,
            src,
            dst,
            node_is_sink,
            leaf_node_index,
            leaf_query_index,
            leaf_value_index,
            value,
            n_queries=n_queries,
            n_iters=4,
        )
    )

    assert leaf_attn is None
    torch.testing.assert_close(edge_prob, expected_edge_prob, atol=atol, rtol=rtol)
    torch.testing.assert_close(p_final, expected_p, atol=atol, rtol=rtol)
    torch.testing.assert_close(readout, expected_readout, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_autograd_backward_matches_torch_reference():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        arhsa_walk_readout_from_scores_fixed_iters_autograd,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    torch.manual_seed(11)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    p0_data = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.float32)
    p0_data[0] = torch.tensor([1.0, 0.7], device=device)
    p0_data[2] = torch.tensor([0.0, 0.3], device=device)
    edge_scores_data = torch.randn(src.numel(), n_heads, device=device, dtype=torch.float32)
    value_data = torch.randn(3, n_heads, 5, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)

    p0_fast = p0_data.clone().requires_grad_()
    edge_scores_fast = edge_scores_data.clone().requires_grad_()
    value_fast = value_data.clone().requires_grad_()
    readout_fast = arhsa_walk_readout_from_scores_fixed_iters_autograd(
        p0_fast,
        edge_scores_fast,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_fast,
        n_queries=n_queries,
        n_iters=4,
    )

    p0_ref = p0_data.clone().requires_grad_()
    edge_scores_ref = edge_scores_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
        p0_ref,
        edge_scores_ref,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
        n_iters=4,
    )

    torch.testing.assert_close(readout_fast, readout_ref, atol=1e-6, rtol=1e-6)
    grad_readout = torch.randn_like(readout_ref)
    readout_fast.backward(grad_readout)
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(p0_fast.grad, p0_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(edge_scores_fast.grad, edge_scores_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(value_fast.grad, value_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_explicit_backward_matches_torch_autograd():
    from flash_attn.cute.arhsa_walk_sm100 import (
        torch_arhsa_walk_readout_from_scores_fixed_iters,
        torch_arhsa_walk_readout_from_scores_fixed_iters_backward,
    )

    device = torch.device("cuda")
    torch.manual_seed(13)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    p0_data = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.float32)
    p0_data[0] = torch.tensor([1.0, 0.7], device=device)
    p0_data[2] = torch.tensor([0.0, 0.3], device=device)
    edge_scores_data = torch.randn(src.numel(), n_heads, device=device, dtype=torch.float32)
    value_data = torch.randn(3, n_heads, 5, device=device, dtype=torch.float32)
    grad_readout = torch.randn(n_queries, n_heads, 5, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)

    got_p0, got_edge_scores, got_value = torch_arhsa_walk_readout_from_scores_fixed_iters_backward(
        p0_data,
        edge_scores_data,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        n_iters=4,
    )

    p0_ref = p0_data.clone().requires_grad_()
    edge_scores_ref = edge_scores_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
        p0_ref,
        edge_scores_ref,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
        n_iters=4,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p0, p0_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_edge_scores, edge_scores_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_value, value_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_cute_backward_matches_torch_autograd():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import (
        run_arhsa_walk_readout_from_scores_fixed_iters_backward,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    torch.manual_seed(19)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    p0_data = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.float32)
    p0_data[0] = torch.tensor([1.0, 0.7], device=device)
    p0_data[2] = torch.tensor([0.0, 0.3], device=device)
    edge_scores_data = torch.randn(src.numel(), n_heads, device=device, dtype=torch.float32)
    value_data = torch.randn(3, n_heads, 5, device=device, dtype=torch.float32)
    grad_readout = torch.randn(n_queries, n_heads, 5, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)

    got_p0, got_edge_scores, got_value = run_arhsa_walk_readout_from_scores_fixed_iters_backward(
        p0_data,
        edge_scores_data,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        n_iters=4,
    )

    p0_ref = p0_data.clone().requires_grad_()
    edge_scores_ref = edge_scores_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
        p0_ref,
        edge_scores_ref,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
        n_iters=4,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(got_p0, p0_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_edge_scores, edge_scores_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(got_value, value_ref.grad, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_cute_backward_matches_torch_autograd_bfloat16():
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        run_arhsa_walk_readout_from_scores_fixed_iters_backward,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    torch.manual_seed(29)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    n_nodes = 6
    n_heads = 2
    n_queries = 2
    p0_data = torch.zeros(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    p0_data[0] = torch.tensor([1.0, 0.7], device=device, dtype=torch.bfloat16)
    p0_data[2] = torch.tensor([0.0, 0.3], device=device, dtype=torch.bfloat16)
    edge_scores_data = torch.randn(src.numel(), n_heads, device=device, dtype=torch.bfloat16)
    value_data = torch.randn(3, n_heads, 5, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, 5, device=device, dtype=torch.bfloat16)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)

    got_p0, got_edge_scores, got_value = run_arhsa_walk_readout_from_scores_fixed_iters_backward(
        p0_data,
        edge_scores_data,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_data,
        grad_readout,
        n_queries=n_queries,
        n_iters=4,
    )

    p0_ref = p0_data.clone().requires_grad_()
    edge_scores_ref = edge_scores_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
        p0_ref,
        edge_scores_ref,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
        n_iters=4,
    )
    readout_ref.backward(grad_readout)

    assert got_p0.dtype is torch.bfloat16
    assert got_edge_scores.dtype is torch.bfloat16
    assert got_value.dtype is torch.bfloat16
    torch.testing.assert_close(got_p0, p0_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(got_edge_scores, edge_scores_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(got_value, value_ref.grad, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_autograd_fast_query_warp_path_matches_torch_bfloat16():
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    from flash_attn.cute.arhsa_walk_sm100 import (
        arhsa_walk_readout_from_scores_fixed_iters_autograd,
        torch_arhsa_walk_readout_from_scores_fixed_iters,
    )

    device = torch.device("cuda")
    torch.manual_seed(59)
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5, 6], dtype=torch.int32, device=device)
    n_nodes = 7
    n_heads = 2
    n_queries = 3
    head_dim_v = 64
    p0_data = torch.rand(n_nodes, n_heads, device=device, dtype=torch.bfloat16)
    edge_scores_data = torch.randn(src.numel(), n_heads, device=device, dtype=torch.bfloat16)
    value_data = torch.randn(n_nodes, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    grad_readout = torch.randn(n_queries, n_heads, head_dim_v, device=device, dtype=torch.bfloat16)
    node_is_sink = torch.tensor([False, False, False, True, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5, 6, 3, 6], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2, 3, 4, 5], dtype=torch.int64, device=device)

    p0 = p0_data.clone().requires_grad_()
    edge_scores = edge_scores_data.clone().requires_grad_()
    value = value_data.clone().requires_grad_()
    readout = arhsa_walk_readout_from_scores_fixed_iters_autograd(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=n_queries,
        n_iters=4,
        leaf_major_stats=True,
        query_warp_stats=True,
        query_warp_readout=True,
        query_warp_scatter=True,
        query_warp_fused=True,
    )
    readout.backward(grad_readout)

    p0_ref = p0_data.clone().requires_grad_()
    edge_scores_ref = edge_scores_data.clone().requires_grad_()
    value_ref = value_data.clone().requires_grad_()
    readout_ref, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
        p0_ref,
        edge_scores_ref,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_ref,
        n_queries=n_queries,
        n_iters=4,
    )
    readout_ref.backward(grad_readout)

    torch.testing.assert_close(readout, readout_ref, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(p0.grad, p0_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(edge_scores.grad, edge_scores_ref.grad, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(value.grad, value_ref.grad, atol=8e-2, rtol=8e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_autograd_uses_cute_backward_for_bfloat16(monkeypatch):
    pytest.importorskip("cutlass")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 CUDA support required")
    import flash_attn.cute.arhsa_walk_sm100 as arhsa_walk

    def _forbid_torch_backward(*_args, **_kwargs):
        raise AssertionError("BF16 autograd must use the CuTe backward path")

    monkeypatch.setattr(
        arhsa_walk,
        "torch_arhsa_walk_readout_from_scores_fixed_iters_backward",
        _forbid_torch_backward,
    )

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    p0 = torch.zeros(6, 2, device=device, dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        p0[0] = torch.tensor([1.0, 0.7], device=device, dtype=torch.bfloat16)
        p0[2] = torch.tensor([0.0, 0.3], device=device, dtype=torch.bfloat16)
    edge_scores = torch.randn(src.numel(), 2, device=device, dtype=torch.bfloat16, requires_grad=True)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)
    value = torch.randn(3, 2, 5, device=device, dtype=torch.bfloat16, requires_grad=True)

    readout = arhsa_walk.arhsa_walk_readout_from_scores_fixed_iters_autograd(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=2,
        n_iters=4,
    )
    readout.square().sum().backward()

    assert p0.grad is not None and p0.grad.dtype is torch.bfloat16
    assert edge_scores.grad is not None and edge_scores.grad.dtype is torch.bfloat16
    assert value.grad is not None and value.grad.dtype is torch.bfloat16
    assert torch.isfinite(p0.grad).all()
    assert torch.isfinite(edge_scores.grad).all()
    assert torch.isfinite(value.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_arhsa_walk_readout_autograd_backward_allows_value_only_grad():
    pytest.importorskip("cutlass")
    from flash_attn.cute.arhsa_walk_sm100 import arhsa_walk_readout_from_scores_fixed_iters_autograd

    device = torch.device("cuda")
    src = torch.tensor([0, 0, 1, 2, 2, 3, 4], dtype=torch.int32, device=device)
    dst = torch.tensor([1, 2, 3, 3, 4, 5, 5], dtype=torch.int32, device=device)
    p0 = torch.zeros(6, 2, device=device, dtype=torch.float32)
    p0[0] = torch.tensor([1.0, 0.7], device=device)
    p0[2] = torch.tensor([0.0, 0.3], device=device)
    edge_scores = torch.randn(src.numel(), 2, device=device, dtype=torch.float32)
    node_is_sink = torch.tensor([False, False, False, True, True, True], device=device)
    leaf_node_index = torch.tensor([3, 4, 5, 4, 5], dtype=torch.int64, device=device)
    leaf_query_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)
    leaf_value_index = torch.tensor([0, 1, 2, 1, 2], dtype=torch.int64, device=device)
    value = torch.randn(3, 2, 5, device=device, dtype=torch.float32, requires_grad=True)

    readout = arhsa_walk_readout_from_scores_fixed_iters_autograd(
        p0,
        edge_scores,
        src,
        dst,
        node_is_sink,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value,
        n_queries=2,
        n_iters=4,
    )
    readout.square().sum().backward()

    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
