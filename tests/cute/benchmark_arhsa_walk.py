import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attn.cute.arhsa_walk_sm100 import (
    arhsa_walk_readout_from_scores_fixed_iters_autograd,
    build_incoming_edge_csr,
    build_leaf_entry_value_packs,
    build_outgoing_edge_csr,
    build_query_leaf_csr,
    build_query_value_packs,
    outgoing_softmax_from_scores,
    readout_arhsa_leaf_attention,
    run_arhsa_gather_edge_prob_by_index,
    run_arhsa_pack_leaf_values,
    run_arhsa_leaf_readout,
    run_arhsa_leaf_readout_backward,
    run_arhsa_markov_backward_step,
    run_arhsa_markov_backward_range_step,
    run_arhsa_markov_incoming_step,
    run_arhsa_markov_incoming_packed_step,
    run_arhsa_markov_incoming_packed_range_step,
    run_arhsa_markov_walk_fixed_iters,
    run_arhsa_outgoing_softmax,
    run_arhsa_outgoing_softmax_with_incoming,
    run_arhsa_outgoing_softmax_backward,
    run_arhsa_walk_readout_fixed_iters,
    torch_arhsa_walk_readout_from_scores_fixed_iters,
)


def _unwrap_output(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _reference_step(p, edge_prob, src, dst, node_is_sink):
    p_next = torch.where(node_is_sink[:, None], p, torch.zeros_like(p))
    if src.numel() > 0:
        p_next.index_add_(0, dst, p[src] * edge_prob)
    return p_next


def _event_ms(fn, *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _mib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _peak_memory(fn) -> dict[str, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    after = torch.cuda.memory_allocated()
    return {
        "peak_mib": _mib(peak),
        "temp_mib": _mib(max(0, peak - before)),
        "after_delta_mib": _mib(after - before),
    }


def _torch_readout_backward_for_benchmark(
    p,
    leaf_node_index,
    leaf_query_index,
    leaf_value_index,
    value,
    grad_readout,
    *,
    n_queries: int,
):
    p_replay = p.detach().requires_grad_(True)
    value_replay = value.detach().requires_grad_(True)
    out, _ = readout_arhsa_leaf_attention(
        p_replay,
        leaf_node_index,
        leaf_query_index,
        leaf_value_index,
        value_replay,
        n_queries=n_queries,
    )
    out.backward(grad_readout)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name}")


def _check_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype is torch.bfloat16:
        return 5e-1, 8e-2
    return 1e-6, 1e-6


def _tensor_delta(prefix: str, got: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    got_f = got.detach().float()
    expected_f = expected.detach().float()
    diff = got_f - expected_f
    denom = expected_f.abs().clamp_min(1.0e-8)
    l2_denom = expected_f.square().sum().sqrt().clamp_min(1.0e-8)
    return {
        f"{prefix}_max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        f"{prefix}_mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        f"{prefix}_l2_rel": float(diff.square().sum().sqrt().div(l2_denom).item()) if diff.numel() else 0.0,
        f"{prefix}_max_rel": float((diff.abs() / denom).max().item()) if diff.numel() else 0.0,
        f"{prefix}_mean_rel": float((diff.abs() / denom).mean().item()) if diff.numel() else 0.0,
    }


def _level_bounds(n_nodes: int, n_levels: int) -> list[int]:
    if n_levels <= 1:
        raise ValueError("structured graph modes require at least 2 levels")
    if n_nodes < n_levels:
        raise ValueError(f"n_nodes={n_nodes} must be >= graph_levels={n_levels}")
    base = n_nodes // n_levels
    rem = n_nodes % n_levels
    bounds = [0]
    for level_idx in range(n_levels):
        bounds.append(bounds[-1] + base + (1 if level_idx < rem else 0))
    return bounds


def _random_dst_from_range(
    *,
    start: int,
    end: int,
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if end <= start:
        raise ValueError(f"empty destination range [{start}, {end})")
    return torch.randint(
        end - start,
        shape,
        device=device,
        dtype=torch.int64,
        generator=generator,
    ) + int(start)


def _make_level_graph_edges(args, device: torch.device, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    n_levels = int(args.graph_levels) if int(args.graph_levels) > 0 else int(args.n_iters) + 1
    bounds = _level_bounds(int(args.n_nodes), n_levels)
    src_parts = []
    dst_parts = []

    for level_idx in range(n_levels - 1):
        start = bounds[level_idx]
        end = bounds[level_idx + 1]
        next_start = bounds[level_idx + 1]
        next_end = bounds[level_idx + 2]
        level_src = torch.arange(start, end, device=device, dtype=torch.int64).repeat_interleave(args.avg_out)
        level_dst = _random_dst_from_range(
            start=next_start,
            end=next_end,
            shape=(level_src.numel(),),
            device=device,
            generator=generator,
        )
        src_parts.append(level_src)
        dst_parts.append(level_dst)

    if args.graph_mode == "structured_walk":
        back_out = max(1, int(args.avg_out) // 2)
        for level_idx in range(1, n_levels - 1):
            start = bounds[level_idx]
            end = bounds[level_idx + 1]
            prev_start = bounds[level_idx - 1]
            prev_end = bounds[level_idx]
            level_src = torch.arange(start, end, device=device, dtype=torch.int64).repeat_interleave(back_out)
            level_dst = _random_dst_from_range(
                start=prev_start,
                end=prev_end,
                shape=(level_src.numel(),),
                device=device,
                generator=generator,
            )
            src_parts.append(level_src)
            dst_parts.append(level_dst)

    if not src_parts:
        raise ValueError("structured graph construction produced no edges")
    return torch.cat(src_parts), torch.cat(dst_parts), bounds


def _make_case(args):
    device = torch.device("cuda")
    dtype = _dtype_from_name(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    n_leaf_entries = args.n_queries * args.leaves_per_query
    level_bounds = None
    if args.graph_mode == "random":
        n_edges = args.n_nodes * args.avg_out
        src = torch.arange(args.n_nodes, device=device, dtype=torch.int64).repeat_interleave(args.avg_out)
        dst = torch.randint(args.n_nodes, (n_edges,), device=device, dtype=torch.int64, generator=generator)
    else:
        src, dst, level_bounds = _make_level_graph_edges(args, device, generator)
        n_edges = int(src.numel())
    src_i32 = src.to(dtype=torch.int32)
    dst_i32 = dst.to(dtype=torch.int32)
    edge_scores = torch.randn(n_edges, args.n_heads, device=device, dtype=dtype, generator=generator)
    edge_prob = outgoing_softmax_from_scores(edge_scores, src, n_nodes=args.n_nodes)
    if args.graph_mode == "random":
        node_is_sink = torch.rand(args.n_nodes, device=device, generator=generator) < args.sink_prob
    else:
        node_is_sink = torch.zeros(args.n_nodes, device=device, dtype=torch.bool)
        node_is_sink[level_bounds[-2] : level_bounds[-1]] = True
    p0 = torch.rand(args.n_nodes, args.n_heads, device=device, dtype=dtype, generator=generator)
    if args.graph_mode != "random":
        p0_mask = torch.zeros(args.n_nodes, 1, device=device, dtype=p0.dtype)
        p0_mask[level_bounds[0] : level_bounds[1]] = 1
        p0 = p0 * p0_mask
    src_row_ptr, src_edge_index = build_outgoing_edge_csr(src, n_nodes=args.n_nodes)
    dst_row_ptr, dst_edge_index = build_incoming_edge_csr(dst, n_nodes=args.n_nodes)
    incoming_src_i32 = src_i32[dst_edge_index.to(dtype=torch.long)].contiguous()
    edge_incoming_index = torch.empty_like(dst_edge_index)
    edge_incoming_index[dst_edge_index.to(dtype=torch.long)] = torch.arange(
        n_edges,
        device=device,
        dtype=torch.int32,
    )

    leaf_query_index = torch.arange(args.n_queries, device=device, dtype=torch.int64).repeat_interleave(
        args.leaves_per_query
    )
    if args.graph_mode == "random":
        leaf_node_index = torch.randint(
            args.n_nodes,
            (n_leaf_entries,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
    else:
        leaf_node_index = _random_dst_from_range(
            start=level_bounds[-2],
            end=level_bounds[-1],
            shape=(n_leaf_entries,),
            device=device,
            generator=generator,
        )
    if args.leaf_value_pattern == "random":
        leaf_value_index = torch.randint(
            args.n_nodes,
            (n_leaf_entries,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
    elif args.leaf_value_pattern == "shared-block16":
        leaf_slot = torch.arange(args.leaves_per_query, device=device, dtype=torch.int64).repeat(args.n_queries)
        query_block = torch.arange(args.n_queries, device=device, dtype=torch.int64).repeat_interleave(
            args.leaves_per_query
        ) // 16
        leaf_value_index = (query_block * args.leaves_per_query + leaf_slot) % args.n_nodes
    else:
        raise ValueError(f"unknown leaf value pattern: {args.leaf_value_pattern}")
    if args.compact_value_rows:
        if leaf_value_index.numel() == 0:
            n_value_rows = 0
        elif args.leaf_value_pattern == "shared-block16":
            n_value_rows = int(leaf_value_index.max().item()) + 1
        else:
            _, inverse = torch.unique(leaf_value_index, sorted=True, return_inverse=True)
            leaf_value_index = inverse
            n_value_rows = int(leaf_value_index.max().item()) + 1
    else:
        n_value_rows = args.n_nodes
    leaf_query_index_i32 = leaf_query_index.to(dtype=torch.int32)
    leaf_node_index_i32 = leaf_node_index.to(dtype=torch.int32)
    leaf_value_index_i32 = leaf_value_index.to(dtype=torch.int32)
    value = torch.randn(
        n_value_rows,
        args.n_heads,
        args.head_dim_v,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    query_leaf_row_ptr, query_leaf_entry_index = build_query_leaf_csr(
        leaf_query_index,
        n_queries=args.n_queries,
    )
    return {
        "src": src,
        "dst": dst,
        "src_i32": src_i32,
        "dst_i32": dst_i32,
        "edge_scores": edge_scores,
        "edge_prob": edge_prob,
        "node_is_sink": node_is_sink,
        "p0": p0,
        "src_row_ptr": src_row_ptr,
        "src_edge_index": src_edge_index,
        "dst_row_ptr": dst_row_ptr,
        "dst_edge_index": dst_edge_index,
        "incoming_src_i32": incoming_src_i32,
        "edge_incoming_index": edge_incoming_index,
        "leaf_query_index": leaf_query_index,
        "leaf_node_index": leaf_node_index,
        "leaf_value_index": leaf_value_index,
        "leaf_query_index_i32": leaf_query_index_i32,
        "leaf_node_index_i32": leaf_node_index_i32,
        "leaf_value_index_i32": leaf_value_index_i32,
        "value": value,
        "query_leaf_row_ptr": query_leaf_row_ptr,
        "query_leaf_entry_index": query_leaf_entry_index,
        "graph_mode": args.graph_mode,
        "level_bounds": level_bounds,
    }


def _check(case, args):
    atol, rtol = _check_tolerances(case["p0"].dtype)
    p_next = torch.empty_like(case["p0"])
    got_edge_prob = run_arhsa_outgoing_softmax(
        case["edge_scores"],
        case["src_row_ptr"],
        case["src_edge_index"],
        n_nodes=args.n_nodes,
    )
    torch.testing.assert_close(got_edge_prob, case["edge_prob"], atol=atol, rtol=rtol)

    got_step = run_arhsa_markov_incoming_step(
        case["p0"],
        case["edge_prob"],
        case["src_i32"],
        case["dst_row_ptr"],
        case["dst_edge_index"],
        case["node_is_sink"],
        p_next,
    )
    expected_step = _reference_step(
        case["p0"],
        case["edge_prob"],
        case["src"],
        case["dst"],
        case["node_is_sink"],
    )
    torch.testing.assert_close(got_step, expected_step, atol=atol, rtol=rtol)
    if args.incoming_packed_step:
        got_edge_prob_again, got_incoming_from_softmax = run_arhsa_outgoing_softmax_with_incoming(
            case["edge_scores"],
            case["src_row_ptr"],
            case["src_edge_index"],
            case["edge_incoming_index"],
            n_nodes=args.n_nodes,
        )
        incoming_edge_prob = run_arhsa_gather_edge_prob_by_index(
            case["edge_prob"],
            case["dst_edge_index"],
        )
        got_packed_step = run_arhsa_markov_incoming_packed_step(
            case["p0"],
            incoming_edge_prob,
            case["incoming_src_i32"],
            case["dst_row_ptr"],
            case["node_is_sink"],
        )
        torch.testing.assert_close(got_edge_prob_again, case["edge_prob"], atol=atol, rtol=rtol)
        torch.testing.assert_close(got_incoming_from_softmax, incoming_edge_prob, atol=atol, rtol=rtol)
        torch.testing.assert_close(incoming_edge_prob, case["edge_prob"][case["dst_edge_index"].long()], atol=0, rtol=0)
        torch.testing.assert_close(got_packed_step, expected_step, atol=atol, rtol=rtol)

    readout_pack_kwargs = {}
    if args.query_value_pack_readout:
        pack_query_index, pack_value_index, pack_query_value_leaf_entry = build_query_value_packs(
            case["leaf_query_index_i32"],
            case["leaf_value_index_i32"],
            n_queries=args.n_queries,
            max_values=args.tensor_core_qv_pack_max_values,
            packing_strategy=args.tensor_core_qv_pack_strategy,
        )
        readout_pack_kwargs = {
            "query_value_pack": True,
            "tensor_core_query_value_pack": args.tensor_core_qv_readout,
            "pack_query_index": pack_query_index,
            "pack_value_index": pack_value_index,
            "pack_query_value_leaf_entry": pack_query_value_leaf_entry,
        }

    readout = run_arhsa_leaf_readout(
        got_step,
        case["leaf_node_index_i32"],
        case["leaf_value_index_i32"],
        case["query_leaf_row_ptr"],
        case["query_leaf_entry_index"],
        case["value"],
        n_queries=args.n_queries,
        query_warp=args.query_warp_readout,
        **readout_pack_kwargs,
    )
    expected_readout, _ = readout_arhsa_leaf_attention(
        got_step,
        case["leaf_node_index"],
        case["leaf_query_index"],
        case["leaf_value_index"],
        case["value"],
        n_queries=args.n_queries,
    )
    torch.testing.assert_close(readout, expected_readout, atol=atol, rtol=rtol)

    got_full, _, got_p, got_prob = run_arhsa_walk_readout_fixed_iters(
        case["p0"],
        got_edge_prob,
        case["src_i32"],
        case["dst_row_ptr"],
        case["dst_edge_index"],
        case["node_is_sink"],
        case["leaf_node_index_i32"],
        case["leaf_query_index_i32"],
        case["leaf_value_index_i32"],
        case["value"],
        n_queries=args.n_queries,
        n_iters=args.n_iters,
        query_leaf_row_ptr=case["query_leaf_row_ptr"],
        query_leaf_entry_index=case["query_leaf_entry_index"],
        return_leaf_attn=False,
        query_warp_readout=args.query_warp_readout,
    ) + (got_edge_prob,)
    expected_full, _, expected_p, expected_prob = torch_arhsa_walk_readout_from_scores_fixed_iters(
        case["p0"],
        case["edge_scores"],
        case["src"],
        case["dst"],
        case["node_is_sink"],
        case["leaf_node_index"],
        case["leaf_query_index"],
        case["leaf_value_index"],
        case["value"],
        n_queries=args.n_queries,
        n_iters=args.n_iters,
    )
    torch.testing.assert_close(got_prob, expected_prob, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_p, expected_p, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_full, expected_full, atol=atol, rtol=rtol)


def _forward_backward_numeric_summary(
    case,
    args,
    grad_readout,
    *,
    tc_pack_leaf_entry_index,
    tc_pack_value_index,
    tc_pack_value_slot,
    tc_qv_pack_query_index,
    tc_qv_pack_value_index,
    tc_qv_pack_leaf_entry,
) -> dict[str, float]:
    p0_cute = case["p0"].detach().clone().requires_grad_(True)
    edge_scores_cute = case["edge_scores"].detach().clone().requires_grad_(True)
    value_cute = case["value"].detach().clone().requires_grad_(True)
    readout_cute = arhsa_walk_readout_from_scores_fixed_iters_autograd(
        p0_cute,
        edge_scores_cute,
        case["src_i32"],
        case["dst_i32"],
        case["node_is_sink"],
        case["leaf_node_index_i32"],
        case["leaf_query_index_i32"],
        case["leaf_value_index_i32"],
        value_cute,
        n_queries=args.n_queries,
        n_iters=args.n_iters,
        src_row_ptr=case["src_row_ptr"],
        src_edge_index=case["src_edge_index"],
        dst_row_ptr=case["dst_row_ptr"],
        dst_edge_index=case["dst_edge_index"],
        query_leaf_row_ptr=case["query_leaf_row_ptr"],
        query_leaf_entry_index=case["query_leaf_entry_index"],
        max_leaves_per_query=args.leaves_per_query if args.fused_readout_bwd else None,
        leaf_major_stats=args.leaf_major_stats,
        query_warp_stats=args.query_warp_stats,
        query_warp_readout=args.query_warp_readout,
        incoming_packed_step=args.incoming_packed_step,
        save_forward_history=args.save_forward_history,
        level_bounds=case["level_bounds"],
        level_range_kernels=args.level_range_kernels,
        incoming_src=case["incoming_src_i32"],
        edge_incoming_index=case["edge_incoming_index"],
        query_warp_scatter=args.query_warp_scatter,
        query_warp_fused=args.query_warp_fused_bwd,
        tensor_core_stats=args.tensor_core_stats_bwd,
        tensor_core_fused=args.tensor_core_fused_bwd,
        tensor_core_packed=args.tensor_core_packed_bwd,
        tensor_core_query_value_packed=args.tensor_core_query_value_packed_bwd,
        query_value_pack_scatter=args.tensor_core_qv_pack_scatter_bwd,
        tensor_core_query_value_pack_scatter_dv=args.tensor_core_qv_pack_scatter_dv_bwd,
        pack_leaf_entry_index=tc_pack_leaf_entry_index,
        pack_value_index=tc_pack_value_index,
        pack_value_slot=tc_pack_value_slot,
        pack_query_index=tc_qv_pack_query_index,
        pack_query_value_index=tc_qv_pack_value_index,
        pack_query_value_leaf_entry=tc_qv_pack_leaf_entry,
    )
    readout_cute.backward(grad_readout)

    p0_ref = case["p0"].detach().clone().requires_grad_(True)
    edge_scores_ref = case["edge_scores"].detach().clone().requires_grad_(True)
    value_ref = case["value"].detach().clone().requires_grad_(True)
    readout_ref, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
        p0_ref,
        edge_scores_ref,
        case["src"],
        case["dst"],
        case["node_is_sink"],
        case["leaf_node_index"],
        case["leaf_query_index"],
        case["leaf_value_index"],
        value_ref,
        n_queries=args.n_queries,
        n_iters=args.n_iters,
    )
    readout_ref.backward(grad_readout)

    summary = {}
    summary.update(_tensor_delta("numeric_readout", readout_cute, readout_ref))
    if args.level_range_kernels and case["level_bounds"] is not None:
        p0_ref_grad = p0_ref.grad.clone()
        p0_cute_grad = p0_cute.grad.clone()
        level0_end = int(case["level_bounds"][1])
        p0_ref_grad[level0_end:] = 0
        p0_cute_grad[level0_end:] = 0
    else:
        p0_ref_grad = p0_ref.grad
        p0_cute_grad = p0_cute.grad
    summary.update(_tensor_delta("numeric_grad_p0", p0_cute_grad, p0_ref_grad))
    summary.update(_tensor_delta("numeric_grad_edge_scores", edge_scores_cute.grad, edge_scores_ref.grad))
    summary.update(_tensor_delta("numeric_grad_value", value_cute.grad, value_ref.grad))
    return summary


def _beam_prune_state(p: torch.Tensor, *, topk: int) -> tuple[torch.Tensor, float]:
    topk = min(int(topk), int(p.shape[0]))
    if topk <= 0 or topk >= int(p.shape[0]):
        return p, 1.0
    values, indices = torch.topk(p.float().abs(), k=topk, dim=0, sorted=False)
    del values
    pruned = torch.zeros_like(p)
    pruned.scatter_(0, indices, p.gather(0, indices))
    before = p.float().abs().sum().clamp_min(1.0e-8)
    retained = pruned.float().abs().sum() / before
    return pruned, float(retained.item())


def _beam_numeric_summary(case, args, *, beam_topk: int) -> dict[str, float]:
    with torch.no_grad():
        exact_readout, _, exact_p, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
            case["p0"],
            case["edge_scores"],
            case["src"],
            case["dst"],
            case["node_is_sink"],
            case["leaf_node_index"],
            case["leaf_query_index"],
            case["leaf_value_index"],
            case["value"],
            n_queries=args.n_queries,
            n_iters=args.n_iters,
        )
        p, retained = _beam_prune_state(case["p0"], topk=beam_topk)
        retained_values = [retained]
        for _ in range(args.n_iters):
            p = _reference_step(
                p,
                case["edge_prob"],
                case["src"],
                case["dst"],
                case["node_is_sink"],
            )
            p, retained = _beam_prune_state(p, topk=beam_topk)
            retained_values.append(retained)
        beam_readout, _ = readout_arhsa_leaf_attention(
            p,
            case["leaf_node_index"],
            case["leaf_query_index"],
            case["leaf_value_index"],
            case["value"],
            n_queries=args.n_queries,
        )
    summary = {
        "beam_topk": float(beam_topk),
        "beam_retained_abs_mass_min": min(retained_values),
        "beam_retained_abs_mass_mean": sum(retained_values) / len(retained_values),
    }
    summary.update(_tensor_delta("beam_p_final", p, exact_p))
    summary.update(_tensor_delta("beam_readout", beam_readout, exact_readout))
    return summary


def _measure_fa4_baseline(args) -> dict[str, float | str]:
    try:
        from flash_attn.cute import flash_attn_func
    except Exception as exc:
        return {"fa4_status": f"import_failed:{type(exc).__name__}:{exc}"}

    dtype = _dtype_from_name(args.fa4_dtype or args.dtype)
    device = torch.device("cuda")
    q = torch.randn(
        args.fa4_batch,
        args.fa4_seqlen,
        args.fa4_n_heads,
        args.fa4_head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    window_size = (-1, -1)
    label = "dense_causal"
    if args.fa4_window_left >= 0:
        window_size = (int(args.fa4_window_left), 0)
        label = f"sliding_left_{args.fa4_window_left}"

    def _forward():
        return _unwrap_output(
            flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=window_size,
            )
        )

    def _forward_grad():
        q_run = q.detach().clone().requires_grad_(True)
        k_run = k.detach().clone().requires_grad_(True)
        v_run = v.detach().clone().requires_grad_(True)
        out = _unwrap_output(
            flash_attn_func(
                q_run,
                k_run,
                v_run,
                causal=True,
                window_size=window_size,
            )
        )
        out.backward(torch.ones_like(out))

    result: dict[str, float | str] = {
        "fa4_status": "measured",
        "fa4_label": label,
        "fa4_batch": float(args.fa4_batch),
        "fa4_seqlen": float(args.fa4_seqlen),
        "fa4_n_heads": float(args.fa4_n_heads),
        "fa4_head_dim": float(args.fa4_head_dim),
    }
    try:
        result["fa4_fwd_ms"] = _event_ms(
            _forward,
            iters=args.fa4_iters,
            warmup=args.fa4_warmup,
        )
        result["fa4_fwd_bwd_ms"] = _event_ms(
            _forward_grad,
            iters=args.fa4_iters,
            warmup=args.fa4_warmup,
        )
        result["fa4_bwd_ms"] = float(result["fa4_fwd_bwd_ms"]) - float(result["fa4_fwd_ms"])
    except Exception as exc:
        result = {
            **result,
            "fa4_status": f"failed:{type(exc).__name__}:{exc}",
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark exact fixed-iteration ARHSA walk kernels.")
    parser.add_argument("--n-nodes", type=int, default=4096)
    parser.add_argument("--avg-out", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-queries", type=int, default=1024)
    parser.add_argument("--leaves-per-query", type=int, default=8)
    parser.add_argument("--head-dim-v", type=int, default=64)
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--sink-prob", type=float, default=0.05)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--graph-mode",
        choices=("random", "level_dag", "structured_walk"),
        default="random",
        help=(
            "Synthetic graph geometry. random is the original arbitrary sparse graph; "
            "level_dag only routes level l -> l+1; structured_walk adds adjacent-level back edges."
        ),
    )
    parser.add_argument(
        "--graph-levels",
        type=int,
        default=0,
        help="Number of structured graph levels. 0 means n_iters + 1 for level_dag/structured_walk.",
    )
    parser.add_argument(
        "--level-range-kernels",
        action="store_true",
        help="For level_dag, use range-limited Markov forward/backward kernels over the active level only.",
    )
    parser.add_argument(
        "--leaf-value-pattern",
        choices=("random", "shared-block16"),
        default="random",
        help="Synthetic leaf-value layout; shared-block16 makes each 16-query block share value columns.",
    )
    parser.add_argument(
        "--compact-value-rows",
        action="store_true",
        help="Allocate/remap value rows to only rows referenced by leaf_value_index.",
    )
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument(
        "--fused-readout-bwd",
        action="store_true",
        help="Use the opt-in fused small-fanout readout backward path.",
    )
    parser.add_argument(
        "--leaf-major-stats",
        action="store_true",
        help="Use a leaf-major readout-backward stats pass with atomic query reductions.",
    )
    parser.add_argument(
        "--reuse-forward-denom",
        action="store_true",
        help="Store readout denominators in forward and reuse them in leaf-major readout backward.",
    )
    parser.add_argument(
        "--fp32-backward-state",
        action="store_true",
        help="Benchmark with FP32 explicit backward scratch/output tensors instead of casting grads to BF16.",
    )
    parser.add_argument(
        "--pack-leaf-values",
        action="store_true",
        help="Pack value[leaf_value_index] before leaf-major readout backward stats.",
    )
    parser.add_argument(
        "--query-warp-stats",
        action="store_true",
        help="Use one warp per query/head for readout-backward stats.",
    )
    parser.add_argument(
        "--query-warp-readout",
        action="store_true",
        help="Use one warp per query/head for the readout forward reduction.",
    )
    parser.add_argument(
        "--query-value-pack-readout",
        action="store_true",
        help="Use qv-pack-owned D=64 readout forward when query/value packs are dense.",
    )
    parser.add_argument(
        "--tensor-core-qv-readout",
        action="store_true",
        help="Use tensor cores for --query-value-pack-readout.",
    )
    parser.add_argument(
        "--incoming-packed-step",
        action="store_true",
        help="Gather edge probabilities into incoming-CSR order and use the packed Markov forward step.",
    )
    parser.add_argument(
        "--save-forward-history",
        action="store_true",
        help="Save edge probabilities and Markov states from autograd forward so backward skips recomputing them.",
    )
    parser.add_argument(
        "--query-warp-scatter",
        action="store_true",
        help="Use one warp per query/head for readout-backward scatter.",
    )
    parser.add_argument(
        "--query-warp-fused-bwd",
        action="store_true",
        help="Fuse query-warp readout-backward stats and scatter for head_dim_v=64.",
    )
    parser.add_argument(
        "--tensor-core-stats-bwd",
        action="store_true",
        help="Use experimental tensor-core D=64 readout-backward stats pass.",
    )
    parser.add_argument(
        "--tensor-core-fused-bwd",
        action="store_true",
        help="Use experimental fused tensor-core D=64 readout backward.",
    )
    parser.add_argument(
        "--tensor-core-packed-bwd",
        action="store_true",
        help="Use experimental leaf-entry/value packed tensor-core D=64 readout backward stats.",
    )
    parser.add_argument(
        "--tensor-core-query-value-packed-bwd",
        action="store_true",
        help="Use experimental query/value packed tensor-core D=64 readout backward stats.",
    )
    parser.add_argument(
        "--tensor-core-qv-pack-scatter-bwd",
        action="store_true",
        help="Use pack-owned qv scatter after query/value packed tensor-core stats.",
    )
    parser.add_argument(
        "--tensor-core-qv-pack-scatter-dv-bwd",
        action="store_true",
        help="Use tensor cores for dV inside pack-owned qv scatter.",
    )
    parser.add_argument(
        "--tensor-core-qv-pack-strategy",
        choices=("lexicographic", "span", "overlap"),
        default="overlap",
        help="CPU grouping strategy for --tensor-core-query-value-packed-bwd metadata.",
    )
    parser.add_argument(
        "--tensor-core-qv-pack-max-values",
        type=int,
        choices=(8, 16),
        default=8,
        help="Value columns per 16-query qv pack; 16 fills tensor-core MMA tiles when fanout allows it.",
    )
    parser.add_argument(
        "--auto-readout-bwd",
        action="store_true",
        help="Choose qv tensor-core backward when qv packing is dense, otherwise use query-warp fused backward.",
    )
    parser.add_argument(
        "--auto-qv-output-util-threshold",
        type=float,
        default=0.25,
        help="Minimum qv pack output utilization for --auto-readout-bwd to select tensor-core qv.",
    )
    parser.add_argument(
        "--report-numerics",
        action="store_true",
        help="Run one custom-vs-Torch forward/backward comparison and print numeric error metrics.",
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="Skip original Torch ARHSA reference timings; useful when the reference OOMs at long sequence lengths.",
    )
    parser.add_argument(
        "--skip-custom-fwd-bwd",
        action="store_true",
        help="Skip allocation-heavy custom autograd fwd+bwd timing; useful for very long preallocated runs.",
    )
    parser.add_argument(
        "--beam-topk",
        type=int,
        default=0,
        help="Diagnostic only: prune the Torch walk state to top-k nodes per head after each step and report error.",
    )
    parser.add_argument(
        "--compare-fa4",
        action="store_true",
        help="Also time a dense/sliding causal FA4 baseline. This is not the same computation as ARHSA.",
    )
    parser.add_argument("--fa4-batch", type=int, default=1)
    parser.add_argument("--fa4-seqlen", type=int, default=8192)
    parser.add_argument("--fa4-n-heads", type=int, default=None)
    parser.add_argument("--fa4-head-dim", type=int, default=None)
    parser.add_argument("--fa4-dtype", choices=("bfloat16", "float32"), default=None)
    parser.add_argument(
        "--fa4-window-left",
        type=int,
        default=-1,
        help="Use sliding-window FA4 with this left window; -1 means dense causal FA4.",
    )
    parser.add_argument("--fa4-iters", type=int, default=None)
    parser.add_argument("--fa4-warmup", type=int, default=None)
    parser.add_argument(
        "--profile-target",
        choices=(
            "cute_fwd_bwd",
            "cute_fwd_bwd_prealloc",
            "torch_fwd_bwd",
            "readout_bwd_cute",
            "full_cute_hot",
            "pack_leaf_values",
        ),
        default=None,
    )
    parser.add_argument("--profile-repeat", type=int, default=3)
    parser.add_argument(
        "--profile-cuda-capture",
        action="store_true",
        help="Wrap --profile-target repeats in cudaProfilerStart/Stop for Nsight capture-range=cudaProfilerApi.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 CUDA support is required for --dtype=bfloat16")
    if args.beam_topk < 0:
        raise ValueError("--beam-topk must be >= 0")
    if args.graph_levels < 0:
        raise ValueError("--graph-levels must be >= 0")
    if args.graph_mode != "random" and args.graph_levels == 1:
        raise ValueError("structured graph modes require --graph-levels 0 or >= 2")
    if args.level_range_kernels and args.graph_mode != "level_dag":
        raise ValueError("--level-range-kernels currently requires --graph-mode level_dag")
    if args.level_range_kernels and not args.incoming_packed_step:
        raise ValueError("--level-range-kernels currently requires --incoming-packed-step")
    if args.fa4_n_heads is None:
        args.fa4_n_heads = args.n_heads
    if args.fa4_head_dim is None:
        args.fa4_head_dim = args.head_dim_v
    if args.fa4_iters is None:
        args.fa4_iters = args.iters
    if args.fa4_warmup is None:
        args.fa4_warmup = args.warmup
    if args.compare_fa4:
        if args.fa4_batch <= 0 or args.fa4_seqlen <= 0 or args.fa4_n_heads <= 0 or args.fa4_head_dim <= 0:
            raise ValueError("FA4 batch, seqlen, heads, and head dim must be positive")
        if args.fa4_dtype == "float32":
            raise ValueError("FA4 baseline currently requires float16/bfloat16-style inputs; use bfloat16")
    if args.auto_readout_bwd and args.head_dim_v != 64:
        raise ValueError("--auto-readout-bwd currently requires --head-dim-v 64")
    if args.query_value_pack_readout and args.head_dim_v != 64:
        raise ValueError("--query-value-pack-readout currently requires --head-dim-v 64")
    if args.tensor_core_qv_readout and not args.query_value_pack_readout:
        raise ValueError("--tensor-core-qv-readout requires --query-value-pack-readout")
    if args.tensor_core_qv_readout and args.dtype != "bfloat16":
        raise ValueError("--tensor-core-qv-readout currently requires --dtype=bfloat16")
    if args.query_value_pack_readout and args.leaves_per_query > args.tensor_core_qv_pack_max_values:
        raise ValueError("--query-value-pack-readout requires leaves_per_query <= --tensor-core-qv-pack-max-values")
    if args.auto_readout_bwd and any(
        (
            args.fused_readout_bwd,
            args.leaf_major_stats,
            args.reuse_forward_denom,
            args.pack_leaf_values,
            args.query_warp_stats,
            args.query_warp_scatter,
            args.query_warp_fused_bwd,
            args.tensor_core_stats_bwd,
            args.tensor_core_fused_bwd,
            args.tensor_core_packed_bwd,
            args.tensor_core_query_value_packed_bwd,
            args.tensor_core_qv_pack_scatter_bwd,
        )
    ):
        raise ValueError("--auto-readout-bwd cannot be combined with explicit readout-backward mode flags")
    if args.reuse_forward_denom and not args.leaf_major_stats and not args.query_warp_fused_bwd:
        raise ValueError("--reuse-forward-denom requires --leaf-major-stats unless --query-warp-fused-bwd is set")
    if args.pack_leaf_values and not args.leaf_major_stats:
        raise ValueError("--pack-leaf-values requires --leaf-major-stats")
    if args.query_warp_stats and not args.leaf_major_stats:
        raise ValueError("--query-warp-stats requires --leaf-major-stats")
    if args.query_warp_stats and args.pack_leaf_values:
        raise ValueError("--query-warp-stats cannot be combined with --pack-leaf-values")
    if args.query_warp_stats and args.reuse_forward_denom:
        raise ValueError("--query-warp-stats cannot be combined with --reuse-forward-denom")
    if args.query_warp_fused_bwd and args.pack_leaf_values:
        raise ValueError("--query-warp-fused-bwd cannot be combined with --pack-leaf-values")
    if args.query_warp_fused_bwd and args.head_dim_v != 64:
        raise ValueError("--query-warp-fused-bwd currently requires --head-dim-v 64")
    if args.tensor_core_stats_bwd and not args.leaf_major_stats:
        raise ValueError("--tensor-core-stats-bwd requires --leaf-major-stats")
    if args.tensor_core_stats_bwd and args.query_warp_fused_bwd:
        raise ValueError("--tensor-core-stats-bwd cannot be combined with --query-warp-fused-bwd")
    if args.tensor_core_stats_bwd and args.query_warp_stats:
        raise ValueError("--tensor-core-stats-bwd cannot be combined with --query-warp-stats")
    if args.tensor_core_stats_bwd and args.pack_leaf_values:
        raise ValueError("--tensor-core-stats-bwd cannot be combined with --pack-leaf-values")
    if args.tensor_core_stats_bwd and args.reuse_forward_denom:
        raise ValueError("--tensor-core-stats-bwd cannot be combined with --reuse-forward-denom")
    if args.tensor_core_stats_bwd and args.head_dim_v != 64:
        raise ValueError("--tensor-core-stats-bwd currently requires --head-dim-v 64")
    if args.tensor_core_fused_bwd and args.head_dim_v != 64:
        raise ValueError("--tensor-core-fused-bwd currently requires --head-dim-v 64")
    if args.tensor_core_fused_bwd and args.tensor_core_stats_bwd:
        raise ValueError("--tensor-core-fused-bwd cannot be combined with --tensor-core-stats-bwd")
    if args.tensor_core_fused_bwd and args.query_warp_fused_bwd:
        raise ValueError("--tensor-core-fused-bwd cannot be combined with --query-warp-fused-bwd")
    if args.tensor_core_fused_bwd and args.fused_readout_bwd:
        raise ValueError("--tensor-core-fused-bwd cannot be combined with --fused-readout-bwd")
    if args.tensor_core_fused_bwd and args.pack_leaf_values:
        raise ValueError("--tensor-core-fused-bwd cannot be combined with --pack-leaf-values")
    if args.tensor_core_fused_bwd and args.reuse_forward_denom:
        raise ValueError("--tensor-core-fused-bwd cannot be combined with --reuse-forward-denom")
    if args.tensor_core_packed_bwd and args.head_dim_v != 64:
        raise ValueError("--tensor-core-packed-bwd currently requires --head-dim-v 64")
    if args.tensor_core_packed_bwd and args.tensor_core_stats_bwd:
        raise ValueError("--tensor-core-packed-bwd cannot be combined with --tensor-core-stats-bwd")
    if args.tensor_core_packed_bwd and args.tensor_core_fused_bwd:
        raise ValueError("--tensor-core-packed-bwd cannot be combined with --tensor-core-fused-bwd")
    if args.tensor_core_packed_bwd and args.query_warp_fused_bwd:
        raise ValueError("--tensor-core-packed-bwd cannot be combined with --query-warp-fused-bwd")
    if args.tensor_core_packed_bwd and args.fused_readout_bwd:
        raise ValueError("--tensor-core-packed-bwd cannot be combined with --fused-readout-bwd")
    if args.tensor_core_packed_bwd and args.pack_leaf_values:
        raise ValueError("--tensor-core-packed-bwd cannot be combined with --pack-leaf-values")
    if args.tensor_core_packed_bwd and args.reuse_forward_denom:
        raise ValueError("--tensor-core-packed-bwd cannot be combined with --reuse-forward-denom")
    if args.tensor_core_query_value_packed_bwd and args.head_dim_v != 64:
        raise ValueError("--tensor-core-query-value-packed-bwd currently requires --head-dim-v 64")
    if args.tensor_core_query_value_packed_bwd and args.tensor_core_stats_bwd:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --tensor-core-stats-bwd")
    if args.tensor_core_query_value_packed_bwd and args.tensor_core_fused_bwd:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --tensor-core-fused-bwd")
    if args.tensor_core_query_value_packed_bwd and args.tensor_core_packed_bwd:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --tensor-core-packed-bwd")
    if args.tensor_core_query_value_packed_bwd and args.query_warp_fused_bwd:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --query-warp-fused-bwd")
    if args.tensor_core_query_value_packed_bwd and args.fused_readout_bwd:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --fused-readout-bwd")
    if args.tensor_core_query_value_packed_bwd and args.pack_leaf_values:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --pack-leaf-values")
    if args.tensor_core_query_value_packed_bwd and args.reuse_forward_denom:
        raise ValueError("--tensor-core-query-value-packed-bwd cannot be combined with --reuse-forward-denom")
    if args.tensor_core_qv_pack_scatter_bwd and not args.tensor_core_query_value_packed_bwd:
        raise ValueError("--tensor-core-qv-pack-scatter-bwd requires --tensor-core-query-value-packed-bwd")
    if args.tensor_core_qv_pack_scatter_bwd and args.query_warp_scatter:
        raise ValueError("--tensor-core-qv-pack-scatter-bwd cannot be combined with --query-warp-scatter")
    if args.tensor_core_qv_pack_scatter_bwd and args.leaves_per_query > args.tensor_core_qv_pack_max_values:
        raise ValueError(
            "--tensor-core-qv-pack-scatter-bwd requires leaves_per_query <= --tensor-core-qv-pack-max-values"
        )
    if args.tensor_core_qv_pack_scatter_dv_bwd and not args.tensor_core_qv_pack_scatter_bwd:
        raise ValueError("--tensor-core-qv-pack-scatter-dv-bwd requires --tensor-core-qv-pack-scatter-bwd")
    case = _make_case(args)
    if not args.no_check:
        _check(case, args)

    p_next = torch.empty_like(case["p0"])
    p_scratch_a = torch.empty_like(case["p0"])
    p_scratch_b = torch.empty_like(case["p0"])
    edge_prob = torch.empty_like(case["edge_scores"])
    edge_prob_incoming = torch.empty_like(case["edge_scores"])
    readout = torch.empty(
        args.n_queries,
        args.n_heads,
        args.head_dim_v,
        device=case["p0"].device,
        dtype=case["value"].dtype,
    )
    grad_readout = torch.randn_like(readout)
    packed_value = torch.empty(
        args.n_queries * args.leaves_per_query,
        args.n_heads,
        args.head_dim_v,
        device=case["p0"].device,
        dtype=case["value"].dtype,
    )
    p_history = [case["p0"]] + [torch.empty_like(case["p0"]) for _ in range(args.n_iters)]
    backward_state_dtype = torch.float32 if args.fp32_backward_state else case["p0"].dtype
    grad_p_final = torch.empty_like(case["p0"], dtype=backward_state_dtype)
    grad_p_scratch = torch.empty_like(case["p0"], dtype=backward_state_dtype)
    grad_edge_prob = torch.empty_like(case["edge_scores"], dtype=backward_state_dtype)
    grad_edge_scores = torch.empty_like(case["edge_scores"], dtype=backward_state_dtype)
    grad_value_prealloc = torch.empty_like(case["value"], dtype=backward_state_dtype)
    leaf_grad_attn = torch.empty(
        args.n_queries * args.leaves_per_query,
        args.n_heads,
        device=case["p0"].device,
        dtype=torch.float32,
    )
    readout_bwd_denom = torch.empty(args.n_queries, args.n_heads, device=case["p0"].device, dtype=torch.float32)
    readout_bwd_weighted_grad_sum = torch.empty(
        args.n_queries,
        args.n_heads,
        device=case["p0"].device,
        dtype=torch.float32,
    )
    grad_p_accum = torch.empty_like(case["p0"], dtype=torch.float32)
    grad_value_accum = torch.empty_like(case["value"], dtype=torch.float32)
    case_edge_prob_incoming = None
    if args.incoming_packed_step:
        case_edge_prob_incoming = run_arhsa_gather_edge_prob_by_index(
            case["edge_prob"],
            case["dst_edge_index"],
            torch.empty_like(case["edge_prob"]),
        )
    use_level_range_kernels = bool(args.level_range_kernels)
    level_bounds = case["level_bounds"] or []

    def _level_range(level_idx: int) -> tuple[int, int]:
        level_idx = max(0, min(int(level_idx), len(level_bounds) - 2))
        start = int(level_bounds[level_idx])
        end = int(level_bounds[level_idx + 1])
        return start, end - start

    def _level_idx(level_idx: int) -> int:
        return max(0, min(int(level_idx), len(level_bounds) - 2))

    def _forward_dst_range(iter_idx: int) -> tuple[int, int]:
        return _level_range(int(iter_idx) + 1)

    def _forward_carry_sinks(iter_idx: int) -> bool:
        return _level_idx(int(iter_idx) + 1) == _level_idx(iter_idx)

    def _backward_src_range(iter_idx: int) -> tuple[int, int]:
        return _level_range(int(iter_idx))

    def _markov_step(source_edge_prob, incoming_edge_prob, p_in, p_out, *, iter_idx: int | None = None):
        if use_level_range_kernels and iter_idx is not None:
            node_start, node_count = _forward_dst_range(iter_idx)
            run_arhsa_markov_incoming_packed_range_step(
                p_in,
                incoming_edge_prob,
                case["incoming_src_i32"],
                case["dst_row_ptr"],
                case["node_is_sink"],
                node_start=node_start,
                node_count=node_count,
                carry_sinks=_forward_carry_sinks(iter_idx),
                p_next=p_out,
            )
            return
        if args.incoming_packed_step:
            run_arhsa_markov_incoming_packed_step(
                p_in,
                incoming_edge_prob,
                case["incoming_src_i32"],
                case["dst_row_ptr"],
                case["node_is_sink"],
                p_out,
            )
        else:
            run_arhsa_markov_incoming_step(
                p_in,
                source_edge_prob,
                case["src_i32"],
                case["dst_row_ptr"],
                case["dst_edge_index"],
                case["node_is_sink"],
                p_out,
            )

    def _markov_walk(source_edge_prob, incoming_edge_prob, scratch_a, scratch_b):
        if not args.incoming_packed_step:
            return run_arhsa_markov_walk_fixed_iters(
                case["p0"],
                source_edge_prob,
                case["src_i32"],
                case["dst_row_ptr"],
                case["dst_edge_index"],
                case["node_is_sink"],
                n_iters=args.n_iters,
                scratch_a=scratch_a,
                scratch_b=scratch_b,
            )
        p_cur = case["p0"]
        for iter_idx in range(args.n_iters):
            p_out = scratch_a if iter_idx % 2 == 0 else scratch_b
            _markov_step(
                source_edge_prob,
                incoming_edge_prob,
                p_cur,
                p_out,
                iter_idx=iter_idx,
            )
            p_cur = p_out
        return p_cur

    def _softmax_for_walk():
        if args.incoming_packed_step:
            run_arhsa_outgoing_softmax_with_incoming(
                case["edge_scores"],
                case["src_row_ptr"],
                case["src_edge_index"],
                case["edge_incoming_index"],
                n_nodes=args.n_nodes,
                edge_prob=edge_prob,
                incoming_edge_prob=edge_prob_incoming,
            )
        else:
            run_arhsa_outgoing_softmax(
                case["edge_scores"],
                case["src_row_ptr"],
                case["src_edge_index"],
                n_nodes=args.n_nodes,
                edge_prob=edge_prob,
            )

    p_readout = _markov_walk(case["edge_prob"], case_edge_prob_incoming, p_scratch_a, p_scratch_b)
    grad_p_readout = torch.empty_like(p_readout)
    grad_value = torch.empty_like(case["value"])
    tc_pack_leaf_entry_index = None
    tc_pack_value_index = None
    tc_pack_value_slot = None
    tc_qv_pack_query_index = None
    tc_qv_pack_value_index = None
    tc_qv_pack_leaf_entry = None
    tc_pack_summary = {}
    if args.tensor_core_packed_bwd:
        tc_pack_leaf_entry_index, tc_pack_value_index, tc_pack_value_slot = build_leaf_entry_value_packs(
            case["leaf_value_index_i32"],
        )
        tc_pack_valid_rows = int((tc_pack_leaf_entry_index >= 0).sum().item())
        tc_pack_count = int(tc_pack_leaf_entry_index.shape[0])
        tc_pack_summary = {
            "tensor_core_pack_count": tc_pack_count,
            "tensor_core_pack_fill": (tc_pack_valid_rows / (tc_pack_count * 16)) if tc_pack_count else 0.0,
            "tensor_core_pack_output_util": (
                tc_pack_valid_rows / (tc_pack_count * 16 * 8)
            ) if tc_pack_count else 0.0,
        }
    auto_selected_readout_bwd = "explicit"
    tc_qv_output_util = 0.0
    if args.query_value_pack_readout or args.tensor_core_query_value_packed_bwd or args.auto_readout_bwd:
        tc_qv_pack_query_index, tc_qv_pack_value_index, tc_qv_pack_leaf_entry = build_query_value_packs(
            case["leaf_query_index_i32"],
            case["leaf_value_index_i32"],
            n_queries=args.n_queries,
            max_values=args.tensor_core_qv_pack_max_values,
            packing_strategy=args.tensor_core_qv_pack_strategy,
        )
        tc_qv_pack_count = int(tc_qv_pack_query_index.shape[0])
        tc_qv_valid_rows = int((tc_qv_pack_query_index >= 0).sum().item())
        tc_qv_valid_outputs = int((tc_qv_pack_leaf_entry >= 0).sum().item())
        tc_qv_used_value_cols = int((tc_qv_pack_leaf_entry >= 0).any(dim=1).sum().item())
        tc_qv_output_util = (
            tc_qv_valid_outputs / (tc_qv_pack_count * 16 * args.tensor_core_qv_pack_max_values)
        ) if tc_qv_pack_count else 0.0
        tc_pack_summary = {
            "tensor_core_qv_pack_count": tc_qv_pack_count,
            "tensor_core_qv_pack_row_fill": (
                tc_qv_valid_rows / (tc_qv_pack_count * 16)
            ) if tc_qv_pack_count else 0.0,
            "tensor_core_qv_pack_value_fill": (
                tc_qv_used_value_cols / (tc_qv_pack_count * args.tensor_core_qv_pack_max_values)
            ) if tc_qv_pack_count else 0.0,
            "tensor_core_qv_pack_output_util": tc_qv_output_util,
        }
    if args.auto_readout_bwd:
        if tc_qv_output_util >= float(args.auto_qv_output_util_threshold):
            args.tensor_core_query_value_packed_bwd = True
            if args.leaves_per_query <= args.tensor_core_qv_pack_max_values:
                args.tensor_core_qv_pack_scatter_bwd = True
                auto_selected_readout_bwd = "tensor_core_query_value_packed_pack_scatter"
            else:
                args.query_warp_scatter = True
                auto_selected_readout_bwd = "tensor_core_query_value_packed_query_warp_scatter"
        else:
            args.query_warp_fused_bwd = True
            auto_selected_readout_bwd = "query_warp_fused"

    readout_pack_kwargs = {}
    if args.query_value_pack_readout:
        if tc_qv_pack_query_index is None or tc_qv_pack_value_index is None or tc_qv_pack_leaf_entry is None:
            raise RuntimeError("query/value readout packs were not built")
        readout_pack_kwargs = {
            "query_value_pack": True,
            "tensor_core_query_value_pack": args.tensor_core_qv_readout,
            "pack_query_index": tc_qv_pack_query_index,
            "pack_value_index": tc_qv_pack_value_index,
            "pack_query_value_leaf_entry": tc_qv_pack_leaf_entry,
        }

    def _pack_leaf_values():
        run_arhsa_pack_leaf_values(
            case["leaf_value_index_i32"],
            case["value"],
            packed_value=packed_value,
        )

    def _cute_forward_backward():
        p0 = case["p0"].detach().requires_grad_(True)
        edge_scores = case["edge_scores"].detach().requires_grad_(True)
        value = case["value"].detach().requires_grad_(True)
        out = arhsa_walk_readout_from_scores_fixed_iters_autograd(
            p0,
            edge_scores,
            case["src_i32"],
            case["dst_i32"],
            case["node_is_sink"],
            case["leaf_node_index_i32"],
            case["leaf_query_index_i32"],
            case["leaf_value_index_i32"],
            value,
            n_queries=args.n_queries,
            n_iters=args.n_iters,
            src_row_ptr=case["src_row_ptr"],
            src_edge_index=case["src_edge_index"],
            dst_row_ptr=case["dst_row_ptr"],
            dst_edge_index=case["dst_edge_index"],
            query_leaf_row_ptr=case["query_leaf_row_ptr"],
            query_leaf_entry_index=case["query_leaf_entry_index"],
            max_leaves_per_query=args.leaves_per_query if args.fused_readout_bwd else None,
            leaf_major_stats=args.leaf_major_stats,
            query_warp_stats=args.query_warp_stats,
            query_warp_readout=args.query_warp_readout,
            query_value_pack_readout=args.query_value_pack_readout,
            tensor_core_query_value_pack_readout=args.tensor_core_qv_readout,
            incoming_packed_step=args.incoming_packed_step,
            save_forward_history=args.save_forward_history,
            level_bounds=case["level_bounds"],
            level_range_kernels=args.level_range_kernels,
            incoming_src=case["incoming_src_i32"],
            edge_incoming_index=case["edge_incoming_index"],
            query_warp_scatter=args.query_warp_scatter,
            query_warp_fused=args.query_warp_fused_bwd,
            tensor_core_stats=args.tensor_core_stats_bwd,
            tensor_core_fused=args.tensor_core_fused_bwd,
            tensor_core_packed=args.tensor_core_packed_bwd,
            tensor_core_query_value_packed=args.tensor_core_query_value_packed_bwd,
            query_value_pack_scatter=args.tensor_core_qv_pack_scatter_bwd,
            tensor_core_query_value_pack_scatter_dv=args.tensor_core_qv_pack_scatter_dv_bwd,
            pack_leaf_entry_index=tc_pack_leaf_entry_index,
            pack_value_index=tc_pack_value_index,
            pack_value_slot=tc_pack_value_slot,
            pack_query_index=tc_qv_pack_query_index,
            pack_query_value_index=tc_qv_pack_value_index,
            pack_query_value_leaf_entry=tc_qv_pack_leaf_entry,
        )
        out.backward(grad_readout)

    def _cute_forward_backward_prealloc():
        _softmax_for_walk()
        for iter_idx in range(args.n_iters):
            _markov_step(
                edge_prob,
                edge_prob_incoming,
                p_history[iter_idx],
                p_history[iter_idx + 1],
                iter_idx=iter_idx,
            )
        run_arhsa_leaf_readout(
            p_history[-1],
            case["leaf_node_index_i32"],
            case["leaf_value_index_i32"],
            case["query_leaf_row_ptr"],
            case["query_leaf_entry_index"],
            case["value"],
            n_queries=args.n_queries,
            readout=readout,
            denom=readout_bwd_denom if args.reuse_forward_denom else None,
            query_warp=args.query_warp_readout,
            **readout_pack_kwargs,
        )
        if args.pack_leaf_values:
            _pack_leaf_values()
        run_arhsa_leaf_readout_backward(
            p_history[-1],
            case["leaf_node_index_i32"],
            case["leaf_query_index_i32"],
            case["leaf_value_index_i32"],
            case["query_leaf_row_ptr"],
            case["query_leaf_entry_index"],
            case["value"],
            grad_readout,
            n_queries=args.n_queries,
            grad_p=grad_p_final,
            grad_value=grad_value_prealloc,
            max_leaves_per_query=args.leaves_per_query if args.fused_readout_bwd else None,
            leaf_grad_attn=leaf_grad_attn,
            denom=readout_bwd_denom,
            weighted_grad_sum=readout_bwd_weighted_grad_sum,
            grad_p_accum=grad_p_accum,
            grad_value_accum=grad_value_accum,
            leaf_major_stats=args.leaf_major_stats,
            denom_precomputed=args.reuse_forward_denom,
            packed_value=packed_value if args.pack_leaf_values else None,
            query_warp_stats=args.query_warp_stats,
            query_warp_scatter=args.query_warp_scatter,
            query_warp_fused=args.query_warp_fused_bwd,
            tensor_core_stats=args.tensor_core_stats_bwd,
            tensor_core_fused=args.tensor_core_fused_bwd,
            tensor_core_packed=args.tensor_core_packed_bwd,
            tensor_core_query_value_packed=args.tensor_core_query_value_packed_bwd,
            query_value_pack_scatter=args.tensor_core_qv_pack_scatter_bwd,
            tensor_core_query_value_pack_scatter_dv=args.tensor_core_qv_pack_scatter_dv_bwd,
            pack_leaf_entry_index=tc_pack_leaf_entry_index,
            pack_value_index=tc_pack_value_index,
            pack_value_slot=tc_pack_value_slot,
            pack_query_index=tc_qv_pack_query_index,
            pack_query_value_index=tc_qv_pack_value_index,
            pack_query_value_leaf_entry=tc_qv_pack_leaf_entry,
        )
        if args.n_iters == 0:
            grad_edge_prob.zero_()
        grad_next = grad_p_final
        grad_scratch = grad_p_scratch
        for iter_idx in range(args.n_iters - 1, -1, -1):
            if use_level_range_kernels:
                if args.n_iters < len(level_bounds) - 1 and iter_idx == args.n_iters - 1:
                    grad_edge_prob.zero_()
                node_start, node_count = _backward_src_range(iter_idx)
                run_arhsa_markov_backward_range_step(
                    grad_next,
                    p_history[iter_idx],
                    edge_prob,
                    case["src_row_ptr"],
                    case["src_edge_index"],
                    case["dst_i32"],
                    case["node_is_sink"],
                    grad_edge_prob,
                    node_start=node_start,
                    node_count=node_count,
                    grad_p_prev=grad_scratch,
                    accumulate_grad_edge_prob=False,
                )
            else:
                run_arhsa_markov_backward_step(
                    grad_next,
                    p_history[iter_idx],
                    edge_prob,
                    case["src_row_ptr"],
                    case["src_edge_index"],
                    case["dst_i32"],
                    case["node_is_sink"],
                    grad_edge_prob,
                    grad_scratch,
                    accumulate_grad_edge_prob=(iter_idx != args.n_iters - 1),
                )
            grad_next, grad_scratch = grad_scratch, grad_next
        run_arhsa_outgoing_softmax_backward(
            edge_prob,
            grad_edge_prob,
            case["src_row_ptr"],
            case["src_edge_index"],
            n_nodes=args.n_nodes,
            grad_edge_scores=grad_edge_scores,
        )

    def _torch_forward_backward():
        p0 = case["p0"].detach().requires_grad_(True)
        edge_scores = case["edge_scores"].detach().requires_grad_(True)
        value = case["value"].detach().requires_grad_(True)
        out, _, _, _ = torch_arhsa_walk_readout_from_scores_fixed_iters(
            p0,
            edge_scores,
            case["src"],
            case["dst"],
            case["node_is_sink"],
            case["leaf_node_index"],
            case["leaf_query_index"],
            case["leaf_value_index"],
            value,
            n_queries=args.n_queries,
            n_iters=args.n_iters,
        )
        out.backward(grad_readout)

    def _readout_backward_cute():
        if args.pack_leaf_values:
            _pack_leaf_values()
        if args.reuse_forward_denom:
            run_arhsa_leaf_readout(
                p_readout,
                case["leaf_node_index_i32"],
                case["leaf_value_index_i32"],
                case["query_leaf_row_ptr"],
                case["query_leaf_entry_index"],
                case["value"],
                n_queries=args.n_queries,
                readout=readout,
                denom=readout_bwd_denom,
                query_warp=args.query_warp_readout,
                **readout_pack_kwargs,
            )
        run_arhsa_leaf_readout_backward(
            p_readout,
            case["leaf_node_index_i32"],
            case["leaf_query_index_i32"],
            case["leaf_value_index_i32"],
            case["query_leaf_row_ptr"],
            case["query_leaf_entry_index"],
            case["value"],
            grad_readout,
            n_queries=args.n_queries,
            grad_p=grad_p_readout,
            grad_value=grad_value,
            max_leaves_per_query=args.leaves_per_query if args.fused_readout_bwd else None,
            leaf_grad_attn=leaf_grad_attn,
            denom=readout_bwd_denom,
            weighted_grad_sum=readout_bwd_weighted_grad_sum,
            grad_p_accum=grad_p_accum,
            grad_value_accum=grad_value_accum,
            leaf_major_stats=args.leaf_major_stats,
            denom_precomputed=args.reuse_forward_denom,
            packed_value=packed_value if args.pack_leaf_values else None,
            query_warp_stats=args.query_warp_stats,
            query_warp_scatter=args.query_warp_scatter,
            query_warp_fused=args.query_warp_fused_bwd,
            tensor_core_stats=args.tensor_core_stats_bwd,
            tensor_core_fused=args.tensor_core_fused_bwd,
            tensor_core_packed=args.tensor_core_packed_bwd,
            tensor_core_query_value_packed=args.tensor_core_query_value_packed_bwd,
            query_value_pack_scatter=args.tensor_core_qv_pack_scatter_bwd,
            tensor_core_query_value_pack_scatter_dv=args.tensor_core_qv_pack_scatter_dv_bwd,
            pack_leaf_entry_index=tc_pack_leaf_entry_index,
            pack_value_index=tc_pack_value_index,
            pack_value_slot=tc_pack_value_slot,
            pack_query_index=tc_qv_pack_query_index,
            pack_query_value_index=tc_qv_pack_value_index,
            pack_query_value_leaf_entry=tc_qv_pack_leaf_entry,
        )

    def _full_cute_hot():
        _softmax_for_walk()
        p_final = _markov_walk(
            edge_prob,
            edge_prob_incoming,
            p_scratch_a,
            p_scratch_b,
        )
        run_arhsa_leaf_readout(
            p_final,
            case["leaf_node_index_i32"],
            case["leaf_value_index_i32"],
            query_leaf_row_ptr=case["query_leaf_row_ptr"],
            query_leaf_entry_index=case["query_leaf_entry_index"],
            value=case["value"],
            n_queries=args.n_queries,
            readout=readout,
            query_warp=args.query_warp_readout,
            **readout_pack_kwargs,
        )

    def _walk_readout_hot():
        p_final = _markov_walk(
            case["edge_prob"],
            case_edge_prob_incoming,
            p_scratch_a,
            p_scratch_b,
        )
        run_arhsa_leaf_readout(
            p_final,
            case["leaf_node_index_i32"],
            case["leaf_value_index_i32"],
            case["query_leaf_row_ptr"],
            case["query_leaf_entry_index"],
            case["value"],
            n_queries=args.n_queries,
            readout=readout,
            query_warp=args.query_warp_readout,
            **readout_pack_kwargs,
        )

    profile_targets = {
        "cute_fwd_bwd": _cute_forward_backward,
        "cute_fwd_bwd_prealloc": _cute_forward_backward_prealloc,
        "torch_fwd_bwd": _torch_forward_backward,
        "readout_bwd_cute": _readout_backward_cute,
        "full_cute_hot": _full_cute_hot,
        "pack_leaf_values": _pack_leaf_values,
    }
    if args.profile_target is not None:
        target = profile_targets[args.profile_target]
        for _ in range(args.warmup):
            target()
        torch.cuda.synchronize()
        if args.profile_cuda_capture:
            torch.cuda.cudart().cudaProfilerStart()
        for repeat_idx in range(int(args.profile_repeat)):
            torch.cuda.nvtx.range_push(f"{args.profile_target}_{repeat_idx}")
            target()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if args.profile_cuda_capture:
            torch.cuda.cudart().cudaProfilerStop()
        return

    timings = {
        "softmax_cute_ms": _event_ms(
            lambda: run_arhsa_outgoing_softmax(
                case["edge_scores"],
                case["src_row_ptr"],
                case["src_edge_index"],
                n_nodes=args.n_nodes,
                edge_prob=edge_prob,
            ),
            iters=args.iters,
            warmup=args.warmup,
        ),
        "step_cute_ms": _event_ms(
            lambda: _markov_step(
                case["edge_prob"],
                case_edge_prob_incoming,
                case["p0"],
                p_next,
                iter_idx=0,
            ),
            iters=args.iters,
            warmup=args.warmup,
        ),
        "walk_cute_ms": _event_ms(
            lambda: _markov_walk(
                case["edge_prob"],
                case_edge_prob_incoming,
                p_scratch_a,
                p_scratch_b,
            ),
            iters=args.iters,
            warmup=args.warmup,
        ),
        "readout_cute_ms": _event_ms(
            lambda: run_arhsa_leaf_readout(
                p_next,
                case["leaf_node_index_i32"],
                case["leaf_value_index_i32"],
                case["query_leaf_row_ptr"],
                case["query_leaf_entry_index"],
                case["value"],
                n_queries=args.n_queries,
                readout=readout,
                query_warp=args.query_warp_readout,
                **readout_pack_kwargs,
            ),
            iters=args.iters,
            warmup=args.warmup,
        ),
        "readout_bwd_cute_ms": _event_ms(
            _readout_backward_cute,
            iters=args.iters,
            warmup=args.warmup,
        ),
        "walk_readout_hot_ms": _event_ms(
            _walk_readout_hot,
            iters=args.iters,
            warmup=args.warmup,
        ),
        "full_cute_hot_ms": _event_ms(
            _full_cute_hot,
            iters=args.iters,
            warmup=args.warmup,
        ),
        "fwd_bwd_cute_prealloc_ms": _event_ms(
            _cute_forward_backward_prealloc,
            iters=args.iters,
            warmup=args.warmup,
        ),
    }
    if not args.skip_custom_fwd_bwd:
        timings["fwd_bwd_cute_custom_ms"] = _event_ms(
            _cute_forward_backward,
            iters=args.iters,
            warmup=args.warmup,
        )
    if not args.skip_torch:
        timings.update(
            {
                "softmax_torch_ms": _event_ms(
                    lambda: outgoing_softmax_from_scores(
                        case["edge_scores"],
                        case["src"],
                        n_nodes=args.n_nodes,
                    ),
                    iters=args.iters,
                    warmup=args.warmup,
                ),
                "step_torch_ms": _event_ms(
                    lambda: _reference_step(
                        case["p0"],
                        case["edge_prob"],
                        case["src"],
                        case["dst"],
                        case["node_is_sink"],
                    ),
                    iters=args.iters,
                    warmup=args.warmup,
                ),
                "readout_torch_ms": _event_ms(
                    lambda: readout_arhsa_leaf_attention(
                        p_next,
                        case["leaf_node_index"],
                        case["leaf_query_index"],
                        case["leaf_value_index"],
                        case["value"],
                        n_queries=args.n_queries,
                    ),
                    iters=args.iters,
                    warmup=args.warmup,
                ),
                "readout_bwd_torch_ms": _event_ms(
                    lambda: _torch_readout_backward_for_benchmark(
                        p_readout,
                        case["leaf_node_index"],
                        case["leaf_query_index"],
                        case["leaf_value_index"],
                        case["value"],
                        grad_readout,
                        n_queries=args.n_queries,
                    ),
                    iters=args.iters,
                    warmup=args.warmup,
                ),
                "full_torch_ms": _event_ms(
                    lambda: torch_arhsa_walk_readout_from_scores_fixed_iters(
                        case["p0"],
                        case["edge_scores"],
                        case["src"],
                        case["dst"],
                        case["node_is_sink"],
                        case["leaf_node_index"],
                        case["leaf_query_index"],
                        case["leaf_value_index"],
                        case["value"],
                        n_queries=args.n_queries,
                        n_iters=args.n_iters,
                    ),
                    iters=args.iters,
                    warmup=args.warmup,
                ),
                "fwd_bwd_torch_autograd_ms": _event_ms(
                    _torch_forward_backward,
                    iters=args.iters,
                    warmup=args.warmup,
                ),
            }
        )
    if args.incoming_packed_step:
        timings["softmax_with_incoming_cute_ms"] = _event_ms(
            _softmax_for_walk,
            iters=args.iters,
            warmup=args.warmup,
        )
        timings["incoming_gather_cute_ms"] = _event_ms(
            lambda: run_arhsa_gather_edge_prob_by_index(
                case["edge_prob"],
                case["dst_edge_index"],
                edge_prob_incoming,
            ),
            iters=args.iters,
            warmup=args.warmup,
        )
    if args.pack_leaf_values:
        timings["pack_leaf_values_cute_ms"] = _event_ms(
            _pack_leaf_values,
            iters=args.iters,
            warmup=args.warmup,
        )
    memory = {}
    if not args.no_memory:
        cute_memory_fn = _cute_forward_backward_prealloc if args.skip_custom_fwd_bwd else _cute_forward_backward
        cute_memory = _peak_memory(cute_memory_fn)
        memory = {
            "setup_allocated_mib": _mib(torch.cuda.memory_allocated()),
            "setup_reserved_mib": _mib(torch.cuda.memory_reserved()),
            "fwd_bwd_cute_peak_mib": cute_memory["peak_mib"],
            "fwd_bwd_cute_temp_mib": cute_memory["temp_mib"],
            "fwd_bwd_cute_after_delta_mib": cute_memory["after_delta_mib"],
        }
        if not args.skip_torch:
            torch_memory = _peak_memory(_torch_forward_backward)
            memory.update(
                {
                    "fwd_bwd_torch_peak_mib": torch_memory["peak_mib"],
                    "fwd_bwd_torch_temp_mib": torch_memory["temp_mib"],
                    "fwd_bwd_torch_after_delta_mib": torch_memory["after_delta_mib"],
                }
            )
    numeric = {}
    if args.report_numerics:
        numeric = _forward_backward_numeric_summary(
            case,
            args,
            grad_readout,
            tc_pack_leaf_entry_index=tc_pack_leaf_entry_index,
            tc_pack_value_index=tc_pack_value_index,
            tc_pack_value_slot=tc_pack_value_slot,
            tc_qv_pack_query_index=tc_qv_pack_query_index,
            tc_qv_pack_value_index=tc_qv_pack_value_index,
            tc_qv_pack_leaf_entry=tc_qv_pack_leaf_entry,
        )
    beam = {}
    if args.beam_topk > 0:
        beam = _beam_numeric_summary(case, args, beam_topk=args.beam_topk)
    fa4 = {}
    if args.compare_fa4:
        fa4 = _measure_fa4_baseline(args)

    print(
        {
            "n_nodes": args.n_nodes,
            "n_edges": int(case["edge_scores"].shape[0]),
            "graph_mode": args.graph_mode,
            "graph_levels": (
                len(case["level_bounds"]) - 1 if case["level_bounds"] is not None else 0
            ),
            "level_range_kernels": args.level_range_kernels,
            "n_heads": args.n_heads,
            "n_queries": args.n_queries,
            "leaf_entries": args.n_queries * args.leaves_per_query,
            "head_dim_v": args.head_dim_v,
            "value_rows": int(case["value"].shape[0]),
            "compact_value_rows": args.compact_value_rows,
            "n_iters": args.n_iters,
            "dtype": args.dtype,
            "leaf_value_pattern": args.leaf_value_pattern,
            "auto_readout_bwd": args.auto_readout_bwd,
            "auto_selected_readout_bwd": auto_selected_readout_bwd,
            "auto_qv_output_util_threshold": args.auto_qv_output_util_threshold,
            "report_numerics": args.report_numerics,
            "compare_fa4": args.compare_fa4,
            "fused_readout_bwd": args.fused_readout_bwd,
            "leaf_major_stats": args.leaf_major_stats,
            "reuse_forward_denom": args.reuse_forward_denom,
            "fp32_backward_state": args.fp32_backward_state,
            "pack_leaf_values": args.pack_leaf_values,
            "query_warp_stats": args.query_warp_stats,
            "query_warp_readout": args.query_warp_readout,
            "query_value_pack_readout": args.query_value_pack_readout,
            "tensor_core_qv_readout": args.tensor_core_qv_readout,
            "incoming_packed_step": args.incoming_packed_step,
            "save_forward_history": args.save_forward_history,
            "query_warp_scatter": args.query_warp_scatter,
            "query_warp_fused_bwd": args.query_warp_fused_bwd,
            "tensor_core_stats_bwd": args.tensor_core_stats_bwd,
            "tensor_core_fused_bwd": args.tensor_core_fused_bwd,
            "tensor_core_packed_bwd": args.tensor_core_packed_bwd,
            "tensor_core_query_value_packed_bwd": args.tensor_core_query_value_packed_bwd,
            "tensor_core_qv_pack_scatter_bwd": args.tensor_core_qv_pack_scatter_bwd,
            "tensor_core_qv_pack_scatter_dv_bwd": args.tensor_core_qv_pack_scatter_dv_bwd,
            "tensor_core_qv_pack_strategy": args.tensor_core_qv_pack_strategy,
            "tensor_core_qv_pack_max_values": args.tensor_core_qv_pack_max_values,
            **{key: round(value, 4) for key, value in tc_pack_summary.items()},
            **{key: round(value, 4) for key, value in timings.items()},
            **{key: round(value, 2) for key, value in memory.items()},
            **{key: round(value, 6) for key, value in numeric.items()},
            **{key: round(value, 6) for key, value in beam.items()},
            **{
                key: (round(value, 4) if isinstance(value, float) else value)
                for key, value in fa4.items()
            },
        }
    )


if __name__ == "__main__":
    main()
