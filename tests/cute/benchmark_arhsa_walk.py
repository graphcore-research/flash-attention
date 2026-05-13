import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flash_attn.cute.arhsa_walk_sm100 import (
    arhsa_walk_readout_from_scores_fixed_iters_autograd,
    build_incoming_edge_csr,
    build_outgoing_edge_csr,
    build_query_leaf_csr,
    outgoing_softmax_from_scores,
    readout_arhsa_leaf_attention,
    run_arhsa_pack_leaf_values,
    run_arhsa_leaf_readout,
    run_arhsa_leaf_readout_backward,
    run_arhsa_markov_backward_step,
    run_arhsa_markov_incoming_step,
    run_arhsa_markov_walk_fixed_iters,
    run_arhsa_outgoing_softmax,
    run_arhsa_outgoing_softmax_backward,
    run_arhsa_walk_readout_fixed_iters,
    torch_arhsa_walk_readout_from_scores_fixed_iters,
)


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


def _make_case(args):
    device = torch.device("cuda")
    dtype = _dtype_from_name(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    n_edges = args.n_nodes * args.avg_out
    n_leaf_entries = args.n_queries * args.leaves_per_query
    src = torch.arange(args.n_nodes, device=device, dtype=torch.int64).repeat_interleave(args.avg_out)
    dst = torch.randint(args.n_nodes, (n_edges,), device=device, dtype=torch.int64, generator=generator)
    src_i32 = src.to(dtype=torch.int32)
    dst_i32 = dst.to(dtype=torch.int32)
    edge_scores = torch.randn(n_edges, args.n_heads, device=device, dtype=dtype, generator=generator)
    edge_prob = outgoing_softmax_from_scores(edge_scores, src, n_nodes=args.n_nodes)
    node_is_sink = torch.rand(args.n_nodes, device=device, generator=generator) < args.sink_prob
    p0 = torch.rand(args.n_nodes, args.n_heads, device=device, dtype=dtype, generator=generator)
    src_row_ptr, src_edge_index = build_outgoing_edge_csr(src, n_nodes=args.n_nodes)
    dst_row_ptr, dst_edge_index = build_incoming_edge_csr(dst, n_nodes=args.n_nodes)

    leaf_query_index = torch.arange(args.n_queries, device=device, dtype=torch.int64).repeat_interleave(
        args.leaves_per_query
    )
    leaf_node_index = torch.randint(
        args.n_nodes,
        (n_leaf_entries,),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    leaf_value_index = torch.randint(
        args.n_nodes,
        (n_leaf_entries,),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    leaf_query_index_i32 = leaf_query_index.to(dtype=torch.int32)
    leaf_node_index_i32 = leaf_node_index.to(dtype=torch.int32)
    leaf_value_index_i32 = leaf_value_index.to(dtype=torch.int32)
    value = torch.randn(
        args.n_nodes,
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
        "leaf_query_index": leaf_query_index,
        "leaf_node_index": leaf_node_index,
        "leaf_value_index": leaf_value_index,
        "leaf_query_index_i32": leaf_query_index_i32,
        "leaf_node_index_i32": leaf_node_index_i32,
        "leaf_value_index_i32": leaf_value_index_i32,
        "value": value,
        "query_leaf_row_ptr": query_leaf_row_ptr,
        "query_leaf_entry_index": query_leaf_entry_index,
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

    readout = run_arhsa_leaf_readout(
        got_step,
        case["leaf_node_index_i32"],
        case["leaf_value_index_i32"],
        case["query_leaf_row_ptr"],
        case["query_leaf_entry_index"],
        case["value"],
        n_queries=args.n_queries,
        query_warp=args.query_warp_readout,
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 CUDA support is required for --dtype=bfloat16")
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
    case = _make_case(args)
    if not args.no_check:
        _check(case, args)

    p_next = torch.empty_like(case["p0"])
    p_scratch_a = torch.empty_like(case["p0"])
    p_scratch_b = torch.empty_like(case["p0"])
    edge_prob = torch.empty_like(case["edge_scores"])
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
    p_history = [torch.empty_like(case["p0"]) for _ in range(args.n_iters + 1)]
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
    p_readout = run_arhsa_markov_walk_fixed_iters(
        case["p0"],
        case["edge_prob"],
        case["src_i32"],
        case["dst_row_ptr"],
        case["dst_edge_index"],
        case["node_is_sink"],
        n_iters=args.n_iters,
        scratch_a=p_scratch_a,
        scratch_b=p_scratch_b,
    )
    grad_p_readout = torch.empty_like(p_readout)
    grad_value = torch.empty_like(case["value"])

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
            query_warp_scatter=args.query_warp_scatter,
            query_warp_fused=args.query_warp_fused_bwd,
        )
        out.backward(grad_readout)

    def _cute_forward_backward_prealloc():
        run_arhsa_outgoing_softmax(
            case["edge_scores"],
            case["src_row_ptr"],
            case["src_edge_index"],
            n_nodes=args.n_nodes,
            edge_prob=edge_prob,
        )
        p_history[0].copy_(case["p0"])
        for iter_idx in range(args.n_iters):
            run_arhsa_markov_incoming_step(
                p_history[iter_idx],
                edge_prob,
                case["src_i32"],
                case["dst_row_ptr"],
                case["dst_edge_index"],
                case["node_is_sink"],
                p_history[iter_idx + 1],
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
        )
        grad_edge_prob.zero_()
        grad_next = grad_p_final
        grad_scratch = grad_p_scratch
        for iter_idx in range(args.n_iters - 1, -1, -1):
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
        )

    def _full_cute_hot():
        run_arhsa_outgoing_softmax(
            case["edge_scores"],
            case["src_row_ptr"],
            case["src_edge_index"],
            n_nodes=args.n_nodes,
            edge_prob=edge_prob,
        )
        run_arhsa_walk_readout_fixed_iters(
            case["p0"],
            edge_prob,
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
            p_scratch_a=p_scratch_a,
            p_scratch_b=p_scratch_b,
            readout=readout,
            query_warp_readout=args.query_warp_readout,
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
        for repeat_idx in range(int(args.profile_repeat)):
            torch.cuda.nvtx.range_push(f"{args.profile_target}_{repeat_idx}")
            target()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
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
        "softmax_torch_ms": _event_ms(
            lambda: outgoing_softmax_from_scores(
                case["edge_scores"],
                case["src"],
                n_nodes=args.n_nodes,
            ),
            iters=args.iters,
            warmup=args.warmup,
        ),
        "step_cute_ms": _event_ms(
            lambda: run_arhsa_markov_incoming_step(
                case["p0"],
                case["edge_prob"],
                case["src_i32"],
                case["dst_row_ptr"],
                case["dst_edge_index"],
                case["node_is_sink"],
                p_next,
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
        "walk_cute_ms": _event_ms(
            lambda: run_arhsa_markov_walk_fixed_iters(
                case["p0"],
                case["edge_prob"],
                case["src_i32"],
                case["dst_row_ptr"],
                case["dst_edge_index"],
                case["node_is_sink"],
                n_iters=args.n_iters,
                scratch_a=p_scratch_a,
                scratch_b=p_scratch_b,
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
        "readout_bwd_cute_ms": _event_ms(
            _readout_backward_cute,
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
        "walk_readout_hot_ms": _event_ms(
            lambda: run_arhsa_walk_readout_fixed_iters(
                case["p0"],
                case["edge_prob"],
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
                p_scratch_a=p_scratch_a,
                p_scratch_b=p_scratch_b,
                readout=readout,
                query_warp_readout=args.query_warp_readout,
            ),
            iters=args.iters,
            warmup=args.warmup,
        ),
        "full_cute_hot_ms": _event_ms(
            _full_cute_hot,
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
        "fwd_bwd_cute_custom_ms": _event_ms(
            _cute_forward_backward,
            iters=args.iters,
            warmup=args.warmup,
        ),
        "fwd_bwd_cute_prealloc_ms": _event_ms(
            _cute_forward_backward_prealloc,
            iters=args.iters,
            warmup=args.warmup,
        ),
        "fwd_bwd_torch_autograd_ms": _event_ms(
            _torch_forward_backward,
            iters=args.iters,
            warmup=args.warmup,
        ),
    }
    if args.pack_leaf_values:
        timings["pack_leaf_values_cute_ms"] = _event_ms(
            _pack_leaf_values,
            iters=args.iters,
            warmup=args.warmup,
        )
    memory = {}
    if not args.no_memory:
        cute_memory = _peak_memory(_cute_forward_backward)
        torch_memory = _peak_memory(_torch_forward_backward)
        memory = {
            "setup_allocated_mib": _mib(torch.cuda.memory_allocated()),
            "setup_reserved_mib": _mib(torch.cuda.memory_reserved()),
            "fwd_bwd_cute_peak_mib": cute_memory["peak_mib"],
            "fwd_bwd_cute_temp_mib": cute_memory["temp_mib"],
            "fwd_bwd_cute_after_delta_mib": cute_memory["after_delta_mib"],
            "fwd_bwd_torch_peak_mib": torch_memory["peak_mib"],
            "fwd_bwd_torch_temp_mib": torch_memory["temp_mib"],
            "fwd_bwd_torch_after_delta_mib": torch_memory["after_delta_mib"],
        }

    print(
        {
            "n_nodes": args.n_nodes,
            "n_edges": args.n_nodes * args.avg_out,
            "n_heads": args.n_heads,
            "n_queries": args.n_queries,
            "leaf_entries": args.n_queries * args.leaves_per_query,
            "head_dim_v": args.head_dim_v,
            "n_iters": args.n_iters,
            "dtype": args.dtype,
            "fused_readout_bwd": args.fused_readout_bwd,
            "leaf_major_stats": args.leaf_major_stats,
            "reuse_forward_denom": args.reuse_forward_denom,
            "fp32_backward_state": args.fp32_backward_state,
            "pack_leaf_values": args.pack_leaf_values,
            "query_warp_stats": args.query_warp_stats,
            "query_warp_readout": args.query_warp_readout,
            "query_warp_scatter": args.query_warp_scatter,
            "query_warp_fused_bwd": args.query_warp_fused_bwd,
            **{key: round(value, 4) for key, value in timings.items()},
            **{key: round(value, 2) for key, value in memory.items()},
        }
    )


if __name__ == "__main__":
    main()
