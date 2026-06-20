import argparse
import json

import torch

from flash_attn.cute import (
    analyze_explicit_2d_sparse_forward,
    summarize_explicit_2d_sparse_forward,
    summarize_explicit_2d_sparse_suite,
)


DEFAULT_CASES = (
    {
        "name": "confetti-4k",
        "case_family": "disjoint_confetti",
        "seqlen": 4096,
        "heads": 16,
        "head_dim": 64,
        "packed_q": 16,
        "support_k": 64,
        "islands_per_row": 4,
        "island_width": 4,
        "row_shift": 9,
    },
    {
        "name": "confetti-16k",
        "case_family": "disjoint_confetti",
        "seqlen": 16384,
        "heads": 16,
        "head_dim": 64,
        "packed_q": 16,
        "support_k": 128,
        "islands_per_row": 8,
        "island_width": 4,
        "row_shift": 9,
    },
    {
        "name": "compact-control",
        "case_family": "compact_control",
        "seqlen": 4096,
        "heads": 16,
        "head_dim": 64,
        "packed_q": 16,
        "support_k": 64,
        "islands_per_row": 4,
        "island_width": 4,
        "row_shift": 2,
    },
)

VALID_VARIANTS = (
    "dense",
    "custom_masked",
    "fa4_packed",
    "direct_2d",
    "direct_2d_compact",
    "direct_2d_tc",
    "shared_support",
)


def _format_ms(payload):
    if not isinstance(payload, dict) or payload.get("status") != "measured":
        return str(payload.get("status", "-")) if isinstance(payload, dict) else "-"
    return f"{float(payload['fwd_ms']):.3f}"


def _format_best(summary):
    best = summary.get("best_variant")
    if not isinstance(best, dict):
        return "-"
    return f"{best['name']}={float(best['fwd_ms']):.3f}"


def _variant_column(variant: str) -> str:
    return {
        "dense": "dense_ms",
        "custom_masked": "custom_ms",
        "fa4_packed": "fa4_ms",
        "direct_2d": "direct2d_ms",
        "direct_2d_compact": "direct2d_compact_ms",
        "direct_2d_tc": "direct2d_tc_ms",
        "shared_support": "shared_ms",
    }[variant]


def _custom_case_requested(args) -> bool:
    return any(
        getattr(args, key) is not None
        for key in (
            "seqlen",
            "heads",
            "head_dim",
            "packed_q",
            "support_k",
            "islands_per_row",
            "island_width",
            "row_shift",
        )
    )


def _default_support_k(seqlen: int) -> int:
    return 64 if seqlen <= 4096 else 128


def _default_islands_per_row(seqlen: int) -> int:
    return 4 if seqlen <= 4096 else 8


def _build_case_specs(args) -> list[dict]:
    if not _custom_case_requested(args):
        if args.case_family == "all":
            return [dict(case) for case in DEFAULT_CASES]
        return [dict(case) for case in DEFAULT_CASES if case["case_family"] == args.case_family]

    families = (
        ("disjoint_confetti", "compact_control")
        if args.case_family == "all"
        else (args.case_family,)
    )
    specs = []
    for family in families:
        seqlen = 4096 if args.seqlen is None else args.seqlen
        heads = 16 if args.heads is None else args.heads
        head_dim = 64 if args.head_dim is None else args.head_dim
        packed_q = 16 if args.packed_q is None else args.packed_q
        support_k = _default_support_k(seqlen) if args.support_k is None else args.support_k
        islands_per_row = _default_islands_per_row(seqlen) if args.islands_per_row is None else args.islands_per_row
        island_width = 4 if args.island_width is None else args.island_width
        row_shift = (9 if family == "disjoint_confetti" else 2) if args.row_shift is None else args.row_shift
        specs.append(
            {
                "name": f"{family}-{seqlen}",
                "case_family": family,
                "seqlen": seqlen,
                "heads": heads,
                "head_dim": head_dim,
                "packed_q": packed_q,
                "support_k": support_k,
                "islands_per_row": islands_per_row,
                "island_width": island_width,
                "row_shift": row_shift,
            }
        )
    return specs


def _parse_variants(raw: str) -> tuple[str, ...]:
    variants = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not variants:
        raise ValueError("expected at least one variant")
    unknown = sorted(set(variants) - set(VALID_VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants {unknown}; expected one of {VALID_VARIANTS}")
    return variants


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark explicit 2D sparse packed forward cases")
    parser.add_argument("--case-family", choices=("disjoint_confetti", "compact_control", "all"), default="all")
    parser.add_argument("--seqlen", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--packed-q", type=int, default=None)
    parser.add_argument("--support-k", type=int, default=None)
    parser.add_argument("--islands-per-row", type=int, default=None)
    parser.add_argument("--island-width", type=int, default=None)
    parser.add_argument("--row-shift", type=int, default=None)
    parser.add_argument("--variants", default="dense,custom_masked,fa4_packed,direct_2d")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=20)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the dense PyTorch oracle and output diff checks for long perf-only runs.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv=None):
    parser = _make_parser()
    args = parser.parse_args(argv)
    variants = _parse_variants(args.variants)
    case_specs = _build_case_specs(args)
    case_payloads = []

    metric_columns = [_variant_column(variant) for variant in variants]
    print(" ".join(["case", "family", "live_pairs", "fill", "payload_s", "geom_s", "build_s", *metric_columns, "best", "go_no_go"]))
    for case_idx, spec in enumerate(case_specs):
        report = analyze_explicit_2d_sparse_forward(
            case_family=spec["case_family"],
            seqlen=spec["seqlen"],
            heads=spec["heads"],
            head_dim=spec["head_dim"],
            packed_q=spec["packed_q"],
            support_k=spec["support_k"],
            islands_per_row=spec["islands_per_row"],
            island_width=spec["island_width"],
            row_shift=spec["row_shift"],
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            variants=variants,
            seed=case_idx,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            check_correctness=not args.skip_correctness,
        )
        summary = summarize_explicit_2d_sparse_forward(report)
        geometry = report["geometry"]
        results = report["results"]
        metric_values = [_format_ms(results.get(variant, {})) for variant in variants]
        print(
            " ".join(
                [
                    spec["name"],
                    spec["case_family"],
                    str(int(geometry["live_pairs"])),
                    f"{float(geometry['fill_rate']):.4f}",
                    f"{float(report.get('payload_build_excluding_diagnostic_geometry_seconds', float('nan'))):.3f}",
                    f"{float(report.get('diagnostic_geometry_seconds', float('nan'))):.3f}",
                    f"{float(report.get('payload_build_seconds', float('nan'))):.3f}",
                    *metric_values,
                    _format_best(summary),
                    str(report["go_no_go"]["status"]),
                ]
            )
        )
        case_payloads.append(
            {
                "name": spec["name"],
                "config": spec,
                "report": report,
                "summary": summary,
            }
        )

    suite_summary = summarize_explicit_2d_sparse_suite(case_payloads)
    primary_confetti = suite_summary["primary_confetti_go_no_go"]
    runtime_routing = suite_summary["runtime_routing_recommendation"]
    print(
        "suite "
        f"primary_confetti={primary_confetti['status']} "
        f"runtime_routing={runtime_routing['status']} "
        f"cases={suite_summary['num_cases']}"
    )

    payload = {
        "cases": case_payloads,
        "suite_summary": suite_summary,
        "variants": list(variants),
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
    }
    if args.json:
        print(json.dumps(payload))
    return payload


if __name__ == "__main__":
    main()
