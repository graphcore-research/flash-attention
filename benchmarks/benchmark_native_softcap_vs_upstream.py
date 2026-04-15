#!/usr/bin/env python3
"""
Benchmark the native SC30 softcap baseline in the current checkout against
Dao-AILab/flash-attention@14f3627d44687513adff00819ec894e54bf92cd7.

Example:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/benchmark_native_softcap_vs_upstream.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Iterable


DEFAULT_UPSTREAM_REPO = "https://github.com/Dao-AILab/flash-attention.git"
DEFAULT_UPSTREAM_COMMIT = "14f3627d44687513adff00819ec894e54bf92cd7"
DEFAULT_SHAPES = ("1,4096,32,128", "1,8192,32,128")


WORKER = r"""
import json
import sys
import torch

repo = sys.argv[1]
payload = json.loads(sys.argv[2])
sys.path.insert(0, repo)

from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
from flash_attn.cute.utils import create_softcap_scoremod

try:
    from flash_attn.cute.utils import create_softcap_scoremod_bwd_native as create_softcap_scoremod_bwd
except ImportError:
    from flash_attn.cute.utils import create_softcap_scoremod_bwd as create_softcap_scoremod_bwd


def bench(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def run_shape(shape, softcap, warmup, reps, causal):
    batch, seqlen, nheads, headdim = shape
    dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=dtype)
    score_mod = create_softcap_scoremod(softcap)
    score_mod_bwd = create_softcap_scoremod_bwd(softcap)
    out, lse = _flash_attn_fwd(
        q,
        k,
        v,
        softcap=0.0,
        score_mod=score_mod,
        causal=causal,
        return_lse=True,
    )
    dout = torch.randn_like(out)
    fwd_us = bench(
        lambda: _flash_attn_fwd(
            q,
            k,
            v,
            softcap=0.0,
            score_mod=score_mod,
            causal=causal,
            return_lse=True,
        ),
        warmup=warmup,
        reps=reps,
    )
    bwd_us = bench(
        lambda: _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=None,
            causal=causal,
            softcap=0.0,
            score_mod=score_mod,
            score_mod_bwd=score_mod_bwd,
        ),
        warmup=warmup,
        reps=reps,
    )
    return {
        "shape": list(shape),
        "fwd_us": fwd_us,
        "bwd_us": bwd_us,
        "total_us": fwd_us + bwd_us,
    }


torch.cuda.init()
torch.manual_seed(payload["seed"])
results = [
    run_shape(tuple(shape), payload["softcap"], payload["warmup"], payload["reps"], payload["causal"])
    for shape in payload["shapes"]
]
print(json.dumps(results))
"""


def parse_shape(text: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Expected shape 'B,S,H,D', got {text!r}"
        )
    return tuple(parts)  # type: ignore[return-value]


def run_checked(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_head(repo: Path) -> str:
    return run_checked(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])


def git_branch(repo: Path) -> str:
    return run_checked(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"])


def ensure_upstream_worktree(
    current_repo: Path,
    upstream_repo: str,
    upstream_commit: str,
    worktree_root: Path,
) -> Path:
    worktree_root.mkdir(parents=True, exist_ok=True)
    worktree = worktree_root / f"flash-attn-upstream-{upstream_commit[:12]}"
    if worktree.exists():
        head = run_checked(["git", "-C", str(worktree), "rev-parse", "HEAD"])
        if head == upstream_commit:
            return worktree
        raise RuntimeError(
            f"Existing worktree at {worktree} is pinned to {head}, expected {upstream_commit}"
        )
    subprocess.run(
        ["git", "-C", str(current_repo), "fetch", "--no-tags", upstream_repo, upstream_commit],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(current_repo),
            "worktree",
            "add",
            "--detach",
            str(worktree),
            upstream_commit,
        ],
        check=True,
    )
    return worktree


def run_worker(
    python_executable: str,
    repo: Path,
    payload: dict[str, object],
) -> list[dict[str, float | list[int]]]:
    env = os.environ.copy()
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    proc = subprocess.run(
        [python_executable, "-c", WORKER, str(repo), json.dumps(payload)],
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker failed for {repo}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def summarize_runs(
    labels: Iterable[str],
    shapes: list[tuple[int, int, int, int]],
    trials: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, float | list[float]]]]:
    summary: dict[str, dict[str, dict[str, float | list[float]]]] = {label: {} for label in labels}
    for label in labels:
        label_trials = [trial for trial in trials if trial["label"] == label]
        for shape in shapes:
            shape_results = [
                next(result for result in trial["results"] if tuple(result["shape"]) == shape)
                for trial in label_trials
            ]
            fwd_runs = [float(result["fwd_us"]) for result in shape_results]
            bwd_runs = [float(result["bwd_us"]) for result in shape_results]
            total_runs = [float(result["total_us"]) for result in shape_results]
            summary[label][str(list(shape))] = {
                "fwd_us_runs": fwd_runs,
                "bwd_us_runs": bwd_runs,
                "total_us_runs": total_runs,
                "fwd_us_median": statistics.median(fwd_runs),
                "bwd_us_median": statistics.median(bwd_runs),
                "total_us_median": statistics.median(total_runs),
            }
    return summary


def print_report(
    current_label: str,
    upstream_label: str,
    shapes: list[tuple[int, int, int, int]],
    summary: dict[str, dict[str, dict[str, float | list[float]]]],
) -> None:
    print(f"current:  {current_label}")
    print(f"upstream: {upstream_label}")
    print()
    for shape in shapes:
        key = str(list(shape))
        current = summary["current"][key]
        upstream = summary["upstream"][key]
        current_total = float(current["total_us_median"])
        upstream_total = float(upstream["total_us_median"])
        ratio = upstream_total / current_total
        delta_pct = (upstream_total - current_total) / current_total * 100.0
        print(f"shape {shape}")
        print(
            f"  current  total={current_total / 1000.0:.3f} ms "
            f"(fwd={float(current['fwd_us_median']) / 1000.0:.3f} ms, "
            f"bwd={float(current['bwd_us_median']) / 1000.0:.3f} ms)"
        )
        print(
            f"  upstream total={upstream_total / 1000.0:.3f} ms "
            f"(fwd={float(upstream['fwd_us_median']) / 1000.0:.3f} ms, "
            f"bwd={float(upstream['bwd_us_median']) / 1000.0:.3f} ms)"
        )
        print(f"  ratio    upstream/current = {ratio:.3f}x ({delta_pct:.1f}% slower)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for workers")
    parser.add_argument("--upstream-repo", default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--upstream-commit", default=DEFAULT_UPSTREAM_COMMIT)
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        type=parse_shape,
        help="Benchmark shape as B,S,H,D. Repeat to add multiple shapes.",
    )
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "flash-attn-worktrees",
    )
    args = parser.parse_args()

    shapes = args.shape if args.shape is not None else [parse_shape(shape) for shape in DEFAULT_SHAPES]
    current_repo = repo_root()
    upstream_repo = ensure_upstream_worktree(
        current_repo=current_repo,
        upstream_repo=args.upstream_repo,
        upstream_commit=args.upstream_commit,
        worktree_root=args.worktree_root,
    )
    payload = {
        "shapes": [list(shape) for shape in shapes],
        "softcap": args.softcap,
        "warmup": args.warmup,
        "reps": args.reps,
        "seed": args.seed,
        "causal": args.causal,
    }

    trials: list[dict[str, object]] = []
    for _ in range(args.trials):
        for label, repo in (("current", current_repo), ("upstream", upstream_repo)):
            results = run_worker(args.python, repo, payload)
            trials.append({"label": label, "repo": str(repo), "results": results})

    summary = summarize_runs(("current", "upstream"), shapes, trials)
    output = {
        "current_branch": git_branch(current_repo),
        "current_head": git_head(current_repo),
        "upstream_commit": args.upstream_commit,
        "upstream_repo": args.upstream_repo,
        "shapes": [list(shape) for shape in shapes],
        "softcap": args.softcap,
        "warmup": args.warmup,
        "reps": args.reps,
        "trials": trials,
        "summary": summary,
    }
    if args.json:
        print(json.dumps(output, indent=2))
        return

    print_report(
        current_label=f"{git_branch(current_repo)}@{git_head(current_repo)}",
        upstream_label=args.upstream_commit[:12],
        shapes=shapes,
        summary=summary,
    )


if __name__ == "__main__":
    main()
