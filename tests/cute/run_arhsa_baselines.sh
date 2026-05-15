#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
PYTHON="${PYTHON:-python}"

COMMON_ARHSA_ARGS=(
  --graph-mode level_dag
  --level-range-kernels
  --n-iters 4
  --no-check
  --query-warp-readout
  --query-value-pack-readout
  --fp32-backward-state
  --auto-readout-bwd
  --leaf-value-pattern shared-block16
  --compact-value-rows
  --incoming-packed-step
  --save-forward-history
)

echo "== ARHSA compact level_dag: 512K nodes / 131K queries =="
CUDA_VISIBLE_DEVICES="${GPU}" timeout 900s "${PYTHON}" tests/cute/benchmark_arhsa_walk.py \
  "${COMMON_ARHSA_ARGS[@]}" \
  --n-nodes 524288 \
  --n-queries 131072 \
  --iters 40 \
  --warmup 8

echo "== ARHSA compact level_dag: 2M nodes / 524K queries =="
CUDA_VISIBLE_DEVICES="${GPU}" timeout 1200s "${PYTHON}" tests/cute/benchmark_arhsa_walk.py \
  "${COMMON_ARHSA_ARGS[@]}" \
  --n-nodes 2097152 \
  --n-queries 524288 \
  --iters 25 \
  --warmup 6

echo "== ARHSA 512K plus dense causal FA4 32K comparison =="
CUDA_VISIBLE_DEVICES="${GPU}" timeout 900s "${PYTHON}" tests/cute/benchmark_arhsa_walk.py \
  "${COMMON_ARHSA_ARGS[@]}" \
  --n-nodes 524288 \
  --n-queries 131072 \
  --iters 10 \
  --warmup 3 \
  --compare-fa4 \
  --fa4-seqlen 32768 \
  --fa4-iters 2 \
  --fa4-warmup 1

echo "== Standalone dense causal FA4 64K =="
CUDA_VISIBLE_DEVICES="${GPU}" timeout 900s "${PYTHON}" -u - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import torch
from flash_attn.cute import flash_attn_func


def unwrap(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def event_ms(fn, *, iters, warmup):
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


B, S, H, D = 1, 65536, 8, 64
q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)


def fwd():
    return unwrap(flash_attn_func(q, k, v, causal=True, window_size=(-1, -1)))


def fwd_bwd():
    q_run = q.detach().clone().requires_grad_(True)
    k_run = k.detach().clone().requires_grad_(True)
    v_run = v.detach().clone().requires_grad_(True)
    out = unwrap(flash_attn_func(q_run, k_run, v_run, causal=True, window_size=(-1, -1)))
    out.backward(torch.ones_like(out))


fwd_ms = event_ms(fwd, iters=2, warmup=1)
fwd_bwd_ms = event_ms(fwd_bwd, iters=2, warmup=1)
print(
    {
        "fa4_status": "measured_standalone",
        "fa4_label": "dense_causal",
        "fa4_seqlen": S,
        "fa4_fwd_ms": round(fwd_ms, 4),
        "fa4_fwd_bwd_ms": round(fwd_bwd_ms, 4),
        "fa4_bwd_ms": round(fwd_bwd_ms - fwd_ms, 4),
        "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
)
PY
