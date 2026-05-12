# HSA Long-Context Benchmark Results

Date: 2026-04-14 UTC

## Scope

These are the long-context benchmark results collected so far for:

- `sparse_mask_plain` / HSA sparse-mask baseline
- regular dense FA4

The relevant harness is [benchmark_hsa.py](./benchmark_hsa.py).

Official measurements below were taken on physical GPU `3` by setting:

```bash
CUDA_VISIBLE_DEVICES=3
```

Inside the benchmark process, that physical GPU appears as `cuda:0`. The script logs:

```text
device=NVIDIA GB200 capability=(10, 0)
```

## Exact Results

### Forward + Backward, exact through 2M

| Seqlen | sparse_mask fwd+bwd ms | dense FA4 fwd+bwd ms | dense / sparse |
| --- | ---: | ---: | ---: |
| 16K | 1.228 | 1.336 | 1.09x |
| 32K | 1.597 | 2.855 | 1.79x |
| 64K | 2.345 | 9.223 | 3.93x |
| 128K | 4.186 | 35.435 | 8.47x |
| 256K | 7.714 | 139.429 | 18.07x |
| 512K | 15.169 | 554.557 | 36.56x |
| 1M | 30.108 | 2216.600 | 73.62x |
| 2M | 59.939 | 8986.195 | 149.92x |

### Forward-only, exact through 4M

| Seqlen | sparse_mask fwd ms | dense FA4 fwd ms | dense / sparse |
| --- | ---: | ---: | ---: |
| 16K | 1.237 | 0.565 | 0.46x |
| 32K | 1.743 | 0.796 | 0.46x |
| 64K | 2.216 | 2.280 | 1.03x |
| 128K | 3.499 | 8.662 | 2.48x |
| 256K | 6.124 | 34.056 | 5.56x |
| 512K | 11.537 | 135.342 | 11.73x |
| 1M | 22.471 | 555.894 | 24.74x |
| 2M | 43.610 | 2216.316 | 50.82x |
| 4M | 85.835 | 8877.439 | 103.42x |

### 10M

Exact dense FA4 forward-only timing:

| Seqlen | sparse_mask fwd ms | dense FA4 fwd ms | status |
| --- | ---: | ---: | --- |
| 10M | n/a | 53625.970 | dense measured, sparse failed |

## Failure Boundary

### Sparse 8M

- `8M` sparse does not currently produce a usable official number.
- The run fails with `CUDA error: an illegal memory access was encountered`.

### Sparse 10M

- `10M` sparse now gets past long schedule/runtime construction and reaches the actual block-sparse forward kernel.
- With `CUDA_LAUNCH_BLOCKING=1`, the failure resolves to:

```text
flash_attn_hsa_sparse_func
  -> _FlashAttnHSABlockSparseFunc.forward
  -> run_hsa_fwd_sm100_blocksparse
  -> _run_hsa_blocksparse_forward
  -> flash_attn_fwd(...)
RuntimeError: CUDA Error: cudaErrorIllegalAddress
```

Relevant call sites in the current local tree:

- [flash_attn/cute/hsa.py](../../flash_attn/cute/hsa.py#L7356)
- [flash_attn/cute/hsa.py](../../flash_attn/cute/hsa.py#L7175)

Current practical boundary for valid exact sparse numbers in this setup:

- last exact sparse forward-only result: `4M`
- `8M`: illegal memory access
- `10M`: illegal memory access

## Commands Used

### Exact full forward + backward through 2M

```bash
CUDA_VISIBLE_DEVICES=3 \
FLASH_ATTN_HSA_BENCH_MAX_RETRIES=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_ONLY=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_SEQLENS=16k,32k,64k,128k,256k,512k,1M,2M \
FLASH_ATTN_HSA_PRIMARY_LONG_WARMUP_ITERS=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_BENCHMARK_ITERS=1 \
/tmp/flash-attn-cute-bench-venv/bin/python tests/cute/benchmark_hsa.py
```

### Exact forward-only through 4M

```bash
CUDA_VISIBLE_DEVICES=3 \
FLASH_ATTN_HSA_BENCH_MAX_RETRIES=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_ONLY=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_FORWARD_ONLY=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_MINIMAL=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_SEQLENS=4M \
FLASH_ATTN_HSA_PRIMARY_LONG_WARMUP_ITERS=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_BENCHMARK_ITERS=1 \
/tmp/flash-attn-cute-bench-venv/bin/python tests/cute/benchmark_hsa.py
```

### Exact dense 10M forward-only line collected from the same harness

```bash
CUDA_VISIBLE_DEVICES=3 \
FLASH_ATTN_HSA_BENCH_CHILD=1 \
FLASH_ATTN_HSA_BENCH_ATTEMPT=1 \
FLASH_ATTN_HSA_BENCH_MAX_RETRIES=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_ONLY=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_FORWARD_ONLY=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_MINIMAL=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_SEQLENS=10M \
FLASH_ATTN_HSA_PRIMARY_LONG_WARMUP_ITERS=1 \
FLASH_ATTN_HSA_PRIMARY_LONG_BENCHMARK_ITERS=1 \
/tmp/flash-attn-cute-bench-venv/bin/python tests/cute/benchmark_hsa.py
```

Observed output:

```text
long-10M: mode=mixed_sparse_mask bwd_block_q=64 bwd_block_k=128 bwd_subtile_factor=1 primary_only=1 forward_only=1 primary_minimal=1 shape=(B=1, T=10485760, H=4, KV=4, D=64) sparse_mask_plain_label=hsa_sparse_mask_plain sparse_mask_plain_status=unavailable_RuntimeError dense_fa4_fwd_ms=53625.970
```

## Notes On Local Benchmarking Changes

These results were gathered with local long-context benchmarking support and sparse setup optimizations added in:

- [tests/cute/benchmark_hsa.py](./benchmark_hsa.py)
- [flash_attn/cute/hsa.py](../../flash_attn/cute/hsa.py)

The setup-side changes were enough to move the `10M` sparse run from schedule/runtime construction bottlenecks to the actual block-sparse forward kernel, but they did not eliminate the kernel-side illegal memory access at `8M+`.

## GPU Availability Notes

- Physical GPU `3` was the clean benchmark device used for the official results above.
- Physical GPU `0` was not usable during follow-up testing and returned `cudaErrorDevicesUnavailable` even for a trivial CUDA tensor allocation.
- Physical GPU `1` accepted work but was contended by other jobs, so it was not used for official result reporting.
