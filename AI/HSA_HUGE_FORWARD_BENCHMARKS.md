# HSA Huge Forward Benchmarks

## Scope

This note captures bespoke forward-only HSA benchmark results at huge sequence
lengths for the two main sparse-HSA forward baselines:

- `plain sparse`: `hsa_sparse_mask_plain`
- `hybrid`: `direct_micro_fwd_sparse_mask_bwd`

The huge-length runner used a forward-only runtime shortcut to avoid paying the
full backward sparse-mask construction cost when the measurement only needs
forward timings. The shortcut was validated against the original hybrid forward
on the same `128k` tensors:

- `out_max_diff = 0.0`

## Forward Results

Format: forward milliseconds.

| Sequence | Plain Sparse | Hybrid | Hybrid Speedup |
| --- | ---: | ---: | ---: |
| `1M` | `21.040 ms` | `3.346 ms` | `6.29x` |
| `2M` | `40.474 ms` | `6.063 ms` | `6.68x` |
| `5M` | `101.037 ms` | `14.771 ms` | `6.84x` |
| `10M` | `200.884 ms` | `62.296 ms` | `3.22x` |

## Setup Cost Breakdown

These runs were dominated by host/runtime setup rather than the kernel itself.
The table below records the major setup stages from the bespoke forward-only
runner.

| Sequence | Metadata | Schedule | Plain Stage | Hybrid Stage |
| --- | ---: | ---: | ---: | ---: |
| `1M` | `3.30 s` | `15.64 s` | `74.01 s` | `90.37 s` |
| `2M` | `6.34 s` | `26.40 s` | `119.88 s` | `181.34 s` |
| `5M` | `15.82 s` | `65.71 s` | `309.83 s` | `492.45 s` |
| `10M` | `32.71 s` | `140.84 s` | `919.96 s` | `1272.86 s` |

Interpretation:

- `metadata`: synthetic `keep_ids/hash_ids` construction
- `schedule`: `build_hsa_schedule(...)`
- `plain stage`: forward sparse runtime construction + timed plain forward
- `hybrid stage`: forward sparse runtime construction + synthetic-grid metadata + timed hybrid forward

## Takeaways

- Hybrid forward remains substantially faster than the plain sparse HSA
  baseline through `10M`.
- The huge-length wall is mostly host/runtime setup, not the forward kernel.
- At `10M`, the run still fit on the device in this forward-only mode; observed
  HBM usage was about `56.9 GB`.
- The speedup is stable through `5M` and then drops at `10M`, suggesting the
  synthetic-grid forward setup cost is becoming a larger fraction of total time.

## Notes

- These numbers came from a bespoke temp runner in `/tmp`, not the full default
  benchmark harness.
- The full harness still pays large backward-runtime construction costs even for
  forward-only sweeps, which makes it impractical for `5M+` interactive runs.
