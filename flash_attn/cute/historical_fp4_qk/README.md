Historical integrated FP4 QK forward kernel snapshots for side-by-side recovery work.

Files:
- `fp4_flash_fwd_sm100_46b15ea.py`
  - source commit: `46b15ea`
  - best remembered MHA checkpoint from the earlier sweep
  - reference rows: `mha d64 s512 = 0.714x`, `mha d128 s512 = 0.834x` vs BF16
- `fp4_flash_fwd_sm100_dcf6b1b.py`
  - source commit: `dcf6b1b`
  - best remembered `gqa d128` checkpoint from the earlier sweep
  - reference row: `gqa d128 s512 = 0.844x` vs BF16

These files are reference-only snapshots copied directly from Git history so we can diff or transplant the old QK-only fast path without rewriting current history.
