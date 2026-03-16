"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import cutlass.cute as cute

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)
from .hsa import (
    HSASchedule,
    backward_packed_masks_to_attend_mask,
    backward_descriptors_to_attend_mask,
    build_hsa_schedule,
    compute_hsa_mask,
    flash_attn_hsa_func,
    flash_attn_hsa_sparse_func,
    fused_forward_to_attend_mask,
    forward_descriptors_to_attend_mask,
    get_hsa_mask_mod,
    hybrid_backward_to_attend_mask,
    hsa_reference_attention,
    hsa_sparse_reference_attention,
    schedule_to_attend_mask,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
    "flash_attn_hsa_func",
    "flash_attn_hsa_sparse_func",
    "get_hsa_mask_mod",
    "build_hsa_schedule",
    "schedule_to_attend_mask",
    "forward_descriptors_to_attend_mask",
    "fused_forward_to_attend_mask",
    "backward_descriptors_to_attend_mask",
    "backward_packed_masks_to_attend_mask",
    "hybrid_backward_to_attend_mask",
    "compute_hsa_mask",
    "hsa_reference_attention",
    "hsa_sparse_reference_attention",
    "HSASchedule",
]
