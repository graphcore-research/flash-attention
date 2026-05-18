"""Flash Attention CUTE (CUDA Template Engine) implementation."""

import re
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir as mlir_ir


_SWIZZLE_TRIPLE_RE = re.compile(r"S<(-?\d+),(-?\d+),(-?\d+)>")


def _install_swizzle_type_attr_compat() -> None:
    """Backfill `SwizzleType.num_*` for newer CUTLASS MLIR type wrappers."""
    try:
        with mlir_ir.Context():
            swizzle_type_cls = type(mlir_ir.Type.parse('!cute.swizzle<"S<3,4,3>">'))
    except Exception:
        return
    if hasattr(swizzle_type_cls, "num_bits"):
        return

    def _get_swizzle_group(self, group_idx: int) -> int:
        match = _SWIZZLE_TRIPLE_RE.search(str(self))
        if match is None:
            raise AttributeError(f"Unable to recover swizzle parameters from {self!r}")
        return int(match.group(group_idx))

    swizzle_type_cls.num_bits = property(lambda self: _get_swizzle_group(self, 1))
    swizzle_type_cls.num_base = property(lambda self: _get_swizzle_group(self, 2))
    swizzle_type_cls.num_shift = property(lambda self: _get_swizzle_group(self, 3))


_install_swizzle_type_attr_compat()

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

from flash_attn.cute.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
