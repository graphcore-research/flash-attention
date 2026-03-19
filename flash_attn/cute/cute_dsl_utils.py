# Copyright (c) 2025, Tri Dao.

import os
import pathlib
import ctypes
from typing import Tuple
from functools import partial, lru_cache
from dataclasses import dataclass, fields

import torch

try:
    from triton.tools.disasm import extract
except ImportError:
    extract = None

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import JitArgument
from cutlass.base_dsl.runtime.dlpack_types import DLTensor, DLDataType
from cutlass.cutlass_dsl import NumericMeta
from cutlass.cute.runtime import from_dlpack, make_fake_tensor

StaticTypes = (cutlass.Constexpr, NumericMeta, int, bool, str, float, type(None))


class _DLManagedTensor(ctypes.Structure):
    pass


_DLPACK_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLPACK_DELETER),
]

_PYTHONAPI = ctypes.pythonapi
_PYTHONAPI.PyCapsule_GetPointer.restype = ctypes.c_void_p
_PYTHONAPI.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_FP4X2_DLPACK_DTYPE = DLDataType(17, 4, 2)


class PackedFP4x2Tensor:
    """Expose packed uint8 storage to TVM-FFI as float4_e2m1fnx2."""

    def __init__(self, tensor):
        self.tensor = tensor
        self._capsules = []

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, stream=None):
        stream_arg = -1 if stream is None else stream
        capsule = self.tensor.__dlpack__(stream=stream_arg)
        ptr = _PYTHONAPI.PyCapsule_GetPointer(capsule, b"dltensor")
        managed = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor))
        managed.contents.dl_tensor.dtype = _FP4X2_DLPACK_DTYPE
        self._capsules.append(capsule)
        return capsule


load_cubin_module_data_og = cutlass.base_dsl.runtime.cuda.load_cubin_module_data
cute_compile_og = cute.compile


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)


@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@dataclass
class ArgumentsBase(JitArgument):
    def __c_pointers__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        c_ptrs = []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__c_pointers__"):
                c_ptrs.extend(obj.__c_pointers__())
        return c_ptrs

    def __get_mlir_types__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, StaticTypes)]
        types, self._values_pos = [], []
        for obj in non_constexpr_fields:
            if hasattr(obj, "__get_mlir_types__"):
                obj_types = obj.__get_mlir_types__()
                types.extend(obj_types)
                self._values_pos.append(len(obj_types))
            else:
                self._values_pos.append(0)
        return types

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, StaticTypes)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, StaticTypes)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


def load_cubin_module_data_patched(cubin_data, filepath):
    pathlib.Path(filepath).write_bytes(cubin_data)
    return load_cubin_module_data_og(cubin_data)


def cute_compile_patched(*args, **kwargs):
    """A patched version of cute.compile that dump the SASS to a file if CUTE_CUBIN_PATH is set."""
    cubin_path = os.getenv("CUTE_CUBIN_PATH", None)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = partial(
            load_cubin_module_data_patched, filepath=cubin_path
        )
    output = cute_compile_og(*args, **kwargs)
    if cubin_path is not None:
        cutlass.base_dsl.runtime.cuda.load_cubin_module_data = load_cubin_module_data_og
        if extract is not None:
            sass = extract(cubin_path, None)
            pathlib.Path(cubin_path).with_suffix(".annotated.sass").write_text(sass)
    return output


def assume_strides_aligned(t):
    """Assume all strides except the last are divisible by 128 bits.

    Python int strides (e.g., stride=0 from GQA expand) are kept as-is
    since they're static and don't need alignment assumptions.
    """
    divby = 128 // t.element_type.width
    strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
    return (*strides, t.stride[-1])


def assume_tensor_aligned(t):
    """Rebuild a tensor with 128-bit aligned stride assumptions. Passes through None."""
    if t is None:
        return None
    return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))


def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)


def to_cute_fp4_tensor(
    t,
    assumed_align=16,
    leading_dim=-1,
    fully_dynamic=False,
    enable_tvm_ffi=True,
):
    """Build a logical FP4 compile-time tensor spec for packed uint8 storage.

    TVM-FFI already supports mapping packed `float4x2` runtime tensors to a
    logical FP4 view when the compile-time spec declares a logical FP4 tensor.
    We take that path here instead of reading `runtime._Tensor.layout`, which
    is unsupported in the installed Cutlass runtime.
    """
    del fully_dynamic, enable_tvm_ffi
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    assert t.stride(leading_dim) == 1, "Packed FP4 tensors must be contiguous in the packed dimension."
    logical_shape = list(t.shape)
    logical_shape[leading_dim] *= 2
    logical_stride = []
    for dim, stride in enumerate(t.stride()):
        logical_stride.append(1 if dim == leading_dim else stride * 2)
    return make_fake_tensor(
        cutlass.Float4E2M1FN,
        tuple(logical_shape),
        tuple(logical_stride),
        assumed_align=assumed_align,
    )


def to_cute_fp4_vt_tensor(
    t,
    assumed_align=16,
):
    """Build a logical FP4 Vt tensor spec from packed transposed storage.

    FP4 PV consumes operand-B as logical ``Vt = (D, S_k, H_k, B)``. The public
    packed runtime contract for that operand is ``(D, S_k // 2, H_k, B)``, where
    mode ``1`` packs pairs of sequence positions after transpose. This
    helper keeps the packed storage untouched while exposing the logical FP4 Vt
    view expected by the PV MMA path.
    """
    assert t.ndim == 4, "FP4 Vt helper expects packed storage shaped (D, S/2, H, B)."
    assert t.stride(1) == 1, "Packed FP4 Vt storage must keep the packed sequence dimension contiguous."
    logical_shape = (t.shape[0], t.shape[1] * 2, t.shape[2], t.shape[3])
    logical_stride = (
        t.stride(0) * 2,
        1,
        t.stride(2) * 2,
        t.stride(3) * 2,
    )
    return make_fake_tensor(
        cutlass.Float4E2M1FN,
        logical_shape,
        logical_stride,
        assumed_align=assumed_align,
    )


def to_cute_aux_tensor(t, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI, tailored to FlexAttention aux tensors.
    This allows the user to specify alignment and leading dimension for aux tensors used in
    custom score_mod callables.
    """
    assumed_align: int = getattr(t, "__assumed_align__", None)
    leading_dim: int = getattr(t, "__leading_dim__", None)
    fully_dynamic: bool = leading_dim is None

    return to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=leading_dim,
        fully_dynamic=fully_dynamic,
        enable_tvm_ffi=enable_tvm_ffi,
    )


def get_aux_tensor_metadata(aux_tensors):
    return tuple(
        (
            getattr(t, "__assumed_align__", 0),
            getattr(t, "__leading_dim__", -1),
            hasattr(t, "__leading_dim__"),
        )
        for t in aux_tensors
    )


def get_broadcast_dims(tensor: torch.Tensor) -> Tuple[bool, ...]:
    """Return tuple of bools indicating which dims have stride=0 (broadcast).

    This is useful for compile keys since CuTe's mark_layout_dynamic() keeps
    stride=0 as static, meaning kernels compiled with different broadcast
    patterns are not interchangeable.
    """
    return tuple(s == 0 for s in tensor.stride())


def to_tvm_ffi_fp4x2_tensor(tensor: torch.Tensor) -> PackedFP4x2Tensor:
    return PackedFP4x2Tensor(tensor)
