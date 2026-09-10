# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Ascend NPU implementation of the cross-platform language_extra primitives.

The dispatch layer (``triton_dist/language/extra/language_extra.py``) routes
calls to this module on Ascend platforms through ``ModuleProxy``.

Design notes
------------
- The CUDA/HIP versions emit PTX / LLVM inline assembly with explicit memory
  ordering.  Ascend (CANN) has no SIMT thread model and global memory is
  hardware-coherent across daVinci cores, so ``scope``/``semantic`` arguments
  are *validated* but not forwarded to the underlying ``tl.*`` calls by
  default.  Dropping the hints does not change correctness on Ascend; it only
  gives up a potential optimization opportunity.  If the installed CANN Triton
  build is confirmed to accept ``sem``/``scope`` keyword arguments on atomics
  (run ``scripts/probe_ascend_language_api.py`` on an NPU machine to check),
  set ``_MEM_ORDER_PASSTHROUGH = True`` below to forward them.
- ``tid(axis=0)`` is emulated with CANN's ``sub_vec_id()`` (the sub-vector
  lane index within one core), matching the election pattern
  (``if sub_vec_id() == 0:``) used by the Ascend tutorials and tests.
  ``axis=1/2`` have no Ascend counterpart and are rejected at compile time.
- ``pack``/``unpack`` are portable pure-Python bit manipulations, ported from
  the HIP implementation (``extra/hip/language_extra.py``).
"""
import triton.language as tl
from triton.language import core
from triton_dist.language import vector, make_vector

try:
    from triton.language.extra.cann.extension import sub_vec_id as _cann_sub_vec_id
except ImportError:  # non-CANN Triton build (e.g. development machine)
    _cann_sub_vec_id = None

# Whether to forward sem/scope kwargs to tl atomic ops.  See module docstring.
_MEM_ORDER_PASSTHROUGH = False

_VALID_SCOPES = ("cta", "gpu", "sys")
_ATOMIC_SEMANTICS = ("relaxed", "acquire", "release", "acq_rel")
_LOAD_SEMANTICS = ("relaxed", "monotonic", "acquire")
_STORE_SEMANTICS = ("relaxed", "monotonic", "release")


def _unwrap(x):
    return core._unwrap_if_constexpr(x)


def _validate_mem_order(scope, semantic, valid_semantics, fn_name):
    """Validate CUDA-style scope/semantic strings accepted by the dispatch layer."""
    scope = _unwrap(scope)
    semantic = _unwrap(semantic)
    if scope not in _VALID_SCOPES:
        raise ValueError(f"{fn_name}: scope should be one of {list(_VALID_SCOPES)}, got {scope!r}")
    if semantic not in valid_semantics:
        raise ValueError(f"{fn_name}: semantic should be one of {list(valid_semantics)}, got {semantic!r}")
    return scope, semantic


@core.extern
def tid(axis: core.constexpr, _semantic=None):
    """Sub-vector lane index within the current daVinci core (``axis=0`` only).

    Ascend has no SIMT thread model; ``tid(0)`` maps to CANN's ``sub_vec_id()``
    so that CUDA-style single-lane election (``if tid(0) == 0:``) keeps working.
    """
    axis = _unwrap(axis)
    if axis != 0:
        tl.static_assert(False,
                         "Ascend NPU has no SIMT thread model; tid() only supports axis=0 (mapped to sub_vec_id())",
                         _semantic=_semantic)
    if _cann_sub_vec_id is None:
        raise RuntimeError("tid() on Ascend requires triton.language.extra.cann.extension.sub_vec_id; "
                           "install the triton-ascend (CANN) build of Triton")
    return _cann_sub_vec_id(_semantic=_semantic)


@core.extern
def atomic_add(ptr, value, scope="gpu", semantic="relaxed", _semantic=None):
    """Atomically add ``value`` to ``*ptr`` and return the old value.

    ``ptr`` must be a pointer to an integer type (same contract as the CUDA
    implementation).  ``scope``/``semantic`` are validated; forwarding is
    controlled by ``_MEM_ORDER_PASSTHROUGH``.
    """
    scope, semantic = _validate_mem_order(scope, semantic, _ATOMIC_SEMANTICS, "atomic_add")
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int(),
        "atomic_add: ptr must be a pointer of int type",
        _semantic=_semantic,
    )
    value = core.cast(value, dtype=ptr.dtype.element_ty, _semantic=_semantic)
    if _MEM_ORDER_PASSTHROUGH:
        return tl.atomic_add(ptr, value, sem=semantic, scope=scope, _semantic=_semantic)
    return tl.atomic_add(ptr, value, _semantic=_semantic)


@core.extern
def atomic_cas(ptr, cmp_val, target_val, scope="gpu", semantic="relaxed", _semantic=None):
    """Atomically compare ``*ptr`` with ``cmp_val``; if equal, store ``target_val``.

    Returns the old value of ``*ptr``.  ``ptr`` must be a pointer to an integer
    type.
    """
    scope, semantic = _validate_mem_order(scope, semantic, _ATOMIC_SEMANTICS, "atomic_cas")
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int(),
        "atomic_cas: ptr must be a pointer of int type",
        _semantic=_semantic,
    )
    if not hasattr(tl, "atomic_cas"):
        raise NotImplementedError("atomic_cas is not provided by this Triton (CANN) build; "
                                  "cannot emulate compare-and-swap on Ascend")
    cmp_val = core.cast(cmp_val, dtype=ptr.dtype.element_ty, _semantic=_semantic)
    target_val = core.cast(target_val, dtype=ptr.dtype.element_ty, _semantic=_semantic)
    if _MEM_ORDER_PASSTHROUGH:
        return tl.atomic_cas(ptr, cmp_val, target_val, sem=semantic, scope=scope, _semantic=_semantic)
    return tl.atomic_cas(ptr, cmp_val, target_val, _semantic=_semantic)


@core.extern
def __syncthreads(_semantic=None):
    """Program-instance barrier (API-compatibility shim).

    daVinci sub-vector lanes execute in SIMD lockstep, so an explicit barrier is
    usually redundant on Ascend; mapped to ``tl.debug_barrier()`` (same strategy
    as the MACA backend).
    """
    if hasattr(tl, "debug_barrier"):
        return tl.debug_barrier(_semantic=_semantic)
    return _semantic.builder.create_barrier()


@core.extern
def ld(ptr, scope="gpu", semantic="relaxed", _semantic=None):
    """Load ``*ptr``.

    ``semantic`` should be one of ["relaxed", "monotonic", "acquire"].
    ``tl.load`` has no memory-order knob; ordering is guaranteed by Ascend's
    hardware-coherent global memory (see module docstring).
    """
    _validate_mem_order(scope, semantic, _LOAD_SEMANTICS, "ld")
    tl.static_assert(ptr.dtype.is_ptr(), "ld: ptr should be a pointer", _semantic=_semantic)
    return tl.load(ptr, _semantic=_semantic)


@core.extern
def st(ptr, val, scope="gpu", semantic="relaxed", _semantic=None):
    """Store ``val`` to ``*ptr``.

    ``semantic`` should be one of ["relaxed", "monotonic", "release"].
    Forwards to :func:`st_vector` when ``val`` is a :class:`vector`
    (same behavior as the CUDA implementation).
    """
    _validate_mem_order(scope, semantic, _STORE_SEMANTICS, "st")
    tl.static_assert(ptr.dtype.is_ptr(), "st: ptr should be a pointer", _semantic=_semantic)
    if isinstance(val, vector):
        return st_vector(ptr, val, scope, semantic, _semantic=_semantic)
    val = core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic)
    return tl.store(ptr, val, _semantic=_semantic)


@core.extern
def ld_vector(ptr, vec_size: core.constexpr = 1, scope="gpu", semantic="relaxed", _semantic=None):
    """Load ``vec_size`` consecutive elements starting at ``ptr`` into a vector."""
    assert isinstance(vec_size, tl.constexpr), "ld_vector: vec_size must be a constexpr"
    ret = []
    for i in range(vec_size):
        ret.append(ld(tl.add(ptr, i, _semantic=_semantic), scope=scope, semantic=semantic, _semantic=_semantic))
    return make_vector(ret, _semantic=_semantic)


@core.extern
def st_vector(ptr, vec, scope="gpu", semantic="relaxed", _semantic=None):
    """Store all elements of ``vec`` to consecutive memory starting at ``ptr``."""
    assert isinstance(vec, vector), "st_vector: vec must be a vector"
    for idx, v in enumerate(vec):
        st(tl.add(ptr, idx, _semantic=_semantic), v, scope=scope, semantic=semantic, _semantic=_semantic)


@core.extern
def pack(src: vector, dst_type, _semantic=None):
    """Pack a vector of smaller-bitwidth elements into one larger scalar.
    E.g. vector([i32_lo, i32_hi]) -> i64 via shift+or (portable, no PTX needed)."""
    assert isinstance(src, vector)
    dst_type = _unwrap(dst_type)  # tolerate constexpr-wrapped dtype from the dispatch layer
    dst_nbits = dst_type.primitive_bitwidth
    src_elem_dtype = src.type.elem_type
    assert src.type.vector_nbits == dst_nbits, (f"src.type.vector_nbits {src.type.vector_nbits} != "
                                                f"dst_type.primitive_bitwidth {dst_nbits}")
    src_nbits = src_elem_dtype.primitive_bitwidth
    src_int_ty = core.get_int_dtype(src_nbits, False)
    dst_int_ty = core.get_int_dtype(dst_nbits, False)
    combined = tl.cast(tl.constexpr(0), dst_int_ty, _semantic=_semantic)
    for j in range(src.type.vec_size):
        bits = tl.cast(src.values[j], src_int_ty, bitcast=True, _semantic=_semantic)
        bits = tl.cast(bits, dst_int_ty, _semantic=_semantic)
        shifted = bits.__lshift__(j * src_nbits, _semantic=_semantic)
        combined = combined.__or__(shifted, _semantic=_semantic)
    return tl.cast(combined, dst_type, bitcast=True, _semantic=_semantic)


@core.extern
def unpack(src, dst_type, _semantic=None):
    """Unpack a larger scalar into multiple smaller-bitwidth elements.
    E.g. i64 -> (i32_lo, i32_hi) via shift+mask (portable, no PTX needed)."""
    dst_type = _unwrap(dst_type)  # tolerate constexpr-wrapped dtype from the dispatch layer
    src_nbits = src.dtype.primitive_bitwidth
    dst_nbits = dst_type.primitive_bitwidth
    assert src_nbits % dst_nbits == 0
    ratio = src_nbits // dst_nbits
    src_int_ty = core.get_int_dtype(src_nbits, False)
    dst_int_ty = core.get_int_dtype(dst_nbits, False)
    int_val = tl.cast(src, src_int_ty, bitcast=True, _semantic=_semantic)
    mask_val = (1 << dst_nbits) - 1
    results = []
    for j in range(ratio):
        shifted = int_val.__rshift__(j * dst_nbits, _semantic=_semantic)
        masked = shifted.__and__(mask_val, _semantic=_semantic)
        elem = tl.cast(masked, dst_int_ty, _semantic=_semantic)
        elem = tl.cast(elem, dst_type, bitcast=True, _semantic=_semantic)
        results.append(elem)
    return results


__all__ = [
    "tid",
    "atomic_cas",
    "atomic_add",
    "__syncthreads",
    "ld",
    "st",
    "ld_vector",
    "st_vector",
    "pack",
    "unpack",
]
