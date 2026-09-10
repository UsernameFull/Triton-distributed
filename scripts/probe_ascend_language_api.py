#!/usr/bin/env python3
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Probe the CANN Triton API surface on an Ascend NPU machine.

Purpose
-------
`python/triton_dist/language/extra/ascend/language_extra.py` makes a few
assumptions about the installed triton-ascend (CANN) build:

  1. `triton.language.extra.cann.extension.sub_vec_id` exists (used by `tid`).
  2. `tl.debug_barrier` exists (used by `__syncthreads`).
  3. `tl.atomic_add` / `tl.atomic_cas` exist; whether they accept the
     `sem=`/`scope=` keyword arguments decides the value of
     `_MEM_ORDER_PASSTHROUGH` in the ascend language_extra module.

Run this script ON THE NPU MACHINE *after* installing the updated package
(`pip install ./python`), then paste the output back when adjusting the
implementation.

Usage:
    python scripts/probe_ascend_language_api.py
"""
import inspect
import shutil
import sys

RESULTS = []


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" -- {detail}" if detail else ""))


def probe_environment():
    print("=" * 72)
    print("1. Environment")
    print("=" * 72)
    report("npu-smi on PATH", shutil.which("npu-smi") is not None)
    try:
        import torch
        import torch_npu  # noqa: F401
        report("torch_npu importable", True, f"torch={torch.__version__}, npu available={torch.npu.is_available()}")
    except Exception as e:  # noqa: BLE001
        report("torch_npu importable", False, repr(e))
    try:
        import triton
        report("triton version", True, f"{triton.__version__} from {triton.__file__}")
    except Exception as e:  # noqa: BLE001
        report("triton importable", False, repr(e))
        print("FATAL: triton not importable, aborting.")
        sys.exit(1)


def probe_symbols():
    print()
    print("=" * 72)
    print("2. Symbol availability")
    print("=" * 72)
    try:
        import triton.language.extra.cann.extension as ext
        symbols = [s for s in dir(ext) if not s.startswith("_")]
        report("cann.extension importable", True, f"symbols: {symbols}")
        report("cann.extension.sub_vec_id", hasattr(ext, "sub_vec_id"))
    except Exception as e:  # noqa: BLE001
        report("cann.extension importable", False, repr(e))

    import triton.language as tl
    for name in ("debug_barrier", "atomic_add", "atomic_cas", "load", "store"):
        report(f"tl.{name}", hasattr(tl, name))


def probe_signatures():
    print()
    print("=" * 72)
    print("3. Signatures (sem/scope support decides _MEM_ORDER_PASSTHROUGH)")
    print("=" * 72)
    import triton.language as tl
    for name in ("atomic_add", "atomic_cas", "load", "store"):
        fn = getattr(tl, name, None)
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters)
            has_sem = "sem" in params
            has_scope = "scope" in params
            report(f"tl.{name} signature", True, f"{sig}  (sem={has_sem}, scope={has_scope})")
        except (TypeError, ValueError) as e:
            report(f"tl.{name} signature", False, f"cannot introspect: {e}")


def probe_smoke_compile():
    print()
    print("=" * 72)
    print("4. Kernel smoke compilation on NPU (requires torch_npu + device)")
    print("=" * 72)
    try:
        import torch
        import torch_npu  # noqa: F401
        import triton
        import triton.language as tl
    except Exception as e:  # noqa: BLE001
        report("smoke prerequisites", False, repr(e))
        return
    if not torch.npu.is_available():
        report("smoke prerequisites", False, "no NPU device available")
        return

    @triton.jit
    def _k_atomic_plain(ptr):
        tl.atomic_add(ptr, 1)

    @triton.jit
    def _k_atomic_sem(ptr):
        tl.atomic_add(ptr, 1, sem="relaxed", scope="gpu")

    @triton.jit
    def _k_barrier(ptr):
        tl.debug_barrier()
        tl.store(ptr, 1)

    # NOTE: the jit kernel below resolves free names through module globals,
    # so the dispatch primitives must be imported with `global`, not as locals.
    global ld, st, atomic_add
    try:
        from triton_dist.language.extra.language_extra import atomic_add, ld, st
    except Exception as e:  # noqa: BLE001
        report("smoke: dispatch layer import (is triton_dist installed?)", False, repr(e))
        return

    @triton.jit
    def _k_dispatch(ptr, out_ptr):
        v = ld(ptr, "gpu", "relaxed")
        st(out_ptr, v, "gpu", "relaxed")
        atomic_add(out_ptr, 1, "gpu", "relaxed")

    def _run(kernel, name, *args):
        try:
            kernel[(1, 1, 1)](*args)
            torch.npu.synchronize()
            report(f"smoke: {name}", True)
        except Exception as e:  # noqa: BLE001
            report(f"smoke: {name}", False, f"{type(e).__name__}: {str(e)[:300]}")

    buf = torch.zeros(4, dtype=torch.int32).npu()
    out = torch.zeros(4, dtype=torch.int32).npu()
    _run(_k_atomic_plain, "tl.atomic_add(ptr, 1)", buf)
    _run(_k_atomic_sem,
         'tl.atomic_add(ptr, 1, sem="relaxed", scope="gpu")  <-- if PASS, set _MEM_ORDER_PASSTHROUGH=True', buf)
    _run(_k_barrier, "tl.debug_barrier()", buf)
    _run(_k_dispatch, "dispatch layer ld/st/atomic_add (new ascend language_extra)", buf, out)


def main():
    probe_environment()
    probe_symbols()
    probe_signatures()
    probe_smoke_compile()

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    failed = [r for r in RESULTS if not r[1]]
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    print()
    if failed:
        print(f"{len(failed)} check(s) failed -- see details above.")
        print("Notable decisions driven by this probe:")
        print("  * sem/scope smoke FAIL  -> keep _MEM_ORDER_PASSTHROUGH=False (default)")
        print("  * tl.atomic_cas missing -> atomic_cas raises NotImplementedError (already handled)")
        print("  * tl.debug_barrier missing -> __syncthreads falls back to builder.create_barrier (already handled)")
        print("  * sub_vec_id missing    -> tid() must be redesigned for this CANN version")
    else:
        print("All checks passed.")
        print("If the sem/scope smoke test passed, consider setting")
        print("_MEM_ORDER_PASSTHROUGH = True in")
        print("python/triton_dist/language/extra/ascend/language_extra.py")


if __name__ == "__main__":
    main()
