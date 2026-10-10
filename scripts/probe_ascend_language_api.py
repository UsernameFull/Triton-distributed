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
import os
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


def probe_libtriton_backends():
    """The imported `triton` must be this checkout's build and carry the
    `distributed` backend.

    `triton_dist/language/distributed_ops.py` does
    `from triton._C.libtriton.distributed import ir`. If a stock
    triton/triton-ascend wheel (whose libtriton.so has no `distributed` plugin)
    shadows the editable install, that import fails with the very confusing
      ModuleNotFoundError: triton._C.libtriton is not a package
    """
    print()
    print("=" * 72)
    print("2. libtriton source + backends (distributed/ascend)")
    print("=" * 72)
    import triton

    print(f"  triton {triton.__version__} from {triton.__file__}")
    c_dir = os.path.join(os.path.dirname(os.path.abspath(triton.__file__)), "_C")
    report("triton/_C directory", os.path.isdir(c_dir), c_dir)
    if not os.path.isdir(c_dir):
        return
    libs = sorted(n for n in os.listdir(c_dir) if n.startswith("libtriton"))
    print(f"  {c_dir}: {libs}")
    report("libtriton.<ext> in triton/_C", any(n.startswith("libtriton.") for n in libs))
    report("libtriton_distributed.<ext> in triton/_C", any(n.startswith("libtriton_distributed.") for n in libs))

    try:
        from triton._C import libtriton
    except Exception as e:  # noqa: BLE001
        report("from triton._C import libtriton", False, repr(e))
        return
    report("from triton._C import libtriton", True, libtriton.__file__)

    for sub in ("ir", "llvm", "ascend", "distributed"):
        report(f"libtriton.{sub} submodule", hasattr(libtriton, sub))

    # This is exactly what triton_dist/language/distributed_ops.py imports.
    try:
        from triton._C.libtriton.distributed import ir  # noqa: F401
    except Exception as e:  # noqa: BLE001
        report("from triton._C.libtriton.distributed import ir", False, repr(e))
        print("  ^ 'triton._C.libtriton is not a package' here means the imported")
        print("    'triton' is NOT this checkout's build: a preinstalled stock")
        print("    triton/triton-ascend wheel shadows the editable install. Fix:")
        print("        pip uninstall -y triton triton-ascend")
        print("        FORCE=1 bash scripts/build_ascend_a3.sh   # or build_ascend_a2.sh")
    else:
        report("from triton._C.libtriton.distributed import ir", True)

    probe_frontend_patch()


def probe_frontend_patch():
    """The vendored ``3rdparty/triton-ascend.patch`` must have been applied to the
    *runtime* triton tree.

    It is what makes the frontend build a ``DistributedOpBuilder`` instead of the
    plain ``ir.builder`` (see ``compiler/code_generator.py``); the distributed
    builder ops -- ``create_symm_at``, ``create_get_rank``, ``create_notify``,
    ... -- live there. When the patch is missing, the build still "succeeds" and
    the first ``dl.symm_at(...)`` in a kernel dies with
      AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute
                      'create_symm_at'
    ``git apply`` no-ops silently (exit 0, "Skipped patch ...") when it is run
    with the wrong working directory, so check the result, not the exit code.
    """
    import triton

    pkg = os.path.dirname(os.path.abspath(triton.__file__))
    # setup.py links the ascend backend's sources into <triton>/backends/ascend/
    # (NOT .../ascend/backend/), with the nested path kept as a fallback.
    checks = (
        (("compiler/code_generator.py",), "distributed.ir.DistributedOpBuilder"),
        (("compiler/compiler.py",), "distributed.ir.load_dialects"),
        (("backends/ascend/compiler.py", "backends/ascend/backend/compiler.py"),
         "add_convert_triton_distributed_to_hivm"),
    )
    for rel_candidates, needle in checks:
        body = None
        tried = []
        for rel in rel_candidates:
            path = os.path.join(pkg, rel)
            try:
                with open(path, "rb") as handle:
                    body = handle.read().decode("utf-8", "replace")
                break
            except OSError as exc:
                tried.append(f"{rel} ({exc})")
        if body is None:
            report(f"{rel_candidates[0]} carries the distributed frontend patch",
                   False, "; ".join(tried))
            continue
        report(f"{rel_candidates[0]} carries the distributed frontend patch",
               needle in body,
               "" if needle in body else
               "3rdparty/triton-ascend.patch is not applied -- fix: "
               "bash scripts/repair_ascend_triton_patch.sh")

    try:
        from triton._C.libtriton.distributed import ir as distributed_ir
    except Exception as exc:  # noqa: BLE001
        report("distributed.ir.DistributedOpBuilder", False, repr(exc))
    else:
        report("distributed.ir.DistributedOpBuilder",
               hasattr(distributed_ir, "DistributedOpBuilder"))


def probe_symbols():
    print()
    print("=" * 72)
    print("3. Symbol availability")
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
    print("4. Signatures (sem/scope support decides _MEM_ORDER_PASSTHROUGH)")
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
    print("5. Kernel smoke compilation on NPU (requires torch_npu + device)")
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
    probe_libtriton_backends()
    probe_symbols()
    probe_signatures()
    probe_smoke_compile()
    probe_npu_ir_patch()

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


def probe_npu_ir_patch():
    """The vendored AscendNPU-IR trees must carry both npuir patches.

    The pinned AscendNPU-IR (triton-ascend's submodule pin, 1b336491) predates
    HIVM's distributed custom-op support, so the backport is split over two
    patch files -- ``3rdparty/AscendNPU-IR.patch`` (the HIVM op definition) and
    ``3rdparty/AscendNPU-IR-hivm-memscope.patch`` (the mem-scope support) --
    applied separately, because a tree that already carries only one half of a
    combined patch makes that combined patch apply *nothing* (``git apply`` is
    atomic per invocation). Together the two patches backport:

      * ``no_side_effect`` on ``hivm.hir.custom`` (HIVMOps.td), set by
        ``lib/Conversion/TritonDistributedToHIVM/ASCEND/DistributedOpToHIVM.cpp``;
      * memory-scope support for the distributed custom ops
        (``InferHIVMMemScope.{h,cpp}``).

    Both are compiled into hivmc/bishengir-compile, so they only take effect
    after a rebuild. Missing them makes BiShengHIR reject every kernel that
    calls an aclshmem helper with

      'hivm.hir.custom' op Unsupported user for root alloc op.
      'func.func' op Failed to propagate memory scope for argument #N
    """
    print()
    print("=" * 72)
    print("6. Vendored AscendNPU-IR patch (C++/hivmc side, needs a rebuild)")
    print("=" * 72)
    repo = os.environ.get("REPO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    built_tree = os.environ.get("NPU_IR_DIR") or os.path.expanduser("~/ascend-build/AscendNPU-IR")
    checks = (
        ("bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td",
         "UnitAttr:$no_side_effect"),
        ("bishengir/lib/Dialect/HIVM/Transforms/InferHIVMMemScope.cpp",
         "inferAndPropagateMemScopeForDistributed"),
    )
    trees = (
        "3rdparty/AscendNPU-IR",
        "3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR",
        built_tree,
    )
    for tree in trees:
        label = tree if os.path.isabs(tree) else tree
        for rel, needle in checks:
            path = os.path.join(tree if os.path.isabs(tree) else os.path.join(repo, tree), rel)
            try:
                with open(path, "rb") as handle:
                    body = handle.read().decode("utf-8", "replace")
            except OSError as exc:
                report(f"{label}: {os.path.basename(rel)} carries the patch", False, repr(exc))
                continue
            report(f"{label}: {os.path.basename(rel)} carries the patch", needle in body,
                   "" if needle in body else
                   "3rdparty/AscendNPU-IR*.patch is not applied -- fix: "
                   "bash scripts/repair_ascend_triton_patch.sh, then rebuild with "
                   "FORCE=1 bash scripts/build_ascend_a3.sh (or build_ascend_a2.sh)")


if __name__ == "__main__":
    main()
