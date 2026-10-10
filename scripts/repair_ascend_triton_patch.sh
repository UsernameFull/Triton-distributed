#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Re-apply (and verify) the vendored patches the Ascend build depends on:
#
#   3rdparty/triton-ascend.patch -> 3rdparty/triton-ascend
#   3rdparty/AscendNPU-IR.patch  -> 3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR
#                                -> 3rdparty/AscendNPU-IR   (outer tree, built in step 3)
#   3rdparty/AscendNPU-IR-hivm-memscope.patch -> the same two npuir trees
#   3rdparty/AscendNPU-IR-distributed.patch   -> the same two npuir trees
#
# The AscendNPU-IR change is split over three patch files on purpose: `git apply`
# is atomic per invocation, so a tree that already carries only one of them
# combined patch makes it fail to apply *entirely* -- and then both `--check`
# and `--reverse --check` fail, so "already applied" and "broken" look the same.
# Applied separately, each half is idempotent on its own.
#
# Why this exists: `git apply` resolves patch paths against the CURRENT working
# directory and silently skips ("Skipped patch '<file>'", exit status still 0)
# every entry that does not live below it. Applied with `cwd=<target>`, our
# repo-root-relative patch paths were all skipped -- and because `--check` and
# `--reverse --check` then both "succeeded", the patches looked applied while
# nothing had changed. Symptom: the build completes, then the first kernel using
# the distributed language dies with
#
#   AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute
#                   'create_symm_at'
#
# This script applies the patches the only way that works -- from the repository
# root, with `git apply --directory=<target>` -- and then verifies the result by
# content instead of by exit status.
#
# triton-ascend's hunks are pure Python (plus one CMakeLists include path), so
# the distributed *frontend* works immediately after this script -- no rebuild.
# The AscendNPU-IR hunks (the HIVM op definition in 3rdparty/AscendNPU-IR.patch
# and the distributed HIVM mem-scope support in
# 3rdparty/AscendNPU-IR-hivm-memscope.patch, plus the distributed HIVM
# lowering / core-type / library-call-name / data-layout support in
# 3rdparty/AscendNPU-IR-distributed.patch) are compiled into
# hivmc/bishengir-compile, so they only take effect at the next build:
#     FORCE=1 bash scripts/build_ascend_a3.sh      # or build_ascend_a2.sh
#
# Usage:
#   bash scripts/repair_ascend_triton_patch.sh
#   RESET=1 bash scripts/repair_ascend_triton_patch.sh   # when a patch no longer
#                                                        # applies, git-restore
#                                                        # that tree first (drops
#                                                        # local edits to it)
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESET="${RESET:-0}"

TA_DIR="3rdparty/triton-ascend"
INNER_NPU_DIR="$TA_DIR/third_party/ascend/AscendNPU-IR"
OUTER_NPU_DIR="3rdparty/AscendNPU-IR"
TA_PATCH="$REPO_DIR/3rdparty/triton-ascend.patch"
NPUIR_PATCH="$REPO_DIR/3rdparty/AscendNPU-IR.patch"
NPUIR_MEMSCOPE_PATCH="$REPO_DIR/3rdparty/AscendNPU-IR-hivm-memscope.patch"
NPUIR_DISTRIBUTED_PATCH="$REPO_DIR/3rdparty/AscendNPU-IR-distributed.patch"
HIVM_TD="bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td"
HIVM_MEMSCOPE="bishengir/lib/Dialect/HIVM/Transforms/InferHIVMMemScope.cpp"
HIVM_DISTUTILS="bishengir/include/bishengir/Dialect/HIVM/Transforms/DistributedTransformUtils.h"
HIVM_CONVERT="bishengir/lib/Dialect/HIVM/Transforms/ConvertToHIVMOp.cpp"

die() { echo "[ERROR] $*" >&2; exit 1; }
note() { echo "[repair] $*"; }

command -v git >/dev/null 2>&1 || die "git not found"
[[ -f "$TA_PATCH" && -f "$NPUIR_PATCH" && -f "$NPUIR_MEMSCOPE_PATCH" \
   && -f "$NPUIR_DISTRIBUTED_PATCH" ]] \
    || die "missing 3rdparty/triton-ascend.patch, 3rdparty/AscendNPU-IR.patch, 3rdparty/AscendNPU-IR-hivm-memscope.patch or 3rdparty/AscendNPU-IR-distributed.patch -- set REPO_DIR to a Triton-distributed checkout"
git -C "$REPO_DIR" rev-parse --git-dir >/dev/null 2>&1 \
    || die "$REPO_DIR is not a git checkout (git apply needs one)"

verify_contains() {  # $1 = file, $2 = literal needle, $3 = what was checked
    grep -qF -- "$2" "$1" 2>/dev/null \
        || die "$3: $1 does not contain '$2' -- the patch did not take effect"
    note "verified $3"
}

apply_patch() {  # $1 = repo-relative target dir, $2 = patch file, $3 = label
    local rel="$1" patch="$2" label="$3"
    if git -C "$REPO_DIR" apply --directory "$rel" --reverse --check "$patch" 2>/dev/null; then
        note "$label already applied ($rel)"
        return 0
    fi
    if [[ "$RESET" == "1" ]]; then
        note "git-restoring $rel (RESET=1)"
        git -C "$REPO_DIR" checkout -- "$rel" 2>/dev/null \
            || die "could not git-restore $rel -- stash/commit your own edits to it first"
    fi
    git -C "$REPO_DIR" apply --directory "$rel" --check "$patch" 2>/dev/null \
        || die "$label does not apply to $rel -- refresh the patch, or re-run with RESET=1 to drop local edits to $rel (see docs/build.md, 'Troubleshooting')"
    git -C "$REPO_DIR" apply --directory "$rel" "$patch"
    note "applied $label to $rel"
}

# --- 1. triton-ascend: the distributed *frontend* ----------------------------
apply_patch "$TA_DIR" "$TA_PATCH" "3rdparty/triton-ascend.patch"
verify_contains "$REPO_DIR/$TA_DIR/python/triton/compiler/code_generator.py" \
    "distributed.ir.DistributedOpBuilder" \
    "the frontend builds a DistributedOpBuilder"
verify_contains "$REPO_DIR/$TA_DIR/python/triton/compiler/compiler.py" \
    "distributed.ir.load_dialects" \
    "the compiler registers the distributed dialects"
verify_contains "$REPO_DIR/$TA_DIR/third_party/ascend/backend/compiler.py" \
    "add_convert_triton_distributed_to_hivm" \
    "the ascend backend runs the distributed->HIVM pass"

# --- 2. AscendNPU-IR: CustomOp attr + distributed HIVM support ----------------
# setup.py patches the INNER copy (used by the root CMake build); step 3 of
# scripts/build_ascend_a{2,3}.sh builds hivmc/bishengir-compile from the OUTER
# copy, so both need everything the patches carry:
#   * the `no_side_effect` unit attr on `hivm.hir.custom`, which
#     lib/Conversion/TritonDistributedToHIVM/ASCEND/DistributedOpToHIVM.cpp sets;
#   * HIVM memory-scope support for the distributed custom ops in
#     InferHIVMMemScope.cpp (backported from AscendNPU-IR's own "Add distributed
#     support"): the pinned AscendNPU-IR predates it, and without it hivmc
#     rejects every distributed kernel with
#       'hivm.hir.custom' op Unsupported user for root alloc op.
#   * the rest of that same upstream commit in
#     3rdparty/AscendNPU-IR-distributed.patch: distributed-aware HIVM lowering
#     (ConvertToHIVMOp), core-type inference, library call names and data
#     layout. Without the lowering part, a masked store into the peer buffer is
#     lowered to an unsupported
#       'hivm.hir.copy' op Unsupported copy from cbuf to gm!
#     Those hunks are compiled into hivmc/bishengir-compile, so they only take
#     effect after a rebuild (unlike the Python frontend below). They are
#     applied one patch at a time, never as a combined file (see the top of
#     this script).
for dir in "$INNER_NPU_DIR" "$OUTER_NPU_DIR"; do
    apply_patch "$dir" "$NPUIR_PATCH" "3rdparty/AscendNPU-IR.patch"
    apply_patch "$dir" "$NPUIR_MEMSCOPE_PATCH" "3rdparty/AscendNPU-IR-hivm-memscope.patch"
    apply_patch "$dir" "$NPUIR_DISTRIBUTED_PATCH" "3rdparty/AscendNPU-IR-distributed.patch"
    verify_contains "$REPO_DIR/$dir/$HIVM_TD" "UnitAttr:\$no_side_effect" \
        "hivm.hir.custom takes no_side_effect ($dir)"
    verify_contains "$REPO_DIR/$dir/$HIVM_MEMSCOPE" \
        "inferAndPropagateMemScopeForDistributed" \
        "HIVM mem-scope pass handles distributed custom ops ($dir)"
    verify_contains "$REPO_DIR/$dir/$HIVM_CONVERT" \
        "isFromDistCallResult" \
        "HIVM lowering handles distributed call results ($dir)"
    verify_contains "$REPO_DIR/$dir/$HIVM_DISTUTILS" \
        "isDistributedTypeCustomOp" \
        "HIVM distributed transform utils are present ($dir)"
done

# --- 3. the *runtime* tree, if `triton` is importable here -------------------
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' || echo "[repair] WARN: 'import triton' failed here -- re-run this script in the environment you run kernels in"
import os
import sys

try:
    import triton
except ImportError as exc:
    sys.exit(f"[repair] triton is not importable by {sys.executable}: {exc}")

pkg = os.path.dirname(os.path.abspath(triton.__file__))
print(f"[repair] triton {triton.__version__} from {triton.__file__}")
missing = []
# setup.py links the ascend backend's sources into <triton>/backends/ascend/
# (NOT .../ascend/backend/), with the nested path kept as a fallback.
for rel_candidates, needle in (
        (("compiler/code_generator.py",), "distributed.ir.DistributedOpBuilder"),
        (("compiler/compiler.py",), "distributed.ir.load_dialects"),
        (("backends/ascend/compiler.py", "backends/ascend/backend/compiler.py"),
         "add_convert_triton_distributed_to_hivm")):
    body = None
    tried = []
    for rel_name in rel_candidates:
        path = os.path.join(pkg, rel_name)
        try:
            with open(path, "rb") as handle:
                body = handle.read().decode("utf-8", "replace")
            break
        except OSError as exc:
            tried.append(f"{rel_name} ({exc})")
    if body is None:
        missing.append("; ".join(tried))
        continue
    if needle not in body:
        missing.append(rel_candidates[0])
if missing:
    sys.exit("[repair] ERROR: the runtime triton tree still lacks the frontend patch:\n"
             + "".join(f"           - {name}\n" for name in missing)
             + f"         (triton package: {pkg})\n"
             "         `import triton` resolves somewhere else; re-install this checkout\n"
             "         (`pip install -e ./python`, see scripts/build_ascend_a3.sh step 4).")
print("[repair] runtime triton tree carries the distributed frontend patch")
try:
    from triton._C.libtriton.distributed import ir as distributed_ir
except Exception as exc:  # noqa: BLE001
    sys.exit(f"[repair] ERROR: cannot import triton._C.libtriton.distributed.ir: {exc!r}")
if not hasattr(distributed_ir, "DistributedOpBuilder"):
    sys.exit("[repair] ERROR: libtriton has no DistributedOpBuilder in "
             "triton._C.libtriton.distributed.ir -- rebuild (FORCE=1) after this repair")
print("[repair] libtriton exposes distributed.ir.DistributedOpBuilder")
PY
fi

cat <<'EOF'

Done. The Python frontend is patched in place, so no rebuild is needed for it:

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PATH=$HOME/ascend-build/AscendNPU-IR/build/bin:$PATH
    torchrun --nproc-per-node=2 --master_port=29501 \
        tutorials/ascend/01-ascend-allgather-gemm.py

If the kernel died inside BiShengHIR
    'hivm.hir.custom' op Unsupported user for root alloc op.
    'hivm.hir.copy' op Unsupported copy from cbuf to gm!
the distributed HIVM hunks (3rdparty/AscendNPU-IR-hivm-memscope.patch and
3rdparty/AscendNPU-IR-distributed.patch) have to be compiled into
hivmc/bishengir-compile first -- and the other C++/tool-side hunks (the proton
CMakeLists include path, the HIVM op definition) need a rebuild too:

    FORCE=1 bash scripts/build_ascend_a3.sh      # or build_ascend_a2.sh
EOF
