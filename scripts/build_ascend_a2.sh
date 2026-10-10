#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# One-shot OFFLINE build script for Triton-distributed (Ascend backend) on
# Atlas A2 (Atlas 800T A2, Ascend 910B), e.g. inside the official triton-ascend
# Docker image with CANN 9.1.0 preinstalled:
#
#   quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-910b-ubuntu24.04-py3.11
#
# A2 counterpart of scripts/build_ascend_a3.sh. What actually differs:
#
#   item             | A2 (this script)                  | A3
#   -----------------+-----------------------------------+-------------------------------
#   docker image     | ...-torch_npu2.7.1.post8-910b-... | ...-torch_npu2.7.1.post8-a3-...
#   Dockerfile arg   | --build-arg CHIP_TYPE=A2          | --build-arg CHIP_TYPE=A3
#                    | (docker/Dockerfile maps A2->910b) |
#   CANN kernel pkg  | Ascend-cann-910b-ops_<ver>.run    | Ascend-cann-kernels-A3_<ver>.run
#   device SoC       | Ascend910B1/B2/B3/B4              | Ascend910_93xx (910C)
#
# Everything else is SoC-agnostic: the vendored LLVM, AscendNPU-IR/bisheng,
# Triton-distributed and shmem artifacts are the same on both families.
# bishengir-compile picks the SoC at JIT time via `--target=<arch>`, and
# triton-ascend takes <arch> from the device itself (TRITON_ASCEND_ARCH can
# override it; on A2 the valid values are Ascend910B1..Ascend910B4, see
# third_party/ascend/backend/utils.py:570). ACLSHMEM's -soc_type only means
# something for Ascend950, so step 5 is byte-identical to the A3 flow.
#
# Hence this script == the A3 script plus two changes:
#
#   1. step 0 gains a device-family preflight: running it on an A3/A5 host, or
#      inside an A3 container, fails in seconds instead of after the ~1h LLVM
#      build and a confusing JIT/runtime failure. Skip it with
#      EXPECTED_SOC_FAMILY="".
#
#   2. step 3 uses VALID AscendNPU-IR build flags: --bisheng-compiler=<dir>
#      (NOT --bisheng-compile) and -t/--build-bishengir-template (NOT
#      --build-shmem-template). An earlier revision of the A3 script passed the
#      two bogus names and aborted at argument-parse time with
#      `Error: Unknown option: --bisheng-compile=<dir>` (verified by running the
#      parser); the A3 script now uses the same valid flags as this one, checked
#      against the vendored build-tools/build.sh --help. -t is REQUIRED: the
#      template build compiles the device-side meta-op bitcode
#      (build/lib/meta_op.*.bc, host.bc) that CANN's hivmc links into every
#      kernel; without it hivmc fails for every kernel with
#      "Failed to compile BiShengLIR to binary".
#
# All build dependencies are VENDORED in this git repo (3rdparty/), so the
# build host needs NO access to github.com / gitcode.com / PyPI:
#
#   3rdparty/llvm-project    LLVM fad3272, triton-ascend's llvm patch
#                            PRE-APPLIED, trimmed to llvm+mlir+lld (no tests)
#   3rdparty/triton-ascend   pinned bfd8f55, pristine; python/setup.py applies
#                            3rdparty/triton-ascend.patch at build time
#   3rdparty/AscendNPU-IR    pinned 1b33649 + its third-party/llvm-project
#                            (patches pre-applied, trimmed) for the standalone
#                            bisheng tool build
#   3rdparty/shmem           pinned 81c95bad (ACLSHMEM), pristine
#   3rdparty/nlohmann-json   v3.11.3 headers (JSON_SYSPATH for offline setup.py)
#
# See 3rdparty/VENDORED.md; regenerate with scripts/vendor_deps.sh (once, on a
# networked Linux machine, then commit).
#
# Steps:
#   0    prerequisite checks (npu-smi, CANN, torch_npu, A2/910B device family,
#        compilers, cmake>=3.28)
#   1    vendored-tree sanity + setup.py patch-compat preflight
#   2    build LLVM from 3rdparty/llvm-project (out-of-source, tests disabled)
#   3    build AscendNPU-IR (bisheng) from a WORK_ROOT copy of the vendored tree
#   4    build & install Triton-distributed (TRITON_USE_ASCEND=ON,
#        TRITON_OFFLINE_BUILD=1, editable)
#   5    build & install shmem (ACLSHMEM python extension)
#   6    verify: probe script + (optionally) ascend tests
#
# Heavy steps are stamp-file guarded: re-running resumes instead of rebuilding
# (FORCE=1 rebuilds everything). The stamps live in WORK_ROOT and are keyed by
# vendored commit / repo HEAD only, i.e. they are deliberately SHARED with
# scripts/build_ascend_a3.sh: the artifacts are SoC-agnostic, so an A2 host that
# already ran the A3 script reuses the LLVM/bisheng builds instead of spending
# another hour on them. Builds happen under WORK_ROOT; the repo tree stays clean
# except for the patches setup.py applies to 3rdparty/triton-ascend during step 4
# (by design -- the editable install reads those sources at runtime).
#
# Usage (on the A2 host / inside the 910b container, from anywhere):
#   bash scripts/build_ascend_a2.sh                # offline build
#   RUN_TESTS=1 bash scripts/build_ascend_a2.sh    # also run pytest at the end
#   FORCE=1 bash scripts/build_ascend_a2.sh        # ignore stamps, rebuild all
#
# Overridable environment variables (defaults in brackets):
#   WORK_ROOT          [$HOME/ascend-build]     build & install root
#   REPO_DIR           [repo containing this script]
#   LLVM_INSTALL_PREFIX[$WORK_ROOT/llvm-install]
#   NPU_IR_DIR         [$WORK_ROOT/AscendNPU-IR]   build copy of 3rdparty/AscendNPU-IR
#   SHMEM_DIR          [$WORK_ROOT/shmem]          build copy of 3rdparty/shmem
#   CANN_ENV           [/usr/local/Ascend/ascend-toolkit/set_env.sh]
#   EXPECTED_SOC_FAMILY[910b]  substrings of the SoC names the host must report
#                              (space separated); "" disables the preflight
#   TRITON_ASCEND_ARCH [unset] optional JIT target override. Leave unset so
#                              triton-ascend auto-detects the SoC; A2 values are
#                              Ascend910B1, Ascend910B2, Ascend910B3, Ascend910B4
#   CLANG_BIN/CLANGXX_BIN/LD_BIN   [auto-detected clang(-15)/lld]
#   JOBS               [min(nproc, MemTotal/2GB)]
#   PIP_NO_INDEX       [1]  pass --no-index to the pip installs performed by the
#                           script itself (the triton-dist editable + the shmem
#                           wheel, both LOCAL builds, so --no-index is correct;
#                           set "" if those two installs should reach the net)
#   AUTO_INSTALL_DEPS  [1]  auto-install MISSING python build tooling
#                           (setuptools/wheel/pybind11/cmake/ninja/pytest) from
#                           PIP_INDEX_URL; 0 = strict offline (check + die only)
#   PIP_INDEX_URL      [Tsinghua PyPI mirror] index used by AUTO_INSTALL_DEPS
#   AUTO_VENDOR        [1]  when 3rdparty/ is incomplete, run
#                           scripts/vendor_deps.sh automatically (needs network
#                           to github.com/gitcode.com, curl, ~8 GB scratch) and
#                           commit the result, so setup.py's clean-tree patch
#                           step still works; 0 = only report the gaps
#   RUN_TESTS          [0]
#
# Step 4 also removes any preinstalled triton / triton-ascend first, so that
# `import triton` resolves to THIS checkout's editable install (whose
# libtriton.so carries the `distributed` backend), and then verifies it.
# Skipping that removal is what makes a successful build fail at runtime with
#     triton._C.libtriton is not a package
# The same verification covers 3rdparty/triton-ascend.patch: without it the
# frontend builds a plain `ir.builder` and the first `dl.symm_at(...)` dies with
#     AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute
#                     'create_symm_at'
# Repair an existing tree (no rebuild) with scripts/repair_ascend_triton_patch.sh.
# 3rdparty/AscendNPU-IR.patch and 3rdparty/AscendNPU-IR-hivm-memscope.patch are
# checked as well (step 1 + step 3), together with
# 3rdparty/AscendNPU-IR-distributed.patch: the pinned AscendNPU-IR predates
# HIVM's distributed custom-op support, so the patches backport it; without it
# hivmc rejects every kernel that calls an aclshmem helper with
#     'hivm.hir.custom' op Unsupported user for root alloc op.
#     'func.func' op Failed to propagate memory scope for argument #N
#     'hivm.hir.copy' op Unsupported copy from cbuf to gm!
# and those hunks DO need a rebuild (step 3) before they take effect.
set -euo pipefail

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
WORK_ROOT="${WORK_ROOT:-$HOME/ascend-build}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-$WORK_ROOT/llvm-install}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$WORK_ROOT/llvm-build}"
NPU_IR_DIR="${NPU_IR_DIR:-$WORK_ROOT/AscendNPU-IR}"
SHMEM_DIR="${SHMEM_DIR:-$WORK_ROOT/shmem}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
EXPECTED_SOC_FAMILY="${EXPECTED_SOC_FAMILY:-910b}"
RUN_TESTS="${RUN_TESTS:-0}"
FORCE="${FORCE:-0}"
PIP_NO_INDEX="${PIP_NO_INDEX-1}"
if [[ -n "$PIP_NO_INDEX" ]]; then PIP_INDEX_FLAGS="--no-index"; else PIP_INDEX_FLAGS=""; fi
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
AUTO_VENDOR="${AUTO_VENDOR:-1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 8)
NPROC=$(nproc 2>/dev/null || echo 4)
JOBS="${JOBS:-$(( MEM_GB / 2 > NPROC ? NPROC : (MEM_GB / 2 < 2 ? 2 : MEM_GB / 2) ))}"
if [[ $EUID -eq 0 ]]; then SUDO=""; elif command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

mkdir -p "$WORK_ROOT"
LOG_FILE="$WORK_ROOT/build_a2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

banner() { echo; echo "============================================================"; echo ">>> $*"; echo "============================================================"; }
stamp_ok() { [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_$1" ]]; }
mark_done() { date > "$WORK_ROOT/.stamp_$1"; }
die() { echo "[ERROR] $*" >&2; echo "[ERROR] see log: $LOG_FILE" >&2; exit 1; }
vendor_sha() { awk -F'=' '$1=="sha"{print $2}' "$1/.vendor-sha" 2>/dev/null || echo unknown; }
version_at_least() { # $1=version $2=minimum -> 0 if $1 >= $2
    [[ "$(printf '%s\n%s' "$2" "$1" | sort -V | head -1)" == "$2" ]]
}
run_pip() {  # "$@" = pip args; falls back to `python3 -m pip` when no pip binary
    if [[ -n "${PIP_BIN:-}" ]]; then "$PIP_BIN" "$@"; else python3 -m pip "$@"; fi
}
install_py_dep() {  # $1=python import name $2=pip spec -> ensure `python3 -c "import $1"`
    local mod="$1" spec="$2"
    python3 -c "import $mod" >/dev/null 2>&1 && return 0
    [[ "$AUTO_INSTALL_DEPS" == "1" ]] \
        || die "python module '$mod' is missing and AUTO_INSTALL_DEPS=0 -- pip install $spec"
    echo "[deps] installing $spec from $PIP_INDEX_URL ..."
    run_pip install --index-url "$PIP_INDEX_URL" "$spec" \
        || die "failed to install $spec from $PIP_INDEX_URL -- point PIP_INDEX_URL at a reachable index (or preinstall it) and re-run"
    python3 -c "import $mod" >/dev/null 2>&1 \
        || die "$spec installed but '$mod' is still not importable -- pip/python version mismatch?"
}
install_cmd_dep() {  # $1=command on PATH $2=pip spec -> ensure `command -v $1`
    local cmd="$1" spec="$2"
    command -v "$cmd" >/dev/null 2>&1 && return 0
    [[ "$AUTO_INSTALL_DEPS" == "1" ]] \
        || die "$cmd not installed and AUTO_INSTALL_DEPS=0 -- pip install $spec"
    echo "[deps] installing $spec from $PIP_INDEX_URL ..."
    run_pip install --index-url "$PIP_INDEX_URL" "$spec" \
        || die "failed to install $spec from $PIP_INDEX_URL -- point PIP_INDEX_URL at a reachable index and re-run"
    hash -r 2>/dev/null || true
    command -v "$cmd" >/dev/null 2>&1 \
        || die "$cmd still not on PATH after installing $spec -- check PATH / activate the right venv"
}
ensure_cmd_version() {  # $1=cmd $2=min version $3=pip spec -> ensure `$1 --version` >= $2
    local cmd="$1" min="$2" spec="$3" ver
    ver=$("$cmd" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    version_at_least "${ver:-0}" "$min" && return 0
    [[ "$AUTO_INSTALL_DEPS" == "1" ]] \
        || die "$cmd >= $min required, got ${ver:-unknown} (AUTO_INSTALL_DEPS=0)"
    echo "[deps] $cmd ${ver:-unknown} is too old (need >= $min) -- installing $spec from $PIP_INDEX_URL ..."
    run_pip install --index-url "$PIP_INDEX_URL" --upgrade "$spec" \
        || die "failed to install $spec from $PIP_INDEX_URL"
    hash -r 2>/dev/null || true
    ver=$("$cmd" --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    version_at_least "${ver:-0}" "$min" \
        || die "$cmd >= $min still required after installing $spec, got ${ver:-unknown} -- check PATH (the pip bin dir may not be first)"
}
# --- foreign-triton handling (step 4) ---------------------------------------
# `pip install -e ./python` exposes the top-level `triton` package through a
# setuptools meta-path finder that is *appended* to sys.meta_path (see
# setuptools/command/editable_wheel.py: sys.meta_path.append(_EditableFinder)),
# so the ordinary PathFinder -- and therefore everything already installed in
# site-packages -- is consulted FIRST. A preinstalled `triton` / `triton-ascend`
# wheel (the official A3/A2 images ship one; both provide the same top-level
# `triton` package) therefore keeps winning over the editable install, and its
# libtriton.so was built without the `distributed` plugin, so the first
# `import triton_dist` dies with
#     triton._C.libtriton is not a package
# (raised inside python/triton_dist/language/distributed_ops.py, which imports
# `triton._C.libtriton.distributed`). docs/build.md tells users to run
# `pip uninstall triton` first; this script now does that itself.
purge_shadowing_triton() {
    echo "[triton] uninstalling any preinstalled triton/triton-ascend/triton_dist ..."
    run_pip uninstall -y triton triton-ascend triton_dist triton-distributed >/dev/null 2>&1 || true
    # `pip uninstall` can leave files behind (partially tracked installs, conda
    # images); a leftover site-packages/triton/ dir shadows us just as badly.
    # Never touch a symlink (setup.py's add_link_to_distributed creates one) and
    # never touch anything inside this checkout.
    local sp real real_repo
    real_repo="$(cd "$REPO_DIR" && pwd -P)"
    while IFS= read -r sp; do
        [[ -n "$sp" && -d "$sp/triton" && ! -L "$sp/triton" ]] || continue
        real="$(cd "$sp/triton" 2>/dev/null && pwd -P || true)"
        [[ -n "$real" && "$real" == "$real_repo"* ]] && continue
        echo "[triton] removing leftover $sp/triton (would shadow the editable install)"
        rm -rf "$sp/triton"
    done < <(python3 -c 'import site; [print(p) for p in dict.fromkeys([*site.getsitepackages(), site.getusersitepackages()]) if p]' 2>/dev/null || true)
}

# Editable install of THIS checkout through the same interpreter used for every
# later `python3` check. `env` supplies the build-time variables so the helper
# is independent of the caller's shell variables. FORCE_PY3_PIP=1 bypasses any
# PIP_BIN override and goes straight through `python3 -m pip`.
install_editable_triton_dist() {
    local pip_cmd
    if [[ "${FORCE_PY3_PIP:-0}" == "1" || -z "${PIP_BIN:-}" ]]; then
        pip_cmd="python3 -m pip"
    else
        pip_cmd="$PIP_BIN"
    fi
    echo "[triton] editable install via: $pip_cmd"
    # shellcheck disable=SC2086
    env \
        LLVM_SYSPATH="$LLVM_INSTALL_PREFIX" \
        TRITON_BUILD_WITH_CLANG_LLD=ON \
        TRITON_BUILD_PROTON=OFF \
        TRITON_BUILD_LITTLE_KERNEL=OFF \
        TRITON_USE_ASCEND=ON \
        TRITON_OFFLINE_BUILD=1 \
        JSON_SYSPATH="$REPO_DIR/3rdparty/nlohmann-json" \
        TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
        $pip_cmd install -e ./python --verbose --no-build-isolation $PIP_INDEX_FLAGS
}

# Hard-verify that `import triton` resolves to THIS checkout and that the loaded
# libtriton.so really carries the `distributed` (and `ascend`) plugins. Without
# this the shadowing above only surfaces as a confusing error deep inside
# triton_dist -- long after "Successfully installed".
verify_triton_dist_install() {
    TRITON_DIST_REPO_DIR="$REPO_DIR" python3 - <<'PY'
import os
import sys

repo = os.path.realpath(os.environ["TRITON_DIST_REPO_DIR"])
print(f"[triton] python {sys.version.split()[0]} ({sys.executable})")

try:
    import triton  # must already resolve to this checkout
except ImportError as exc:
    # Almost always: pip installed into a different interpreter, or a previous
    # purge removed the (image) triton and the editable install never landed.
    import glob
    import site
    import subprocess

    print(f"[triton] FATAL: import triton failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(f"[triton]   interpreter    : {sys.executable}", file=sys.stderr)
    print(f"[triton]   repo (expected): {repo}", file=sys.stderr)
    for p in sys.path:
        print(f"[triton]   sys.path       : {p}", file=sys.stderr)
    for d in dict.fromkeys([*site.getsitepackages(), site.getusersitepackages()]):
        if not d or not os.path.isdir(d):
            continue
        print(f"[triton]   site-packages  : {d}", file=sys.stderr)
        for pat in ("__editable__*triton*", "__editable___triton*"):
            for f in glob.glob(os.path.join(d, pat)):
                print(f"[triton]     editable finder: {f}", file=sys.stderr)
    for mod in ("triton-dist", "triton", "triton-ascend"):
        try:
            out = subprocess.run([sys.executable, "-m", "pip", "show", mod],
                                 capture_output=True, text=True, timeout=120)
        except Exception as sub_exc:  # noqa: BLE001
            print(f"[triton]   pip show {mod}: {sub_exc}", file=sys.stderr)
            continue
        state = "installed" if out.returncode == 0 else "NOT installed"
        print(f"[triton]   pip show {mod}: {state}", file=sys.stderr)
        body = (out.stdout or out.stderr).strip()
        if body:
            for line in body.splitlines():
                print(f"[triton]     {line}", file=sys.stderr)
    sys.exit(
        "ERROR: 'triton' is not importable by this interpreter at all.\n"
        "       Step 4 ran pip, but the editable install does not provide\n"
        "       'triton' here -- pip targeted a different python, or the install\n"
        "       silently no-op'd. Re-run with FORCE=1 and check the diagnostics\n"
        "       above (interpreter / site-packages / pip show)."
    )

where = os.path.realpath(triton.__file__)
print(f"[triton] triton {triton.__version__} from {triton.__file__}")
if not where.startswith(repo + os.sep):
    sys.exit(
        f"ERROR: 'triton' is imported from {where}\n"
        f"       instead of this checkout ({repo}).\n"
        "       A preinstalled triton/triton-ascend wheel shadows the editable\n"
        "       install; its libtriton.so has no 'distributed' plugin, which shows\n"
        "       up as \"triton._C.libtriton is not a package\".\n"
        "       Fix: pip uninstall -y triton triton-ascend, then FORCE=1 re-run."
    )

from triton._C import libtriton  # loads <checkout>/python/triton/_C/libtriton.so
print(f"[triton] libtriton from {libtriton.__file__}")
missing = [name for name in ("ir", "llvm", "ascend", "distributed") if not hasattr(libtriton, name)]
if missing:
    sys.exit(
        f"ERROR: the loaded libtriton.so has no {', '.join(missing)} submodule(s)\n"
        f"       ({libtriton.__file__}).\n"
        "       It was not built from this checkout: rebuild with TRITON_USE_ASCEND=ON\n"
        "       (this script) and make sure no stock triton/triton-ascend wheel is\n"
        "       installed."
    )

# The exact import that failed at runtime.
from triton._C.libtriton.distributed import ir as _distributed_ir  # noqa: F401
from triton._C.libtriton.distributed import ascend_passes as _ascend_passes  # noqa: F401
print("[triton] triton._C.libtriton.distributed.{ir,ascend_passes} importable")

# The vendored 3rdparty/triton-ascend.patch is what makes the *frontend* build a
# DistributedOpBuilder: code_generator.py swaps the plain `ir.builder` (which has
# no create_symm_at/create_get_rank/...) for `distributed.ir.DistributedOpBuilder`,
# compiler.py registers the distributed dialects, and the ascend backend adds the
# distributed->HIVM pass. If the patch did not land, the build still "succeeds"
# and the first `dl.symm_at(...)` in a kernel dies with
#   AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute
#                   'create_symm_at'
# End of the build is the last cheap place to catch that, so check the *runtime*
# tree here instead of during kernel triage.
if not hasattr(_distributed_ir, "DistributedOpBuilder"):
    sys.exit(
        "ERROR: triton._C.libtriton.distributed.ir has no DistributedOpBuilder --\n"
        "       the loaded libtriton.so is missing the distributed frontend ops."
    )

triton_pkg = os.path.dirname(os.path.abspath(triton.__file__))
# setup.py links the ascend backend's sources into
# <triton>/backends/ascend/ (NOT .../ascend/backend/), so look there first and
# keep the nested path as a fallback for grafted layouts.
frontend_needles = (
    (("compiler/code_generator.py",), "distributed.ir.DistributedOpBuilder"),
    (("compiler/compiler.py",), "distributed.ir.load_dialects"),
    (("backends/ascend/compiler.py", "backends/ascend/backend/compiler.py"),
     "add_convert_triton_distributed_to_hivm"),
)
unpatched = []
for rel_candidates, needle in frontend_needles:
    body = None
    missing = []
    for rel in rel_candidates:
        path = os.path.join(triton_pkg, rel)
        try:
            with open(path, "rb") as handle:
                body = handle.read().decode("utf-8", "replace")
            break
        except OSError as exc:
            missing.append(f"{rel} ({exc})")
    if body is None:
        unpatched.append("; ".join(missing))
        continue
    if needle not in body:
        unpatched.append(rel_candidates[0])
if unpatched:
    sys.exit(
        "ERROR: the runtime triton tree does NOT carry 3rdparty/triton-ascend.patch:\n"
        + "".join(f"         - {name}\n" for name in unpatched)
        + f"         (triton package: {triton_pkg})\n"
        "       Without it the frontend builds a plain `ir.builder`, and the first\n"
        "       dl.symm_at(...) fails with\n"
        "         AttributeError: 'triton._C.libtriton.ir.builder' object has no\n"
        "                         attribute 'create_symm_at'\n"
        "       Fix: bash scripts/repair_ascend_triton_patch.sh   (the Python side\n"
        "       needs no rebuild -- re-run the tutorial straight after)"
    )
print("[triton] frontend patch applied: DistributedOpBuilder + distributed passes")

# Soft check: the full user-facing import chain (torch/torch_npu are heavy).
try:
    import triton_dist.language  # noqa: F401
    print("[triton] import triton_dist.language OK")
except Exception as exc:  # noqa: BLE001
    print(f"[triton] WARN: import triton_dist.language failed: {type(exc).__name__}: {exc}")
    print("[triton]       the triton/libtriton checks above passed -- see step 6 probe")
PY
}

trap 'echo "[ERROR] failed at line $LINENO (step: ${STEP:-unknown}), log: $LOG_FILE" >&2' ERR

# ---------------------------------------------------------------------------
# step 0: prerequisites
# ---------------------------------------------------------------------------
STEP=0-prechecks
banner "step 0: prerequisite checks (Atlas A2 / Ascend 910B)"

[[ -f "$REPO_DIR/python/setup.py" ]] || die "$REPO_DIR is not a Triton-distributed checkout (set REPO_DIR=...)"

command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found -- run inside the Ascend container / on the A2 host"
npu-smi info | head -8 || true

[[ -f "$CANN_ENV" ]] || die "CANN env script not found at $CANN_ENV (set CANN_ENV=...)"
# shellcheck disable=SC1090
source "$CANN_ENV"
echo "CANN: ${ASCEND_TOOLKIT_HOME:-unknown} | ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}"
echo "expected SoC family: ${EXPECTED_SOC_FAMILY:-<check disabled>} (Atlas 800T A2 / 910B)"

for tool in git python3; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not installed"
done

# Always drive pip through `python3 -m pip`. A bare `pip`/`pip3` binary can
# belong to a DIFFERENT interpreter (conda vs system, /usr/bin vs /usr/local),
# in which case step 4 "Successfully installed"s into that other environment
# and the very next `python3` call dies with
#     ModuleNotFoundError: No module named 'triton'
# PIP_BIN is honored only when it provably belongs to the same interpreter.
PY3_REAL="$(python3 -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
[[ -n "$PY3_REAL" ]] || PY3_REAL="$(command -v python3)"
PIP_BIN="${PIP_BIN:-$(command -v pip || command -v pip3 || true)}"
if [[ -n "$PIP_BIN" ]]; then
    _pip_py="$("$PIP_BIN" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    if [[ "$_pip_py" != "$PY3_REAL" ]]; then
        echo "[deps] ignoring PIP_BIN=$PIP_BIN (python: ${_pip_py:-unknown}); using 'python3 -m pip'"
        PIP_BIN=""
    fi
fi
python3 -m pip --version >/dev/null 2>&1 \
    || die "no pip available for $PY3_REAL (tried pip, pip3, python3 -m pip)"
echo "[deps] python3: $PY3_REAL | pip: $(python3 -m pip --version 2>/dev/null)"

# Build tooling: auto-install ONLY what is missing (or too old), from the
# mirror. AUTO_INSTALL_DEPS=1 by default; 0 = strict offline check-and-die.
install_py_dep setuptools setuptools
install_py_dep wheel      wheel
install_py_dep pybind11   pybind11
install_py_dep pytest     pytest
install_cmd_dep cmake cmake
install_cmd_dep ninja ninja
ensure_cmd_version cmake 3.28 "cmake>=3.28"
ensure_cmd_version ninja 1.12 "ninja>=1.12"

# torch/torch_npu are CANN-paired wheels that are NOT on PyPI -- never try to
# auto-install them, just fail loudly with the pairing hint.
python3 -c "import torch, torch_npu" 2>/dev/null \
    || die "torch/torch_npu not importable (CANN 9.1.0 pairs with torch_npu==2.7.1.post8; use the official triton-ascend 910b (A2) image)"
python3 -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__, '| npu:', torch.npu.is_available())"

# --- A2 device-family preflight -------------------------------------------
# Cheap and early: the A2 and A3 docker images differ only in the CANN kernel
# package they carry (Ascend-cann-910b-ops vs Ascend-cann-kernels-A3), and the
# two SoCs are not interchangeable, so a mismatched host/container otherwise
# shows up much later as a JIT/launch failure. Collect every SoC name that can
# be seen (npu-smi + torch_npu) and require EXPECTED_SOC_FAMILY to appear.
if [[ -n "${EXPECTED_SOC_FAMILY:-}" ]]; then
    SOC_LIST="$(
        {
            npu-smi info 2>/dev/null | grep -oE 'Ascend[0-9A-Za-z_]+' || true
            python3 -c 'import torch, torch_npu
try:
    print(torch.npu.get_device_name(0))
except Exception:
    pass' 2>/dev/null || true
        } | sort -u | tr '\n' ' '
    )"
    SOC_LIST="${SOC_LIST% }"
    if [[ -z "$SOC_LIST" ]]; then
        echo "[WARN] no NPU SoC detected (device-less/compile-only host) -- skipping the A2 family check"
    else
        echo "detected SoC(s): $SOC_LIST"
        matched=0
        for fam in $EXPECTED_SOC_FAMILY; do
            if echo "$SOC_LIST" | grep -qi -- "$fam"; then
                matched=1
                echo "[OK] device family matches A2 ($fam)"
                break
            fi
        done
        [[ "$matched" == "1" ]] || die "this host reports [$SOC_LIST] but scripts/build_ascend_a2.sh targets A2 (${EXPECTED_SOC_FAMILY}); use scripts/build_ascend_a3.sh on an A3 host, or set EXPECTED_SOC_FAMILY=\"\" to bypass"
    fi
fi

if [[ -n "${TRITON_ASCEND_ARCH:-}" ]]; then
    case "$TRITON_ASCEND_ARCH" in
        Ascend910B1|Ascend910B2|Ascend910B3|Ascend910B4)
            echo "[OK] TRITON_ASCEND_ARCH=$TRITON_ASCEND_ARCH (A2/910B)" ;;
        *)
            echo "[WARN] TRITON_ASCEND_ARCH=$TRITON_ASCEND_ARCH is not an A2 (910B) arch;" \
                 "A2 values are Ascend910B1..Ascend910B4, A3 values look like Ascend910_93xx -- unset it if unintended" ;;
    esac
else
    echo "[INFO] TRITON_ASCEND_ARCH unset -> triton-ascend reads the SoC from the device at JIT time (recommended)"
fi

# compilers for the LLVM build: prefer clang-15 (docs/build.md), accept any clang
CLANG_BIN="${CLANG_BIN:-$(command -v clang-15 || command -v clang || true)}"
CLANGXX_BIN="${CLANGXX_BIN:-$(command -v clang++-15 || command -v clang++ || true)}"
LD_BIN="${LD_BIN:-$(command -v ld.lld-15 || command -v ld.lld || true)}"
if [[ -z "$CLANG_BIN" || -z "$LD_BIN" ]]; then
    echo "[WARN] clang/lld not found; trying the package manager (needs network)..."
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -y && $SUDO apt-get install -y clang lld ccache || die "apt install clang/lld failed -- install manually or set CLANG_BIN/CLANGXX_BIN/LD_BIN"
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y clang lld ccache || die "dnf install clang/lld failed -- install manually or set CLANG_BIN/CLANGXX_BIN/LD_BIN"
    else
        die "no clang/lld found and no supported package manager; set CLANG_BIN/CLANGXX_BIN/LD_BIN manually"
    fi
    CLANG_BIN=$(command -v clang-15 || command -v clang)
    CLANGXX_BIN=$(command -v clang++-15 || command -v clang++)
    LD_BIN=$(command -v ld.lld-15 || command -v ld.lld)
fi
echo "compilers: $CLANG_BIN / $CLANGXX_BIN / linker: $LD_BIN"
echo "parallel jobs: $JOBS (mem ${MEM_GB}GB, nproc $NPROC)"
df -h "$WORK_ROOT" | tail -1

# ---------------------------------------------------------------------------
# step 1: vendored trees sanity + patch preflight
# ---------------------------------------------------------------------------
STEP=1-vendored
banner "step 1: vendored dependency check ($REPO_DIR/3rdparty)"

check_vendored() {  # non-zero (and logs [MISSING] lines) if 3rdparty/ is incomplete
    local f missing=0
    for f in \
        3rdparty/llvm-project/llvm/CMakeLists.txt \
        3rdparty/llvm-project/third-party/siphash/include/siphash/SipHash.h \
        3rdparty/llvm-project/libunwind/include/mach-o/compact_unwind_encoding.h \
        3rdparty/llvm-project/mlir/CMakeLists.txt \
        3rdparty/llvm-project/lld/CMakeLists.txt \
        3rdparty/triton-ascend/cmake/llvm-hash.txt \
        3rdparty/triton-ascend/third_party/ascend/backend/compiler.py \
        3rdparty/triton-ascend/third_party/amd/include/TritonAMDGPUToLLVM/TargetUtils.h \
        3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR/CMakeLists.txt \
        3rdparty/AscendNPU-IR/build-tools/build.sh \
        3rdparty/AscendNPU-IR/third-party/llvm-project/llvm/CMakeLists.txt \
        3rdparty/shmem/scripts/build.sh \
        3rdparty/nlohmann-json/include/nlohmann/json.hpp \
        3rdparty/triton-ascend.patch \
        3rdparty/AscendNPU-IR.patch \
        3rdparty/AscendNPU-IR-hivm-memscope.patch \
        3rdparty/AscendNPU-IR-distributed.patch
    do
        if [[ -e "$REPO_DIR/$f" ]]; then
            echo "[OK] $f"
        else
            echo "[MISSING] $f" >&2
            missing=1
        fi
    done
    return "$missing"
}

if ! check_vendored; then
    [[ "$AUTO_VENDOR" == "1" ]] \
        || die "vendored dependencies incomplete -- run scripts/vendor_deps.sh on a networked machine and commit the result, or re-run with AUTO_VENDOR=1 (see 3rdparty/VENDORED.md)"
    echo "[vendor] 3rdparty/ is incomplete -- running scripts/vendor_deps.sh"
    echo "[vendor] this needs network (github.com + gitcode.com), curl and ~8 GB free scratch"
    index_dirty_before=0
    git -C "$REPO_DIR" diff-index --cached --quiet HEAD -- 2>/dev/null \
        || index_dirty_before=1
    bash "$REPO_DIR/scripts/vendor_deps.sh" \
        || die "scripts/vendor_deps.sh failed -- verify network access to github.com/gitcode.com, or vendor 3rdparty/ on another machine and commit it (see 3rdparty/VENDORED.md)"
    check_vendored \
        || die "3rdparty/ is still incomplete after vendor_deps.sh -- see the [MISSING] lines above and 3rdparty/VENDORED.md"
    [[ "$index_dirty_before" == "0" ]] \
        || die "vendor_deps.sh staged the vendored trees but the index already had other staged changes; commit 3rdparty/ yourself and re-run (setup.py applies 3rdparty/*.patch only when the whole repo is clean)"
    # setup.py git-applies 3rdparty/*.patch only when the repo is clean, and
    # vendor_deps.sh leaves its work STAGED -- so commit it. The index was clean
    # before vendoring, hence this commit contains only the vendored files.
    echo "[vendor] committing the vendored 3rdparty/ trees (index was clean before)"
    if git -C "$REPO_DIR" config user.email >/dev/null 2>&1 \
       && git -C "$REPO_DIR" config user.name >/dev/null 2>&1; then
        git -C "$REPO_DIR" commit -q -m "vendor: offline Ascend build dependencies (auto; see 3rdparty/VENDORED.md)" \
            || die "failed to commit the vendored trees -- commit 3rdparty/ by hand and re-run"
    else
        git -C "$REPO_DIR" -c user.name="triton-dist offline build" \
            -c user.email="offline-build@localhost" \
            commit -q -m "vendor: offline Ascend build dependencies (auto; see 3rdparty/VENDORED.md)" \
            || die "failed to commit the vendored trees -- commit 3rdparty/ by hand and re-run"
    fi
fi

# The vendored trees must be COMMITTED: python/setup.py decides whether to
# apply 3rdparty/*.patch via a whole-repo `git diff-index --quiet HEAD` check,
# and untracked/staged-but-uncommitted vendored trees make the repo look
# dirty -> setup.py would skip patching and the build would fail cryptically.
git -C "$REPO_DIR" ls-files --error-unmatch 3rdparty/triton-ascend/cmake/llvm-hash.txt >/dev/null 2>&1 \
    || die "3rdparty/triton-ascend is not tracked by git -- commit the vendored trees first (scripts/vendor_deps.sh stages them; see 3rdparty/VENDORED.md)"

TA_DIR="$REPO_DIR/3rdparty/triton-ascend"
INNER_NPU_DIR="$TA_DIR/third_party/ascend/AscendNPU-IR"
LLVM_SHA="$(vendor_sha "$REPO_DIR/3rdparty/llvm-project")"
HASH_SHA="$(tr -d '[:space:]' < "$TA_DIR/cmake/llvm-hash.txt")"
echo "vendored LLVM: $LLVM_SHA (triton-ascend expects $HASH_SHA)"
[[ "$LLVM_SHA" == "$HASH_SHA" || "$LLVM_SHA" == "unknown" ]] \
    || echo "[WARN] vendored LLVM commit differs from triton-ascend's cmake/llvm-hash.txt"

# setup.py git-applies these two patches during step 4 -- from the repository
# ROOT, with `git apply --directory=<target>`. That combination matters:
# `git apply` resolves patch paths against the *current directory* and silently
# skips ("Skipped patch '<file>'", on stdout) every entry that does not live
# below it, while STILL exiting 0 -- so with `cwd=<target>` our repo-root-relative
# patch paths skipped everything, and `--check` *and* `--reverse --check` both
# reported success. That is exactly how both vendored patches used to no-op while
# looking applied; the first symptom is a kernel that reaches the frontend and
# dies with
#   AttributeError: 'triton._C.libtriton.ir.builder' object has no attribute
#                   'create_symm_at'
# Mirror the fixed decision here (cheap) instead of after the ~1h LLVM build.
patch_target_rel() {  # $1 = absolute dir under $REPO_DIR -> path relative to it
    echo "${1#"$REPO_DIR"/}"
}
preflight_patch() {  # $1=target dir, $2=patch file, $3=label
    local out rel
    rel="$(patch_target_rel "$1")"
    if out=$(git -C "$REPO_DIR" apply --directory "$rel" --check "$2" 2>&1); then
        echo "[OK] $3 applies cleanly (target: $rel)"
    elif git -C "$REPO_DIR" apply --directory "$rel" --reverse --check "$2" 2>/dev/null; then
        echo "[OK] $3 already applied (target: $rel)"
    else
        echo "[ERROR] $3 does NOT apply to $rel:" >&2
        echo "$out" >&2
        die "$3 neither applies nor is already applied in $rel -- refresh 3rdparty/*.patch, or repair the tree (see docs/build.md 'Troubleshooting')"
    fi
}
preflight_patch "$TA_DIR" "$REPO_DIR/3rdparty/triton-ascend.patch" "3rdparty/triton-ascend.patch"
# Each patch is preflighted SEPARATELY, on both trees. `git apply` is atomic per
# invocation: a tree that already carries only one half of a combined patch
# (e.g. `no_side_effect` from an older revision of that patch, which is what
# every checkout that ran the pre-split patch looks like) makes the combined
# patch fail to apply *entirely* -- and then BOTH `--check` and
# `--reverse --check` fail, so "already applied" is indistinguishable from
# "broken" and this preflight would `die` before step 3 ever rebuilt hivmc.
NPUIR_PATCHES=(
    "3rdparty/AscendNPU-IR.patch"
    "3rdparty/AscendNPU-IR-hivm-memscope.patch"
    "3rdparty/AscendNPU-IR-distributed.patch"
)
for npuir_tree in "$INNER_NPU_DIR" "$REPO_DIR/3rdparty/AscendNPU-IR"; do
    for npuir_p in "${NPUIR_PATCHES[@]}"; do
        preflight_patch "$npuir_tree" "$REPO_DIR/$npuir_p" "$npuir_p"
    done
done

# setup.py patches only the INNER AscendNPU-IR copy
# (3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR), which is what the
# root CMake build compiles. Step 3 builds hivmc / bishengir-compile from the
# OUTER 3rdparty/AscendNPU-IR, so that tree has to carry the same changes.
# The distributed flow needs three independent things, carried by three patches:
#   * 3rdparty/AscendNPU-IR.patch -- the `no_side_effect` unit attr on
#     `hivm.hir.custom` (HIVMOps.td), set by
#     lib/Conversion/TritonDistributedToHIVM/ASCEND/DistributedOpToHIVM.cpp;
#   * 3rdparty/AscendNPU-IR-hivm-memscope.patch -- HIVM memory-scope support
#     for the distributed custom ops (InferHIVMMemScope.{h,cpp}, backported
#     from AscendNPU-IR's own "Add distributed support"). The pinned
#     AscendNPU-IR predates it, and without it BiShengHIR rejects EVERY kernel
#     that calls an aclshmem helper:
#       'hivm.hir.custom' op Unsupported user for root alloc op.
#       'func.func' op Failed to propagate memory scope for argument #N
#   * 3rdparty/AscendNPU-IR-distributed.patch -- the rest of that same upstream
#     commit: distributed-aware HIVM lowering (ConvertToHIVMOp treats values
#     derived from a distributed custom op as GM, so a masked store into remote
#     memory becomes hivm.store/hivm.load instead of an unsupported
#     `hivm.hir.copy ... cbuf to gm`), plus the distributed core-type / library
#     call-name / data-layout handling (InferCoreType, LibraryFunctionOpInterface,
#     InferHIVMDataLayout, SplitMixKernel, DistributedTransformUtils.h).
# Apply them one at a time -- again from the repo root with --directory:
# `git apply` is atomic per invocation, so a tree that already carries only one
# half of a combined patch would make that combined patch apply *nothing*.
apply_npuir_patch() {  # $1=tree rel, $2=patch rel, $3=probe file rel, $4=needle
    local rel="$1" patch="$2" probe="$3" needle="$4"
    if grep -qF "$needle" "$REPO_DIR/$probe" 2>/dev/null; then
        echo "[OK] $patch already applied to $rel"
        return 0
    fi
    git -C "$REPO_DIR" apply --directory "$rel" "$REPO_DIR/$patch" \
        || die "$patch does not apply to $rel -- refresh the patch (see docs/build.md 'Troubleshooting')"
    grep -qF "$needle" "$REPO_DIR/$probe" \
        || die "$patch did not take effect in $rel ('$needle' missing from $probe)"
    echo "[OK] applied $patch to $rel"
}
apply_outer_npuir_patch() {
    local rel="3rdparty/AscendNPU-IR"
    local td="$rel/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td"
    local memscope="$rel/bishengir/lib/Dialect/HIVM/Transforms/InferHIVMMemScope.cpp"
    local distutils="$rel/bishengir/include/bishengir/Dialect/HIVM/Transforms/DistributedTransformUtils.h"
    apply_npuir_patch "$rel" "3rdparty/AscendNPU-IR.patch" \
        "$td" 'UnitAttr:$no_side_effect'
    apply_npuir_patch "$rel" "3rdparty/AscendNPU-IR-hivm-memscope.patch" \
        "$memscope" 'inferAndPropagateMemScopeForDistributed'
    apply_npuir_patch "$rel" "3rdparty/AscendNPU-IR-distributed.patch" \
        "$distutils" 'isDistributedTypeCustomOp'
}
apply_outer_npuir_patch

# ---------------------------------------------------------------------------
# step 2: build LLVM (vendored, patch pre-applied, out-of-source)
# ---------------------------------------------------------------------------
STEP=2-llvm
banner "step 2: build vendored LLVM ($LLVM_SHA)"

# Stamp records WHICH vendored LLVM was built; re-vendoring a different commit
# invalidates it. SoC-independent, hence shared with the A3 script.
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_llvm" \
      && "$(cat "$WORK_ROOT/.stamp_llvm")" == "$LLVM_SHA" \
      && -x "$LLVM_INSTALL_PREFIX/bin/mlir-opt" ]]; then
    echo "LLVM $LLVM_SHA already built (stamp matches), skipping. FORCE=1 to rebuild."
else
    # The vendored tree carries triton-ascend's llvm patch pre-applied and has
    # its test suites trimmed, hence *_INCLUDE_TESTS=OFF (FileCheck is gated by
    # LLVM_INCLUDE_UTILS and is still built; llvm-lit is not built and nothing
    # in this flow needs it). third-party/benchmark (google/benchmark
    # submodule) was never vendored, hence INCLUDE_BENCHMARKS=OFF.
    # *** Guard: a `#` comment must NEVER appear inside the continued `cmake`
    # *** command below. bash ends the command at the `#` even when the
    # *** previous line continued it, so every following -D... line is then
    # *** executed as a shell command and never reaches cmake -- which is
    # *** exactly what produced the historical failure this fixes:
    # ***   CMake Error ... add_subdirectory given source "unittests"
    # ***   which is not an existing directory
    # *** (-DLLVM_INCLUDE_TESTS=OFF / -DMLIR_INCLUDE_TESTS=OFF never reached
    # *** cmake, so the test-trimmed tree still tried to build its tests).
    #
    # Pin the LLVM version explicitly: a stale CMakeCache.txt can otherwise
    # carry these as DEFINED-but-empty, which makes project(VERSION ..) fail
    # with `VERSION ".." format invalid` at llvm/CMakeLists.txt:46. Values
    # match the vendored tree's own cmake/Modules/LLVMVersion.cmake defaults
    # (fad3272); command-line -D always wins over the cache.
    cmake -S "$REPO_DIR/3rdparty/llvm-project/llvm" -B "$LLVM_BUILD_DIR" -G Ninja \
        -DCMAKE_C_COMPILER="$CLANG_BIN" \
        -DCMAKE_CXX_COMPILER="$CLANGXX_BIN" \
        -DCMAKE_LINKER="$LD_BIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        -DLLVM_VERSION_MAJOR=22 \
        -DLLVM_VERSION_MINOR=0 \
        -DLLVM_VERSION_PATCH=0 \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DMLIR_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"
    ninja -C "$LLVM_BUILD_DIR" -j "$JOBS" install
    cp "$LLVM_BUILD_DIR/bin/FileCheck" "$LLVM_INSTALL_PREFIX/bin/FileCheck"
    echo "$LLVM_SHA" > "$WORK_ROOT/.stamp_llvm"
fi

# ---------------------------------------------------------------------------
# step 3: build AscendNPU-IR (bisheng tools used at JIT runtime)
# ---------------------------------------------------------------------------
STEP=3-npu-ir
banner "step 3: build vendored AscendNPU-IR"

NPU_SHA="$(vendor_sha "$REPO_DIR/3rdparty/AscendNPU-IR")"
# The last clause keeps pre-fix build dirs from being reused: they were created
# without the AscendNPU-IR patches (see apply_outer_npuir_patch), so their
# hivmc carries neither `no_side_effect` on hivm.hir.custom nor the distributed
# mem-scope / distributed-lowering support.
NPUIR_TD="$NPU_IR_DIR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td"
NPUIR_MEMSCOPE="$NPU_IR_DIR/bishengir/lib/Dialect/HIVM/Transforms/InferHIVMMemScope.cpp"
NPUIR_DISTUTILS="$NPU_IR_DIR/bishengir/include/bishengir/Dialect/HIVM/Transforms/DistributedTransformUtils.h"
NPUIR_PATCHED=0
if grep -qF 'UnitAttr:$no_side_effect' "$NPUIR_TD" 2>/dev/null \
   && grep -qF 'inferAndPropagateMemScopeForDistributed' "$NPUIR_MEMSCOPE" 2>/dev/null \
   && grep -qF 'isDistributedTypeCustomOp' "$NPUIR_DISTUTILS" 2>/dev/null; then
    NPUIR_PATCHED=1
fi
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_npu_ir" \
      && "$(cat "$WORK_ROOT/.stamp_npu_ir")" == "$NPU_SHA" \
      && -d "$NPU_IR_DIR/build/bin" \
      && -f "$NPU_IR_DIR/build/lib/meta_op.aic.bc" \
      && "$NPUIR_PATCHED" == "1" ]]; then
    echo "AscendNPU-IR $NPU_SHA already built (stamp matches), skipping. FORCE=1 to rebuild."
else
    : "${ASCEND_HOME_PATH:?CANN set_env.sh did not define ASCEND_HOME_PATH -- check CANN_ENV}"
    # Build from a WORK_ROOT copy so the repo tree stays pristine; build.sh
    # derives paths from `git rev-parse --show-toplevel`, hence the git init.
    rm -rf "$NPU_IR_DIR"; mkdir -p "$NPU_IR_DIR"
    cp -a "$REPO_DIR/3rdparty/AscendNPU-IR/." "$NPU_IR_DIR/"
    git -C "$NPU_IR_DIR" init -q .
    cd "$NPU_IR_DIR"
    # Differences vs AscendNPU-IR's docs/source/*/installing_guide.md recipe,
    # because the vendored tree has npuir's llvm patches PRE-APPLIED and its
    # test suites trimmed:
    #   * no --apply-patches (would `git reset --hard` + re-patch; also needs
    #     third-party/torch-mlir which we intentionally do not vendor)
    #   * no --build-test (check-mlir/check-bishengir lit suites need the
    #     trimmed test trees; plain `ninja` builds a superset of the required
    #     binaries)
    #   * *_INCLUDE_TESTS=OFF + INCLUDE_BENCHMARKS=OFF via --add-cmake-options
    #     (test suites and third-party/benchmark were trimmed from the tree)
    # Flag names verified against 3rdparty/AscendNPU-IR/build-tools/build.sh
    # (--help): the compiler path option is --bisheng-compiler=<dir>, NOT
    # --bisheng-compile, and the template switch is -t/--build-bishengir-template,
    # NOT --build-shmem-template. An unknown option aborts at argument-parse time
    # with `Error: Unknown option: --bisheng-compile=<dir>`.
    # -t (BISHENGIR_BUILD_TEMPLATE=ON) is REQUIRED: it compiles the Template
    # sources with CANN's ccec/llvm-link and links them into the device-side
    # meta-op bitcode at build/lib/{meta_op.aic,meta_op.aiv,meta_op.mix.aic,
    # meta_op.mix.aiv,host}.bc. bishengir-compile attaches those files to every
    # HIVM module and CANN's hivmc links them into the kernel binary; without
    # them hivmc fails for EVERY kernel with
    #     error: Failed to compile BiShengLIR to binary
    # (after bishengir-compile itself reported success), which surfaces as a
    # runtime "kernel smoke compilation" failure, not a build failure.
    bash ./build-tools/build.sh -o ./build -j "$JOBS" --build-type Release \
        -t \
        --bisheng-compiler="$ASCEND_HOME_PATH/bin" \
        --add-cmake-options="-DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF"
    for bc in meta_op.aic.bc meta_op.aiv.bc meta_op.mix.aic.bc meta_op.mix.aiv.bc host.bc; do
        [[ -f "$NPU_IR_DIR/build/lib/$bc" ]] \
            || die "AscendNPU-IR built without build/lib/$bc -- hivmc cannot compile any kernel without the meta-op bitcode (needs CANN's $ASCEND_HOME_PATH/bin/{ccec,llvm-link}; never drop -t)"
    done
    echo "$NPU_SHA" > "$WORK_ROOT/.stamp_npu_ir"
fi
export PATH="$NPU_IR_DIR/build/bin:$PATH"

# ---------------------------------------------------------------------------
# step 4: build & install Triton-distributed (editable, fully offline)
# ---------------------------------------------------------------------------
STEP=4-triton-dist
banner "step 4: pip install -e ./python (TRITON_USE_ASCEND=ON, offline)"

cd "$REPO_DIR"
DIST_FINGERPRINT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_triton_dist" \
      && "$(cat "$WORK_ROOT/.stamp_triton_dist")" == "$DIST_FINGERPRINT" ]] \
      && verify_triton_dist_install >/dev/null 2>&1; then
    echo "Triton-distributed already installed (and importable) for this exact source combo, skipping. FORCE=1 to rebuild."
    echo "(editable install: pure-Python changes take effect without rebuilding)"
else
    # TRITON_OFFLINE_BUILD=1: setup.py skips ALL downloads (nvidia ptxas/cudart/
    # cupti stubs, prebuilt-LLVM tarballs) and forces TRITON_BUILD_UT=OFF;
    # JSON_SYSPATH satisfies its offline guard; LLVM_SYSPATH points at step 2.
    # A preinstalled triton/triton-ascend wheel has to go first: the editable
    # install only *appends* a meta-path finder, so site-packages would keep
    # winning and its libtriton.so (no `distributed` plugin) breaks triton_dist
    # at runtime with "triton._C.libtriton is not a package".
    purge_shadowing_triton
    # setup.py applies 3rdparty/*.patch (preflighted in step 1).
    install_editable_triton_dist \
        || die "editable install of ./python failed (see the pip output above)"
    # 'Successfully installed' from pip is not enough: if pip targeted another
    # interpreter, `import triton` here is a bare ModuleNotFoundError. Retry
    # once, explicitly through python3, before the detailed verify below.
    if ! python3 -c 'import triton' >/dev/null 2>&1; then
        echo "[triton] 'import triton' failed right after the install -- retrying via 'python3 -m pip'"
        FORCE_PY3_PIP=1 install_editable_triton_dist \
            || die "editable reinstall via 'python3 -m pip' failed (see the pip output above)"
    fi
    verify_triton_dist_install \
        || die "triton does not resolve to this checkout, or its libtriton.so lacks the 'distributed'/'ascend' plugins (see the ERROR above) -- fix the environment, then re-run with FORCE=1"
    echo "$DIST_FINGERPRINT" > "$WORK_ROOT/.stamp_triton_dist"
fi

# ---------------------------------------------------------------------------
# step 5: build & install shmem (ACLSHMEM)
# ---------------------------------------------------------------------------
STEP=5-shmem
banner "step 5: build vendored shmem (ACLSHMEM)"

# Identical to the A3 flow: ACLSHMEM's -soc_type / -rdma_backend only apply to
# Ascend950, so the 910B (A2) build takes no extra option here.
SHMEM_SHA="$(vendor_sha "$REPO_DIR/3rdparty/shmem")"
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_shmem" \
      && "$(cat "$WORK_ROOT/.stamp_shmem")" == "$SHMEM_SHA" ]] \
      && python3 -c "import shmem" 2>/dev/null; then
    echo "shmem $SHMEM_SHA already installed (stamp matches), skipping. FORCE=1 to rebuild."
else
    # Build from a WORK_ROOT copy: keeps build artifacts out of the repo tree
    # (setup.py's whole-repo dirty check). `-python_extension` performs no
    # downloads (catlass etc. are only fetched by other flags).
    rm -rf "$SHMEM_DIR"; mkdir -p "$SHMEM_DIR"
    cp -a "$REPO_DIR/3rdparty/shmem/." "$SHMEM_DIR/"
    cd "$SHMEM_DIR"
    bash scripts/build.sh -python_extension
    whl=$(ls -t dist/shmem-*.whl 2>/dev/null | head -1 || true)
    [[ -n "$whl" ]] || die "shmem wheel not found in $SHMEM_DIR/dist"
    run_pip install $PIP_INDEX_FLAGS "$whl"
    echo "$SHMEM_SHA" > "$WORK_ROOT/.stamp_shmem"
fi

# ---------------------------------------------------------------------------
# step 6: verify
# ---------------------------------------------------------------------------
STEP=6-verify
banner "step 6: verification"

cd "$REPO_DIR"
# shellcheck disable=SC1090
source "$CANN_ENV"
export PATH="$NPU_IR_DIR/build/bin:$PATH"

echo "--- devices ---"
npu-smi info | head -8 || true

echo "--- CANN API probe ---"
python3 scripts/probe_ascend_language_api.py || echo "[WARN] probe reported failures -- paste the output back for triage"

if [[ "$RUN_TESTS" == "1" ]]; then
    echo "--- ascend tests (new language_extra) ---"
    python3 -m pytest python/triton_dist/test/ascend/test_language_extra.py -v -m dist || true
    echo "--- ascend tests (regression) ---"
    python3 -m pytest python/triton_dist/test/ascend/ -m dist || true
fi

banner "DONE"
cat <<EOF
Build finished (fully offline from vendored 3rdparty/ sources).
Target platform: Atlas A2 / Ascend 910B (image ...-post8-910b-ubuntu24.04-py3.11).

To use the installation in a new shell:

    source $CANN_ENV
    export PATH=$NPU_IR_DIR/build/bin:\$PATH
    # optional, only if the SoC cannot be auto-detected (e.g. compile-only):
    # export TRITON_ASCEND_ARCH=Ascend910B1   # A2: Ascend910B1..Ascend910B4

Smoke test:
    cd $REPO_DIR
    torchrun --nproc-per-node=2 tutorials/ascend/01-ascend-allgather-gemm.py

Log: $LOG_FILE
EOF
