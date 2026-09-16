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
#   2. step 3 uses VALID AscendNPU-IR build flags. The A3 script passes
#      "--bisheng-compile=<dir> --build-shmem-template"; neither option exists
#      in the vendored build-tools/build.sh, which aborts with
#      `Error: Unknown option: --bisheng-compile=<dir>` (verified by running the
#      parser). The documented names are --bisheng-compiler and
#      -t/--build-bishengir-template (see AscendNPU-IR docs, installing_guide.md);
#      the template library is not needed by this flow (it is the only consumer
#      of BISHENG_COMPILER_PATH, guarded by BISHENGIR_BUILD_TEMPLATE=ON), so only
#      the correctly spelled compiler path is passed here.
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
#   PIP_NO_INDEX       [1]  pass --no-index to pip (set to "" if the host has
#                           a reachable PyPI mirror and needs missing deps)
#   RUN_TESTS          [0]
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

for tool in git python3 cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not installed"
done
PIP_BIN=$(command -v pip || command -v pip3 || true)
[[ -n "$PIP_BIN" ]] || die "neither pip nor pip3 found"
CMAKE_VER=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
version_at_least "$CMAKE_VER" 3.28 \
    || die "cmake >= 3.28 required (AscendNPU-IR), got $CMAKE_VER"
NINJA_VER=$(ninja --version 2>/dev/null || echo 0)
version_at_least "$NINJA_VER" 1.12 \
    || die "ninja >= 1.12 required (AscendNPU-IR), got $NINJA_VER"

python3 -c "import torch, torch_npu" 2>/dev/null \
    || die "torch/torch_npu not importable (CANN 9.1.0 pairs with torch_npu==2.7.1.post8)"
python3 -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__, '| npu:', torch.npu.is_available())"
python3 -c "import pybind11" 2>/dev/null \
    || die "python pybind11 not importable (needed by setup.py under --no-build-isolation; pip install pybind11 or set PYBIND11_SYSPATH)"

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

vendored_missing=0
for f in \
    3rdparty/llvm-project/llvm/CMakeLists.txt \
    3rdparty/llvm-project/mlir/CMakeLists.txt \
    3rdparty/llvm-project/lld/CMakeLists.txt \
    3rdparty/triton-ascend/cmake/llvm-hash.txt \
    3rdparty/triton-ascend/third_party/ascend/backend/compiler.py \
    3rdparty/triton-ascend/third_party/ascend/AscendNPU-IR/CMakeLists.txt \
    3rdparty/AscendNPU-IR/build-tools/build.sh \
    3rdparty/AscendNPU-IR/third-party/llvm-project/llvm/CMakeLists.txt \
    3rdparty/shmem/scripts/build.sh \
    3rdparty/nlohmann-json/include/nlohmann/json.hpp \
    3rdparty/triton-ascend.patch \
    3rdparty/AscendNPU-IR.patch
do
    if [[ -e "$REPO_DIR/$f" ]]; then
        echo "[OK] $f"
    else
        echo "[MISSING] $f" >&2
        vendored_missing=1
    fi
done
[[ "$vendored_missing" == "0" ]] || die "vendored dependencies incomplete -- run scripts/vendor_deps.sh on a networked Linux machine and commit/copy the repo over (see 3rdparty/VENDORED.md)"

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

# setup.py git-applies these two patches during step 4 and HARD-FAILS if they
# don't apply to a clean tree (dirty trees are skipped). The vendored trees are
# plain directories of the outer repo now, so setup.py's `git diff-index
# --quiet HEAD` check spans the WHOLE repo -- mirror it exactly here (cheap)
# instead of after the ~1h LLVM build.
preflight_patch() {  # $1=target dir, $2=patch file, $3=label
    local out
    if git -C "$1" diff-index --quiet HEAD -- 2>/dev/null; then
        if out=$(git -C "$1" apply --check "$2" 2>&1); then
            echo "[OK] $3 applies cleanly"
        else
            echo "[ERROR] $3 does NOT apply to the vendored tree:" >&2
            echo "$out" >&2
            die "$3 incompatible with 3rdparty/triton-ascend -- re-run scripts/vendor_deps.sh or refresh 3rdparty/*.patch"
        fi
    elif git -C "$1" apply --reverse --check "$2" 2>/dev/null; then
        echo "[OK] $3 already applied (re-run with patched tree)"
    else
        echo "[WARN] $3 neither applies nor is already applied -- repo has unrelated modifications; setup.py will skip patching it"
    fi
}
preflight_patch "$TA_DIR" "$REPO_DIR/3rdparty/triton-ascend.patch" "3rdparty/triton-ascend.patch"
preflight_patch "$INNER_NPU_DIR" "$REPO_DIR/3rdparty/AscendNPU-IR.patch" "3rdparty/AscendNPU-IR.patch"

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
    cmake -S "$REPO_DIR/3rdparty/llvm-project/llvm" -B "$LLVM_BUILD_DIR" -G Ninja \
        -DCMAKE_C_COMPILER="$CLANG_BIN" \
        -DCMAKE_CXX_COMPILER="$CLANGXX_BIN" \
        -DCMAKE_LINKER="$LD_BIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        # Pin the LLVM version explicitly: a stale CMakeCache.txt can otherwise
        # carry these as DEFINED-but-empty, which makes project(VERSION ..)
        # fail with `VERSION ".." format invalid` at llvm/CMakeLists.txt:46.
        # Values match the vendored tree's own cmake/Modules/LLVMVersion.cmake
        # defaults (fad3272); command-line -D always wins over the cache.
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
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_npu_ir" \
      && "$(cat "$WORK_ROOT/.stamp_npu_ir")" == "$NPU_SHA" \
      && -d "$NPU_IR_DIR/build/bin" ]]; then
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
    # Flag names are exactly those accepted by the vendored build.sh (run
    # `bash build-tools/build.sh --help`): --bisheng-compile / --build-shmem-template
    # as used by scripts/build_ascend_a3.sh are NOT valid and abort the build.
    bash ./build-tools/build.sh -o ./build -j "$JOBS" --build-type Release \
        --bisheng-compiler="$ASCEND_HOME_PATH/bin" \
        --add-cmake-options="-DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF"
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
      && "$(cat "$WORK_ROOT/.stamp_triton_dist")" == "$DIST_FINGERPRINT" ]]; then
    echo "Triton-distributed already installed for this exact source combo, skipping. FORCE=1 to rebuild."
    echo "(editable install: pure-Python changes take effect without rebuilding)"
else
    # TRITON_OFFLINE_BUILD=1: setup.py skips ALL downloads (nvidia ptxas/cudart/
    # cupti stubs, prebuilt-LLVM tarballs) and forces TRITON_BUILD_UT=OFF;
    # JSON_SYSPATH satisfies its offline guard; LLVM_SYSPATH points at step 2.
    # NOTE: pip replaces any preinstalled triton/triton-ascend wheel from the
    # Docker image; setup.py applies 3rdparty/*.patch (preflighted in step 1).
    LLVM_SYSPATH="$LLVM_INSTALL_PREFIX" \
    TRITON_BUILD_WITH_CLANG_LLD=ON \
    TRITON_BUILD_PROTON=OFF \
    TRITON_BUILD_LITTLE_KERNEL=OFF \
    TRITON_USE_ASCEND=ON \
    TRITON_OFFLINE_BUILD=1 \
    JSON_SYSPATH="$REPO_DIR/3rdparty/nlohmann-json" \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    "$PIP_BIN" install -e ./python --verbose --no-build-isolation $PIP_INDEX_FLAGS
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
    "$PIP_BIN" install $PIP_INDEX_FLAGS "$whl"
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
