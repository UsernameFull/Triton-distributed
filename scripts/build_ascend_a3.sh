#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# One-shot build script for Triton-distributed (Ascend backend) on an
# Atlas A3 server (Ascend 910B, Ubuntu/openEuler).
#
# It automates the full flow documented in docs/build.md ("To use
# Triton-distributed with the Ascend backend"):
#   step 0  prerequisite checks (npu-smi, CANN, torch_npu, toolchain)
#   step 1  clone repo + submodules (incl. gitcode-hosted shmem)
#   step 2  build the pinned & patched LLVM (fad3272 + fad3272.patch)
#   step 3  build AscendNPU-IR (bisheng)
#   step 4  build & install Triton-distributed (TRITON_USE_ASCEND=ON, editable)
#   step 5  build & install shmem (ACLSHMEM python extension)
#   step 6  verify: probe script + (optionally) ascend tests
#
# Heavy steps are stamp-file guarded, so re-running the script resumes
# instead of rebuilding. FORCE=1 rebuilds everything.
#
# Usage:
#   bash scripts/build_ascend_a3.sh                # full build
#   RUN_TESTS=1 bash scripts/build_ascend_a3.sh    # also run pytest at the end
#   FORCE=1 bash scripts/build_ascend_a3.sh        # ignore stamps, rebuild all
#
# Overridable environment variables (defaults in brackets):
#   WORK_ROOT          [$HOME/ascend-build]        clone & build root
#   REPO_URL           [https://github.com/UsernameFull/Triton-distributed.git]
#   REPO_BRANCH        [feat/ascend-language-extra]
#   REPO_DIR           [$WORK_ROOT/Triton-distributed]  existing clone is reused
#   LLVM_INSTALL_PREFIX[$WORK_ROOT/llvm-install]
#   LLVM_COMMIT        [fad3272286528b8a491085183434c5ad4b59ab92]
#   LLVM_PATCH_URL     [gitcode raw URL of fad3272.patch]
#   LLVM_PATCH_FILE    []                          use a local patch file instead
#   CANN_ENV           [/usr/local/Ascend/ascend-toolkit/set_env.sh]
#   CLANG_BIN/CLANGXX_BIN/LD_BIN  [auto-detected clang-15/lld-15]
#   JOBS               [min(nproc, MemTotal/2GB)]
#   RUN_TESTS          [0]
set -euo pipefail

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
WORK_ROOT="${WORK_ROOT:-$HOME/ascend-build}"
REPO_URL="${REPO_URL:-https://github.com/UsernameFull/Triton-distributed.git}"
REPO_BRANCH="${REPO_BRANCH:-feat/ascend-language-extra}"
REPO_DIR="${REPO_DIR:-$WORK_ROOT/Triton-distributed}"
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-$WORK_ROOT/llvm-install}"
LLVM_COMMIT="${LLVM_COMMIT:-fad3272286528b8a491085183434c5ad4b59ab92}"
LLVM_PATCH_URL="${LLVM_PATCH_URL:-https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch}"
LLVM_PATCH_FILE="${LLVM_PATCH_FILE:-}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_IR_DIR="${NPU_IR_DIR:-$WORK_ROOT/AscendNPU-IR}"
RUN_TESTS="${RUN_TESTS:-0}"
FORCE="${FORCE:-0}"

MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 8)
NPROC=$(nproc 2>/dev/null || echo 4)
JOBS="${JOBS:-$(( MEM_GB / 2 > NPROC ? NPROC : (MEM_GB / 2 < 2 ? 2 : MEM_GB / 2) ))}"

mkdir -p "$WORK_ROOT"
LOG_FILE="$WORK_ROOT/build_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

banner() { echo; echo "============================================================"; echo ">>> $*"; echo "============================================================"; }
stamp_ok() { [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_$1" ]]; }
mark_done() { date > "$WORK_ROOT/.stamp_$1"; }
die() { echo "[ERROR] $*" >&2; echo "[ERROR] see log: $LOG_FILE" >&2; exit 1; }
trap 'echo "[ERROR] failed at line $LINENO (step: ${STEP:-unknown}), log: $LOG_FILE" >&2' ERR

# ---------------------------------------------------------------------------
# step 0: prerequisites
# ---------------------------------------------------------------------------
STEP=0-prechecks
banner "step 0: prerequisite checks"

command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found -- install the Ascend driver, or run on the A3 host"
npu-smi info | head -8 || true

[[ -f "$CANN_ENV" ]] || die "CANN env script not found at $CANN_ENV (set CANN_ENV=...)"
# shellcheck disable=SC1090
source "$CANN_ENV"

for tool in git curl python3 cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not installed"
done
PIP_BIN=$(command -v pip || command -v pip3 || true)
[[ -n "$PIP_BIN" ]] || die "neither pip nor pip3 found"
CMAKE_VER=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
[[ "$(printf '%s\n3.20' "$CMAKE_VER" | sort -V | head -1)" == "3.20" ]] \
    || die "cmake >= 3.20 required, got $CMAKE_VER"

python3 -c "import torch, torch_npu" 2>/dev/null \
    || die "torch/torch_npu not importable -- install the CANN-matching torch_npu wheel first"
python3 -c "import torch; print('torch', torch.__version__, '| npu available:', torch.npu.is_available())"

# clang/lld-15 (LLVM build compilers); allow overrides for openEuler
CLANG_BIN="${CLANG_BIN:-$(command -v clang-15 || command -v clang || echo /usr/bin/clang-15)}"
CLANGXX_BIN="${CLANGXX_BIN:-$(command -v clang++-15 || command -v clang++ || echo /usr/bin/clang++-15)}"
LD_BIN="${LD_BIN:-$(command -v ld.lld-15 || command -v ld.lld || echo /usr/bin/lld-15)}"
if [[ ! -x "$CLANG_BIN" ]]; then
    echo "[WARN] clang-15 not found; trying to install via package manager (needs sudo)..."
    if command -v apt-get >/dev/null; then
        sudo apt-get install -y clang-15 lld-15 ccache || die "apt install clang-15/lld-15 failed"
    elif command -v dnf >/dev/null; then
        sudo dnf install -y clang lld ccache || die "dnf install clang/lld failed"
    else
        die "no clang found and no supported package manager; set CLANG_BIN/CLANGXX_BIN/LD_BIN manually"
    fi
    CLANG_BIN=$(command -v clang-15 || command -v clang)
    CLANGXX_BIN=$(command -v clang++-15 || command -v clang++)
    LD_BIN=$(command -v ld.lld-15 || command -v ld.lld)
fi
echo "compilers: $CLANG_BIN / $CLANGXX_BIN / linker: $LD_BIN"
echo "parallel jobs: $JOBS (mem ${MEM_GB}GB, nproc $NPROC)"
df -h "$WORK_ROOT" | tail -1

# ---------------------------------------------------------------------------
# step 1: clone repo + submodules
# ---------------------------------------------------------------------------
STEP=1-clone
banner "step 1: clone Triton-distributed ($REPO_BRANCH)"

if [[ -d "$REPO_DIR/.git" ]]; then
    echo "reusing existing clone at $REPO_DIR"
    git -C "$REPO_DIR" fetch origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$REPO_BRANCH" \
        || echo "[WARN] pull --ff-only failed (local changes?); continuing with the current checkout"
else
    git clone -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git submodule update --init --depth=1
# 3rdparty/shmem is hosted on gitcode.com and marked `update = none`;
# --checkout overrides it (see docs/build.md).
git submodule update --init --checkout --depth=1 3rdparty/shmem || \
    echo "[WARN] shmem submodule fetch failed -- step 5 will fail; check gitcode.com reachability"
git submodule update --init --checkout --depth=1 3rdparty/triton-ascend
cd 3rdparty/triton-ascend && git submodule update --init --depth=1 && cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# step 2: build pinned & patched LLVM
# ---------------------------------------------------------------------------
STEP=2-llvm
banner "step 2: build LLVM @ ${LLVM_COMMIT:0:8} + fad3272.patch (this takes 30-60 min)"

if stamp_ok llvm && [[ -x "$LLVM_INSTALL_PREFIX/bin/mlir-opt" ]]; then
    echo "LLVM already built (stamp found), skipping. FORCE=1 to rebuild."
else
    LLVM_SRC="$WORK_ROOT/llvm-project"
    [[ -d "$LLVM_SRC/.git" ]] || git clone --no-checkout https://github.com/llvm/llvm-project.git "$LLVM_SRC"
    cd "$LLVM_SRC"
    git fetch --depth=1 origin "$LLVM_COMMIT"
    # -f: reset to a pristine tree on re-runs (a previously applied patch would
    # otherwise make the checkout fail); the patch is re-applied below.
    git checkout -f "$LLVM_COMMIT"

    if [[ -n "$LLVM_PATCH_FILE" ]]; then
        cp "$LLVM_PATCH_FILE" ./fad3272.patch
    else
        curl -fsSL "$LLVM_PATCH_URL" -o ./fad3272.patch || die "failed to download LLVM patch from $LLVM_PATCH_URL"
    fi
    git apply ./fad3272.patch || die "fad3272.patch does not apply cleanly to $LLVM_COMMIT"

    mkdir -p build && cd build
    cmake ../llvm -G Ninja \
        -DCMAKE_C_COMPILER="$CLANG_BIN" \
        -DCMAKE_CXX_COMPILER="$CLANGXX_BIN" \
        -DCMAKE_LINKER="$LD_BIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"
    ninja -j "$JOBS" install
    cp bin/FileCheck "$LLVM_INSTALL_PREFIX/bin/FileCheck"
    cp bin/llvm-lit "$LLVM_INSTALL_PREFIX/bin/llvm-lit"
    cd "$WORK_ROOT"
    mark_done llvm
fi

# ---------------------------------------------------------------------------
# step 3: build AscendNPU-IR
# ---------------------------------------------------------------------------
STEP=3-npu-ir
banner "step 3: build AscendNPU-IR"

if stamp_ok npu_ir && [[ -d "$NPU_IR_DIR/build/bin" ]]; then
    echo "AscendNPU-IR already built (stamp found), skipping. FORCE=1 to rebuild."
else
    : "${ASCEND_HOME_PATH:?CANN set_env.sh did not define ASCEND_HOME_PATH -- check CANN_ENV}"
    [[ -d "$NPU_IR_DIR/.git" ]] || git clone https://gitcode.com/Ascend/AscendNPU-IR.git "$NPU_IR_DIR"
    cd "$NPU_IR_DIR"
    git submodule update --init --depth=1
    mkdir -p build
    ./build-tools/build.sh -o ./build -t --build-type Release --apply-patches \
        --bisheng-compile="$ASCEND_HOME_PATH/bin" --build-shmem-template
    mark_done npu_ir
fi
export PATH="$NPU_IR_DIR/build/bin:$PATH"

# ---------------------------------------------------------------------------
# step 4: build & install Triton-distributed (editable)
# ---------------------------------------------------------------------------
STEP=4-triton-dist
banner "step 4: pip install -e ./python (TRITON_USE_ASCEND=ON)"

cd "$REPO_DIR"
if stamp_ok triton_dist; then
    echo "Triton-distributed already installed (stamp found), skipping. FORCE=1 to rebuild."
    echo "(editable install: pure-Python changes take effect without rebuilding)"
else
    LLVM_SYSPATH="$LLVM_INSTALL_PREFIX" \
    TRITON_BUILD_WITH_CLANG_LLD=ON \
    TRITON_BUILD_PROTON=OFF \
    TRITON_BUILD_LITTLE_KERNEL=OFF \
    TRITON_USE_ASCEND=ON \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    "$PIP_BIN" install -e ./python --verbose --no-build-isolation
    mark_done triton_dist
fi

# ---------------------------------------------------------------------------
# step 5: build & install shmem (ACLSHMEM)
# ---------------------------------------------------------------------------
STEP=5-shmem
banner "step 5: build shmem (ACLSHMEM)"

cd "$REPO_DIR/3rdparty/shmem"
if stamp_ok shmem && python3 -c "import shmem" 2>/dev/null; then
    echo "shmem already installed (stamp found), skipping. FORCE=1 to rebuild."
else
    bash scripts/build.sh -python_extension
    "$PIP_BIN" install dist/shmem-*.whl
    mark_done shmem
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
Build finished. To use the installation in a new shell:

    source $CANN_ENV
    export PATH=$NPU_IR_DIR/build/bin:\$PATH

Smoke test:
    cd $REPO_DIR
    torchrun --nproc-per-node=2 tutorials/ascend/01-ascend-allgather-gemm.py

Log: $LOG_FILE
EOF
