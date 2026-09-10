#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# One-shot build script for Triton-distributed (Ascend backend) on Atlas A3
# (Ascend 910B).  Primary target: the official triton-ascend Docker images
# with CANN 9.1.0 preinstalled, e.g.
#
#   quay.io/ascend/triton:3.2.2-cann9.1.0-torch_npu2.7.1.post8-a3-ubuntu24.04-py3.11
#
# In that mode the script checks out the LATEST triton-ascend (main), reads
# the LLVM commit from its cmake/llvm-hash.txt and applies the matching
# in-repo patch (third_party/ascend/patch/llvm_patch_<hash>.patch) -- the
# combination officially paired with CANN 9.1.0.
#
# It also works on a bare-metal host with CANN 8.x: set
# TRITON_ASCEND_REF=pinned to keep the submodule commit pinned by
# Triton-distributed (LLVM fad3272 + fad3272.patch, per docs/build.md).
#
# Steps:
#   0    prerequisite checks (npu-smi, CANN, torch_npu, compilers, cmake)
#   1    clone repo + submodules (incl. gitcode-hosted shmem)
#   1.5  checkout requested triton-ascend ref + patch-compat preflight
#   2    build LLVM (commit & patch resolved from triton-ascend)
#   3    build AscendNPU-IR (bisheng)
#   4    build & install Triton-distributed (TRITON_USE_ASCEND=ON, editable)
#   5    build & install shmem (ACLSHMEM python extension)
#   6    verify: probe script + (optionally) ascend tests
#
# Heavy steps are stamp-file guarded: re-running resumes instead of
# rebuilding (FORCE=1 rebuilds everything).
#
# Usage (inside the CANN 9.1.0 container, or on the host):
#   bash build_ascend_a3.sh                     # latest triton-ascend (CANN 9.1.0)
#   RUN_TESTS=1 bash build_ascend_a3.sh         # also run pytest at the end
#   TRITON_ASCEND_REF=pinned bash build_ascend_a3.sh   # legacy CANN 8.5.0 combo
#   FORCE=1 bash build_ascend_a3.sh             # ignore stamps, rebuild all
#
# Overridable environment variables (defaults in brackets):
#   WORK_ROOT          [$HOME/ascend-build]     clone & build root
#   REPO_URL           [https://github.com/UsernameFull/Triton-distributed.git]
#   REPO_BRANCH        [feat/ascend-language-extra]
#   REPO_DIR           [$WORK_ROOT/Triton-distributed]  existing clone is reused
#   TRITON_ASCEND_REF  [main]                   "pinned" keeps the submodule commit
#   LLVM_INSTALL_PREFIX[$WORK_ROOT/llvm-install]
#   LLVM_COMMIT        [auto: 3rdparty/triton-ascend/cmake/llvm-hash.txt]
#   LLVM_PATCH_FILE    [auto: in-repo llvm_patch_<hash>.patch, else LLVM_PATCH_URL]
#   LLVM_PATCH_URL     [gitcode raw URL of fad3272.patch, legacy fallback]
#   CANN_ENV           [/usr/local/Ascend/ascend-toolkit/set_env.sh]
#   CLANG_BIN/CLANGXX_BIN/LD_BIN   [auto-detected clang(-15)/lld]
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
TRITON_ASCEND_REF="${TRITON_ASCEND_REF:-main}"
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-$WORK_ROOT/llvm-install}"
LLVM_COMMIT="${LLVM_COMMIT:-}"   # empty = read from triton-ascend cmake/llvm-hash.txt
LLVM_PATCH_FILE="${LLVM_PATCH_FILE:-}"
LLVM_PATCH_URL="${LLVM_PATCH_URL:-https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_IR_DIR="${NPU_IR_DIR:-$WORK_ROOT/AscendNPU-IR}"
RUN_TESTS="${RUN_TESTS:-0}"
FORCE="${FORCE:-0}"

MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 8)
NPROC=$(nproc 2>/dev/null || echo 4)
JOBS="${JOBS:-$(( MEM_GB / 2 > NPROC ? NPROC : (MEM_GB / 2 < 2 ? 2 : MEM_GB / 2) ))}"
if [[ $EUID -eq 0 ]]; then SUDO=""; elif command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO=""; fi

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

command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found -- run inside the Ascend container / on the A3 host"
npu-smi info | head -8 || true

[[ -f "$CANN_ENV" ]] || die "CANN env script not found at $CANN_ENV (set CANN_ENV=...)"
# shellcheck disable=SC1090
source "$CANN_ENV"
echo "CANN: ${ASCEND_TOOLKIT_HOME:-unknown} | ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset}"

for tool in git curl python3 cmake ninja; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not installed"
done
PIP_BIN=$(command -v pip || command -v pip3 || true)
[[ -n "$PIP_BIN" ]] || die "neither pip nor pip3 found"
CMAKE_VER=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
[[ "$(printf '%s\n3.20' "$CMAKE_VER" | sort -V | head -1)" == "3.20" ]] \
    || die "cmake >= 3.20 required, got $CMAKE_VER"

python3 -c "import torch, torch_npu" 2>/dev/null \
    || die "torch/torch_npu not importable (CANN 9.1.0 pairs with torch_npu==2.7.1.post8)"
python3 -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__, '| npu:', torch.npu.is_available())"

# compilers for the LLVM build: prefer clang-15 (docs/build.md), accept any
# clang, fall back to installing the distro default (clang-18 on ubuntu 24.04).
CLANG_BIN="${CLANG_BIN:-$(command -v clang-15 || command -v clang || true)}"
CLANGXX_BIN="${CLANGXX_BIN:-$(command -v clang++-15 || command -v clang++ || true)}"
LD_BIN="${LD_BIN:-$(command -v ld.lld-15 || command -v ld.lld || true)}"
if [[ -z "$CLANG_BIN" || -z "$LD_BIN" ]]; then
    echo "[WARN] clang/lld not found; installing via package manager..."
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -y && $SUDO apt-get install -y clang lld ccache || die "apt install clang/lld failed"
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y clang lld ccache || die "dnf install clang/lld failed"
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
    die "shmem submodule fetch failed -- check gitcode.com reachability"
git submodule update --init --checkout --depth=1 3rdparty/triton-ascend

# ---------------------------------------------------------------------------
# step 1.5: select triton-ascend version + patch-compat preflight
# ---------------------------------------------------------------------------
STEP=1.5-triton-ascend
banner "step 1.5: triton-ascend @ ${TRITON_ASCEND_REF} + Triton-distributed patch preflight"

TA_DIR="$REPO_DIR/3rdparty/triton-ascend"
if [[ "$TRITON_ASCEND_REF" != "pinned" ]]; then
    cd "$TA_DIR"
    git fetch --depth=1 origin "$TRITON_ASCEND_REF"
    if [[ "$(git rev-parse HEAD)" == "$(git rev-parse FETCH_HEAD)" ]]; then
        # Re-run: keep the working tree as-is -- it may carry the patches that
        # setup.py applied during a previous step 4 (editable installs read
        # these sources at runtime; `checkout -f` would wipe them).
        echo "triton-ascend already at $TRITON_ASCEND_REF tip ($(git rev-parse --short HEAD)); keeping working tree"
    else
        git checkout -f FETCH_HEAD
        echo "triton-ascend checked out at: $(git rev-parse HEAD) ($(git log -1 --format=%cs))"
    fi
    # inner submodule: third_party/ascend/AscendNPU-IR (gitcode)
    git submodule sync --recursive >/dev/null
    git submodule update --init --recursive --depth=1 || \
        die "failed to init triton-ascend inner submodules (AscendNPU-IR on gitcode.com)"
    cd "$REPO_DIR"
else
    echo "keeping submodule-pinned triton-ascend: $(git -C "$TA_DIR" rev-parse HEAD)"
fi

# setup.py git-applies these two patches during step 4 and HARD-FAILS if they
# don't apply to a clean tree (dirty trees are skipped by setup.py).  Check now
# (cheap) instead of after the ~1h LLVM build.
preflight_patch() {  # $1=target dir, $2=patch file, $3=label
    local out
    if git -C "$1" diff-index --quiet HEAD -- 2>/dev/null; then
        if out=$(git -C "$1" apply --check "$2" 2>&1); then
            echo "[OK] $3 applies cleanly"
        else
            echo "[ERROR] $3 does NOT apply to this triton-ascend checkout:" >&2
            echo "$out" >&2
            die "$3 incompatible with TRITON_ASCEND_REF=$TRITON_ASCEND_REF -- use TRITON_ASCEND_REF=pinned, or update 3rdparty/*.patch for the new triton-ascend"
        fi
    elif git -C "$1" apply --reverse --check "$2" 2>/dev/null; then
        echo "[OK] $3 already applied (re-run with patched tree)"
    else
        echo "[WARN] $3 neither applies nor is already applied -- tree has unrelated modifications; setup.py will skip patching it"
    fi
}
preflight_patch "$TA_DIR" "$REPO_DIR/3rdparty/triton-ascend.patch" "3rdparty/triton-ascend.patch"
if [[ -d "$TA_DIR/third_party/ascend/AscendNPU-IR/.git" || -f "$TA_DIR/third_party/ascend/AscendNPU-IR/CMakeLists.txt" ]]; then
    preflight_patch "$TA_DIR/third_party/ascend/AscendNPU-IR" "$REPO_DIR/3rdparty/AscendNPU-IR.patch" "3rdparty/AscendNPU-IR.patch"
else
    echo "[WARN] inner AscendNPU-IR submodule not populated -- skipping its patch preflight"
fi

# ---------------------------------------------------------------------------
# step 2: build LLVM (commit & patch resolved from the triton-ascend checkout)
# ---------------------------------------------------------------------------
STEP=2-llvm
banner "step 2: resolve & build LLVM"

if [[ -z "$LLVM_COMMIT" ]]; then
    HASH_FILE="$TA_DIR/cmake/llvm-hash.txt"
    [[ -f "$HASH_FILE" ]] || die "no cmake/llvm-hash.txt in triton-ascend and LLVM_COMMIT not set"
    LLVM_COMMIT=$(tr -d '[:space:]' < "$HASH_FILE")
fi
echo "LLVM commit (from triton-ascend): $LLVM_COMMIT"

# Resolve the LLVM patch: explicit file > in-repo llvm_patch_<short-hash>.patch
# > single llvm_patch_*.patch in repo > download LLVM_PATCH_URL (legacy).
if [[ -z "$LLVM_PATCH_FILE" ]]; then
    TA_PATCH_DIR="$TA_DIR/third_party/ascend/patch"
    if [[ -f "$TA_PATCH_DIR/llvm_patch_${LLVM_COMMIT:0:7}.patch" ]]; then
        LLVM_PATCH_FILE="$TA_PATCH_DIR/llvm_patch_${LLVM_COMMIT:0:7}.patch"
    else
        shopt -s nullglob
        candidates=("$TA_PATCH_DIR"/llvm_patch_*.patch)
        shopt -u nullglob
        if [[ ${#candidates[@]} -eq 1 ]]; then
            LLVM_PATCH_FILE="${candidates[0]}"
            echo "[WARN] no patch matching ${LLVM_COMMIT:0:7}; using the only one found: $LLVM_PATCH_FILE"
        elif [[ ${#candidates[@]} -gt 1 ]]; then
            die "multiple LLVM patches in $TA_PATCH_DIR (${candidates[*]##*/}) -- set LLVM_PATCH_FILE explicitly"
        else
            echo "[INFO] no in-repo LLVM patch (older triton-ascend); downloading $LLVM_PATCH_URL"
            LLVM_PATCH_FILE="$WORK_ROOT/fad3272.patch"
            curl -fsSL "$LLVM_PATCH_URL" -o "$LLVM_PATCH_FILE" || die "failed to download LLVM patch"
        fi
    fi
fi
echo "LLVM patch: $LLVM_PATCH_FILE"

# The stamp records WHICH LLVM commit was built: switching triton-ascend refs
# changes the required LLVM hash and must invalidate the stamp.
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_llvm" \
      && "$(tr -d '[:space:]' < "$WORK_ROOT/.stamp_llvm")" == "$LLVM_COMMIT" \
      && -x "$LLVM_INSTALL_PREFIX/bin/mlir-opt" ]]; then
    echo "LLVM $LLVM_COMMIT already built (stamp matches), skipping. FORCE=1 to rebuild."
else
    LLVM_SRC="$WORK_ROOT/llvm-project"
    [[ -d "$LLVM_SRC/.git" ]] || git clone --no-checkout https://github.com/llvm/llvm-project.git "$LLVM_SRC"
    cd "$LLVM_SRC"
    git fetch --depth=1 origin "$LLVM_COMMIT"
    # -f: reset to a pristine tree on re-runs (a previously applied patch would
    # otherwise make the checkout fail); the patch is re-applied below.
    git checkout -f "$LLVM_COMMIT"
    git apply "$LLVM_PATCH_FILE" || die "LLVM patch does not apply cleanly to $LLVM_COMMIT"

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
    echo "$LLVM_COMMIT" > "$WORK_ROOT/.stamp_llvm"
fi

# ---------------------------------------------------------------------------
# step 3: build AscendNPU-IR (bisheng tools used at JIT runtime)
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
# Stamp records repo HEAD + triton-ascend HEAD: re-run rebuilds when either moves.
DIST_FINGERPRINT="$(git -C "$REPO_DIR" rev-parse HEAD)|$(git -C "$TA_DIR" rev-parse HEAD)"
if [[ "$FORCE" != "1" && -f "$WORK_ROOT/.stamp_triton_dist" \
      && "$(cat "$WORK_ROOT/.stamp_triton_dist")" == "$DIST_FINGERPRINT" ]]; then
    echo "Triton-distributed already installed for this exact source combo, skipping. FORCE=1 to rebuild."
    echo "(editable install: pure-Python changes take effect without rebuilding)"
else
    # NOTE: pip replaces any preinstalled triton/triton-ascend wheel from the
    # Docker image; setup.py applies 3rdparty/*.patch (preflighted in step 1.5).
    LLVM_SYSPATH="$LLVM_INSTALL_PREFIX" \
    TRITON_BUILD_WITH_CLANG_LLD=ON \
    TRITON_BUILD_PROTON=OFF \
    TRITON_BUILD_LITTLE_KERNEL=OFF \
    TRITON_USE_ASCEND=ON \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    "$PIP_BIN" install -e ./python --verbose --no-build-isolation
    echo "$DIST_FINGERPRINT" > "$WORK_ROOT/.stamp_triton_dist"
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
