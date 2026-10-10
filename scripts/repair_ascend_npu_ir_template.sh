#!/usr/bin/env bash
# Repair an EXISTING AscendNPU-IR build tree so that it contains the
# device-side meta-op bitcode that CANN's hivmc needs.
#
# Why this exists:
#   bishengir-compile attaches <bishengir-compile>/../lib/{meta_op.aic,
#   meta_op.aiv,meta_op.mix.aic,meta_op.mix.aiv,host}.bc to every HIVM module
#   and hivmc links them into the kernel binary. build-tools/build.sh only
#   produces those files with -t (BISHENGIR_BUILD_TEMPLATE=ON); AND it only
#   re-runs CMake when build/CMakeCache.txt is absent:
#
#       elif [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then cmake_generate; fi
#
#   so a tree configured WITHOUT -t silently ignores a later `-t` re-run
#   (it just rebuilds nothing and the .bc files stay missing). Symptom:
#   every kernel fails at runtime with
#       error: Failed to compile BiShengLIR to binary
#   even though `build.sh` printed "Build Done!!!".
#
# This script deletes build/CMakeCache.txt (NOT the whole build tree: the .o
# files live under build/CMakeFiles, so keeping them makes the rebuild
# incremental) and re-runs build.sh with -t, which now re-runs CMake.
#
# Usage (on the A3/A2 host or inside the CANN container):
#   bash scripts/repair_ascend_npu_ir_template.sh
# Overridable: WORK_ROOT, NPU_IR_DIR, CANN_ENV, JOBS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$HOME/ascend-build}"
NPU_IR_DIR="${NPU_IR_DIR:-$WORK_ROOT/AscendNPU-IR}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
BCS=(meta_op.aic.bc meta_op.aiv.bc meta_op.mix.aic.bc meta_op.mix.aiv.bc host.bc)

die() { echo "[ERROR] $*" >&2; exit 1; }
say() { echo "[repair] $*"; }

have_all() {
    local bc
    for bc in "${BCS[@]}"; do
        [[ -f "$NPU_IR_DIR/build/lib/$bc" ]] || return 1
    done
}

[[ -d "$NPU_IR_DIR/build/bin" ]] \
    || die "$NPU_IR_DIR/build/bin not found -- run scripts/build_ascend_a3.sh first, or set NPU_IR_DIR=..."
[[ -f "$NPU_IR_DIR/build-tools/build.sh" ]] \
    || die "$NPU_IR_DIR/build-tools/build.sh not found -- $NPU_IR_DIR is not an AscendNPU-IR tree"

if have_all; then
    say "meta-op bitcode already present in $NPU_IR_DIR/build/lib -- nothing to do"
    exit 0
fi

[[ -f "$CANN_ENV" ]] || die "$CANN_ENV not found -- source CANN first or set CANN_ENV=..."
# shellcheck disable=SC1090
source "$CANN_ENV"
: "${ASCEND_HOME_PATH:?CANN set_env.sh did not define ASCEND_HOME_PATH -- check CANN_ENV}"
for tool in ccec llvm-link; do
    [[ -x "$ASCEND_HOME_PATH/bin/$tool" ]] \
        || die "$ASCEND_HOME_PATH/bin/$tool is missing -- -t compiles the Template sources with ccec and links them with llvm-link, so CANN must provide both"
done

say "re-configuring $NPU_IR_DIR/build with -t (dropping only CMakeCache.txt; objects stay)"
cd "$NPU_IR_DIR"
# build.sh skips cmake_generate() whenever build/CMakeCache.txt exists (see
# build-tools/build.sh: main()), so the cached -DBISHENGIR_BUILD_TEMPLATE=OFF
# would win. Dropping just the cache forces a re-configure; `-r/--rebuild`
# would also delete build/CMakeFiles and therefore every object file.
rm -f "$NPU_IR_DIR/build/CMakeCache.txt"
bash ./build-tools/build.sh -o ./build -j "$JOBS" --build-type Release \
    -t \
    --bisheng-compiler="$ASCEND_HOME_PATH/bin" \
    --add-cmake-options="-DLLVM_INCLUDE_TESTS=OFF -DMLIR_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF"

missing=()
for bc in "${BCS[@]}"; do
    [[ -f "$NPU_IR_DIR/build/lib/$bc" ]] || missing+=("$bc")
done
[[ ${#missing[@]} -eq 0 ]] \
    || die "still missing after the rebuild: ${missing[*]} -- inspect the '-t' section of the output above (BISHENGIR_BUILD_TEMPLATE must be ON and ccec/llvm-link must actually work)"

say "OK:"
ls -l "$NPU_IR_DIR/build/lib"/meta_op.*.bc "$NPU_IR_DIR/build/lib"/host.bc

cat <<EOF

Next:
    export PATH=$NPU_IR_DIR/build/bin:\$PATH
    python3 $REPO_DIR/scripts/probe_ascend_language_api.py
    # or the full ladder:
    bash $REPO_DIR/scripts/triage_ascend_runtime.sh
EOF
