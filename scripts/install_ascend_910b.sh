#!/usr/bin/env bash
# Install the Ascend backend from this checkout; run on the target 910B host.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=${WORK_DIR:-"${ROOT}/.ascend-910b"}
CANN_ENV=${CANN_ENV:-}
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-32}
TEST_TIMEOUT=${TEST_TIMEOUT:-1800}
LLVM_COMMIT=fad3272286528b8a491085183434c5ad4b59ab92
LLVM_PATCH_URL=${LLVM_PATCH_URL:-https://raw.gitcode.com/Ascend/triton-ascend/blobs/2b0a06eb21438359d6d0576b622e3bb5e0292d17/fad3272.patch}
usage() {
    cat <<'EOF'
Usage: bash scripts/install_ascend_910b.sh [--help|--prereqs-only|--tests-only]
Run on a 910B host with CANN and >=4 visible NPUs for the full suite (2 for prereqs).
Environment variables:
  CANN_ENV       Optional explicit set_env.sh path
  ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME / ASCEND_TOOLKIT_LATEST_HOME
                 CANN installation root used to locate set_env.sh (in this order)
  WORK_DIR       Build, venv and log directory (default: .ascend-910b)
  PYTHON         Base Python with compatible torch/torch_npu (default: python3)
  TORCH_VERSION / TORCH_NPU_VERSION Exact package versions if installing from index
  CC / CXX / LD  Optional bootstrap tools (default: clang-15 or clang >=15, ld.lld-15 or ld.lld)
  PIP_INDEX_URL  Python package index (default: https://repo.huaweicloud.com/repository/pypi/simple)
  PIP_DEFAULT_TIMEOUT / PIP_RETRIES Pip timeout in seconds / retries (default: 600 / 10)
  TORCH_WHEEL    Optional exact torch wheel path/URL; use with TORCH_NPU_WHEEL
  TORCH_NPU_WHEEL Optional exact torch_npu wheel path/URL, matched to CANN
  NPUIR_RUNTIME_BIN Optional existing SHMEM-capable runtime compiler bin directory
  NPUIR_RUNTIME_COMMIT Optional runtime source revision (default: fixed August 2026 commit)
  LLVM_INSTALL_PREFIX Optional existing LLVM installation matching docs/build.md
  JOBS           Parallel build jobs (default: 32)
  TEST_TIMEOUT   Timeout in seconds per test phase (default: 1800)
  LLVM_SOURCE_ARCHIVE Optional tar.gz from the exact LLVM_COMMIT (GitHub codeload)
  LLVM_PATCH_URL Optional mirror for the documented LLVM patch
System prerequisites (Debian/Ubuntu example, install as administrator):
  apt-get install git curl build-essential clang-15 lld-15 ccache python3-venv python3-dev
Missing system prerequisites are installed using apt-get or dnf (root required).
CANN and drivers are never updated. --prereqs-only stops after NPU matmul checks.
EOF
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
PREREQS_ONLY=0
TESTS_ONLY=0
if [[ ${1:-} == --prereqs-only ]]; then PREREQS_ONLY=1; shift; fi
if [[ ${1:-} == --tests-only ]]; then TESTS_ONLY=1; shift; fi
[[ $# == 0 ]] || { usage; exit 2; }
[[ $JOBS =~ ^[1-9][0-9]*$ && $TEST_TIMEOUT =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$WORK_DIR"
WORK_DIR=$(cd "$WORK_DIR" && pwd)
ENV_FILE="$WORK_DIR/env.sh"
mkdir -p "$WORK_DIR/logs"
LOG="$WORK_DIR/logs/install-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
STAGE=preflight
trap 'rc=$?; printf "\nFAILED stage=%s line=%s status=%s\nLog: %s\n" "$STAGE" "$LINENO" "$rc" "$LOG" >&2; exit "$rc"' ERR
fail() { printf 'ERROR: %s\nStage: %s\nLog: %s\n' "$*" "$STAGE" "$LOG" >&2; exit 1; }
stage() { STAGE=$1; printf '\n[%s] %s\n' "$(date -u +%FT%TZ)" "$STAGE"; }
if [[ -z $CANN_ENV ]]; then
    cann_root=${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-${ASCEND_TOOLKIT_LATEST_HOME:-}}}
    [[ -n $cann_root ]] || fail 'Set ASCEND_HOME_PATH (or ASCEND_TOOLKIT_HOME/ASCEND_TOOLKIT_LATEST_HOME), or explicitly set CANN_ENV'
    [[ -d $cann_root ]] || fail "CANN installation directory does not exist: $cann_root"
    cann_root=$(cd "$cann_root" && pwd)
    # Some toolkit layouts put set_env.sh alongside the version directory.
    for candidate in "$cann_root/set_env.sh" "$cann_root/../set_env.sh"; do
        if [[ -f $candidate ]]; then CANN_ENV=$candidate; break; fi
    done
    [[ -n $CANN_ENV ]] || fail "No set_env.sh found under $cann_root or its parent; set CANN_ENV explicitly"
fi
[[ -f $CANN_ENV ]] || fail "CANN_ENV does not exist: $CANN_ENV"
CANN_ENV="$(cd "$(dirname "$CANN_ENV")" && pwd)/$(basename "$CANN_ENV")"
printf 'CANN environment: %s\n' "$CANN_ENV"
# Vendor environment scripts are not necessarily nounset-safe.
set +u
source "$CANN_ENV"
set -u
stage system-prerequisites
command -v npu-smi >/dev/null || fail 'npu-smi missing; install the Ascend driver first'
command -v "$PYTHON" >/dev/null || fail "Python not found: $PYTHON"
# Prefer a compatible pair, including versioned binaries not named "clang".
compiler_major() {
    local version
    version=$("$1" -dumpversion 2>/dev/null) || return 1
    version=${version%%.*}
    [[ $version =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$version"
}
select_clang() {
    local c cpp major cpp_major suffix
    local requested_cc=${CC:-} requested_cxx=${CXX:-}
    for suffix in custom -15 '' -22 -21 -20 -19 -18 -17 -16; do
        if [[ $suffix == custom ]]; then
            [[ -n $requested_cc && -n $requested_cxx ]] || continue
            c=$requested_cc; cpp=$requested_cxx
        else
            c=clang$suffix; cpp=clang++$suffix
        fi
        c=$(command -v "$c") || continue
        cpp=$(command -v "$cpp") || continue
        major=$(compiler_major "$c") || continue
        cpp_major=$(compiler_major "$cpp") || continue
        if (( major >= 15 && major == cpp_major )); then
            CC=$c; CXX=$cpp
            return 0
        fi
    done
    return 1
}
clang_ready=0
if select_clang; then clang_ready=1; fi
LD=${LD:-$(command -v ld.lld-15 || command -v ld.lld || true)}
missing=0
for tool in git curl make g++ ccache timeout "${CC:-clang}" "${CXX:-clang++}" "${LD:-ld.lld}"; do
    command -v "$tool" >/dev/null || missing=1
done
if (( missing || ! clang_ready )); then
    [[ $(id -u) == 0 ]] || fail 'Missing or outdated build prerequisites; rerun as root for automatic installation, or provide a compatible CC/CXX pair and install remaining tools listed in --help'
    if command -v apt-get >/dev/null; then
        packages=()
        for package in git curl build-essential clang-15 lld-15 ccache python3-venv python3-dev; do
            if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -qx 'install ok installed'; then
                packages+=("$package")
            fi
        done
        if (( ! clang_ready )); then
            # Install the versioned pair alongside old clang; do not replace it.
            packages+=(clang-15 lld-15)
        fi
        if (( ${#packages[@]} )); then
            apt-get update
            apt-get install -y "${packages[@]}"
        fi
    elif command -v dnf >/dev/null; then
        packages=()
        for package in git curl gcc gcc-c++ make clang lld ccache python3-devel; do
            if ! rpm -q "$package" >/dev/null 2>&1; then packages+=("$package"); fi
        done
        if (( ! clang_ready )); then
            # Installed-but-old packages must remain in this transaction so dnf
            # can upgrade them to the newest version in configured repositories.
            packages+=(clang lld)
        fi
        if (( ${#packages[@]} )); then dnf install -y "${packages[@]}"; fi
    else
        fail 'Unsupported package manager; install build prerequisites manually'
    fi
fi
if ! select_clang; then
    fail 'Configured repositories did not provide clang/clang++ >=15 with matching major versions. Install a newer LLVM toolchain or enable its distro repository, then set both CC and CXX (and optionally LD); existing build files are retained'
fi
if [[ -z $LD ]] || ! command -v "$LD" >/dev/null; then
    LD=$(command -v ld.lld-15 || command -v ld.lld) || fail 'Install lld or specify LD'
fi
"$CC" --version
"$CXX" --version
"$LD" --version
# Bootstrap compiler paths are passed to LLVM and NPU IR, not forced onto SHMEM.
npu-smi info
[[ -n ${ASCEND_HOME_PATH:-} && -d $ASCEND_HOME_PATH ]] || fail 'CANN did not set ASCEND_HOME_PATH'
find "$ASCEND_HOME_PATH" -maxdepth 2 -name '*version*.cfg' -print 2>/dev/null
[[ -z ${TORCH_WHEEL:-} && -z ${TORCH_NPU_WHEEL:-} || -n ${TORCH_WHEEL:-} && -n ${TORCH_NPU_WHEEL:-} ]] || fail 'Supply both torch wheels together'

stage python-environment
VENV="$WORK_DIR/venv"
if [[ ! -x $VENV/bin/python ]]; then "$PYTHON" -m venv --system-site-packages "$VENV"; fi
source "$VENV/bin/activate"
# Export once so nested builds use the same index and download settings.
export PIP_INDEX_URL=${PIP_INDEX_URL:-https://repo.huaweicloud.com/repository/pypi/simple}
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-600}
export PIP_RETRIES=${PIP_RETRIES:-10}
python -m pip install 'setuptools>=40.8.0' wheel 'cmake>=3.20,<4' 'ninja>=1.11.1' 'pybind11>=2.13.1' packaging pytest numpy pyyaml
if [[ -n ${TORCH_WHEEL:-} ]]; then
    python -m pip install "$TORCH_WHEEL" "$TORCH_NPU_WHEEL"
elif [[ -n ${TORCH_VERSION:-} || -n ${TORCH_NPU_VERSION:-} ]]; then
    [[ -n ${TORCH_VERSION:-} && -n ${TORCH_NPU_VERSION:-} ]] || fail 'Supply both TORCH_VERSION and TORCH_NPU_VERSION'
    python -m pip install "torch==$TORCH_VERSION" "torch_npu==$TORCH_NPU_VERSION"
elif ! python -c 'import importlib.util; assert importlib.util.find_spec("torch") and importlib.util.find_spec("torch_npu")'; then
    fail 'torch/torch_npu missing. Supply TORCH_VERSION and TORCH_NPU_VERSION matched to CANN, or both wheel paths; no compatible version is guessed'
fi
# Platform wrappers may request a hardware family check without overriding detection.
export TRITON_DIST_EXPECTED_PLATFORM=${TRITON_DIST_EXPECTED_PLATFORM:-}
required_npus=4
if (( PREREQS_ONLY )); then required_npus=2; fi
REQUIRED_NPUS="$required_npus" timeout --kill-after=30s "$TEST_TIMEOUT" python -u - <<'PY'
import os
import torch
import torch_npu
print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__)
from packaging.version import Version
assert Version(torch.__version__).release[:2] == Version(torch_npu.__version__).release[:2], 'torch/torch_npu major.minor versions differ'
assert torch.npu.is_available(), 'NPU unavailable; check torch_npu/CANN/driver compatibility'
required = int(os.environ['REQUIRED_NPUS'])
assert torch.npu.device_count() >= required, f'This mode requires at least {required} visible NPUs'
for device in range(2):
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(device)
    if os.environ.get('TRITON_DIST_EXPECTED_PLATFORM') == 'A3':
        assert name.startswith(('Ascend910_93', 'Ascend910C')), f'A3 device expected, detected {name}'
    x = torch.ones((16, 16), device=f'npu:{device}')
    torch.testing.assert_close((x @ x).cpu(), torch.full((16, 16), 16.))
    print('NPU', device, torch.npu.get_device_name(device), 'matmul PASS')
PY

if (( PREREQS_ONLY )); then
    printf 'PASS: build tools and torch/torch_npu two-device matmul. Log: %s\n' "$LOG"
    exit 0
fi

if (( ! TESTS_ONLY )); then
# setup.py patches tracked files. Permit only exactly those patches on a rerun.
check_clean() {
    local path=$1 patch=${2:-}
    [[ -d $path ]] || return 0
    git -C "$path" rev-parse --show-toplevel >/dev/null 2>&1 || return 0
    [[ $(git -C "$path" rev-parse --show-toplevel) == "$path" ]] || return 0
    if [[ -z $(git -C "$path" status --porcelain --untracked-files=no --ignore-submodules=all) ]]; then return 0; fi
    if [[ -n $patch && -f $patch ]] && git -C "$path" apply --reverse --check "$patch"; then
        local expected actual
        expected=$(mktemp "$WORK_DIR/expected.XXXXXX")
        actual=$(mktemp "$WORK_DIR/actual.XXXXXX")
        git -C "$path" diff --ignore-submodules=all --binary HEAD -- > "$actual"
        # Compare resulting trees in a temporary index, without changing source files.
        local index
        index=$(mktemp "$WORK_DIR/index.XXXXXX"); rm "$index"
        GIT_INDEX_FILE="$index" git -C "$path" read-tree HEAD
        GIT_INDEX_FILE="$index" git -C "$path" apply --cached "$patch"
        GIT_INDEX_FILE="$index" git -C "$path" diff --cached --binary HEAD -- > "$expected"
        if cmp -s "$expected" "$actual"; then rm -f "$expected" "$actual" "$index"; return 0; fi
        rm -f "$expected" "$actual" "$index"
    fi
    fail "Local tracked modifications in $path; preserve them before installing"
}
# Some GitCode repositories advertise branches but no default HEAD. Bootstrap
# their submodules by the parent's gitlink instead of cloning the default branch.
init_pinned_submodule() {
    local parent=$1 relative=$2 path revision url
    path="$parent/$relative"
    revision=$(git -C "$parent" rev-parse "HEAD:$relative")
    git -C "$parent" submodule init -- "$relative"
    url=$(git -C "$parent" config --get "submodule.$relative.url")
    if [[ ! -e $path/.git ]]; then
        if [[ -d $path && -n $(find "$path" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
            fail "Nonempty submodule directory without Git metadata: $path; preserve its contents before installing"
        fi
        git init "$path"
        git -C "$path" remote add origin "$url"
    fi
    if ! git -C "$path" cat-file -e "$revision^{commit}" 2>/dev/null; then
        git -C "$path" fetch --depth=1 origin "$revision"
    fi
    # A normal checkout preserves local changes or refuses conflicts; never force.
    git -C "$path" checkout --detach "$revision"
    git -C "$parent" submodule absorbgitdirs -- "$relative"
}
TA="$ROOT/3rdparty/triton-ascend"
NPUIR="$TA/third_party/ascend/AscendNPU-IR"
SHMEM="$ROOT/3rdparty/shmem"
stage submodules
check_clean "$TA" "$ROOT/3rdparty/triton-ascend.patch"
check_clean "$NPUIR" "$ROOT/3rdparty/AscendNPU-IR.patch"
check_clean "$SHMEM"
git -C "$ROOT" submodule update --init --checkout 3rdparty/triton-ascend 3rdparty/shmem
git -C "$TA" submodule update --init --depth=1 third_party/ascend/AscendNPU-IR
git -C "$SHMEM" submodule update --init --recursive
[[ -f $NPUIR/build-tools/build.sh ]] || fail 'Pinned AscendNPU-IR submodule missing'
for dir in "$TA" "$NPUIR" "$SHMEM"; do git -C "$dir" log -1 --format='%H %s'; done
# Never fetch a floating AscendNPU-IR HEAD: use the compiler's pinned submodule.
for entry in 'triton-ascend' 'AscendNPU-IR'; do
    path=$TA; [[ $entry == AscendNPU-IR ]] && path=$NPUIR
    patch="$ROOT/3rdparty/$entry.patch"
    if git -C "$path" apply --reverse --check "$patch" 2>/dev/null; then
        printf 'Already applied: %s\n' "$patch"
    else
        git -C "$path" apply --check "$patch"
        git -C "$path" apply "$patch"
    fi
done

stage llvm
if [[ -z ${LLVM_INSTALL_PREFIX:-} ]]; then
    LLVM_INSTALL_PREFIX="$WORK_DIR/llvm-install"
    LLVM_SRC="$WORK_DIR/llvm-project"
    if [[ -n ${LLVM_SOURCE_ARCHIVE:-} ]]; then
        if [[ ! -f $LLVM_SRC/.archive-revision ]]; then
            [[ ! -e $LLVM_SRC ]] || fail 'Use a fresh WORK_DIR for LLVM_SOURCE_ARCHIVE'
            tar -tzf "$LLVM_SOURCE_ARCHIVE" | cut -d/ -f1 | sort -u > "$WORK_DIR/archive-roots.txt"
            [[ $(cat "$WORK_DIR/archive-roots.txt") == "llvm-project-$LLVM_COMMIT" ]] || fail 'LLVM archive revision differs from pinned commit'
            mkdir -p "$LLVM_SRC"
            tar -xzf "$LLVM_SOURCE_ARCHIVE" --strip-components=1 -C "$LLVM_SRC"
            printf '%s\n' "$LLVM_COMMIT" > "$LLVM_SRC/.archive-revision"
        fi
        [[ $(cat "$LLVM_SRC/.archive-revision") == "$LLVM_COMMIT" ]] || fail 'LLVM archive marker mismatch'
    elif [[ ! -d $LLVM_SRC/.git ]]; then
        git init "$LLVM_SRC"
        git -C "$LLVM_SRC" remote add origin https://github.com/llvm/llvm-project.git
    fi
    if [[ ! -f $LLVM_INSTALL_PREFIX/.triton-dist-llvm-ready ]]; then
        if git -C "$LLVM_SRC" rev-parse HEAD >/dev/null 2>&1; then
            check_clean "$LLVM_SRC" "$WORK_DIR/llvm.patch"
        fi
        if [[ -z ${LLVM_SOURCE_ARCHIVE:-} ]]; then
            git -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=60 -C "$LLVM_SRC" fetch --depth=1 origin "$LLVM_COMMIT"
            git -C "$LLVM_SRC" checkout --detach FETCH_HEAD
        fi
        if [[ ! -s $WORK_DIR/llvm.patch ]]; then
            curl -fL --retry 3 --connect-timeout 30 "$LLVM_PATCH_URL" -o "$WORK_DIR/llvm.patch.download"
            mv "$WORK_DIR/llvm.patch.download" "$WORK_DIR/llvm.patch"
        fi
        if ! git -C "$LLVM_SRC" apply --reverse --check "$WORK_DIR/llvm.patch" 2>/dev/null; then
            git -C "$LLVM_SRC" apply "$WORK_DIR/llvm.patch"
        fi
        cmake -S "$LLVM_SRC/llvm" -B "$WORK_DIR/llvm-build" -G Ninja \
          -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
          -DCMAKE_LINKER="$LD" -DCMAKE_BUILD_TYPE=Release \
          -DLLVM_ENABLE_ASSERTIONS=ON '-DLLVM_ENABLE_PROJECTS=mlir;llvm;lld' \
          '-DLLVM_TARGETS_TO_BUILD=host;NVPTX;AMDGPU' -DLLVM_ENABLE_LLD=ON \
          -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX"
        cmake --build "$WORK_DIR/llvm-build" --target install FileCheck --parallel "$JOBS"
        cp "$WORK_DIR/llvm-build/bin/FileCheck" "$LLVM_INSTALL_PREFIX/bin/"
        cp "$WORK_DIR/llvm-build/bin/llvm-lit" "$LLVM_INSTALL_PREFIX/bin/"
        touch "$LLVM_INSTALL_PREFIX/.triton-dist-llvm-ready"
    fi
fi
[[ -f $LLVM_INSTALL_PREFIX/lib/cmake/llvm/LLVMConfig.cmake ]] || fail 'Invalid LLVM_INSTALL_PREFIX'
export LLVM_INSTALL_PREFIX LLVM_SYSPATH="$LLVM_INSTALL_PREFIX"
export PATH="$LLVM_INSTALL_PREFIX/bin:$PATH"
export MAX_JOBS="$JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

stage ascend-npu-ir
# The headers pinned by Triton predate the SHMEM runtime templates. Build a
# separate, fixed runtime compiler; never replace Triton's header submodule.
NPUIR_RUNTIME_COMMIT=${NPUIR_RUNTIME_COMMIT:-f16704a2545db71c75b1a4f9f8c325c3e236954e}
NPUIR_RUNTIME_DIR=${NPUIR_RUNTIME_DIR:-"$WORK_DIR/npuir-runtime"}
if [[ -z ${NPUIR_RUNTIME_BIN:-} ]]; then
    NPUIR_RUNTIME_BIN="$WORK_DIR/npuir-build/bin"
    npuir_marker="$WORK_DIR/npuir-build/.triton-dist-npuir-ready"
    npuir_config=$(printf '%s\n' "$NPUIR_RUNTIME_COMMIT" "$NPUIR_RUNTIME_DIR" \
        "$ASCEND_HOME_PATH" "$CC" "$CXX" 'Release;shmem-template;apply-patches' | sha256sum)
    if [[ -f $npuir_marker && $(cat "$npuir_marker") == "$npuir_config" ]] &&
       [[ -x $NPUIR_RUNTIME_BIN/bishengir-compile ]] &&
       "$NPUIR_RUNTIME_BIN/bishengir-compile" --version >/dev/null 2>&1; then
        printf 'Reusing NPU IR build: %s\n' "$NPUIR_RUNTIME_BIN"
    else
        # An interrupted or changed build must not retain a success marker.
        rm -f "$npuir_marker"
        if [[ ! -d $NPUIR_RUNTIME_DIR/.git ]]; then
            git clone --no-checkout https://gitcode.com/Ascend/AscendNPU-IR.git "$NPUIR_RUNTIME_DIR"
        fi
        git -C "$NPUIR_RUNTIME_DIR" checkout "$NPUIR_RUNTIME_COMMIT"
        for dependency in llvm-project torch-mlir; do
            init_pinned_submodule "$NPUIR_RUNTIME_DIR" "third-party/$dependency"
        done
        git -C "$NPUIR_RUNTIME_DIR" submodule update --init --depth=1 --jobs=1
        (
            cd "$NPUIR_RUNTIME_DIR"
            # The upstream script skips configuration whenever a cache exists,
            # even after a failed configure or a compiler change. Regenerate it
            # while retaining sources and the remaining build artifacts.
            npuir_reconfigure=()
            if [[ -f $WORK_DIR/npuir-build/CMakeCache.txt ]]; then
                npuir_reconfigure+=(--rebuild)
            fi
            ASM="$CC" bash build-tools/build.sh -o "$WORK_DIR/npuir-build" -t --build-type Release \
              --c-compiler "$CC" --cxx-compiler "$CXX" "${npuir_reconfigure[@]}" \
              --apply-patches --bisheng-compiler="$ASCEND_HOME_PATH/bin" --build-shmem-template -j "$JOBS"
        )
        [[ -x $NPUIR_RUNTIME_BIN/bishengir-compile ]] || fail 'bishengir-compile was not produced'
        "$NPUIR_RUNTIME_BIN/bishengir-compile" --version >/dev/null
        printf '%s\n' "$npuir_config" > "$npuir_marker"
    fi
fi
[[ -x $NPUIR_RUNTIME_BIN/bishengir-compile ]] || fail 'bishengir-compile was not produced'
export PATH="$NPUIR_RUNTIME_BIN:$PATH"
export TRITON_USE_ASCEND=ON TRITON_BUILD_WITH_CLANG_LLD=ON
export TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF
# setup.py's Clang/LLD mode defaults to unversioned binary names. Its appended
# arguments take precedence, including over compiler paths in an old cache.
TRITON_APPEND_CMAKE_ARGS=$(python - "$CC" "$CXX" "$(command -v "$LD")" <<'PY'
import shlex
import sys

cc, cxx, linker = sys.argv[1:]
args = ["-DTRITON_BUILD_UT=OFF", f"-DCMAKE_C_COMPILER={cc}",
        f"-DCMAKE_CXX_COMPILER={cxx}", f"-DCMAKE_LINKER={linker}"]
args += [f"-DCMAKE_{kind}_LINKER_FLAGS=-fuse-ld={linker}"
         for kind in ("EXE", "MODULE", "SHARED")]
print(shlex.join(args))
PY
)
export TRITON_APPEND_CMAKE_ARGS
stage triton-distributed
# setup.py enumerates packages before its wheel hook creates this link.
# Create it before metadata generation, otherwise a first install omits triton/*.py.
if [[ -L $ROOT/python/triton ]]; then
    case $(readlink "$ROOT/python/triton") in
        ../3rdparty/triton/python/triton|../3rdparty/triton-ascend/python/triton)
            ln -sfn ../3rdparty/triton-ascend/python/triton "$ROOT/python/triton" ;;
        *) fail 'Unknown python/triton symlink; preserve it before building' ;;
    esac
elif [[ ! -e $ROOT/python/triton ]]; then
    ln -s ../3rdparty/triton-ascend/python/triton "$ROOT/python/triton"
fi
[[ $(readlink -f "$ROOT/python/triton") == "$TA/python/triton" ]] || fail 'python/triton points to another backend; preserve it and select the Ascend link first'
python -m pip install --no-build-isolation --verbose "$ROOT/python"
stage ascend-shmem
(
    cd "$SHMEM"
    bash scripts/build.sh -python_extension
)
# Select only SHMEM wheels; never install an arbitrary stale glob of all wheels.
mapfile -t wheels < <(find "$SHMEM/dist" -maxdepth 1 -name 'shmem-*.whl' -type f)
[[ ${#wheels[@]} == 1 ]] || fail 'Expected exactly one shmem wheel in 3rdparty/shmem/dist; remove stale wheels and rerun'
python -m pip install --force-reinstall --no-deps "${wheels[0]}"

stage write-environment
ENV_FILE="$WORK_DIR/env.sh"
{
    printf '# Generated Ascend runtime environment\n_ascend_restore_nounset=0\n[[ $- == *u* ]] && _ascend_restore_nounset=1\nset +u\nsource %q\nsource %q\n[[ $_ascend_restore_nounset == 0 ]] || set -u\nunset _ascend_restore_nounset\n' "$CANN_ENV" "$VENV/bin/activate"
    printf 'export LLVM_INSTALL_PREFIX=%q\nexport LLVM_SYSPATH=%q\n' "$LLVM_INSTALL_PREFIX" "$LLVM_INSTALL_PREFIX"
    printf 'export PATH=%q:%q:"$PATH"\n' "$NPUIR_RUNTIME_BIN" "$LLVM_INSTALL_PREFIX/bin"
    printf 'export TRITON_USE_ASCEND=ON\n'
    printf 'export TRITON_DIST_EXPECTED_PLATFORM=%q\n' "$TRITON_DIST_EXPECTED_PLATFORM"
} > "$ENV_FILE"
else
    [[ -f $WORK_DIR/env.sh ]] || fail 'No installed environment; run full installation first'
    source "$WORK_DIR/env.sh"
fi
cd "$ROOT"
stage backend-import
python - <<'PY'
import torch, torch_npu, triton, triton_dist, shmem
from triton.runtime import driver
torch.npu.set_device(0)
target = driver.active.get_current_target()
print('Triton:', triton.__file__, 'target:', target)
assert target.backend in ('npu', 'ascend'), f'Unexpected backend: {target}'
import os
if os.environ.get('TRITON_DIST_EXPECTED_PLATFORM') == 'A3':
    assert str(target.arch).startswith('Ascend910_93'), f'A3 compiler target expected: {target.arch}'
assert torch.npu.device_count() >= 4, 'The full suite includes a four-rank sub-team test; expose >=4 NPUs'
assert hasattr(shmem, 'aclshmem_init'), 'Wrong SHMEM Python package'
PY
stage hccl-test
cat > "$WORK_DIR/test_hccl.py" <<'PY'
import os
from datetime import timedelta
import torch
import torch_npu
import torch.distributed as dist
rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(rank)
dist.init_process_group('hccl', timeout=timedelta(seconds=180))
try:
    x = torch.tensor([float(rank + 1)], device=f'npu:{rank}')
    dist.all_reduce(x)
    torch.testing.assert_close(x.cpu(), torch.tensor([3.]))
    print(f'HCCL rank {rank} PASS', flush=True)
finally:
    dist.destroy_process_group()
PY
timeout --kill-after=30s "$TEST_TIMEOUT" python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 "$WORK_DIR/test_hccl.py"
stage communication-primitives
# Serial execution: upstream tests bind fixed localhost ports (including 8666).
# Register torch_npu in pytest's parent before multiprocessing spawns workers.
timeout --kill-after=30s "$TEST_TIMEOUT" python -c 'import torch_npu, pytest; raise SystemExit(pytest.main(["--import-mode=importlib", "-v", "-ra", "python/triton_dist/test/ascend/"]))'
stage allgather-gemm
timeout --kill-after=30s "$TEST_TIMEOUT" python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 tutorials/ascend/01-ascend-allgather-gemm.py
stage complete
printf 'PASS: NPU compute, backend imports, HCCL, communication primitives, AllGather GEMM.\nEnvironment: source %q\nLog: %s\n' "$ENV_FILE" "$LOG"
