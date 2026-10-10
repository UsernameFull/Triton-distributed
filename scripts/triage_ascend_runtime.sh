#!/usr/bin/env bash
# Ascend runtime triage: pin down WHERE a segfault (torchrun "exitcode -11" /
# SIGSEGV) or other hard crash happens when running tutorials/ascend/*.py.
#
# Rationale: a SIGSEGV with no Python traceback can come from three very
# different layers, and they need different fixes:
#
#   1. in-process Triton/Ascend lowering  -> python/triton_dist/test/ascend/
#      test_gm_addr_args_indices.py runs the HIVM distributed passes on MLIR
#      strings and needs NO device, so it isolates this layer;
#   2. the ACLSHMEM / CANN / torch_npu / HCCL runtime -> the `-m dist` tests
#      form a ladder from "aclshmem init + my_pe" (test_my_pe.py) up to
#      "distributed kernel" (test_language_extra.py);
#   3. the tutorial's own kernel (swizzle + symm_at + barrier_all + sub_vec_id)
#      -> only the tutorial itself exercises that.
#
# The ladder is incremental: the FIRST stage that fails bounds the problem.
#
# Usage (from the repo root, on the A3/A2 host or inside the CANN container):
#   bash scripts/triage_ascend_runtime.sh
#   STOP_ON_FAIL=0 bash scripts/triage_ascend_runtime.sh   # run every stage
#   RUN_TUTORIAL=0 bash scripts/triage_ascend_runtime.sh   # skip the tutorial
#   CLEAR_CACHE=1  bash scripts/triage_ascend_runtime.sh   # wipe ~/.triton/cache first
#
# Overridable environment variables (defaults in brackets):
#   REPO_DIR     [repo containing this script]
#   CANN_ENV     [/usr/local/Ascend/ascend-toolkit/set_env.sh]
#   WORK_ROOT    [$HOME/ascend-build]        logs go to $WORK_ROOT/triage-logs
#   TUTORIAL     [tutorials/ascend/01-ascend-allgather-gemm.py]
#   WORLD        [2]                         ranks for the tutorial stage
#   MASTER_PORT  [29500]   torchrun rendezvous port; change it if a stale
#                          torchrun/TCPStore still holds 29500 (EADDRINUSE)
#   RUN_TUTORIAL [1]
#   STOP_ON_FAIL [1]
#   PYTEST_EXTRA [-x -q]
#   CLEAR_CACHE  [0]  remove ~/.triton/cache before running (a cache written by
#                     a different/mismatched libtriton can itself cause SIGSEGV)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
WORK_ROOT="${WORK_ROOT:-$HOME/ascend-build}"
TUTORIAL="${TUTORIAL:-tutorials/ascend/01-ascend-allgather-gemm.py}"
WORLD="${WORLD:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"
RUN_TUTORIAL="${RUN_TUTORIAL:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
PYTEST_EXTRA="${PYTEST_EXTRA:--x -q}"
CLEAR_CACHE="${CLEAR_CACHE:-0}"

LOG_DIR="$WORK_ROOT/triage-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/triage-$(date +%Y%m%d-%H%M%S).log"

say()    { echo "[triage] $*"; }
banner() { echo; echo "============================================================"; echo ">>> $*"; echo "============================================================"; }
signal_note() {  # $1=rc -> ", SIGSEGV (segfault)" style suffix
    local rc="$1" sig
    [[ "$rc" -ge 128 ]] || return 0
    sig=$((rc - 128))
    case "$sig" in
        4)  echo ", SIGILL" ;;
        6)  echo ", SIGABRT" ;;
        11) echo ", SIGSEGV (segfault)" ;;
        *)  echo ", killed by signal $sig" ;;
    esac
}

RESULTS=()
FAILED=()

run_stage() {  # $1=label $2..=command
    local label="$1"; shift
    banner "stage: $label"
    echo "\$ $*"
    local rc=0
    "$@" || rc=$?
    if [[ $rc -eq 0 ]]; then
        say "PASS  $label"
        RESULTS+=("PASS|$label|0")
    else
        local note; note="$(signal_note "$rc")"
        say "FAIL  $label (rc=$rc$note)"
        RESULTS+=("FAIL|$label|$rc$note")
        FAILED+=("$label")
    fi
    return "$rc"
}

exec > >(tee -a "$LOG_FILE") 2>&1

say "repo: $REPO_DIR"
say "log : $LOG_FILE"

# ---------------------------------------------------------------------------
# stage 0: environment / version matrix
# ---------------------------------------------------------------------------
banner "stage: environment"
if [[ -f "$CANN_ENV" ]]; then
    # shellcheck disable=SC1090
    source "$CANN_ENV"
    say "CANN: ${ASCEND_TOOLKIT_HOME:-unknown} (ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-unset})"
    for vf in "$ASCEND_TOOLKIT_HOME/version.cfg" /usr/local/Ascend/ascend-toolkit/latest/version.cfg; do
        [[ -f "$vf" ]] && { echo "--- $vf ---"; grep -iE '^version|^Version' "$vf" || true; break; }
    done
else
    say "WARN: $CANN_ENV not found -- run inside the container or set CANN_ENV=..."
fi
command -v npu-smi >/dev/null 2>&1 && { echo "--- npu-smi info ---"; npu-smi info | head -12 || true; }
if command -v bishengir-compile >/dev/null 2>&1; then
    echo "--- bishengir-compile ---"; bishengir-compile --version 2>&1 | head -3 || true
else
    say "bishengir-compile not on PATH (the build exports \$WORK_ROOT/AscendNPU-IR/build/bin)"
fi

python3 - <<'PY'
import importlib
import sys

print("--- python module matrix ---")
print(f"  python       {sys.version.split()[0]}  ({sys.executable})")
for mod in ("torch", "torch_npu", "triton", "triton_dist", "shmem"):
    try:
        m = importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        print(f"  {mod:12s} IMPORT FAILED: {type(e).__name__}: {e}")
        continue
    print(f"  {mod:12s} {str(getattr(m, '__version__', '?')):28s} {getattr(m, '__file__', '?')}")
try:
    from triton._C import libtriton
    print(f"  {'libtriton':12s} {'':28s} {libtriton.__file__}")
    print(f"  {'submodules':12s} {[n for n in ('ir', 'llvm', 'ascend', 'distributed') if hasattr(libtriton, n)]}")
except Exception as e:  # noqa: BLE001
    print(f"  {'libtriton':12s} IMPORT FAILED: {type(e).__name__}: {e}")
PY

TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
say "triton cache: $TRITON_CACHE"
[[ -d "$TRITON_CACHE" ]] && { du -sh "$TRITON_CACHE" 2>/dev/null || true; }
if [[ "$CLEAR_CACHE" == "1" && -d "$TRITON_CACHE" ]]; then
    say "CLEAR_CACHE=1: removing $TRITON_CACHE"
    rm -rf "$TRITON_CACHE"
fi

cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# stages 1..N: the incremental ladder (first failure bounds the problem)
# ---------------------------------------------------------------------------
# Ordering: device-free compiler passes first, then aclshmem init (the most
# basic distributed capability), then progressively heavier distributed work.
STAGES=(
    "compiler-ir (no device)|python/triton_dist/test/ascend/test_gm_addr_args_indices.py"
    "aclshmem-init|python/triton_dist/test/ascend/test_my_pe.py -m dist"
    "aclshmem-ranks|python/triton_dist/test/ascend/test_num_ranks.py -m dist"
    "symm_at|python/triton_dist/test/ascend/test_symm_at.py -m dist"
    "barrier|python/triton_dist/test/ascend/test_barrier_ops.py -m dist"
    "put_get_mem|python/triton_dist/test/ascend/test_put_get_mem.py -m dist"
    "signal_op|python/triton_dist/test/ascend/test_signal_op.py -m dist"
    "wait_notify|python/triton_dist/test/ascend/test_wait_notify.py -m dist"
    "language_extra|python/triton_dist/test/ascend/test_language_extra.py -m dist"
)

for spec in "${STAGES[@]}"; do
    label="${spec%%|*}"
    args="${spec#*|}"
    # shellcheck disable=SC2086
    run_stage "$label" python3 -m pytest $args $PYTEST_EXTRA || true
    if [[ ${#FAILED[@]} -gt 0 && "$STOP_ON_FAIL" == "1" ]]; then
        say "first failure reached -- stopping (STOP_ON_FAIL=0 runs every stage)"
        break
    fi
done

# ---------------------------------------------------------------------------
# final stage: the tutorial itself
# ---------------------------------------------------------------------------
if [[ "$RUN_TUTORIAL" == "1" && ( ${#FAILED[@]} -eq 0 || "$STOP_ON_FAIL" != "1" ) ]]; then
    # PYTHONFAULTHANDLER -> python traceback on SIGSEGV
    # ASCEND_LAUNCH_BLOCKING -> kernels run synchronously, so an error surfaces
    #   at the launch that caused it instead of somewhere later
    # TRITON_ALWAYS_COMPILE -> ignores cached device binaries (a stale cache is
    #   itself a common cause of a segfault right after switching libtriton)
    export PYTHONFAULTHANDLER=1
    export ASCEND_LAUNCH_BLOCKING=1
    export TRITON_ALWAYS_COMPILE=1
    if command -v torchrun >/dev/null 2>&1; then
        run_stage "tutorial $TUTORIAL ($WORLD ranks)" \
            torchrun --nproc-per-node="$WORLD" --master_port="$MASTER_PORT" \
                "$TUTORIAL" || true
    else
        say "torchrun not found -- skipping the tutorial stage"
    fi
fi

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
banner "summary"
for r in "${RESULTS[@]}"; do
    printf '  %-6s %-34s %s\n' "${r%%|*}" "$(echo "$r" | cut -d'|' -f2)" "$(echo "$r" | cut -d'|' -f3)"
done
echo
if [[ ${#FAILED[@]} -eq 0 ]]; then
    say "every stage passed -- the crash is NOT reproduced by the ladder"
else
    say "first failing stage: ${FAILED[0]}"
    cat <<EOF

Next steps:
  * 'compiler-ir' failed        -> crash is inside Triton/Ascend lowering
                                   (libtriton). Report it with the MLIR dump:
                                     MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 \\
                                       python3 -m pytest python/triton_dist/test/ascend/test_gm_addr_args_indices.py
  * 'aclshmem-init' failed      -> crash is below the tutorial: ACLSHMEM / CANN /
                                   torch_npu / HCCL version or ABI mismatch
                                   (check the version matrix printed above).
  * only 'tutorial' failed      -> the ladder works, so the tutorial's own kernel
                                   is the trigger; re-run it with the Triton dump:
                                     MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 \\
                                       torchrun --nproc-per-node=$WORLD \
                                         --master_port=$MASTER_PORT $TUTORIAL
  * anything fails with SIGSEGV -> also try a clean Triton cache:
                                     CLEAR_CACHE=1 bash scripts/triage_ascend_runtime.sh

Full log: $LOG_FILE
EOF
fi