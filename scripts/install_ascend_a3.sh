#!/usr/bin/env bash
# A3 entry point; reuse the tested A2 installer and let CANN detect the exact SoC.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash scripts/install_ascend_a3.sh [--help|--prereqs-only|--tests-only]
Install and test on Ascend A3 (Ascend910_93 / 910C), with CANN already installed.
Full suite: >=4 visible NPUs; --prereqs-only: >=2 visible NPUs.
Default WORK_DIR: <repository>/.ascend-a3 (separate from the A2 installation).
Uses the same pinned toolchain, SHMEM and tests as install_ascend_910b.sh.
Device and compiler target are detected automatically and checked for A3.
No A3-specific torch/CANN version is guessed; supply compatible versions/wheels.
All other environment variables are documented by:
  bash scripts/install_ascend_910b.sh --help
HELP
    exit 0
fi
export WORK_DIR=${WORK_DIR:-"${SCRIPT_DIR}/../.ascend-a3"}
export TRITON_DIST_EXPECTED_PLATFORM=A3
exec bash "$SCRIPT_DIR/install_ascend_910b.sh" "$@"
