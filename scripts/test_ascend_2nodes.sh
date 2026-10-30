#!/usr/bin/env bash
# Launch the Triton-distributed two-node test, one NPU on each 910B host.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
    cat <<'HELP'
Usage: bash scripts/test_ascend_2nodes.sh NODE_RANK MASTER_IP [test options]
Run on both hosts: NODE_RANK is 0 on the first host, 1 on the second.
MASTER_IP is the first host's reachable IPv4 address on both hosts.
Environment:
  WORK_DIR   Installation directory (default: <repository>/.ascend-910b)
  ASCEND_RT_VISIBLE_DEVICES  One physical NPU to use (default: 0)
  MASTER_PORT  Torchrun bootstrap port (default: 29501)
  SHMEM_PORT   SHMEM bootstrap port (default: 8765)
  RUN_TIMEOUT  Total timeout in seconds (default: 600)
Test options: --elements N --iterations N --timeout SECONDS
Requires an installed Ascend backend and RDMA-enabled SHMEM on both hosts.
HELP
}
if [[ ${1:-} == --help ]]; then usage; exit 0; fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
node_rank=$1
master_ip=$2
shift 2
[[ $node_rank == 0 || $node_rank == 1 ]] || { echo 'NODE_RANK must be 0 or 1' >&2; exit 2; }
[[ $master_ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo 'MASTER_IP must be an IPv4 address' >&2; exit 2; }
IFS=. read -r -a octets <<< "$master_ip"
for octet in "${octets[@]}"; do
    (( 10#$octet <= 255 )) || { echo 'Invalid IPv4 address' >&2; exit 2; }
done
[[ ${octets[0]} != 127 && $master_ip != 0.0.0.0 ]] || { echo 'Use a reachable host IP, not loopback/unspecified' >&2; exit 2; }
WORK_DIR=${WORK_DIR:-"$ROOT/.ascend-910b"}
MASTER_PORT=${MASTER_PORT:-29501}
SHMEM_PORT=${SHMEM_PORT:-8765}
RUN_TIMEOUT=${RUN_TIMEOUT:-600}
for port in "$MASTER_PORT" "$SHMEM_PORT"; do
    [[ $port =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo 'Ports must be 1..65535' >&2; exit 2; }
done
[[ $MASTER_PORT != "$SHMEM_PORT" ]] || { echo 'MASTER_PORT and SHMEM_PORT must differ' >&2; exit 2; }
[[ $RUN_TIMEOUT =~ ^[1-9][0-9]*$ ]] || { echo 'RUN_TIMEOUT must be positive seconds' >&2; exit 2; }
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
[[ $ASCEND_RT_VISIBLE_DEVICES =~ ^[0-9]+$ ]] || { echo 'Select exactly one NPU with ASCEND_RT_VISIBLE_DEVICES' >&2; exit 2; }
[[ -f $WORK_DIR/env.sh ]] || { echo "Installation environment missing: $WORK_DIR/env.sh" >&2; exit 1; }
WORK_DIR=$(cd "$WORK_DIR" && pwd)
# CANN vendor scripts may reference unset variables.
set +u
source "$WORK_DIR/env.sh"
set -u
command -v python >/dev/null
command -v timeout >/dev/null
cd "$ROOT"
mkdir -p "$WORK_DIR/logs"
log="$WORK_DIR/logs/test-2nodes-rank${node_rank}-$(date -u +%Y%m%dT%H%M%SZ).log"
printf 'Triton-distributed two-node test: node=%s master=%s:%s SHMEM=%s NPU=%s\nLog: %s\n' \
    "$node_rank" "$master_ip" "$MASTER_PORT" "$SHMEM_PORT" "$ASCEND_RT_VISIBLE_DEVICES" "$log"
timeout --kill-after=15s "$RUN_TIMEOUT" python -u -m torch.distributed.run \
    --nnodes=2 --node-rank="$node_rank" --nproc-per-node=1 \
    --rdzv-backend=static --master-addr="$master_ip" --master-port="$MASTER_PORT" \
    "$ROOT/scripts/test_ascend_2nodes.py" --shmem-port="$SHMEM_PORT" "$@" 2>&1 | tee "$log"
