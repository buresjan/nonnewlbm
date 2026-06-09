#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../solver/sim_tcpc"
make "$@"
