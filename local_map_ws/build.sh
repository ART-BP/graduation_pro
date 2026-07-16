#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${WORKSPACE_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash

if [[ -f "${PROJECT_DIR}/lio_ws/devel/setup.bash" ]]; then
  source "${PROJECT_DIR}/lio_ws/devel/setup.bash"
fi

exec catkin_make -C "${WORKSPACE_DIR}" -DCMAKE_BUILD_TYPE=Release "$@"

