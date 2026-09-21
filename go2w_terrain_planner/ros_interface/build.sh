#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${WORKSPACE_DIR}/../.." && pwd)"

source /opt/ros/noetic/setup.bash
if [[ ! -f "${PROJECT_DIR}/local_map_ws/devel/setup.bash" ]]; then
  echo "local_map_ws is not built: ${PROJECT_DIR}/local_map_ws" >&2
  exit 1
fi
source "${PROJECT_DIR}/local_map_ws/devel/setup.bash" --extend

exec catkin_make -C "${WORKSPACE_DIR}" -DCMAKE_BUILD_TYPE=Release "$@"
