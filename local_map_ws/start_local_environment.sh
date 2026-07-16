#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${WORKSPACE_DIR}/.." && pwd)"

source /opt/ros/noetic/setup.bash

if [[ -f "${PROJECT_DIR}/lio_ws/devel/setup.bash" ]]; then
  source "${PROJECT_DIR}/lio_ws/devel/setup.bash"
fi

if [[ ! -f "${WORKSPACE_DIR}/devel/setup.bash" ]]; then
  echo "Local-map workspace is not built. Run: ${WORKSPACE_DIR}/build.sh" >&2
  exit 1
fi

source "${WORKSPACE_DIR}/devel/setup.bash"

exec roslaunch go2w_local_environment local_environment.launch "$@"

