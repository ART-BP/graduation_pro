#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/noetic/setup.bash

if [[ ! -f "${WORKSPACE_DIR}/devel/setup.bash" ]]; then
  echo "FAST-LIO workspace is not built. Run: catkin_make -C ${WORKSPACE_DIR}" >&2
  exit 1
fi

source "${WORKSPACE_DIR}/devel/setup.bash"

exec roslaunch fast_lio mapping_xt16.launch "$@"
