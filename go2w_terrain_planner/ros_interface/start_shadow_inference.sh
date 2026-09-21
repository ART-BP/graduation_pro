#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${WORKSPACE_DIR}/../.." && pwd)"
ONNX_PATH="${GO2W_ONNX_PATH:-${PROJECT_DIR}/data/export/planner.onnx}"
ENGINE_PATH="${GO2W_ENGINE_PATH:-${PROJECT_DIR}/data/export/planner_fp32.engine}"

source /opt/ros/noetic/setup.bash
source "${PROJECT_DIR}/local_map_ws/devel/setup.bash" --extend
if [[ ! -f "${WORKSPACE_DIR}/devel/setup.bash" ]]; then
  echo "ros_interface is not built. Run: ${WORKSPACE_DIR}/build.sh" >&2
  exit 1
fi
source "${WORKSPACE_DIR}/devel/setup.bash" --extend

if [[ ! -s "${ENGINE_PATH}" && ! -s "${ONNX_PATH}" ]]; then
  echo "No TensorRT engine or ONNX model was found." >&2
  echo "Expected engine: ${ENGINE_PATH}" >&2
  echo "Expected ONNX:   ${ONNX_PATH}" >&2
  exit 1
fi

exec roslaunch go2w_terrain_planner_ros shadow_inference.launch \
  onnx_path:="${ONNX_PATH}" engine_path:="${ENGINE_PATH}" "$@"
