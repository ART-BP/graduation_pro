#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CHECKPOINT.pt [OUTPUT.onnx]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -d "${SOURCE_PACKAGE_DIR}/models" ]]; then
  PACKAGE_DIR="${SOURCE_PACKAGE_DIR}"
elif command -v rospack >/dev/null 2>&1; then
  PACKAGE_DIR="$(rospack find go2w_policy_deployment)"
else
  PACKAGE_DIR=""
fi
if [[ $# -eq 1 && -z "${PACKAGE_DIR}" ]]; then
  echo "Cannot locate go2w_policy_deployment; pass an explicit OUTPUT.onnx" >&2
  exit 1
fi
CHECKPOINT="$(realpath "$1")"
OUTPUT="${2:-${PACKAGE_DIR}/models/planner.onnx}"
OUTPUT_DIR="$(dirname "${OUTPUT}")"
OUTPUT_NAME="$(basename "${OUTPUT}")"
CHECKPOINT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
IMAGE="${GO2W_PLANNER_IMAGE:-go2w-terrain-planner:0.1.0}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(realpath "${OUTPUT_DIR}")"

DOCKER_ARGS=(
  --rm
  -v "${CHECKPOINT_DIR}:/workspace/input:ro"
  -v "${OUTPUT_DIR}:/workspace/output"
)
EXPORT_ARGS=(
  export
  --checkpoint "/workspace/input/${CHECKPOINT_NAME}"
  --output "/workspace/output/${OUTPUT_NAME}"
)
VERIFY_ARGS=(
  verify-onnx
  --model "/workspace/output/${OUTPUT_NAME}"
)
if [[ -n "${GO2W_PROJECT_CONFIG_DIR:-}" ]]; then
  CONFIG_DIR="$(realpath "${GO2W_PROJECT_CONFIG_DIR}")"
  DOCKER_ARGS+=(-v "${CONFIG_DIR}:/workspace/project_configs:ro")
  EXPORT_ARGS+=(--project-config-dir /workspace/project_configs)
  VERIFY_ARGS+=(--project-config-dir /workspace/project_configs)
fi

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${EXPORT_ARGS[@]}"

docker run "${DOCKER_ARGS[@]}" "${IMAGE}" "${VERIFY_ARGS[@]}"

echo "Exported and verified: ${OUTPUT_DIR}/${OUTPUT_NAME}"
