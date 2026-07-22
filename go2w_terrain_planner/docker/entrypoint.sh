#!/usr/bin/env bash
set -euo pipefail

ISAAC_LAB_ROOT="${ISAAC_LAB_ROOT:-/workspace/isaaclab}"
PROJECT_ROOT="${GO2W_PROJECT_ROOT:-/workspace/go2w_terrain_planner}"
COMMAND="${1:-shell}"
shift || true

run_python() {
    exec "${ISAAC_LAB_ROOT}/isaaclab.sh" -p "$@"
}

case "${COMMAND}" in
    unit-test)
        export GO2W_SKIP_TASK_REGISTRATION=1
        run_python -m pytest -q "${PROJECT_ROOT}/tests/unit" "$@"
        ;;
    smoke)
        run_python "${PROJECT_ROOT}/tests/smoke/smoke_env.py" --headless "$@"
        ;;
    train)
        run_python "${PROJECT_ROOT}/scripts/rsl_rl/train.py" \
            --task Go2W-Terrain-Navigation-Direct-v0 --headless "$@"
        ;;
    play)
        run_python "${PROJECT_ROOT}/scripts/rsl_rl/play.py" \
            --task Go2W-Terrain-Navigation-Direct-v0  "$@"
        ;;
    evaluate)
        run_python "${PROJECT_ROOT}/scripts/evaluate.py" \
            --task Go2W-Terrain-Navigation-Direct-v0 --headless "$@"
        ;;
    export)
        run_python "${PROJECT_ROOT}/scripts/export_onnx.py" "$@"
        ;;
    verify-onnx)
        run_python "${PROJECT_ROOT}/scripts/verify_onnx.py" "$@"
        ;;
    shell)
        exec /bin/bash "$@"
        ;;
    *)
        echo "未知命令：${COMMAND}" >&2
        echo "可用命令：unit-test、smoke、train、play、evaluate、export、verify-onnx、shell" >&2
        exit 2
        ;;
esac
