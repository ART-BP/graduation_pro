#!/usr/bin/env bash

set -euo pipefail

ISAAC_LAB_ROOT="/workspace/isaaclab"
PROJECT_ROOT="/workspace/go2w_terrain_planner"

COMMAND="${1:-shell}"
shift || true

case "${COMMAND}" in
    unit-test)
        exec "${ISAAC_LAB_ROOT}/isaaclab.sh" \
            -p -m pytest \
            -q "${PROJECT_ROOT}/tests/unit" \
            "$@"
        ;;

    smoke)
        exec "${ISAAC_LAB_ROOT}/isaaclab.sh" \
            -p "${PROJECT_ROOT}/scripts/smoke_isaac.py" \
            --headless \
            "$@"
        ;;

    train-official)
        cd "${ISAAC_LAB_ROOT}"
        exec ./isaaclab.sh \
            -p scripts/reinforcement_learning/rsl_rl/train.py \
            --headless \
            "$@"
        ;;

    play-official)
        cd "${ISAAC_LAB_ROOT}"
        exec ./isaaclab.sh \
            -p scripts/reinforcement_learning/rsl_rl/play.py \
            --headless \
            "$@"
        ;;

    shell)
        exec /bin/bash "$@"
        ;;

    *)
        echo "未知命令：${COMMAND}" >&2
        echo "可用命令：unit-test、smoke、train-official、play-official、shell" >&2
        exit 2
        ;;
esac