# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning-based local terrain planner for Unitree Go2W."""

import importlib.util
import os

# Pure Python unit tests can import preprocessing/model modules without
# launching or installing Isaac Sim. Environments register only when the
# Isaac Lab task package is available.
if (
    os.environ.get("GO2W_SKIP_TASK_REGISTRATION") != "1"
    and importlib.util.find_spec("isaaclab_tasks") is not None
):
    import os

# Training/play/evaluation require task registration, while standalone
# model export and ONNX verification must not import Isaac Sim modules.
if os.environ.get(
    "GO2W_SKIP_TASK_IMPORT",
    "0",
).strip().lower() not in {"1", "true", "yes"}:
    from .tasks import *

