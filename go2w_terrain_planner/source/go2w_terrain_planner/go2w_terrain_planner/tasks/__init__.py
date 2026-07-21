# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

import importlib.util
import os

##
# Register Gym environments.
##

if (
    os.environ.get("GO2W_SKIP_TASK_REGISTRATION") != "1"
    and importlib.util.find_spec("isaaclab_tasks") is not None
):
    from isaaclab_tasks.utils import import_packages

    _BLACKLIST_PKGS = ["utils", ".mdp"]
    import_packages(__name__, _BLACKLIST_PKGS)
