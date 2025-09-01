# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import sys

from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.utilities.registry import _register_classes
from lightning.pytorch.accelerators import AcceleratorRegistry

from backend.qaic_accelerator import QAICAccelerator

_register_classes(
    AcceleratorRegistry, "register_accelerators", sys.modules[__name__], Accelerator
)
