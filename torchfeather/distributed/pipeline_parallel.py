import copy
import math
from collections.abc import Callable

import torch
from loguru import logger
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    ScheduleZBVZeroBubble,
    _PipelineSchedule,
    get_schedule_class,
)

from torchfeather.components.loss import LossFunction
from torchfeather.config import JobConfig
from torchfeather.distributed import ParallelDims

__all__ = [
    "build_pipeline_schedule",
    "generate_llm_fqn_per_model_part",
    "pipeline_llm",
    "pipeline_module_split",
]
