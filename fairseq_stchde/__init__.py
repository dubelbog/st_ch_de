# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from fairseq_stchde.X import Y`
from fairseq_stchde.distributed import utils as distributed_utils
from fairseq_stchde.logging import meters, metrics, progress_bar  # noqa

sys.modules["fairseq_stchde.distributed_utils"] = distributed_utils
sys.modules["fairseq_stchde.meters"] = meters
sys.modules["fairseq_stchde.metrics"] = metrics
sys.modules["fairseq_stchde.progress_bar"] = progress_bar

# initialize hydra
from fairseq_stchde.dataclass.initialize import hydra_init
hydra_init()

import fairseq_stchde.criterions  # noqa
import fairseq_stchde.distributed  # noqa
import fairseq_stchde.models  # noqa
import fairseq_stchde.modules  # noqa
import fairseq_stchde.optim  # noqa
import fairseq_stchde.optim.lr_scheduler  # noqa
import fairseq_stchde.pdb  # noqa
import fairseq_stchde.scoring  # noqa
import fairseq_stchde.tasks  # noqa
import fairseq_stchde.token_generation_constraints  # noqa

import fairseq_stchde.benchmark  # noqa
import fairseq_stchde.model_parallel  # noqa
