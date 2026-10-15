"""BioMGL training entry.

The complete training code will be released after the paper is accepted.
"""

from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import torch


def set_seed(seed, cuda=False):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

