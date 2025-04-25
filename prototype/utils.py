import torch
import random
import os
import numpy as np


def seed_everything(seed: int):
    # python random, np and torch seed no longer needed, bc we use generator objects (less side effects)
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # keep for reproducibility on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
