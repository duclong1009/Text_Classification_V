import numpy as np
import random
import os
import torch


def seed_all(seed=1009):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # tf.random.set_seed(seed)
