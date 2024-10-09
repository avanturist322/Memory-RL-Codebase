import torch
import numpy as np
import random
import os
import gym
from typing import Optional


def set_global_seed(
    seed: int, env: gym.Env, eval_env: Optional[gym.Env] = None
) -> None:
    """Sets seed for PyTorch, NumPy, and random.

    Args:
        seed: The random seed to use.
        env: The gym environment to seed.
        eval_env: Optional. the evaluation gym environment to seed.
    """
    random.seed(seed)
    tseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    np.random.seed(npseed)
    env.seed(seed=seed)
    if eval_env is not None:
        eval_env.seed(seed=seed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)
