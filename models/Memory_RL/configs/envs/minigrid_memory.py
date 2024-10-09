from ml_collections import ConfigDict
from typing import Tuple
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal
from envs.key_to_door import visual_match


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = 'MiniGrid-MemoryS13-v0'
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "MiniGrid-Memory"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 50   
    config.save_interval = 50
    config.eval_episodes = 50

    config.env_name = 9999999

    return config
