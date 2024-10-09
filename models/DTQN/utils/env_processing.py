import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
import numpy as np
from typing import Union
import gymnasium

try:
    from gym_gridverse.gym import GymEnvironment
    from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
    from gym_gridverse.outer_env import OuterEnv
    from gym_gridverse.representations.observation_representations import (
        make_observation_representation,
    )
    from gym_gridverse.representations.state_representations import (
        make_state_representation,
    )
except ImportError:
    print(
        f"WARNING: ``gym_gridverse`` is not installed. This means you cannot run an experiment with the `gv_*` domains."
    )
    GymEnvironment = None
# from envs.gv_wrapper import GridVerseWrapper
from environments.POPGym.popgym_env import POPGymWrapper
#from environments.Wind.wind_env import WindEnv
from environments.Bsuite.bsuite_env import BsuiteWrapper
from environments.Passive_T_Maze_Flag.env.env_passive_t_maze_flag import TMazeClassicPassive
from environments.Minigrid_Memory.env.env_minigrid import Minigrid  
from environments.MemoryCards.memory_cards_env import MemoryCards

import os
from enum import Enum
from typing import Tuple


class ObsType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


def get_env_obs_type(obs_space: spaces.Space) -> int:
    if isinstance(
        obs_space, (spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)
    ):
        return ObsType.DISCRETE
    else:
        return ObsType.CONTINUOUS


def make_env(id_or_path: str, config) -> GymEnvironment:
    """Makes a GV gym environment."""
    try:
        if 'MiniGrid' in id_or_path:
            env = gym.make('MemoryCards') # fix this later for skipping 
        else:
            print("Loading using gym.make")
            env = gym.make(id_or_path)

    except gym.error.Error:

        print(f"Environment with id {id_or_path} not found.")
        print("Loading using YAML")

        if id_or_path in ['gv_memory.5x5.yaml', 'gv_memory.7x7.yaml', 'gv_memory.9x9.yaml', 'gv_memory_four_rooms.7x7.yaml']:
            inner_env = factory_env_from_yaml(
                os.path.join(os.getcwd(), "envs", "gridverse", id_or_path)
            )
            state_representation = make_state_representation(
                "default", inner_env.state_space
            )
            observation_representation = make_observation_representation(
                "default", inner_env.observation_space
            )
            outer_env = OuterEnv(
                inner_env,
                state_representation=state_representation,
                observation_representation=observation_representation,
            )
            env = GymEnvironment(outer_env)
            env = TimeLimit(GridVerseWrapper(env), max_episode_steps=500)
        elif id_or_path in ['AutoencodeEasy', 'AutoencodeMedium','AutoencodeHard', 'RepeatPreviousEasy', 'RepeatPreviousMedium', 'RepeatPreviousHard', 'ConcentrationEasy', 'ConcentrationMedium','ConcentrationHard']:
            env = POPGymWrapper(env_name = id_or_path)
        elif 'MemoryLength' in id_or_path:
            env = BsuiteWrapper(id_or_path)
            # print(env.observation_space)
            # print(env.observation_space.obs_type)
        elif id_or_path.split('/')[0] == 'DiscountingChain':
            name, N = id_or_path.split('/')
            env = BsuiteWrapper(name, (int(N), ))
        elif id_or_path in ['Wind']:
            env = WindEnv()
        elif id_or_path in ['Passive_T_Maze_Flag']:
            env = TMazeClassicPassive(episode_length=config['episode_timeout'], 
                            corridor_length=config['corridor_length'], 
                            goal_reward=config['goal_reward'],
                            penalty=-1/(config['episode_timeout']-1))
        elif 'MemoryCards' in id_or_path:
            env = MemoryCards(id_or_path)
        elif 'MiniGrid' in id_or_path:
            env =  Minigrid(id_or_path, config['length'])
        else: 
            raise NotImplementedError

    return env


def get_env_obs_length(env: gym.Env) -> int:
    """Gets the length of the observations in an environment"""
    print(env.observation_space, env.action_space)
    if env.observation_space.obs_type == 'discrete':
        return 1
    elif env.observation_space.obs_type == 'vector':
        return env.observation_space.obs_shape[0] 
    elif env.observation_space.obs_type == 'multidiscrete':
        return len(env.observation_space.nvec) # +1
    elif env.observation_space.obs_type == 'image':
        return env.observation_space.shape
    else: 
        raise NotImplementedError(f"We do not yet support {env.observation_space.obs_type}")




    # if isinstance(env.observation_space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
    #     return 1
    # elif isinstance(env.observation_space, (gym.spaces.MultiDiscrete, gym.spaces.Box)):
    #     if len(env.observation_space.shape) != 1:
    #         raise NotImplementedError(f"We do not yet support 2D observation spaces")
    #     return env.observation_space.shape[0]
    # elif isinstance(env.observation_space, spaces.MultiBinary):
    #     return env.observation_space.n
    # else:
    #     raise NotImplementedError(f"We do not yet support {env.observation_space}")


def get_env_obs_mask(env: gym.Env) -> Union[int, np.ndarray]:
    """Gets the number of observations possible (for discrete case).
    For continuous case, please edit the -5 to something lower than
    lowest possible observation (while still being finite) so the
    network knows it is padding.
    """

    if env.observation_space.obs_type == 'discrete':
        return env.observation_space.n
    elif env.observation_space.obs_type == 'vector':
        return -5
    elif env.observation_space.obs_type == 'image':
        return 0 # -1 maybe....
    elif env.observation_space.obs_type == 'multidiscrete':
        #print(env.observation_space.nvec)
        return max(env.observation_space.nvec) + 1
    else: 
        raise NotImplementedError(f"We do not yet support {env.observation_space}")





    # if isinstance(env.observation_space, (gym.spaces.Discrete, gymnasium.spaces.Discrete)):
    #     return env.observation_space.n
    # elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
    #     return env.observation_space.nvec + 1
    # elif isinstance(env.observation_space, gym.spaces.Box):
    #     # If you would like to use DTQN with a continuous action space, make sure this value is
    #     #       below the minimum possible observation. Otherwise it will appear as a real observation
    #     #       to the network which may cause issues. In our case, Car Flag has min of -1 so this is
    #     #       fine.
    #     return -5
    # else:
    #     raise NotImplementedError(f"We do not yet support {env.observation_space}")


def add_to_history(history, obs) -> np.ndarray:
    """Append an observation to the history, and removes the first observation."""
    if isinstance(obs, np.ndarray):
        return np.concatenate((history[1:], [obs.copy()]))
    elif isinstance(obs, (int, float)):
        return np.concatenate((history[1:], [[obs]]))
    # hack to get numpy values
    elif hasattr(obs, "dtype"):
        return np.concatenate((history[1:], [[obs]]))
    else:
        raise ValueError(f"Tried to add to {history} with {obs}, type {obs.dtype}")


def make_empty_contexts(
    context_len: int,
    env_obs_length: int,
    obs_type: type,
    obs_mask: int,
    num_actions: int,
    reward_mask: float,
    done_mask: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(env_obs_length, tuple):
        obs_context = np.full([context_len, *env_obs_length], obs_mask, dtype=obs_type)
        next_obs_context = np.full([context_len, *env_obs_length], obs_mask, dtype=obs_type)
    else:
        obs_context = np.full([context_len, env_obs_length], obs_mask, dtype=obs_type)
        next_obs_context = np.full([context_len, env_obs_length], obs_mask, dtype=obs_type)


    action_context = np.full(
        [context_len, 1], np.random.randint(num_actions), dtype=np.int_
    )
    reward_context = np.full([context_len, 1], reward_mask, dtype=np.float32)
    done_context = np.full([context_len, 1], done_mask, dtype=np.bool_)
    return obs_context, next_obs_context, action_context, reward_context, done_context
