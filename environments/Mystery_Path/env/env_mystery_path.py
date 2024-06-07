import memory_gym
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
from random import randint


class MultiDiscreteToDiscreteWrapper(gym.ActionWrapper):
    """
    A wrapper that converts a multidiscrete action space to discrete
    by taking the Cartesian product of all possible actions.
    """
    def __init__(self, env):
        super().__init__(env)
        # Get the number of actions for each dimension
        self.nvec = env.action_space.nvec
        # Compute the total number of discrete actions
        self.n = np.prod(self.nvec)
        # Create a discrete action space
        self.action_space = gym.spaces.Discrete(self.n)

    def action(self, action):
        # Convert the discrete action to a multidimensional index
        index = np.unravel_index(action, self.nvec)
        # Return the original action
        return index


class MPEasyWrapper(gym.Wrapper):

    def reset(self, seed = None, return_info = True, options = None):
        options={
                "show_origin": True,
                "show_goal": True,
                "reward_path_progress":0.1,
            }
        obs, info = self.env.reset(seed = seed, return_info = True, options = options)
        return obs, info
    
class FiveToFourWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        state, reward, done, _, info = self.env.step(action)

        return state, reward, done, info

def make_mmgrid():
    # see the section below explaining arguments
    env=gym.make("MysteryPath-Grid-v0")
    
    env = FiveToFourWrapper(env)
    return env

def make_mmgrid_easy():
    # see the section below explaining arguments
    env=gym.make("MysteryPath-Grid-v0")
    env=MPEasyWrapper(env)
    env = FiveToFourWrapper(env)
    return env


def make_mm():
    # see the section below explaining arguments
    env=gym.make("MysteryPath-v0")
    env=MultiDiscreteToDiscreteWrapper(env)
    env = FiveToFourWrapper(env)
    return env

def make_mm_easy():
    # see the section below explaining arguments
    env=gym.make("MysteryPath-v0")
    env=MPEasyWrapper(env)
    env=MultiDiscreteToDiscreteWrapper(env)
    env = FiveToFourWrapper(env)
    return env



# Register the environment with OpenAI Gym
def create_env(env_name: str = "MysteryPath-Grid-v0"):
    if env_name == "MysteryPath-v0":
        return make_mm()
    elif env_name == "MysteryPath-Grid-v0":
        return make_mmgrid()
    elif env_name == "MysteryPath-Easy-v0":
        return make_mm_easy()
    elif env_name == "MysteryPath-Grid-Easy-v0":
        return make_mmgrid_easy()
    elif env_name=='MortarMayhem-v0':
        return MultiDiscreteToDiscreteWrapper(gym.make('MortarMayhem-v0'))
    elif env_name=='MortarMayhem-Grid-v0':
        return gym.make('MortarMayhem-Grid-v0')
    elif env_name=='SearingSpotlights-v0':
        return MultiDiscreteToDiscreteWrapper(gym.make('SearingSpotlights-v0'))
    else:
        raise NotImplementedError("Unknown environment: {}".format(env_name))
