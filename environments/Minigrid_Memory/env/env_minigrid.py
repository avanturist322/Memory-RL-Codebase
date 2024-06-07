import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *


class Minigrid:
    def __init__(self, name, length):
        self._env = gym.make(name)

        self._env.height = length
        self._env.width = length
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        if "Memory" in name:
            view_size = 3
            self.tile_size = 28
            hw = view_size * self.tile_size
            self.max_episode_steps = 96
            self._action_space = spaces.Discrete(3)
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)
        else:
            view_size = 7
            self.tile_size = 8
            hw = view_size * self.tile_size
            self.max_episode_steps = 64
            self._action_space = self._env.action_space
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)

        self._env = ImgObsWrapper(self._env)

        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, hw, hw),
                dtype = np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return self._action_space
    
    def reset(self, seed=None):
        self._env.seed(np.random.randint(0, 9999) if seed is None else seed)
        self.t = 0
        self._rewards = []
        obs = self._env.reset()
        obs = obs.astype(np.float32) / 255.

        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        obs = obs.astype(np.float32) / 255.

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        self.t += 1
        return obs, reward, done, info

    def render(self):
        img = self._env.render(tile_size = 96)
        plt.close()
        # time.sleep(0.5)
        return img

    def close(self):
        self._env.close()


def create_env(config:dict, length, render:bool=False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        env_name {str}: Name of the to be instantiated environment
        render {bool}: Whether to instantiate the environment in render mode. (default: {False})

    Returns:
        {env}: Returns the selected environment instance.
    """
    # if config["type"] == "PocMemoryEnv":
    #     return PocMemoryEnv(glob=False, freeze=True, max_episode_steps=32)
    # if config["type"] == "CartPole":
    #     return CartPole(mask_velocity=False)
    # if config["type"] == "CartPoleMasked":
    #     return CartPole(mask_velocity=True)
    if config["type"] == "Minigrid":
    # if 'minigrid' in config['type'].lower():
        return Minigrid(config["name"], length)
    # if config["type"] in ["SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"]:
    #     return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode=render)

