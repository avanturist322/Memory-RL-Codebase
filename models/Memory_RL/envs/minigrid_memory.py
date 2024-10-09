import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *
from collections.abc import Iterable


class Minigrid:
    def __init__(self, name, length = 31):
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
            self.episode_length = 96
            self._action_space = spaces.Discrete(3)
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)
        else:
            view_size = 7
            self.tile_size = 8
            hw = view_size * self.tile_size
            self.max_episode_steps = 64
            self.episode_length = 64
            self._action_space = self._env.action_space
            self._env = ViewSizeWrapper(self._env, view_size)
            self._env = RGBImgPartialObsWrapper(self._env, tile_size = self.tile_size)

        self._env = ImgObsWrapper(self._env)

        # self.observation_space = spaces.Box(
        #         low = 0,
        #         high = 1.0,
        #         shape = (3, hw, hw),
        #         dtype = np.float32)

        self.observation_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (np.array((3, hw, hw)).prod(), ), # flatten
            dtype = np.uint8)

        self.image_space = spaces.Box(
            low = 0,
            high = 255,
            shape = (3, hw, hw), # flatten
            dtype = np.uint8)


        self.observation_space.obs_shape = self.observation_space.shape 
        self.observation_space.obs_type = 'image'

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
        obs = obs.astype(np.uint8) 

        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        return obs.flatten() 

    def seed(self, seed):
        self._env.seed(np.random.randint(0, 9999) if seed is None else seed)

    def step(self, action):

        if isinstance(action, Iterable):
            if len(action) == 1:
                action = action[0]



        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        obs = obs.astype(np.uint8)

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = {}
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        self.t += 1
        return obs.flatten(), reward, done, info

    def render(self):
        img = self._env.render(tile_size = 96)
        plt.close()
        # time.sleep(0.5)
        return img

    def close(self):
        self._env.close()


# class SingleActionWrapper(Minigrid):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env

#     def step(self, action):
#         return self.env.step((action, ))
    
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
        env = Minigrid(config["name"], length)
        # env = SingleActionWrapper(env)
    # if 'minigrid' in config['type'].lower():
        return env
    # if config["type"] in ["SearingSpotlights", "MortarMayhem", "MortarMayhem-Grid", "MysteryPath", "MysteryPath-Grid"]:
    #     return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode=render)

