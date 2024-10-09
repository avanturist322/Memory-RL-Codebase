import bsuite
# from bsuite.utils import gym_wrapper

import numpy as np
import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
import numpy as np
from collections.abc import Iterable
import gym 
import gymnasium

import bsuite
# from bsuite.utils import gym_wrapper
from gym import spaces
import gym
import dm_env
from dm_env import specs
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import bsuite


class BsuiteWrapper(gym.Env):
    """
    This class wraps bsuite enviroments
    Available Environments: 
    * memory_len/0
    * discounting_chain/0
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    def __init__(self, env_name, reset_params = None, realtime_mode = False):
        """Instantiates the bsuite environment.
        
        Arguments:
            env_name {string} -- Name of the bsuite environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        Attributes: 
            observation_space_shape {tuple} -- Shape of the observation space
            
        """

        self._env = bsuite.load_from_id(env_name)  # type: dm_env.Environment
        self._env.reset()

        if env_name.split('/')[0] == 'discounting_chain':
            self.max_episode_steps = 100
        elif env_name.split('/')[0] == 'discounting_chain':
            self.max_episode_steps = 100
        else:
            pass

        try:
            self.max_episode_steps = self._env.bsuite_num_episodes
        except Exception as e:
            print(f'ERROR: {e}')
            #TO-DO: define for envs when None!!!
            # max_lenght = 96 # define 
            raise e

        # self._env.seed(self._default_reset_params['start-seed'])

        self._last_observation = None  # type: Optional[np.ndarray]
        self.viewer = None
        self.game_over = False  # Needed for Dopamine agents.

        obs_spec = self._env.observation_spec()  # type: specs.Array
        if isinstance(obs_spec, specs.BoundedArray):
            self.observation_space = spaces.Box(
                low=float(obs_spec.minimum),
                high=float(obs_spec.maximum),
                shape=obs_spec.shape,
                dtype=obs_spec.dtype)
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype)

        self._rewards = []
        self.t = 0

        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        # TO-DO: add another envs
        # if you want to use another popgym enviroment just add desired rows below

        if isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            if len(self.observation_space.shape) > 2:
                self.observation_space.obs_type = 'image'
                self.observation_space.obs_shape = self.observation_space.shape # fix!
            else:
                self.observation_space.obs_type = 'vector'
                self.observation_space.obs_shape = (self.observation_space.shape[-1],)
        elif isinstance(self.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
            # discrete observation
            self.observation_space.obs_shape = (1, )
            self.observation_space.obs_type = 'discrete'
        elif isinstance(self.observation_space, Iterable) and isinstance(self.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
            # multi discrete observation - like discrete vector 
            self.observation_space.obs_shape = (1 for obs_space in env.observation_space )
            self.observation_space.obs_type = 'iterable_discrete'
        elif isinstance(self.observation_space, Iterable) and isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            # Case: list/tuple etc. of image/vector observation is available e.g list of observations from different cameras
            raise NotImplementedError

    # @property
    # def max_episode_steps(self):
    #     """Returns the maximum number of steps that an episode can last."""



    def step(self, action): # Tuple[np.ndarray, float, bool, Dict[str, Any]]
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {list} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """

        if isinstance(action, list): # ???
            if len(action) == 1:
                action = action[0]

        timestep = self._env.step(action)
        if timestep.last():
            self.game_over = True
        self._last_observation = timestep.observation

        reward = timestep.reward or 0.
        obs, terminated, info = timestep.observation, self.game_over, {}
        # print(tuple(reversed(obs.shape)))
        self._rewards.append(reward)

        if isinstance(obs, int):
            obs = np.array([obs, ]) #.reshape((self.observation_space.obs_shape))
        elif isinstance(obs, Iterable):
            obs = obs.flatten() #.reshape((self.observation_space.obs_shape))
        if self.t == self.max_episode_steps - 1:
            terminated = True


        if terminated:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}

        self.t += 1

        return obs, reward, terminated, info

    def reset(self) -> np.ndarray:
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
        """
        self.t = 0
        self._rewards = []
        self.game_over = False

        # Reset the environment to retrieve the initial observation
        timestep = self._env.reset()
        # self._env.seed(self._default_reset_params['start-seed'])

        self._last_observation = timestep.observation
        obs = timestep.observation


        if isinstance(obs, int):
            obs = np.array([obs, ])
        elif isinstance(obs, Iterable):
            obs = obs.flatten()
        return obs

    def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
        if self._last_observation is None:
          raise ValueError('Environment not ready to render. Call reset() first.')

        if mode == 'rgb_array':
          return self._last_observation

        if mode == 'human':
          if self.viewer is None:
            # pylint: disable=import-outside-toplevel
            # pylint: disable=g-import-not-at-top
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
          self.viewer.imshow(self._last_observation)
          return self.viewer.isopen


    @property
    def action_space(self) -> spaces.Discrete:
        action_spec = self._env.action_spec()  # type: specs.DiscreteArray
        return spaces.Discrete(action_spec.num_values)

    @property
    def reward_range(self) -> Tuple[float, float]:
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
          return reward_spec.minimum, reward_spec.maximum
        return -float('inf'), float('inf')
