# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import bsuite
import numpy as np
import os

from random import randint
import numpy as np
from collections.abc import Iterable
import gym 
import gymnasium

import bsuite
from gym import spaces
import gym
import dm_env
from dm_env import specs
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
import bsuite
from bsuite.utils import gym_wrapper


from typing import Any, Dict, Optional

from bsuite.environments import base
from bsuite.experiments.discounting_chain import sweep

import dm_env
from dm_env import specs
import numpy as np


class DiscountingChain(base.Environment):
    """Simple diagnostic discounting challenge.
    
    Observation is two pixels: (context, time_to_live)
    
    Context will only be -1 in the first step, then equal to the action selected in
    the first step. For all future decisions the agent is in a "chain" for that
    action. Reward of +1 come  at one of: 1, 3, 10, 30, 100
    
    However, depending on the seed, one of these chains has a 10% bonus.
    """

    def __init__(self, mapping_seed: Optional[int] = None):
        """Builds the Discounting Chain environment.

        Args:
          mapping_seed: Optional integer, specifies which reward is bonus.
        """
        super().__init__()
        self._episode_len = 100
        self._reward_timestep = [1, 3, 10, 30, 100]
        self._n_actions = len(self._reward_timestep)
        if mapping_seed is None:
            self.mapping_seed = np.random.randint(0, self._n_actions)
        else:
            self.mapping_seed = mapping_seed % self._n_actions

        self._rewards = np.ones(self._n_actions)
        self._rewards[mapping_seed] += 0.1

        self._timestep = 0
        self._context = -1

        self.bsuite_num_episodes = 10000 #sweep.NUM_EPISODES

    def _get_observation(self):
        obs = np.zeros(shape=(1, 2), dtype=np.float32)
        obs[0, 0] = self._context
        obs[0, 1] = self._timestep / self._episode_len
        return obs

    def _reset(self) -> dm_env.TimeStep:
        self._timestep = 0
        self._context = -1
        observation = self._get_observation()
        return dm_env.restart(observation)

    def _step(self, action: int) -> dm_env.TimeStep:
        if self._timestep == 0:
            self._context = action

        self._timestep += 1
        if self._timestep == self._reward_timestep[self._context]:
            reward = self._rewards[self._context]
        else:
            reward = 0.0

        observation = self._get_observation()
        if self._timestep == self._episode_len:
            return dm_env.termination(reward=reward, observation=observation)
        return dm_env.transition(reward=reward, observation=observation)

    def observation_spec(self):
        return specs.Array(shape=(1, 2), dtype=np.float32, name="observation")

    def action_spec(self):
        return specs.DiscreteArray(self._n_actions, name="action")

    def _save(self, observation):
        self._raw_observation = (observation * 255).astype(np.uint8)

    @property
    def optimal_return(self):
        # Returns the maximum total reward achievable in an episode.
        return 1.1

    def bsuite_info(self) -> Dict[str, Any]:
        return {}




class MemoryChain(base.Environment):
    """Simple diagnostic memory challenge.

    Observation is given by n+1 pixels: (context, time_to_live).
    
    Context will only be nonzero in the first step, when it will be +1 or -1 iid
    by component. All actions take no effect until time_to_live=0, then the agent
    must repeat the observations that it saw bit-by-bit.
    """
    def __init__(
        self, memory_length: int, num_bits: int = 1, seed: Optional[int] = 1337
    ):
        """Builds the memory chain environment."""
        super(MemoryChain, self).__init__()
        self._memory_length = memory_length
        self._num_bits = num_bits
        self._rng = np.random.RandomState(seed)

        # Contextual information per episode
        self._timestep = 0
        self._context = self._rng.binomial(1, 0.5, num_bits)
        self._query = self._rng.randint(num_bits)

        # Logging info
        self._total_perfect = 0
        self._total_regret = 0
        self._episode_mistakes = 0

        # bsuite experiment length.
        self.bsuite_num_episodes = 10_000  # Overridden by experiment load().

    def _get_observation(self):
        """Observation of form [time, query, num_bits of context]."""
        obs = np.zeros(shape=(1, self._num_bits + 2), dtype=np.float32)
        # Show the time, on every step.
        obs[0, 0] = 1 - self._timestep / self._memory_length
        # Show the query, on the last step
        if self._timestep == self._memory_length - 1:
            obs[0, 1] = self._query
        # Show the context, on the first step
        if self._timestep == 0:
            obs[0, 2:] = 2 * self._context - 1
        return obs

    def _step(self, action: int) -> dm_env.TimeStep:
        observation = self._get_observation()
        self._timestep += 1

        if self._timestep - 1 < self._memory_length:
            # On all but the last step provide a reward of 0.
            return dm_env.transition(reward=0.0, observation=observation)
        if self._timestep - 1 > self._memory_length:
            raise RuntimeError("Invalid state.")  # We shouldn't get here.

        if action == self._context[self._query]:
            reward = 1.0
            self._total_perfect += 1
        else:
            reward = -1.0
            self._total_regret += 2.0
        return dm_env.termination(reward=reward, observation=observation)

    def _reset(self) -> dm_env.TimeStep:
        self._timestep = 0
        self._episode_mistakes = 0
        self._context = self._rng.binomial(1, 0.5, self._num_bits)
        self._query = self._rng.randint(self._num_bits)
        observation = self._get_observation()
        return dm_env.restart(observation)

    def observation_spec(self):
        return specs.Array(
            shape=(1, self._num_bits + 2), dtype=np.float32, name="observation"
        )

    def action_spec(self):
        return specs.DiscreteArray(2, name="action")

    def _save(self, observation):
        self._raw_observation = (observation * 255).astype(np.uint8)

    def bsuite_info(self):
        return dict(total_perfect=self._total_perfect, total_regret=self._total_regret)


class BsuiteWrapper(gym.Env):
    """
    This class wraps bsuite enviroments
    Available Environments: 
    * memory_len/N/K
    * discounting_chain/N
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

        if env_name == 'MemoryLengthEasy':
            #self._env = MemoryChain(env_params[0], env_params[1])
            self._env = MemoryChain(memory_length = 30, num_bits = 1)
            self.max_episode_steps = 31
        elif env_name == 'MemoryLengthMedium':
            #self._env = MemoryChain(env_params[0], env_params[1])
            self._env = MemoryChain(memory_length = 60, num_bits = 1)
            self.max_episode_steps = 61
        elif env_name == 'MemoryLengthHard':
            #self._env = MemoryChain(env_params[0], env_params[1])
            self._env = MemoryChain(memory_length = 90, num_bits = 1)
            self.max_episode_steps = 91
        elif env_name == 'DiscountingChain':
            self._env = DiscountingChain(mapping_seed = None)
            self.max_episode_steps = 100

        # self._env = bsuite.load_from_id(env_name)  # type: dm_env.Environment
        self._env.reset()


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
    def seed(self, seed):
        np.random.seed(seed)


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

        if isinstance(action, Iterable):
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
