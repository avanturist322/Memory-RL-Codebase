import numpy as np
import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
import numpy as np
from collections.abc import Iterable
import gym 
import gymnasium

import popgym
from environments.POPGym.repeat_previous import RepeatPreviousEasy, RepeatPreviousMedium, RepeatPreviousHard
from environments.POPGym.concentration import ConcentrationEasy, ConcentrationMedium, ConcentrationHard
from environments.POPGym.autoencode import AutoencodeEasy, AutoencodeMedium, AutoencodeHard

# from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction

class POPGymWrapper():
    """
    This class wraps POPGym discrete enviroments
    Available Environments:

    * RepeatPrevious
    * Autoencode
    * Concentration
    """
    def __init__(self, env_name, reset_params = None, realtime_mode = False) -> None:
        """Instantiates the POPGym environment.
        
        Arguments:
            env_name {string} -- Name of the POPGym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        Attributes: 
            observation_space_shape {tuple} -- Shape of the observation space
            
        """

        self._rewards = []
        self.t = 0

        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        # TO-DO: add other envs
        # if you want to use other popgym enviroment just add desired rows below
        if env_name == 'AutoencodeEasy':
            self._env = AutoencodeEasy()
            self.max_episode_steps = self._env.max_episode_length

        elif env_name == 'AutoencodeMedium':
            self._env = AutoencodeMedium()
            self.max_episode_steps = self._env.max_episode_length

        elif env_name == 'AutoencodeHard':
            self._env = AutoencodeHard()
            self.max_episode_steps = self._env.max_episode_length

        elif env_name == 'RepeatPreviousEasy':
            self._env = RepeatPreviousEasy()
            self.max_episode_steps = self._env.max_episode_length


        elif env_name == 'RepeatPreviousMedium':
            self._env = RepeatPreviousMedium()
            self.max_episode_steps = self._env.max_episode_length

        elif env_name == 'RepeatPreviousHard':
            self._env = RepeatPreviousHard()
            self.max_episode_steps = self._env.max_episode_length


        elif env_name == 'ConcentrationEasy':
            self._env = ConcentrationEasy()
            self.max_episode_steps = 104
        elif env_name == 'ConcentrationMedium':
            self._env = ConcentrationMedium()
            self.max_episode_steps = 208
        elif env_name == 'ConcentrationHard':
            self._env = ConcentrationHard()
            self.max_episode_steps = 104
        else:
            print('ERROR: prease select correct enviroment')
            raise NotImplementedError

        self.observation_space = self._env.observation_space



        if isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            self.observation_space.obs_shape = self.observation_space.shape 
            if len(self._env.observation_space.obs_shape) > 1:
                self.observation_space.obs_type = 'image'
            else:
                self.observation_space.obs_type = 'vector'
        elif isinstance(self.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
            # discrete observation
            self.observation_space.obs_shape = (1, )
            self.observation_space.obs_type = 'discrete'
        elif isinstance(self.observation_space, Iterable) and all(isinstance(space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)) for space in self._env.observation_space.spaces):
            # multi discrete observation - like discrete vector 
            self.observation_space.obs_shape = (len(self.observation_space.spaces),)#tuple([1 for obs_space in self._env.observation_space.spaces])
            self.observation_space.obs_type = 'multidiscrete' 
            self.observation_space.nvec = [env.n for env in self._env.observation_space.spaces]

        elif isinstance(self.observation_space, (gymnasium.spaces.MultiDiscrete, gym.spaces.MultiDiscrete)):
            # multi discrete observation - like discrete vector 
            self.observation_space.obs_shape = self.observation_space.shape 
            self.observation_space.obs_type = 'multidiscrete'
        else:
            raise NotImplementedError


    # @property
    # def observation_space(self):
    #     """Returns the shape of the observation space of the agent."""
    #     return self._env.observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    def reset(self, seed = None, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
        """
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        self.t = 0
        self._rewards = []

        # Sample seed
        if seed is None:
            self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)
        else:
            self._seed = seed

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Reset the environment to retrieve the initial observation
        obs, _ = self._env.reset(seed=self._seed, options=options)
        if isinstance(obs, int):
            obs = np.array([obs, ])
        elif isinstance(obs, Iterable):
            obs = np.array(obs)

        return obs

    def step(self, action):
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
        obs, reward, terminated, truncated, info  = self._env.step(action)
        self._rewards.append(reward)


        if isinstance(obs, int):
            obs = np.array([obs, ])
        elif isinstance(obs, Iterable):
            obs = np.array(obs)

        # if self.t == self.max_episode_steps:
        #     terminated = True


        if terminated:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}

        self.t += 1

        return obs, reward, terminated, info

    def seed(self, seed):
        """Returns the shape of the action space of the agent."""
        return self._env.reset(seed = seed)
    
    def render(self):
        """Renders the environment."""
        self._env.render()

    def close(self):
        """Shuts down the environment."""
        self._env.close()
