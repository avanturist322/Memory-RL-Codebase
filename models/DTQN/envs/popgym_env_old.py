import numpy as np
import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
import numpy as np
from collections.abc import Iterable
import gym 
import gymnasium

import popgym
from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction

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

        # TO-DO: add another envs
        # if you want to use another popgym enviroment just add desired rows below
        if env_name == 'AutoencodeEasy':
            self._env = popgym.envs.autoencode.AutoencodeEasy()
        elif env_name == 'AutoencodeMedium':
            self._env = popgym.envs.autoencode.AutoencodeMedium()
        elif env_name == 'AutoencodeHard':
            self._env = popgym.envs.autoencode.AutoencodeHard()


        elif env_name == 'RepeatPreviousEasy':
            self._env = popgym.envs.repeat_previous.RepeatPreviousEasy()
        elif env_name == 'RepeatPreviousMedium':
            self._env = popgym.envs.repeat_previous.RepeatPreviousMedium()
        elif env_name == 'RepeatPreviousHard':
            self._env = popgym.envs.repeat_previous.RepeatPreviousHard()

        elif env_name == 'ConcentrationEasy':
            self._env = popgym.envs.concentration.ConcentrationEasy()
        elif env_name == 'ConcentrationMedium':
            self._env = popgym.envs.concentration.ConcentrationMedium()
        elif env_name == 'ConcentrationHard':
            self._env = popgym.envs.concentration.ConcentrationHard()
        else:
            print('ERROR: prease select correct enviroment')
            raise NotImplementedError

        if isinstance(self._env.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            self._env.observation_space.obs_shape = self.observation_space.shape 
            if len(self._env.observation_space.obs_shape) > 1:
                self._env.observation_space.obs_type = 'image'
            else:
                self._env.observation_space.obs_type = 'vector'
        elif isinstance(self._env.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
            # discrete observation
            self._env.observation_space.obs_shape = (1, )
            self._env.observation_space.obs_type = 'discrete'
        elif isinstance(self._env.observation_space, Iterable) and all(isinstance(space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)) for space in self._env.observation_space.spaces):
            # multi discrete observation - like discrete vector 
            self._env.observation_space.obs_shape = tuple([1 for obs_space in self._env.observation_space.spaces])
            self._env.observation_space.obs_type = 'iterable_discrete'
        # elif isinstance(self._env.observation_space, Iterable) and isinstance(self._env.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
        #     # Case: list/tuple etc. of image/vector observation is available e.g list of observations from different cameras
        #     raise NotImplementedError
        else:
            raise NotImplementedError


        self._env.reset()
        try:
            self.max_episode_steps = self._env.max_episode_length
        except Exception as e:
            print(f'ERROR: {e}')
            #TO-DO: define for envs when None!!!
            self.max_episode_steps = 96 # define 

    @property
    def observation_space(self):
        """Returns the shape of the observation space of the agent."""
        return self._env.observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    def seed(self, seed):
        """Returns the shape of the action space of the agent."""
        return self._env.reset(seed = seed)

    def reset(self, reset_params = None):
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
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

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
        if isinstance(action, list):
            if len(action) == 1:
                action = action[0]
        obs, reward, terminated, truncated, info  = self._env.step(action)
        self._rewards.append(reward)



        if isinstance(obs, int):
            obs = np.array([obs, ])
        elif isinstance(obs, Iterable):
            obs = np.array(obs)

        if self.t == self.max_episode_steps - 1:
            terminated = True


        if terminated:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}

        self.t += 1

        return obs, reward, terminated, info
    
    def render(self):
        """Renders the environment."""
        self._env.render()

    def close(self):
        """Shuts down the environment."""
        self._env.close()
