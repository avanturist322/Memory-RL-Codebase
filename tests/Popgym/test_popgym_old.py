# import popgym
# from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction
# env = popgym.envs.position_only_cartpole.RepeatPreviousEasy()
# print(env.reset(seed=0))
# wrapped = DiscreteAction(Flatten(PreviousAction(env))) # Append prev action to obs, flatten obs/action spaces, then map the multidiscrete action space to a single discrete action for Q learning
# print(wrapped.reset(seed=0))

import popgym
import gymnasium
import gym
from popgym.wrappers import PreviousAction, Antialias, Markovian, Flatten, DiscreteAction
from popgym.core.observability import Observability, STATE

from torch import nn
import torch
import numpy as np
from gym import spaces
import popgym
from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction



# env_classes = popgym.envs.ALL.keys()
# print(env_classes)

env = popgym.envs.repeat_previous.RepeatPreviousEasy()
#env = popgym.envs.autoencode.AutoencodeEasy()
#env = popgym.envs.concentration.ConcentrationEasy()
num_steps = 1

# print(env.observation_space)
# try:
#     print(env.max_episode_length)
# except Exception as e:
#     print(e)

# env.observation_space.shape = None

# print(env.reset(seed=0))
# for i in range(num_steps):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(obs)
#     print(type(obs))
#     print(np.array(np.array(np.array([1, 2, 3]))))
    # wrapped = DiscreteAction(Flatten(PreviousAction(env))) # Append prev action to obs, flatten obs/action spaces, then map the multidiscrete action space to a single discrete action for Q learning
    # print(wrapped.reset(seed=0))
    # print(env.action_space.sample())

# print(isinstance((), list))

# print(env.observation_space)
# print(env.observation_space.__dict__)
# print(type(env.observation_space))
# print(env.observation_space.shape)
# # print(isinstance(type(spaces.discrete.Discrete(n =2 )), env.observation_space.shape))
# # print(np.array([1,]))
# #print('image' if len(env.observation_space.shape) > 1 else 'vector')

# # print(len([None]))
# print(isinstance(env.observation_space, (gym.spaces.box.Box, gymnasium.spaces.discrete.Discrete)))
# print(gymnasium.spaces.discrete.Discrete)
# print(gym.spaces.box.Box == gym.spaces.Box)
# print(gym.spaces.Discrete)
# # print(type(env))


# for obs_space in env.observation_space:
#     print(obs_space.n)

# print(sum([obs_space.n for obs_space in env.observation_space ]))

# def _calc_grad_norm(*modules):
#     """Computes the norm of the gradients of the given modules.
#     Arguments:
#         modules {list} -- List of modules to compute the norm of the gradients of.
#     Returns:
#         {float} -- Norm of the gradients of the given modules. 
#     """
#     grads = []
#     for module in modules:
#         for name, parameter in module.named_parameters():
#             grads.append(parameter.grad.view(-1))
#     return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None




lin_hidden = nn.Linear(2, 10)
out = lin_hidden(torch.tensor(np.array([0, 2])).float())
print(out, out.shape)


# # print(lin_hidden(torch.tensor([1, 2]).float()))
# # loss = sum(lin_hidden(torch.tensor([1, 2]).float()))

lin_hidden = nn.Sequential(nn.Embedding(3, 10), nn.Flatten(0, 1))
out = lin_hidden(torch.tensor([0]))
print(out, out.shape)

# loss = lin_hidden(torch.tensor([1])).sum()
# print(lin_hidden(torch.tensor([1])))


# print(loss)
# loss.backward()
# print(_calc_grad_norm(lin_hidden))


import numpy as np
import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
import numpy as np
from collections.abc import Iterable


import popgym
from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction



        # if isinstance(observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
        #     if len(self.observation_space.shape) > 1:
        #         self.observation_type = 'image'
        #     else:
        #         self.observation_type = 'vector'
        # elif isinstance(observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
        #     self.observation_type = 'discrete'
        # elif isinstance(observation_space, Iterable) and isinstance(observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
        #     self.observation_type = 'iterable_discrete'
        # elif isinstance(observation_space, Iterable) and isinstance(observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
        #     # Case: list/tuple etc. of image/vector observation is available
        #     raise NotImplementedError

# class POPGym_space

class obs_Box():
    pass

class obs_Discrete():
    pass

class POPGymWrapper():
    """
    This class wraps POPGym discrete enviroments
    Available Environments:

    * RepeatPrevious
    * Autoencode
    * Concentration
    """
    def __init__(self, env_name, reset_params = None, realtime_mode = False) -> None:
        """Instantiates the memory-gym environment.
        
        Arguments:
            env_name {string} -- Name of the memory-gym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        Attributes: 
            observation_space_shape {tuple} -- Shape of the observation space
            
        """

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
        elif env_name == 'RepeatPreviousEasy':
            self._env = popgym.envs.repeat_previous.RepeatPreviousHard()

        elif env_name == 'ConcentrationHard':
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
            self._env.observation_space.obs_shape = (1 for obs_space in self._env.observation_space.spaces)
            self._env.observation_space.obs_type = 'iterable_discrete'
        # elif isinstance(self._env.observation_space, Iterable) and isinstance(self._env.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
        #     # Case: list/tuple etc. of image/vector observation is available e.g list of observations from different cameras
        #     raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def observation_space(self):
        """Returns the shape of the observation space of the agent."""
        return self._env.observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        self._env.reset()

        try:
            max_lenght = self._env.max_episode_length
        except Exception as e:
            print(f'ERROR: {e}')
            max_lenght = -1 # for None max lenght
        return max_lenght

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

        if isinstance(obs, int):
            obs = np.array([obs, ])
        elif isinstance(obs, Iterable):
            obs = np.array(obs)

        return obs, reward, terminated, info
    
    def render(self):
        """Renders the environment."""
        self._env.render()

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    # def __init__(self, env_name, reset_params = None, realtime_mode = False) -> None:

env = POPGymWrapper(env_name = 'RepeatPreviousEasy') # AutoencodeEasy
print(env.reset())
for i in range(3):
    action = env.action_space.sample()
    obs, reward, terminated, info = env.step(action)
    print(obs, action)
    print(type(obs))
    print(obs.shape)


# a = np.array([1, 3, 2, 2, 0, 0, 3, 1, 3, 1, 1, 3, 1, 3, 2, 1])
# print(a)
# print(torch.tensor(a))
print(env.observation_space)

# print(isinstance(env.observation_space, Iterable) and isinstance(env.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)))
# print(isinstance(env.observation_space, Iterable))
# print(env.observation_space.spaces)


# # print(env.observation_space.shape)
# if :
#     print('Trudsdfsdfsdfsde')

print(env.observation_space.obs_type)
print(env.observation_space.obs_shape)
# # print([1, 2, 3])


print('uddddddddd')
print(np.array([[0, 2], [1, 2], [3, 4], [5, 4]]).shape)