import matplotlib.pyplot as plt
import random
import numpy as np
from gymnasium import spaces
from gymnasium import Env
import numpy as np
import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from random import randint
import numpy as np
from collections.abc import Iterable
import gym 
import gymnasium

class WindEnv(Env):
    """
    Modification of https://github.com/twni2016/pomdp-baselines/blob/main/envs/meta/toy_navigation/wind.py

    point robot on a 2-D plane with position control
    tasks vary in noise term in dynamics for fixed goal-reaching
     - noise is fixed for a task
     - reward is sparse
    """

    def __init__(
        self,
        max_episode_steps=75,
        n_tasks=1,  # this will be modified in config
        goal_radius=0.03,
        **kwargs
    ):

        self._rewards = []
        self.t = 0

        self.n_tasks = n_tasks
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        np.random.seed(1337)  # let's fix winds for reproducibility
        self.winds = [
            [np.random.uniform(-0.08, 0.08), np.random.uniform(-0.08, 0.08)]
            for _ in range(n_tasks)
        ]
        # fixed hidden goal
        self._goal = np.array([0.0, 1.0])
        self.goal_radius = goal_radius

        self.reset_task(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)

        if isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            self.observation_space.obs_shape = self.observation_space.shape 
            if len(self.observation_space.obs_shape) > 1:
                self.observation_space.obs_type = 'image'
            else:
                self.observation_space.obs_type = 'vector'
        elif isinstance(self.observation_space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)):
            # discrete observation
            self.observation_space.obs_shape = (1, )
            self.observation_space.obs_type = 'discrete'
        elif isinstance(self.observation_space, Iterable) and all(isinstance(space, (gymnasium.spaces.discrete.Discrete, gym.spaces.discrete.Discrete)) for space in self.observation_space.spaces):
            # multi discrete observation - like discrete vector 
            self.observation_space.obs_shape = (len(self.observation_space.spaces),)#tuple([1 for obs_space in self.observation_space.spaces])
            self.observation_space.obs_type = 'multidiscrete' 
            self.observation_space.nvec = [env.n for env in self.observation_space.spaces]

        elif isinstance(self.observation_space, (gymnasium.spaces.MultiDiscrete, gym.spaces.MultiDiscrete)):
            # multi discrete observation - like discrete vector 
            self.observation_space.obs_shape = self.observation_space.shape 
            self.observation_space.obs_type = 'multidiscrete'
        # elif isinstance(self.observation_space, Iterable) and isinstance(self.observation_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
        #     # Case: list/tuple etc. of image/vector observation is available e.g list of observations from different cameras
        #     raise NotImplementedError
        else:
            raise NotImplementedError



    def reset_task(self, idx):
        """reset goal AND reset the agent"""
        if idx is not None:
            self._wind = np.array(self.winds[idx])

    def set_goal(self, wind):
        self._wind = np.asarray(wind)

    def get_current_task(self):
        # for multi-task MDP
        return self._wind.copy()

    def get_all_task_idx(self):
        return range(len(self.winds))

    def reset_model(self):
        # reset to a fixed location on the unit square
        self._state = np.array([0.0, 0.0])
        return self._get_obs()

    def reset(self, new_task=True, **kwargs):
        if new_task:
            task = random.choice(self.get_all_task_idx())
            self.reset_task(task)


        self.step_count = 0
        self.t = 0
        self._rewards = []
        return self.reset_model()#, {}

    def _get_obs(self):
        return np.copy(self._state).astype(np.float32)

    def step(self, action):
        info = {}

        if isinstance(action, Iterable):
            if len(action) == 1:
                action = action[0]


        self._state = self._state + action + self._wind  # add wind to transition

        # sparse reward
        if self.is_goal_state():
            reward = 1.0
        else:
            reward = 0.0
        self._rewards.append(reward)

        # check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            done = True
        else:
            done = False

        ob = self._get_obs()

        if done:
            info.update({"reward": sum(self._rewards),
                    "length": len(self._rewards)})

        return ob, reward, done, dict()

    def viewer_setup(self):
        print("no viewer")
        pass

    def render(self):
        print("current state:", self._state)

    def is_goal_state(self):
        if np.linalg.norm(self._state - self._goal) <= self.goal_radius:
            return True
        else:
            return False

    def plot_env(self):
        ax = plt.gca()
        # fix visualization
        plt.axis("scaled")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 1.5)
        plt.xticks([])
        plt.yticks([])
        # plot goal position
        circle = plt.Circle(
            (self._goal[0], self._goal[1]), radius=self.goal_radius, alpha=0.3
        )
        ax.add_artist(circle)
        # plot wind vector field
        X, Y = np.meshgrid(np.linspace(-0.8, 1, 5), np.linspace(-0.4, 1.5, 5))
        plt.quiver(X, Y, [self._wind[0]], [self._wind[1]])

    def plot_behavior(self, observations, plot_env=True, **kwargs):
        # kwargs are color and label
        if plot_env:  # whether to plot circle and goal pos..(maybe already exists)
            self.plot_env()
        # plot trajectory
        plt.plot(observations[:, 0], observations[:, 1], **kwargs)