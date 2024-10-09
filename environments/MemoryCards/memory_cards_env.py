import gymnasium
from gymnasium import spaces
from typing import Union, Tuple, Optional
import numpy as np
from numpy.random import Generator
from collections.abc import Iterable



class MemoryCards(gymnasium.Env):
    """
    Memory Card Game. The agent is presented with N cards, and tries to find all the pairs.
    Each round the agent is shown one card, and has to select the card it thinks will pair it.
    Once a pair is found, it is removed from the pile.

    Consider the following episode:
        Say 1 represents a dog card, 2 Represents a cat card, 3 a bird card
        0 is hidden, and -1 means removed

    Reset:
    State: [1, 3, 1, 2, 2, 3]

    Obs:   [0, 0, 0, 2, 0, 0]
    Action: 2
    Reward: 0 # 0 reward for selecting incorrectly
    Obs: [0, 0, 0, 0, 0, 3]
    Action: 1
    Reward: 1 # Positive reward for selecting correctly
    Obs: [1, -1, 0, 0, 0, -1]
    Action: 2
    Reward: 1
    Obs: [-1, -1, -1, 0, 2, -1]
    Action: 0
    Reward: -1 # Negative reward for picking a removed card
    Obs: [-1, -1, -1, 0, 2, -1]
    Action: 3
    Reward: 1
    Obs: [-1, -1, -1, -1, -1, -1]
    Done: True


    shoutout @dmklee on github
    """

    def __init__(self, level_id):
        super(MemoryCards, self).__init__()

        self._rewards = []
        self.t = 0

        # self.num_pairs = num_pairs #5
        # self.max_episode_steps= 50

        if 'Easy' in level_id:
            self.num_pairs = 5
            self.max_episode_steps= 51 # +- mean reward based on 500 samples for random agent
        elif 'Medium' in level_id:
            self.num_pairs = 8
            self.max_episode_steps= 126 # +- mean reward based on 500 samples for random agent
        elif 'Hard' in level_id:
            self.num_pairs = 12
            self.max_episode_steps= 285 # +- mean reward based on 500 samples for random agent

        self.num_cards = self.num_pairs * 2

        # Each card can be -1 for removed, 0 for hidden, or 1, 2, ..., num_cards / 2
        self.observation_space = gymnasium.spaces.MultiDiscrete(
            [self.num_pairs + 2] * self.num_cards
        )
        self.action_space = gymnasium.spaces.Discrete(self.num_cards)

        self.card_removed = self.num_pairs + 1
        self.card_hidden = 0

        # Initialize state with everything removed
        self.state = self.card_removed * np.ones(self.num_cards)
        self.observation = self.card_hidden * np.ones(self.num_cards)
        self.current_card = -1
        self.np_random = None


        self.observation_space.obs_shape = self.observation_space.shape 
        self.observation_space.obs_type = 'multidiscrete'

    def seed(self, seed: int) -> None:
        if self.np_random is None:
            seed_seq = np.random.SeedSequence(seed)
            self.np_random = Generator(np.random.PCG64(seed_seq))

    def reset(self) -> np.ndarray:

        self.t = 0
        self._rewards = []

        # Create hand and shuffle the cards
        self.state = np.repeat(np.arange(1, self.num_pairs + 1), 2)
        self.np_random.shuffle(self.state)
        # Reset observation
        self.observation = self.card_hidden * np.ones(self.num_cards)
        # Reveal one card randomly
        self.current_card = self.np_random.integers(self.num_cards)
        self.observation[self.current_card] = self.state[self.current_card]

        return self.observation

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:

        if isinstance(action, Iterable):
            if len(action) == 1:
                action = action[0]

        if np.array_equal(self.state, self.card_removed * np.ones(self.num_cards)):
            raise ValueError("Trying to take step in invalid state. Did you reset?")

        done = False
        info = {}
        # Choosing the shown card is wrong
        if action == self.current_card:
            self.observation[self.current_card] = self.card_hidden
            reward = -1
        # Card selected is correct
        elif self.state[action] == self.observation[self.current_card]:
            # set both cards to removed
            self.observation[action] = self.card_removed
            self.observation[self.current_card] = self.card_removed
            reward = 0
            # Removed all cards
            if np.array_equal(
                self.observation, self.card_removed * np.ones(self.num_cards)
            ):
                done = True
                info.update({"is_success": True})
        else:
            self.observation[self.current_card] = self.card_hidden
            reward = -1
        
        self._rewards.append(reward)

        if self.t == self.max_episode_steps - 1:
            done = True

        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        self.t += 1


        if not done:
            # Reveal one card randomly
            self.current_card = self.np_random.integers(self.num_cards)
            # Make sure the card isn't already removed
            while self.observation[self.current_card] == self.card_removed:
                self.current_card = self.np_random.integers(self.num_cards)
            self.observation[self.current_card] = self.state[self.current_card]

        return self.observation, reward, done, info