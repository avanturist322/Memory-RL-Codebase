import copy
import itertools
import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import math

from . import base_environment, constants, utils

class KeyDoorEnv(base_environment.BaseEnvironment):
    """Grid world environment with multiple rooms.
    Between each room is a door, that requires a key to unlock.
    """

    def __init__(
        self,
        map_ascii_path: str,
        map_yaml_path: str,
        representation: str,
        episode_timeout: Optional[Union[int, None]] = None,
        frame_stack: Optional[Union[int, None]] = None,
        scaling: Optional[int] = 1,
        field_x: Optional[int] = 1,
        field_y: Optional[int] = 1,
        grayscale: bool = True,
        batch_dimension: bool = True,
        torch_axes: bool = True,
        seed: int = None,
    ) -> None:
        """Class constructor.

        Args:
            map_ascii_path: path to txt or other ascii file with map specifications.
            map_yaml_path: path to yaml file with map settings (reward locations etc.)
            representation: agent_position (for tabular) or pixel
                (for function approximation).
            episode_timeout: number of steps before episode automatically terminates.
            frame_stack: number of frames to stack together in representation.
            scaling: optional integer (for use with pixel representations)
                specifying how much to expand state by.
            field_x: integer (required for use with partial observability
                in pixel representations) specifying how many pixels in each x
                direction the agent can see.
            field_y: integer (required for use with partial observability
                in pixel representations) specifying how many pixels in each y
                direction the agent can see.
            grayscale: whether to grayscale representation.
            batch_dimension: whether to add dummy batch dimension to representation.
            torch_axes: whether to use torch (else tf) axis ordering.
        """

        self._key_ids = [constants.KEYS]
        self._rewards_state: np.ndarray
        self._keys_state: np.ndarray

        # if seed is not None:
        #     self.seed(seed)

        super().__init__(
            representation=representation,
            episode_timeout=episode_timeout,
            frame_stack=frame_stack,
            scaling=scaling,
            field_x=field_x,
            field_y=field_y,
            grayscale=grayscale,
            batch_dimension=batch_dimension,
            torch_axes=torch_axes,
        )
        
        self.map_ascii_path = map_ascii_path
        self.map_yaml_path = map_yaml_path

        # self._setup_environment(
        #     map_ascii_path=map_ascii_path, map_yaml_path=map_yaml_path
        # )

        # # states are zero, -1 removes walls from counts.
        # self._visitation_counts = -1 * copy.deepcopy(self._map)

    def _setup_environment(
        self, map_yaml_path: str, map_ascii_path: Optional[str] = None
    ):
        """Setup environment according to geometry of ascii file
        and settings of yaml file.

        Args:
            map_ascii_path: path to txt or other ascii file with map specifications.
            map_yaml_path: path to yaml file with map settings (reward locations etc.)
        """

        if map_ascii_path is not None:
            self._map = utils.parse_map_outline(
                map_file_path=map_ascii_path, mapping=self.MAPPING
            )

        (
            self._starting_xy,
            self._key_positions,
            self._door_positions,
            reward_positions,
            reward_statistics,
        ) = utils.parse_map_positions(map_yaml_path)

        
        self._rewards = utils.setup_reward_statistics(
            reward_positions, reward_statistics
        )
        self._total_rewards = len(self._rewards)

        (
            self._positional_state_space,
            self._key_possession_state_space,
            self._rewards_received_state_space,
            self._state_space,
            self._wall_state_space,
        ) = utils.configure_state_space(
            map_outline=self._map,
            key_positions=self._key_positions,
            reward_positions=reward_positions,
        )
        
        excluded_points = [(1, 3), (5, 3), (13, 3), (17, 3)]

        x_pos, y_pos = random.randint(1, 5), random.randint(1, 5)
        x_key, y_key = random.randint(1, 5), random.randint(1, 5)

        #while (x_pos, y_pos) == (x_key, y_key) or (x_pos, y_pos) in excluded_points or (x_key, y_key) in excluded_points:
        while np.sqrt((x_pos - x_key) ** 2 + (y_pos - y_key) ** 2) < 2 or (x_pos, y_pos) in excluded_points or (x_key, y_key) in excluded_points:
                x_pos, y_pos = random.randint(1, 5), random.randint(1, 5)
                x_key, y_key = random.randint(1, 5), random.randint(1, 5)

        x_door, y_door = random.randint(13, 17), random.randint(1, 5)

        while (x_door, y_door) in excluded_points:
            x_door, y_door = random.randint(13, 17), random.randint(1, 5)
            
        self._starting_xy = x_pos, y_pos
        self._key_positions[0] = x_key, y_key
        self._door_positions[0] = x_door, y_door
        
        self._key_is_taken = []

        self._episode_timeout -= 1

        self.PERCENT = 0.2
        
        #self._rewards = [(0,0)]
        #print(self._rewards)
        
            

    def average_values_over_positional_states(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], float]:
        """For certain analyses (e.g. plotting value functions) we want to
        average the values for each position over all non-positional state information--
        in this case the key posessions.
        Args:
            values: full state-action value information
        Returns:
            averaged_values: (positional-state)-action values.
        """
        averaged_values = {}
        for state in self._positional_state_space:
            non_positional_set = [
                values[state + i[0] + i[1]]
                for i in itertools.product(
                    self._key_possession_state_space, self._rewards_received_state_space
                )
            ]
            non_positional_mean = np.mean(non_positional_set, axis=0)
            averaged_values[state] = non_positional_mean
        return averaged_values

    def get_value_combinations(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], Dict[Tuple[int], float]]:
        """Get each possible combination of positional state-values
        over non-positional states.
        Args:
            values: values over overall state-space
        Returns:
            value_combinations: values over positional state space
                for each combination of non-positional state.
        """
        value_combinations = {}
        for key_state in self._key_possession_state_space:
            for reward_state in self._rewards_received_state_space:
                value_combination = {}
                for state in self._positional_state_space:
                    value_combination[state] = values[state + key_state + reward_state]
                value_combinations[key_state + reward_state] = value_combination

        return value_combinations

    def _env_skeleton(
        self,
        rewards: Union[None, str, Tuple[int]] = "state",
        keys: Dict[str, Union[None, str, Tuple[int]]] = {"keys": "state"},
        doors: Union[None, str] = "state",
        agent: Union[None, str, np.ndarray] = "state",
        cue: Union[None, str, np.ndarray] = None,
    ) -> np.ndarray:
        """Get a 'skeleton' of map e.g. for visualisation purposes.

        Args:
            rewards: # TODO whether or not to mark out rewards (ignores magnitudes).
            show_doors: whether or not to mark out doors.
            show_keys: whether or not to mark out keys.
            show_agent: whether or not to mark out agent position

        Returns:
            skeleton: np array of map.
        """
        # flip size so indexing is consistent with axis dimensions
        skeleton = np.ones(self._map.shape + (3,))

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        if rewards is not None:
            if isinstance(rewards, str):
                if rewards == constants.STATIONARY:
                    reward_iterate = list(self._rewards.keys())
                elif rewards == constants.STATE:
                    reward_positions = list(self._rewards.keys())
                    reward_iterate = [
                        reward_positions[i]
                        for i, r in enumerate(self._rewards_state)
                        if not r
                    ]
                else:
                    raise ValueError(f"Rewards keyword {rewards} not identified.")
            elif isinstance(rewards, tuple):
                reward_positions = list(self._rewards.keys())
                reward_iterate = [
                    reward_positions[i] for i, r in enumerate(rewards) if not r
                ]
            else:
                raise ValueError(
                    "Rewards must be string with relevant keyword or keystate list"
                )
            # show reward in red
            #for reward in reward_iterate:
                #skeleton[reward[::-1]] = [1.0, 0.0, 0.0]

        keys = keys[constants.KEYS]
        if keys is not None:
            if isinstance(keys, str):
                if keys == constants.STATIONARY:
                    keys_iterate = self._key_positions
                elif keys == constants.STATE:
                    keys_iterate = [
                        self._key_positions[i]
                        for i, k in enumerate(self._keys_state)
                        if not k
                    ]
                else:
                    raise ValueError(f"Keys keyword {keys} not identified.")
            elif isinstance(keys, tuple):
                keys_iterate = [
                    self._key_positions[i] for i, k in enumerate(keys) if not k
                ]
            else:
                raise ValueError(
                    "Keys must be string with relevant keyword or keystate list"
                )
            # show key in yellow
            for key_position in keys_iterate:
                skeleton[tuple(key_position[::-1])] = [1.0, 1.0, 0.0]

        if doors is not None:
            if isinstance(doors, str):
                if doors == constants.STATE:
                    doors_iterate = self._door_positions
                elif doors == constants.STATIONARY:
                    doors_iterate = self._door_positions
                else:
                    raise ValueError(f"Doors keyword {doors} not identified.")
            # show door in maroon
            for door in doors_iterate:
                skeleton[tuple(door[::-1])] = [0.5, 0.0, 0.0]

        if agent is not None:
            if isinstance(agent, str):
                if agent == constants.STATE:
                    agent_position = self._agent_position
                elif agent == constants.STATIONARY:
                    agent_position = self._starting_xy
            else:
                agent_position = agent
            # show agent
            skeleton[tuple(agent_position[::-1])] = 0.5 * np.ones(3)

        return skeleton

    def _partial_observation(self, state, agent_position):

        height = state.shape[0]
        width = state.shape[1]

        # out of bounds needs to be different from wall pixels
        OUT_OF_BOUNDS_PIXEL = 0.2 * np.ones(3)

        # nominal bounds on field of view (pre-edge cases)
        x_min = agent_position[1] - self._field_x
        x_max = agent_position[1] + self._field_x
        y_min = agent_position[0] - self._field_y
        y_max = agent_position[0] + self._field_y

        state = state[
            max(0, x_min) : min(x_max, width) + 1,
            max(0, y_min) : min(y_max, height) + 1,
            :,
        ]

        # edge case contingencies
        if 0 > x_min:
            append_left = 0 - x_min

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((append_left, state.shape[1], 1)),
            )
            state = np.concatenate(
                (fill, state),
                axis=0,
            )
        if x_max >= width:
            append_right = x_max + 1 - width

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((append_right, state.shape[1], 1)),
            )
            state = np.concatenate(
                (state, fill),
                axis=0,
            )
        if 0 > y_min:
            append_below = 0 - y_min

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((state.shape[0], append_below, 1)),
            )
            state = np.concatenate(
                (fill, state),
                axis=1,
            )
        if y_max >= height:
            append_above = y_max + 1 - height

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((state.shape[0], append_above, 1)),
            )
            state = np.concatenate(
                (state, fill),
                axis=1,
            )

        return state

    def get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        """From current state, produce a representation of it.
        This can either be a tuple of the agent and key positions,
        or a top-down pixel view of the environment (for DL)."""
        if self._representation == constants.AGENT_POSITION:
            return (
                tuple(self._agent_position)
                + tuple(self._keys_state)
                + tuple(self._rewards_state)
            )
        elif self._representation in [constants.PIXEL, constants.PO_PIXEL]:
            if tuple_state is None:
                agent_position = self._agent_position
                state = self._env_skeleton()  # H x W x C
            else:
                agent_position = tuple_state[:2]
                keys = tuple_state[2 : 2 + len(self._key_positions)]
                rewards = tuple_state[2 + len(self._key_positions) :]
                state = self._env_skeleton(
                    rewards=rewards, keys=keys, agent=agent_position
                )  # H x W x C

            if self._representation == constants.PO_PIXEL:
                state = self._partial_observation(
                    state=state, agent_position=agent_position
                )

            if self._grayscale:
                state = utils.rgb_to_grayscale(state)

            if self._torch_axes:
                state = np.transpose(state, axes=(2, 0, 1))  # C x H x W
                state = np.kron(state, np.ones((1, self._scaling, self._scaling)))
            else:
                state = np.kron(state, np.ones((self._scaling, self._scaling, 1)))

            if self._batch_dimension:
                # add batch dimension
                state = np.expand_dims(state, 0)

            return state

    def _move_agent(self, delta: np.ndarray) -> float:
        """Move agent. If provisional new position is a wall, no-op."""
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._wall_state_space
        locked_door = tuple(provisional_new_position) in self._door_positions

        if locked_door:
            door_index = self._door_positions.index(tuple(provisional_new_position))
            if self._keys_state[door_index]:
                locked_door = False

        if not moving_into_wall and not locked_door:
            self._agent_position = provisional_new_position

        if tuple(self._agent_position) in self._key_positions:
            key_index = self._key_positions.index(tuple(self._agent_position))
            if not self._keys_state[key_index]:
                self._keys_state[key_index] = 1
        return self._compute_reward()










    def lets_go(self, action):

        if action is not None:
            reward = self._move_agent(delta=self.DELTAS[action])
        new_state = self.get_state_representation()
        skeleton = self._env_skeleton()
        if tuple(self._agent_position) == self._key_positions[0]:
            self._key_is_taken.append(True)
        else:
            self._key_is_taken.append(False)
        
        self._full_obs = np.sum(skeleton, axis=2) * 2
        #self._full_obs[self._agent_position[1], self._agent_position[0]] = 3.0
        #self._full_obs[self._key_positions[0][1], self._key_positions[0][0]] = 4.0
        #print(self._key_positions)
        #print(self._door_positions)

        self._active = self._remain_active(reward=reward)
        self._episode_step_count += 1

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            self._train_episode_position_history.append(tuple(self._agent_position))
            self._train_episode_history.append(skeleton)
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history.append(
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                )
        else:
            self._test_episode_position_history.append(tuple(self._agent_position))
            self._test_episode_history.append(skeleton)
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history.append(
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                )
        return reward, self._full_obs

        
    def make_style(self, action):
        timer = self._episode_step_count

        if timer == math.floor(self._episode_timeout * self.PERCENT):
            self.agent_position = (random.randint(self.ROOM_2_X1, self.ROOM_2_X2), 
                                   random.randint(self.ROOM_2_Y1, self.ROOM_2_Y2))
        
        if timer == math.floor(self._episode_timeout - self._episode_timeout * self.PERCENT):
            x_room_3 = random.randint(self.ROOM_3_X1, 
                                      self.ROOM_3_X2)
            
            y_room_3 = random.randint(self.ROOM_3_Y1, 
                                      self.ROOM_3_Y2)
            
            while np.sqrt((x_room_3 - self._door_positions[0][0]) ** 2 + (y_room_3 - self._door_positions[0][1]) ** 2) < 2:
                x_room_3 = random.randint(self.ROOM_3_X1, 
                                          self.ROOM_3_X2)
                
                y_room_3 = random.randint(self.ROOM_3_Y1, 
                                          self.ROOM_3_Y2)
                
            self.agent_position = (x_room_3, y_room_3)
            
            
        if self._episode_timeout * self.PERCENT > timer:
            r, state = self.lets_go(action)
            state = self._full_obs[0:self.ROOM_2_Y2+2, 0:self.ROOM_2_X1]

        if self._episode_timeout - self._episode_timeout * self.PERCENT > timer >= self._episode_timeout * self.PERCENT:
            r, state = self.lets_go(action)
            state = self._full_obs[0:self.ROOM_2_Y2+2, self.ROOM_2_X1-1:self.ROOM_2_X2+2]

        if timer >= self._episode_timeout - self._episode_timeout * self.PERCENT:
            r, state = self.lets_go(action)
            state = self._full_obs[0:self.ROOM_2_Y2+2, self.ROOM_3_X1-1:self.ROOM_3_X2+2]

        return state, r
        
    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            self._active
        ), "Environment not active. call reset() to reset environment and make it active."
        assert (
            action in self.ACTION_SPACE
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."
        
        #print(self._env_skeleton())



        # ! BEGIN OF TELEPORTATION BLOCK

        state, r = self.make_style(action)

        # ! END OF TELEPORTATION BLOCK

        #return obs, rew, done, info
        return state, r, not self._active, {}

    # def _compute_reward(self) -> float:
    #     """Check for reward, i.e. whether agent position is equal to a reward position.
    #     If reward is found, add to rewards received log.
    #     """
        
    #     if (
    #         tuple(self._agent_position) in self._rewards
    #         and tuple(self._agent_position) not in self._rewards_received
    #     ):
    #         reward = self._rewards.get(tuple(self._agent_position))()
    #         print(reward)#, self._rewards.get(tuple(self._agent_position))())
    #         reward_index = list(self._rewards.keys()).index(tuple(self._agent_position))
    #         self._rewards_state[reward_index] = 1
    #         self._rewards_received.append(tuple(self._agent_position))
    #     else:
    #         reward = 0.0

    #     return reward
    
    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        #print(self._door_positions, tuple(self._agent_position))
        # print(list(self._rewards.keys()), self._door_positions)
        if (
            tuple(self._agent_position) in self._door_positions
            #and tuple(self._agent_position) not in self._rewards_received
            and np.sum(self._key_is_taken) != 0
            # and self._keys_state[0] == 1
        ):
            #reward = self._rewards.get(tuple(self._door_positions))()
            reward = 1.0
            #reward_index = list(self._rewards.keys()).index(tuple(self._agent_position))
            reward_index = list(self._door_positions).index(tuple(self._agent_position))
            self._rewards_state[reward_index] = 1
            self._rewards_received.append(tuple(self._agent_position))
        else:
            reward = 0.0

        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """

        conditions = [
            self._episode_step_count == self._episode_timeout-1,
            len(self._rewards_received) == self._total_rewards,
            ((tuple(self._agent_position) == self._door_positions[0]) and (np.sum(self._key_is_taken) != 0))
        ] 
        return not any(conditions)

    def reset(
        self, seed: int = 42, options = None, train: bool = True, map_yaml_path: Optional[str] = None
    ) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        self.seed(seed)

        self._setup_environment(
            map_ascii_path=self.map_ascii_path, map_yaml_path=self.map_yaml_path
        )

        # states are zero, -1 removes walls from counts.
        self._visitation_counts = -1 * copy.deepcopy(self._map)
        # if map_yaml_path is not None:
        #     self._setup_environment(map_yaml_path=map_yaml_path)
        

        self._active = True
        self._episode_step_count = 0
        self._training = train
        self._agent_position = np.array(self._starting_xy)
        self._rewards_received = []
        self._keys_state = np.zeros(len(self._key_positions), dtype=int)
        self._rewards_state = np.zeros(len(self._rewards), dtype=int)

        initial_state = self.get_state_representation()
        skeleton = self._env_skeleton()
        full_obs_init = np.sum(skeleton, axis=2) * 2

        if train:
            self._train_episode_position_history = [tuple(self._agent_position)]
            self._train_episode_history = [skeleton]
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history = [
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                ]
        else:
            self._test_episode_position_history = [tuple(self._agent_position)]
            self._test_episode_history = [skeleton]
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history = [
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                ]

        #return initial_state
        return full_obs_init[0:self.ROOM_2_Y2+2, 0:self.ROOM_2_X1]

    def close(self):
        pass

    def seed(self, seed=None):
        # self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
