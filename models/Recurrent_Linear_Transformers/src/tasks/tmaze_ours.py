import numpy as np
import gymnasium as gym


"""
T-Maze: originated from (Bakker, 2001) and earlier neuroscience work, 
    and here extended to unit-test several key challenges in RL:
- Exploration
- Memory and credit assignment
- Discounting and distraction
- Generalization

Finite horizon problem: episode_length
Has a corridor of corridor_length
Looks like
                        g1
o--s---------------------j
                        g2
o is the oracle point, (x, y) = (0, 0)
s is starting point, (x, y) = (o, 0)
j is T-juncation, (x, y) = (o + corridor_length, 0)
g1 is goal candidate, (x, y) = (o + corridor_length, 1)
g2 is goal candidate, (x, y) = (o + corridor_length, -1)
"""


class TMazeBase(gym.Env):
    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        ambiguous_position: bool = False,
        expose_goal: bool = False,
        add_timestep: bool = False,
        seed: int = None,
    ):
        """
        The Base class of TMaze, decouples episode_length and corridor_length

        Other variants:
            (Osband, 2016): distract_reward = eps > 0, goal_reward is given at T-junction (no choice).
                This only tests the exploration and discounting of agent, no memory required
            (Osband, 2020): ambiguous_position = True, add_timestep = True, supervised = True.
                This only tests the memory of agent, no exploration required (not implemented here)
        """
        super().__init__()
        assert corridor_length >= 1 and episode_length >= 1
        assert penalty <= 0.0

        self.episode_length = episode_length
        self.corridor_length = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = goal_reward
        self.penalty = penalty
        self.distract_reward = distract_reward

        self.ambiguous_position = ambiguous_position
        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(4)  # four directions
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        self.tmaze_map = np.zeros(
            (3 + 2, self.oracle_length + self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[
            [self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1
        ] = True  # goal candidates
        #print(self.tmaze_map.astype(np.int32))

        obs_dim = 2 if self.ambiguous_position else 3
        if self.expose_goal:  # test Markov policies
            assert self.ambiguous_position is False
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        if seed is not None:
            self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def position_encoding(self, x: int, y: int, goal_y: int):
        if x == 0:
            # oracle position
            if not self.oracle_visited:
                # only appear at first
                exposure = goal_y
                self.oracle_visited = True
            else:
                exposure = 0

        if self.ambiguous_position:
            if x == 0:
                # oracle position
                return [0, exposure]
            elif x < self.oracle_length + self.corridor_length:
                # intermediate positions (on the corridor)
                return [0, 0]
            else:
                # T-junction or goal candidates
                return [1, y]
        else:
            if self.expose_goal:
                return [x, y, goal_y if self.oracle_visited else 0]
            if x == 0:
                # oracle position
                return [x, y, exposure]
            else:
                return [x, y, 0]

    def timestep_encoding(self):
        return (
            [
                self.time_step,
            ]
            if self.add_timestep
            else []
        )

    def get_obs(self):
        return np.array(
            self.position_encoding(self.x, self.y, self.goal_y)
            + self.timestep_encoding(),
            dtype=np.float32,
        )

    def reward_fn(self, done: bool, x: int, y: int, goal_y: int):
        if done:  # only give bonus at the final time step
            return float(y == goal_y) * self.goal_reward
        else:
            # a penalty (when t > o) if x < t - o (desired: x = t - o)
            rew = float(x < self.time_step - self.oracle_length) * self.penalty
            if x == 0:
                return rew + self.distract_reward
            else:
                return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        # transition
        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            # valid move
            self.x, self.y = self.x + move_x, self.y + move_y
            

        if self.time_step >= self.episode_length or (self.x == self.corridor_length and (self.y == 1 or self.y == -1)):
            done = True
        else:
            done = False
        
        
        rew = self.reward_fn(done, self.x, self.y, self.goal_y)
        truncated = False
        return self.get_obs(), rew, done, truncated, {}

    def reset(self):
        self.x, self.y = self.oracle_length, 0
        self.goal_y = np.random.choice([-1, 1])

        self.oracle_visited = False
        self.time_step = 0
        return self.get_obs(), {}


class TMazeOurs(TMazeBase):
    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        goal_penalty: float = 0.0,
        timestep_penalty: float = 0.0,
        seed: int = None,
    ):
        """
        Transforms observation of the  base environment: [x, y, cue] -> [y, cue, flag, noise]
            flag: whether the agent is at the T junction of the maze
            noise: random int from [-1; 1]
        """
        super().__init__(
            episode_length=episode_length,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=0.0,
            distract_reward=0.0,
            expose_goal=False,
            ambiguous_position=False,
            add_timestep=False,
            seed=seed,
        )
        obs_dim = 4
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.goal_penalty = goal_penalty
        self.timestep_penalty = timestep_penalty
        self.last_success = False
    
    def get_obs(self):
        flag = 1 if (self.x == self.corridor_length and self.y == 0) else 0
        noise = np.random.randint(-1, 2)
        obs = super().get_obs()
        obs = obs[1:]  # [x, y, cue] -> [y, cue]
        obs = np.append(obs, [flag, noise])
        return obs
    
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        obs, rew, done, trunc, info = super().step(action)
        if self.y != 0:
            self.last_success = self.y == self.goal_y
            rew -= (not self.last_success) * self.goal_penalty
            info['episode_extra_stats'] = {"success": int(self.last_success), "new_cue": self.current_cue}
        else:
            rew -= self.timestep_penalty
        return obs, rew, done, trunc, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        obs, info = super().reset()
        self.current_cue = self.goal_y
        info = {"success": int(self.last_success), "new_cue": self.current_cue}
        self.last_success = False  # reset
        return obs, info
    
    def render(self):
        frame = np.zeros((3, self.corridor_length+1, 3), dtype=np.uint8)
        frame[1-self.y, self.x, :] = 128
        frame[1-self.goal_y, self.corridor_length, :] = 255
        return frame


if __name__ == "__main__":
    episode_length = 1000
    corridor_length = 160
    goal_reward = 4
    goal_penalty = 1
    timestep_penalty = 0.1
    seed = 123
    env = TMazeOurs(
        episode_length=episode_length, 
        corridor_length=corridor_length,
        goal_reward=goal_reward,
        goal_penalty=goal_penalty,
        timestep_penalty=timestep_penalty,
        seed=None,
    )
    s0, info0 = env.reset(seed=seed)
    x0 = env.x
    cue = s0[1]
    done = False
    step = 0
    R = 0
    while not done:
        a = 0 if step != corridor_length else (1 if cue == 1 else 3)
        s, r, done, trunc, info = env.step(a)
        step += 1
        R += r
        x = env.x
    print(f"init_state: {s0}, init_info: {info0}")
    print(f"final_state: {s}, final_info: {info}")
    print(f"total_reward: {R}, total_steps: {step}, done: {done}, truncated: {trunc}")
    print(f"init_x: {x0}, final_x: {x}")