import warnings
import copy
import random

import gymnasium as gym
import numpy as np

from amago.envs import AMAGOEnv
from amago.hindsight import GoalSeq

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from environments.ViZDoom_Two_Colors.env import env_vizdoom


SCENARIO_NAME = 'custom_scenario{:003}.cfg' # 'custom_scenario_no_pil{:003}.cfg'
SCENARIO_DIR = 'environments/ViZDoom_Two_Colors/env_configs/'

env_args = {
    'simulator':'doom', 
    'scenario': SCENARIO_NAME, #custom_scenario{:003}.cfg
    'test_scenario':'', 
    'screen_size':'320X180', 
    'screen_height':64, 
    'screen_width':112, 
    'num_environments':16,# 16
    'limit_actions':True, 
    'scenario_dir': SCENARIO_DIR,
    'test_scenario_dir':'', 
    'show_window':False, 
    'resize':True, 
    'multimaze':True, 
    'num_mazes_train':16, 
    'num_mazes_test':1, # 64 
    'disable_head_bob':False, 
    'use_shaping':False, 
    'fixed_scenario':False, 
    'use_pipes':False, 
    'num_actions':0, 
    'hidden_size':128, 
    'reload_model':'', 
    # 'model_checkpoint': A2C_CHECKPOINT,   # two_col_p0_checkpoint_0049154048.pth.tar',  #two_col_p0_checkpoint_0198658048.pth.tar', 
    'conv1_size':16, 
    'conv2_size':32, 
    'conv3_size':16, 
    'learning_rate':0.0007, 
    'momentum':0.0, 
    'gamma':0.99, 
    'frame_skip':4, 
    'train_freq':4, 
    'train_report_freq':100, 
    'max_iters':5000000, 
    'eval_freq':1000, 
    'eval_games':50, 
    'model_save_rate':1000, 
    'eps':1e-05, 
    'alpha':0.99, 
    'use_gae':False, 
    'tau':0.95, 
    'entropy_coef':0.001, 
    'value_loss_coef':0.5, 
    'max_grad_norm':0.5, 
    'num_steps':128, 
    'num_stack':1, 
    'num_frames':200000000, 
    'use_em_loss':False, 
    'skip_eval':False, 
    'stoc_evals':False, 
    'model_dir':'', 
    'out_dir':'./', 
    'log_interval':100, 
    'job_id':12345, 
    'test_name':'test_000', 
    'use_visdom':False, 
    'visdom_port':8097, 
    'visdom_ip':'http://10.0.0.1'                 
}

scenario = env_args['scenario_dir'] + env_args['scenario'].format(0)
config_env = scenario


class GymEnv(AMAGOEnv):
    def __init__(
        self,
        gym_env: gym.Env,
        env_name: str,
        horizon: int,
        start: int = 0,
        zero_shot: bool = True,
        convert_from_old_gym: bool = False,
        soft_reset_kwargs={},
    ):
        if convert_from_old_gym:
            gym_env = gym.wrappers.EnvCompatibility(gym_env)

        super().__init__(gym_env, horizon=horizon, start=start)
        self.zero_shot = zero_shot
        self._env_name = env_name
        self.soft_reset_kwargs = soft_reset_kwargs

    @property
    def env_name(self):
        return self._env_name

    @property
    def blank_goal(self):
        return np.zeros((1,), dtype=np.float32)

    @property
    def achieved_goal(self) -> np.ndarray:
        return [self.blank_goal + 1]

    @property
    def kgoal_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=0.0, high=0.0, shape=(1, 1))

    @property
    def goal_sequence(self) -> GoalSeq:
        goal_seq = [self.blank_goal]
        return GoalSeq(seq=goal_seq, active_idx=0)

    def step(self, action):
        return super().step(
            action,
            normal_rl_reward=True,
            normal_rl_reset=self.zero_shot,
            soft_reset_kwargs=self.soft_reset_kwargs,
        )


class _DiscreteToBox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n,), dtype=np.int64
        )

    def observation(self, obs):
        arr = np.zeros((self.n,), dtype=np.int64)
        arr[obs] = 1
        return arr


class _MultiDiscreteToBox(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.MultiDiscrete)
        self.max_options = env.observation_space.nvec.max()
        self.categories = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.categories * self.max_options,), dtype=np.int64
        )

    def observation(self, obs):
        arr = np.zeros((self.categories, self.max_options), dtype=np.int64)
        for i, o in enumerate(obs):
            arr[i, o] = 1
        return arr.flatten()

class FrameProcessing(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env.new_step_api = False

    def reset(self, **kwargs):
        # print('im the reset method and im working')
        data = self.env.reset(**kwargs)
        obs = data['image'] # * (3,64,112)
        # print(obs.shape) 
        # print(obs.dtype, obs.min(), obs[0], obs.max())
        # print(obs.shape)
        # obs = obs.transpose(1,2,0) # !!!!!!
        # print(obs.shape)
        self._last_frame = obs.copy()
        # obs = obs.astype(np.float32) # / 255.
        # print('reset still working')

        return obs, {}
    
    def step(self, action):
        # print('im the step method and im workng')
        # print(self.env.step(action))
        obs, reward, done, info = self.env.step(action)
        # print('is red', info['is_red'])
        # print('step still working')
        obs = obs['image']
        self._last_frame = obs.copy()
        # obs = obs.astype(np.float32) #  / 255.
        # print('ok i love gymnasium')

        return obs, reward, done, False, info
    
class DoomGymEnv(GymEnv):
    def __init__(self, env_name: str, horizon: int):
        try:
            from environments.ViZDoom_Two_Colors.env import env_vizdoom, VizdoomAsGym
        except ImportError:
            msg = "vizdoom required"
            print(msg)
            exit()

        env = env_vizdoom.DoomEnvironmentDisappear(
                                        scenario=config_env,
                                        show_window=False,
                                        use_info=True,
                                        use_shaping=False, # * if False rew only +1 if True rew +1 or -1
                                        frame_skip=2,
                                        no_backward_movement=True,
                                        # seed=2
                                        )
        env = VizdoomAsGym(env)

        env = FrameProcessing(env)

        # print(111111111111111111)

        env.metadata = []
        # metadata = {"render_modes": ["human", "rgb_array"]}

        # env = Flatten(env)
        # if isinstance(env.action_space, gym.spaces.Discrete | gym.spaces.MultiDiscrete):
        #     env = DiscreteAction(env)
        if isinstance(env.observation_space, gym.spaces.Discrete):
            env = _DiscreteToBox(env)
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            env = _MultiDiscreteToBox(env)
        super().__init__(
            env, env_name=env_name, horizon=horizon, start=0, zero_shot=True
        )


class RandomLunar(gym.Env):
    def __init__(
        self,
        k_shots=2,
    ):
        self.reset()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.k_shots = k_shots

    def reset(self, *args, **kwargs):
        self.current_gravity = random.uniform(-3.0, -0.1)
        self.current_wind = random.uniform(0.0, 20.0)
        self.current_turbulence = random.uniform(0.0, 2.0)
        self.env = gym.make(
            "LunarLander-v2",
            continuous=True,
            gravity=self.current_gravity,
            enable_wind=True,
            wind_power=self.current_wind,
            turbulence_power=self.current_turbulence,
        )
        self.current_k = 0
        return self.env.reset()

    def step(self, action):
        done = False
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            next_state, info = self.env.reset()
            self.current_k += 1
        if self.current_k >= self.k_shots:
            done = True
        return next_state, reward, done, False, info


from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class MetaFrozenLake(gym.Env):
    def __init__(
        self,
        size: int,
        k_shots: int = 10,
        hard_mode: bool = False,
        recover_mode: bool = False,
    ):
        self.size = size
        self.k_shots = k_shots
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(shape=(4,), low=0.0, high=1.0)
        self.hard_mode = hard_mode
        self.recover_mode = recover_mode
        self.reset()

    def reset(self, *args, **kwargs):
        self.current_map = [[t for t in row] for row in generate_random_map(self.size)]
        self.action_mapping = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.hard_mode and random.random() < 0.5:
            temp = self.action_mapping[1]
            self.action_mapping[1] = self.action_mapping[3]
            self.action_mapping[3] = temp
        self.current_k = 0
        return self.soft_reset()

    def make_obs(self, reset_signal: bool):
        if self.hard_mode and random.random() < 0.25:
            x = min(max(self.x + random.choice([-1, 0, 1]), 0), self.size - 1)
            y = min(max(self.y + random.choice([-1, 0, 1]), 0), self.size - 1)
        else:
            x, y = self.x, self.y
        return np.array(
            [x / self.size, y / self.size, reset_signal, self.current_k / self.k_shots],
            dtype=np.float32,
        )

    def soft_reset(self):
        self.active_map = copy.deepcopy(self.current_map)
        self.x, self.y = 0, 0
        obs = self.make_obs(reset_signal=True)
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action)
        move_x, move_y = self.action_mapping[action]
        next_x = max(min(self.x + move_x, self.size - 1), 0)
        next_y = max(min(self.y + move_y, self.size - 1), 0)

        if (
            (self.x, self.y) != (next_y, next_y)
            and self.hard_mode
            and random.random() < 0.33
        ):
            self.active_map[self.x][self.y] = "H"

        on = self.active_map[next_x][next_y]
        if on == "G":
            reward = 1.0
            soft_reset = True
        elif on == "H":
            reward = 0.0 if not self.recover_mode else -0.1
            soft_reset = not self.recover_mode
            if self.recover_mode:
                next_x = self.x
                next_y = self.y
        else:
            reward = 0.0
            soft_reset = False

        self.x = next_x
        self.y = next_y

        if soft_reset:
            self.current_k += 1
            next_state, info = self.soft_reset()
        else:
            next_state, info = self.make_obs(False), {}

        terminated = self.current_k >= self.k_shots
        return next_state, reward, terminated, False, info

    def render(self, *args, **kwargs):
        render_map = copy.deepcopy(self.active_map)
        render_map[self.x][self.y] = "A"
        print(f"\nFrozen Lake (k={self.k_shots}, Hard Mode={self.hard_mode})")
        for row in render_map:
            print(" ".join(row))


if __name__ == "__main__":
    env = MetaFrozenLake(5, 2, False, True)
    env.reset()
    env.render()
    done = False
    while not done:
        action = input()
        action = {"a": 4, "w": 2, "d": 3, "s": 1, "x": 0}[action]
        next_state, reward, done, _, info = env.step(action)
        env.render()
        print(next_state)
        print(reward)
        print(done)
