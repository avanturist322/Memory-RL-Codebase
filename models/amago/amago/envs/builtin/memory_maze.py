import warnings
import copy
import random
import gym as gym0

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

from environments.Memory_Maze.env.env_memory_maze import create_env


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

    
# class FiveToFourWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)

#     def step(self, action):
#         state, reward, done, info = self.env.step(action)

#         return state, reward, done, _, info



class FrameProcessing(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
    def reset(self, seed, options):
        # print(seed, options)
        self.env.seed(seed)
        obs = self.env.reset()
        self._last_frame = obs.copy()

        return obs, {}
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._last_frame = obs.copy()
        # obs = obs['observation']
        return obs, reward, done, truncated, info

# class FrameProcessing(gym.Wrapper):
#     def __init__(self, env: gym.Env):
#         super().__init__(env)
#         self.env.new_step_api = False

#     def reset(self, **kwargs):
#         # print('im the reset method and im working')
#         data = self.env.reset(**kwargs)
#         obs = data['image'] # * (3,64,112)
#         # print(obs.shape) 
#         # print(obs.dtype, obs.min(), obs[0], obs.max())
#         # print(obs.shape)
#         # obs = obs.transpose(1,2,0) # !!!!!!
#         # print(obs.shape)
#         self._last_frame = obs.copy()
#         # obs = obs.astype(np.float32) # / 255.
#         # print('reset still working')

#         return obs, {}
    
#     def step(self, action):
#         # print('im the step method and im workng')
#         # print(self.env.step(action))
#         obs, reward, done, info = self.env.step(action)
#         # print('is red', info['is_red'])
#         # print('step still working')
#         obs = obs['image']
#         self._last_frame = obs.copy()
#         # obs = obs.astype(np.float32) #  / 255.
#         # print('ok i love gymnasium')

#         return obs, reward, done, False, info

def create_memorymaze(env_name):

    env = gym0.make(env_name)

    env = FrameProcessing(env) # !

    return env 
    
class MemoryMazeGymEnv(GymEnv):
    def __init__(self, env_name: str, horizon: int):
        # print(env_name)
        env = create_memorymaze(env_name) # 'memory_maze:MemoryMaze-9x9-v0'

#         # env.metadata = []
#         # metadata = {"render_modes": ["human", "rgb_array"]}

#         # try:
#         #     from environments.ViZDoom_Two_Colors.env import env_vizdoom, VizdoomAsGym
#         # except ImportError:
#         #     msg = "vizdoom required"
#         #     print(msg)
#         #     exit()
#         # SCENARIO_NAME = 'custom_scenario{:003}.cfg' # 'custom_scenario_no_pil{:003}.cfg'
#         # SCENARIO_DIR = 'environments/ViZDoom_Two_Colors/env_configs/'

#         # env_args = {
#         #     'simulator':'doom', 
#         #     'scenario': SCENARIO_NAME, #custom_scenario{:003}.cfg
#         #     'test_scenario':'', 
#         #     'screen_size':'320X180', 
#         #     'screen_height':64, 
#         #     'screen_width':112, 
#         #     'num_environments':16,# 16
#         #     'limit_actions':True, 
#         #     'scenario_dir': SCENARIO_DIR,
#         #     'test_scenario_dir':'', 
#         #     'show_window':False, 
#         #     'resize':True, 
#         #     'multimaze':True, 
#         #     'num_mazes_train':16, 
#         #     'num_mazes_test':1, # 64 
#         #     'disable_head_bob':False, 
#         #     'use_shaping':False, 
#         #     'fixed_scenario':False, 
#         #     'use_pipes':False, 
#         #     'num_actions':0, 
#         #     'hidden_size':128, 
#         #     'reload_model':'', 
#         #     # 'model_checkpoint': A2C_CHECKPOINT,   # two_col_p0_checkpoint_0049154048.pth.tar',  #two_col_p0_checkpoint_0198658048.pth.tar', 
#         #     'conv1_size':16, 
#         #     'conv2_size':32, 
#         #     'conv3_size':16, 
#         #     'learning_rate':0.0007, 
#         #     'momentum':0.0, 
#         #     'gamma':0.99, 
#         #     'frame_skip':4, 
#         #     'train_freq':4, 
#         #     'train_report_freq':100, 
#         #     'max_iters':5000000, 
#         #     'eval_freq':1000, 
#         #     'eval_games':50, 
#         #     'model_save_rate':1000, 
#         #     'eps':1e-05, 
#         #     'alpha':0.99, 
#         #     'use_gae':False, 
#         #     'tau':0.95, 
#         #     'entropy_coef':0.001, 
#         #     'value_loss_coef':0.5, 
#         #     'max_grad_norm':0.5, 
#         #     'num_steps':128, 
#         #     'num_stack':1, 
#         #     'num_frames':200000000, 
#         #     'use_em_loss':False, 
#         #     'skip_eval':False, 
#         #     'stoc_evals':False, 
#         #     'model_dir':'', 
#         #     'out_dir':'./', 
#         #     'log_interval':100, 
#         #     'job_id':12345, 
#         #     'test_name':'test_000', 
#         #     'use_visdom':False, 
#         #     'visdom_port':8097, 
#         #     'visdom_ip':'http://10.0.0.1'                 
#         # }

#         # scenario = env_args['scenario_dir'] + env_args['scenario'].format(0)
#         # config_env = scenario
#         # env = env_vizdoom.DoomEnvironmentDisappear(
#         #                                 scenario=config_env,
#         #                                 show_window=False,
#         #                                 use_info=True,
#         #                                 use_shaping=False, # * if False rew only +1 if True rew +1 or -1
#         #                                 frame_skip=2,
#         #                                 no_backward_movement=True,
#         #                                 # seed=2
#         #                                 )
#         # env = VizdoomAsGym(env)

#         # env = FrameProcessing(env)
#         # env.metadata = {}

        
        if isinstance(env.observation_space, gym.spaces.Discrete):
            env = _DiscreteToBox(env)
        elif isinstance(env.observation_space, gym.spaces.MultiDiscrete):
            env = _MultiDiscreteToBox(env)
        
        super().__init__(
            env, env_name=env_name, horizon=horizon, start=0, zero_shot=True
        )
        


# if __name__ == "__main__":
#     env = MetaFrozenLake(5, 2, False, True)
#     env.reset()
#     env.render()
#     done = False
#     while not done:
#         action = input()
#         action = {"a": 4, "w": 2, "d": 3, "s": 1, "x": 0}[action]
#         next_state, reward, done, _, info = env.step(action)
#         env.render()
#         print(next_state)
#         print(reward)
#         print(done)
