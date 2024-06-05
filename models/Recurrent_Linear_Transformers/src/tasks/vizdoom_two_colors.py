# import gym
import numpy as np
import gymnasium as gym

import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import os
# to enable rendering on a headless machine
os.environ['MUJOCO_GL']='osmesa'

from environments.ViZDoom_Two_Colors.env import env_vizdoom, VizdoomAsGym

class FrameProcessing(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env.new_step_api = False

    def reset(self, **kwargs):
        # print('im the reset method and im working')
        data = self.env.reset(**kwargs)
        obs = data['image']
        self._last_frame = obs.copy()
        obs = obs.astype(np.float32) / 255.
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
        obs = obs.astype(np.float32) / 255.
        # print('ok i love gymnasium')

        return obs, reward, done, False, info
    
    def render(self):
        return self._last_frame

def create_vizdoom_two_colors(env_name):
    if env_name == 'doom_with_pillar':
        config_env = 'environments/ViZDoom_Two_Colors/env_configs/custom_scenario000.cfg'
    elif env_name == 'doow_without_pillar':
        config_env = 'environments/ViZDoom_Two_Colors/env_configs/custom_scenario_no_pil000.cfg'

    env = env_vizdoom.DoomEnvironmentDisappear(
                                                scenario=config_env,
                                                show_window=False,
                                                use_info=True,
                                                use_shaping=False, # * if False rew only +1 if True rew +1 or -1
                                                frame_skip=2,
                                                no_backward_movement=True,
                                                # seed=2,
                                                )

    env = VizdoomAsGym(env)

    env = FrameProcessing(env)

    # print(111111111111111111)

    env.metadata = []
    

    return env 