import gymnasium as gym
# import gym
import numpy as np
import os

# to enable rendering on a headless machine
os.environ['MUJOCO_GL']='osmesa'

class FrameProcessing(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_frame = obs.copy()
        obs = obs.astype(np.float32) / 255
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._last_frame = obs.copy()
        obs = obs.astype(np.float32) / 255
        return obs, reward, done, truncated, info
    
    def render(self):
        return self._last_frame

def create_memorymaze(env_name):

    env = gym.make(env_name)

    env = FrameProcessing(env) # !
    return env 