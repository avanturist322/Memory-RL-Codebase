import gymnasium as gym
import numpy as np
import os

# to enable rendering on a headless machine
os.environ['MUJOCO_GL']='osmesa'

def create_mujoco_env(env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda rwd: np.clip(rwd, -10, 10))
    return env 