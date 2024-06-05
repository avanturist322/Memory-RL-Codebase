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

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs['image']
        self._last_frame = obs.copy()
        obs = obs.astype(np.float32) / 255
        self.env.metadata = []
        return obs, info
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs['image']
        self._last_frame = obs.copy()
        obs = obs.astype(np.float32) / 255
        self.env.metadata = []
        return obs, reward, done, info
    
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

    print(env.observation_space)
    env.metadata = []
    return env 


"""

[2024-06-05 21:45:49,267][src.trainers.trainers_control][INFO] - Using async env
[2024-06-05 21:45:50,173][src.trainers.trainers_control][INFO] - Observation space: Box(-1.0, 1.0, (4,), float32)
[2024-06-05 21:45:50,173][src.trainers.trainers_control][INFO] - Action space: Discrete(4)
/home/jovyan/Egor_C/REPOSITORIES/Memory-RL-Codebase/models/Recurrent_Linear_Transformers/src/agents/base_agent.py:69: DeprecationWarning: jax.tree_map is deprecated: use jax.tree.map (jax v0.4.25 or newer) or jax.tree_util.tree_map (any JAX version).
  self.h_tickminus1=jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x,axis=0),self.env.num_envs,axis=0),self.seq_init())
/home/jovyan/Egor_C/REPOSITORIES/Memory-RL-Codebase/models/Recurrent_Linear_Transformers/src/agents/base_agent.py:73: DeprecationWarning: jax.tree_map is deprecated: use jax.tree.map (jax v0.4.25 or newer) or jax.tree_util.tree_map (any JAX version).
  return sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape),params)))
[2024-06-05 21:45:57,568][src.agents.base_agent][INFO] - Total Number of params: 1772741
[2024-06-05 21:45:57,569][src.agents.base_agent][INFO] - Number of params in Seq Model: 1739072
111111111111111111
<src.trainers.trainers_control.ControlTrainer object at 0x7f466c094f40>

"""