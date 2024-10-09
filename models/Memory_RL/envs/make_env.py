import gym
from gym.wrappers import RescaleAction
from envs.minigrid_memory import Minigrid


def make_env(
    env_name: str,
    seed: int,
) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids and env_name != 'MiniGrid-MemoryS13-v0':
        env = gym.make(env_name)
        print(50 * '#' + f' Made env: {env_name} ' + 50 * '#' )

        env.max_episode_steps = getattr(env, "max_episode_steps", env.spec.max_episode_steps)


    elif env_name == 'MiniGrid-MemoryS13-v0':
        env = Minigrid(env_name, 31)
        print(50 * '#' + f' Initialized env: {env_name} ' + 50 * '#' )



    if isinstance(env.action_space, gym.spaces.Box):
        env = RescaleAction(env, -1.0, 1.0)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    print("obs space", env.observation_space)
    print("act space", env.action_space)

    return env
