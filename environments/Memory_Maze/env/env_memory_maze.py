import gym
import logging

logging.disable(logging.WARNING)
gym.logger.set_level(40)


def create_env(name: str = 'memory_maze:MemoryMaze-9x9-v0', seed: int = 42):
    """
    'memory_maze:MemoryMaze-9x9-v0'
    'memory_maze:MemoryMaze-11x11-v0'
    'memory_maze:MemoryMaze-13x13-v0'
    'memory_maze:MemoryMaze-15x15-v0'
    """

    env = gym.make(name, seed=seed)
    
    return env
