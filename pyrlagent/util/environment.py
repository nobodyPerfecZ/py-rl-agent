from typing import Type, Union

import gymnasium as gym


def get_env(name: str, **env_kwargs) -> gym.Env:
    """
    Creates a single Gymnasium environment given the name and kwargs.

    Args:
        name (str):
            The name of the Gymnasium environment

        **env_kwargs:
            Additional parameters to the environment

    Returns:
        gym.Env:
            The Gymnasium environment with the specified kwargs
    """
    env = gym.make(name, **env_kwargs)
    return env


def get_vector_env(name: str, num_envs: int, **env_kwargs) -> gym.experimental.vector.VectorEnv:
    """
    Creates NUM_ENVS Gymnasium environments given the name and kwargs.

    Args:
        name (str):
            The name of the Gymnasium environments

        num_envs (int):
            The number of parallel Gymnasium environments

        **env_kwargs:
            Additional parameters to the environments

    Returns:
        gym.experimental.vector.VectorEnv:
            The Gymnasium environment with the specified kwargs
    """
    envs = gym.make_vec(name, num_envs=num_envs, **env_kwargs)
    return envs


def transform_env(env: gym.Env, wrappers: Union[Type[gym.Wrapper], list[Type[gym.Wrapper]]]) -> gym.Env:
    """
    Transform the Gymnasium environment with the given wrappers.

    Args:
        env (gym.Env):
            The Gymnasium environment we want to transform.

        wrappers (Type[gym.Wrapper] | list[Type[gym.Wrapper]]):
            Single value or a list of Gymnasium wrappers to transform the environment

    Returns:
        gym.Env:
            The transformed Gymnasium environment
    """
    if not isinstance(wrappers, list):
        # Case: single wrapper is given
        env = wrappers(env)
    else:
        # Case: multiple wrappers are given
        for wrapper in wrappers:
            env = wrapper(env)
    return env