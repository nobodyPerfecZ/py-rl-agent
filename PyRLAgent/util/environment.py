from typing import Optional, Type

import gymnasium as gym


def get_env(name: str, **env_kwargs) -> gym.Env:
    """
    Creates a Gymnasium environment given the name and kwargs.

    Args:
        name (str):
            The name of the Gymnasium environment

    Returns:
        gym.Env:
            The Gymnasium environment with the specified kwargs
    """
    env = gym.make(name, **env_kwargs)
    return env


def transform_env(env: gym.Env, wrappers: Optional[list[Type[gym.Wrapper]]] = None) -> gym.Env:
    """
    Transform the Gymnasium environment with the given wrappers.

    If no wrappers are provided, then the given environment will be returned.

    Args:
        env (gym.Env):
            The Gymnasium environment we want to transform.

        wrappers (list[Type[gym.Wrapper]], optional):
            The list of Gymnasium wrappers to transform the environment

    Returns:
        gym.Env:
            The transformed Gymnasium environment
    """
    if wrappers is None:
        # Case: No wrappers are provided
        return env

    for wrapper in wrappers:
        env = wrapper(env)
    return env
