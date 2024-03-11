from typing import Union

import gymnasium as gym


def get_env(name: str, return_render: bool = False) -> Union[gym.Env, tuple[gym.Env, gym.Env]]:
    """
    Creates a Gymnasium environment given the name.

    Args:
        name (str):
            The name of the Gymnasium environment

        return_render (bool):
            Controls whether to return a rendered environment

    Returns:
        gym.Env | tuple[gym.Env, gym.Env]:
            env (gym.Env):
                The Gymnasium environment (without rendering)

            render_env (gym.Env, optional):
                The Gymnasium environment with rendering
    """
    env = gym.make(name, render_mode="rgb_array")
    if return_render:
        render_env = gym.make(name, render_mode="human")
        return env, render_env
    return env
