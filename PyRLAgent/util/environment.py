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
