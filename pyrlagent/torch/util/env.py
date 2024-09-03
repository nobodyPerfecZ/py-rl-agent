import gymnasium as gym


def get_env(
    env_id: str,
    device: str,
    **env_kwargs,
) -> gym.Env:
    """
    Creates an environment given the id and its kwargs.

    Args:
        env_id (str):
            The identifier of the environment

        **env_kwargs:
            Additional parameters to the environment

    Returns:
        gym.Env:
            The environment with the specified kwargs
    """
    # Create the environment
    env = gym.make(env_id, **env_kwargs)

    # Wrap the environment with the appropriate wrappers
    if isinstance(env.action_space, gym.spaces.Discrete):
        # Case: Environment has discrete action spaces
        return gym.wrappers.NumpyToTorch(
            env=gym.wrappers.Autoreset(env),
            device=device,
        )
    else:
        # Case: Environment has continuous action spaces
        return gym.wrappers.NumpyToTorch(
            env=gym.wrappers.Autoreset(gym.wrappers.ClipAction(env)),
            device=device,
        )


def get_vector_env(
    env_id: str,
    num_envs: int,
    device: str,
    **env_kwargs,
) -> gym.vector.VectorEnv:
    """
    Creates NUM_ENVS environments given the name and kwargs.

    Args:
        env_id (str):
            The identifier of the environment

        num_envs (int):
            The number of parallel environments

        **env_kwargs:
            Additional parameters to the environments

    Returns:
        gym.vector.VectorEnv:
            The environment with the specified kwargs
    """
    # Create the vectorized environment
    envs = gym.make_vec(
        id=env_id,
        num_envs=num_envs,
        vectorization_mode="sync",
        **env_kwargs,
    )

    # Wrap the environments with the appropriate wrappers
    if isinstance(envs.single_action_space, gym.spaces.Discrete):
        # Case: Environments have continuous action spaces
        return gym.wrappers.vector.NumpyToTorch(
            env=envs,
            device=device,
        )
    else:
        # Case: Environments have discrete action spaces
        return gym.wrappers.vector.NumpyToTorch(
            env=gym.wrappers.vector.ClipAction(envs),
            device=device,
        )


def get_obs_act_space(env: gym.Env) -> tuple[gym.spaces.Space, gym.spaces.Space]:
    """
    Get the observation and action space of the environment.

    Args:
        env (gym.Env):
            The gymnasium environment

    Returns:
        tuple[gym.spaces.Space, gym.spaces.Space]:
            The observation and action space
    """
    if isinstance(env, gym.vector.VectorEnv):
        return env.single_observation_space, env.single_action_space
    return env.observation_space, env.action_space


def get_obs_act_dims(
    obs_space: gym.spaces.Space,
    act_space: gym.spaces.Space,
) -> tuple[int, int]:
    """
    Get the observation and action dimensions of the environment.

    Args:
        env (gym.Env):
            The gymnasium environment

    Returns:
        tuple[int, int]:
            The observation and action dimensions
    """
    obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
    act_dim = (
        act_space.n
        if isinstance(act_space, gym.spaces.Discrete)
        else act_space.shape[0]
    )
    return obs_dim, act_dim
