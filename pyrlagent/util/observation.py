import numpy as np
import torch


def obs_to_tensor(observation: np.ndarray) -> torch.Tensor:
    """
    Converts the given numpy observation to a tensor object.

    Args:
        observation (np.ndarray):
            The observation to convert

    Returns:
        torch.Tensor:
            The converted observation
    """
    if is_rgb_image_observation(observation):
        # Case: observation is a (batch of) rgb-images
        # Transform image format from (B, H, W, C) to (B, C, H, W)
        transformed = observation.transpose(0, 3, 1, 2)
        return torch.from_numpy(transformed)
    else:
        # Case: observation is an array of different values
        return torch.from_numpy(observation)


def is_rgb_image_observation(observation: np.ndarray) -> bool:
    """
    Returns True if the given observation contains rgb images.

    Args:
        observation (np.ndarray):
            The observation you want to check.

    Returns:
        bool:
            True if the given observation contains rgb images
    """
    if observation.ndim <= 3:
        # Case: Observation is no rgb-image
        return False
    elif observation.ndim == 4:
        # Case: Observation has the format of rgb-images: (B, H, W, C)
        return True
    else:
        # Case: Observation has unknown format
        raise ValueError(
            "Unknown format of the observation. "
            "The observation should be in a format of (B, H, W, C) for rgb-images!"
        )
