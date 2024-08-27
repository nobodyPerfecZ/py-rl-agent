import torch


def get_device(device: str) -> str:
    """
    Get the device on which to run the neural network.

    Args:
        device (str):
            The device on which to run the neural network

    Returns:
        str:
            The device on which to run the neural network
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")
    return device
