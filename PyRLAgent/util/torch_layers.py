from typing import List

import torch.nn as nn


def create_mlp(
        input_dim: int,
        output_dim: int,
        architecture: List[int] = [],
        activation_fn: List[nn.Module] = [],
        bias: bool = True
) -> nn.Sequential:
    """
    This method constructs a feedforward neural network with the specified architecture, using the given
    activation functions.

    Args:
        input_dim (int):
            The dimension of the input data.

        output_dim (int):
            The dimension of the output data.

        architecture (list[int]):
            A list specifying the dimension of each hidden layer.

        activation_fn (list[nn.Module]):
            A list of activation functions for each hidden layer.

        bias (bool, optional):
            Whether to include bias terms in the network (default is True).

    Returns:
        nn.Sequential:
            A Pytorch MLP model.

    Raises:
        ValueError:
            If the number of activation functions does not match the number of hidden layers.

    Example:
        To create a simple neural network with two hidden layers, you can call this function like this:

        >>> # Create the following neural network:
        >>> # [nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 10)]
        >>> input_dim = 64
        >>> output_dim = 10
        >>> architecture = [128, 64]
        >>> activation_fn = [nn.ReLU(), nn.Tanh()]
        >>> model = create_architecture(input_dim, output_dim, architecture, activation_fn)
    """
    if len(activation_fn) != len(architecture):
        raise ValueError("The number of activation functions should match the number of (hidden) layers.")

    # Create the deep neural network
    modules = []
    if len(architecture) > 0:
        # Case: At least one hidden layer should be added to modules
        modules += [nn.Linear(input_dim, architecture[0], bias=bias), activation_fn[0]]

    for i in range(len(architecture) - 1):
        # Case: More than one hidden layer should be added to modules
        modules += [nn.Linear(architecture[i], architecture[i + 1], bias=bias), activation_fn[i + 1]]

    # Calculate the hidden dimension from the last hidden layer
    hidden_dim = architecture[-1] if len(architecture) > 0 else input_dim

    # Add the output layer to modules
    modules += [nn.Linear(hidden_dim, output_dim, bias=bias)]
    return nn.Sequential(*modules)
