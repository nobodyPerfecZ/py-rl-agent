from typing import Optional

import torch.nn as nn


def create_mlp(
        input_dim: int,
        output_dim: int,
        architecture: Optional[list[int]],
        activation_fn: Optional[list[nn.Module]],
        bias: bool = True
) -> nn.Sequential:
    """
    Creates a feed forward neural network with the specified architecture, using the given activation functions.

    Args:
        input_dim (int):
            The dimension of the input data.

        output_dim (int):
            The dimension of the output data.

        architecture (list[int], optional):
            A list of the dimension for each hidden layer.

        activation_fn (list[nn.Module], optional):
            A list of activation functions for each hidden layer.

        bias (bool, optional):
            Whether to include bias terms in the network (default is True).

    Returns:
        nn.Sequential:
            The feed forward neural network
    """
    if len(activation_fn) != len(architecture):
        raise ValueError("The number of activation functions should match the number of (hidden) layers!")
    if architecture is None:
        architecture = []
    if activation_fn is None:
        activation_fn = []

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
