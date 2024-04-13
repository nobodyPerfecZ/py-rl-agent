from typing import Optional

from torch import nn


def create_mlp(
        input_dim: int,
        output_dim: int,
        architecture: Optional[list[int]] = None,
        activation_fn: Optional[nn.Module] = None,
        output_activation_fn: Optional[nn.Module] = None,
        bias: bool = True,
) -> nn.Module:
    """
    Creates a feed forward neural network with the specified architecture and activation functions.

    Args:
        input_dim (int):
            The dimension of the input data.

        output_dim (int):
            The dimension of the output data.

        architecture (list[int]):
            A list of the output dimensions of each hidden layer.

        activation_fn (nn.Module, optional):
            The activation function of each hidden layer.

        output_activation_fn (nn.Module, optional):
            The activation function after the output layer.

        bias (bool, optional):
            Whether to include bias terms in the network (default is True).

    Returns:
        nn.Module:
            The feed forward neural network
    """
    if architecture is None:
        architecture = []

    # Create the deep neural network
    modules = []
    for i in range(len(architecture)):
        modules.append(nn.Linear(input_dim if i == 0 else architecture[i - 1], architecture[i], bias=bias))
        if activation_fn:
            modules.append(activation_fn)

    # Add the output layer
    modules.append(nn.Linear(architecture[-1] if architecture else input_dim, output_dim, bias=bias))

    if output_activation_fn:
        modules.append(output_activation_fn)

    return nn.Sequential(*modules)
