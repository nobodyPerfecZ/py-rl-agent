from typing import Optional

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):

    def __init__(self, feature_extractor: Optional[nn.Module], values: nn.Module, advantages: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.values = values
        self.advantages = advantages

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x) if self.feature_extractor else x
        value = self.values(features)
        advantage = self.advantages(features)
        return value + (advantage - advantage.mean(dim=-1).unsqueeze(-1))


def create_mlp(
        input_dim: int,
        output_dim: int,
        architecture: Optional[list[int]],
        activation_fn: Optional[list[nn.Module]],
        bias: bool = True
) -> nn.Module:
    """
    Creates a feed forward neural network with the specified architecture and activation functions.

    Args:
        input_dim (int):
            The dimension of the input data.

        output_dim (int):
            The dimension of the output data.

        architecture (list[int]):
            A list of the output dimensions of each layer.

        activation_fn (list[nn.Module], optional):
            A list of activation functions of each layer.

        bias (bool, optional):
            Whether to include bias terms in the network (default is True).

    Returns:
        nn.Module:
            The feed forward neural network
    """
    if architecture is None:
        architecture = []
    if activation_fn is None:
        activation_fn = []
    if len(activation_fn) != len(architecture) and len(activation_fn) - 1 != len(architecture):
        raise ValueError("The number of activation functions should match the number of (hidden) layers!")

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
    if len(activation_fn) - 1 == len(architecture):
        modules += [nn.Linear(hidden_dim, output_dim, bias=bias), activation_fn[-1]]
    else:
        modules += [nn.Linear(hidden_dim, output_dim, bias=bias)]
    return nn.Sequential(*modules)


def create_dueling_mlp(
        input_dim: int,
        output_dim: int,
        feature_architecture: Optional[list[int]],
        feature_activation_fn: Optional[list[nn.Module]],
        value_architecture: Optional[list[int]],
        value_activation_fn: Optional[list[nn.Module]],
        advantage_architecture: Optional[list[int]],
        advantage_activation_fn: Optional[list[nn.Module]],
        bias: bool = True
) -> nn.Module:
    """
    Creates a dueling network with the specified architecture and activation functions
    for the feature extractor f(s) value function V(f(s)) and advantages A(f(s), a).

    The corresponding paper can be found here:
    https://arxiv.org/abs/1511.06581

    Args:
        input_dim (int):
            The dimension of the input data.

        output_dim (int):
            The dimension of the output data.

        feature_architecture (list[int], optional):
             A list of the output dimension of each hidden layer for f(s).

        feature_activation_fn (list[nn.Module], optional):
             A list of the output dimension of each hidden layer for f(s).

        value_architecture (list[int], optional):
             A list of the output dimension of each hidden layer for V(f(s)).

        value_activation_fn (list[nn.Module], optional):
            A list of activation functions of each hidden layer for V(f(s)).

        advantage_architecture (list[int], optional):
            A list of output dimensions of each hidden layer for A(f(s), a).

        advantage_activation_fn (list[nn.Module], optional):
            A list of activation functions of each hidden layer for A(f(s), a).

        bias (bool):
            Whether to include bias terms in the network (default is True).

    Returns:
        tuple[nn.Sequential, nn.Sequential]:
            values (nn.Sequential):
                The neural network that approximate the value function V(s)

            advantages (nn.Sequential):
                The neural network that approximate the advantages A(s,a)
    """
    if ((feature_architecture is None or feature_architecture == []) and
            (feature_activation_fn is None or feature_activation_fn == [])):
        # Case: No feature extractor used
        feature_output = input_dim
        feature_extractor = None
    else:
        # Case: Create the feature extractor
        feature_output = feature_architecture[-1]
        feature_extractor = create_mlp(
            input_dim=input_dim,
            output_dim=feature_output,
            architecture=feature_architecture[:-1],
            activation_fn=feature_activation_fn,
            bias=bias,
        )

    # Create the value function architecture
    values = create_mlp(
        input_dim=feature_output,
        output_dim=1,
        architecture=value_architecture,
        activation_fn=value_activation_fn,
        bias=bias,
    )

    # Create the advantage function architecture
    advantages = create_mlp(
        input_dim=feature_output,
        output_dim=output_dim,
        architecture=advantage_architecture,
        activation_fn=advantage_activation_fn,
        bias=bias,
    )

    return DuelingNetwork(feature_extractor, values, advantages)
