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


def create_dueling_mlp(
        input_dim: int,
        output_dim: int,
        feature_architecture: Optional[list[int]] = None,
        feature_activation_fn: Optional[nn.Module] = None,
        feature_output_activation_fn: Optional[nn.Module] = None,
        value_architecture: Optional[list[int]] = None,
        value_activation_fn: Optional[nn.Module] = None,
        value_output_activation_fn: Optional[nn.Module] = None,
        advantage_architecture: Optional[list[int]] = None,
        advantage_activation_fn: Optional[nn.Module] = None,
        advantage_output_activation_fn: Optional[nn.Module] = None,
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
             The activation function of each hidden layer for f(s).

        feature_output_activation_fn (nn.Module, optional):
            The activation function after the output layer for f(s).

        value_architecture (list[int], optional):
             A list of the output dimension of each hidden layer for V(f(s)).

        value_activation_fn (list[nn.Module], optional):
           The activation function of each hidden layer for V(f(s)).

        value_output_activation_fn (nn.Module, optional):
            The activation function after the output layer for V(f(s)).

        advantage_architecture (list[int], optional):
            A list of output dimensions of each hidden layer for A(f(s), a).

        advantage_activation_fn (list[nn.Module], optional):
            The activation function of each hidden layer for A(f(s), a).

        advantage_output_activation_fn (nn.Module, optional):
            The activation function after the output layer for A(f(s), a).

        bias (bool):
            Whether to include bias terms in the network (default is True).

    Returns:
        nn.Module:
            The dueling network
    """
    if not feature_architecture:
        # Case: Use no feature extractor
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
            output_activation_fn=feature_output_activation_fn,
            bias=bias,
        )

    # Create the value function architecture
    values = create_mlp(
        input_dim=feature_output,
        output_dim=1,
        architecture=value_architecture,
        activation_fn=value_activation_fn,
        output_activation_fn=value_output_activation_fn,
        bias=bias,
    )

    # Create the advantage function architecture
    advantages = create_mlp(
        input_dim=feature_output,
        output_dim=output_dim,
        architecture=advantage_architecture,
        activation_fn=advantage_activation_fn,
        output_activation_fn=advantage_output_activation_fn,
        bias=bias,
    )

    return DuelingNetwork(feature_extractor, values, advantages)
