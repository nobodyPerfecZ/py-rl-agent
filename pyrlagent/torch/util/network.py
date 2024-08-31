import torch
import torch.nn as nn


def cnn_in_features(obs_shape: tuple[int, int, int], conv_layers: list[nn.Module]):
    """
    Computes the in_features for the first linear layer in a CNN.
    This method supports only nn.Conv2d, nn.MaxPool2d, and nn.AvgPool2d layers.

    Args:
        obs_shape (tuple[int, int, int]):
            The shape of the input observation in (C, H, W) format

        conv_layers (list):
            The list of convolutional and pooling layers

    Returns:
        int:
            The number of in_features for the first linear layer
    """
    c, h, w = obs_shape
    for layer in conv_layers:
        # Update height and width
        if isinstance(layer, nn.Conv2d):
            h = (
                h
                + 2 * layer.padding[0]
                - layer.dilation[0] * (layer.kernel_size[0] - 1)
                - 1
            ) // layer.stride[0] + 1
            w = (
                w
                + 2 * layer.padding[1]
                - layer.dilation[1] * (layer.kernel_size[1] - 1)
                - 1
            ) // layer.stride[1] + 1
            c = layer.out_channels
        elif isinstance(layer, nn.MaxPool2d):
            h = (
                h + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1
            ) // layer.stride + 1
            w = (
                w + 2 * layer.padding - layer.dilation * (layer.kernel_size - 1) - 1
            ) // layer.stride + 1
        elif isinstance(layer, nn.AvgPool2d):
            h = (h + 2 * layer.padding - layer.kernel_size) // layer.stride + 1
            w = (w + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

    # Compute the in_features for the first linear layer
    in_features = c * h * w
    return in_features


def mlp(
    in_features: int,
    hidden_features: list[int],
    out_features: int,
    activation: nn.Module,
) -> nn.Module:
    """
    Creates a Multi-Layer Perceptron (MLP) network.

    Args:
        in_features (int):
            The number of input features

        hidden_features (int):
            The number of hidden features

        out_features (int):
            The number of output features

        activations (nn.Module):
            The activation functions for each layer

        activation_kwargs (dict):
            The arguments for the activation function

    Examples:
        >>> # TODO: Add examples

    Returns:
        nn.Module:
            The MLP network
    """
    if in_features <= 0:
        raise ValueError("in_features must be greater than 0.")
    if any(hf <= 0 for hf in hidden_features):
        raise ValueError("hidden_features must be greater than 0.")
    if out_features <= 0:
        raise ValueError("out_features must be greater than 0.")

    # Combine the features together
    features = [in_features] + hidden_features + [out_features]

    layers = []
    for j in range(len(features) - 1):
        activation = activation if j < len(features) - 2 else nn.Identity
        layers += [
            nn.Linear(
                in_features=features[j],
                out_features=features[j + 1],
                dtype=torch.float32,
            ),
            activation(),
        ]
    return nn.Sequential(*layers)


def cnn(
    input_shape: tuple[int, int, int],
    hidden_channels: list[int],
    hidden_features: list[int],
    out_features: int,
    pooling: nn.Module,
    activation: nn.Module,
    conv_kernel_sizes: list[int],
    pooling_kernel_sizes: list[int],
) -> nn.Module:
    """
    # TODO: Add documentation

    Args:
        input_shape (tuple[int, int, int]): _description_
        hidden_channels (list[int]): _description_
        hidden_features (list[int]): _description_
        out_features (int): _description_
        pooling (nn.Module): _description_
        activation (nn.Module): _description_
        conv_kernel_sizes (list[int]): _description_
        pooling_kernel_sizes (list[int]): _description_

    Examples:
        >>> # TODO: Add examples here

    Returns:
        nn.Module: _description_
    """
    if any(i <= 0 for i in input_shape):
        raise ValueError("input_shape must be greater than 0.")
    if any(hc <= 0 for hc in hidden_channels):
        raise ValueError("hidden_channels must be greater than 0.")
    if any(hf <= 0 for hf in hidden_features):
        raise ValueError("hidden_features must be greater than 0.")
    if out_features <= 0:
        raise ValueError("out_features must be greater than 0.")
    if any(cks <= 0 for cks in conv_kernel_sizes):
        raise ValueError("conv_kernel_sizes must be greater than 0.")
    if any(pks <= 0 for pks in pooling_kernel_sizes):
        raise ValueError("pooling_kernel_sizes must be greater than 0.")
    if len(hidden_channels) != len(conv_kernel_sizes):
        raise ValueError(
            "hidden_channels and conv_kernel_sizes must have the same length."
        )
    if len(hidden_channels) != len(pooling_kernel_sizes):
        raise ValueError(
            "hidden_channels and pooling_kernel_sizes must have the same length."
        )

    # Combine the channels together
    cnn_channels = [input_shape[0]] + hidden_channels

    # Build the CNN layers
    cnn_layers = []
    for j in range(len(cnn_channels) - 1):
        pooling = pooling if j < len(cnn_channels) - 1 else nn.Identity
        activation = activation if j < len(cnn_channels) - 1 else nn.Identity
        cnn_layers += [
            nn.Conv2d(
                in_channels=cnn_channels[j],
                out_channels=cnn_channels[j + 1],
                kernel_size=conv_kernel_sizes[j],
                dtype=torch.float32,
            ),
            pooling(kernel_size=pooling_kernel_sizes[j]),
            activation(),
        ]

    # Combine the features together
    in_features = cnn_in_features(input_shape, cnn_layers)
    features = [in_features] + hidden_features + [out_features]

    # Build the MLP layers
    mlp_layers = []
    for j in range(len(features) - 1):
        activation = activation if j < len(features) - 2 else nn.Identity
        mlp_layers += [
            nn.Linear(in_features=features[j], out_features=features[j + 1]),
            activation(),
        ]

    layers = cnn_layers + [nn.Flatten()] + mlp_layers
    return nn.Sequential(*layers)
