import torch
import torch.nn as nn


def cnn_in_features(
    input_shape: tuple[int, int, int],
    conv_layers: list[nn.Module],
) -> int:
    """
    Computes the in_features for the first linear layer in a CNN.

    This method supports only nn.Conv2d, nn.MaxPool2d, and nn.AvgPool2d layers.

    Args:
        input_shape (tuple[int, int, int]):
            The shape of the input observation in (C, H, W) format

        conv_layers (list):
            The list of convolutional and pooling layers

    Returns:
        int:
            The number of in_features for the first linear layer
    """
    c, h, w = input_shape
    for layer in conv_layers:
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

    Returns:
        nn.Module:
            The MLP network

    Examples:
        >>> mlp(4, [64, 64], 2, nn.ReLU)
        Sequential(
          (0): Linear(in_features=4, out_features=64, bias=True)
          (1): ReLU()
          (2): Linear(in_features=64, out_features=64, bias=True)
          (3): ReLU()
          (4): Linear(in_features=64, out_features=2, bias=True)
        )
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
    pooling: nn.Module,
    activation: nn.Module,
    conv_kernel_sizes: list[int],
    pooling_kernel_sizes: list[int],
) -> nn.Module:
    """
    Creates a Convolutional Neural Network (CNN) network.

    Args:
        input_shape (tuple[int, int, int]):
            The shape of the input observation in (C, H, W) format

        hidden_channels (list[int]):
            The number of hidden channels for each convolutional layer

        pooling (nn.Module):
            The pooling layer after each convolutional layer

        activation (nn.Module):
            The activation function after each pooling layer

        conv_kernel_sizes (list[int]):
            The kernel sizes for each convolutional layer

        pooling_kernel_sizes (list[int]):
            The kernel sizes for each pooling layer

    Returns:
        nn.Module:
            The CNN network

    Examples:
        >>> cnn((3, 100, 100), [32, 64], nn.MaxPool2d, nn.Tanh, [7, 5], [3, 3])
        Sequential(
            (0): Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1))
            (1): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
            (2): Tanh()
            (3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
            (4): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
            (5): Tanh()
            (6): Flatten(start_dim=1, end_dim=-1)
        )
    """
    if any(i <= 0 for i in input_shape):
        raise ValueError("input_shape must be greater than 0.")
    if any(hc <= 0 for hc in hidden_channels):
        raise ValueError("hidden_channels must be greater than 0.")
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
    cnn_layers += [nn.Flatten()]
    return nn.Sequential(*cnn_layers)


def cnn_mlp(
    input_shape: tuple[int, int, int],
    hidden_channels: list[int],
    hidden_features: list[int],
    out_features: int,
    pooling: nn.Module,
    activation: nn.Module,
    conv_kernel_sizes: list[int],
    pooling_kernel_sizes: list[int],
) -> tuple[nn.Module, nn.Module]:
    """
    Creates a Convolutional Neural Network (CNN) followed by Multi-Layer Perceptron (MLP) network.

    Args:
        input_shape (tuple[int, int, int]):
            The shape of the input observation in (C, H, W) format

        hidden_channels (list[int]):
            The number of hidden channels for each convolutional layer

        hidden_features (int):
            The number of hidden features for each linear layer

        pooling (nn.Module):
            The pooling layer after each convolutional layer

        activation (nn.Module):
            The activation function after each pooling layer / linear layer

        conv_kernel_sizes (list[int]):
            The kernel sizes for each convolutional layer

        pooling_kernel_sizes (list[int]):
            The kernel sizes for each pooling layer

    Returns:
        tuple[nn.Module, nn.Module]:
            cnn_net (nn.Module):
                The CNN part of the network

            mlp_net (nn.Module):
                The MLP part of the network

    Examples:
        >>> cnn_net, mlp_net = cnn_mlp((3, 100, 100), [32, 64], [128], 1, nn.MaxPool2d, nn.Tanh, [7, 5], [3, 3])
        >>> cnn_net
        Sequential(
            (0): Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1))
            (1): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
            (2): Tanh()
            (3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
            (4): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
            (5): Tanh()
            (6): Flatten(start_dim=1, end_dim=-1)
        )
        >>> mlp_net
        Sequential(
            (0): Linear(in_features=5184, out_features=128, bias=True)
            (1): Tanh()
            (2): Linear(in_features=128, out_features=1, bias=True)
            (3): Identity()
        )
    """
    # Build the CNN layers
    cnn_net = cnn(
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        pooling=pooling,
        activation=activation,
        conv_kernel_sizes=conv_kernel_sizes,
        pooling_kernel_sizes=pooling_kernel_sizes,
    )

    # Build the MLP layers
    mlp_net = mlp(
        in_features=cnn_in_features(input_shape, list(cnn_net.modules())),
        hidden_features=hidden_features,
        out_features=out_features,
        activation=activation,
    )

    return cnn_net, mlp_net
