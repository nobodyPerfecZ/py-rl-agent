import unittest

import torch.nn as nn

from pyrlagent.torch.config import NetworkConfig, create_network
from pyrlagent.torch.network import (
    CNNDiscretePGActorCriticNetwork,
    CNNContinuousPGActorCriticNetwork,
    MLPDiscretePGActorCriticNetwork,
    MLPContinuousPGActorCriticNetwork,
)


class TestNetworkConfig(unittest.TestCase):
    """Tests the NetworkConfig class."""

    def setUp(self):
        self.cnn_discrete_id = "pg-cnn-discrete"
        self.cnn_discrete_kwargs = {
            "hidden_channels": [32, 64],
            "hidden_features": [64, 64],
            "pooling": nn.MaxPool2d,
            "activation": nn.ReLU,
            "conv_kernel_sizes": [3, 3],
            "pooling_kernel_sizes": [3, 3],
        }

        self.mlp_discrete_id = "pg-mlp-discrete"
        self.mlp_discrete_kwargs = {
            "hidden_features": [64, 64],
            "activation": nn.ReLU,
        }

        self.cnn_continuous_id = "pg-cnn-continuous"
        self.cnn_continuous_kwargs = {
            "hidden_channels": [32, 64],
            "hidden_features": [64, 64],
            "pooling": nn.MaxPool2d,
            "activation": nn.ReLU,
            "conv_kernel_sizes": [3, 3],
            "pooling_kernel_sizes": [3, 3],
        }

        self.mlp_continuous_id = "pg-mlp-continuous"
        self.mlp_continuous_kwargs = {
            "hidden_features": [64, 64],
            "activation": nn.ReLU,
        }

        self.obs_img_dim = (3, 100, 100)
        self.obs_dim = 10
        self.act_dim = 5

    def test_init(self):
        """Tests the __init__() method."""
        cnn_discrete_config = NetworkConfig(
            id=self.cnn_discrete_id, kwargs=self.cnn_discrete_kwargs
        )
        self.assertEqual(cnn_discrete_config.id, self.cnn_discrete_id)
        self.assertEqual(cnn_discrete_config.kwargs, self.cnn_discrete_kwargs)

        mlp_discrete_config = NetworkConfig(
            id=self.mlp_discrete_id, kwargs=self.mlp_discrete_kwargs
        )
        self.assertEqual(mlp_discrete_config.id, self.mlp_discrete_id)
        self.assertEqual(mlp_discrete_config.kwargs, self.mlp_discrete_kwargs)

        cnn_continuous_config = NetworkConfig(
            id=self.cnn_continuous_id, kwargs=self.cnn_continuous_kwargs
        )
        self.assertEqual(cnn_continuous_config.id, self.cnn_continuous_id)
        self.assertEqual(cnn_continuous_config.kwargs, self.cnn_continuous_kwargs)

        mlp_continuous_config = NetworkConfig(
            id=self.mlp_continuous_id, kwargs=self.mlp_continuous_kwargs
        )
        self.assertEqual(mlp_continuous_config.id, self.mlp_continuous_id)
        self.assertEqual(mlp_continuous_config.kwargs, self.mlp_continuous_kwargs)

    def test_create_network(self):
        """Tests the create_network() method."""
        # Create an invalid network
        with self.assertRaises(ValueError):
            create_network(
                network_config=NetworkConfig(id="invalid_id", kwargs={}),
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
            )

        # Create a CNN discrete network
        cnn_discrete = create_network(
            network_config=NetworkConfig(
                id=self.cnn_discrete_id, kwargs=self.cnn_discrete_kwargs
            ),
            obs_dim=self.obs_img_dim,
            act_dim=self.act_dim,
        )
        self.assertIsInstance(cnn_discrete, CNNDiscretePGActorCriticNetwork)

        # Create a MLP discrete network
        mlp_discrete = create_network(
            network_config=NetworkConfig(
                id=self.mlp_discrete_id, kwargs=self.mlp_discrete_kwargs
            ),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        self.assertIsInstance(mlp_discrete, MLPDiscretePGActorCriticNetwork)

        # Create a CNN continuous network
        cnn_continuous = create_network(
            network_config=NetworkConfig(
                id=self.cnn_continuous_id, kwargs=self.cnn_continuous_kwargs
            ),
            obs_dim=self.obs_img_dim,
            act_dim=self.act_dim,
        )
        self.assertIsInstance(cnn_continuous, CNNContinuousPGActorCriticNetwork)

        # Create a MLP continuous network
        mlp_continuous = create_network(
            network_config=NetworkConfig(
                id=self.mlp_continuous_id, kwargs=self.mlp_continuous_kwargs
            ),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
        )
        self.assertIsInstance(mlp_continuous, MLPContinuousPGActorCriticNetwork)


if __name__ == "__main__":
    unittest.main()
