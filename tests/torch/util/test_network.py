import unittest

import torch.nn as nn

from pyrlagent.torch.util.network import cnn, cnn_in_features, mlp


class TestNetwork(unittest.TestCase):
    """Tests the methods under network.py."""

    def test_cnn_in_features(self):
        """Tests the cnn_in_features() method."""
        features1 = cnn_in_features(
            obs_shape=(3, 100, 100),
            conv_layers=[
                nn.Conv2d(3, 32, 7),
                nn.MaxPool2d(3),
                nn.Conv2d(32, 64, 5),
                nn.MaxPool2d(3),
            ],
        )
        self.assertEqual(5184, features1)

        features2 = cnn_in_features(
            obs_shape=(3, 50, 50),
            conv_layers=[
                nn.Conv2d(3, 32, 7),
                nn.MaxPool2d(3),
                nn.Conv2d(32, 64, 5),
                nn.MaxPool2d(3),
            ],
        )
        self.assertEqual(576, features2)

    def test_mlp(self):
        """Tests the mlp() method."""
        # Create the MLP model:
        # [
        #   nn.Linear(8, 64),
        #   nn.ReLU(),
        #   nn.Linear(64, 64),
        #   nn.Tanh(),
        #   nn.Linear(64, 1)
        # ]
        network = mlp(
            in_features=8,
            hidden_features=[64, 64],
            out_features=1,
            activation=nn.ReLU,
        )
        linear1, activation1, linear2, activation2, linear3, activation3 = network[:6]

        self.assertIsInstance(linear1, nn.Linear)
        self.assertEqual(8, linear1.in_features)
        self.assertEqual(64, linear1.out_features)

        self.assertIsInstance(activation1, nn.ReLU)

        self.assertIsInstance(linear2, nn.Linear)
        self.assertEqual(64, linear2.in_features)
        self.assertEqual(64, linear2.out_features)

        self.assertIsInstance(activation2, nn.ReLU)

        self.assertIsInstance(linear3, nn.Linear)
        self.assertEqual(64, linear3.in_features)
        self.assertEqual(1, linear3.out_features)

        self.assertIsInstance(activation3, nn.Identity)

    def test_cnn(self):
        """Tests the cnn() method."""
        # Create the CNN model:
        # [
        #   nn.Conv2d(3, 32, 7),
        #   nn.MaxPool2d(3),
        #   nn.Tanh(),
        #   nn.Conv2d(32, 64, 5),
        #   nn.MaxPool2d(3),
        #   nn.Tanh(),
        #   nn.Flatten(),
        #   nn.Linear(6000, 128),
        #   nn.Tanh(),
        #   nn.Linear(128, 1)
        # ]
        network = cnn(
            input_shape=(3, 100, 100),
            hidden_channels=[32, 64],
            hidden_features=[128],
            out_features=1,
            pooling=nn.MaxPool2d,
            activation=nn.Tanh,
            conv_kernel_sizes=[7, 5],
            pooling_kernel_sizes=[3, 3],
        )
        (
            conv1,
            pooling1,
            activation1,
            conv2,
            pooling2,
            activation2,
            flatten,
            linear1,
            activation3,
            linear2,
            activation4,
        ) = network[:11]

        self.assertIsInstance(conv1, nn.Conv2d)
        self.assertEqual(3, conv1.in_channels)
        self.assertEqual(32, conv1.out_channels)
        self.assertEqual((7, 7), conv1.kernel_size)

        self.assertIsInstance(pooling1, nn.MaxPool2d)
        self.assertEqual(3, pooling1.kernel_size)

        self.assertIsInstance(activation1, nn.Tanh)

        self.assertIsInstance(conv2, nn.Conv2d)
        self.assertEqual(32, conv2.in_channels)
        self.assertEqual(64, conv2.out_channels)
        self.assertEqual((5, 5), conv2.kernel_size)

        self.assertIsInstance(pooling2, nn.MaxPool2d)
        self.assertEqual(3, pooling2.kernel_size)

        self.assertIsInstance(activation2, nn.Tanh)

        self.assertIsInstance(flatten, nn.Flatten)

        self.assertIsInstance(linear1, nn.Linear)
        self.assertEqual(5184, linear1.in_features)
        self.assertEqual(128, linear1.out_features)

        self.assertIsInstance(activation3, nn.Tanh)

        self.assertIsInstance(linear2, nn.Linear)
        self.assertEqual(128, linear2.in_features)
        self.assertEqual(1, linear2.out_features)

        self.assertIsInstance(activation4, nn.Identity)


if __name__ == "__main__":
    unittest.main()
