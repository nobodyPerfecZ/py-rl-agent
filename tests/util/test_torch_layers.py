import unittest
import numpy as np
import torch.nn as nn

from PyRLAgent.util.torch_layers import create_mlp


class TestTorchLayers(unittest.TestCase):
    """
    Tests all methods from util.torch_layers.
    """

    def test_create_mlp(self):
        """
        Tests the method create_mlp().
        """
        # Create the minimal neural network:
        # [nn.Linear(64, 10, bias=True)]
        input_dim = 64
        output_dim = 10
        architecture = []
        activation_fn = []
        bias = False

        model = create_mlp(input_dim, output_dim, architecture, activation_fn, bias)
        linear = model[0]

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(1, len(model))
        self.assertEqual(input_dim, linear.in_features)
        self.assertEqual(output_dim, linear.out_features)
        self.assertIsNone(linear.bias)

        # Create the following neural network:
        # [nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 10)]
        input_dim = 64
        output_dim = 10
        architecture = [128, 64]
        activation_fn = [nn.ReLU(), nn.Tanh()]
        bias = True

        model = create_mlp(input_dim, output_dim, architecture, activation_fn, bias)
        linear1, act1, linear2, act2, linear3 = model[0], model[1], model[2], model[3], model[4]

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(5, len(model))

        self.assertEqual(input_dim, linear1.in_features)
        self.assertEqual(architecture[0], linear1.out_features)
        self.assertIsNotNone(linear1.bias)

        self.assertIsInstance(act1, nn.ReLU)

        self.assertEqual(architecture[0], linear2.in_features)
        self.assertEqual(architecture[1], linear2.out_features)
        self.assertIsNotNone(linear2.bias)

        self.assertIsInstance(act2, nn.Tanh)

        self.assertEqual(architecture[1], linear3.in_features)
        self.assertEqual(output_dim, linear3.out_features)
        self.assertIsNotNone(linear3.bias)


if __name__ == '__main__':
    unittest.main()
