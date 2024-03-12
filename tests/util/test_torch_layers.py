import unittest

import torch.nn as nn

from PyRLAgent.util.torch_layers import create_dueling_mlp, create_mlp


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
        architecture = None
        activation_fn = None
        output_activation_fn = None
        bias = False

        model = create_mlp(input_dim, output_dim, architecture, activation_fn, output_activation_fn, bias)
        linear = model[0]

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(1, len(model))
        self.assertEqual(input_dim, linear.in_features)
        self.assertEqual(output_dim, linear.out_features)
        self.assertIsNone(linear.bias)

        # Create another minimal neural network:
        # [nn.Linear(64, 10, bias=True), nn.Tanh()]
        input_dim = 64
        output_dim = 10
        architecture = None
        activation_fn = None
        output_activation_fn = nn.Tanh()

        bias = False

        model = create_mlp(input_dim, output_dim, architecture, activation_fn, output_activation_fn, bias)
        linear, activation = model[0], model[1]

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(2, len(model))
        self.assertEqual(input_dim, linear.in_features)
        self.assertEqual(output_dim, linear.out_features)
        self.assertIsNone(linear.bias)
        self.assertIsInstance(activation, nn.Tanh)

        # Create the following neural network:
        # [nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10) nn.Tanh()]
        input_dim = 64
        output_dim = 10
        architecture = [128, 64]
        activation_fn = nn.ReLU()
        output_activation_fn = nn.Tanh()
        bias = True

        model = create_mlp(input_dim, output_dim, architecture, activation_fn, output_activation_fn, bias)
        linear1, act1, linear2, act2, linear3, act3 = model[0], model[1], model[2], model[3], model[4], model[5]

        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(6, len(model))

        self.assertEqual(input_dim, linear1.in_features)
        self.assertEqual(architecture[0], linear1.out_features)
        self.assertIsNotNone(linear1.bias)

        self.assertIsInstance(act1, nn.ReLU)

        self.assertEqual(architecture[0], linear2.in_features)
        self.assertEqual(architecture[1], linear2.out_features)
        self.assertIsNotNone(linear2.bias)

        self.assertIsInstance(act2, nn.ReLU)

        self.assertEqual(architecture[1], linear3.in_features)
        self.assertEqual(output_dim, linear3.out_features)
        self.assertIsNotNone(linear3.bias)

        self.assertIsInstance(act3, nn.Tanh)

    def test_create_dueling_mlp(self):
        """
        Tests the method create_dueling_mlp().
        """
        # Create the minimal neural network:
        input_dim = 64
        output_dim = 10
        feature_architecture = None
        feature_activation_fn = None
        feature_output_activation_fn = None
        value_architecture = None
        value_activation_fn = None
        value_output_activation_fn = None
        advantage_architecture = None
        advantage_activation_fn = None
        advantage_output_activation_fn = None
        bias = False

        model = create_dueling_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            feature_architecture=feature_architecture,
            feature_activation_fn=feature_activation_fn,
            feature_output_activation_fn=feature_output_activation_fn,
            value_architecture=value_architecture,
            value_activation_fn=value_activation_fn,
            value_output_activation_fn=value_output_activation_fn,
            advantage_architecture=advantage_architecture,
            advantage_activation_fn=advantage_activation_fn,
            advantage_output_activation_fn=advantage_output_activation_fn,
            bias=bias,
        )

        feature_extractor = model.feature_extractor
        self.assertIsNone(feature_extractor)

        values = model.values
        self.assertIsInstance(values, nn.Sequential)
        self.assertEqual(1, len(values))
        self.assertEqual(input_dim, values[0].in_features)
        self.assertEqual(1, values[0].out_features)
        self.assertIsNone(values[0].bias)

        advantages = model.advantages
        self.assertIsInstance(advantages, nn.Sequential)
        self.assertEqual(1, len(advantages))
        self.assertEqual(input_dim, advantages[0].in_features)
        self.assertEqual(output_dim, advantages[0].out_features)
        self.assertIsNone(advantages[0].bias)

        # Create a non-minimal network
        input_dim = 64
        output_dim = 10
        feature_architecture = [64]
        feature_activation_fn = None
        feature_output_activation_fn = nn.Tanh()
        value_architecture = [32]
        value_activation_fn = nn.ReLU()
        value_output_activation_fn = None
        advantage_architecture = [32]
        advantage_activation_fn = nn.LeakyReLU()
        advantage_output_activation_fn = None
        bias = False

        model = create_dueling_mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            feature_architecture=feature_architecture,
            feature_activation_fn=feature_activation_fn,
            feature_output_activation_fn = feature_output_activation_fn,
            value_architecture=value_architecture,
            value_activation_fn=value_activation_fn,
            value_output_activation_fn=value_output_activation_fn,
            advantage_architecture=advantage_architecture,
            advantage_activation_fn=advantage_activation_fn,
            advantage_output_activation_fn=advantage_output_activation_fn,
            bias=bias,
        )
        feature_extractor = model.feature_extractor
        self.assertIsInstance(feature_extractor, nn.Sequential)
        self.assertEqual(2, len(feature_extractor))
        self.assertEqual(input_dim, feature_extractor[0].in_features)
        self.assertEqual(feature_architecture[0], feature_extractor[0].out_features)
        self.assertIsNone(feature_extractor[0].bias)
        self.assertIsInstance(feature_extractor[1], nn.Tanh)

        values = model.values
        self.assertIsInstance(values, nn.Sequential)
        self.assertEqual(3, len(values))
        self.assertEqual(feature_architecture[0], values[0].in_features)
        self.assertEqual(32, values[0].out_features)
        self.assertIsNone(values[0].bias)
        self.assertIsInstance(values[1], nn.ReLU)
        self.assertEqual(32, values[2].in_features)
        self.assertEqual(1, values[2].out_features)
        self.assertIsNone(values[2].bias)

        advantages = model.advantages
        self.assertIsInstance(advantages, nn.Sequential)
        self.assertEqual(3, len(advantages))
        self.assertEqual(feature_architecture[0], advantages[0].in_features)
        self.assertEqual(32, advantages[0].out_features)
        self.assertIsNone(advantages[0].bias)
        self.assertIsInstance(advantages[1], nn.LeakyReLU)
        self.assertEqual(32, advantages[2].in_features)
        self.assertEqual(output_dim, advantages[2].out_features)
        self.assertIsNone(advantages[2].bias)


if __name__ == '__main__':
    unittest.main()
