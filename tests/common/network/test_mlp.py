import unittest

from torch import nn

from PyRLAgent.common.network.mlp import create_mlp


class TestMLP(unittest.TestCase):
    """
    Tests all methods under mlp.py
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


if __name__ == '__main__':
    unittest.main()
