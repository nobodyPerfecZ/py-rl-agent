import unittest

import torch
import torch.nn as nn

from pyrlagent.torch.config import OptimizerConfig, create_optimizer


class TestOptimizerConfig(unittest.TestCase):
    """Tests the OptimizerConfig class."""

    def setUp(self):
        self.network = nn.Sequential(*[nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)])

        self.sgd_id = "sgd"
        self.sgd_kwargs = {"lr": 0.01}

        self.adam_id = "adam"
        self.adam_kwargs = {"lr": 0.001}

    def test_init(self):
        """Tests the __init__() method."""
        sgd_config = OptimizerConfig(id=self.sgd_id, kwargs=self.sgd_kwargs)
        self.assertEqual(sgd_config.id, self.sgd_id)
        self.assertEqual(sgd_config.kwargs, self.sgd_kwargs)

        adam_config = OptimizerConfig(id=self.adam_id, kwargs=self.adam_kwargs)
        self.assertEqual(adam_config.id, self.adam_id)
        self.assertEqual(adam_config.kwargs, self.adam_kwargs)

    def test_create_optimizer(self):
        """Tests the create_optimizer() method."""
        # Create an invalid optimizer
        with self.assertRaises(ValueError):
            create_optimizer(
                optimizer_config=OptimizerConfig(id="invalid_id", kwargs={}),
                network=self.network,
            )

        # Create a SGD optimizer
        sgd_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(id=self.sgd_id, kwargs=self.sgd_kwargs),
            network=self.network,
        )
        self.assertIsInstance(sgd_optimizer, torch.optim.SGD)

        # Create an Adam optimizer
        adam_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(id=self.adam_id, kwargs=self.adam_kwargs),
            network=self.network,
        )
        self.assertIsInstance(adam_optimizer, torch.optim.Adam)


if __name__ == "__main__":
    unittest.main()
