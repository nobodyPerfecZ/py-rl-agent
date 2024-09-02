import unittest

import torch
import torch.nn as nn

from pyrlagent.torch.config import OptimizerConfig, create_optimizer


class TestOptimizerConfig(unittest.TestCase):
    """Tests the OptimizerConfig class."""

    def setUp(self):
        self.network = nn.Sequential(*[nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)])

        self.adadeta_id = "adadelta"
        self.adadelta_kwargs = {"lr": 0.001}

        self.adagrad_id = "adagrad"
        self.adagrad_kwargs = {"lr": 0.001}

        self.adam_id = "adam"
        self.adam_kwargs = {"lr": 0.001}

        self.adamw_id = "adamw"
        self.adamw_kwargs = {"lr": 0.001}

        self.sparse_adam_id = "sparse_adam"
        self.sparse_adam_kwargs = {"lr": 0.001}

        self.adamax_id = "adamax"
        self.adamax_kwargs = {"lr": 0.001}

        self.asgd_id = "asgd"
        self.asgd_kwargs = {"lr": 0.001}

        self.lbfgs_id = "lbfgs"
        self.lbfgs_kwargs = {"lr": 0.001}

        self.nadam_id = "nadam"
        self.nadam_kwargs = {"lr": 0.001}

        self.radam_id = "radam"
        self.radam_kwargs = {"lr": 0.001}

        self.rmsprop_id = "rmsprop"
        self.rmsprop_kwargs = {"lr": 0.001}

        self.rprop_id = "rprop"
        self.rprop_kwargs = {"lr": 0.001}

        self.sgd_id = "sgd"
        self.sgd_kwargs = {"lr": 0.01}

    def test_init(self):
        """Tests the __init__() method."""
        adadelta_config = OptimizerConfig(
            id=self.adadeta_id, kwargs=self.adadelta_kwargs
        )
        self.assertEqual(adadelta_config.id, self.adadeta_id)
        self.assertEqual(adadelta_config.kwargs, self.adadelta_kwargs)

        adagrad_config = OptimizerConfig(id=self.adagrad_id, kwargs=self.adagrad_kwargs)
        self.assertEqual(adagrad_config.id, self.adagrad_id)
        self.assertEqual(adagrad_config.kwargs, self.adagrad_kwargs)

        adam_config = OptimizerConfig(id=self.adam_id, kwargs=self.adam_kwargs)
        self.assertEqual(adam_config.id, self.adam_id)
        self.assertEqual(adam_config.kwargs, self.adam_kwargs)

        adamw_config = OptimizerConfig(id=self.adamw_id, kwargs=self.adamw_kwargs)
        self.assertEqual(adamw_config.id, self.adamw_id)
        self.assertEqual(adamw_config.kwargs, self.adamw_kwargs)

        sparse_adam_config = OptimizerConfig(
            id=self.sparse_adam_id, kwargs=self.sparse_adam_kwargs
        )
        self.assertEqual(sparse_adam_config.id, self.sparse_adam_id)
        self.assertEqual(sparse_adam_config.kwargs, self.sparse_adam_kwargs)

        adamax_config = OptimizerConfig(id=self.adamax_id, kwargs=self.adamax_kwargs)
        self.assertEqual(adamax_config.id, self.adamax_id)
        self.assertEqual(adamax_config.kwargs, self.adamax_kwargs)

        asgd_config = OptimizerConfig(id=self.asgd_id, kwargs=self.asgd_kwargs)
        self.assertEqual(asgd_config.id, self.asgd_id)
        self.assertEqual(asgd_config.kwargs, self.asgd_kwargs)

        lbfgs_config = OptimizerConfig(id=self.lbfgs_id, kwargs=self.lbfgs_kwargs)
        self.assertEqual(lbfgs_config.id, self.lbfgs_id)
        self.assertEqual(lbfgs_config.kwargs, self.lbfgs_kwargs)

        nadam_config = OptimizerConfig(id=self.nadam_id, kwargs=self.nadam_kwargs)
        self.assertEqual(nadam_config.id, self.nadam_id)
        self.assertEqual(nadam_config.kwargs, self.nadam_kwargs)

        radam_config = OptimizerConfig(id=self.radam_id, kwargs=self.radam_kwargs)
        self.assertEqual(radam_config.id, self.radam_id)
        self.assertEqual(radam_config.kwargs, self.radam_kwargs)

        rmsprop_config = OptimizerConfig(id=self.rmsprop_id, kwargs=self.rmsprop_kwargs)
        self.assertEqual(rmsprop_config.id, self.rmsprop_id)
        self.assertEqual(rmsprop_config.kwargs, self.rmsprop_kwargs)

        rprop_config = OptimizerConfig(id=self.rprop_id, kwargs=self.rprop_kwargs)
        self.assertEqual(rprop_config.id, self.rprop_id)
        self.assertEqual(rprop_config.kwargs, self.rprop_kwargs)

        sgd_config = OptimizerConfig(id=self.sgd_id, kwargs=self.sgd_kwargs)
        self.assertEqual(sgd_config.id, self.sgd_id)
        self.assertEqual(sgd_config.kwargs, self.sgd_kwargs)

    def test_create_optimizer(self):
        """Tests the create_optimizer() method."""
        # Create an invalid optimizer
        with self.assertRaises(ValueError):
            create_optimizer(
                optimizer_config=OptimizerConfig(id="invalid_id", kwargs={}),
                network=self.network,
            )

        # Create an Adadelta optimizer
        adadelta_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.adadeta_id, kwargs=self.adadelta_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(adadelta_optimizer, torch.optim.Adadelta)

        # Create an Adagrad optimizer
        adagrad_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.adagrad_id, kwargs=self.adagrad_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(adagrad_optimizer, torch.optim.Adagrad)

        # Create an Adam optimizer
        adam_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(id=self.adam_id, kwargs=self.adam_kwargs),
            network=self.network,
        )
        self.assertIsInstance(adam_optimizer, torch.optim.Adam)

        # Create an AdamW optimizer
        adamw_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.adamw_id, kwargs=self.adamw_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(adamw_optimizer, torch.optim.AdamW)

        # Create a SparseAdam optimizer
        sparse_adam_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.sparse_adam_id, kwargs=self.sparse_adam_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(sparse_adam_optimizer, torch.optim.SparseAdam)

        # Create an Adamax optimizer
        adamax_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.adamax_id, kwargs=self.adamax_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(adamax_optimizer, torch.optim.Adamax)

        # Create an ASGD optimizer
        asgd_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(id=self.asgd_id, kwargs=self.asgd_kwargs),
            network=self.network,
        )
        self.assertIsInstance(asgd_optimizer, torch.optim.ASGD)

        # Create an L-BFGS optimizer
        lbfgs_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.lbfgs_id, kwargs=self.lbfgs_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(lbfgs_optimizer, torch.optim.LBFGS)

        # Create a NAdam optimizer
        nadam_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.nadam_id, kwargs=self.nadam_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(nadam_optimizer, torch.optim.NAdam)

        # Create a RAdam optimizer
        radam_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.radam_id, kwargs=self.radam_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(radam_optimizer, torch.optim.RAdam)

        # Create a RMSprop optimizer
        rmsprop_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.rmsprop_id, kwargs=self.rmsprop_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(rmsprop_optimizer, torch.optim.RMSprop)

        # Create a Rprop optimizer
        rprop_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(
                id=self.rprop_id, kwargs=self.rprop_kwargs
            ),
            network=self.network,
        )
        self.assertIsInstance(rprop_optimizer, torch.optim.Rprop)

        # Create a SGD optimizer
        sgd_optimizer = create_optimizer(
            optimizer_config=OptimizerConfig(id=self.sgd_id, kwargs=self.sgd_kwargs),
            network=self.network,
        )
        self.assertIsInstance(sgd_optimizer, torch.optim.SGD)


if __name__ == "__main__":
    unittest.main()
