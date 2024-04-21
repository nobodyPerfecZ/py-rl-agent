import unittest

import torch
import torch.nn as nn
from torch.optim import (
    SGD,
    Adam,
    Adadelta,
    Adagrad,
    AdamW,
    SparseAdam,
    Adamax,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop
)

from PyRLAgent.util.optimizer import OptimizerEnum


class TestOptimizerEnum(unittest.TestCase):
    """
    Tests the enum class OptimizerEnum.
    """

    def setUp(self):
        self.theta1 = nn.Parameter(data=torch.tensor(1.0), requires_grad=True)
        self.theta2 = nn.Parameter(data=torch.tensor(2.0), requires_grad=True)
        self.theta = [self.theta1, self.theta2]
        self.lr = 1e-3

        self.wrapper = {
            OptimizerEnum.ADADELTA: Adadelta,
            OptimizerEnum.ADAGRAD: Adagrad,
            OptimizerEnum.ADAM: Adam,
            OptimizerEnum.ADAMW: AdamW,
            OptimizerEnum.SPARSEADAM: SparseAdam,
            OptimizerEnum.ADAMAX: Adamax,
            OptimizerEnum.ASGD: ASGD,
            OptimizerEnum.LBFGS: LBFGS,
            OptimizerEnum.NADAM: NAdam,
            OptimizerEnum.RADAM: RAdam,
            OptimizerEnum.RMSPROP: RMSprop,
            OptimizerEnum.RPROP: Rprop,
            OptimizerEnum.SGD: SGD,
        }

        self.optimizer_kwargs1 = {"rho": 0.8, "weight_decay": 0.0001}
        self.optimizer_kwargs2 = {"lr_decay": 0.001, "weight_decay": 0.0001}
        self.optimizer_kwargs3 = {"betas": (0.99, 0.9999), "weight_decay": 0.0001}
        self.optimizer_kwargs4 = {"betas": (0.99, 0.9999), "weight_decay": 0.0001}
        self.optimizer_kwargs5 = {"betas": (0.99, 0.9999)}
        self.optimizer_kwargs6 = {"betas": (0.99, 0.9999), "weight_decay": 0.0001}
        self.optimizer_kwargs7 = {"lambd": 0.1, "alpha": 0.8}
        self.optimizer_kwargs8 = {"history_size": 200}
        self.optimizer_kwargs9 = {"betas": (0.99, 0.9999), "weight_decay": 0.0001}
        self.optimizer_kwargs10 = {"betas": (0.99, 0.9999), "weight_decay": 0.0001}
        self.optimizer_kwargs11 = {"alpha": 0.999, "weight_decay": 0.0001}
        self.optimizer_kwargs12 = {"etas": (0.6, 1.3)}
        self.optimizer_kwargs13 = {"momentum": 0.0001, "dampening": 0.0001}

    def test_wrapper(self):
        """
        Tests the class method wrapper().
        """
        self.assertDictEqual(self.wrapper, OptimizerEnum.wrapper())

    def test_to_optimizer(self):
        """
        Tests the method to_optimizer().
        """
        optimizer1 = OptimizerEnum.ADADELTA.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs1)
        optimizer2 = OptimizerEnum.ADAGRAD.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs2)
        optimizer3 = OptimizerEnum.ADAM.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs3)
        optimizer4 = OptimizerEnum.ADAMW.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs4)
        optimizer5 = OptimizerEnum.SPARSEADAM.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs5)
        optimizer6 = OptimizerEnum.ADAMAX.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs6)
        optimizer7 = OptimizerEnum.ASGD.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs7)
        optimizer8 = OptimizerEnum.LBFGS.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs8)
        optimizer9 = OptimizerEnum.NADAM.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs9)
        optimizer10 = OptimizerEnum.RADAM.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs10)
        optimizer11 = OptimizerEnum.RMSPROP.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs11)
        optimizer12 = OptimizerEnum.RPROP.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs12)
        optimizer13 = OptimizerEnum.SGD.to_optimizer(params=self.theta, lr=self.lr, **self.optimizer_kwargs13)

        self.assertIsInstance(optimizer1, Adadelta)
        self.assertEqual(self.lr, optimizer1.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs1["rho"], optimizer1.param_groups[0]["rho"])
        self.assertEqual(self.optimizer_kwargs1["weight_decay"], optimizer1.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer2, Adagrad)
        self.assertEqual(self.lr, optimizer2.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs2["lr_decay"], optimizer2.param_groups[0]["lr_decay"])
        self.assertEqual(self.optimizer_kwargs2["weight_decay"], optimizer2.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer3, Adam)
        self.assertEqual(self.lr, optimizer3.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs3["betas"], optimizer3.param_groups[0]["betas"])
        self.assertEqual(self.optimizer_kwargs3["weight_decay"], optimizer3.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer4, AdamW)
        self.assertEqual(self.lr, optimizer4.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs4["betas"], optimizer4.param_groups[0]["betas"])
        self.assertEqual(self.optimizer_kwargs4["weight_decay"], optimizer4.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer5, SparseAdam)
        self.assertEqual(self.lr, optimizer5.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs5["betas"], optimizer5.param_groups[0]["betas"])

        self.assertIsInstance(optimizer6, Adamax)
        self.assertEqual(self.lr, optimizer6.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs6["betas"], optimizer6.param_groups[0]["betas"])
        self.assertEqual(self.optimizer_kwargs6["weight_decay"], optimizer6.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer7, ASGD)
        self.assertEqual(self.lr, optimizer7.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs7["lambd"], optimizer7.param_groups[0]["lambd"])
        self.assertEqual(self.optimizer_kwargs7["alpha"], optimizer7.param_groups[0]["alpha"])

        self.assertIsInstance(optimizer8, LBFGS)
        self.assertEqual(self.lr, optimizer8.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs8["history_size"], optimizer8.param_groups[0]["history_size"])

        self.assertIsInstance(optimizer9, NAdam)
        self.assertEqual(self.lr, optimizer9.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs9["betas"], optimizer9.param_groups[0]["betas"])
        self.assertEqual(self.optimizer_kwargs9["weight_decay"], optimizer9.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer10, RAdam)
        self.assertEqual(self.lr, optimizer10.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs10["betas"], optimizer10.param_groups[0]["betas"])
        self.assertEqual(self.optimizer_kwargs10["weight_decay"], optimizer10.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer11, RMSprop)
        self.assertEqual(self.lr, optimizer11.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs11["alpha"], optimizer11.param_groups[0]["alpha"])
        self.assertEqual(self.optimizer_kwargs11["weight_decay"], optimizer11.param_groups[0]["weight_decay"])

        self.assertIsInstance(optimizer12, Rprop)
        self.assertEqual(self.lr, optimizer12.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs12["etas"], optimizer12.param_groups[0]["etas"])

        self.assertIsInstance(optimizer13, SGD)
        self.assertEqual(self.lr, optimizer13.param_groups[0]["lr"])
        self.assertEqual(self.optimizer_kwargs13["momentum"], optimizer13.param_groups[0]["momentum"])
        self.assertEqual(self.optimizer_kwargs13["dampening"], optimizer13.param_groups[0]["dampening"])


if __name__ == "__main__":
    unittest.main()
