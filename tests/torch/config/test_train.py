import unittest

import gymnasium as gym
import torch
import torch.nn as nn

from pyrlagent.torch.config import (
    EnvConfig,
    LRSchedulerConfig,
    NetworkConfig,
    OptimizerConfig,
    RLTrainConfig,
    create_rl_components_eval,
    create_rl_components_train,
)
from pyrlagent.torch.network import MLPCategoricalActorCriticNetwork


class TestRLTrainConfig(unittest.TestCase):
    """Tests the RLTrainConfig class."""

    def setUp(self):
        self.env_config = EnvConfig(id="CartPole-v1", kwargs={})
        self.network_config = NetworkConfig(
            id="mlp-discrete",
            kwargs={
                "hidden_features": [64, 64],
                "activation": nn.ReLU,
            },
        )
        self.optimizer_config = OptimizerConfig(
            id="adam",
            kwargs={"lr": 0.001},
        )
        self.lr_scheduler_config = LRSchedulerConfig(
            id="step",
            kwargs={"step_size": 10, "gamma": 0.1},
        )

        self.device = "cpu"

    def test_init(self):
        """Tests the __init__() method."""
        train_config = RLTrainConfig(
            env_config=self.env_config,
            network_config=self.network_config,
            optimizer_config=self.optimizer_config,
            lr_scheduler_config=self.lr_scheduler_config,
        )
        self.assertEqual(train_config.env_config, self.env_config)
        self.assertEqual(train_config.network_config, self.network_config)
        self.assertEqual(train_config.optimizer_config, self.optimizer_config)
        self.assertEqual(train_config.lr_scheduler_config, self.lr_scheduler_config)

    def test_create_rl_components_train(self):
        """Tests the create_rl_components_train() method."""
        env, network, optimizer, lr_scheduler = create_rl_components_train(
            train_config=RLTrainConfig(
                env_config=self.env_config,
                network_config=self.network_config,
                optimizer_config=self.optimizer_config,
                lr_scheduler_config=self.lr_scheduler_config,
            ),
            num_envs=1,
            device="cpu",
        )
        self.assertIsInstance(env, gym.vector.VectorEnv)
        self.assertIsInstance(network, MLPCategoricalActorCriticNetwork)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_rl_components_eval(self):
        """Tests the create_rl_components_eval() method."""
        env, network = create_rl_components_eval(
            train_config=RLTrainConfig(
                env_config=self.env_config,
                network_config=self.network_config,
                optimizer_config=self.optimizer_config,
                lr_scheduler_config=self.lr_scheduler_config,
            ),
            train_state=None,
            device=self.device,
        )
        self.assertIsInstance(env, gym.Env)
        self.assertIsInstance(network, MLPCategoricalActorCriticNetwork)


if __name__ == "__main__":
    unittest.main()
