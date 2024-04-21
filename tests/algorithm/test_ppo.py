import copy
import unittest

import torch
import yaml
from torch import nn

from PyRLAgent.algorithm.ppo import PPO


class TestPPO(unittest.TestCase):
    """
    Tests the class PPO.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.agent = PPO(
            env_type="CartPole-v1",
            env_wrappers="none",
            policy_type="actor-critic-net",
            policy_kwargs={
                "actor_architecture": [128],
                "actor_activation_fn": nn.Tanh(),
                "actor_output_activation_fn": None,
                "critic_architecture": [128],
                "critic_activation_fn": nn.Tanh(),
                "critic_output_activation_fn": None,
                "bias": True
            },
            optimizer_type="adamw",
            optimizer_kwargs={"lr": 1e-3},
            lr_scheduler_type="linear-lr",
            lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.8, "total_iters": 100000},
            batch_size=32,
            steps_per_trajectory=16,
            clip_ratio=0.2,
            gamma=0.98,
            gae_lambda=0.8,
            target_kl=0.01,
            vf_coef=0.5,
            ent_coef=0.0,
            render_freq=50,
            gradient_steps=16,
        )

    def test_compute_gae(self):
        """
        Tests the method compute_loss().
        """
        advantages, targets = self.agent.compute_gae(
            rewards=torch.rand(size=(32 * 16,)),
            dones=torch.randint(low=0, high=2, size=(32 * 16,)),
            values=torch.rand(size=(32 * 16,)),
        )

        self.assertIsInstance(advantages, torch.Tensor)
        self.assertEqual((32 * 16,), advantages.shape)

        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual((32 * 16,), targets.shape)

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        loss, loss_info = self.agent.compute_loss(
            states=torch.rand(size=(32 * 16, 4)),
            actions=torch.randint(low=0, high=2, size=(32 * 16,)),
            log_probs=torch.rand(size=(32 * 16,)),
            advantages=torch.rand(size=(32 * 16,)),
            values=torch.rand(size=(32 * 16,)),
            targets=torch.rand(size=(32 * 16,)),
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(loss_info, dict)

    def test_fit(self):
        """
        Tests the method fit().
        """
        old_params = copy.deepcopy(self.agent.policy.state_dict())
        self.agent.fit(n_timesteps=1e3)
        new_params = copy.deepcopy(self.agent.policy.state_dict())

        # Check if the weights gets updated
        self.assertTrue(
            all(
                key1 == key2 and torch.any(torch.not_equal(item1, item2))
                for (key1, item1), (key2, item2) in zip(old_params.items(), new_params.items())
            )
        )

    def test_eval(self):
        """
        Tests the method eval().
        """
        old_params = copy.deepcopy(self.agent.policy.state_dict())
        self.agent.eval(n_timesteps=1e2)
        new_params = copy.deepcopy(self.agent.policy.state_dict())

        # Check if the weights gets not updated
        self.assertTrue(
            all(
                key1 == key2 and torch.all(torch.eq(item1, item2))
                for (key1, item1), (key2, item2) in zip(old_params.items(), new_params.items())
            )
        )

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.agent, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            agent = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal (overriding __eq__ causes much slower training)
        self.assertEqual(str(self.agent), str(agent))


if __name__ == '__main__':
    unittest.main()
