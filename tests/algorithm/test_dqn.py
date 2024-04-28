import copy
import unittest

import torch
import torch.nn as nn
import yaml

from PyRLAgent.algorithm.dqn import DQN


class TestDQN(unittest.TestCase):
    """
    Tests the class DQN.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.agent = DQN(
            env_type="CartPole-v1",
            env_wrappers="none",
            policy_type="q-net",
            policy_kwargs={
                "architecture": [128],
                "activation_fn": nn.Tanh(),
                "output_activation_fn": None,
                "bias": True
            },
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring-buffer",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            lr_scheduler_type="linear-lr",
            lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.3, "total_iters": 1000},
            loss_type="huber",
            loss_kwargs={},
            max_gradient_norm=100,
            num_envs=8,
            steps_per_trajectory=16,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        loss, loss_info = self.agent.compute_loss(
            states=torch.rand(size=(8, 64, 4)),
            actions=torch.randint(low=0, high=2, size=(8, 64,)),
            rewards=torch.rand(size=(8, 64,)),
            next_states=torch.rand(size=(8, 64, 4)),
            dones=torch.randint(low=0, high=2, size=(8, 64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(loss.item(), float)
        self.assertIsInstance(loss_info, dict)

    def test_fit(self):
        """
        Tests the method fit().
        """
        old_params = copy.deepcopy(self.agent.q_net.state_dict())
        self.agent.fit(n_timesteps=1e3)
        new_params = copy.deepcopy(self.agent.q_net.state_dict())

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
        old_params = copy.deepcopy(self.agent.q_net.state_dict())
        self.agent.eval(n_timesteps=1e2)
        new_params = copy.deepcopy(self.agent.q_net.state_dict())

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

        # Train/Eval the agent
        agent.fit(n_timesteps=1e3)
        agent.eval(n_timesteps=1e2)


if __name__ == '__main__':
    unittest.main()
