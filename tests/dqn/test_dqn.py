import unittest
import numpy as np
import torch.nn as nn
import torch
import yaml
import copy

from PyRLAgent.dqn.dqn import DQN, DDQN, ClippedDDQN


class TestDQN(unittest.TestCase):
    """
    Tests the class DQN.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.agent = DQN(
            env_type="CartPole-v1",
            policy_type="q-net",
            policy_kwargs={"architecture": [128], "activation_fn": nn.Tanh(), "bias": True},
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            loss_type="huber",
            loss_kwargs={},
            max_gradient_norm=100,
            learning_starts=16,
            batch_size=16,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            train_freq=(1, "steps"),
            render_freq=50,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        result = self.agent.compute_loss(
            states=torch.rand(size=(64, 4)),
            actions=torch.randint(low=0, high=2, size=(64,)),
            rewards=torch.rand(size=(64,)),
            next_states=torch.rand(size=(64, 4)),
            dones=torch.randint(low=0, high=2, size=(64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(result.item(), float)

    def test_predict_deterministic(self):
        """
        Tests the method predict() with deterministic=True.
        """
        result = self.agent.predict(np.random.random(size=(4,)).astype(dtype=np.float32), deterministic=True)

        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(result.item(), int)

    def test_predict_non_deterministic(self):
        """
        Tests the method predict() with deterministic=False.
        """
        result = self.agent.predict(np.random.random(size=(4,)).astype(dtype=np.float32), deterministic=False)

        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(result.item(), int)

    def test_fit(self):
        """
        Tests the method fit().
        """
        old_params = copy.deepcopy(self.agent.q_net.state_dict())
        rewards = self.agent.fit(n_episodes=10, episode_length=500)
        new_params = copy.deepcopy(self.agent.q_net.state_dict())
        self.assertEqual(10, len(rewards))

        # Check if the weights gets updated
        self.assertTrue(all(key1 == key2 and torch.any(torch.not_equal(item1, item2)) for (key1, item1), (key2, item2) in zip(old_params.items(), new_params.items())))

    def test_evaluate(self):
        """
        Tests the method evaluate().
        """
        old_params = copy.deepcopy(self.agent.q_net.state_dict())
        rewards = self.agent.evaluate(n_episodes=10, episode_length=500)
        new_params = copy.deepcopy(self.agent.q_net.state_dict())
        self.assertEqual(10, len(rewards))

        # Check if the weights gets updated
        self.assertTrue(all(key1 == key2 and torch.all(torch.eq(item1, item2)) for (key1, item1), (key2, item2) in zip(old_params.items(), new_params.items())))

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


class TestDDQN(unittest.TestCase):
    """
    Tests the class DDQN.
    """

    def setUp(self) -> None:
        self.agent = DDQN(
            env_type="CartPole-v1",
            policy_type="q-net",
            policy_kwargs={"architecture": [128], "activation_fn": nn.Tanh(), "bias": True},
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            loss_type="huber",
            loss_kwargs={},
            max_gradient_norm=100,
            learning_starts=64,
            batch_size=64,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            train_freq=(1, "steps"),
            render_freq=50,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        result = self.agent.compute_loss(
            states=torch.rand(size=(64, 4)),
            actions=torch.randint(low=0, high=2, size=(64,)),
            rewards=torch.rand(size=(64,)),
            next_states=torch.rand(size=(64, 4)),
            dones=torch.randint(low=0, high=2, size=(64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(result.item(), float)


class TestClippedDDQN(unittest.TestCase):
    """
    Test the class Clipped DDQN.
    """

    def setUp(self) -> None:
        self.agent = ClippedDDQN(
            env_type="CartPole-v1",
            policy_type="q-net",
            policy_kwargs={"architecture": [128], "activation_fn": nn.Tanh(), "bias": True},
            strategy_type="linear-epsilon",
            strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
            replay_buffer_type="ring",
            replay_buffer_kwargs={"max_size": 10000},
            optimizer_type="adam",
            optimizer_kwargs={"lr": 5e-4},
            loss_type="huber",
            loss_kwargs={},
            max_gradient_norm=100,
            learning_starts=64,
            batch_size=64,
            tau=5e-3,
            gamma=0.99,
            target_freq=1,
            train_freq=(1, "steps"),
            render_freq=50,
            gradient_steps=1,
        )

    def test_compute_loss(self):
        """
        Tests the method compute_loss().
        """
        result = self.agent.compute_loss(
            states=torch.rand(size=(64, 4)),
            actions=torch.randint(low=0, high=2, size=(64,)),
            rewards=torch.rand(size=(64,)),
            next_states=torch.rand(size=(64, 4)),
            dones=torch.randint(low=0, high=2, size=(64,)).to(dtype=torch.bool),
        )

        self.assertIsInstance(result, torch.Tensor)
        self.assertIsInstance(result.item(), float)


if __name__ == '__main__':
    unittest.main()
