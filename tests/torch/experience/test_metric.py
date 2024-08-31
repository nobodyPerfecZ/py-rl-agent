import unittest

import numpy as np
import torch

from pyrlagent.torch.experience import discounted_return, gae, non_discounted_return


class TestMetric(unittest.TestCase):
    """Tests the methods under metric.py."""

    def setUp(self):
        self.reward = np.ones((10, 1), dtype=np.float32)
        self.done = np.zeros((10, 1), dtype=np.float32)
        self.value = 0.25 * np.ones((10, 1), dtype=np.float32)
        self.next_value = 0.5 * np.ones((10, 1), dtype=np.float32)
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def test_numpy_non_discounted_return(self):
        """Tests the non_discounted_return_numpy() method."""
        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, False, False, False, False, False, False, False, False]
        non_discounted_returns1 = non_discounted_return(self.reward, self.done)
        self.assertIsInstance(non_discounted_returns1, np.ndarray)
        np.testing.assert_allclose(
            np.array(
                [
                    [10.0],
                    [9.0],
                    [8.0],
                    [7.0],
                    [6.0],
                    [5.0],
                    [4.0],
                    [3.0],
                    [2.0],
                    [1.0],
                ]
            ),
            non_discounted_returns1,
        )

        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, True, False, False, True, False, False, True, False]
        self.done[[2, 5, 8]] = True
        non_discounted_returns2 = non_discounted_return(self.reward, self.done)
        self.assertIsInstance(non_discounted_returns2, np.ndarray)
        np.testing.assert_allclose(
            np.array(
                [
                    [3.0],
                    [2.0],
                    [1.0],
                    [3.0],
                    [2.0],
                    [1.0],
                    [3.0],
                    [2.0],
                    [1.0],
                    [1.0],
                ]
            ),
            non_discounted_returns2,
        )

    def test_torch_non_discounted_return(self):
        """Tests the non_discounted_return_numpy() method."""
        rewards = torch.tensor(self.reward, dtype=torch.float32)
        dones = torch.tensor(self.done, dtype=torch.float32)

        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, False, False, False, False, False, False, False, False]
        non_discounted_returns1 = non_discounted_return(rewards, dones)
        self.assertIsInstance(non_discounted_returns1, torch.Tensor)
        torch.testing.assert_close(
            torch.tensor(
                [
                    [10.0],
                    [9.0],
                    [8.0],
                    [7.0],
                    [6.0],
                    [5.0],
                    [4.0],
                    [3.0],
                    [2.0],
                    [1.0],
                ]
            ),
            non_discounted_returns1,
        )

        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, True, False, False, True, False, False, True, False]
        dones[[2, 5, 8]] = True
        non_discounted_returns2 = non_discounted_return(rewards, dones)
        self.assertIsInstance(non_discounted_returns2, torch.Tensor)
        torch.testing.assert_close(
            torch.tensor(
                [
                    [3.0],
                    [2.0],
                    [1.0],
                    [3.0],
                    [2.0],
                    [1.0],
                    [3.0],
                    [2.0],
                    [1.0],
                    [1.0],
                ]
            ),
            non_discounted_returns2,
        )

    def test_numpy_discounted_return(self):
        """Tests the discounted_return_numpy() method."""
        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, False, False, False, False, False, False, False, False]
        discounted_returns1 = discounted_return(self.reward, self.done, self.gamma)
        self.assertIsInstance(discounted_returns1, np.ndarray)
        np.testing.assert_allclose(
            np.array(
                [
                    [9.5617925],
                    [8.64827525],
                    [7.72553056],
                    [6.79346521],
                    [5.85198506],
                    [4.90099501],
                    [3.940399],
                    [2.9701],
                    [1.99],
                    [1.0],
                ]
            ),
            discounted_returns1,
        )

        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, True, False, False, True, False, False, True, False]
        self.done[[2, 5, 8]] = True
        discounted_returns2 = discounted_return(self.reward, self.done, self.gamma)
        self.assertIsInstance(discounted_returns2, np.ndarray)
        np.testing.assert_allclose(
            np.array(
                [
                    [2.9701],
                    [1.99],
                    [1.0],
                    [2.9701],
                    [1.99],
                    [1.0],
                    [2.9701],
                    [1.99],
                    [1.0],
                    [1.0],
                ]
            ),
            discounted_returns2,
        )

    def test_torch_discounted_return(self):
        """Tests the discounted_return_torch() method."""
        rewards = torch.tensor(self.reward, dtype=torch.float32)
        dones = torch.tensor(self.done, dtype=torch.float32)

        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, False, False, False, False, False, False, False, False]
        discounted_returns1 = discounted_return(rewards, dones, self.gamma)
        self.assertIsInstance(discounted_returns1, torch.Tensor)
        torch.testing.assert_close(
            torch.tensor(
                [
                    [9.5617925],
                    [8.64827525],
                    [7.72553056],
                    [6.79346521],
                    [5.85198506],
                    [4.90099501],
                    [3.940399],
                    [2.9701],
                    [1.99],
                    [1.0],
                ]
            ),
            discounted_returns1,
        )

        # Rewards := [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # Dones := [False, False, True, False, False, True, False, False, True, False]
        dones[[2, 5, 8]] = True
        discounted_returns2 = discounted_return(rewards, dones, self.gamma)
        self.assertIsInstance(discounted_returns2, torch.Tensor)
        torch.testing.assert_close(
            torch.tensor(
                [
                    [2.9701],
                    [1.99],
                    [1.0],
                    [2.9701],
                    [1.99],
                    [1.0],
                    [2.9701],
                    [1.99],
                    [1.0],
                    [1.0],
                ]
            ),
            discounted_returns2,
        )

    def test_numpy_gae(self):
        """Tests the gae() method."""
        advantages, returns = gae(
            self.reward,
            self.value,
            self.next_value,
            self.done,
            self.gamma,
            self.gae_lambda,
        )
        self.assertIsInstance(advantages, np.ndarray)
        np.testing.assert_allclose(
            np.array(
                [
                    [9.59409651],
                    [8.8772956],
                    [8.11514683],
                    [7.30478131],
                    [6.44314865],
                    [5.52700547],
                    [4.55290321],
                    [3.51717512],
                    [2.41592251],
                    [1.245],
                ]
            ),
            advantages,
        )
        self.assertIsInstance(returns, np.ndarray)
        np.testing.assert_allclose(
            np.array(
                [
                    [9.84409651],
                    [9.1272956],
                    [8.36514683],
                    [7.55478131],
                    [6.69314865],
                    [5.77700547],
                    [4.80290321],
                    [3.76717512],
                    [2.66592251],
                    [1.495],
                ]
            ),
            returns,
        )

    def test_torch_gae(self):
        """Tests the gae_torch() method."""
        rewards = torch.tensor(self.reward, dtype=torch.float32)
        values = torch.tensor(self.value, dtype=torch.float32)
        next_values = torch.tensor(self.next_value, dtype=torch.float32)
        dones = torch.tensor(self.done, dtype=torch.float32)

        advantages, returns = gae(
            rewards,
            values,
            next_values,
            dones,
            self.gamma,
            self.gae_lambda,
        )
        self.assertIsInstance(advantages, torch.Tensor)
        torch.testing.assert_close(
            torch.tensor(
                [
                    [9.59409651],
                    [8.8772956],
                    [8.11514683],
                    [7.30478131],
                    [6.44314865],
                    [5.52700547],
                    [4.55290321],
                    [3.51717512],
                    [2.41592251],
                    [1.245],
                ]
            ),
            advantages,
        )
        self.assertIsInstance(returns, torch.Tensor)
        torch.testing.assert_close(
            torch.tensor(
                [
                    [9.84409651],
                    [9.1272956],
                    [8.36514683],
                    [7.55478131],
                    [6.69314865],
                    [5.77700547],
                    [4.80290321],
                    [3.76717512],
                    [2.66592251],
                    [1.495],
                ]
            ),
            returns,
        )


if __name__ == "__main__":
    unittest.main()
