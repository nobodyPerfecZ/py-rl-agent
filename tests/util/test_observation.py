import unittest
import numpy as np

from PyRLAgent.util.observation import obs_to_tensor, is_rgb_image_observation


class TestUtilObservation(unittest.TestCase):
    """
    Tests all methods from util.observation.
    """
    def test_obs_to_tensor(self):
        """
        Tests the method obs_to_tensor().
        """
        observation1 = np.random.random(size=(10,))
        observation2 = np.random.random(size=(10, 10))
        observation3 = np.random.random(size=(10, 10, 3))
        observation4 = np.random.random(size=(10, 10, 10, 3))

        tensor1 = obs_to_tensor(observation1)
        tensor2 = obs_to_tensor(observation2)
        tensor3 = obs_to_tensor(observation3)
        tensor4 = obs_to_tensor(observation4)

        self.assertTrue(np.array_equal(observation1, tensor1))
        self.assertTrue(np.array_equal(observation2, tensor2))
        self.assertTrue(np.array_equal(observation3, tensor3))
        self.assertFalse(np.array_equal(observation4, tensor4))

    def test_is_rgb_image_observation(self):
        """
        Tests the method is_rgb_image_observation().
        """
        observation1 = np.random.random(size=(10,))
        observation2 = np.random.random(size=(10, 10))
        observation3 = np.random.random(size=(10, 10, 3))
        observation4 = np.random.random(size=(10, 10, 10, 3))

        self.assertFalse(is_rgb_image_observation(observation1))
        self.assertFalse(is_rgb_image_observation(observation2))
        self.assertFalse(is_rgb_image_observation(observation3))
        self.assertTrue(is_rgb_image_observation(observation4))


if __name__ == '__main__':
    unittest.main()
