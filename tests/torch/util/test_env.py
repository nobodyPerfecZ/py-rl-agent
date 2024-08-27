import unittest

import gymnasium as gym

from pyrlagent.torch.util.env import (
    get_env,
    get_obs_act_dims,
    get_obs_act_space,
    get_vector_env,
)


class TestEnv(unittest.TestCase):
    """Tests the methods under env.py."""

    def setUp(self):
        self.continuous_env_name = "Ant-v5"
        self.discrete_env_name = "CartPole-v1"
        self.num_envs = 3
        self.cpu_device = "cpu"
        self.cuda_device = "cuda"

    def test_get_env(self):
        """Tests the get_env() method."""
        # Create an environment with continuous action spaces with cpu device
        continuous_env = get_env(
            env_id=self.continuous_env_name, device=self.cpu_device, render_mode=None
        )
        self.assertEqual(self.continuous_env_name, continuous_env.spec.id)
        continuous_env.close()

        # Create an environment with continuous action spaces with cuda device
        continuous_env = get_env(
            env_id=self.continuous_env_name, device=self.cuda_device, render_mode=None
        )
        self.assertEqual(self.continuous_env_name, continuous_env.spec.id)
        continuous_env.close()

        # Create an environment with discrete action spaces with cpu device
        discrete_env = get_env(
            env_id=self.discrete_env_name, device=self.cpu_device, render_mode=None
        )
        self.assertEqual(self.discrete_env_name, discrete_env.spec.id)
        discrete_env.close()

        # Create an environment with discrete action spaces with cuda device
        discrete_env = get_env(
            env_id=self.discrete_env_name, device=self.cuda_device, render_mode=None
        )
        self.assertEqual(self.discrete_env_name, discrete_env.spec.id)
        discrete_env.close()

    def test_get_vector_env(self):
        """Tests the get_vector_env() method."""
        # Create a vector environment with continuous action spaces with cpu device
        continous_envs = get_vector_env(
            env_id=self.continuous_env_name,
            num_envs=self.num_envs,
            device=self.cpu_device,
            render_mode=None,
        )
        self.assertIsInstance(continous_envs, gym.vector.VectorEnv)
        self.assertEqual(self.continuous_env_name, continous_envs.spec.id)
        continous_envs.close()

        # Create a vector environment with continuous action spaces with cuda device
        continous_envs = get_vector_env(
            env_id=self.continuous_env_name,
            num_envs=self.num_envs,
            device=self.cuda_device,
            render_mode=None,
        )
        self.assertIsInstance(continous_envs, gym.vector.VectorEnv)
        self.assertEqual(self.continuous_env_name, continous_envs.spec.id)
        continous_envs.close()

        # Create a vector environment with discrete action spaces with cpu device
        discrete_envs = get_vector_env(
            env_id=self.discrete_env_name,
            num_envs=self.num_envs,
            device=self.cpu_device,
            render_mode=None,
        )
        self.assertIsInstance(discrete_envs, gym.vector.VectorEnv)
        self.assertEqual(self.discrete_env_name, discrete_envs.spec.id)
        discrete_envs.close()

        # Create a vector environment with discrete action spaces with cuda device
        discrete_envs = get_vector_env(
            env_id=self.discrete_env_name,
            num_envs=self.num_envs,
            device=self.cuda_device,
            render_mode=None,
        )
        self.assertIsInstance(discrete_envs, gym.vector.VectorEnv)
        self.assertEqual(self.discrete_env_name, discrete_envs.spec.id)
        discrete_envs.close()

    def test_get_obs_act_space(self):
        """Tests the get_obs_act_space() method."""
        # Test for an environment with continuous action spaces with cpu device
        continous_env = get_env(
            env_id=self.continuous_env_name,
            device=self.cpu_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(continous_env)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Box)
        continous_env.close()

        # Test for an environment with continuous action spaces with cuda device
        continous_env = get_env(
            env_id=self.continuous_env_name,
            device=self.cuda_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(continous_env)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Box)
        continous_env.close()

        # Test for a vectorized environment with continuous action spaces with cpu device
        continous_envs = get_vector_env(
            env_id=self.continuous_env_name,
            num_envs=self.num_envs,
            device=self.cpu_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(continous_envs)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Box)
        continous_envs.close()

        # Test for a vectorized environment with continuous action spaces with cuda device
        continous_envs = get_vector_env(
            env_id=self.continuous_env_name,
            num_envs=self.num_envs,
            device=self.cuda_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(continous_envs)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Box)
        continous_envs.close()

        # Test for an environment with discrete action spaces with cpu device
        discrete_env = get_env(
            env_id=self.discrete_env_name, device=self.cpu_device, render_mode=None
        )
        obs_space, act_space = get_obs_act_space(discrete_env)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Discrete)
        discrete_env.close()

        # Test for an environment with discrete action spaces with cuda device
        discrete_env = get_env(
            env_id=self.discrete_env_name, device=self.cuda_device, render_mode=None
        )
        obs_space, act_space = get_obs_act_space(discrete_env)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Discrete)
        discrete_env.close()

        # Test for a vectorized environment with discrete action spaces with cpu device
        discrete_envs = get_vector_env(
            env_id=self.discrete_env_name,
            num_envs=self.num_envs,
            device=self.cpu_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(discrete_envs)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Discrete)
        discrete_envs.close()

        # Test for a vectorized environment with discrete action spaces with cuda device
        discrete_envs = get_vector_env(
            env_id=self.discrete_env_name,
            num_envs=self.num_envs,
            device=self.cuda_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(discrete_envs)
        self.assertIsInstance(obs_space, gym.spaces.Box)
        self.assertIsInstance(act_space, gym.spaces.Discrete)
        discrete_envs.close()

    def test_get_obs_act_dims(self):
        """Tests the get_obs_act_dims() method."""
        # Test for an environment with continuous action spaces with cpu device
        continous_env = get_env(
            env_id=self.continuous_env_name, device=self.cpu_device, render_mode=None
        )
        obs_space, act_space = get_obs_act_space(continous_env)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(105, obs_dim)
        self.assertEqual(8, act_dim)
        continous_env.close()

        # Test for an environment with continuous action spaces with cuda device
        continous_env = get_env(
            env_id=self.continuous_env_name, device=self.cuda_device, render_mode=None
        )
        obs_space, act_space = get_obs_act_space(continous_env)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(105, obs_dim)
        self.assertEqual(8, act_dim)
        continous_env.close()

        # Test for a vectorized environment with continuous action spaces with cpu device
        continous_envs = get_vector_env(
            env_id=self.continuous_env_name,
            num_envs=self.num_envs,
            device=self.cpu_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(continous_envs)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(105, obs_dim)
        self.assertEqual(8, act_dim)
        continous_envs.close()

        # Test for a vectorized environment with continuous action spaces with cuda device
        continous_envs = get_vector_env(
            env_id=self.continuous_env_name,
            num_envs=self.num_envs,
            device=self.cuda_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(continous_envs)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(105, obs_dim)
        self.assertEqual(8, act_dim)
        continous_envs.close()

        # Test for an environment with discrete action spaces with cpu device
        discrete_env = get_env(
            env_id=self.discrete_env_name, device=self.cpu_device, render_mode=None
        )
        obs_space, act_space = get_obs_act_space(discrete_env)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(4, obs_dim)
        self.assertEqual(2, act_dim)
        discrete_env.close()

        # Test for an environment with discrete action spaces with cuda device
        discrete_env = get_env(
            env_id=self.discrete_env_name, device=self.cuda_device, render_mode=None
        )
        obs_space, act_space = get_obs_act_space(discrete_env)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(4, obs_dim)
        self.assertEqual(2, act_dim)
        discrete_env.close()

        # Test for a vectorized environment with discrete action spaces with cpu device
        discrete_envs = get_vector_env(
            env_id=self.discrete_env_name,
            num_envs=self.num_envs,
            device=self.cpu_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(discrete_envs)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(4, obs_dim)
        self.assertEqual(2, act_dim)
        discrete_envs.close()

        # Test for a vectorized environment with discrete action spaces with cuda device
        discrete_envs = get_vector_env(
            env_id=self.discrete_env_name,
            num_envs=self.num_envs,
            device=self.cuda_device,
            render_mode=None,
        )
        obs_space, act_space = get_obs_act_space(discrete_envs)
        obs_dim, act_dim = get_obs_act_dims(obs_space, act_space)
        self.assertEqual(4, obs_dim)
        self.assertEqual(2, act_dim)
        discrete_envs.close()


if __name__ == "__main__":
    unittest.main()
