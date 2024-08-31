import unittest

from pyrlagent.torch.buffer import ReplayBuffer, RolloutBuffer
from pyrlagent.torch.config import BufferConfig, create_buffer


class TestBufferConfig(unittest.TestCase):
    """Tests the BufferConfig class."""

    def setUp(self):
        self.rollout_id = "rollout"
        self.rollout_kwargs = {}

        self.replay_id = "replay"
        self.replay_kwargs = {}

        self.obs_dim = 10
        self.act_dim = 5
        self.env_dim = 10
        self.max_size = 100
        self.device = "cpu"

    def test_init(self):
        """Tests the __init__() method."""
        # Create a rollout buffer configuration
        rollout_config = BufferConfig(id=self.rollout_id, kwargs=self.rollout_kwargs)
        self.assertEqual(rollout_config.id, self.rollout_id)
        self.assertEqual(rollout_config.kwargs, self.rollout_kwargs)

        # Create a replay buffer configuration
        replay_config = BufferConfig(id=self.replay_id, kwargs=self.replay_kwargs)
        self.assertEqual(replay_config.id, self.replay_id)
        self.assertEqual(replay_config.kwargs, self.replay_kwargs)

    def test_create_buffer(self):
        """Tests the create_buffer() method."""
        # Create an invalid buffer
        with self.assertRaises(ValueError):
            create_buffer(
                buffer_config=BufferConfig(id="invalid_id", kwargs={}),
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                env_dim=self.env_dim,
                max_size=self.max_size,
                device=self.device,
            )

        # Create a rollout buffer
        rollout_buffer = create_buffer(
            buffer_config=BufferConfig(id=self.rollout_id, kwargs={}),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            env_dim=self.env_dim,
            max_size=self.max_size,
            device=self.device,
        )
        self.assertIsInstance(rollout_buffer, RolloutBuffer)

        # Create a replay buffer
        replay_buffer = create_buffer(
            buffer_config=BufferConfig(id=self.replay_id, kwargs={}),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            env_dim=self.env_dim,
            max_size=self.max_size,
            device=self.device,
        )
        self.assertIsInstance(replay_buffer, ReplayBuffer)


if __name__ == "__main__":
    unittest.main()
