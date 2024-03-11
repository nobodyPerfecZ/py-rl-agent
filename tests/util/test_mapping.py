import unittest

import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR

from PyRLAgent.algorithm.policy import QNetwork
from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.util.mapping import get_value


class TestMapping(unittest.TestCase):
    """
    Tests all methods from util.mapping.py
    """

    def setUp(self):
        self.replay_buffer_map = {
            "ring": RingBuffer,
        }
        self.replay_buffer_type_name = "ring"
        self.replay_buffer_type_class = RingBuffer

        self.lr_scheduler_map = {
            "linear-lr": LinearLR,
            "exp-lr": ExponentialLR,
            "step-lr": StepLR,
        }
        self.lr_scheduler_type_name = "linear-lr"
        self.lr_scheduler_type_class = LinearLR

        self.optimizer_map = {
            "adam": Adam,
            "adamw": AdamW,
            "sgd": SGD,
        }
        self.optimizer_type_name = "sgd"
        self.optimizer_type_class = SGD

        self.policy_map = {
            "q-net": QNetwork,
        }
        self.policy_type_name = "q-net"
        self.policy_type_class = QNetwork

        self.loss_map = {
            "mae": F.l1_loss,
            "mse": F.mse_loss,
            "bce": F.binary_cross_entropy,
            "bce-logits": F.binary_cross_entropy_with_logits,
            "ce-logits": F.cross_entropy,
            "huber": F.huber_loss,
        }
        self.loss_type_name = "huber"
        self.loss_type_class = F.huber_loss

    def test_get_value(self):
        """
        Tests the method get_value().
        """
        replay_buffer_type1 = get_value(self.replay_buffer_map, self.replay_buffer_type_name)
        replay_buffer_type2 = get_value(self.replay_buffer_map, self.replay_buffer_type_class)
        self.assertEqual(self.replay_buffer_type_class, replay_buffer_type1)
        self.assertEqual(self.replay_buffer_type_class, replay_buffer_type2)

        lr_scheduler_type1 = get_value(self.lr_scheduler_map, self.lr_scheduler_type_name)
        lr_scheduler_type2 = get_value(self.lr_scheduler_map, self.lr_scheduler_type_class)
        self.assertEqual(self.lr_scheduler_type_class, lr_scheduler_type1)
        self.assertEqual(self.lr_scheduler_type_class, lr_scheduler_type2)

        optimizer_type1 = get_value(self.optimizer_map, self.optimizer_type_name)
        optimizer_type2 = get_value(self.optimizer_map, self.optimizer_type_class)
        self.assertEqual(self.optimizer_type_class, optimizer_type1)
        self.assertEqual(self.optimizer_type_class, optimizer_type2)

        policy_type1 = get_value(self.policy_map, self.policy_type_name)
        policy_type2 = get_value(self.policy_map, self.policy_type_class)
        self.assertEqual(self.policy_type_class, policy_type1)
        self.assertEqual(self.policy_type_class, policy_type2)

        loss_type1 = get_value(self.loss_map, self.loss_type_name)
        loss_type2 = get_value(self.loss_map, self.loss_type_class)
        self.assertEqual(self.loss_type_class, loss_type1)
        self.assertEqual(self.loss_type_class, loss_type2)


if __name__ == '__main__':
    unittest.main()
