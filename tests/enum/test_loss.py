import unittest

import torch.nn.functional as F

from PyRLAgent.enum.loss import LossEnum


class TestLossEnum(unittest.TestCase):
    """
    Tests the enum class LossEnum.
    """

    def setUp(self):
        self.wrapper = {
            LossEnum.MAE: F.l1_loss,
            LossEnum.MSE: F.mse_loss,
            LossEnum.HUBER: F.huber_loss,
            LossEnum.CE_LOGITS: F.cross_entropy,
            LossEnum.KL_LOGITS: F.kl_div,
        }

    def test_wrapper(self):
        """
        Tests the method test_wrapper().
        """
        self.assertDictEqual(self.wrapper, LossEnum.wrapper())

    def test_to(self):
        """
        Tests the method to().
        """
        loss_fn1 = LossEnum.MAE.to()
        loss_fn2 = LossEnum.MSE.to()
        loss_fn3 = LossEnum.HUBER.to()
        loss_fn4 = LossEnum.CE_LOGITS.to()
        loss_fn5 = LossEnum.KL_LOGITS.to()

        self.assertTrue(callable(loss_fn1))
        self.assertTrue(callable(loss_fn2))
        self.assertTrue(callable(loss_fn3))
        self.assertTrue(callable(loss_fn4))
        self.assertTrue(callable(loss_fn5))


if __name__ == '__main__':
    unittest.main()
