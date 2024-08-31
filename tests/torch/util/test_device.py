import unittest

from pyrlagent.torch.util import get_device


class TestDevice(unittest.TestCase):
    """Tests the methods under device.py."""

    def test_get_device(self):
        """Tests the get_device() method."""
        device1 = get_device(device="auto")
        device2 = get_device(device="cpu")
        device3 = get_device(device="cuda")

        self.assertEqual("cuda", device1)
        self.assertEqual("cpu", device2)
        self.assertEqual("cuda", device3)


if __name__ == "__main__":
    unittest.main()
