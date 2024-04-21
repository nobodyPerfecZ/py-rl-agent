import unittest

from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.enum.buffer import BufferEnum


class TestBufferEnum(unittest.TestCase):
    """
    Tests the enum class BufferEnum.
    """

    def setUp(self):
        self.wrapper = {
            BufferEnum.RING_BUFFER: RingBuffer,
        }

        self.buffer_kwargs1 = {"max_size": 1000}

    def test_wrapper(self):
        """
        Tests the method test_wrapper().
        """
        self.assertDictEqual(self.wrapper, BufferEnum.wrapper())

    def test_to(self):
        """
        Tests the method to().
        """
        buffer1 = BufferEnum.RING_BUFFER.to(**self.buffer_kwargs1)

        self.assertIsInstance(buffer1, RingBuffer)


if __name__ == '__main__':
    unittest.main()
