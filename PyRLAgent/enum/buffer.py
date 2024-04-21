from enum import Enum
from typing import Type

from PyRLAgent.common.buffer.abstract_buffer import Buffer
from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.enum.abstract_enum import AbstractStrEnum


class BufferEnum(AbstractStrEnum):
    """
    An enumeration of supported buffer types.
    """
    RING_BUFFER = "ring-buffer"

    @classmethod
    def wrapper(cls) -> dict[Enum, Type[Buffer]]:
        return {
            cls.RING_BUFFER: RingBuffer,
        }

    def to(self, **buffer_kwargs) -> Buffer:
        return BufferEnum.wrapper()[self](**buffer_kwargs)
