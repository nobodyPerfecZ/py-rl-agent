from collections import deque

import torch

from PyRLAgent.common.buffer.abstract_buffer import Buffer, Transition


class RingBuffer(Buffer):
    """
    This class represents a circular replay buffer.

    The circular replay buffer stores a specified maximum number of Transitions (s_t, a_t, r_t, s_t+1, done).

    If the maximum number of Transitions are reached (~ full), adding new Transitions will overwrite the oldest
    Transition in a circular manner after the LIFO (Last In, First Out) principle.

    Attributes:
        max_size (int):
            The maximum number of Transitions allowed to be stored
    """

    def __init__(self, max_size: int):
        if max_size <= 0:
            raise ValueError("Illegal max_size. The argument should be >= 1!")

        self.max_size = max_size
        self.memory = deque(maxlen=max_size)

    def reset(self):
        self.memory.clear()

    def full(self) -> bool:
        return len(self) == self.max_size

    def filled(self, min_size: int) -> bool:
        if min_size <= 0 or min_size > self.max_size:
            raise ValueError("Illegal min_size. The argument should be 1 <= min_size <= max_size!")
        return len(self) >= min_size

    def _push(self, transition: Transition):
        if self.full():
            self.memory.popleft()
        self.memory.append(transition)

    def _get(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("Illegal batch_size. The argument should be >= 1!")
        if not self.filled(batch_size):
            raise ValueError("Illegal call of get()! There are not enough transitions stored!")

        return [self.memory[idx] for idx in range(batch_size)]

    def _sample(self, batch_size: int) -> list[Transition]:
        if batch_size <= 0:
            raise ValueError("Illegal batch_size. The argument should be >= 1!")
        if not self.filled(batch_size):
            raise ValueError("Illegal call of sample()! There are not enough transitions stored!")

        indices = torch.randint(0, len(self), size=(batch_size,))
        return [self.memory[idx] for idx in indices]

    def __len__(self) -> int:
        return self.memory.__len__()

    def __str__(self) -> str:
        return f"RingBuffer(max_size={self.max_size})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, RingBuffer):
            return self.max_size == other.max_size and \
                all(item1 == item2 for item1, item2 in zip(self.memory, other.memory))
        raise NotImplementedError

    def __getstate__(self) -> dict:
        return {
            "max_size": self.max_size,
            "memory": self.memory,
        }

    def __setstate__(self, state: dict):
        self.max_size = state["max_size"]
        self.memory = state["memory"]
