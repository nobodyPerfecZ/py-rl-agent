from collections import deque
import torch

from PyRLAgent.common.buffer.abstract_buffer import Buffer, Transition


class RingBuffer(Buffer):
    """
    A class for representing a circular replay buffer.

    The fixed-size circular buffer stores a specified maximum number of Transitions (s, a, r, s', done).
    When the buffer is full, adding new Transitions will overwrite the oldest items in a circular manner after the
    LIFO (Last In, First Out) principle.
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError(
                "Illegal max_size!"
                "The argument should be higher than or equal to 1!"
            )
        self.max_size = max_size
        self.memory = deque(maxlen=max_size)

    def reset(self):
        self.memory.clear()

    def full(self) -> bool:
        return len(self) == self.max_size

    def filled(self, minimum_size: int) -> bool:
        return len(self) >= minimum_size

    def _push(self, transition: Transition):
        if self.full():
            # Case: Drop the oldest transition from the memory
            self.memory.popleft()

        # Append a single transition to the memory
        self.memory.append(transition)

    def _sample(self, batch_size: int) -> list[Transition]:
        if not self.filled(batch_size):
            raise ValueError(
                "Illegal call of sample()!"
                "There are not enough transitions stored to sample from the buffer!"
            )
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
        """ Magic function to save a custom class as yaml file. """
        return {
            "max_size": self.max_size,
            "memory": self.memory,
        }

    def __setstate__(self, state: dict):
        """ Magic function to load a custom class from yaml file. """
        self.max_size = state["max_size"]
        self.memory = state["memory"]
