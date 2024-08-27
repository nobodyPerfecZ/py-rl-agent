from abc import abstractmethod
from enum import Enum
from typing import Any


class AbstractStrEnum(str, Enum):
    """
    An abstract class of a StrEnum.
    """

    @classmethod
    @abstractmethod
    def wrapper(cls) -> dict[Enum, Any]:
        """
        Returns the wrapper dictionary {key: value}, where
            - key represents enum type
            - value represents the class

        Returns:
            dict[Enum, Any]:
                The wrapper of the Enum class
        """
        pass

    @abstractmethod
    def to(self, **kwargs) -> Any:
        """
        Initialize a new instance given the arguments.

        Args:
            **kwargs:
                Arguments for the enum class

        Returns:
            Any:
                The new instance
        """
        pass
