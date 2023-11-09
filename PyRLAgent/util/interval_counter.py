class IntervalCounter:
    """
    A simple counter for tracking intervals or periods of time.

    This class provides functionality to count and track intervals or periods of time.
    It is commonly used to monitor when a specified interval has elapsed.
    """

    def __init__(self, initial_value: int = 0, modulo: int = 1):
        if initial_value < 0:
            raise ValueError("Illegal initial_value! The argument should be higher or equal to 0!")

        if modulo <= 0:
            raise ValueError("Illegal modulo! The argument should be higher or equal to 1!")

        self.initial_value = initial_value
        self.curr_value = initial_value
        self.modulo = modulo

    def reset(self):
        """
        Resets the counter to the initial value.
        """
        self.curr_value = self.initial_value

    def increment(self, value: int = 1):
        """
        Increments the counter by the given value.

        Args:
            value (int):
                Add to the current counter value
        """
        if value <= 0:
            raise ValueError("Illegal value! The argument should be higher or equal to 1!")
        self.curr_value = (self.curr_value + value) % self.modulo

    def get_value(self) -> int:
        """
        Returns:
            int:
                Current counter value
        """
        return self.curr_value

    def is_interval_reached(self) -> bool:
        """
        Returns:
            bool:
                True, if the interval is reached (counter value is equal to the given initial value)
        """
        return self.curr_value == self.initial_value

    def __str__(self) -> str:
        return f"IntervalCounter(initial_value={self.initial_value}, curr_value={self.curr_value}, modulo={self.modulo})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, IntervalCounter):
            return self.initial_value == other.initial_value and \
                   self.curr_value == other.curr_value and \
                   self.modulo == other.modulo
        raise NotImplementedError

    def __getstate__(self) -> dict:
        """ Magic function to save a custom class as yaml file. """
        return {
            "initial_value": self.initial_value,
            "curr_value": self.curr_value,
            "modulo": self.modulo,
        }

    def __setstate__(self, state: dict):
        """ Magic function to load a custom class from yaml file. """
        self.initial_value = state["initial_value"]
        self.curr_value = state["curr_value"]
        self.modulo = state["modulo"]
