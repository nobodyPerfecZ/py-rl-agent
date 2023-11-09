import numpy as np
import torch

from PyRLAgent.common.strategy.abstract_strategy import Strategy


class UCB(Strategy):
    """
    A class representing the Upper Confidence Bound (UCB) exploration strategy.

    The UCB strategy balances exploration and exploitation from a reinforcement learning agent.
    It calculates the UCB score for each action, which takes into account both the estimated
    value of the action and the exploration bonus. Actions with higher UCB scores are selected
    to explore and exploit efficiently.

    UCB(s,a) = Q(s,a) + c * sqrt(ln(t) / N(a))
    """

    def __init__(self, c: float):
        if c < 0:
            raise ValueError(
                "Illegal c!"
                "The argument should be higher or equal to 0!"
            )
        self.c = c
        self.counter = {}
        self.timestep = 1

    def _setup_counter(self, state: np.ndarray, output: torch.Tensor):
        """
        Set up the counter tensor to a value that is close to 0 with a shape of the given number of actions.
        This function is only called (once) at the beginning, where counter is not initialized.

        Args:

            state (np.ndarray).
                Current state

            output (torch.Tensor):
                Output of a Pytorch model
        """
        state_as_hash = state.tobytes()
        if state_as_hash not in self.counter:
            self.counter[state_as_hash] = torch.full((output.size()[-1],), 1e-10)

    def _update_counter(self, actions: torch.Tensor):
        """
        Updates the counter tensor according to the given selected actions, where each value of actions represents the
        index of counter. For each selected action we increment the counter for that index by 1.

        Args:
            actions (torch.Tensor):
                Selected actions
        """
        if actions.dim() > 0:
            # Case: multiple actions are given
            for action in actions:
                self.counter[action] += 1
        else:
            # Case: single action is given
            self.counter[actions] += 1

    def _update_timestep(self):
        """
        Updates the timestep counter t by 1.
        """
        self.timestep += 1

    def choose_action(self, state: np.ndarray, output: torch.Tensor) -> torch.Tensor:
        # Set up the counter (if state is newly visited)
        self._setup_counter(state, output)

        # Calculate the ucb bounds
        ucb = output + self.c * torch.sqrt(torch.log(torch.tensor(self.timestep)) / self.counter[state.tobytes()])
        actions = torch.argmax(ucb, dim=-1)

        return actions

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        # Update timestep
        self.timestep += 1

        # Update the counter
        self.counter[state.tobytes()][action] += 1

    def __str__(self) -> str:
        return f"UCB(c={self.c}, timestep={self.timestep})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, UCB):
            return self.c == other.c and \
                   self.timestep == other.timestep and \
                   all(key1 == key2 and torch.all(torch.eq(item1, item2)) for (key1, item1), (key2, item2) in
                       zip(self.counter.items(), other.counter.items()))
        raise NotImplementedError

    def __getstate__(self) -> dict:
        return {
            "c": self.c,
            "counter": self.counter,
            "timestep": self.timestep,
        }

    def __setstate__(self, state: dict):
        self.c = state["c"]
        self.counter = state["counter"]
        self.timestep = state["timestep"]
