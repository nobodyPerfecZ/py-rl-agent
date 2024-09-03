from typing import NamedTuple

import numpy as np
import torch


class Trajectory(NamedTuple):
    """
    A class for representing a Transition in a reinforcement learning environment.

    A Transition (s_t, a_t, r_t, s_t+1, done, log_prob, value) is used to capture the key elements of an agent
    interaction with an environment:
        - s_t := current state
        - a_t := taken action
        - r_t := reward for doing action a_t from state s_t
        - s_t+1 := next state after taking action a_t from state s_t
        - done := Is state s_t+1 a terminal state or not
        - (Optional:) log_prob := log probability of P(a_t | s_t)
        - (Optional:) value := state-value function V(s_t)

    Attributes:
        state (np.ndarray | torch.Tensor):
            The current state s_t in time step t

        action (np.ndarray | torch.Tensor):
            The taken action a_t in time step t

        reward (np.ndarray | torch.Tensor):
            The reward r_t for doing action a_t from state s_t

        next_state (np.ndarray | torch.Tensor):
            The next state s_t+1 after taking action a_t from state s_t

        done (np.ndarray | torch.Tensor):
            Signalizes whether the next state s_t+1 is a terminal state

        log_prob(np.ndarray | torch.Tensor, optional):
            The log probability p(a_t | s_t)

        value(np.ndarray | torch.Tensor, optional):
            The state value V(s_t)

        next_value(np.ndarray | torch.Tensor, optional):
            The state value V(s_t+1)
    """

    state: np.ndarray | torch.Tensor
    action: np.ndarray | torch.Tensor
    reward: np.ndarray | torch.Tensor
    next_state: np.ndarray | torch.Tensor
    done: np.ndarray | torch.Tensor
    log_prob: np.ndarray | torch.Tensor | None = None
    value: np.ndarray | torch.Tensor | None = None
    next_value: np.ndarray | torch.Tensor | None = None

    def __getitem__(self, index: int) -> "Trajectory":
        return Trajectory(
            state=self.state[index],
            action=self.action[index],
            reward=self.reward[index],
            next_state=self.next_state[index],
            done=self.done[index],
            log_prob=None if self.log_prob is None else self.log_prob[index],
            value=None if self.value is None else self.value[index],
            next_value=None if self.next_value is None else self.next_value[index],
        )
