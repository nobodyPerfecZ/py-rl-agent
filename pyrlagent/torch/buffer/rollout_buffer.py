from typing import Any, Optional

import numpy as np
import torch

from pyrlagent.torch.buffer.abstract_buffer import AbstractBuffer
from pyrlagent.torch.experience.trajectory import Trajectory


class RolloutBuffer(AbstractBuffer):
    """A time-related sequence of transitions."""

    def __init__(
        self, obs_dim: int, act_dim: int, env_dim: int, max_size: int, device: str
    ):
        assert obs_dim > 0, "obs_dim must be greater than 0."
        assert act_dim > 0, "act_dim must be greater than 0."
        assert env_dim > 0, "env_dim must be greater than 0."
        assert max_size > 0, "max_size must be greater than 0."

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.max_size = max_size
        self.device = device

        self.ptr = None
        self.trajectory = None

        # Initialize memory
        self.reset()

    def reset(self):
        state_shape = (
            (self.max_size, self.env_dim, self.obs_dim)
            if self.obs_dim > 1
            else (self.max_size, self.env_dim)
        )
        action_shape = (
            (self.max_size, self.env_dim, self.act_dim)
            if self.act_dim > 1
            else (self.max_size, self.env_dim)
        )
        single_value_shape = (self.max_size, self.env_dim)

        self.ptr = -1

        if self.device == "cuda":
            self.trajectory = Trajectory(
                state=torch.zeros(state_shape, dtype=torch.float32, device=self.device),
                action=torch.zeros(
                    action_shape, dtype=torch.float32, device=self.device
                ),
                reward=torch.zeros(
                    single_value_shape, dtype=torch.float32, device=self.device
                ),
                next_state=torch.zeros(
                    state_shape, dtype=torch.float32, device=self.device
                ),
                done=torch.zeros(
                    single_value_shape, dtype=torch.float32, device=self.device
                ),
                log_prob=torch.zeros(
                    single_value_shape, dtype=torch.float32, device=self.device
                ),
                value=torch.zeros(
                    single_value_shape, dtype=torch.float32, device=self.device
                ),
                next_value=torch.zeros(
                    single_value_shape, dtype=torch.float32, device=self.device
                ),
            )
        else:
            self.trajectory = Trajectory(
                state=np.zeros(state_shape, dtype=np.float32),
                action=np.zeros(action_shape, dtype=np.float32),
                reward=np.zeros(single_value_shape, dtype=np.float32),
                next_state=np.zeros(state_shape, dtype=np.float32),
                done=np.zeros(single_value_shape, dtype=np.float32),
                log_prob=np.zeros(single_value_shape, dtype=np.float32),
                value=np.zeros(single_value_shape, dtype=np.float32),
                next_value=np.zeros(single_value_shape, dtype=np.float32),
            )

    def push(
        self,
        state: Any,
        action: Any,
        reward: Any,
        next_state: Any,
        done: Any,
        log_prob: Optional[Any] = None,
        value: Optional[Any] = None,
        next_value: Optional[Any] = None,
    ):
        # Update pointer
        self.ptr = (self.ptr + 1) % self.max_size

        # Update transition
        self.trajectory.state[self.ptr] = state
        self.trajectory.action[self.ptr] = action
        self.trajectory.reward[self.ptr] = reward
        self.trajectory.next_state[self.ptr] = next_state
        self.trajectory.done[self.ptr] = done

        if log_prob is not None:
            self.trajectory.log_prob[self.ptr] = log_prob

        if value is not None:
            self.trajectory.value[self.ptr] = value

        if next_value is not None:
            self.trajectory.next_value[self.ptr] = next_value

    def sample(self, batch_size: int) -> Trajectory:
        assert batch_size > 0, "batch_size must be greater than 0."
        assert batch_size <= len(
            self
        ), "batch_size must be less than or equal to the current capacity."

        # Get the trajectory
        sample = self.trajectory[:batch_size]

        # Replace 0s with None
        if self.device == "cuda":
            if torch.all(sample.log_prob == 0):
                sample = sample._replace(log_prob=None)

            if torch.all(sample.value == 0):
                sample = sample._replace(value=None)

            if torch.all(sample.next_value == 0):
                sample = sample._replace(next_value=None)
        else:
            if np.all(sample.log_prob == 0):
                sample = sample._replace(log_prob=None)

            if np.all(sample.value == 0):
                sample = sample._replace(value=None)

            if np.all(sample.next_value == 0):
                sample = sample._replace(next_value=None)

        return sample

    def __len__(self) -> int:
        return self.ptr + 1

    def __str__(self) -> str:
        return f"Trajectory(max_size={self.max_size})"

    def __repr__(self) -> str:
        return str(self)

    def __getstate__(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "env_dim": self.env_dim,
            "max_size": self.max_size,
            "device": self.device,
            "ptr": self.ptr,
            "trajectory": self.trajectory,
        }

    def __setstate__(self, state: dict):
        self.obs_dim = state["obs_dim"]
        self.act_dim = state["act_dim"]
        self.env_dim = state["env_dim"]
        self.max_size = state["max_size"]
        self.device = state["device"]
        self.ptr = state["ptr"]
        self.trajectory = state["trajectory"]
