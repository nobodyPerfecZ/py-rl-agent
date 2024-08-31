from typing import Optional

import torch

from pyrlagent.torch.buffer import AbstractBuffer
from pyrlagent.torch.experience import Trajectory
from pyrlagent.torch.util import get_device


class ReplayBuffer(AbstractBuffer):
    """A replay buffer for storing transitions in RL after the LIFO principle."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        env_dim: int,
        max_size: int,
        device: str,
    ):
        if obs_dim <= 0:
            raise ValueError(f"obs_dim ({obs_dim}) must be greather than 0.")
        if act_dim <= 0:
            raise ValueError(f"act_dim ({act_dim}) must be greather than 0.")
        if env_dim <= 0:
            raise ValueError(f"env_dim ({env_dim}) must be greather than 0.")
        if max_size <= 0:
            raise ValueError(f"max_size ({max_size}) must be greather than 0.")

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.env_dim = env_dim
        self.max_size = max_size
        self.device = get_device(device)

        self.ptr = None
        self.curr_size = None
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
        self.curr_size = 0
        self.trajectory = Trajectory(
            state=torch.zeros(state_shape, dtype=torch.float32, device=self.device),
            action=torch.zeros(action_shape, dtype=torch.float32, device=self.device),
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

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        log_prob: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        next_value: Optional[torch.Tensor] = None,
    ):
        # Update pointer and current size
        self.ptr = (self.ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

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
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if batch_size > len(self):
            raise ValueError(
                "batch_size must be less than or equal to the current capacity."
            )

        # Get the transitions
        indices = torch.randint(0, self.curr_size, (batch_size,), device=self.device)
        sample = self.trajectory[indices]

        # Replace 0s with None
        if torch.all(sample.log_prob == 0):
            sample = sample._replace(log_prob=None)

        if torch.all(sample.value == 0):
            sample = sample._replace(value=None)

        if torch.all(sample.next_value == 0):
            sample = sample._replace(next_value=None)

        return sample

    def __len__(self) -> int:
        return self.curr_size

    def __str__(self) -> str:
        return f"RingBuffer(max_size={self.max_size})"

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
            "curr_size": self.curr_size,
            "trajectory": self.trajectory,
        }

    def __setstate__(self, state: dict):
        self.obs_dim = state["obs_dim"]
        self.act_dim = state["act_dim"]
        self.env_dim = state["env_dim"]
        self.max_size = state["max_size"]
        self.device = state["device"]
        self.ptr = state["ptr"]
        self.curr_size = state["curr_size"]
        self.trajectory = state["trajectory"]
