from typing import Any

import torch

from PyRLAgent.algorithm.dqn import DQN


class DDQN(DQN):

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            Q_next = self.target_q_net.forward(next_states)
            a_next = self.q_net.forward(next_states).argmax(dim=-1)
            targets = rewards + ~dones * self.gamma * Q_next.gather(dim=-1, index=a_next.unsqueeze(-1)).squeeze()

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        predicted = Q.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()

        return self.loss_fn(
            predicted.reshape(self.num_envs * self.steps_per_trajectory, -1),
            targets.reshape(self.num_envs * self.steps_per_trajectory, -1),
            **self.loss_kwargs
        ), {}


class ClippedDDQN(DDQN):

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            Q_next1 = self.q_net.forward(next_states)
            Q_next2 = self.target_q_net.forward(next_states)
            a_next1 = Q_next1.argmax(dim=-1)
            a_next2 = Q_next2.argmax(dim=-1)
            targets1 = Q_next1.gather(dim=-1, index=a_next1.unsqueeze(-1)).squeeze()
            targets2 = Q_next2.gather(dim=-1, index=a_next2.unsqueeze(-1)).squeeze()
            targets = rewards + ~dones * self.gamma * torch.min(targets1, targets2)

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        predicted = Q.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze()

        return self.loss_fn(
            predicted.reshape(self.num_envs * self.steps_per_trajectory, -1),
            targets.reshape(self.num_envs * self.steps_per_trajectory, -1),
            **self.loss_kwargs
        ), {}
