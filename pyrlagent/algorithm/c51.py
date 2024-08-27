from typing import Any

import torch
import torch.nn.functional as F

from PyRLAgent.algorithm.dqn import DQN


class C51(DQN):

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Calculate the target probability distribution
        with torch.no_grad():
            # Calculate the probabilities of each bin
            next_logits = self.target_q_net.forward(next_states)
            next_probabilities = F.softmax(next_logits, dim=-1)

            # Calculate the q-values := sum(z * probabilities)
            next_q_values = torch.sum(self.target_q_net.Z * next_probabilities, dim=-1)

            # Select the best action with argmax (N,)
            best_actions = torch.argmax(next_q_values, dim=-1)

            # Select from the target probabilities the ones with the best actions
            next_probabilities = next_probabilities.gather(
                dim=2, index=best_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, next_probabilities.shape[-1])
            ).squeeze()

            # Compute the projection of t_z onto the support
            t_z = (rewards + ~dones * self.gamma).unsqueeze(-1) * self.target_q_net.Z.unsqueeze(0)
            t_z = torch.clamp_(t_z, min=self.target_q_net.Q_min, max=self.target_q_net.Q_max)
            b = (t_z - self.target_q_net.Q_min) / self.target_q_net.delta_Z
            l = b.floor().long()
            u = b.ceil().long()

            # Distribute probability of Tz
            offset = (
                torch.linspace(
                    0, (self.steps_per_trajectory - 1) * self.target_q_net.num_atoms, self.steps_per_trajectory
                ).long()
                .unsqueeze(-1)
                .unsqueeze(-1)
                .expand(self.steps_per_trajectory, self.num_envs, self.target_q_net.num_atoms)
            )
            proj_dist = torch.zeros(next_probabilities.shape)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_probabilities * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_probabilities * (b - l.float())).view(-1)
            )

        dist = self.q_net.forward(states)
        dist = dist.gather(
            dim=2, index=actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, dist.shape[-1])
        ).squeeze()

        return self.loss_fn(
            dist,
            proj_dist,
            **self.loss_kwargs
        ), {}
