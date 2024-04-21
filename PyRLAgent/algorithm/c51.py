from typing import Any

import torch
import torch.nn.functional as F

from PyRLAgent.algorithm.dqn import DQN


class C51(DQN):
    """
    Categorical DQN (C51) agent for reinforcement learning.

    The corresponding paper can be found here:
    https://arxiv.org/abs/1707.06887

    The C51 agent uses a neural network-based policy to approximate the distribution of returns for stable
    distributional Q-Learning.

    Attributes:
        env_type (str):
            The environment where we want to optimize our agent.
            Either the name or the class itself can be given.

        policy_type (str | Type[Policy]):
            The type of the policy, that we want to use.
            Either the name or the class itself can be given.

        policy_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the policy.

        strategy_type (str | Type[Strategy]):
            The type of exploration strategy that is used for the action selection (exploration-exploitation problem).
            Either the name or the class itself can be given.

        strategy_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the strategy.

        replay_buffer_type (str | Type[Buffer]):
            The type of replay buffer used for storing experiences (experience replay).
            Either the name or the class itself can be given.

        replay_buffer_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the replay buffer.

        optimizer_type (str | Type[Optimizer]):
            The type of optimizer used for training the policy.
            Either the name or the class itself can be given.

        optimizer_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the optimizer.

        loss_type (str | Type["F"]):
            The type of loss function used for training the policy.
            Either the name or the class itself can be given.

        loss_kwargs (Dict[str, Any]):
            Keyword arguments for initializing the loss function.

        max_gradient_norm (float | None):
            The maximum gradient norm for gradient clipping.
            If the value is None, then no gradient clipping is used.

        batch_size (int):
            The batch size for the gradient upgrades.

        tau (float):
            The update rate of the target network (soft update).

        gamma (float):
            The discount factor (necessary for computing the TD-Target).

        target_freq (int):
            The frequency of target network updates.
            After N gradient updates, the target network will be updated.

        train_freq (int):
            The number of steps per gradient update.

        render_freq (int):
            The frequency of rendering the environment.
            After N episodes the environment gets rendered.

        gradient_steps (int):
            The number of gradient updates per training step.
    """

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
            next_probabilities = next_probabilities[range(self.batch_size), best_actions]

            # Compute the projection of t_z onto the support
            t_z = (rewards + ~dones * self.gamma).unsqueeze(1) * self.target_q_net.Z.unsqueeze(0)
            t_z = torch.clamp_(t_z, min=self.target_q_net.Q_min, max=self.target_q_net.Q_max)
            b = (t_z - self.target_q_net.Q_min) / self.target_q_net.delta_Z
            l = b.floor().long()
            u = b.ceil().long()

            # Distribute probability of Tz
            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.target_q_net.num_atoms, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.target_q_net.num_atoms)
            )
            proj_dist = torch.zeros(next_probabilities.size())
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_probabilities * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_probabilities * (b - l.float())).view(-1)
            )

        dist = self.q_net.forward(states)
        dist = dist[range(self.batch_size), actions]

        return self.loss_fn(dist, proj_dist, **self.loss_kwargs), {}
