import torch

from PyRLAgent.algorithm.dqn import DQN


class DDQN(DQN):
    """
    Double DQN agent for reinforcement learning.

    The corresponding paper can be found here:
    https://arxiv.org/abs/1509.06461

    The DDQN agent uses a neural network-based policy to approximate the Q-function and employs techniques
    such as experience replay, target networks and double Q-learning for stable Q-learning.

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

        learning_starts (int):
            The number of steps before starting Q-learning updates.

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
    ) -> torch.Tensor:
        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            Q_next = self.target_q_net.forward(next_states)
            a_next = self.q_net.forward(next_states).argmax(dim=1)
            q_targets = rewards + ~dones * self.gamma * Q_next.gather(dim=1, index=a_next.unsqueeze(1)).squeeze()

            # Clip the target q-values
            q_targets = torch.clamp_(q_targets, min=self.target_q_net.Q_min, max=self.target_q_net.Q_max)

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        q_values = Q.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        return self.loss_fn(q_values, q_targets, **self.loss_kwargs)


class ClippedDDQN(DDQN):
    """
    Clipped DDQN agent for reinforcement learning.

    The corresponding paper can be found here:
    https://arxiv.org/abs/1802.09477

    The DDQN agent uses a neural network-based policy to approximate the Q-function and employs techniques
    such as experience replay, target networks and pessimistic double Q-learning for stable Q-learning.

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

        learning_starts (int):
            The number of steps before starting Q-learning updates.

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
    ) -> torch.Tensor:
        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            Q_next1 = self.q_net.forward(next_states)
            Q_next2 = self.target_q_net.forward(next_states)
            a_next1 = Q_next1.argmax(dim=1)
            a_next2 = Q_next2.argmax(dim=1)
            q_targets1 = Q_next1.gather(dim=1, index=a_next1.unsqueeze(1)).squeeze()
            q_targets2 = Q_next2.gather(dim=1, index=a_next2.unsqueeze(1)).squeeze()
            q_targets = rewards + ~dones * self.gamma * torch.min(q_targets1, q_targets2)

            # Clip the target q-values
            q_targets = torch.clamp_(q_targets, min=self.target_q_net.Q_min, max=self.target_q_net.Q_max)

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        q_values = Q.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        return self.loss_fn(q_values, q_targets, **self.loss_kwargs)
