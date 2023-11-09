from typing import Union, Type, Any, Dict, Tuple
from gymnasium import Env
import torch.nn.functional as F
import numpy as np
import gymnasium
import torch.nn as nn
import torch
import copy

from PyRLAgent.common.strategy.abstract_strategy import Strategy
from PyRLAgent.common.policy.abstract_policy import Policy
from PyRLAgent.common.buffer.abstract_buffer import Buffer
from PyRLAgent.common.buffer.ring_buffer import RingBuffer
from PyRLAgent.dqn.policy import QNetwork

from PyRLAgent.util.interval_counter import IntervalCounter


class DQN:
    """
    Deep Q-Network (DQN) agent for reinforcement learning.
    The corresponding paper can be found here: https://arxiv.org/abs/1312.5602

    The DQN agent uses a neural network-based policy to approximate the Q-function and employs techniques
    such as experience replay and target networks for stable Q-learning.

        Args:
            env_type (str):
                The environment where we want to optimize our agent.
                Either the name or the class itself can be given.

            policy_type (Union[str, Type[Policy]]):
                The type of the policy, that we want to use.
                Either the name or the class itself can be given.

            policy_kwargs (Dict[str, Any]):
                Keyword arguments for initializing the policy.

            strategy_type (Union[str, Type[Strategy]]):
                The type of exploration strategy that is used for the action selection (exploration-exploitation
                problem).
                Either the name or the class itself can be given.

            strategy_kwargs (Dict[str, Any]):
                Keyword arguments for initializing the strategy.

            replay_buffer_type (Union[str, Type[Buffer]]):
                The type of replay buffer used for storing experiences (experience replay).
                Either the name or the class itself can be given.

            replay_buffer_kwargs (Dict[str, Any]):
                Keyword arguments for initializing the replay buffer.

            optimizer_type (Union[str, Type[torch.optim.Optimizer]]):
                The type of optimizer used for training the policy.
                Either the name or the class itself can be given.

            optimizer_kwargs (Dict[str, Any]):
                Keyword arguments for initializing the optimizer.

            loss_type (Union[str, Type["torch.nn.functional"]]):
                The type of loss function used for training the policy.
                Either the name or the class itself can be given.

            loss_kwargs (Dict[str, Any]):
                Keyword arguments for initializing the loss function.

            learning_starts (int):
                The number of steps before starting Q-learning updates.

            max_gradient_norm (Union[float, None]):
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

            train_freq (tuple[int, str]):
                The frequency of training steps.
                The first entry specifies the frequency of the scope. The second entry specifies the scope,
                which is either "steps" or "episodes".

            render_freq (int):
                The frequency of rendering the environment.
                After N episodes the environment gets rendered.

            gradient_steps (int):
                The number of gradient updates per training step.
        """

    def __init__(
            self,
            env_type: str,
            policy_type: Union[str, Type[Policy]],
            policy_kwargs: Dict[str, Any],
            strategy_type: Union[str, Type[Strategy]],
            strategy_kwargs: Dict[str, Any],
            replay_buffer_type: Union[str, Type[Buffer]],
            replay_buffer_kwargs: Dict[str, Any],
            optimizer_type: Union[str, Type[torch.optim.Optimizer]],
            optimizer_kwargs: Dict[str, Any],
            loss_type: Union[str, Type["torch.nn.functional"]],
            loss_kwargs: Dict[str, Any],
            learning_starts: int,
            max_gradient_norm: Union[float, None],
            batch_size: int,
            tau: float,
            gamma: float,
            target_freq: int,
            train_freq: tuple[int, str],
            render_freq: int,
            gradient_steps: int,
    ):
        self.env_type = env_type
        self.env, self.render_env = DQN._get_env(env_type)
        self.q_net = DQN._get_policy_type(policy_type)(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            **policy_kwargs,
            strategy_type=strategy_type,
            strategy_kwargs=strategy_kwargs,
        )
        self.target_q_net = copy.deepcopy(self.q_net)
        self.target_q_net.freeze()

        self.replay_buffer = DQN._get_replay_buffer_type(replay_buffer_type)(**replay_buffer_kwargs)
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer = DQN._get_optimizer_type(self.optimizer_type)(params=self.q_net.parameters(), **self.optimizer_kwargs)
        self.loss_fn = DQN._get_loss_type(loss_type)
        self.loss_kwargs = loss_kwargs
        self.max_gradient_norm = max_gradient_norm
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma

        self.target_freq = target_freq
        self.train_freq = train_freq
        self.render_freq = render_freq
        self.gradient_steps = gradient_steps
        self.target_count = IntervalCounter(initial_value=0, modulo=self.target_freq)
        self.train_count = IntervalCounter(initial_value=0, modulo=self.train_freq[0])
        self.render_count = IntervalCounter(initial_value=0, modulo=self.render_freq)

    @staticmethod
    def _get_env(env_type: str) -> Tuple[Env, Env]:
        """
        Create and return a pair of Gymnasium environments for a given environment name.

        This method is a convenience function for creating two instances of the same Gym environment, one configured
        with a "human" render mode for visualization and the other for training or interaction without rendering.

        Args:
            env_type (str):
                The name of the Gym environment to create.

        Returns:
            Tuple[Env, Env]:
                A tuple with the following information's:
                    - [0] (Env): environment for training (without rendering)
                    - [1] (Env): environment for visualization (with rendering)
        """
        env = gymnasium.make(env_type, render_mode="rgb_array")
        render_env = gymnasium.make(env_type, render_mode="human")
        return env, render_env

    @staticmethod
    def _get_policy_type(policy_type: Union[str, Type[Policy]]) -> Type[Policy]:
        """
        Get and return a policy class based on the specified policy type.

        This method allows for checking and returning a policy class based on the given policy type.
        It can be either a string key or the class itself.

        The following keys are allowed for the DQN algorithm:
            - "q-net" := QNetwork

        Args:
            policy_type (Union[str, Type[Policy]]):
                The policy type, which can be a string or the class itself.

        Returns:
            Type[Policy]:
                The concrete policy class corresponding to the specified policy type.
        """
        policy_type_map = {
            "q-net": QNetwork,
        }

        if isinstance(policy_type, str):
            policy_type = policy_type_map.get(policy_type)
            if policy_type is None:
                raise ValueError(
                    "Illegal policy_type!"
                    "The argument should be 'q-net'!"
                )
        else:
            if policy_type not in policy_type_map.items():
                raise ValueError(
                    "Illegal policy_type."
                    "The argument should be 'QNetwork'!"
                )
        return policy_type

    @staticmethod
    def _get_replay_buffer_type(replay_buffer_type: Union[str, Type[Buffer]]) -> Type[Buffer]:
        """
        Get and return a replay buffer class based on the specified buffer type.

        This method allows for checking and returning a buffer class based on the given buffer type.
        It can be either a string key or the class itself.

        The following keys are allowed for the DQN algorithm:
            - "ring" := RingBuffer

        Args:
            replay_buffer_type (Union[str, Type[Buffer]]):
                The buffer type, which can be a string or a buffer class.

        Returns:
            Type[Buffer]:
                The concrete replay buffer class corresponding to the specified replay buffer type.
        """
        replay_buffer_type_map = {
            "ring": RingBuffer,
        }

        if isinstance(replay_buffer_type, str):
            replay_buffer_type = replay_buffer_type_map.get(replay_buffer_type)
            if replay_buffer_type is None:
                raise ValueError(
                    "Illegal replay_buffer_type!"
                    "The argument should be either 'ring'!"
                )
        else:
            if replay_buffer_type not in replay_buffer_type_map.items():
                raise ValueError(
                    "Illegal replay_buffer_type!"
                    "The argument should be either 'RingBuffer'!"
                )
        return replay_buffer_type

    @staticmethod
    def _get_optimizer_type(optimizer_type: Union[str, Type[torch.optim.Optimizer]]) -> Type[torch.optim.Optimizer]:
        """
        Get and return an optimizer class based on the specified optimizer type.

        This method allows for checking and returning an optimizer class based on the given optimizer type.
        It can be either a string key or the class itself.

        The following keys are allowed for the DQN algorithm:
            - "adam" := Adam
            - "adamw" := AdamW
            - "sgd" := SGD

        Args:
            optimizer_type (Union[str, Type[torch.optim.Optimizer]]):
                The optimizer type, which can be a string or an optimizer class.

        Returns:
            Type[torch.optim.Optimizer]:
                The concrete optimizer class corresponding to the specified optimizer type.
        """
        optimizer_type_map = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }

        if isinstance(optimizer_type, str):
            optimizer_type = optimizer_type_map.get(optimizer_type)
            if optimizer_type is None:
                raise ValueError(
                    "Illegal optimizer_type! The argument should be either 'adam', 'adamw' or 'sgd'!"
                )
        else:
            if optimizer_type not in optimizer_type_map.items():
                raise ValueError(
                    "Illegal optimizer_type! The argument should be either 'Adam', 'AdamW' or 'SGD'!"
                )
        return optimizer_type

    @staticmethod
    def _get_loss_type(loss_type: Union[str, Type["torch.nn.functional"]]) -> Type["torch.nn.functional"]:
        """
        Get and return a loss function based on the specified loss function type.

        This method allows for checking and returning a loss function based on the given loss function type.
        It can be either a string key or the class itself.

        The following keys are allowed for the DQN algorithm:
            - "mse" := mse_loss
            - "huber" := huber_loss

        Args:
            loss_type (Union[str, Type["torch.nn.functional"]]):
                The loss type, which can be a string or a loss function.

        Returns:
            Type["torch.nn.functional"]:
                The concrete loss function corresponding to the specified loss type.
        """
        loss_type_map = {
            "mse": F.mse_loss,
            "huber": F.huber_loss,
        }

        if isinstance(loss_type, str):
            loss_type = loss_type_map.get(loss_type)
            if loss_type is None:
                raise ValueError(
                    "Illegal loss_type!"
                    "The argument should be either 'mse' or 'huber'!"
                )
        else:
            if loss_type not in loss_type_map.items():
                raise ValueError(
                    "Illegal optimizer_type!"
                    "The argument should be either 'mse_loss' or 'huber_loss'!"
                )
        return loss_type

    def _reset(self):
        """
        Resets the environments to the start state and counters to 0.
        This method is necessary before starting with training the agent.
        """
        self.env.reset()
        self.render_env.reset()
        self.target_count.reset()
        self.train_count.reset()
        self.render_count.reset()

    def _apply_gradient_norm(self):
        """
        Applies gradient clipping on the policy weights based on the given maximum gradient norm.
        If the maximum gradient norm is not given, then no gradient norm will be performed.
        """
        if self.max_gradient_norm is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_gradient_norm)

    def _soft_update(self):
        """
        Performs a soft update of the target network parameters.

        The soft update is performed by blending the current target network parameters with the current model
        network parameters based on the given tau.
        """
        if self.target_count.is_interval_reached():
            for model_params, target_params in zip(self.q_net.parameters(), self.target_q_net.parameters()):
                target_params.copy_(
                    self.tau * model_params + (1 - self.tau) * target_params
                )
        # Update the counter for soft updates
        self.target_count.increment()

    def train(self):
        """
        Train the DQN agent using experience stored in the replay buffer.

        This method is responsible for training the DQN agent using experiences collected in the replay buffer.
        It typically involves sampling mini-batches of experiences, computing Q-value targets, and updating
        the Q-network's parameters through gradient descent.

        After updating the Q-network parameters we apply gradient clipping and soft update of the target network.
        """
        self.q_net.train(True)
        self.target_q_net.train(False)

        losses = []
        for _ in range(self.gradient_steps):
            # Sample from replay buffer
            samples = self.replay_buffer.sample(self.batch_size)

            self.optimizer.zero_grad()
            loss = self.compute_loss(
                states=samples.states,
                actions=samples.actions,
                rewards=samples.rewards,
                next_states=samples.next_states,
                dones=samples.dones
            )
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self._apply_gradient_norm()
            self._soft_update()

        return np.mean(losses)

    def compute_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            next_states: torch.Tensor,
            dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the loss between the predicted q-values and the q-targets based on the given loss function.

        In DQN (with target network) the update rule is:
            - loss = L(Q_predicted, Q_target), where
            - Q_predicted := Q(s,a)
            - Q_target := r(s,a) + gamma * max_a' Q_target(s', a')

        Args:
            states (torch.Tensor):
                Minibatch of states s

            actions (torch.Tensor):
                Minibatch of actions a

            rewards (torch.Tensor):
                Minibatch of rewards r

            next_states (torch.Tensor):
                Minibatch of next_states s'

            dones (torch.Tensor):
                Minibatch of dones

        Returns:
            torch.Tensor:
                Loss between predicted and target Q-Values
        """

        # Calculate the target q-values (TD-Target)
        with torch.no_grad():
            # DQN learning rule
            Q_next = self.target_q_net.forward(next_states)
            q_targets = rewards + ~dones * self.gamma * Q_next.max(dim=1)[0]

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        q_values = Q.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        return self.loss_fn(q_values, q_targets, **self.loss_kwargs)

    def predict(self, observation: np.ndarray, deterministic: bool) -> torch.Tensor:
        """
        Predicts the next action a given the observation by interacting with the environment.

        Args:
            observation (np.ndarray):
                Current observation getting by interacting with the given environment.

            deterministic (bool):
                Either the deterministic exploration strategy (:= True) or non-deterministic
                exploration strategy (:= False) should be used for selecting the action.

        Returns:
            torch.Tensor:
                Selected action as Pytorch Tensor
        """
        return self.q_net.predict(observation, deterministic)

    def fit(self, n_episodes: int, episode_length: int) -> list[float]:
        """
        The main training loop, where the agent is trained for N episodes with an episode length of L.

        The Gradient-based approach is based on the given algorithm (e. g.: DQN).

        Args:
            n_episodes (int):
                Number of episodes to train the agent.

            episode_length (int):
                The maximum length of each individual episode.

        Returns:
            list[float]:
                Accumulated rewards for each episode
        """
        # Reset parameters
        self._reset()
        rewards = []

        for generation in range(n_episodes):
            if self.render_count.is_interval_reached():
                curr_env = self.render_env
            else:
                curr_env = self.env
            observation, info = curr_env.reset()
            state = observation
            acc_reward = 0

            for _ in range(episode_length):
                # Get the next action
                action = self.q_net.predict(state, deterministic=False)
                action = action.item()

                # Do a step on the environment
                observation, reward, terminated, truncated, info = curr_env.step(action)
                next_state = observation
                done = terminated or truncated
                acc_reward += reward

                # Update the exploration strategy with the given transition
                self.q_net.update_strategy(state, action, reward, next_state, done)

                # Update the replay buffer by pushing the given transition
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Update the state
                state = next_state

                if self.replay_buffer.filled(self.learning_starts):
                    if self.train_freq[1] == "steps":
                        if self.train_count.is_interval_reached():
                            self.train()
                        # Update count (for each step)
                        self.train_count.increment()

                if done:
                    # Case: Terminated State reached
                    break

            if self.replay_buffer.filled(self.learning_starts):
                # Case: Replay buffer is filled with at least learning_starts Transition
                if self.train_freq[1] == "episodes":
                    if self.train_count.is_interval_reached():
                        self.train()
                    # Update count (for each episode)
                    self.train_count.increment()

            # TODO: Use of logger instead of print-statements
            if self.render_count.is_interval_reached():
                print(f"Generation {generation}:")
                print(f"Reward: {acc_reward}")

            # Update render count
            self.render_count.increment()

            # Add accumulated rewards to the list of rewards for each episode
            rewards += [acc_reward]
        return rewards

    def evaluate(self, n_episodes: int, episode_length: int):
        """
        The main evaluation loop, where the agent is tested for N episodes with an episode length of L.
        So that means no backpropagation (upgrading weights with gradients) are performed here.

        Args:
            n_episodes (int):
                Number of episodes to train the agent.

            episode_length (int):
                The maximum length of each individual episode.

        Returns:
            list[float]:
                Accumulated rewards for each episode
        """
        # Reset parameters
        self._reset()
        rewards = []

        for generation in range(n_episodes):
            if self.render_count.is_interval_reached():
                curr_env = self.render_env
            else:
                curr_env = self.env
            observation, info = curr_env.reset()
            state = observation
            acc_reward = 0

            for _ in range(episode_length):
                # Get the next action
                action = self.q_net.predict(state, deterministic=True)
                action = action.item()

                # Do a step on the environment
                observation, reward, terminated, truncated, info = curr_env.step(action)
                next_state = observation
                done = terminated or truncated
                acc_reward += reward

                # Update the state
                state = next_state

                if done:
                    # Case: Terminated State reached
                    break

            # TODO: Use of logger instead of print-statements
            if self.render_count.is_interval_reached():
                print(f"Generation {generation}:")
                print(f"Reward: {acc_reward}")

            # Update render count
            self.render_count.increment()

            # Add accumulated rewards to the list of rewards for each episode
            rewards += [acc_reward]
        return rewards

    def __str__(self) -> str:
        header = f"{self.__class__.__name__}("
        env_line = f"env={self.env_type},"
        q_net_line = f"q_net={self.q_net},"
        target_q_net_line = f"target_q_net={self.target_q_net},"
        replay_buffer_line = f"replay_buffer={self.replay_buffer},"
        optimizer_line = f"optimizer={self.optimizer},"
        loss_fn_line = f"loss_fn={self.loss_fn},"
        learning_starts_line = f"learning_starts={self.learning_starts},"
        max_gradient_norm_line = f"max_gradient_norm={self.max_gradient_norm},"
        batch_size_line = f"batch_size={self.batch_size},"
        tau_line = f"tau={self.tau},"
        gamma_line = f"gamma={self.gamma},"
        target_freq_line = f"target_freq={self.target_freq},"
        train_freq_line = f"train_freq={self.train_freq},"
        render_freq_line = f"render_freq={self.render_freq},"
        gradient_steps_line = f"gradient_steps={self.gradient_steps},"
        end = ")"
        return "\n".join([header, env_line, q_net_line, target_q_net_line, replay_buffer_line, optimizer_line,
                          loss_fn_line, learning_starts_line, max_gradient_norm_line, batch_size_line, tau_line,
                          gamma_line, target_freq_line, train_freq_line, render_freq_line, gradient_steps_line, end])

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self) -> dict:
        return {
            "env_type": self.env_type,
            "q_net": self.q_net,
            "target_q_net": self.target_q_net,
            "replay_buffer": self.replay_buffer,
            "optimizer_type": self.optimizer_type,
            "optimizer_kwargs": self.optimizer_kwargs,
            "loss_fn": self.loss_fn,
            "loss_kwargs": self.loss_kwargs,
            "max_gradient_norm": self.max_gradient_norm,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "gamma": self.gamma,
            "target_freq": self.target_freq,
            "train_freq": self.train_freq,
            "render_freq": self.render_freq,
            "gradient_steps": self.gradient_steps,
            "target_count": self.target_count,
            "train_count": self.train_count,
            "render_count": self.render_count,
        }

    def __setstate__(self, state: dict):
        self.env_type = state["env_type"]
        self.env, self.render_env = DQN._get_env(self.env_type)
        self.q_net = state["q_net"]
        self.target_q_net = state["target_q_net"]
        self.target_q_net.freeze()
        self.replay_buffer = state["replay_buffer"]
        self.optimizer_type = state["optimizer_type"]
        self.optimizer_kwargs = state["optimizer_kwargs"]
        self.optimizer = DQN._get_optimizer_type(self.optimizer_type)(params=self.q_net.parameters(), **self.optimizer_kwargs)
        self.loss_fn = state["loss_fn"]
        self.loss_kwargs = state["loss_kwargs"]
        self.loss_fn = state["loss_fn"]
        self.loss_kwargs = state["loss_kwargs"]
        self.max_gradient_norm = state["max_gradient_norm"]
        self.learning_starts = state["learning_starts"]
        self.batch_size = state["batch_size"]
        self.tau = state["tau"]
        self.gamma = state["gamma"]
        self.target_freq = state["target_freq"]
        self.train_freq = state["train_freq"]
        self.render_freq = state["render_freq"]
        self.gradient_steps = state["gradient_steps"]
        self.target_count = state["target_count"]
        self.train_count = state["train_count"]
        self.render_count = state["render_count"]


class DDQN(DQN):

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
            # Double DQN learning rule
            Q_next = self.target_q_net.forward(next_states)
            a_next = self.q_net.forward(next_states).argmax(dim=1)
            q_targets = rewards + ~dones * self.gamma * Q_next.gather(dim=1, index=a_next.unsqueeze(1)).squeeze()

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        q_values = Q.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        return self.loss_fn(q_values, q_targets, **self.loss_kwargs)


class ClippedDDQN(DQN):

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
            # Clipped Double DQN
            Q_next1 = self.q_net.forward(next_states)
            Q_next2 = self.target_q_net.forward(next_states)
            a_next1 = Q_next1.argmax(dim=1)
            a_next2 = Q_next2.argmax(dim=1)
            q_targets1 = Q_next1.gather(dim=1, index=a_next1.unsqueeze(1)).squeeze()
            q_targets2 = Q_next2.gather(dim=1, index=a_next2.unsqueeze(1)).squeeze()
            q_targets = rewards + ~dones * self.gamma * torch.min(q_targets1, q_targets2)

        # Calculate the predicted q-values
        Q = self.q_net.forward(states)
        q_values = Q.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        return self.loss_fn(q_values, q_targets, **self.loss_kwargs)
