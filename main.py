import matplotlib.pyplot as plt
import torch.nn as nn

from PyRLAgent.algorithm.ddqn import ClippedDDQN
from PyRLAgent.algorithm.ppo import PPO

if __name__ == "__main__":
    # For CartPole
    ppo = PPO(
        env_type="CartPole-v1",
        env_wrappers=None,
        policy_type="actor-critic",
        policy_kwargs={
            "actor_architecture": [128],
            "actor_activation_fn": nn.Tanh(),
            "actor_output_activation_fn": None,
            "critic_architecture": [128],
            "critic_activation_fn": nn.Tanh(),
            "critic_output_activation_fn": None,
            "bias": True
        },
        optimizer_type="adamw",
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_type="linear-lr",
        lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.8, "total_iters": 100000},
        batch_size=32,
        steps_per_trajectory=16,
        clip_ratio=0.2,
        gamma=0.98,
        gae_lambda=0.8,
        target_kl=0.01,
        vf_coef=0.5,
        ent_coef=0.0,
        render_freq=50,
        gradient_steps=16,
    )
    # Train the agent
    train_returns = ppo.fit(n_timesteps=1e5)
    plt.plot(train_returns)
    plt.title("Training - CartPole v1")
    plt.ylabel("Returns")
    plt.xlabel("Episodes")
    plt.show()

    # Evaluate the agent
    eval_returns = ppo.eval(n_timesteps=1e4)
    plt.plot(eval_returns)
    plt.title("Evaluation - CartPole v1")
    plt.ylabel("Returns")
    plt.xlabel("Episodes")
    plt.show()

    # For CartPole
    clipped_ddqn = ClippedDDQN(
        env_type="CartPole-v1",
        env_wrappers=None,
        policy_type="q-dueling-net",
        policy_kwargs={
            "feature_architecture": [64],
            "feature_activation_fn": None,
            "feature_output_activation_fn": nn.Tanh(),
            "value_architecture": [64],
            "value_activation_fn": nn.Tanh(),
            "value_output_activation_fn": None,
            "advantage_architecture": [64, 64],
            "advantage_activation_fn": nn.Tanh(),
            "advantage_output_activation_fn": None,
            "bias": True
        },
        strategy_type="exp-epsilon",
        strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "decay_factor": 0.95},
        replay_buffer_type="ring",
        replay_buffer_kwargs={"max_size": 15000},
        optimizer_type="adam",
        optimizer_kwargs={"lr": 1e-4},
        lr_scheduler_type="linear-lr",
        lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.5, "total_iters": 100000},
        loss_type="huber",
        loss_kwargs={},
        max_gradient_norm=20,
        batch_size=64,
        tau=5e-3,
        gamma=0.99,
        target_freq=5,
        train_freq=20,
        render_freq=50,
        gradient_steps=4,
    )
    # Train the agent
    train_returns = clipped_ddqn.fit(n_timesteps=1e4)
    plt.plot(train_returns)
    plt.title("Training - CartPole v1")
    plt.ylabel("Returns")
    plt.xlabel("Episodes")
    plt.show()

    # Evaluate the agent
    eval_returns = clipped_ddqn.eval(n_timesteps=1e4)
    plt.plot(eval_returns)
    plt.title("Evaluation - CartPole v1")
    plt.ylabel("Returns")
    plt.xlabel("Episodes")
    plt.show()
