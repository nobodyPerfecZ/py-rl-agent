import matplotlib.pyplot as plt
import torch.nn as nn

from PyRLAgent.algorithm.ddqn import DQN

if __name__ == "__main__":
    agent = DQN(
        env_type="CartPole-v1",
        env_wrappers="none",
        policy_type="q-dueling-net",
        policy_kwargs={
            "feature_architecture": [256],
            "feature_activation_fn": None,
            "feature_output_activation_fn": nn.Tanh(),
            "value_architecture": [256],
            "value_activation_fn": nn.Tanh(),
            "value_output_activation_fn": None,
            "advantage_architecture": [256],
            "advantage_activation_fn": nn.Tanh(),
            "advantage_output_activation_fn": None,
            "bias": True
        },
        strategy_type="exp-epsilon",
        strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "decay_factor": 0.99},
        replay_buffer_type="ring-buffer",
        replay_buffer_kwargs={"max_size": 200000},
        optimizer_type="adam",
        optimizer_kwargs={"lr": 2.5e-4},
        lr_scheduler_type="none",
        lr_scheduler_kwargs={},
        loss_type="huber",
        loss_kwargs={},
        max_gradient_norm=0.5,
        num_envs=8,
        steps_per_trajectory=64,
        tau=5e-3,
        gamma=0.99,
        target_freq=4,
        gradient_steps=4,
    )
    # Train the agent
    train_returns = agent.fit(n_timesteps=5e5)
    plt.plot(train_returns)
    plt.title("Training - CartPole v1")
    plt.ylabel("Returns")
    plt.xlabel("Episodes")
    plt.show()

    # Evaluate the agent
    eval_returns = agent.eval(n_timesteps=5e3)
    plt.plot(eval_returns)
    plt.title("Evaluation - CartPole v1")
    plt.ylabel("Returns")
    plt.xlabel("Episodes")
    plt.show()
