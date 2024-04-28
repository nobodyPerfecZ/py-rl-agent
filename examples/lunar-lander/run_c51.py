from matplotlib import pyplot as plt
from torch import nn

from PyRLAgent.algorithm.c51 import C51

if __name__ == "__main__":
    agent = C51(
        env_type="LunarLander-v2",
        env_wrappers="none",
        policy_type="q-prob-net",
        policy_kwargs={
            "Q_min": -1,
            "Q_max": 1,
            "num_atoms": 51,
            "architecture": [256, 256],
            "activation_fn": nn.Tanh(),
            "output_activation_fn": None,
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
