import matplotlib.pyplot as plt
import torch.nn as nn

from PyRLAgent.algorithm.ppo import PPO

if __name__ == "__main__":
    agent = PPO(
        env_type="CartPole-v1",
        env_wrappers="none",
        policy_type="actor-critic-net",
        policy_kwargs={
            "actor_architecture": [256, 256],
            "actor_activation_fn": nn.Tanh(),
            "actor_output_activation_fn": None,
            "critic_architecture": [256, 256],
            "critic_activation_fn": nn.Tanh(),
            "critic_output_activation_fn": None,
            "bias": True
        },
        optimizer_type="adam",
        optimizer_kwargs={"lr": 2.5e-4},
        lr_scheduler_type="none",
        lr_scheduler_kwargs={},
        max_gradient_norm=0.5,
        num_envs=32,
        steps_per_trajectory=64,
        clip_ratio=0.1,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=None,
        vf_coef=0.5,
        ent_coef=0.01,
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
