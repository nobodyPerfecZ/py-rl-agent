import torch.nn as nn

from pyrlagent.torch.algorithm import PPO
from pyrlagent.torch.config import (
    EnvConfig,
    LRSchedulerConfig,
    NetworkConfig,
    OptimizerConfig,
    RLTrainConfig,
)


if __name__ == "__main__":
    agent = PPO(
        train_config=RLTrainConfig(
            env_config=EnvConfig(
                id="LunarLander-v3",
                kwargs={},
            ),
            network_config=NetworkConfig(
                id="mlp-discrete",
                kwargs={
                    "hidden_features": [256, 256, 256, 256],
                    "activation": nn.Tanh,
                },
            ),
            optimizer_config=OptimizerConfig(
                id="adam",
                kwargs={"lr": 2.5e-4},
            ),
            lr_scheduler_config=LRSchedulerConfig(
                id="exponential",
                kwargs={"gamma": 0.99},
            ),
        ),
        max_gradient_norm=1.0,
        num_envs=128,
        steps_per_trajectory=32,
        clip_ratio=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=2.0,
        ent_coef=0.0,
        update_steps=4,
        device="auto",
    )
    # Train the agent
    agent.fit(num_timesteps=3e6)

    # Create a checkpoint of the agent
    train_state = agent.state_dict()

    # Evaluate the agent
    agent.eval(num_timesteps=5e3, train_state=train_state)
