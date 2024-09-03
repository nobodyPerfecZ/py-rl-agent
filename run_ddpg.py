import torch.nn as nn

from pyrlagent.torch.algorithm import DDPG
from pyrlagent.torch.config import (
    EnvConfig,
    LRSchedulerConfig,
    NetworkConfig,
    OptimizerConfig,
    RLTrainConfig,
)


if __name__ == "__main__":
    agent = DDPG(
        train_config=RLTrainConfig(
            env_config=EnvConfig(
                id="Ant-v4",
                kwargs={},
            ),
            network_config=NetworkConfig(
                id="ddpg-mlp-continuous",
                kwargs={
                    "hidden_features": [256, 256, 256, 256],
                    "activation": nn.Tanh,
                    "noise_scale": 0.1,
                    "low_action": -1.0,
                    "high_action": 1.0,
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
        steps_per_update=32,
        max_size=1e5,
        gamma=0.99,
        polyak=0.95,
        vf_coef=2.0,
        update_steps=4,
        device="auto",
    )
    # Train the agent
    agent.fit(num_timesteps=1e5)

    # Create a checkpoint of the agent
    train_state = agent.state_dict()

    # Evaluate the agent
    agent.eval(num_timesteps=5e3, train_state=train_state)
