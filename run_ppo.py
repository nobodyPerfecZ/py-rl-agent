import torch.nn as nn

from pyrlagent.torch.algorithm import PPO
from pyrlagent.torch.config.env import EnvConfig
from pyrlagent.torch.config.lr_scheduler import LRSchedulerConfig
from pyrlagent.torch.config.network import NetworkConfig
from pyrlagent.torch.config.optimizer import OptimizerConfig
from pyrlagent.torch.config.train import RLTrainConfig, RLTrainState

if __name__ == "__main__":
    agent = PPO(
        train_config=RLTrainConfig(
            env_config=EnvConfig(
                env_type="LunarLander-v3",
                env_kwargs={},
            ),
            network_config=NetworkConfig(
                network_type="mlp-discrete",
                network_kwargs={
                    "hidden_features": [256, 256, 256, 256],
                    "activation": nn.Tanh,
                },
            ),
            optimizer_config=OptimizerConfig(
                optimizer_type="adam",
                optimizer_kwargs={"lr": 2.5e-4},
            ),
            lr_scheduler_config=LRSchedulerConfig(
                lr_scheduler_type="exponential",
                lr_scheduler_kwargs={"gamma": 0.99},
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
        device="cpu",
    )
    # Train the agent
    agent.fit(num_timesteps=1e5)

    # Create a checkpoint of the agent
    train_state = RLTrainState(
        network_state=agent.network.state_dict(),
        optimizer_state=agent.optimizer.state_dict(),
        lr_scheduler_state=agent.lr_scheduler.state_dict(),
    )

    # Evaluate the agent
    agent.eval(num_timesteps=5e3, train_state=train_state)
