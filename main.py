from PyRLAgent.dqn.dqn import DQN, DDQN, ClippedDDQN
import torch.nn as nn
import yaml

if __name__ == "__main__":
    # For Cartpole
    dqn_agent = ClippedDDQN(
        env_type="CartPole-v1",
        policy_type="q-net",
        policy_kwargs={"architecture": [128], "activation_fn": nn.Tanh(), "bias": True},
        strategy_type="linear-epsilon",
        strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 1000},
        replay_buffer_type="ring",
        replay_buffer_kwargs={"max_size": 10000},
        optimizer_type="adam",
        optimizer_kwargs={"lr": 5e-4},
        loss_type="huber",
        loss_kwargs={},
        max_gradient_norm=100,
        learning_starts=64,
        batch_size=64,
        tau=5e-3,
        gamma=0.99,
        target_freq=1,
        train_freq=(1, "steps"),
        render_freq=50,
        gradient_steps=1,
    )
    # print(dqn_agent)

    dqn_agent.fit(n_episodes=100, episode_length=500)

    with open("test_data.yaml", "w") as yaml_file:
        yaml.dump(dqn_agent, yaml_file)

    with open("test_data.yaml", "r") as yaml_file:
        dqn_agent = yaml.load(yaml_file, Loader=yaml.Loader)

    dqn_agent.fit(n_episodes=200, episode_length=500)
    """
    # For Lunar Lander
    dqn_agent = ClippedDDQN(
        env="LunarLander-v2",
        policy_type="q-net",
        policy_kwargs={"architecture": [64, 64, 64], "activation_fn": nn.Tanh(), "bias": True},
        strategy_type="linear-epsilon",
        strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "steps": 30000},
        replay_buffer_type="ring",
        replay_buffer_kwargs={"max_size": 50000},
        optimizer_type="adam",
        optimizer_kwargs={"lr": 1e-4},
        loss_type="huber",
        loss_kwargs={},
        max_gradient_norm=10,
        learning_starts=64,
        batch_size=64,
        tau=5e-3,
        gamma=0.99,
        target_freq=250,
        train_freq=(4, "steps"),
        render_freq=50,
        gradient_steps=1,
    )
    dqn_agent.fit(n_episodes=1000, episode_length=1000)
    """
