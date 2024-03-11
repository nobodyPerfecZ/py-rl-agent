import torch.nn as nn

from PyRLAgent.algorithm.ddqn import ClippedDDQN

if __name__ == "__main__":
    # For CartPole
    dqn = ClippedDDQN(
        env_type="CartPole-v1",
        policy_type="q-dueling-net",
        policy_kwargs={
            "Q_min": -10,
            "Q_max": 10,
            "feature_architecture": [64],
            "feature_activation_fn": nn.Tanh(),
            "value_architecture": [64],
            "value_activation_fn": nn.Tanh(),
            "advantage_architecture": [64, 64],
            "advantage_activation_fn": nn.Tanh(),
            "bias": True
        },
        strategy_type="exp-epsilon",
        strategy_kwargs={"epsilon_min": 0.01, "epsilon_max": 1.0, "decay_factor": 0.995},
        replay_buffer_type="ring",
        replay_buffer_kwargs={"max_size": 15000},
        optimizer_type="adam",
        optimizer_kwargs={"lr": 1e-4},
        lr_scheduler_type="linear-lr",
        lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.5, "total_iters": 100000},
        loss_type="huber",
        loss_kwargs={},
        max_gradient_norm=20,
        learning_starts=64,
        batch_size=64,
        tau=5e-3,
        gamma=0.99,
        target_freq=5,
        train_freq=20,
        render_freq=50,
        gradient_steps=4,
    )
    train_returns = dqn.fit(n_timesteps=1e3)
    eval_returns = dqn.eval(n_timesteps=1e4)
    print(train_returns)
    print(eval_returns)
