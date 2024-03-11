import torch.nn as nn

from PyRLAgent.algorithm.ddqn import ClippedDDQN

if __name__ == "__main__":
    # For CartPole
    dqn = ClippedDDQN(
        env_type="CartPole-v1",
        policy_type="q-net",
        policy_kwargs={
            "Q_min": -1,
            "Q_max": 1,
            "architecture": [128, 128],
            "activation_fn": nn.LeakyReLU(),
            "bias": True
        },
        strategy_type="exp-epsilon",
        strategy_kwargs={"epsilon_min": 0.01, "epsilon_max": 1.0, "decay_factor": 0.95},
        replay_buffer_type="ring",
        replay_buffer_kwargs={"max_size": 2000},
        optimizer_type="adam",
        optimizer_kwargs={"lr": 1e-3},
        lr_scheduler_type="linear-lr",
        lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.5, "total_iters": 10000},
        loss_type="huber",
        loss_kwargs={},
        max_gradient_norm=10,
        learning_starts=64,
        batch_size=64,
        tau=1e-2,
        gamma=0.99,
        target_freq=10,
        train_freq=10,
        render_freq=50,
        gradient_steps=4,
    )
    train_returns = dqn.fit(n_timesteps=1e4)
    eval_returns = dqn.eval(n_timesteps=5e3)
    print(train_returns)
    print(eval_returns)
