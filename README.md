# py-rl-agent
Python/Pytorch Implementation of common RL algorithms

# PyRLAgent
PyRLAgent is a Python Framework for using common RL algorithms like DQN, DDQN, ClippedDDQN and many more.

### How to use PyRLAgent
In the following we want to solve the CartPole environment from Gymnasium.

We choose the ClippedDDQN algorithm to solve the CartPole environment:

```python
import torch.nn as nn

from PyRLAgent.algorithm.ddqn import ClippedDDQN

# For Cartpole
agent = ClippedDDQN(
    env_type="CartPole-v1",
    policy_type="q-net",
    policy_kwargs={"architecture": [64, 64], "activation_fn": nn.LeakyReLU(), "bias": True},
    strategy_type="exp-epsilon",
    strategy_kwargs={"epsilon_min": 0.1, "epsilon_max": 1.0, "decay_factor": 0.95},
    replay_buffer_type="ring",
    replay_buffer_kwargs={"max_size": 10000},
    optimizer_type="adam",
    optimizer_kwargs={"lr": 1e-3},
    lr_scheduler_type="linear-lr",
    lr_scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.5, "total_iters": 30000},
    loss_type="huber",
    loss_kwargs={},
    max_gradient_norm=10,
    learning_starts=64,
    batch_size=64,
    tau=5e-3,
    gamma=0.99,
    target_freq=4,
    train_freq=100,
    render_freq=50,
    gradient_steps=4,
)

# With the `.fit()` method we can train our agent on the given environment:
train_rewards = agent.fit(n_timesteps=1e5)

# With `.eval()` method we can evaluate our agent on the given environment:
# For Cartpole
test_rewards = agent.eval(n_timesteps=1e4)
```

### Further Resources
- DQN Paper: https://arxiv.org/abs/1312.5602
- DDQN Paper: https://arxiv.org/abs/1509.06461
- Clipped DDQN Paper: https://arxiv.org/abs/1802.09477
- C51 Paper: https://arxiv.org/abs/1707.06887
