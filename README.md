# py-rl-agent
Python/Pytorch Implementation of common RL algorithms

# PyRLAgent
PyRLAgent is a Python Framework for using common RL algorithms like DQN, DDQN, ClippedDDQN and many more.

### How to use PyRLAgent
In the following we want to solve the CartPole environment from Gymnasium.

We choose the ClippedDDQN algorithm to solve the CartPole environment:
```python
from PyRLAgent.dqn.dqn import ClippedDDQN
import torch.nn as nn

# For Cartpole
agent = ClippedDDQN(
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
```

With the `.fit()` method we can train our agent on the given environment:
```python
# For Cartpole
train_rewards = agent.fit(n_episodes=100, episode_length=500)
```

With `.evaluate()` method we can evaluate our agent on the given environment:
```python
# For Cartpole
test_rewards = agent.evaluate(n_episodes=100, episode_length=500)
```

### Further Resources
- DQN Paper: https://arxiv.org/abs/1312.5602
- DDQN Paper: https://arxiv.org/abs/1509.06461
- Clipped DDQN Paper: https://arxiv.org/abs/1802.09477
