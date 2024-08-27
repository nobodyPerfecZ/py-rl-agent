from pyrlagent.torch.experience import metric
from pyrlagent.torch.experience import trajectory

non_discounted_return = metric.non_discounted_return
discounted_return = metric.discounted_return
gae = metric.gae

Trajectory = trajectory.Trajectory

__all__ = [
    "non_discounted_return",
    "discounted_return",
    "gae",
    "Trajectory",
]
