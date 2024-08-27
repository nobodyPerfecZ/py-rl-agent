from pyrlagent.torch.network import actor
from pyrlagent.torch.network import critic
from pyrlagent.torch.network import actor_critic

# Actor networks
AbstractActorNetwork = actor.AbstractActorNetwork
MLPCategoricalActorNetwork = actor.MLPCategoricalActorNetwork
CNNCategoricalActorNetwork = actor.CNNCategoricalActorNetwork
MLPGaussianActorNetwork = actor.MLPGaussianActorNetwork
CNNGaussianActorNetwork = actor.CNNGaussianActorNetwork

# Critic networks
AbstractCriticNetwork = critic.AbstractCriticNetwork
MLPCriticNetwork = critic.MLPCriticNetwork
CNNCriticNetwork = critic.CNNCriticNetwork

# Actor Critic networks
AbstractActorCriticNetwork = actor_critic.AbstractActorCriticNetwork
MLPCategoricalActorCriticNetwork = actor_critic.MLPCategoricalActorCriticNetwork
CNNCategoricalActorCriticNetwork = actor_critic.CNNCategoricalActorCriticNetwork
MLPGaussianActorCriticNetwork = actor_critic.MLPGaussianActorCriticNetwork
CNNGaussianActorCriticNetwork = actor_critic.CNNGaussianActorCriticNetwork

__all__ = [
    "AbstractActorNetwork",
    "MLPCategoricalActorNetwork",
    "CNNCategoricalActorNetwork",
    "MLPGaussianActorNetwork",
    "CNNGaussianActorNetwork",
    "AbstractCriticNetwork",
    "MLPCriticNetwork",
    "CNNCriticNetwork",
    "AbstractActorCriticNetwork",
    "MLPCategoricalActorCriticNetwork",
    "CNNCategoricalActorCriticNetwork",
    "MLPGaussianActorCriticNetwork",
    "CNNGaussianActorCriticNetwork",
]
