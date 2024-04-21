import unittest

from PyRLAgent.common.strategy.epsilon_greedy import LinearDecayEpsilonGreedy, ExponentialDecayEpsilonGreedy
from PyRLAgent.common.strategy.greedy import Greedy
from PyRLAgent.common.strategy.random import Random
from PyRLAgent.common.strategy.ucb import UCB
from PyRLAgent.enum.strategy import StrategyEnum


class TestStrategyEnum(unittest.TestCase):
    """
    Tests the enum class Strategy.
    """

    def setUp(self):
        self.wrapper = {
            StrategyEnum.RANDOM: Random,
            StrategyEnum.GREEDY: Greedy,
            StrategyEnum.LINEAR_EPSILON: LinearDecayEpsilonGreedy,
            StrategyEnum.EXP_EPSILON: ExponentialDecayEpsilonGreedy,
            StrategyEnum.UCB: UCB,
        }

        self.strategy_kwargs1 = {}
        self.strategy_kwargs2 = {}
        self.strategy_kwargs3 = {"epsilon_min": 0.0, "epsilon_max": 1.0, "steps": 25000}
        self.strategy_kwargs4 = {"epsilon_min": 0.0, "epsilon_max": 1.0, "decay_factor": 0.9}
        self.strategy_kwargs5 = {"c": 0.4}

    def test_wrapper(self):
        """
        Tests the method test_wrapper().
        """
        self.assertDictEqual(self.wrapper, StrategyEnum.wrapper())

    def test_to(self):
        """
        Tests the method to().
        """
        strategy1 = StrategyEnum.RANDOM.to(**self.strategy_kwargs1)
        strategy2 = StrategyEnum.GREEDY.to(**self.strategy_kwargs2)
        strategy3 = StrategyEnum.LINEAR_EPSILON.to(**self.strategy_kwargs3)
        strategy4 = StrategyEnum.EXP_EPSILON.to(**self.strategy_kwargs4)
        strategy5 = StrategyEnum.UCB.to(**self.strategy_kwargs5)

        self.assertIsInstance(strategy1, Random)
        self.assertIsInstance(strategy2, Greedy)
        self.assertIsInstance(strategy3, LinearDecayEpsilonGreedy)
        self.assertIsInstance(strategy4, ExponentialDecayEpsilonGreedy)
        self.assertIsInstance(strategy5, UCB)


if __name__ == '__main__':
    unittest.main()
