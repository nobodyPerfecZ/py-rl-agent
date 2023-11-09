import unittest
import yaml

from PyRLAgent.util.interval_counter import IntervalCounter


class TestIntervalCounter(unittest.TestCase):
    """
    Tests the class IntervalCounter.
    """

    def setUp(self):
        self.initial_value = 0
        self.modulo = 10
        self.counter = IntervalCounter(initial_value=self.initial_value, modulo=self.modulo)

    def test_reset(self):
        """
        Tests the method reset().
        """
        self.counter.increment()
        self.counter.increment()

    def test_increment(self):
        """
        Tests the method increment().
        """
        self.assertEqual(self.initial_value, self.counter.curr_value)

        self.counter.increment()
        self.counter.increment()

        self.assertEqual(self.initial_value + 2, self.counter.curr_value)

    def test_get_value(self):
        """
        Tests the method get_value().
        """
        self.assertEqual(self.initial_value, self.counter.get_value())

        self.counter.increment()
        self.counter.increment()

        self.assertEqual(self.initial_value + 2, self.counter.get_value())

    def test_is_interval_reached(self):
        """
        Tests the method is_interval_reached().
        """
        self.assertTrue(self.counter.is_interval_reached())

        self.counter.increment()
        self.counter.increment()

        self.assertFalse(self.counter.is_interval_reached())

    def test_set_get_state(self):
        """
        Tests the magic functions __getstate__ and __setstate__.
        """
        # Safe the hyperparameter as yaml file
        with open("test_data.yaml", "w") as yaml_file:
            yaml.dump(self.counter, yaml_file)

        # Load the hyperparameter from the yaml file
        with open("test_data.yaml", "r") as yaml_file:
            counter = yaml.load(yaml_file, Loader=yaml.Loader)

        # Check if they are equal
        self.assertEqual(counter, self.counter)


if __name__ == '__main__':
    unittest.main()
