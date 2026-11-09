import unittest
from stats import mean, median


class StatsHidden(unittest.TestCase):
    def test_mean_of_nothing(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_two_values(self):
        self.assertEqual(median([1, 2]), 1.5)
        self.assertEqual(median([10, 0]), 5.0)
