import unittest
from stats import mean, median


class StatsKept(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_single_value(self):
        self.assertEqual(median([5]), 5)

    def test_empty_median(self):
        with self.assertRaises(ValueError):
            median([])
