import unittest
from stats import mean, median


class Stats(unittest.TestCase):
    def test_mean_of_nothing(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_even_median(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_odd_median(self):
        self.assertEqual(median([3, 1, 2]), 2)
