import unittest
from search import find


class FindHidden(unittest.TestCase):
    def test_larger_list(self):
        self.assertEqual(find([2, 4, 6, 8, 10], 10), 4)

    def test_absent_above_the_end(self):
        self.assertEqual(find([1, 3], 4), -1)
