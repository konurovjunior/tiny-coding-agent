import unittest
from search import find


class FindKept(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(find([], 1), -1)

    def test_first_and_only(self):
        self.assertEqual(find([1, 3, 5], 1), 0)
        self.assertEqual(find([5], 5), 0)
