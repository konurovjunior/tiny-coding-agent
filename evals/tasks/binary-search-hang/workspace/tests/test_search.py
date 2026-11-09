import unittest
from search import find


class Find(unittest.TestCase):
    def test_last_item(self):
        self.assertEqual(find([1, 3, 5, 7], 7), 3)

    def test_first_item(self):
        self.assertEqual(find([1, 3, 5], 1), 0)
