import unittest
from basket import describe


class Describe(unittest.TestCase):
    def test_describe_unchanged(self):
        self.assertEqual(describe(7, 3), "3 x 7")
