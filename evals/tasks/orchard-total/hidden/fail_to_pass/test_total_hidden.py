import unittest
from basket import total


class TotalHidden(unittest.TestCase):
    def test_other_values(self):
        self.assertEqual(total(2, 5), 10)
        self.assertEqual(total(0, 9), 0)
        self.assertEqual(total(1.5, 2), 3.0)
