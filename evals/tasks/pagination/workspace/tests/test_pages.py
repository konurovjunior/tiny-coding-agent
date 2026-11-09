import unittest
from pages import page


class Pages(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(page(list(range(10)), 1, 3), [0, 1, 2])

    def test_second_page(self):
        self.assertEqual(page(list(range(10)), 2, 3), [3, 4, 5])
