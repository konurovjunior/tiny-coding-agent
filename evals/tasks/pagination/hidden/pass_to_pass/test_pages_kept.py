import unittest
from pages import page, page_count


class PagesKept(unittest.TestCase):
    def test_past_the_end_is_empty(self):
        self.assertEqual(page(list(range(10)), 5, 3), [])

    def test_page_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            page([1, 2], 0, 3)

    def test_page_count(self):
        self.assertEqual(page_count(10, 3), 4)
