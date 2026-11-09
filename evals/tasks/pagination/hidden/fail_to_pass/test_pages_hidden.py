import unittest
from pages import page


class PagesHidden(unittest.TestCase):
    def test_last_partial_page(self):
        self.assertEqual(page(list(range(10)), 4, 3), [9])

    def test_one_page_holds_everything(self):
        self.assertEqual(page(list("abc"), 1, 5), ["a", "b", "c"])
