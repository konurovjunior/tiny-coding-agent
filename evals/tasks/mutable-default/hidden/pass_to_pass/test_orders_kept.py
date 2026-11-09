import unittest
from orders import add_line


class OrdersKept(unittest.TestCase):
    def test_appends_to_the_given_order(self):  # a fix that always copies breaks this one
        existing = [("a", 1)]
        result = add_line("b", 2, existing)
        self.assertIs(result, existing)
        self.assertEqual(existing, [("a", 1), ("b", 2)])
