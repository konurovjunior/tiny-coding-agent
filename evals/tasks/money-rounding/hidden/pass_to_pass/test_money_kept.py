import unittest
from cart import cart_total
from money import format_tenge, to_tiyn


class MoneyKept(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_tenge(5), "0.05 ₸")

    def test_whole_tenge(self):
        self.assertEqual(to_tiyn(2), 200)

    def test_empty_cart(self):
        self.assertEqual(cart_total([]), 0)
