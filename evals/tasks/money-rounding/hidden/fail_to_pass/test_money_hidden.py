import unittest
from cart import cart_total
from money import to_tiyn


class MoneyHidden(unittest.TestCase):
    def test_conversion_itself_is_exact(self):  # a fix in cart.py alone leaves these wrong
        self.assertEqual(to_tiyn(0.29), 29)
        self.assertEqual(to_tiyn(1.15), 115)

    def test_two_lines(self):
        self.assertEqual(cart_total([0.29, 0.29]), 58)
