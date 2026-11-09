import unittest
from cart import cart_total, receipt


class Cart(unittest.TestCase):
    def test_one_price(self):
        self.assertEqual(cart_total([19.99]), 1999)

    def test_receipt(self):
        self.assertEqual(receipt([19.99, 0.01]), "20.00 ₸")
