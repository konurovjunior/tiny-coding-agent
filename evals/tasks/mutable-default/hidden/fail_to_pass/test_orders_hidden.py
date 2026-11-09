import unittest
from orders import add_line


class OrdersHidden(unittest.TestCase):
    def test_first_order_is_not_changed_by_the_next(self):
        first = add_line("x", 1)
        add_line("y", 1)
        self.assertEqual(first, [("x", 1)])
