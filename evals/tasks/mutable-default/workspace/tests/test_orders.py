import unittest
from orders import add_line


class Orders(unittest.TestCase):
    def test_new_orders_are_separate(self):
        add_line("apple", 1)
        second = add_line("pear", 2)
        self.assertEqual(second, [("pear", 2)])
