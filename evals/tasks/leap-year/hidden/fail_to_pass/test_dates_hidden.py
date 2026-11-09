import unittest
from dates import days_in_february, is_leap


class DatesHidden(unittest.TestCase):
    def test_other_centuries(self):
        self.assertFalse(is_leap(2100))
        self.assertFalse(is_leap(1800))

    def test_february_in_a_century(self):
        self.assertEqual(days_in_february(1900), 28)
