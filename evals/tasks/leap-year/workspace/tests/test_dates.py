import unittest
from dates import days_in_february, is_leap


class Dates(unittest.TestCase):
    def test_ordinary_leap_year(self):
        self.assertTrue(is_leap(2024))

    def test_century_is_not_leap(self):
        self.assertFalse(is_leap(1900))

    def test_february(self):
        self.assertEqual(days_in_february(2023), 28)
