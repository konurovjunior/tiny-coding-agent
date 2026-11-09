import unittest
from dates import days_in_february, is_leap


class DatesKept(unittest.TestCase):
    def test_every_400th_year_stays_leap(self):  # the half-fix "not a century" breaks this one
        self.assertTrue(is_leap(2000))
        self.assertTrue(is_leap(1600))
        self.assertEqual(days_in_february(2000), 29)

    def test_ordinary_years(self):
        self.assertFalse(is_leap(2023))
        self.assertTrue(is_leap(2024))
