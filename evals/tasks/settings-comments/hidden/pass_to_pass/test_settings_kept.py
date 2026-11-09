import unittest
from settings import parse


class SettingsKept(unittest.TestCase):
    def test_comment_after_a_value(self):  # dropping comment handling altogether breaks this one
        self.assertEqual(parse("a = 1 # one"), {"a": "1"})

    def test_indented_comment_line(self):
        self.assertEqual(parse("   # indented comment\nb=2"), {"b": "2"})
