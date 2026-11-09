import unittest
from settings import parse


class Settings(unittest.TestCase):
    def test_value_with_a_hash(self):
        self.assertEqual(parse("color = #ff8800"), {"color": "#ff8800"})

    def test_comments_and_blank_lines(self):
        self.assertEqual(parse("a = 1\n# note\n\nb = 2"), {"a": "1", "b": "2"})
