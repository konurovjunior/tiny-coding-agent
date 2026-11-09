import unittest
from text_stats import word_count


class WordCount(unittest.TestCase):
    def test_two_spaces(self):
        self.assertEqual(word_count("hello  world"), 2)

    def test_empty(self):
        self.assertEqual(word_count(""), 0)
