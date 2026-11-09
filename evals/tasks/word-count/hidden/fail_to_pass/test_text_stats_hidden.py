import unittest
from text_stats import word_count


class WordCountHidden(unittest.TestCase):
    def test_tabs_and_newlines_separate_words(self):  # filtering out empty strings passes the visible tests, not this
        self.assertEqual(word_count("a\nb\tc"), 3)

    def test_only_spaces(self):
        self.assertEqual(word_count("   "), 0)

    def test_leading_and_trailing_spaces(self):
        self.assertEqual(word_count(" lead and trail "), 3)
