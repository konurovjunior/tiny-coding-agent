import unittest
from text_stats import longest_word, word_count


class TextStatsKept(unittest.TestCase):
    def test_longest_word(self):
        self.assertEqual(longest_word("a bbb cc"), "bbb")

    def test_one_word(self):
        self.assertEqual(word_count("one"), 1)
