import unittest
from settings import parse


class SettingsHidden(unittest.TestCase):
    def test_url_fragment(self):
        self.assertEqual(parse("url = http://x/#top"), {"url": "http://x/#top"})

    def test_hash_value_then_comment(self):
        self.assertEqual(parse("tag=#1 # first"), {"tag": "#1"})
