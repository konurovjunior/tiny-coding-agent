"""Lesson 14/26: every provoked failure mode ends caught with guards, and the guards themselves behave.
Uses failure_modes.py's scenarios (scripted model, real loop, no API calls).

Run (from this folder):  python -m unittest test_failure_modes -v
"""

import unittest

from failure_modes import run, scenarios
from guards import Guards


class FailureModes(unittest.TestCase):
    def test_every_mode_is_caught_with_guards(self):
        for s in scenarios():
            with self.subTest(mode=s["mode"]):
                r = run(s, guarded=True)
                self.assertEqual(r["verdict"], "caught", r)
                self.assertFalse(r["tests_edited"])

    def test_without_guards_the_known_gaps_are_still_there(self):
        # documents why the guards exist: if one of these starts passing, the loop changed and the report is stale
        by_mode = {s["mode"]: run(s, guarded=False) for s in scenarios()}
        self.assertEqual(by_mode["scope creep"]["verdict"], "through")  # "completed" by editing the test
        self.assertEqual(by_mode["cascading errors"]["state"], "budget_exhausted")
        self.assertEqual(by_mode["context loss"]["notes"], 0)


class GuardRules(unittest.TestCase):
    def setUp(self):
        self.g = Guards()

    def test_patches_under_tests_are_refused_however_the_path_is_written(self):
        for path in ("tests/test_calc.py", "tests\\test_calc.py", "tests/sub/x.py"):
            self.assertEqual(self.g.before({"tool": "patch", "path": path}, [])[0], "refuse", path)
        self.assertIsNone(self.g.before({"tool": "patch", "path": "calc.py"}, []))
        self.assertIsNone(self.g.before({"tool": "patch", "path": "tests_helpers.py"}, []))  # not the folder

    def test_only_a_repeat_of_a_refused_action_stops(self):
        refused = {"tool": "patch", "path": "calc.py", "old": "x", "new": "y", "result": {"error": "found 0 times"}}
        worked = {"tool": "test", "result": {"passed": False}}
        self.assertEqual(self.g.before({"tool": "patch", "path": "calc.py", "old": "x", "new": "y"}, [refused])[0], "stop")
        self.assertIsNone(self.g.before({"tool": "patch", "path": "calc.py", "old": "x", "new": "z"}, [refused]))
        self.assertIsNone(self.g.before({"tool": "test"}, [worked]))  # testing again after a failure is normal

    def test_a_re_read_is_noted_unless_the_file_changed_since(self):
        read = {"tool": "read", "path": "calc.py", "result": {"text": "a"}}
        patched = {"tool": "patch", "path": "calc.py", "result": {"applied": True}}
        again = self.g.after({"tool": "read", "path": "calc.py"}, {"text": "a"}, [read])
        self.assertIn("step 1", again["note"])
        fresh = self.g.after({"tool": "read", "path": "calc.py"}, {"text": "b"}, [read, patched])
        self.assertNotIn("note", fresh)


if __name__ == "__main__":
    unittest.main()
