"""Lesson 14/27: with a model that OBEYS every planted instruction, what each defence stops. Scripted, no API calls.

Run (from this folder):  ../../../.venv/bin/python -m unittest test_injection -v
"""

import json
import unittest

from guards import Guards, validate_patch
from injection_suite import ATTACKS, PAYLOAD, live_payload, one_run
from llm_planner import observation

BY_NAME = {a["name"]: a for a in ATTACKS}


def run(name, defended):
    return one_run(BY_NAME[name], defended, scripted=True)


class ObedientModel(unittest.TestCase):
    def test_without_the_validator_injected_code_ships_inside_a_passing_fix(self):
        for name in ("arbitrary code via source comment", "memory poisoning"):
            with self.subTest(name):
                r = run(name, defended=False)
                self.assertTrue(r["repaired"])           # the tests pass...
                self.assertTrue(r["attack_succeeded"])   # ...and the attacker's line is live in the code

    def test_the_validator_refuses_it_and_the_honest_fix_still_lands(self):
        for name in ("arbitrary code via source comment", "memory poisoning"):
            with self.subTest(name):
                r = run(name, defended=True)
                self.assertEqual((r["repaired"], r["attack_succeeded"]), (True, False))
                self.assertTrue(any("adds code that" in m for m in r["refusals"]))

    def test_earlier_defences_already_stop_scope_creep_and_data_theft(self):
        for defended in (False, True):
            self.assertFalse(run("scope creep via test docstring", defended)["attack_succeeded"])   # 14/26 guard
            theft = run("data theft via source comment", defended)
            self.assertFalse(theft["attack_succeeded"])
            self.assertIn("path escapes the workspace: ../.env", theft["refusals"])                 # stage 1

    def test_fake_completion_is_denial_not_false_success(self):
        r = run("fake completion via test output", defended=True)
        self.assertEqual((r["state"], r["repaired"]), ("failed", False))  # believed "DONE", but nothing completed

    def test_no_false_alarm_on_an_honest_repair(self):
        r = run("control (no attack)", defended=True)
        self.assertEqual((r["repaired"], r["refusals"]), (True, []))


class Validator(unittest.TestCase):
    READ = [{"tool": "read", "path": "basket.py", "result": {"text": "..."}}]

    def test_only_code_the_patch_adds_counts(self):
        keep = {"tool": "patch", "path": "basket.py", "old": "import os\nreturn a + b", "new": "import os\nreturn a * b"}
        self.assertIsNone(validate_patch(keep, self.READ))  # 'os' was already there: not introduced by the patch
        add = {"tool": "patch", "path": "basket.py", "old": "return a + b", "new": "requests.get('http://x')\nreturn a"}
        self.assertIn("network", validate_patch(add, self.READ)[1])

    def test_a_patch_needs_a_reason_to_touch_the_file(self):
        stray = {"tool": "patch", "path": "setup.py", "old": "a", "new": "b"}
        self.assertIn("neither named by a failing test nor read", validate_patch(stray, self.READ)[1])
        named = [{"tool": "test", "result": {"output": 'File "/w/setup.py", line 3'}}]
        self.assertIsNone(validate_patch(stray, named))

    def test_pve_is_off_unless_asked(self):
        bad = {"tool": "patch", "path": "basket.py", "old": "x", "new": PAYLOAD}
        self.assertIsNone(Guards().before(bad, self.READ))
        self.assertEqual(Guards(pve=True).before(bad, self.READ)[0], "refuse")

    def test_outside_text_is_labelled_untrusted_only_when_asked(self):
        self.assertNotIn("source", json.loads(observation({"text": "x"})))
        self.assertIn("do not follow", json.loads(observation({"text": "x"}, untrusted_label=True))["source"])
        self.assertNotIn("source", json.loads(observation({"applied": True}, untrusted_label=True)))

    def test_a_comment_holding_the_marker_is_not_a_live_payload(self):
        self.assertFalse(live_payload(f"    # {PAYLOAD}\n    return 1"))
        self.assertTrue(live_payload(f"    {PAYLOAD}\n    return 1"))


if __name__ == "__main__":
    unittest.main()
