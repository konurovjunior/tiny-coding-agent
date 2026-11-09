"""Lesson 14/30: the eval suite's own checks. No API calls (a scripted model plays the agent).

Run (from this folder):  ../../../.venv/bin/python -m unittest test_eval_suite -v
"""

import tempfile
import unittest
from pathlib import Path

from eval_suite import (fresh_workspace, gate, grade, load_tasks, run_agent, summarize, validate_task,
                        wilson)
from fake_model import FakeModel, call, reply
from main import apply_patch

TASKS = {t["id"]: t for t in load_tasks()}


def graded_after(task, patches):
    with tempfile.TemporaryDirectory() as tmp:
        ws = fresh_workspace(task, Path(tmp))
        for p in patches:
            apply_patch(ws, p["path"], p["old"], p["new"])
        return grade(task, ws)


def scripted(patches, read_path):
    """A model that runs the tests, reads one file, applies the given patches and runs the tests again."""
    turns = [reply(call("c1", "run_tests")), reply(call("c2", "read_file", path=read_path))]
    turns += [reply(call(f"p{i}", "apply_patch", **p)) for i, p in enumerate(patches)]
    turns += [reply(call("c9", "run_tests")), reply(text="Fixed: the tests pass.")]
    return FakeModel(*turns)


class Tasks(unittest.TestCase):
    def test_the_trap_tasks_are_fair_questions(self):  # the CLI's --validate checks all nine (about 20 s)
        for task_id in ("leap-year", "money-rounding", "word-count"):
            with self.subTest(task_id):
                self.assertEqual(validate_task(TASKS[task_id]), [])

    def test_a_half_fix_passes_the_visible_tests_and_is_graded_overfit(self):
        task = TASKS["leap-year"]
        g = graded_after(task, task["decoys"][0]["patches"])
        self.assertTrue(g["visible_passed"])
        self.assertTrue(g["fail_to_pass"])
        self.assertFalse(g["pass_to_pass"])  # 2000 is a leap year again
        self.assertEqual((g["resolved"], g["overfit"]), (False, True))

    def test_editing_the_visible_tests_never_counts(self):
        task = TASKS["orchard-total"]
        g = graded_after(task, [{"path": "tests/test_basket.py", "old": "21", "new": "10"},
                                {"path": "tests/test_basket.py", "old": "total(7, 0), 0)", "new": "total(7, 0), 7)"}])
        self.assertTrue(g["visible_passed"])  # the edited tests pass...
        self.assertTrue(g["tests_edited"])
        self.assertFalse(g["resolved"])       # ...but nothing was fixed

    def test_a_fix_outside_the_allowed_files_is_reported(self):
        task = TASKS["money-rounding"]
        g = graded_after(task, task["decoys"][0]["patches"])  # fixes the symptom in cart.py
        self.assertEqual(g["changed_files"], ["cart.py"])
        self.assertFalse(g["fail_to_pass"])   # to_tiyn(0.29) is still 28


class LiveRunShape(unittest.TestCase):
    def test_the_gold_fix_through_the_real_loop_and_planner_is_resolved(self):
        task = TASKS["leap-year"]
        model = scripted(task["gold"], "dates.py")
        r = run_agent(task, "t-gold", client=model)
        self.assertEqual((r["state"], r["resolved"], r["overfit"]), ("completed", True, False))
        self.assertEqual(r["steps"], r["gold_steps"])
        self.assertIn("Issue: is_leap must follow the Gregorian calendar", model.requests[0]["input"][0]["content"])

    def test_the_half_fix_completes_the_run_but_is_not_resolved(self):
        task = TASKS["leap-year"]
        r = run_agent(task, "t-decoy", client=scripted(task["decoys"][0]["patches"], "dates.py"))
        self.assertEqual(r["state"], "completed")  # every earlier lesson would have called this a success
        self.assertEqual((r["resolved"], r["overfit"]), (False, True))


class Numbers(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(5, 5)
        self.assertAlmostEqual(lo, 0.566, places=3)  # 5/5 only says "at least about 57%"
        self.assertEqual(hi, 1.0)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))

    def test_pass_at_k_and_pass_hat_k(self):
        runs = [{"task": "a", "resolved": ok, "overfit": False, "tests_edited": False, "out_of_scope": [],
                 "refusals": [], "steps": 4, "gold_steps": 4, "cost_usd": 0.001} for ok in (True, True, False)]
        runs += [{**runs[0], "task": "b"} for _ in range(3)]
        s = summarize(runs)
        self.assertEqual((s["resolved"], s["runs"]), (5, 6))
        self.assertEqual(s["per_task"]["a"]["pass@1"], 2 / 3)
        self.assertEqual((s["per_task"]["a"]["pass^k"], s["per_task"]["a"]["pass@k"]), (False, True))
        self.assertEqual((s["pass^k_tasks"], s["pass@k_tasks"]), (1, 2))

    def test_gate(self):
        def result(rate, a, b):
            return {"rate": rate, "per_task": {"a": {"k": 3, "resolved": a}, "b": {"k": 3, "resolved": b}}}
        base = result(0.83, 3, 2)
        self.assertEqual(gate(result(0.83, 2, 3), base), (True, []))        # one run either way is noise
        ok, why = gate(result(0.67, 1, 3), base)                              # task a lost 2 of 3
        self.assertFalse(ok)
        self.assertIn("a: 3/3 → 1/3", why)
        ok, why = gate(result(0.5, 2, 1), base)                               # overall −33 points
        self.assertFalse(ok)
        self.assertTrue(why[0].startswith("resolve rate"))


if __name__ == "__main__":
    unittest.main()
