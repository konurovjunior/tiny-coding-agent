"""Durable runs (lesson 14/13): checkpoints, resume after a crash, the approval gate, refusing a changed workspace.
A fake model plays the agent: no API calls, no cost.

Run (from this folder):  ../../../.venv/bin/python -m unittest test_durable -v
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import durable
from durable import DurableRunner, ResumeRefused, SimulatedCrash
from fake_model import FakeModel, call, reply

REAL_EXECUTE = durable._execute


def repair_script():
    """The model's turns for a clean repair: test, read, patch, test."""
    return [reply(call("c1", "run_tests")),
            reply(call("c2", "read_file", path="calc.py")),
            reply(call("c3", "apply_patch", path="calc.py", old="a - b", new="a + b")),
            reply(call("c4", "run_tests"))]


class Durable(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        root = Path(self.t.name)
        self.ws = root / "ws"
        (self.ws / "tests").mkdir(parents=True)
        (self.ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.ws / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n"
            "class T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n")
        self.db = root / "runs.sqlite"

    def runner(self, model):
        return DurableRunner(self.db, client=model)  # a new runner each time = a new process after a crash

    def run_id(self, runner):
        with runner.store._db() as db:
            return db.execute("SELECT run_id FROM checkpoints LIMIT 1").fetchone()[0]

    def test_a_crashed_run_resumes_where_it_stopped(self):
        model = FakeModel(*repair_script())
        first = self.runner(model)
        with self.assertRaises(SimulatedCrash):
            first.start(self.ws, crash_after=2)  # killed after test + read
        run_id = self.run_id(first)
        result = self.runner(model).resume(run_id)  # a fresh runner: only SQLite carries the state over
        self.assertEqual((result["status"], result["tools"]), ("completed", ["test", "read", "patch", "test"]))
        self.assertEqual(result["model_calls"], 4)  # the same 4 turns as an uninterrupted run: nothing re-asked
        self.assertIn("a + b", (self.ws / "calc.py").read_text())

    def test_every_action_is_checkpointed_before_and_after(self):
        runner = self.runner(FakeModel(*repair_script()))
        result = runner.start(self.ws)
        statuses = [status for _, status in runner.store.history(result["run_id"])]
        # start, then intent + result for 3 actions, then intent + the final "completed" for the 4th
        self.assertEqual(len(statuses), 1 + 2 * 3 + 2)
        self.assertEqual(statuses[-1], "completed")

    def test_the_gate_stops_before_a_patch_and_approval_runs_it(self):
        model = FakeModel(*repair_script())
        paused = self.runner(model).start(self.ws, gate_patches=True)
        self.assertEqual(paused["status"], "awaiting_approval")
        self.assertEqual(paused["awaiting"], {"tool": "patch", "path": "calc.py", "old": "a - b", "new": "a + b"})
        self.assertIn("a - b", (self.ws / "calc.py").read_text())  # nothing changed while waiting
        with self.assertRaises(ValueError):
            self.runner(model).resume(paused["run_id"])  # no decision: refused
        done = self.runner(model).resume(paused["run_id"], decision="approve")
        self.assertEqual(done["status"], "completed")

    def test_a_rejection_reaches_the_model_and_it_must_try_again(self):
        model = FakeModel(reply(call("c0", "read_file", path="calc.py")),  # read first: the validator wants a reason
                          reply(call("c1", "apply_patch", path="calc.py", old="a - b", new="b - a")),
                          reply(call("c2", "apply_patch", path="calc.py", old="a - b", new="a + b")),
                          reply(call("c3", "run_tests")))
        paused = self.runner(model).start(self.ws, gate_patches=True)
        again = self.runner(model).resume(paused["run_id"], decision="reject", reason="b - a is still wrong")
        self.assertEqual(again["awaiting"]["new"], "a + b")  # the model proposed something else
        [told] = [i for i in model.requests[2]["input"] if i.get("call_id") == "c1"]
        self.assertIn("rejected by a human reviewer: b - a is still wrong", json.loads(told["output"])["error"])
        self.assertEqual(self.runner(model).resume(paused["run_id"], decision="approve")["status"], "completed")

    def test_a_guard_refusal_never_reaches_the_human_gate(self):
        model = FakeModel(reply(call("c0", "read_file", path="calc.py")),
                          reply(call("c1", "apply_patch", path="tests/test_calc.py", old="5", new="-1")),
                          reply(call("c2", "apply_patch", path="calc.py", old="a - b", new="a + b")),
                          reply(call("c3", "run_tests")))
        paused = self.runner(model).start(self.ws, gate_patches=True)
        self.assertEqual(paused["awaiting"]["path"], "calc.py")  # the test edit was refused before any review
        self.assertEqual(self.runner(model).resume(paused["run_id"], decision="approve")["status"], "completed")
        self.assertIn("assertEqual(add(2, 3), 5)", (self.ws / "tests" / "test_calc.py").read_text())

    def test_a_changed_workspace_is_refused_on_resume(self):
        model = FakeModel(*repair_script())
        with self.assertRaises(SimulatedCrash):
            self.runner(model).start(self.ws, crash_after=1)
        (self.ws / "calc.py").write_text("def add(a, b):\n    return 42\n")  # someone edits the code meanwhile
        with self.assertRaisesRegex(ResumeRefused, "calc.py"):
            self.runner(model).resume(self.run_id(self.runner(model)))

    def test_crash_before_acting_runs_the_saved_intent_without_asking_again(self):
        model = FakeModel(*repair_script())
        calls = {"n": 0}

        def dies_before_the_third_action(workspace, action):
            calls["n"] += 1
            if calls["n"] == 3:
                raise SimulatedCrash("killed before the patch ran")
            return REAL_EXECUTE(workspace, action)

        with mock.patch.object(durable, "_execute", dies_before_the_third_action), self.assertRaises(SimulatedCrash):
            self.runner(model).start(self.ws)
        result = self.runner(model).resume(self.run_id(self.runner(model)))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 4)  # the patch came from the saved intent, not a new model turn

    def test_crash_after_acting_but_before_saving_is_refused_not_repeated(self):
        model = FakeModel(*repair_script())
        calls = {"n": 0}

        def dies_after_the_patch_ran(workspace, action):
            calls["n"] += 1
            result = REAL_EXECUTE(workspace, action)
            if calls["n"] == 3:
                raise SimulatedCrash("killed after the patch, before its checkpoint")
            return result

        with mock.patch.object(durable, "_execute", dies_after_the_patch_ran), self.assertRaises(SimulatedCrash):
            self.runner(model).start(self.ws)
        with self.assertRaisesRegex(ResumeRefused, "calc.py"):  # the files moved on, the checkpoint didn't
            self.runner(model).resume(self.run_id(self.runner(model)))

    def test_resuming_a_finished_run_changes_nothing(self):
        model = FakeModel(*repair_script())
        done = self.runner(model).start(self.ws)
        self.assertEqual(self.runner(model).resume(done["run_id"])["status"], "completed")
        self.assertEqual(len(model.requests), 4)


if __name__ == "__main__":
    unittest.main()
