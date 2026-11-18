"""The model planner, driven by a fake model: no API calls, no cost.

Run (from this folder):  python -m unittest test_llm_planner -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from fake_model import FakeModel, call, reply
from llm_planner import LlmPlanner
from main import drive


class Planner(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.ws = Path(self.t.name)
        (self.ws / "tests").mkdir()
        (self.ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (self.ws / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n"
            "class T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n")
        self.addCleanup(self.t.cleanup)

    def run_with(self, model, max_steps=12):
        planner = LlmPlanner(self.ws, client=model, model="gpt-6-luna")
        return planner, drive(self.ws, planner, max_steps=max_steps, on_invalid="observe")

    def tool_results(self, model, request_index):
        return [i for i in model.requests[request_index]["input"] if i.get("type") == "function_call_output"]

    def test_repairs_and_completes_on_real_test_evidence(self):
        model = FakeModel(
            reply(call("c1", "run_tests"), call("c2", "read_file", path="calc.py")),  # two calls in one turn
            reply(call("c3", "apply_patch", path="calc.py", old="a - b", new="a + b")),
            reply(call("c4", "run_tests")),
        )
        planner, result = self.run_with(model)
        self.assertEqual(result["state"], "completed")
        self.assertEqual([s["tool"] for s in result["trace"]], ["test", "read", "patch", "test"])
        self.assertEqual(planner.model_calls, 3)
        self.assertAlmostEqual(planner.cost_usd, 3 * (1000 * 0.10 + 100 * 0.50) / 1e6)

    def test_parallel_results_go_back_under_their_own_ids(self):
        model = FakeModel(reply(call("c1", "run_tests"), call("c2", "read_file", path="calc.py")),
                          reply(text="I give up."))
        self.run_with(model)
        results = {i["call_id"]: json.loads(i["output"]) for i in self.tool_results(model, 1)}
        self.assertIsNotNone(model.requests[1]["previous_response_id"])  # turn 2 continues turn 1
        self.assertEqual(set(results), {"c1", "c2"})
        self.assertIn("passed", results["c1"])           # the test run's evidence...
        self.assertIn("return a - b", results["c2"]["text"])  # ...and the file text, each under its own call

    def test_a_bad_call_goes_back_as_an_error_and_never_runs(self):
        model = FakeModel(reply(call("c1", "read_file", path="calc.py", mode="w")),  # unknown argument
                          reply(text="stopping"))
        _, result = self.run_with(model)
        self.assertEqual(result["trace"], [])  # drive() never saw it
        [error] = self.tool_results(model, 1)
        self.assertIn("unknown argument 'mode'", json.loads(error["output"])["error"])

    def test_a_refused_patch_is_an_observation_the_model_can_fix(self):
        model = FakeModel(
            reply(call("c1", "apply_patch", path="calc.py", old="a-b", new="a+b")),  # not the exact text
            reply(call("c2", "apply_patch", path="calc.py", old="a - b", new="a + b")),
            reply(call("c3", "run_tests")),
        )
        _, result = self.run_with(model)
        self.assertEqual(result["state"], "completed")
        self.assertIn("found 0 times", result["trace"][0]["result"]["error"])

    def test_claiming_success_without_passing_tests_is_not_completion(self):
        model = FakeModel(reply(text="Fixed it, all good."))  # words, no tool calls
        planner, result = self.run_with(model)
        self.assertEqual(result["state"], "failed")
        self.assertEqual(planner.final_text, "Fixed it, all good.")

    def test_a_same_size_patch_in_the_same_second_is_seen_by_the_next_test_run(self):
        # regression: a stale __pycache__ made the second run test the old code (the size and second didn't change)
        from main import apply_patch, run_tests
        self.assertFalse(run_tests(self.ws)["passed"])
        apply_patch(self.ws, "calc.py", "a - b", "a + b")  # same length, immediately after
        self.assertTrue(run_tests(self.ws)["passed"])
        self.assertFalse((self.ws / "__pycache__").exists())  # and the workspace is not littered with caches

    def test_the_step_budget_still_holds(self):
        model = FakeModel(*[reply(call(f"c{i}", "read_file", path="calc.py")) for i in range(5)])
        _, result = self.run_with(model, max_steps=3)
        self.assertEqual((result["state"], result["steps"]), ("budget_exhausted", 3))


if __name__ == "__main__":
    unittest.main()
