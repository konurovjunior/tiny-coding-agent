"""Lesson 14/24: the trace shape the agent sends to Langfuse, checked in memory (no Langfuse, no API calls).

Run (from this folder):  ../../../.venv/bin/python -m unittest test_telemetry -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import telemetry
from eval_suite import load_tasks, run_agent
from fake_model import FakeModel, call, reply
from job_queue import JobQueue, Worker, handle_eval
from durable import DurableRunner

LEAP = load_tasks(["leap-year"])[0]
GOLD = LEAP["gold"][0]


def gold_model():
    return FakeModel(reply(call("c1", "run_tests")), reply(call("c2", "read_file", path="dates.py")),
                     reply(call("c3", "apply_patch", **GOLD)), reply(call("c4", "run_tests")))


class Traced(unittest.TestCase):
    def setUp(self):
        telemetry.setup("memory")
        self.addCleanup(telemetry.setup, "off")

    def spans(self):
        return telemetry.memory_spans.get_finished_spans()

    def by_type(self, kind):
        return [s for s in self.spans() if s.attributes.get("langfuse.observation.type") == kind]

    def test_one_run_is_one_trace_agent_then_generations_and_tools(self):
        r = run_agent(LEAP, "batch-1-leap-year-1", client=gold_model())
        (agent,) = self.by_type("agent")
        chats, tools = self.by_type("generation"), self.by_type("tool")
        self.assertEqual((len(chats), len(tools)), (r["model_calls"], r["steps"]))  # 4 and 4
        self.assertEqual([t.attributes["gen_ai.tool.name"] for t in tools],
                         ["run_tests", "read_file", "apply_patch", "run_tests"])
        for s in chats + tools:  # all children of the run, all in its trace
            self.assertEqual(s.parent.span_id, agent.context.span_id)
            self.assertEqual(s.context.trace_id, agent.context.trace_id)
        self.assertEqual(format(agent.context.trace_id, "032x"), r["trace_id"])
        for s in self.spans():  # what every span carries, so Langfuse can filter on it
            self.assertEqual(s.attributes["langfuse.session.id"], "batch-1")
            self.assertEqual(s.attributes["langfuse.version"], telemetry.prompt_version())
        usage = json.loads(chats[0].attributes["langfuse.observation.usage_details"])
        self.assertEqual(usage, {"input": 1000, "output": 100, "total": 1100})
        self.assertEqual(chats[0].attributes["gen_ai.usage.input_tokens"], 1000)
        self.assertAlmostEqual(json.loads(chats[0].attributes["langfuse.observation.cost_details"])["total"], 0.00015)
        self.assertIn('"resolved": true', agent.attributes["langfuse.observation.output"])

    def test_the_eval_verdict_becomes_scores_on_the_trace(self):
        r = run_agent(LEAP, "batch-1-leap-year-1", client=gold_model())
        scores = {s["name"]: (s["value"], s["dataType"], s["traceId"]) for s in telemetry.memory_scores}
        self.assertEqual(scores, {"resolved": (1, "BOOLEAN", r["trace_id"]), "overfit": (0, "BOOLEAN", r["trace_id"]),
                                  "steps_over_gold": (1.0, "NUMERIC", r["trace_id"])})

    def test_a_refused_action_is_a_warning_with_the_guards_reason(self):
        model = FakeModel(reply(call("c1", "run_tests")),
                          reply(call("c2", "apply_patch", path="tests/test_dates.py", old="1900", new="1904")),
                          reply(text="giving up"))
        run_agent(LEAP, "b-leap-year-1", client=model)
        refused = [t for t in self.by_type("tool") if t.attributes.get("langfuse.observation.level") == "WARNING"]
        self.assertEqual(len(refused), 1)
        self.assertIn("not allowed", refused[0].attributes["langfuse.observation.status_message"])
        self.assertEqual([e.name for e in refused[0].events], ["guard_refused"])

    def test_a_model_timeout_shows_as_an_error_generation_and_a_failed_run(self):
        class Timeout(FakeModel):
            def create(self, **kwargs):
                raise TimeoutError("model timeout")
        with self.assertRaises(TimeoutError):
            run_agent(LEAP, "b-leap-year-1", client=Timeout())
        (chat,) = self.by_type("generation")
        (agent,) = self.by_type("agent")
        self.assertEqual(chat.attributes["langfuse.observation.level"], "ERROR")
        self.assertEqual(agent.attributes["langfuse.observation.level"], "ERROR")

    def test_a_queued_job_attempt_is_traced_with_its_job_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = JobQueue(Path(tmp) / "jobs.sqlite")
            job = q.enqueue("eval", {"task": "leap-year"}, batch="nightly-x")
            worker = Worker(q, {"eval": lambda j, w: handle_eval(j, w, Path(tmp) / "jobs")}, heartbeat=False,
                            runner=DurableRunner(Path(tmp) / "runs.sqlite", client=gold_model()))
            self.assertEqual(worker.run_once()["status"], "done")
        (agent,) = self.by_type("agent")
        self.assertEqual(agent.attributes["langfuse.trace.metadata.job_id"], job)
        self.assertEqual(agent.attributes["langfuse.trace.metadata.attempt"], "1")
        self.assertEqual(agent.attributes["langfuse.session.id"], "nightly-x")
        self.assertEqual(len(self.by_type("tool")), 4)
        self.assertEqual({s["name"] for s in telemetry.memory_scores}, {"resolved", "overfit", "steps_over_gold"})


class Off(unittest.TestCase):
    def test_off_sends_nothing_and_changes_nothing(self):
        telemetry.setup("off")
        r = run_agent(LEAP, "b-leap-year-1", client=gold_model())
        self.assertTrue(r["resolved"])
        self.assertIsNone(r["trace_id"])


if __name__ == "__main__":
    unittest.main()
