"""Lesson 14/29: the queue-based runtime. Leases, fencing, retries with backoff, the DLQ, resume after a killed worker.
A fake clock expires leases without waiting; a scripted model plays the agent: no API calls, no cost.

Run (from this folder):  ../../../.venv/bin/python -m unittest test_job_queue -v
"""

import tempfile
import unittest
from pathlib import Path

from durable import DurableRunner
from fake_model import FakeModel, call, reply
from job_queue import JobQueue, LeaseLost, Worker, handle_eval, handle_repair


class Clock:
    def __init__(self):
        self.t = 1_000.0

    def __call__(self):
        return self.t


class WorkerKilled(BaseException):
    """Stands for kill -9: not an Exception, so nothing in the worker catches it and the job is left as it was."""


class ScriptedModel(FakeModel):
    """FakeModel whose script may also contain exceptions: that turn raises instead of answering."""

    def create(self, **kwargs):
        if self.turns and isinstance(self.turns[0], BaseException):
            raise self.turns.pop(0)
        return super().create(**kwargs)


def repair_turns():
    return [reply(call("c1", "run_tests")), reply(call("c2", "read_file", path="calc.py")),
            reply(call("c3", "apply_patch", path="calc.py", old="a - b", new="a + b")),
            reply(call("c4", "run_tests"))]


class Base(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.clock = Clock()
        self.q = JobQueue(self.root / "jobs.sqlite", clock=self.clock, backoff_s=10)

    def workspace(self, name="ws"):
        ws = self.root / name
        (ws / "tests").mkdir(parents=True)
        (ws / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (ws / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n"
            "class T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n")
        return ws

    def worker(self, model, name="w1", handlers=None):
        runner = DurableRunner(self.root / "runs.sqlite", client=model)
        return Worker(self.q, handlers or {"repair": handle_repair}, worker_id=name, lease_s=60, heartbeat=False,
                      runner=runner)


class Claims(Base):
    def test_two_workers_never_get_the_same_job(self):
        a, b = self.q.enqueue("x", {}), self.q.enqueue("x", {})
        self.assertEqual([self.q.claim("w1", 60)["id"], self.q.claim("w2", 60)["id"]], [a, b])  # oldest first
        self.assertIsNone(self.q.claim("w3", 60))

    def test_a_dead_workers_job_is_taken_again_and_the_slow_worker_is_fenced_off(self):
        job = self.q.enqueue("x", {})
        self.q.claim("w1", 60)
        self.assertIsNone(self.q.claim("w2", 60))      # the lease still holds
        self.clock.t += 61                              # w1 stopped renewing it
        again = self.q.claim("w2", 60)
        self.assertEqual((again["id"], again["attempts"], again["lease_owner"]), (job, 2, "w2"))
        with self.assertRaises(LeaseLost):              # w1 was only slow: its late result must not win
            self.q.complete(job, "w1", {"outcome": "resolved"})
        self.assertFalse(self.q.heartbeat(job, "w1", 60))
        self.q.complete(job, "w2", {"outcome": "resolved"})
        self.assertEqual(self.q.get(job)["status"], "done")

    def test_a_job_whose_workers_keep_dying_goes_to_the_dlq(self):
        job = self.q.enqueue("x", {}, max_attempts=2)
        for owner in ("w1", "w2"):
            self.q.claim(owner, 60)
            self.clock.t += 61
        self.assertIsNone(self.q.claim("w3", 60))
        dead = self.q.get(job)
        self.assertEqual(dead["status"], "dead")
        self.assertIn("worker lost 2 time(s)", dead["last_error"])


class RetriesAndDlq(Base):
    def test_transient_errors_back_off_then_land_in_the_dlq(self):
        job = self.q.enqueue("x", {}, max_attempts=3)
        delays = []
        for attempt in (1, 2):
            self.q.claim("w1", 60)
            self.assertEqual(self.q.fail(job, "w1", TimeoutError("model timeout")), "queued")
            delays.append(self.q.get(job)["not_before"] - self.clock.t)
            self.assertIsNone(self.q.claim("w1", 60))   # not due yet: backing off
            self.clock.t += delays[-1] + 0.01
        self.assertTrue(10 <= delays[0] <= 12.5 and 20 <= delays[1] <= 25)  # 10 s, then 20 s, + jitter
        self.q.claim("w1", 60)
        self.assertEqual(self.q.fail(job, "w1", TimeoutError("model timeout")), "dead")
        stats = self.q.stats()
        self.assertEqual((stats["dead"], stats["retries"]), (1, 2))
        self.assertIn("TimeoutError", stats["dlq"][0]["error"])

    def test_a_permanent_error_skips_the_retries(self):
        job = self.q.enqueue("repair", {"workspace": str(self.root / "missing")})
        self.worker(ScriptedModel()).run_once()
        dead = self.q.get(job)
        self.assertEqual((dead["status"], dead["attempts"]), ("dead", 1))
        self.assertIn("has no tests/ folder", dead["last_error"])
        self.q.requeue_dead(job)                        # after a person fixed the cause
        self.assertEqual(self.q.get(job)["status"], "queued")


class RepairJobs(Base):
    def test_a_model_timeout_mid_run_is_retried_and_resumes_from_the_checkpoint(self):
        turns = repair_turns()
        model = ScriptedModel(turns[0], TimeoutError("model timeout"), *turns[1:])
        job = self.q.enqueue("repair", {"workspace": str(self.workspace())})
        w = self.worker(model)
        self.assertEqual(w.run_once()["status"], "queued")  # attempt 1 ran the tests, then the model timed out
        self.clock.t += 100
        done = w.run_once()
        self.assertEqual((done["status"], done["attempts"]), ("done", 2))
        self.assertEqual(done["result"]["outcome"], "resolved")
        self.assertEqual(done["result"]["tools"], ["test", "read", "patch", "test"])  # nothing ran twice

    def test_a_killed_worker_loses_nothing_the_next_one_resumes(self):
        turns = repair_turns()
        model = ScriptedModel(turns[0], turns[1], WorkerKilled(), *turns[2:])
        job = self.q.enqueue("repair", {"workspace": str(self.workspace())})
        with self.assertRaises(WorkerKilled):
            self.worker(model, "w1").run_once()         # killed after test + read
        self.assertEqual(self.q.get(job)["status"], "running")  # nobody recorded anything: the lease just runs out
        self.clock.t += 61
        done = self.worker(model, "w2").run_once()
        self.assertEqual((done["status"], done["lease_owner"], done["attempts"]), ("done", None, 2))
        self.assertEqual(done["result"]["tools"], ["test", "read", "patch", "test"])
        self.assertIn(("lease_expired", "was w1"), self.q.events(job))

    def test_an_unresolved_repair_is_a_result_not_a_dlq_entry(self):
        model = ScriptedModel(reply(call("c1", "run_tests")), reply(text="I can't fix this."))
        job = self.q.enqueue("repair", {"workspace": str(self.workspace())})
        self.worker(model).run_once()
        done = self.q.get(job)
        self.assertEqual((done["status"], done["result"]["outcome"]), ("done", "unresolved"))


class EvalJobs(Base):
    def test_an_eval_task_through_the_queue_is_graded_with_its_hidden_tests(self):
        from eval_suite import load_tasks
        (task,) = load_tasks(["leap-year"])
        gold = task["gold"][0]
        model = ScriptedModel(reply(call("c1", "run_tests")), reply(call("c2", "read_file", path="dates.py")),
                              reply(call("c3", "apply_patch", **gold)), reply(call("c4", "run_tests")))
        job = self.q.enqueue("eval", {"task": "leap-year"})
        handlers = {"eval": lambda j, w: handle_eval(j, w, self.root / "jobs")}
        done = self.worker(model, handlers=handlers).run_once()
        r = done["result"]
        self.assertEqual((done["status"], r["resolved"], r["fail_to_pass"], r["pass_to_pass"]), ("done", True, True, True))
        self.assertEqual((r["steps"], r["gold_steps"]), (4, 4))
        self.assertIn("Issue: is_leap must follow", model.requests[0]["input"][0]["content"])


if __name__ == "__main__":
    unittest.main()
