"""The queue-based runtime for the agent (lesson 14/29): jobs in SQLite, workers with leases, retries, a dead-letter queue.

Until now the agent ran request-response (run_llm_repair.py: start it and wait) or durable (durable.py: checkpoints,
resume). A real "fix this failing build" agent runs in the background: jobs arrive, workers pick them up, and a job
that fails must never just vanish. This module is that shape, with the standard library only.

A job's life:
    queued ──claim──▶ running ──▶ done      the handler returned a result (the agent may still be "unresolved":
       ▲                 │                    that is an answer about the bug, not a fault in the system)
       └──retry later────┤ transient error (OpenAI timeout / 5xx / 429, a lost connection): back to queued with
                         │ exponential backoff, while attempts < max_attempts
                         └──▶ dead       a permanent error (bad payload, a workspace changed under a run), or the
                                         retries ran out: parked in the DLQ with its last error, for a person

Leases (why a dead worker never loses a job): claiming a job writes lease_owner and lease_until. A live worker renews
the lease from a heartbeat thread. A worker that dies stops renewing, the lease expires, and the next claim takes the
job again (that counts as an attempt). Every write a worker makes afterwards checks lease_owner, so a worker that was
only slow, not dead, cannot overwrite the result of the worker that took over (fencing; in Java terms, optimistic
locking on a version column). Repair jobs run through durable.py with the job id as the run id, so a re-claimed job
resumes from its last checkpoint instead of starting over.

Java analogy: a Kafka consumer group with a dead-letter topic, or Spring Batch's retry + skip, as one table.
"""

import json
import random
import shutil
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import openai

from durable import DurableRunner
from llm_planner import MODEL, PRICES

TRANSIENT = (TimeoutError, ConnectionError, openai.APITimeoutError, openai.APIConnectionError,
             openai.RateLimitError, openai.InternalServerError)  # worth another try; anything else is permanent


def is_transient(error: BaseException) -> bool:
    return isinstance(error, TRANSIENT)


SCHEMA = (
    """CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, batch TEXT,
        status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'dead')),
        attempts INTEGER NOT NULL DEFAULT 0, max_attempts INTEGER NOT NULL,
        lease_owner TEXT, lease_until REAL, not_before REAL NOT NULL,
        last_error TEXT, result TEXT,
        created REAL NOT NULL, started REAL, finished REAL)""",
    """CREATE INDEX IF NOT EXISTS jobs_claim ON jobs (status, not_before, created)""",
    """CREATE TABLE IF NOT EXISTS job_events (
        job_id TEXT NOT NULL, at REAL NOT NULL, event TEXT NOT NULL, detail TEXT)""",
)


class LeaseLost(Exception):
    """This worker no longer owns the job (its lease expired and another worker took it): drop the result."""


class JobQueue:
    def __init__(self, path, clock=time.time, backoff_s: float = 2.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock          # tests pass a fake clock to expire leases without waiting
        self.backoff_s = backoff_s  # retry n waits backoff_s * 2**(n-1), plus up to 25% jitter
        # executescript() would COMMIT the open transaction first, so the statements run one by one
        with self._tx() as db:
            for statement in SCHEMA:
                db.execute(statement)

    @contextmanager
    def _tx(self):
        """One connection per operation. BEGIN IMMEDIATE takes SQLite's write lock at the start, so two workers can
        never claim the same job: the second waits (busy timeout) and then sees it already taken."""
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def _event(self, db, job_id, event, detail=""):
        db.execute("INSERT INTO job_events VALUES (?, ?, ?, ?)", (job_id, self.clock(), event, detail))

    # --- producers -------------------------------------------------------------------------------------------------
    def enqueue(self, kind: str, payload: dict, batch: str | None = None, max_attempts: int = 3) -> str:
        job_id = "job-" + uuid.uuid4().hex[:10]
        now = self.clock()
        with self._tx() as db:
            db.execute("INSERT INTO jobs (id, kind, payload, batch, status, max_attempts, not_before, created) "
                       "VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)",
                       (job_id, kind, json.dumps(payload, ensure_ascii=False), batch, max_attempts, now, now))
            self._event(db, job_id, "enqueued")
        return job_id

    # --- workers ---------------------------------------------------------------------------------------------------
    def claim(self, owner: str, lease_s: float) -> dict | None:
        """Take the oldest job that is due: queued and past its backoff, or running with an expired lease (its worker
        died). A job whose lease expired with no attempts left goes to the DLQ instead of being run again."""
        now = self.clock()
        with self._tx() as db:
            for row in db.execute("SELECT id, attempts, lease_owner FROM jobs WHERE status = 'running' "
                                  "AND lease_until < ? AND attempts >= max_attempts", (now,)).fetchall():
                db.execute("UPDATE jobs SET status = 'dead', finished = ?, lease_owner = NULL, last_error = ? "
                           "WHERE id = ?", (now, f"worker lost {row['attempts']} time(s) (last: {row['lease_owner']})",
                                            row["id"]))
                self._event(db, row["id"], "dead", "lease expired, no attempts left")
            row = db.execute("SELECT * FROM jobs WHERE (status = 'queued' AND not_before <= ?) "
                             "OR (status = 'running' AND lease_until < ?) ORDER BY created LIMIT 1",
                             (now, now)).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                self._event(db, row["id"], "lease_expired", f"was {row['lease_owner']}")
            db.execute("UPDATE jobs SET status = 'running', attempts = attempts + 1, lease_owner = ?, "
                       "lease_until = ?, started = COALESCE(started, ?) WHERE id = ?",
                       (owner, now + lease_s, now, row["id"]))
            self._event(db, row["id"], "claimed", f"{owner}, attempt {row['attempts'] + 1}")
            return self._job(db, row["id"])

    def heartbeat(self, job_id: str, owner: str, lease_s: float) -> bool:
        with self._tx() as db:
            cur = db.execute("UPDATE jobs SET lease_until = ? WHERE id = ? AND status = 'running' AND lease_owner = ?",
                             (self.clock() + lease_s, job_id, owner))
            return cur.rowcount == 1

    def complete(self, job_id: str, owner: str, result: dict) -> None:
        with self._tx() as db:
            cur = db.execute("UPDATE jobs SET status = 'done', result = ?, finished = ?, lease_owner = NULL, "
                             "last_error = NULL WHERE id = ? AND status = 'running' AND lease_owner = ?",
                             (json.dumps(result, ensure_ascii=False), self.clock(), job_id, owner))
            if cur.rowcount != 1:
                raise LeaseLost(job_id)
            self._event(db, job_id, "done", result.get("outcome", ""))

    def fail(self, job_id: str, owner: str, error: BaseException) -> str:
        """Record a failed attempt. Returns the new status: 'queued' (will be retried) or 'dead' (in the DLQ)."""
        message = f"{type(error).__name__}: {error}"[:2000]
        now = self.clock()
        with self._tx() as db:
            row = db.execute("SELECT attempts, max_attempts FROM jobs WHERE id = ? AND status = 'running' "
                             "AND lease_owner = ?", (job_id, owner)).fetchone()
            if row is None:
                raise LeaseLost(job_id)
            if is_transient(error) and row["attempts"] < row["max_attempts"]:
                delay = self.backoff_s * 2 ** (row["attempts"] - 1) * (1 + random.random() / 4)
                db.execute("UPDATE jobs SET status = 'queued', not_before = ?, lease_owner = NULL, last_error = ? "
                           "WHERE id = ?", (now + delay, message, job_id))
                self._event(db, job_id, "retry", f"in {delay:.1f} s: {message}")
                return "queued"
            why = "permanent error" if not is_transient(error) else f"{row['attempts']} attempts used"
            db.execute("UPDATE jobs SET status = 'dead', finished = ?, lease_owner = NULL, last_error = ? "
                       "WHERE id = ?", (now, message, job_id))
            self._event(db, job_id, "dead", f"{why}: {message}")
            return "dead"

    def requeue_dead(self, job_id: str) -> None:
        """After a person fixed the cause: give a DLQ job a fresh set of attempts."""
        with self._tx() as db:
            cur = db.execute("UPDATE jobs SET status = 'queued', attempts = 0, not_before = ?, finished = NULL "
                             "WHERE id = ? AND status = 'dead'", (self.clock(), job_id))
            if cur.rowcount != 1:
                raise KeyError(f"no dead job {job_id!r}")
            self._event(db, job_id, "requeued")

    # --- reading -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _job(db, job_id) -> dict:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        job = dict(row)
        job["payload"] = json.loads(job["payload"])
        job["result"] = json.loads(job["result"]) if job["result"] else None
        return job

    def get(self, job_id: str) -> dict:
        with self._tx() as db:
            return self._job(db, job_id)

    def jobs(self, batch: str | None = None) -> list[dict]:
        with self._tx() as db:
            ids = [r["id"] for r in db.execute(
                "SELECT id FROM jobs WHERE ? IS NULL OR batch = ? ORDER BY created", (batch, batch))]
            return [self._job(db, i) for i in ids]

    def events(self, job_id: str) -> list[tuple[str, str]]:
        with self._tx() as db:
            return [(r["event"], r["detail"]) for r in db.execute(
                "SELECT event, detail FROM job_events WHERE job_id = ? ORDER BY rowid", (job_id,))]

    def stats(self, batch: str | None = None) -> dict:
        """What the lesson says to watch for this shape: queue depth, per-job latency, retries, DLQ size."""
        jobs, now = self.jobs(batch), self.clock()
        by_status = {s: sum(j["status"] == s for j in jobs) for s in ("queued", "running", "done", "dead")}
        finished = [j for j in jobs if j["finished"] is not None and j["started"] is not None]
        return {
            **by_status, "total": len(jobs),
            "depth_due": sum(j["status"] == "queued" and j["not_before"] <= now for j in jobs),
            "retries": sum(max(0, j["attempts"] - 1) for j in jobs),
            "wait_s": percentiles([j["started"] - j["created"] for j in finished]),
            "run_s": percentiles([j["finished"] - j["started"] for j in finished]),
            "dlq": [{"id": j["id"], "kind": j["kind"], "attempts": j["attempts"], "error": j["last_error"]}
                    for j in jobs if j["status"] == "dead"],
        }


def percentiles(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None, "max": None}
    v = sorted(values)
    pick = lambda q: round(v[min(len(v) - 1, int(q * len(v)))], 2)  # nearest rank, enough for a handful of jobs
    return {"p50": pick(0.5), "p95": pick(0.95), "max": round(v[-1], 2)}


# ------------------------------------------------------------------------------------------------------- the worker

class Worker:
    """Claims jobs and runs each through its kind's handler, with a heartbeat that renews the lease meanwhile.

    handlers: {kind: handler(job, worker) -> result dict}. A handler raises to fail the attempt; the queue decides
    between retry and DLQ from the error's type. A BaseException that isn't an Exception (KeyboardInterrupt, or the
    tests' simulated kill) leaves the job untouched, exactly as a killed process would: its lease just runs out."""

    def __init__(self, queue: JobQueue, handlers: dict, worker_id: str | None = None, lease_s: float = 120,
                 heartbeat: bool = True, runner: DurableRunner | None = None):
        self.queue, self.handlers = queue, handlers
        self.id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self.lease_s, self.use_heartbeat = lease_s, heartbeat
        self.runner = runner  # the durable runner repair handlers use (its own checkpoint database)

    def run_once(self) -> dict | None:
        job = self.queue.claim(self.id, self.lease_s)
        if job is None:
            return None
        stop = threading.Event()
        beat = threading.Thread(target=self._beat, args=(job["id"], stop), daemon=True)
        if self.use_heartbeat:
            beat.start()
        try:
            handler = self.handlers.get(job["kind"])
            if handler is None:
                raise ValueError(f"no handler for job kind {job['kind']!r}")
            result = handler(job, self)
        except Exception as e:  # the attempt failed; the job itself is safe in the queue
            status = self.queue.fail(job["id"], self.id, e)
            return {**job, "status": status, "last_error": str(e)}
        finally:
            stop.set()
        self.queue.complete(job["id"], self.id, result)
        return self.queue.get(job["id"])

    def run(self, max_jobs: int | None = None, idle_exit: bool = True, idle_sleep: float = 1.0) -> int:
        """Work until the queue has nothing due (idle_exit) or max_jobs were taken. Returns how many were taken."""
        taken = 0
        while max_jobs is None or taken < max_jobs:
            job = self.run_once()
            if job is None:
                if idle_exit and not self._anything_pending():
                    return taken
                time.sleep(idle_sleep)  # jobs are waiting out a backoff, or another worker holds them
                continue
            taken += 1
        return taken

    def _anything_pending(self) -> bool:
        s = self.queue.stats()
        return s["queued"] + s["running"] > 0

    def _beat(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(self.lease_s / 3):
            if not self.queue.heartbeat(job_id, self.id, self.lease_s):
                return  # lease lost: complete()/fail() will refuse this worker's write


# ----------------------------------------------------------------------------------------------------- the handlers

def run_cost(runner: DurableRunner, run_id: str) -> tuple[dict, float]:
    state = runner.store.latest(run_id)
    planner = state["planner"] or {}
    price_in, price_out = PRICES[MODEL]  # durable runs use the configured model
    cost = (planner.get("input_tokens", 0) * price_in + planner.get("output_tokens", 0) * price_out) / 1_000_000
    return state, cost


def durable_repair(runner: DurableRunner, job: dict, workspace: Path, max_steps: int, issue: str | None) -> dict:
    """Start the job's durable run, or resume it if an earlier attempt got as far as a checkpoint."""
    try:
        runner.store.latest(job["id"])
    except KeyError:
        return runner.start(workspace, max_steps=max_steps, run_id=job["id"], issue=issue)
    return runner.resume(job["id"])  # ResumeRefused (files changed under the run) is permanent: the DLQ


def handle_repair(job: dict, worker: Worker) -> dict:
    """payload: {workspace, max_steps?, issue?}. Repairs the workspace in place, like durable_run.py start."""
    p = job["payload"]
    workspace = Path(p.get("workspace", ""))
    if not p.get("workspace") or not (workspace / "tests").is_dir():
        raise ValueError(f"workspace {p.get('workspace')!r} has no tests/ folder")  # permanent: retrying won't help
    summary = durable_repair(worker.runner, job, workspace, p.get("max_steps", 12), p.get("issue"))
    _, cost = run_cost(worker.runner, job["id"])
    return {**summary, "outcome": "resolved" if summary["status"] == "completed" else "unresolved", "cost_usd": cost}


def handle_eval(job: dict, worker: Worker, jobs_dir: Path) -> dict:
    """payload: {task, max_steps?, with_issue?}. One eval_suite run through the queue: a copy of the task's workspace
    in the job's own folder (made once, so a retry resumes on the same files), a durable run, hidden-test grading."""
    from eval_suite import MAX_STEPS, gold_steps, grade, load_tasks
    p = job["payload"]
    (task,) = load_tasks([p["task"]])
    ws = jobs_dir / job["id"] / "workspace"
    if not ws.exists():
        shutil.copytree(task["dir"] / "workspace", ws, ignore=shutil.ignore_patterns("__pycache__"))
    summary = durable_repair(worker.runner, job, ws, p.get("max_steps", MAX_STEPS),
                             task["issue"] if p.get("with_issue", True) else None)
    graded = grade(task, ws)
    state, cost = run_cost(worker.runner, job["id"])
    trace = state["trace"]
    return {"task": task["id"], "run_id": job["id"], "state": summary["status"], **graded,
            "outcome": "resolved" if graded["resolved"] else "unresolved",
            "steps": len(trace), "gold_steps": gold_steps(task), "tools": [s["tool"] for s in trace],
            "refusals": [s["result"]["error"] for s in trace if s["result"].get("guard")],
            "model_calls": summary["model_calls"], "cost_usd": cost}
