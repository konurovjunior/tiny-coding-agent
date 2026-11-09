"""Durable repair runs: a checkpoint in SQLite after every step, resume after a crash, and a human approval gate
before every patch (lesson 14/13). Built with the standard library to see the mechanism; LangGraph packages the same
ideas (checkpointers, resume, interrupt).

It runs the same loop as main.drive() with on_invalid="observe", guards.Guards and the same tools and stop rules, plus:

  Two checkpoints per action, like a database's write-ahead log:
    1. intent: "about to run this action" (saved BEFORE acting)
    2. result: the trace entry and the workspace's file fingerprints (saved AFTER acting)
  On resume, the saved fingerprints are compared with the workspace as it is now:
    - they match            -> continue; a saved intent that never ran is run now
    - they differ           -> refuse: something changed the files since the last checkpoint (a person, or an
                               action that ran just before a crash); continuing would plan from a false picture
  Approval gate (gate_patches=True): before a patch runs, the run stops as "awaiting_approval" with the exact patch.
    resume(decision="approve") runs it; resume(decision="reject", reason=...) gives the reason to the model as the
    patch's result, so it has to try something else. Either way the decision is recorded in the trace.

A model turn that happens after the last checkpoint and before a crash is lost and asked again on resume: one
extra model call, never a repeated workspace action.
"""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from guards import Guards
from llm_planner import LlmPlanner
from main import _execute
from memory import RepairMemory

FINISHED = {"completed", "failed", "budget_exhausted"}


class SimulatedCrash(Exception):
    """Raised by crash_after=N right after step N's checkpoint, as if the process had been killed there."""


class ResumeRefused(Exception):
    pass


def fingerprints(workspace) -> dict[str, str]:
    """sha256 of every file in the workspace (tests included): the part of the run's state that lives on disk."""
    root = Path(workspace).resolve()
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


class CheckpointStore:
    """Every checkpoint of every run, in one SQLite file. Rows are never updated: the history stays inspectable."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS checkpoints (
                              run_id TEXT NOT NULL, seq INTEGER NOT NULL, status TEXT NOT NULL,
                              state TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY (run_id, seq))""")

    @contextmanager
    def _db(self):
        """One connection per operation, always closed. Careful: `with sqlite3.connect(...)` alone only commits or
        rolls back the transaction; unlike Java's try-with-resources on a Connection, it does NOT close it."""
        conn = sqlite3.connect(self.path)
        try:
            with conn:  # the transaction: commit on success, roll back on an exception
                yield conn
        finally:
            conn.close()

    def save(self, state: dict) -> int:
        with self._db() as db:
            (seq,) = db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM checkpoints WHERE run_id = ?",
                                (state["run_id"],)).fetchone()
            db.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
                       (state["run_id"], seq, state["status"], json.dumps(state, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return seq

    def latest(self, run_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT state FROM checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
                             (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"no run {run_id!r}")
        return json.loads(row[0])

    def history(self, run_id: str) -> list[tuple[int, str]]:
        with self._db() as db:
            return db.execute("SELECT seq, status FROM checkpoints WHERE run_id = ? ORDER BY seq",
                              (run_id,)).fetchall()


class DurableRunner:
    def __init__(self, db_path, client=None):
        self.store = CheckpointStore(db_path)
        self.client = client  # None: a real OpenAI client per planner; tests pass a fake model

    # --- public ------------------------------------------------------------------------------------------------
    def start(self, workspace, max_steps=12, gate_patches=False, memory=None, memory_first=False,
              crash_after=None, run_id=None, issue=None) -> dict:
        """run_id: chosen by the caller when it must find the run again after a crash (a queued job uses its own id,
        lesson 14/29); issue: the ticket text for the model (lesson 14/30)."""
        workspace = Path(workspace).resolve()
        state = {
            "run_id": run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            "issue": issue,
            "status": "running", "reason": "", "workspace": str(workspace), "max_steps": max_steps,
            "gate_patches": gate_patches, "memory": str(memory) if memory else None, "memory_first": memory_first,
            "trace": [], "next_action": None, "fingerprints": fingerprints(workspace), "planner": None,
        }
        planner = self._planner(state)
        state["planner"] = planner.state()
        self.store.save(state)
        return self._loop(state, planner, crash_after)

    def resume(self, run_id, decision=None, reason="", crash_after=None) -> dict:
        state = self.store.latest(run_id)
        if state["status"] in FINISHED:
            return self._summary(state)
        now = fingerprints(state["workspace"])
        if now != state["fingerprints"]:
            changed = sorted(k for k in set(now) | set(state["fingerprints"])
                             if now.get(k) != state["fingerprints"].get(k))
            raise ResumeRefused(f"workspace changed since the last checkpoint: {', '.join(changed)}")
        if state["status"] == "awaiting_approval":
            if decision not in ("approve", "reject"):
                raise ValueError("this run is waiting for a patch decision: approve or reject")
            state["decision"] = {"decision": decision, "reason": reason}
            state["status"] = "running"
        planner = self._planner(state).restore(state["planner"])
        return self._loop(state, planner, crash_after)

    def status(self, run_id) -> dict:
        return {**self._summary(self.store.latest(run_id)), "checkpoints": self.store.history(run_id)}

    # --- the loop (main.drive with checkpoints) ------------------------------------------------------------------
    def _loop(self, state, planner, crash_after) -> dict:
        trace = state["trace"]
        while True:
            action = state["next_action"]
            if action is None:  # nothing saved from before: ask the planner
                action = planner(trace)
                if action is None:
                    return self._finish(state, planner, "failed", "planner abstained: no action fits the last observation")
                if len(trace) >= state["max_steps"]:
                    return self._finish(state, planner, "budget_exhausted", f"used all {state['max_steps']} steps")
                state["next_action"] = action
                self._checkpoint(state, planner)  # 1. intent, before acting

            decision = state.pop("decision", None)
            verdict = Guards(pve=True).before(action, trace) if decision is None else None  # a human decision overrides
            if verdict and verdict[0] == "stop":
                trace.append({**action, "result": {"error": verdict[1], "guard": "stop"}})
                state["next_action"] = None
                return self._finish(state, planner, "failed", f"stopped by a guard: {verdict[1]}")
            if verdict and verdict[0] == "refuse":  # refused before any human is asked: never worth a review
                trace.append({**action, "result": {"error": verdict[1], "guard": "refuse"}})
                state["next_action"] = None
                self._checkpoint(state, planner)
                continue
            if state["gate_patches"] and action["tool"] == "patch" and decision is None:
                state["status"] = "awaiting_approval"
                state["reason"] = "patch waiting for a human decision"
                self._checkpoint(state, planner)
                return self._summary(state)

            if decision and decision["decision"] == "reject":
                result = {"error": f"rejected by a human reviewer: {decision['reason'] or 'no reason given'}"}
            else:
                try:
                    result = _execute(state["workspace"], action)
                except ValueError as e:  # an invalid action is an observation for the model (on_invalid="observe")
                    result = {"error": str(e)}
            if decision:
                result["human_decision"] = decision["decision"]
            result = Guards(pve=True).after(action, result, trace)
            trace.append({**action, "result": result})
            state["next_action"] = None
            state["fingerprints"] = fingerprints(state["workspace"])
            if action["tool"] == "test" and result.get("passed"):
                return self._finish(state, planner, "completed", f"{result['tests']} tests passed")
            self._checkpoint(state, planner)  # 2. result, after acting
            if crash_after is not None and len(trace) >= crash_after:
                raise SimulatedCrash(f"simulated crash after step {len(trace)} (run {state['run_id']})")

    # --- helpers -------------------------------------------------------------------------------------------------
    def _planner(self, state) -> LlmPlanner:
        memory = RepairMemory(state["memory"]) if state["memory"] else None
        kwargs = {"client": self.client} if self.client else {}
        return LlmPlanner(state["workspace"], memory=memory, memory_first=state["memory_first"],
                          untrusted_labels=True, issue=state.get("issue"), **kwargs)  # .get: older checkpoints

    def _checkpoint(self, state, planner) -> None:
        state["planner"] = planner.state()
        self.store.save(state)

    def _finish(self, state, planner, status, reason) -> dict:
        state["status"], state["reason"] = status, reason
        self._checkpoint(state, planner)
        return self._summary(state)

    @staticmethod
    def _summary(state) -> dict:
        planner = state["planner"] or {}
        return {"run_id": state["run_id"], "status": state["status"], "reason": state["reason"],
                "steps": len(state["trace"]), "tools": [s["tool"] for s in state["trace"]],
                "awaiting": state["next_action"] if state["status"] == "awaiting_approval" else None,
                "model_calls": planner.get("model_calls", 0)}
