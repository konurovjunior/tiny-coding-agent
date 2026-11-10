"""The agent as an A2A agent (lesson 13/19): other agents delegate a whole repair to it, over JSON-RPC 2.0 + HTTP.

MCP (mcp_server.py, 13/06) exposes the agent's TOOLS: the caller decides every read, patch and test run.
A2A exposes the AGENT: the caller sends a goal ("these tests fail, fix it") and gets back task states and artifacts.
How the agent works inside (its model calls, its tool calls, its trace) stays opaque to the caller.

Wire names follow A2A 1.0.1, as in the course's lesson harness:
  GET  /.well-known/agent-card.json   the Agent Card, signed (a JWS over the card's canonical JSON, HS256)
  POST /a2a                           JSON-RPC 2.0; every request needs the header A2A-Version: 1.0 (without it a
                                      server must read the request as v0.3, so it is refused: -32009) and
                                      Authorization: Bearer <token> (the card's securitySchemes)
     SendMessage            new task (a zip of the workspace + the issue text), or the answer to an input request
     SendStreamingMessage   the same, then Server-Sent Events until the task ends or needs input
     SubscribeToTask        the event stream of an existing task
     GetTask, CancelTask

The task is our existing machinery seen from outside:
  A2A                         this agent
  TASK_STATE_SUBMITTED        a queued job (job_queue.py)
  TASK_STATE_WORKING          a running job: a durable run (durable.py) with the guards, checkpoints and retries
  TASK_STATE_INPUT_REQUIRED   the durable run's approval gate: a patch waits for a decision (14/13)
  TASK_STATE_COMPLETED        the run ended; the 'report' artifact says resolved or unresolved (an answer, not a fault)
  TASK_STATE_FAILED           the job ended in the dead-letter queue (14/29)
  TASK_STATE_REJECTED         refused before any work: no workspace, an unsafe zip, no tests/ folder
  TASK_STATE_CANCELED         canceled while queued or while waiting for input (a running step can't be interrupted)

Artifacts (what the caller gets; not the trace): 'patch' (text/x-diff), 'changed_files' (data: path -> new content),
'report' (data: outcome, steps, model calls, cost). A caller must not trust them blindly: see a2a_client.py.

Usage:  python a2a_server.py [--port 8765] [--workers 2] [--trace langfuse]
State in state/a2a/ (git-ignored): tasks, the queue, checkpoints, workspaces, and secrets.json (token + card key).
"""

import argparse
import base64
import difflib
import hashlib
import hmac
import io
import json
import secrets
import shutil
import sqlite3
import stat
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

import telemetry
from durable import DurableRunner
from job_queue import JobQueue, Worker, job_trace, run_cost

HERE = Path(__file__).resolve().parent
A2A_VERSION = "1.0"
PART_FIELDS = ("text", "raw", "url", "data")
MAX_ZIP_BYTES, MAX_FILES, MAX_UNPACKED = 2_000_000, 200, 5_000_000
MAX_STEPS_CAP = 30
TERMINAL = {"TASK_STATE_COMPLETED", "TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}
INTERRUPTED = {"TASK_STATE_INPUT_REQUIRED"}

# JSON-RPC errors: the standard ones and A2A's own (as named in the A2A specification)
INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32600, -32601, -32602
TASK_NOT_FOUND, TASK_NOT_CANCELABLE, UNSUPPORTED_OPERATION, CONTENT_TYPE_NOT_SUPPORTED = -32001, -32002, -32004, -32005
VERSION_NOT_SUPPORTED = -32009


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------------------------- card and signing

def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def canonical(card: dict) -> bytes:
    """RFC 8785-style canonical JSON of the card without 'signatures': sorted keys, no spaces. Exact for this card,
    which holds only strings, booleans, lists and objects (RFC 8785's number rules never come into play)."""
    body = {k: v for k, v in card.items() if k != "signatures"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign_card(card: dict, key: bytes) -> dict:
    """One JWS (RFC 7515) with a detached payload: {protected, signature}, over protected.payload."""
    protected = b64url(json.dumps({"alg": "HS256", "typ": "JOSE"}, separators=(",", ":")).encode())
    signing_input = f"{protected}.{b64url(canonical(card))}".encode()
    signature = b64url(hmac.new(key, signing_input, hashlib.sha256).digest())
    return {**card, "signatures": [{"protected": protected, "signature": signature}]}


def verify_card(card: dict, key: bytes) -> bool:
    for sig in card.get("signatures", []):
        header = json.loads(base64.urlsafe_b64decode(sig["protected"] + "=" * (-len(sig["protected"]) % 4)))
        if header.get("alg") != "HS256":  # never let the card choose a weaker algorithm (e.g. "none")
            continue
        signing_input = f"{sig['protected']}.{b64url(canonical(card))}".encode()
        expected = b64url(hmac.new(key, signing_input, hashlib.sha256).digest())
        if hmac.compare_digest(expected, sig["signature"]):
            return True
    return False


def agent_card(url: str) -> dict:
    return {
        "name": "tiny-coding-agent",
        "description": "Repairs a small Python workspace whose unit tests fail. Edits code, never tests. "
                       "Returns the patch, the changed files and a report; may ask for approval before a patch.",
        "version": "0.7.0",
        "supportedInterfaces": [{"url": url, "protocolBinding": "JSONRPC", "protocolVersion": A2A_VERSION}],
        "capabilities": {"streaming": True, "pushNotifications": False},
        "securitySchemes": {"bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
        "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
        "defaultInputModes": ["application/zip", "text/plain", "application/json"],
        "defaultOutputModes": ["text/x-diff", "application/json"],
        "skills": [{
            "id": "repair_failing_tests",
            "name": "Repair failing tests",
            "description": (
                "Use when a Python workspace (zipped, with a tests/ folder run by unittest) has failing tests and you "
                "want the bug in the code fixed. Send the zip as a raw part (application/zip) and the issue as text; "
                "optional data part {\"gate_patches\": true, \"max_steps\": 16} pauses before every patch "
                "(TASK_STATE_INPUT_REQUIRED; answer with data {\"decision\": \"approve\"|\"reject\", \"reason\": ...}). "
                "Do not use to write new features, change tests, or run arbitrary commands; URLs are not fetched."),
            "tags": ["coding", "repair", "python", "tests"],
            "inputModes": ["application/zip", "text/plain", "application/json"],
            "outputModes": ["text/x-diff", "application/json"],
        }],
    }


# ------------------------------------------------------------------------------------------------- parts and zips

def part_kind(part: dict) -> str:
    present = [f for f in PART_FIELDS if f in part]
    if len(present) != 1:
        raise RpcError(INVALID_PARAMS, f"a Part holds exactly one of text, raw, url, data; got {present}")
    return present[0]


def unpack_workspace(data: bytes, dest: Path) -> None:
    """Extract a zip only after checking every entry: no absolute paths, no '..', no symlinks, no zip bombs.
    The caller never chooses where files land (the same rule as safe_path)."""
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"zip is {len(data)} bytes, limit {MAX_ZIP_BYTES}")
    dest.mkdir(parents=True)
    root = dest.resolve()
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("not a zip file")
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_UNPACKED:
            raise ValueError("zip has too many files or unpacks too large")
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if info.filename.startswith(("/", "\\")) or ".." in parts or ":" in info.filename:
                raise ValueError(f"unsafe path in zip: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"symlink in zip: {info.filename}")
            if not (root / info.filename).resolve().is_relative_to(root):
                raise ValueError(f"path escapes the workspace: {info.filename}")
        archive.extractall(root)
    if not (root / "tests").is_dir():
        raise ValueError("the workspace has no tests/ folder")


def zip_folder(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(folder).as_posix())
    return buffer.getvalue()


def text_files(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


# ------------------------------------------------------------------------------------------------------ the service

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class A2AService:
    """The protocol logic, without HTTP: handle(headers, request) -> response, events(task_id) -> stream.
    Tasks live in their own table; their work runs as jobs on the queue (batch = task id)."""

    def __init__(self, state_dir: Path, client=None, card_url: str = "http://localhost:8765/a2a",
                 poll_s: float = 0.3):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.queue = JobQueue(self.dir / "jobs.sqlite")
        self.runner = DurableRunner(self.dir / "runs.sqlite", client=client)
        self.poll_s = poll_s
        secrets_file = self.dir / "secrets.json"
        if not secrets_file.exists():
            secrets_file.write_text(json.dumps({"token": secrets.token_hex(16), "card_key": secrets.token_hex(32)}))
            secrets_file.chmod(0o600)
        s = json.loads(secrets_file.read_text())
        self.token, self.card_key = s["token"], bytes.fromhex(s["card_key"])
        self.card = sign_card(agent_card(card_url), self.card_key)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                              id TEXT PRIMARY KEY, context_id TEXT NOT NULL, created TEXT NOT NULL,
                              rejected TEXT, canceled INTEGER NOT NULL DEFAULT 0, history TEXT NOT NULL)""")
        self.stop = threading.Event()

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.dir / "tasks.sqlite", timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # --- workers --------------------------------------------------------------------------------------------------
    def start_workers(self, n: int) -> list[threading.Thread]:
        threads = []
        for i in range(n):
            w = Worker(self.queue, {"a2a": self.handle_job}, worker_id=f"a2a-w{i + 1}", runner=self.runner)
            t = threading.Thread(target=w.run, kwargs={"idle_exit": False, "idle_sleep": self.poll_s,
                                                       "stop": self.stop}, daemon=True)
            t.start()
            threads.append(t)
        return threads

    def handle_job(self, job: dict, worker: Worker) -> dict:
        """One step of a task: start its durable run, continue it after a decision, or resume it after a crash."""
        p, task_id = job["payload"], job["payload"]["task_id"]
        with job_trace(job, f"a2a {task_id}", p.get("issue")) as run:
            try:
                state = self.runner.store.latest(task_id)
            except KeyError:
                state = None
            if state is None:
                summary = self.runner.start(p["workspace"], max_steps=p["max_steps"], gate_patches=p["gate"],
                                            run_id=task_id, issue=p.get("issue"), instrument=run.instrument)
            elif state["status"] == "awaiting_approval" and not p.get("decision"):
                summary = self.runner.status(task_id)  # a retried job after the gate was reached: nothing to do
            else:
                summary = self.runner.resume(task_id, p.get("decision"), p.get("reason", ""),
                                             instrument=run.instrument)
            _, summary["cost_usd"] = run_cost(self.runner, task_id)
            run.finish({k: summary[k] for k in ("status", "reason", "steps")})
        return summary

    # --- JSON-RPC ---------------------------------------------------------------------------------------------------
    def check_headers(self, headers) -> None:
        if headers.get("A2A-Version") != A2A_VERSION:
            raise RpcError(VERSION_NOT_SUPPORTED, f"Version not supported: send the header A2A-Version: {A2A_VERSION}")

    def authorized(self, headers) -> bool:
        return hmac.compare_digest(headers.get("Authorization", ""), f"Bearer {self.token}")

    def handle(self, headers, request: dict) -> dict:
        response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None}
        try:
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or "method" not in request:
                raise RpcError(INVALID_REQUEST, "not a JSON-RPC 2.0 request")
            self.check_headers(headers)
            method, params = request["method"], request.get("params") or {}
            if method == "SendMessage":
                response["result"] = self.send_message(params)
            elif method == "GetTask":
                response["result"] = self.get_task(params)
            elif method == "CancelTask":
                response["result"] = self.cancel_task(params)
            elif method in ("ListTasks", "GetExtendedAgentCard") or "PushNotificationConfig" in method:
                raise RpcError(UNSUPPORTED_OPERATION, f"{method} is not supported by this agent")
            else:
                raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")
        except RpcError as e:
            response.pop("result", None)
            response["error"] = {"code": e.code, "message": str(e)}
        return response

    def send_message(self, params: dict) -> dict:
        message = params.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("parts"), list) or not message["parts"]:
            raise RpcError(INVALID_PARAMS, "params.message with at least one part is required")
        kinds = [part_kind(p) for p in message["parts"]]
        if "url" in kinds:  # fetching caller-supplied URLs would make this server an SSRF tool
            raise RpcError(CONTENT_TYPE_NOT_SUPPORTED, "url parts are not fetched; send the workspace as a raw zip")
        if message.get("taskId"):
            return {"task": self.answer(message)}
        return {"task": self.create(message)}

    def create(self, message: dict) -> dict:
        task_id, context_id = "task-" + uuid.uuid4().hex[:12], message.get("contextId") or str(uuid.uuid4())
        texts = [p["text"] for p in message["parts"] if "text" in p]
        zips = [p for p in message["parts"] if "raw" in p]
        config = {}
        for p in message["parts"]:
            if "data" in p and isinstance(p["data"], dict):
                config.update(p["data"])
        workspace = self.dir / "tasks" / task_id / "workspace"
        rejected = None
        try:
            if len(zips) != 1 or zips[0].get("mediaType", "application/zip") != "application/zip":
                raise ValueError("send exactly one raw part with mediaType application/zip (the workspace)")
            unpack_workspace(base64.b64decode(zips[0]["raw"], validate=True), workspace)
            shutil.copytree(workspace, workspace.parent / "original")  # to diff against when the task completes
        except (ValueError, base64.binascii.Error) as e:
            rejected = str(e)
        stored = {k: v for k, v in message.items() if k != "parts"}  # history keeps the message, not the zip bytes
        stored["parts"] = [{"raw": f"<{len(p['raw'])} base64 chars>", **{k: v for k, v in p.items() if k != "raw"}}
                           if "raw" in p else p for p in message["parts"]]
        with self._db() as db:
            db.execute("INSERT INTO tasks (id, context_id, created, rejected, history) VALUES (?, ?, ?, ?, ?)",
                       (task_id, context_id, now_iso(), rejected, json.dumps([stored], ensure_ascii=False)))
        if rejected is None:
            max_steps = max(1, min(int(config.get("max_steps", 16)), MAX_STEPS_CAP))
            self.queue.enqueue("a2a", {"task_id": task_id, "workspace": str(workspace), "issue": "\n".join(texts),
                                       "gate": bool(config.get("gate_patches", False)), "max_steps": max_steps},
                               batch=task_id)
        return self.task(task_id)

    def answer(self, message: dict) -> dict:
        """A follow-up with the same taskId: the decision on a patch the task is waiting on."""
        task = self.task(message["taskId"])
        if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
            raise RpcError(INVALID_PARAMS, f"task is {task['status']['state']}, not waiting for input")
        decision, reason = None, ""
        for p in message["parts"]:
            if "data" in p and isinstance(p["data"], dict):
                decision, reason = p["data"].get("decision"), p["data"].get("reason", "")
            elif "text" in p and decision is None:
                word, _, rest = p["text"].strip().partition(":")
                decision, reason = word.strip().lower(), rest.strip()
        if decision not in ("approve", "reject"):
            raise RpcError(INVALID_PARAMS, 'answer with data {"decision": "approve"|"reject", "reason": "..."}')
        with self._db() as db:
            (history,) = db.execute("SELECT history FROM tasks WHERE id = ?", (task["id"],)).fetchone()
            db.execute("UPDATE tasks SET history = ? WHERE id = ?",
                       (json.dumps(json.loads(history) + [message], ensure_ascii=False), task["id"]))
        last = self.queue.jobs(task["id"])[-1]
        self.queue.enqueue("a2a", {**last["payload"], "decision": decision, "reason": reason}, batch=task["id"])
        return self.task(task["id"])

    def get_task(self, params: dict) -> dict:
        return self.task(params.get("id", ""))

    def cancel_task(self, params: dict) -> dict:
        task = self.task(params.get("id", ""))
        state = task["status"]["state"]
        if state in TERMINAL:
            raise RpcError(TASK_NOT_CANCELABLE, f"task is already {state}")
        last = self.queue.jobs(task["id"])[-1]
        if state == "TASK_STATE_INPUT_REQUIRED" or self.queue.cancel(last["id"]):
            with self._db() as db:
                db.execute("UPDATE tasks SET canceled = 1 WHERE id = ?", (task["id"],))
            return self.task(task["id"])
        raise RpcError(TASK_NOT_CANCELABLE, "the task is working on a step; it can be canceled while queued or "
                                            "while waiting for input")

    # --- the task as A2A sees it --------------------------------------------------------------------------------
    def task(self, task_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT context_id, rejected, canceled, history FROM tasks WHERE id = ?",
                             (task_id,)).fetchone()
        if row is None:
            raise RpcError(TASK_NOT_FOUND, f"Task not found: {task_id}")
        context_id, rejected, canceled, history = row
        task = {"id": task_id, "contextId": context_id, "artifacts": [], "history": json.loads(history)}
        if rejected:
            return {**task, "status": self._status("TASK_STATE_REJECTED", task, rejected)}
        if canceled:
            return {**task, "status": self._status("TASK_STATE_CANCELED", task, "canceled by the caller")}
        jobs = self.queue.jobs(task_id)
        last = jobs[-1]
        if last["status"] == "queued":
            state = "TASK_STATE_SUBMITTED" if len(jobs) == 1 and last["attempts"] == 0 else "TASK_STATE_WORKING"
            return {**task, "status": self._status(state, task)}
        if last["status"] == "running":
            return {**task, "status": self._status("TASK_STATE_WORKING", task)}
        if last["status"] == "dead":
            return {**task, "status": self._status("TASK_STATE_FAILED", task, last["last_error"])}
        result = last["result"]
        if result["status"] == "awaiting_approval":
            patch = result["awaiting"]
            ask = [{"text": f"Approve this patch to {patch['path']}? Answer with data "
                            '{"decision": "approve"|"reject", "reason": "..."} and the same taskId.'},
                   {"data": {"patch": {k: patch.get(k) for k in ("path", "old", "new")}},
                    "mediaType": "application/json"}]
            return {**task, "status": self._status("TASK_STATE_INPUT_REQUIRED", task, parts=ask)}
        return {**task, "status": self._status("TASK_STATE_COMPLETED", task,
                                               "resolved" if result["status"] == "completed" else "unresolved"),
                "artifacts": self.artifacts(task_id, result)}

    @staticmethod
    def _status(state: str, task: dict, text: str | None = None, parts: list | None = None) -> dict:
        status = {"state": state, "timestamp": now_iso()}
        if text or parts:
            status["message"] = {"messageId": f"{task['id']}-{state}", "role": "ROLE_AGENT", "taskId": task["id"],
                                 "contextId": task["contextId"], "parts": parts or [{"text": text}]}
        return status

    def artifacts(self, task_id: str, result: dict) -> list[dict]:
        folder = self.dir / "tasks" / task_id
        before, after = text_files(folder / "original"), text_files(folder / "workspace")
        changed = {path: text for path, text in after.items() if before.get(path) != text}
        diff = "".join(line for path in sorted(changed) for line in difflib.unified_diff(
            before.get(path, "").splitlines(keepends=True), changed[path].splitlines(keepends=True),
            f"a/{path}", f"b/{path}"))
        outcome = "resolved" if result["status"] == "completed" else "unresolved"
        return [
            {"artifactId": f"{task_id}-patch", "name": "patch", "parts": [{"text": diff, "mediaType": "text/x-diff"}]},
            {"artifactId": f"{task_id}-files", "name": "changed_files",
             "parts": [{"data": {"files": changed}, "mediaType": "application/json"}]},
            {"artifactId": f"{task_id}-report", "name": "report", "parts": [{"data": {
                "outcome": outcome, "reason": result["reason"], "steps": result["steps"],
                "model_calls": result["model_calls"], "cost_usd": round(result.get("cost_usd", 0.0), 6)},
                "mediaType": "application/json"}]},
        ]

    # --- streaming --------------------------------------------------------------------------------------------------
    def events(self, task_id: str, timeout_s: float = 600):
        """StreamResponse payloads for one task: the task first, then a statusUpdate on every state change, the
        artifacts as artifactUpdate events when it completes. Ends at a terminal state, or when the task needs input
        (the caller answers with SendStreamingMessage, which opens a new stream)."""
        task = self.task(task_id)
        yield {"task": task}
        last_state, deadline = task["status"]["state"], time.monotonic() + timeout_s
        while last_state not in TERMINAL | INTERRUPTED and time.monotonic() < deadline:
            time.sleep(self.poll_s)
            task = self.task(task_id)
            if task["status"]["state"] == last_state:
                continue
            last_state = task["status"]["state"]
            for artifact in task["artifacts"]:
                yield {"artifactUpdate": {"taskId": task_id, "contextId": task["contextId"], "artifact": artifact,
                                          "append": False, "lastChunk": True}}
            yield {"statusUpdate": {"taskId": task_id, "contextId": task["contextId"], "status": task["status"]}}


# ---------------------------------------------------------------------------------------------------------- HTTP

def make_handler(service: A2AService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet: the tasks table and Langfuse are the record
            pass

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/.well-known/agent-card.json":
                self._json(200, service.card)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/a2a":
                return self._json(404, {"error": "not found"})
            if not service.authorized(self.headers):
                self.send_response(401)
                self.send_header("WWW-Authenticate", "Bearer")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_ZIP_BYTES * 2:
                return self._json(413, {"error": "request too large"})
            try:
                request = json.loads(self.rfile.read(length) or b"null")
            except json.JSONDecodeError:
                return self._json(200, {"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32700, "message": "Parse error"}})
            method = request.get("method") if isinstance(request, dict) else None
            if method not in ("SendStreamingMessage", "SubscribeToTask"):
                return self._json(200, service.handle(self.headers, request))
            self._stream(request)

        def _stream(self, request: dict):
            rid = request.get("id")
            try:
                service.check_headers(self.headers)
                params = request.get("params") or {}
                if request["method"] == "SendStreamingMessage":
                    task_id = service.send_message(params)["task"]["id"]
                else:
                    task_id = service.task(params.get("id", ""))["id"]
            except RpcError as e:
                return self._json(200, {"jsonrpc": "2.0", "id": rid, "error": {"code": e.code, "message": str(e)}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for event in service.events(task_id):
                frame = json.dumps({"jsonrpc": "2.0", "id": rid, "result": event}, ensure_ascii=False)
                self.wfile.write(f"data: {frame}\n\n".encode())
                self.wfile.flush()

    return Handler


def serve(service: A2AService, host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--state", type=Path, default=HERE / "state" / "a2a")
    parser.add_argument("--trace", choices=["off", "langfuse"], default="off")
    args = parser.parse_args()
    telemetry.setup(args.trace)
    service = A2AService(args.state, card_url=f"http://{args.host}:{args.port}/a2a")
    service.start_workers(args.workers)
    server = serve(service, args.host, args.port)
    print(f"A2A agent on http://{args.host}:{args.port} (card: /.well-known/agent-card.json); "
          f"token in {args.state / 'secrets.json'}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop.set()
        telemetry.flush()


if __name__ == "__main__":
    main()
