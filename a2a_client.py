"""A calling agent for a2a_server.py (lesson 13/19): a small "CI agent" that delegates repairs over A2A.

What it does, in A2A terms:
  1. discovery   GET /.well-known/agent-card.json, verify the card's signature, pick the JSONRPC interface, and pick
                 the skill by its tags (it never hard-codes the URL or the skill id)
  2. delegate    SendStreamingMessage with the workspace zip (raw part), the issue (text) and options (data)
  3. follow      read the Server-Sent Events: statusUpdate / artifactUpdate until the task ends or needs input
  4. answer      on TASK_STATE_INPUT_REQUIRED the server shows the exact patch it wants to apply; this agent reviews it
                 with its OWN rules (no edits under tests/, no added code that runs commands, reads secrets, opens
                 the network...) and answers approve or reject with the same taskId
  5. verify      opacity cuts both ways: the caller can't see how the result was made, so it never trusts the
                 report's "resolved". It applies the returned files to its own copy of the workspace and runs the
                 hidden tests itself (eval_suite.grade)

Usage:  python a2a_client.py card                                   show and verify the card
        python a2a_client.py eval --runs 1 --label a2a [--gate-patches] [--gate]
                                                                    every eval task through A2A, graded locally
Needs the server running (python a2a_server.py) and its token (state/a2a/secrets.json, or --token / A2A_TOKEN).
"""

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

from a2a_server import A2A_VERSION, verify_card, zip_folder
from guards import SUSPICIOUS_CODE

HERE = Path(__file__).resolve().parent
SECRETS = HERE / "state" / "a2a" / "secrets.json"


class A2AClient:
    def __init__(self, base_url: str, token: str, card_key: bytes | None = None, timeout: float = 60):
        self.base_url, self.token, self.card_key, self.timeout = base_url.rstrip("/"), token, card_key, timeout
        self.card, self.endpoint, self.skill = None, None, None

    # 1. discovery ------------------------------------------------------------------------------------------------
    def discover(self, needed_tags: set[str]) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/.well-known/agent-card.json", timeout=self.timeout) as r:
            card = json.load(r)
        if self.card_key is not None and not verify_card(card, self.card_key):
            raise RuntimeError("the agent card's signature does not verify: refusing to talk to this agent")
        interface = next((i for i in card["supportedInterfaces"]
                          if i["protocolBinding"] == "JSONRPC" and i["protocolVersion"] == A2A_VERSION), None)
        if interface is None:
            raise RuntimeError("the agent offers no JSONRPC 1.0 interface")
        skill = next((s for s in card["skills"] if needed_tags <= set(s.get("tags", []))), None)
        if skill is None:
            raise RuntimeError(f"no skill with tags {sorted(needed_tags)}")
        self.card, self.endpoint, self.skill = card, interface["url"], skill
        return card

    # transport -----------------------------------------------------------------------------------------------------
    def _post(self, method: str, params: dict):
        body = json.dumps({"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}).encode()
        return urllib.request.urlopen(urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json", "A2A-Version": A2A_VERSION,
                     "Authorization": f"Bearer {self.token}"}), timeout=self.timeout)

    def call(self, method: str, params: dict) -> dict:
        with self._post(method, params) as r:
            response = json.load(r)
        if "error" in response:
            raise RuntimeError(f"{method}: {response['error']['code']} {response['error']['message']}")
        return response["result"]

    def stream(self, method: str, params: dict):
        """Yields each event's result (task / statusUpdate / artifactUpdate) until the server closes the stream."""
        with self._post(method, params) as r:
            if not r.headers.get("Content-Type", "").startswith("text/event-stream"):
                response = json.load(r)
                raise RuntimeError(f"{method}: {response.get('error')}")
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    yield json.loads(line[len("data: "):])["result"]

    # 2-4. delegate, follow, answer -------------------------------------------------------------------------------
    def repair(self, workspace: Path, issue: str, gate_patches: bool = False, max_steps: int = 16,
               log=print) -> dict:
        message = {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [
            {"text": issue},
            {"raw": base64.b64encode(zip_folder(workspace)).decode(), "filename": "workspace.zip",
             "mediaType": "application/zip"},
            {"data": {"gate_patches": gate_patches, "max_steps": max_steps}, "mediaType": "application/json"}]}
        task, decisions = None, []
        events = self.stream("SendStreamingMessage", {"message": message})
        while True:
            for event in events:
                if "task" in event:
                    task = event["task"]
                elif "statusUpdate" in event:
                    task["status"] = event["statusUpdate"]["status"]
                elif "artifactUpdate" in event:
                    task.setdefault("artifacts", []).append(event["artifactUpdate"]["artifact"])
                log(f"    {task['id']}: {task['status']['state']}")
            if task["status"]["state"] != "TASK_STATE_INPUT_REQUIRED":
                return {"task": task, "decisions": decisions}
            patch = next(p["data"]["patch"] for p in task["status"]["message"]["parts"] if "data" in p)
            decision, reason = review_patch(patch)
            decisions.append({"path": patch["path"], "decision": decision, "reason": reason})
            log(f"    review {patch['path']}: {decision}{' (' + reason + ')' if reason else ''}")
            answer = {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "taskId": task["id"],
                      "parts": [{"data": {"decision": decision, "reason": reason}, "mediaType": "application/json"}]}
            events = self.stream("SendStreamingMessage", {"message": answer})


def review_patch(patch: dict) -> tuple[str, str]:
    """The calling agent's own policy: it doesn't rely on the server's guards, which it can't see."""
    if PurePosixPath(patch["path"]).parts[:1] == ("tests",):
        return "reject", "do not change the tests; fix the code"
    for pattern, why in SUSPICIOUS_CODE.items():
        if re.search(pattern, patch["new"], re.IGNORECASE) and not re.search(pattern, patch["old"], re.IGNORECASE):
            return "reject", f"the patch adds code that {why}"
    return "approve", ""


def artifact(task: dict, name: str) -> dict:
    return next(a for a in task.get("artifacts", []) if a["name"] == name)["parts"][0]


# 5. verify -------------------------------------------------------------------------------------------------------
def verify_with_hidden_tests(task_def: dict, result_task: dict) -> dict:
    """Write the returned files into a fresh copy of the task's workspace, then grade it with the hidden tests."""
    from eval_suite import fresh_workspace, grade
    files = artifact(result_task, "changed_files")["data"]["files"]
    with tempfile.TemporaryDirectory() as tmp:
        ws = fresh_workspace(task_def, Path(tmp))
        for path, text in files.items():
            target = (ws / path).resolve()
            if not target.is_relative_to(ws.resolve()):  # a returned path is data from another agent: confine it
                raise RuntimeError(f"returned file escapes the workspace: {path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return grade(task_def, ws)


def run_eval(client: A2AClient, runs: int, label: str, gate_patches: bool, tasks=None, log=print) -> list[dict]:
    from eval_suite import gold_steps, load_tasks
    results = []
    for task_def in load_tasks(tasks):
        for i in range(1, runs + 1):
            log(f"{task_def['id']} run {i}")
            out = client.repair(task_def["dir"] / "workspace", task_def["issue"], gate_patches=gate_patches, log=log)
            task = out["task"]
            if task["status"]["state"] != "TASK_STATE_COMPLETED":
                graded = {"resolved": False, "overfit": False, "tests_edited": False, "out_of_scope": [],
                          "changed_files": []}
                report = {"steps": 0, "cost_usd": 0.0, "outcome": task["status"]["state"]}
            else:
                graded = verify_with_hidden_tests(task_def, task)
                report = artifact(task, "report")["data"]
            results.append({
                "task": task_def["id"], "run_id": f"{label}-{task_def['id']}-{i}", "a2a_task": task["id"],
                "final_state": task["status"]["state"], "claimed": report["outcome"], **graded,
                "claim_matches": (report["outcome"] == "resolved") == graded["resolved"],
                "decisions": out["decisions"], "refusals": [],
                "steps": report["steps"], "gold_steps": gold_steps(task_def), "cost_usd": report["cost_usd"]})
            log(f"  claimed {report['outcome']}, verified {'RESOLVED' if graded['resolved'] else 'NOT resolved'}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--token", default=os.environ.get("A2A_TOKEN"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("card")
    ev = sub.add_parser("eval")
    ev.add_argument("--runs", type=int, default=1)
    ev.add_argument("--label", default="a2a")
    ev.add_argument("--tasks")
    ev.add_argument("--gate-patches", action="store_true", help="ask the server to pause before every patch")
    ev.add_argument("--gate", action="store_true", help="compare with evals/baseline.json")
    args = parser.parse_args()
    local = json.loads(SECRETS.read_text()) if SECRETS.exists() else {}
    client = A2AClient(args.url, args.token or local.get("token", ""),
                       bytes.fromhex(local["card_key"]) if "card_key" in local else None)
    card = client.discover({"coding", "repair"})
    print(f"card ok: {card['name']} {card['version']}, signature {'verified' if client.card_key else 'NOT checked'}; "
          f"skill {client.skill['id']} at {client.endpoint}")
    if args.command == "card":
        print(json.dumps(card, indent=2, ensure_ascii=False))
        return 0

    from eval_suite import BASELINE, gate, load_tasks, report, summarize
    results = run_eval(client, args.runs, args.label, args.gate_patches, args.tasks.split(",") if args.tasks else None)
    summary = summarize(results)
    text = report(summary, args.label, "a2a", load_tasks())
    mismatches = [r for r in results if not r["claim_matches"]]
    approvals = sum(len(r["decisions"]) for r in results)
    text += (f"\n\nthrough A2A: {sum(r['final_state'] == 'TASK_STATE_COMPLETED' for r in results)}/{len(results)} "
             f"tasks COMPLETED; the server's claim matched the hidden-test check in "
             f"{len(results) - len(mismatches)}/{len(results)}; patch decisions: {approvals}")
    ok = True
    if args.gate:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        ok, reasons = gate(summary, base)
        text += f"\ngate vs baseline '{base['label']}': " + ("PASS" if ok else "FAIL\n  " + "\n  ".join(reasons))
    print("\n" + text)
    out = HERE / "results" / f"a2a-{args.label}"
    out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
