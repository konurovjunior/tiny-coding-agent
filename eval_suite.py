"""An eval suite for the agent (lesson 14/30): varied tasks, hidden tests, k runs each, a baseline and a gate.

The Orchard bug is one line, so 5/5 on it proves the machinery, not the agent. This suite asks harder questions, in
the SWE-bench shape:

  evals/tasks/<id>/workspace/   what the agent sees: the code and the VISIBLE failing tests
  evals/tasks/<id>/hidden/      what it never sees, copied in only after the run ends:
      fail_to_pass/  tests that fail before the fix and must pass after it (does the fix generalise?)
      pass_to_pass/  tests that pass before and must still pass (did the fix break something else?)
  evals/tasks/<id>/task.json    the issue text (the ticket), which files a fix may touch, the gold fix, and decoys:
                                known WRONG fixes that pass the visible tests (half-fixes, symptom fixes, overfits)

A run is RESOLVED only if the visible tests pass, both hidden sets pass, and tests/ is unchanged. A run whose visible
tests pass but whose hidden tests fail is OVERFIT: it would have looked like a success in every earlier lesson.

Each task runs k times, because the model samples: pass@1 (share of runs resolved), pass^k (all k resolved: what
autonomy needs) and pass@k (at least one: what a human picking the best attempt gets). Rates come with 95% Wilson
intervals, so 9 tasks x 3 runs is read as what it is: a regression alarm, not a fine ruler.

Usage (from this folder):
  python eval_suite.py --validate                     free: every task is a real test (unpatched fails, gold
                                                      resolves, every decoy is caught by the hidden tests)
  python eval_suite.py --runs 3 --label luna          live: results/eval-<label>.txt and .jsonl
  python eval_suite.py --runs 3 --model gpt-5.4-mini  another arm, same tasks
  python eval_suite.py --runs 3 --no-issue --gate     ablation: no ticket text, only the visible tests
  python eval_suite.py --runs 3 --save-baseline       store this result as evals/baseline.json (last known good)
  python eval_suite.py --runs 3 --gate                compare with the baseline; exit code 1 on a regression
"""

import argparse
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from guards import Guards, validate_patch
from llm_planner import MODEL, LlmPlanner
from main import apply_patch, drive, run_tests

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "evals" / "tasks"
BASELINE = HERE / "evals" / "baseline.json"
TEST_TIMEOUT = 5            # seconds per test run, as in the agent's own run_tests
MAX_STEPS = 16              # more than Orchard's 12: two-bug and two-file tasks need extra reads and patches
GATE_RATE_DROP = 0.10       # FAIL if the overall resolve rate drops by more than 10 points...
GATE_TASK_DROP = 2          # ...or any task loses 2 or more of its k successes compared with the baseline


# ---------------------------------------------------------------------------------------------------------- tasks

def load_tasks(names: list[str] | None = None) -> list[dict]:
    tasks = []
    for folder in sorted(p for p in TASKS_DIR.iterdir() if (p / "task.json").is_file()):
        if names and folder.name not in names:
            continue
        task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        tasks.append({**task, "id": folder.name, "dir": folder})
    if names and len(tasks) != len(names):
        raise SystemExit(f"unknown task(s): {sorted(set(names) - {t['id'] for t in tasks})}")
    return tasks


def files_of(folder: Path) -> dict[str, str]:
    """Every text file under folder, by relative path (bytecode caches excluded)."""
    return {str(p.relative_to(folder)): p.read_text(encoding="utf-8")
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def fresh_workspace(task: dict, parent: Path) -> Path:
    ws = parent / "workspace"
    shutil.copytree(task["dir"] / "workspace", ws, ignore=shutil.ignore_patterns("__pycache__"))
    return ws


def gold_steps(task: dict) -> int:
    """The shortest sensible trajectory: run the tests, read each file the fix touches, patch, run the tests."""
    return 2 + len({p["path"] for p in task["gold"]}) + len(task["gold"])


# -------------------------------------------------------------------------------------------------------- grading

def run_hidden(ws: Path, hidden: Path) -> dict:
    """Run one hidden test set against the agent's final code: a copy of the workspace whose tests/ holds only the
    hidden tests. The agent's workspace itself is never touched, and it never saw these files."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "graded"
        shutil.copytree(ws, target, ignore=shutil.ignore_patterns("__pycache__", "tests"))
        shutil.copytree(hidden, target / "tests")
        result = run_tests(target, timeout=TEST_TIMEOUT)
    return {"passed": result["passed"], "tests": result["tests"], "state": result["state"],
            "output": result["output"][-1500:]}


def grade(task: dict, ws: Path) -> dict:
    original = files_of(task["dir"] / "workspace")
    final = files_of(ws)
    tests_edited = {k: v for k, v in original.items() if k.startswith("tests/")} != \
                   {k: v for k, v in final.items() if k.startswith("tests/")}
    changed = sorted(k for k in final if not k.startswith("tests/") and final[k] != original.get(k))
    visible = run_tests(ws, timeout=TEST_TIMEOUT)
    f2p = run_hidden(ws, task["dir"] / "hidden" / "fail_to_pass")
    p2p = run_hidden(ws, task["dir"] / "hidden" / "pass_to_pass")
    resolved = visible["passed"] and f2p["passed"] and p2p["passed"] and not tests_edited
    return {
        "resolved": resolved,
        "visible_passed": visible["passed"],
        "fail_to_pass": f2p["passed"], "pass_to_pass": p2p["passed"],
        "overfit": visible["passed"] and not tests_edited and not (f2p["passed"] and p2p["passed"]),
        "tests_edited": tests_edited,
        "changed_files": changed,
        "out_of_scope": [f for f in changed if f not in task["allowed"]],
        "diff": {f: final[f] for f in changed},
        "hidden_output": {"fail_to_pass": f2p["output"] if not f2p["passed"] else "",
                          "pass_to_pass": p2p["output"] if not p2p["passed"] else ""},
    }


def validate_task(task: dict) -> list[str]:
    """SWE-bench's own check that a task is a fair question, free and deterministic:
    unpatched, the visible and fail_to_pass tests fail and pass_to_pass passes; the gold fix resolves it;
    every decoy passes the visible tests and is still caught by the hidden ones."""
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        g = grade(task, fresh_workspace(task, Path(tmp)))
        if g["visible_passed"]:
            problems.append("unpatched: visible tests already pass")
        if g["fail_to_pass"]:
            problems.append("unpatched: fail_to_pass already passes (it tests nothing new)")
        if not g["pass_to_pass"]:
            problems.append("unpatched: pass_to_pass fails (it must pass before the fix)")
    for name, patches in [("gold", task["gold"])] + [(f"decoy {i}", d["patches"])
                                                    for i, d in enumerate(task["decoys"], 1)]:
        with tempfile.TemporaryDirectory() as tmp:
            ws = fresh_workspace(task, Path(tmp))
            for p in patches:
                apply_patch(ws, p["path"], p["old"], p["new"])
            g = grade(task, ws)
        if name == "gold":  # a fair task: the guards that are on in every live run must let the right fix through
            read = [{"tool": "read", "path": p["path"], "result": {"text": ""}} for p in patches]
            for p in patches:
                verdict = validate_patch({"tool": "patch", **p}, read)
                if verdict:
                    problems.append(f"the PVE validator refuses the gold fix: {verdict[1]}")
        if name == "gold" and not g["resolved"]:
            problems.append(f"gold fix does not resolve the task: {g['hidden_output']}")
        if name != "gold" and not g["visible_passed"]:
            problems.append(f"{name} fails the visible tests, so it isn't a trap")
        if name != "gold" and g["resolved"]:
            problems.append(f"{name} is not caught by the hidden tests")
        if g["out_of_scope"]:
            problems.append(f"{name} touches files outside 'allowed': {g['out_of_scope']}")
    return problems


# ----------------------------------------------------------------------------------------------------------- runs

def run_agent(task: dict, run_id: str, model: str = MODEL, max_steps: int = MAX_STEPS, client=None,
              with_issue: bool = True) -> dict:
    """One live attempt: a fresh copy, the model planner with the issue text, the loop with all guards on
    (scope, repeats, re-reads, PVE, untrusted labels), then grading with the hidden tests."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = fresh_workspace(task, Path(tmp))
        planner = LlmPlanner(ws, client=client, model=model, untrusted_labels=True,
                               issue=task["issue"] if with_issue else None)
        result = drive(ws, planner, max_steps=max_steps, on_invalid="observe", guards=Guards(pve=True))
        graded = grade(task, ws)
    trace = result["trace"]
    return {
        "task": task["id"], "run_id": run_id, "model": model,
        "state": result["state"], "reason": result["reason"], **graded,
        "steps": result["steps"], "gold_steps": gold_steps(task),
        "tools": [s["tool"] for s in trace],
        "refusals": [s["result"]["error"] for s in trace if s["result"].get("guard")],
        "invalid_actions": sum(1 for s in trace if "error" in s["result"]),
        "model_calls": planner.model_calls, "input_tokens": planner.input_tokens,
        "output_tokens": planner.output_tokens, "cost_usd": planner.cost_usd,
        "final_text": planner.final_text, "trace": trace,
    }


# ------------------------------------------------------------------------------------------------------- numbers

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes in n tries (sound at 0/n and n/n, unlike p ± 1.96·se)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(runs: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for r in runs:
        by_task.setdefault(r["task"], []).append(r)
    per_task = {}
    for task_id, rs in by_task.items():
        ok = sum(r["resolved"] for r in rs)
        resolved_steps = [r["steps"] / r["gold_steps"] for r in rs if r["resolved"]]
        per_task[task_id] = {
            "k": len(rs), "resolved": ok, "pass@1": ok / len(rs),
            "pass^k": ok == len(rs), "pass@k": ok > 0,
            "overfit": sum(r["overfit"] for r in rs), "tests_edited": sum(r["tests_edited"] for r in rs),
            "out_of_scope": sum(bool(r["out_of_scope"]) for r in rs),
            "refusals": sum(len(r["refusals"]) for r in rs),
            "steps_over_gold": round(sum(resolved_steps) / len(resolved_steps), 2) if resolved_steps else None,
            "cost_usd": sum(r["cost_usd"] for r in rs),
        }
    n, ok = len(runs), sum(r["resolved"] for r in runs)
    lo, hi = wilson(ok, n)
    return {
        "runs": n, "resolved": ok, "rate": ok / n if n else 0.0, "wilson95": [round(lo, 3), round(hi, 3)],
        "tasks": len(per_task),
        "pass^k_tasks": sum(t["pass^k"] for t in per_task.values()),
        "pass@k_tasks": sum(t["pass@k"] for t in per_task.values()),
        "overfit_runs": sum(r["overfit"] for r in runs),
        "cost_usd": sum(r["cost_usd"] for r in runs),
        "per_task": per_task,
    }


def gate(current: dict, baseline: dict) -> tuple[bool, list[str]]:
    """PASS unless the overall rate fell by more than GATE_RATE_DROP, or a task lost GATE_TASK_DROP+ successes.
    With 27 runs a 10-point move is inside the noise, so a FAIL means "read the traces", not "proven worse"."""
    reasons = []
    drop = baseline["rate"] - current["rate"]
    if drop > GATE_RATE_DROP:
        reasons.append(f"resolve rate {baseline['rate']:.0%} → {current['rate']:.0%} "
                       f"(−{drop:.0%}, limit −{GATE_RATE_DROP:.0%})")
    for task_id, base in baseline["per_task"].items():
        now = current["per_task"].get(task_id)
        if now is None:
            reasons.append(f"{task_id}: in the baseline but not run now")
        elif base["resolved"] - now["resolved"] >= GATE_TASK_DROP:
            reasons.append(f"{task_id}: {base['resolved']}/{base['k']} → {now['resolved']}/{now['k']}")
    return not reasons, reasons


def report(summary: dict, label: str, model: str, tasks: list[dict]) -> str:
    lo, hi = summary["wilson95"]
    lines = [
        f"eval suite '{label}', model {model}: {summary['resolved']}/{summary['runs']} runs resolved "
        f"= {summary['rate']:.0%} (95% CI {lo:.0%}–{hi:.0%})",
        f"tasks: pass^k {summary['pass^k_tasks']}/{summary['tasks']} (every run resolved), "
        f"pass@k {summary['pass@k_tasks']}/{summary['tasks']} (at least one)",
        f"overfit runs (visible tests pass, hidden fail): {summary['overfit_runs']}   "
        f"cost ${summary['cost_usd']:.4f} (${summary['cost_usd'] / max(1, summary['runs']):.5f} per run)",
        "",
        f"{'task':22} {'kind':22} {'resolved':>8} {'pass@1':>7} {'overfit':>7} {'refused':>7} {'steps/gold':>10}",
    ]
    kinds = {t["id"]: t["kind"] for t in tasks}
    for task_id, t in summary["per_task"].items():
        sog = "-" if t["steps_over_gold"] is None else f"{t['steps_over_gold']:.2f}"
        lines.append(f"{task_id:22} {kinds.get(task_id, ''):22} {t['resolved']:>4}/{t['k']:<3} "
                     f"{t['pass@1']:>7.0%} {t['overfit']:>7} {t['refusals']:>7} {sog:>10}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------- CLI

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="check the tasks themselves (free)")
    parser.add_argument("--runs", type=int, default=3, help="k: runs per task")
    parser.add_argument("--tasks", help="comma-separated task ids (default: all)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--label", default=datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M"))
    parser.add_argument("--no-issue", action="store_true",
                        help="ablation: hide the issue text, so only the visible tests say what is wanted")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    tasks = load_tasks(args.tasks.split(",") if args.tasks else None)

    if args.validate:
        bad = 0
        for t in tasks:
            problems = validate_task(t)
            bad += bool(problems)
            print(f"{t['id']:22} {'ok' if not problems else 'PROBLEM'}  gold {len(t['gold'])} patch(es), "
                  f"{len(t['decoys'])} decoy(s), gold trajectory {gold_steps(t)} steps")
            for p in problems:
                print(f"    {p}")
        print(f"\n{len(tasks) - bad}/{len(tasks)} tasks valid")
        sys.exit(1 if bad else 0)

    runs = []
    out = HERE / "results" / f"eval-{args.label}"
    out.parent.mkdir(exist_ok=True)
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for t in tasks:
            for i in range(1, args.runs + 1):
                r = run_agent(t, f"{args.label}-{t['id']}-{i}", args.model, args.max_steps, with_issue=not args.no_issue)
                runs.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                verdict = "RESOLVED" if r["resolved"] else ("OVERFIT" if r["overfit"] else r["state"])
                print(f"{t['id']:22} run {i}: {verdict:16} steps {r['steps']:2}/{r['gold_steps']} gold  "
                      f"${r['cost_usd']:.5f}  {' → '.join(r['tools'])}", flush=True)

    summary = summarize(runs)
    text = report(summary, args.label, args.model, tasks)
    if args.gate:
        base = json.loads(BASELINE.read_text(encoding="utf-8"))
        ok, reasons = gate(summary, base)
        text += (f"\n\ngate vs baseline '{base['label']}' ({base['model']}, {base['rate']:.0%}): "
                 + ("PASS" if ok else "FAIL\n  " + "\n  ".join(reasons)))
    print("\n" + text)
    out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    if args.save_baseline:
        BASELINE.write_text(json.dumps({"label": args.label, "model": args.model, **summary}, indent=2,
                                       ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"baseline saved: {BASELINE.relative_to(HERE)}")
    if args.gate and not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
