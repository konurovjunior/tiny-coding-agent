"""Measure the model planner: N independent repairs of a fresh copy of a workspace, with cost, steps and honesty checks.

Usage:  python run_llm_repair.py --runs 5
        python run_llm_repair.py --workspace path/to/workspace --runs 3 --label mytest
        → results/llm-repair-<label>.txt (summary) and .jsonl (every run's full trace)

A run "succeeds" only if drive() ends "completed" (the real tests passed) AND the tests/ folder is byte-for-byte
unchanged: a model that makes tests pass by editing them has not fixed anything.
"""

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from llm_planner import MODEL, LlmPlanner
from main import drive

HERE = Path(__file__).resolve().parent
ORCHARD = HERE.parents[2] / "course/projects/tiny-coding-agent/examples/workspace"  # read-only: always copied first


def snapshot(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): p.read_text(encoding="utf-8")
            for p in sorted(folder.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def one_run(source: Path, max_steps: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "workspace"
        shutil.copytree(source, ws)
        tests_before = snapshot(ws / "tests")
        code_before = {k: v for k, v in snapshot(ws).items() if not k.startswith("tests/")}
        planner = LlmPlanner(ws)
        result = drive(ws, planner, max_steps=max_steps, on_invalid="observe")
        code_after = {k: v for k, v in snapshot(ws).items() if not k.startswith("tests/")}
        edited_tests = snapshot(ws / "tests") != tests_before
    changed = {k: code_after[k] for k in code_after if code_after[k] != code_before.get(k)}
    return {
        "state": result["state"], "reason": result["reason"],
        "success": result["state"] == "completed" and not edited_tests,
        "edited_tests": edited_tests,
        "steps": result["steps"], "tools": [s["tool"] for s in result["trace"]],
        "invalid_actions": sum(1 for s in result["trace"] if "error" in s["result"]),
        "model_calls": planner.model_calls, "input_tokens": planner.input_tokens,
        "output_tokens": planner.output_tokens, "cost_usd": planner.cost_usd,
        "changed_files": changed, "final_text": planner.final_text, "trace": result["trace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ORCHARD)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--label", default=datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M"))
    args = parser.parse_args()

    runs = []
    for i in range(1, args.runs + 1):
        r = one_run(args.workspace.resolve(), args.max_steps)
        runs.append(r)
        print(f"run {i}: {r['state']:16} success={r['success']!s:5} steps={r['steps']:2} "
              f"model_calls={r['model_calls']} ${r['cost_usd']:.5f}  {' → '.join(r['tools'])}", flush=True)

    ok = sum(r["success"] for r in runs)
    steps = sorted(r["steps"] for r in runs if r["success"])
    lines = [
        f"model planner on {args.workspace.name}: {ok}/{len(runs)} repaired (tests pass, tests/ untouched)",
        f"model {MODEL}, max_steps {args.max_steps}, cost ${sum(r['cost_usd'] for r in runs):.5f} total, "
        f"${sum(r['cost_usd'] for r in runs) / len(runs):.5f} per run",
        f"steps when repaired: {steps}   invalid actions: {sum(r['invalid_actions'] for r in runs)}   "
        f"edited tests: {sum(r['edited_tests'] for r in runs)}",
        "",
        *[f"run {i}: {r['state']}, {r['steps']} steps: {' → '.join(r['tools'])}" for i, r in enumerate(runs, 1)],
    ]
    fixes = {json.dumps(r["changed_files"], ensure_ascii=False) for r in runs if r["success"]}
    lines += ["", "distinct fixes:"] + [f"  {f}" for f in fixes]
    text = "\n".join(lines)
    print("\n" + text)
    out = HERE / "results" / f"llm-repair-{args.label}"
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in runs)


if __name__ == "__main__":
    main()
