"""Do the traces in Langfuse say the same as our own results file? (lesson 14/24: inspect numbers before trusting them)

Usage:  python langfuse_check.py --session 1424-luna-traced --results results/eval-1424-luna-traced.jsonl

Reads the session's observations through Langfuse's public API (v2 observations: in v4 "events only" mode the older
/traces and /scores read endpoints are gone) and compares, per run and in total: trace count, model calls, input and
output tokens, and cost. Exit code 1 on any mismatch.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict

import telemetry


def observations(cfg: dict, session: str) -> list[dict]:
    found, cursor = [], None
    while True:
        query = {"sessionId": session, "limit": 1000, "fields": "core,basic,usage"}
        if cursor:
            query["cursor"] = cursor
        url = f"{cfg['host']}/api/public/v2/observations?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": cfg["auth"]}),
                                    timeout=30) as response:
            page = json.load(response)
        found += page["data"]
        cursor = page.get("meta", {}).get("cursor")
        if not cursor or not page["data"]:
            return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    runs = [json.loads(line) for line in open(args.results, encoding="utf-8")]
    obs = observations(telemetry._langfuse_config(), args.session)

    per_trace = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0, "tools": 0, "warnings": 0})
    for o in obs:
        t = per_trace[o["traceId"]]
        if o["type"] == "GENERATION":
            t["calls"] += 1
            t["in"] += o.get("inputUsage") or 0
            t["out"] += o.get("outputUsage") or 0
            t["cost"] += o.get("totalCost") or 0.0
        elif o["type"] == "TOOL":
            t["tools"] += 1
            t["warnings"] += o.get("level") == "WARNING"

    problems = []
    for r in runs:
        t = per_trace.get(r["trace_id"])
        if t is None:
            problems.append(f"{r['run_id']}: trace {r['trace_id']} not in Langfuse")
            continue
        ours = (r["model_calls"], r["input_tokens"], r["output_tokens"], r["steps"])
        theirs = (t["calls"], t["in"], t["out"], t["tools"])
        if ours != theirs or abs(r["cost_usd"] - t["cost"]) > 1e-9:
            problems.append(f"{r['run_id']}: ours calls/in/out/steps {ours} ${r['cost_usd']:.6f}, "
                            f"Langfuse {theirs} ${t['cost']:.6f}")
    total = {k: sum(t[k] for t in per_trace.values()) for k in ("calls", "in", "out", "cost", "tools", "warnings")}
    print(f"Langfuse session {args.session}: {len(per_trace)} traces, {len(obs)} observations; "
          f"{total['calls']} model calls, {total['in']:,} in / {total['out']:,} out tokens, ${total['cost']:.5f}, "
          f"{total['tools']} tool steps ({total['warnings']} warnings)")
    print(f"our results file: {len(runs)} runs, {sum(r['model_calls'] for r in runs)} model calls, "
          f"{sum(r['input_tokens'] for r in runs):,} in / {sum(r['output_tokens'] for r in runs):,} out tokens, "
          f"${sum(r['cost_usd'] for r in runs):.5f}")
    if len(per_trace) != len(runs):
        problems.append(f"{len(per_trace)} traces in Langfuse, {len(runs)} runs in the file")
    print("MATCH" if not problems else "MISMATCH\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
