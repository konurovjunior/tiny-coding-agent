"""Command line for durable repair runs (durable.py).

  start   copy a workspace to a new folder and start repairing it
            python durable_run.py start --source ../../../course/projects/tiny-coding-agent/examples/workspace \\
                                        --copy-to /tmp/orchard-1 [--gate] [--crash-after N] [--max-steps 12]
  resume  continue a run from its last checkpoint
            python durable_run.py resume RUN_ID [--approve | --reject "reason"] [--crash-after N]
  status  show a run's state and its checkpoint history
            python durable_run.py status RUN_ID

Checkpoints live in state/runs.sqlite next to this file (git-ignored), or --db PATH.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from durable import DurableRunner, ResumeRefused, SimulatedCrash

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=HERE / "state" / "runs.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--source", type=Path, required=True)
    start.add_argument("--copy-to", type=Path, required=True)
    start.add_argument("--gate", action="store_true", help="stop for approval before every patch")
    start.add_argument("--max-steps", type=int, default=12)
    start.add_argument("--crash-after", type=int)
    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    choice = resume.add_mutually_exclusive_group()
    choice.add_argument("--approve", action="store_true")
    choice.add_argument("--reject", metavar="REASON")
    resume.add_argument("--crash-after", type=int)
    status = sub.add_parser("status")
    status.add_argument("run_id")
    args = parser.parse_args()

    runner = DurableRunner(args.db)
    try:
        if args.command == "start":
            source, target = args.source.resolve(), args.copy_to.resolve()
            if target.exists() or target.is_relative_to(source):
                parser.error("--copy-to must be a new folder outside the source")
            shutil.copytree(source, target)  # never repair the original
            result = runner.start(target, args.max_steps, args.gate, crash_after=args.crash_after)
        elif args.command == "resume":
            decision = "approve" if args.approve else "reject" if args.reject is not None else None
            result = runner.resume(args.run_id, decision, args.reject or "", args.crash_after)
        else:
            result = runner.status(args.run_id)
    except SimulatedCrash as e:
        print(f"CRASH: {e}. Resume with: python durable_run.py resume <run_id>", file=sys.stderr)
        return 2
    except (ResumeRefused, ValueError, KeyError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in ("completed", "awaiting_approval") else 1


if __name__ == "__main__":
    raise SystemExit(main())
