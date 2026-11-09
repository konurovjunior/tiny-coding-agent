"""Command line for the queue-based runtime (job_queue.py, lesson 14/29).

  enqueue-repair  python jobs.py enqueue-repair --source <workspace> --copy-to <new folder> [--issue "..."]
  enqueue-eval    python jobs.py enqueue-eval --runs 3 [--batch NAME] [--tasks a,b] [--no-issue]
  work            python jobs.py work [--workers 3] [--chaos 0.1]     run until nothing is due, then exit
  stats           python jobs.py stats [--batch NAME]                  depth, latency, retries, DLQ
  requeue         python jobs.py requeue JOB_ID                        a DLQ job, after fixing its cause
  report          python jobs.py report BATCH [--gate]                 eval_suite numbers for a batch of eval jobs
  nightly         python jobs.py nightly [--runs 3] [--workers 3]      the scheduled shape: enqueue the eval suite,
                                                                       work it off, gate it against the baseline

nightly is what a scheduler runs. Its exit code is the gate's (0 PASS, 1 FAIL), so cron/launchd mail or a CI job can
alert on it. A crontab line for 03:00 every night (the model key comes from work/.env, as for every live run):
  0 3 * * *  cd <this folder> && ../../../.venv/bin/python jobs.py nightly >> state/nightly.log 2>&1

--trace langfuse (work, nightly) sends one trace per job attempt, with eval scores, to Langfuse (lesson 14/24).
--chaos P makes P of all model calls fail with a timeout before they reach OpenAI: the retries, backoff and resume
from checkpoints are then exercised on real runs, and the eval gate shows whether the results changed.
State: state/jobs.sqlite (queue), state/job-runs.sqlite (durable checkpoints), state/jobs/<id>/ (eval workspaces).
"""

import argparse
import json
import random
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import telemetry
from durable import DurableRunner
from job_queue import JobQueue, Worker, handle_eval, handle_repair

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"


class ChaosClient:
    """Wraps the OpenAI client: a share of calls raise TimeoutError instead of being sent (fault injection)."""

    def __init__(self, inner, rate: float, seed: int = 7):
        self.inner, self.rate = inner, rate
        self.rng, self.lock = random.Random(seed), threading.Lock()
        self.injected = 0
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self.lock:
            fail = self.rng.random() < self.rate
            self.injected += fail
        if fail:
            raise TimeoutError("chaos: simulated model timeout")
        return self.inner.responses.create(**kwargs)


def make_handlers():
    jobs_dir = STATE / "jobs"
    return {"repair": handle_repair, "eval": lambda job, worker: handle_eval(job, worker, jobs_dir)}


def work(queue: JobQueue, workers: int, chaos: float = 0.0, lease_s: float = 120) -> ChaosClient | None:
    client = None
    if chaos:
        from openai import OpenAI
        client = ChaosClient(OpenAI(timeout=60, max_retries=0), chaos)
    runner = DurableRunner(STATE / "job-runs.sqlite", client=client)
    handlers = make_handlers()
    taken = [0] * workers

    def loop(i):
        taken[i] = Worker(queue, handlers, worker_id=f"w{i + 1}", lease_s=lease_s, runner=runner).run()

    threads = [threading.Thread(target=loop, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"workers done: {taken} jobs taken" + (f", {client.injected} model calls failed by chaos" if client else ""))
    return client


def enqueue_eval(queue: JobQueue, runs: int, batch: str, tasks: list[str] | None, with_issue: bool) -> int:
    from eval_suite import load_tasks
    n = 0
    for task in load_tasks(tasks):
        for _ in range(runs):
            queue.enqueue("eval", {"task": task["id"], "with_issue": with_issue}, batch=batch)
            n += 1
    return n


def report(queue: JobQueue, batch: str, gate_it: bool) -> tuple[str, bool]:
    from eval_suite import BASELINE, gate, load_tasks, report as table, summarize
    jobs = queue.jobs(batch)
    runs = [j["result"] for j in jobs if j["status"] == "done"]
    s = queue.stats(batch)
    lines = [f"batch {batch}: {s['done']} done, {s['dead']} dead, {s['queued'] + s['running']} not finished; "
             f"retries {s['retries']}; wait p50/p95 {s['wait_s']['p50']}/{s['wait_s']['p95']} s, "
             f"run p50/p95 {s['run_s']['p50']}/{s['run_s']['p95']} s"]
    lines += [f"  DLQ {d['id']}: {d['error']}" for d in s["dlq"]]
    ok = True
    if runs:
        summary = summarize(runs)
        lines += ["", table(summary, batch, "queue", load_tasks())]
        if gate_it:
            base = json.loads(BASELINE.read_text(encoding="utf-8"))
            ok, reasons = gate(summary, base)
            if s["dead"]:  # a run that never finished is a missing result, not a pass
                ok, reasons = False, reasons + [f"{s['dead']} job(s) in the DLQ"]
            lines.append(f"\ngate vs baseline '{base['label']}' ({base['rate']:.0%}): "
                         + ("PASS" if ok else "FAIL\n  " + "\n  ".join(reasons)))
    return "\n".join(lines), ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=STATE / "jobs.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("enqueue-repair")
    rp.add_argument("--source", type=Path, required=True)
    rp.add_argument("--copy-to", type=Path, required=True)
    rp.add_argument("--issue")
    ev = sub.add_parser("enqueue-eval")
    ev.add_argument("--runs", type=int, default=3)
    ev.add_argument("--batch")
    ev.add_argument("--tasks")
    ev.add_argument("--no-issue", action="store_true")
    wk = sub.add_parser("work")
    wk.add_argument("--workers", type=int, default=3)
    wk.add_argument("--chaos", type=float, default=0.0)
    wk.add_argument("--trace", choices=["off", "langfuse"], default="off")
    st = sub.add_parser("stats")
    st.add_argument("--batch")
    rq = sub.add_parser("requeue")
    rq.add_argument("job_id")
    rep = sub.add_parser("report")
    rep.add_argument("batch")
    rep.add_argument("--gate", action="store_true")
    ni = sub.add_parser("nightly")
    ni.add_argument("--runs", type=int, default=3)
    ni.add_argument("--workers", type=int, default=3)
    ni.add_argument("--chaos", type=float, default=0.0)
    ni.add_argument("--label", default="")
    ni.add_argument("--trace", choices=["off", "langfuse"], default="off")
    args = parser.parse_args()
    queue = JobQueue(args.db)
    telemetry.setup(getattr(args, "trace", "off"))  # lesson 14/24: one trace per job attempt, eval scores on it

    if args.command == "enqueue-repair":
        source, target = args.source.resolve(), args.copy_to.resolve()
        if target.exists() or target.is_relative_to(source):
            parser.error("--copy-to must be a new folder outside the source")
        shutil.copytree(source, target)  # never repair the original
        print(queue.enqueue("repair", {"workspace": str(target), "issue": args.issue}))
    elif args.command == "enqueue-eval":
        batch = args.batch or "eval-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        n = enqueue_eval(queue, args.runs, batch, args.tasks.split(",") if args.tasks else None, not args.no_issue)
        print(f"{n} eval jobs in batch {batch}")
    elif args.command == "work":
        work(queue, args.workers, args.chaos)
        telemetry.flush()
    elif args.command == "stats":
        print(json.dumps(queue.stats(args.batch), indent=2, ensure_ascii=False))
    elif args.command == "requeue":
        queue.requeue_dead(args.job_id)
        print(f"{args.job_id} queued again")
    elif args.command == "report":
        text, ok = report(queue, args.batch, args.gate)
        print(text)
        return 0 if ok else 1
    elif args.command == "nightly":
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
        batch = f"nightly-{stamp}{'-' + args.label if args.label else ''}"
        print(f"{enqueue_eval(queue, args.runs, batch, None, True)} eval jobs in batch {batch}", flush=True)
        work(queue, args.workers, args.chaos)
        telemetry.flush()
        text, ok = report(queue, batch, gate_it=True)
        print(text)
        out = HERE / "results" / f"{batch}.txt"
        out.write_text(text + "\n", encoding="utf-8")
        print(f"report: {out.relative_to(HERE)}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
