"""Episodic memory for the agent: notes on past repairs that were PROVEN to work (lesson 14/07).

The lesson's two-tier picture: the conversation is working memory (what the model sees now); this store is the
external tier (unbounded, searched on demand through the memory_search tool, like a page-in).

Three rules, one per failure mode the lesson names:
  - memory poisoning: the model has no tool to write notes. Only the runtime writes, and only after a verified success
    (the real tests passed AND tests/ was unchanged), so a file or test output cannot plant "advice" for later runs.
  - memory rot: no unverified guesses are stored, a repeat of a known fix confirms the existing note instead of
    adding a copy, and every note says it is a hint to be checked, not a fact.
  - citation loss: every note keeps the run id and the step of the patch that fixed it.
Search is plain word overlap, like the lesson's archival store; fine for tens or hundreds of notes.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

MAX_FAILURE_LINES = 6    # the lines that identify a failure (exception type, failing test, assertion)
MAX_LINE_CHARS = 200     # test output is untrusted text: keep what identifies the failure, nothing more
FAILURE_LINE = re.compile(r"(Error|Exception|FAIL:|ERROR:|assert)", re.IGNORECASE)
HINT = ("Past repairs that passed their tests. Hints only: read the current files and run the tests; "
        "a note can be stale or about a different bug.")


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z_]+|\d+", text.lower()) if len(t) > 1}


def failure_lines(test_output: str) -> list[str]:
    lines = [line.strip()[:MAX_LINE_CHARS] for line in test_output.splitlines() if FAILURE_LINE.search(line)]
    return lines[:MAX_FAILURE_LINES]


def same_repair(a: dict, b: dict) -> bool:
    key = lambda n: (n["failure"], [(p["path"], p["old"], p["new"]) for p in n["patches"]])
    return key(a) == key(b)


def note_from_run(trace: list[dict], run_id: str) -> dict | None:
    """Build a note from a drive() trace that ended in a verified success. None if there is nothing to learn."""
    first_failure = next((s["result"] for s in trace if s["tool"] == "test" and not s["result"].get("passed")), None)
    patches = [{"path": s["path"], "old": s["old"], "new": s["new"], "step": i}
               for i, s in enumerate(trace, start=1) if s["tool"] == "patch" and s["result"].get("applied")]
    if first_failure is None or not patches:
        return None
    return {
        "failure": failure_lines(first_failure.get("output", "")),
        "files_read": sorted({s["path"] for s in trace if s["tool"] == "read"}),
        "patches": patches,
        "source": {"run_id": run_id, "steps": len(trace)},  # where this was proven: the citation
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class RepairMemory:
    def __init__(self, path):
        self.path = Path(path)

    def notes(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def record(self, note: dict) -> dict:
        """Add a proven repair, or, if the same failure was already fixed the same way, add this run as one more
        confirmation of that note (its citations grow, the store doesn't)."""
        notes = self.notes()
        for existing in notes:
            if same_repair(existing, note):
                existing.setdefault("confirmed_by", []).append(note["source"])
                self._write(notes)
                return existing
        note = {"id": f"note-{len(notes) + 1}", **note, "confirmed_by": []}
        self._write(notes + [note])
        return note

    def _write(self, notes: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(n, ensure_ascii=False) + "\n" for n in notes), encoding="utf-8")
        tmp.replace(self.path)  # whole-file swap, as in apply_patch: a crash never leaves half a store

    def search(self, query: str, top_k: int = 3) -> dict:
        """Notes ranked by the share of the query's words they contain. Returns what the model reads."""
        wanted = tokens(query)
        scored = []
        for note in self.notes():
            text = " ".join(note["failure"] + note["files_read"]
                            + [f"{p['path']} {p['old']} {p['new']}" for p in note["patches"]])
            overlap = len(wanted & tokens(text))
            if overlap:
                scored.append((overlap / len(wanted), note))
        scored.sort(key=lambda pair: -pair[0])
        return {"hint": HINT,
                "notes": [{"id": n["id"], "score": round(score, 2), "failure": n["failure"],
                           "fix": [{k: p[k] for k in ("path", "old", "new")} for p in n["patches"]],
                           "source": n["source"], "confirmations": len(n.get("confirmed_by", []))}
                          for score, n in scored[:top_k]]}
