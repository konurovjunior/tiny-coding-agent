"""Step gates for the failure modes the course's 14/26 lesson names, checked at every step of the loop.

The loop already catches several of them by design (see results/failure-modes-*.txt):
  hallucinated action    unknown tools and invented arguments never run (the planner's schema check)
  tool misuse            a patch must match its text exactly once, or it is refused and the model is told why
  success hallucination  only a passing test run completes a run; "Fixed it!" is not evidence
These gates close the three that otherwise got through:
  scope creep      a patch under tests/ is refused: making tests pass by editing them fixes nothing
  cascading errors repeating an action that was already refused stops the run at once, instead of burning the
                   whole step budget on the same mistake
  context loss     re-reading a file that hasn't changed since the last read still runs (cheap, and the model may
                   need it), but the result tells the model it already has that text, and from which step
"""

from pathlib import PurePosixPath

PROTECTED_DIR = "tests"


def _same_action(a: dict, b: dict) -> bool:
    keys = ("tool", "path", "old", "new")
    return all(a.get(k) == b.get(k) for k in keys)


class Guards:
    """drive(..., guards=Guards()) calls before() ahead of every action and after() on every result."""

    def before(self, action: dict, trace: list[dict]) -> tuple[str, str] | None:
        """None to go ahead, ("refuse", why) to record an error instead of running it, ("stop", why) to end the run."""
        if action.get("tool") == "patch" and _under(action.get("path", ""), PROTECTED_DIR):
            return ("refuse", f"patches under {PROTECTED_DIR}/ are not allowed: fix the code, not the tests")
        for step in trace:
            if "error" in step["result"] and _same_action(step, action):
                return ("stop", f"repeated an action that was already refused ({action['tool']} "
                                f"{action.get('path', '')}): stopping instead of spending the budget on it")
        return None

    def after(self, action: dict, result: dict, trace: list[dict]) -> dict:
        if action.get("tool") == "read" and "text" in result:
            for i in range(len(trace), 0, -1):  # the most recent earlier step first
                step = trace[i - 1]
                if step["tool"] == "patch" and step.get("path") == action["path"] and step["result"].get("applied"):
                    break  # the file changed since: reading it again is useful
                if step["tool"] == "read" and step.get("path") == action["path"] and "text" in step["result"]:
                    return {**result, "note": f"unchanged since you read it at step {i}; you already have this text"}
        return result


def _under(path: str, folder: str) -> bool:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    return bool(parts) and parts[0] == folder
