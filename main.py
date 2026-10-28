"""Fill in the stage interfaces in your own workspace."""

import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

OUTPUT_LIMIT = 8_000  # characters of test output kept for the planner (the end, where the failures are)


def safe_path(workspace, relative):
    """Resolve `relative` to an existing file inside `workspace`, or raise ValueError.

    The order matters: resolve first (follows symlinks and collapses ".."), then check containment on the
    resolved path. Checking the raw string would let "link -> /etc/hosts" or "sub/../../x" through.
    """
    root = Path(workspace).resolve()
    requested = Path(relative)
    if requested.is_absolute():  # the caller chooses the root; a model must not pick a new one
        raise ValueError(f"absolute path not allowed: {relative}")
    target = (root / requested).resolve()
    # Compare path components, not string prefixes: "/tmp/work-other".startswith("/tmp/work") is True.
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes the workspace: {relative}")
    if not target.is_file():  # a directory (".") or a missing file is not something to read or patch
        raise ValueError(f"not a file in the workspace: {relative}")
    return target


def apply_patch(workspace, relative, old, new):
    """Replace the single exact occurrence of `old` with `new` in a workspace file, or raise ValueError.

    A patch is a claim about the file as it is now: `old` must appear exactly once. Zero means the model's view
    is stale, two or more means the edit is ambiguous; either way the file is left untouched.
    The new content goes to a temporary file in the same folder, gets the original's permission bits, and then
    replaces the original in one atomic step, so a crash mid-write can never leave half a file behind.
    """
    target = safe_path(workspace, relative)
    if not old:  # "" occurs everywhere ("abc".count("") == 4): an empty anchor is never a real patch
        raise ValueError("old text is empty")
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found != 1:
        raise ValueError(f"old text found {found} times in {relative}, expected exactly once")
    patched = text.replace(old, new, 1)

    mode = stat.S_IMODE(target.stat().st_mode)  # e.g. 0o751: keep executable scripts executable
    # Same folder = same filesystem, which is what makes os.replace atomic (a rename, never a copy).
    tmp_path = None
    try:
        # delete=False: we decide the file's fate ourselves (replace it in, or remove it in finally)
        with tempfile.NamedTemporaryFile("w", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp",
                                         delete=False, encoding="utf-8") as f:  # with = try-with-resources
            tmp_path = f.name  # captured before writing: if write or close fails, finally still knows the file
            f.write(patched)
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)  # atomic: readers see the old file or the new one, never a mix
    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)  # already gone after a successful replace; otherwise clean up


def run_tests(workspace, timeout=5):
    """Run the workspace's unittest suite through the one command this tool allows, and judge it from evidence.

    The tool owns the command: nothing from a model or a request becomes part of it. Passing needs all of:
    exit code 0, the final unittest summary saying OK, at least one test run, and none skipped.
    """
    root = Path(workspace).resolve()
    if not (root / "tests").is_dir():
        return {"passed": False, "tests": 0, "skipped": 0, "state": "no_tests", "returncode": None,
                "output": "no tests/ folder in the workspace"}
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]  # fixed argv, never a shell string
    proc = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)  # own process group, so a timeout can stop its children too
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # the whole group: a test may have started processes of its own
        stdout, stderr = proc.communicate()  # reap it and collect what it printed
        return {"passed": False, "tests": 0, "skipped": 0, "state": "timeout", "returncode": None,
                "output": _tail(stdout + stderr) + f"\n[stopped after {timeout} s]"}

    # unittest writes its summary to stderr. Only the LAST "Ran N tests" counts: test code can print a fake one
    # (to stdout or even stderr) before the real run ends.
    runs = list(re.finditer(r"^Ran (\d+) tests? in ", stderr, re.MULTILINE))
    tests = int(runs[-1].group(1)) if runs else 0
    verdict = stderr[runs[-1].end():] if runs else ""
    skipped_match = re.search(r"skipped=(\d+)", verdict)
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    ok = re.search(r"^OK\b", verdict, re.MULTILINE) is not None
    passed = proc.returncode == 0 and ok and tests > 0 and skipped == 0  # zero or skipped tests prove nothing
    return {"passed": passed, "tests": tests, "skipped": skipped, "state": "passed" if passed else "failed",
            "returncode": proc.returncode, "output": _tail(stdout + stderr)}


def _tail(text):
    return text if len(text) <= OUTPUT_LIMIT else "[...]\n" + text[-OUTPUT_LIMIT:]


def _execute(workspace, action):
    """Run one typed action. Raises ValueError when the call itself is invalid (unknown tool, bad fields,
    a patch the file refuses): the loop treats that as a stop, not as an observation to retry on."""
    tool = action.get("tool") if isinstance(action, dict) else None
    if tool == "test":
        return run_tests(workspace)
    if tool == "patch":
        fields = [action.get(k) for k in ("path", "old", "new")]
        if not all(isinstance(f, str) for f in fields):
            raise ValueError("patch needs string fields path, old and new")
        apply_patch(workspace, *fields)
        return {"applied": True}
    raise ValueError(f"unknown tool: {tool!r}")  # e.g. "shell": only test and patch exist


def drive(workspace, planner, max_steps=6):
    """The agent loop: ask the planner for one action, run it, record action + observation, repeat.

    planner(trace) sees everything so far and returns the next action, or None to stop (abstain).
    Stops with:  completed         a test run passed (the only success: evidence, not a claim)
                 failed            the planner has nothing more to do, or an action was invalid
                 budget_exhausted  max_steps actions were used and the planner still wanted more
    """
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    trace = []
    while True:
        action = planner(trace)
        if action is None:
            return _finish("failed", trace, "planner abstained: no action fits the last observation")
        if len(trace) >= max_steps:  # checked before acting: the step that would exceed the budget never runs
            return _finish("budget_exhausted", trace, f"used all {max_steps} steps")
        try:
            result = _execute(workspace, action)
        except ValueError as e:
            trace.append({**action, "result": {"error": str(e)}})
            return _finish("failed", trace, f"invalid action: {e}")
        trace.append({**action, "result": result})
        if action["tool"] == "test" and result["passed"]:
            return _finish("completed", trace, f"{result['tests']} tests passed")


def _finish(state, trace, reason):
    return {"state": state, "steps": len(trace), "reason": reason, "trace": trace}


def agent_loop(workspace, actions, max_steps=6):
    """Run a fixed, recorded list of actions under the same loop and stop rules as drive()."""
    actions = list(actions)
    return drive(workspace, lambda trace: actions[len(trace)] if len(trace) < len(actions) else None, max_steps)


def proposal_planner(proposals):
    """A transparent rule planner: test first; after a failing test, pick the first unused prewritten repair whose
    failure marker appears in the test output; after a patch, test again. It never invents code: that is the
    model's job in later lessons, behind the same tools, preconditions and completion rule."""
    proposals = list(proposals)

    def plan(trace):
        if not trace or trace[-1].get("tool") == "patch":
            return {"tool": "test"}
        output = trace[-1].get("result", {}).get("output", "")
        used = {step.get("proposal") for step in trace}
        for p in proposals:
            if p["id"] not in used and p["failure_contains"] in output:
                return {"tool": "patch", "path": p["path"], "old": p["old"], "new": p["new"], "proposal": p["id"]}
        return None  # no proposal matches what was observed: abstain rather than guess

    return plan
