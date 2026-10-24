"""Fill in the stage interfaces in your own workspace."""

from pathlib import Path


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
    raise NotImplementedError("apply_patch")


def run_tests(workspace, timeout=5):
    raise NotImplementedError("run_tests")


def agent_loop(workspace, actions, max_steps=6):
    raise NotImplementedError("agent_loop")


def drive(workspace, planner, max_steps=6):
    raise NotImplementedError("drive: feed each observation to the planner")


def proposal_planner(proposals):
    raise NotImplementedError("proposal_planner: select repairs from observed failures")
