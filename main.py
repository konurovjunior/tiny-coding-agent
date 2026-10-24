"""Fill in the stage interfaces in your own workspace."""

import os
import stat
import tempfile
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
    raise NotImplementedError("run_tests")


def agent_loop(workspace, actions, max_steps=6):
    raise NotImplementedError("agent_loop")


def drive(workspace, planner, max_steps=6):
    raise NotImplementedError("drive: feed each observation to the planner")


def proposal_planner(proposals):
    raise NotImplementedError("proposal_planner: select repairs from observed failures")
