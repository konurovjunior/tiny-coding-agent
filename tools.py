"""The agent's tools as the model sees them: name, description, input schema, and whether a call changes anything.

Lesson 13/05: the model chooses a tool only from these words, so they decide how often it picks the right one.
Rules followed here, and checked by test_tools.py with the lesson's own linter:
  - snake_case verb_noun names, nothing embedded in the name (read_file(path), not read_basket_py)
  - descriptions say "Use when ..." and "Do not use for ..." and point at the right neighbour tool
  - every string field is described; "additionalProperties": False rejects arguments the schema doesn't know
  - consequential tools (they change files or run code) are marked, so the loop can gate them (lesson 13/01)
"""

import os
from pathlib import Path

from main import apply_patch, run_tests, safe_path
from memory import RepairMemory

MAX_READ_CHARS = 20_000  # a whole large file would crowd out everything else in the model's context


def read_file(workspace, path):
    """Return the text of one workspace file (through safe_path, so it cannot leave the workspace)."""
    text = safe_path(workspace, path).read_text(encoding="utf-8")
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n[truncated: file has {len(text)} characters, showing the first {MAX_READ_CHARS}]"
    return text


PATH_FIELD = {
    "type": "string",
    "description": "File path relative to the workspace root, e.g. 'basket.py'. Absolute paths and '..' are rejected.",
}

# Where memory_search looks when the tool is served on its own (the MCP server); runs can pass their own store.
DEFAULT_MEMORY = Path(os.environ.get("TINY_AGENT_MEMORY", Path(__file__).resolve().parent / "memory" / "notes.jsonl"))

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Use when you need the current text of one file in the workspace, for example before writing a patch "
            "or to understand a failing test. Returns the file content. "
            "Do not use to change a file (use apply_patch) or to check whether the code works (use run_tests)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": PATH_FIELD},
            "required": ["path"],
            "additionalProperties": False,
        },
        "consequential": False,
        "executor": lambda workspace, args: read_file(workspace, args["path"]),
    },
    {
        "name": "apply_patch",
        "description": (
            "Use when you want to change one file by replacing a piece of its text. 'old' must be copied exactly "
            "from the current file (read it with read_file first) and must occur exactly once; otherwise the patch "
            "is refused and the file is left unchanged, so include enough surrounding text to be unique. "
            "Do not use to create, rename or delete files, or to rewrite a whole file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": PATH_FIELD,
                "old": {
                    "type": "string",
                    "description": "Exact text to replace, copied from the current file; must occur exactly once.",
                },
                "new": {
                    "type": "string",
                    "description": "Replacement text. May be empty to delete 'old'.",
                },
            },
            "required": ["path", "old", "new"],
            "additionalProperties": False,
        },
        "consequential": True,  # changes a file
        "executor": lambda workspace, args: apply_patch(workspace, args["path"], args["old"], args["new"]),
    },
    {
        "name": "run_tests",
        "description": (
            "Use when you need to know whether the code in the workspace works: runs the workspace's unit tests "
            "and returns whether they passed, how many ran, and their output. Run it after every patch. "
            "Do not use to read file contents (use read_file); it takes no arguments and runs only the test suite."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "consequential": True,  # runs the workspace's code (stage 3)
        "executor": lambda workspace, args: run_tests(workspace),
    },
    {
        "name": "memory_search",
        "description": (
            "Use when a test fails and you want to know how a similar failure was fixed before: searches notes on "
            "past repairs whose tests passed, and returns the closest ones with their patches. Notes are hints "
            "that can be stale; always check them with read_file and run_tests. "
            "Do not use to read the workspace's files (use read_file) or to store anything (notes are written "
            "automatically after a proven fix)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words describing the failure, e.g. the assertion line and the file name.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "consequential": False,  # reads notes only; the model cannot write them (memory poisoning, lesson 14/07)
        "executor": lambda workspace, args: RepairMemory(DEFAULT_MEMORY).search(args["query"]),
    },
]

TOOLS_BY_NAME = {tool["name"]: tool for tool in TOOLS}
