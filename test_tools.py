"""The tool registry passes lesson 13/05's linter, and read_file keeps stage 1's guarantees.

Run (from this folder):  ../../../.venv/bin/python -m unittest test_tools -v
The linter is loaded read-only from the course folder; the course is never edited.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from tools import TOOLS, TOOLS_BY_NAME, read_file

LINTER = (Path(__file__).resolve().parents[3]
          / "course/phases/13-tools-and-protocols/05-tool-schema-design/code/main.py")


def load_linter():
    # A unique module name: the lesson's file is also called main.py, like this project's main.py.
    spec = importlib.util.spec_from_file_location("lesson_13_05_linter", LINTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # register before running it: its @dataclass looks itself up there
    spec.loader.exec_module(module)
    return module


class Registry(unittest.TestCase):
    def test_linter_finds_nothing_to_block_or_warn(self):
        findings = load_linter().lint_registry(TOOLS)
        serious = [str(f) for f in findings if f.severity in ("block", "warn")]
        self.assertEqual(serious, [])

    def test_consequential_tools_are_marked(self):
        flags = {t["name"]: t["consequential"] for t in TOOLS}
        self.assertEqual(flags, {"read_file": False, "apply_patch": True, "run_tests": True})

    def test_unknown_arguments_are_rejected_by_every_schema(self):
        for tool in TOOLS:
            self.assertIs(tool["input_schema"]["additionalProperties"], False, tool["name"])


class ReadFile(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.ws = Path(self.t.name)
        (self.ws / "basket.py").write_text("return price + quantity\n")
        self.addCleanup(self.t.cleanup)

    def test_reads_through_the_registry(self):
        text = TOOLS_BY_NAME["read_file"]["executor"](self.ws, {"path": "basket.py"})
        self.assertEqual(text, "return price + quantity\n")

    def test_cannot_leave_the_workspace(self):
        with self.assertRaises(ValueError):
            read_file(self.ws, "../basket.py")

    def test_large_files_are_truncated_with_a_note(self):
        (self.ws / "big.py").write_text("x" * 30_000)
        text = read_file(self.ws, "big.py")
        self.assertIn("[truncated: file has 30000 characters", text)


if __name__ == "__main__":
    unittest.main()
