"""The MCP server follows lesson 13/06's stateless rules, and refusals reach the model as tool errors.

Run (from this folder):  ../../../.venv/bin/python -m unittest test_mcp_server -v
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mcp_server import CAPABILITIES_KEY, CLIENT_INFO_KEY, PROTOCOL_VERSION, VERSION_KEY, McpServer

HERE = Path(__file__).resolve().parent


def request(method, params=None, request_id=1, version=PROTOCOL_VERSION, capabilities=True):
    meta = {VERSION_KEY: version, CLIENT_INFO_KEY: {"name": "test-client", "version": "1.0"}}
    if capabilities:
        meta[CAPABILITIES_KEY] = {}
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": {**(params or {}), "_meta": meta}}


def call(name, arguments, request_id=1):
    return request("tools/call", {"name": name, "arguments": arguments}, request_id)


class McpServerTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.ws = Path(self.t.name)
        (self.ws / "basket.py").write_text("def total(price, quantity):\n    return price + quantity\n")
        self.addCleanup(self.t.cleanup)
        self.server = McpServer(self.ws)

    def result(self, message):
        response = self.server.handle(message)
        self.assertNotIn("error", response, response)
        self.assertEqual(response["result"]["resultType"], "complete")
        self.assertEqual(response["result"]["_meta"]["io.modelcontextprotocol/serverInfo"]["name"], "tiny-coding-agent")
        return response["result"]

    def error_code(self, message):
        return self.server.handle(message)["error"]["code"]

    # --- discovery and listing ---------------------------------------------------------------------------------
    def test_discover_lists_versions_and_capabilities(self):
        r = self.result(request("server/discover"))
        self.assertEqual(r["supportedVersions"], [PROTOCOL_VERSION])
        self.assertIn("tools", r["capabilities"])
        self.assertEqual((r["cacheScope"], r["ttlMs"] > 0), ("public", True))

    def test_tools_list_is_sorted_camelcase_and_hides_executors(self):
        tools = self.result(request("tools/list"))["tools"]
        self.assertEqual([t["name"] for t in tools], ["apply_patch", "read_file", "run_tests"])
        for t in tools:
            self.assertEqual(set(t), {"name", "description", "inputSchema", "annotations"})
        read_only = {t["name"]: t["annotations"]["readOnlyHint"] for t in tools}
        self.assertEqual(read_only, {"apply_patch": False, "read_file": True, "run_tests": False})

    # --- tools/call --------------------------------------------------------------------------------------------
    def test_read_file(self):
        r = self.result(call("read_file", {"path": "basket.py"}))
        self.assertFalse(r["isError"])
        self.assertIn("price + quantity", r["content"][0]["text"])

    def test_apply_patch_changes_the_file(self):
        r = self.result(call("apply_patch", {"path": "basket.py", "old": "price + quantity", "new": "price * quantity"}))
        self.assertFalse(r["isError"])
        self.assertIn("price * quantity", (self.ws / "basket.py").read_text())

    def test_a_refused_path_is_a_tool_error_the_model_can_read(self):
        r = self.result(call("read_file", {"path": "../.env"}))
        self.assertTrue(r["isError"])
        self.assertIn("escapes the workspace", r["content"][0]["text"])

    def test_an_ambiguous_patch_is_a_tool_error_and_the_file_is_unchanged(self):
        before = (self.ws / "basket.py").read_text()
        r = self.result(call("apply_patch", {"path": "basket.py", "old": "quantity", "new": "qty"}))  # appears twice
        self.assertTrue(r["isError"])
        self.assertEqual((self.ws / "basket.py").read_text(), before)

    def test_unknown_tool_and_bad_arguments_are_protocol_errors(self):
        self.assertEqual(self.error_code(call("delete_file", {"path": "basket.py"})), -32602)
        self.assertEqual(self.error_code(call("read_file", {})), -32602)                               # missing
        self.assertEqual(self.error_code(call("read_file", {"path": "basket.py", "mode": "w"})), -32602)  # unknown
        self.assertEqual(self.error_code(call("read_file", {"path": 7})), -32602)                       # wrong type

    def test_not_yet_implemented_tool_is_a_tool_error(self):
        self.assertTrue(self.result(call("run_tests", {}))["isError"])  # stage 3 will implement run_tests

    # --- the stateless envelope --------------------------------------------------------------------------------
    def test_missing_meta_or_capabilities_is_32602(self):
        no_meta = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        self.assertEqual(self.error_code(no_meta), -32602)
        self.assertEqual(self.error_code(request("tools/list", capabilities=False)), -32602)

    def test_nothing_is_remembered_between_requests(self):
        self.result(request("tools/list", request_id=1))  # a complete request first...
        # ...does not let the next one skip its own capabilities, as a handshake-era server would
        self.assertEqual(self.error_code(request("tools/list", request_id=2, capabilities=False)), -32602)

    def test_unsupported_version_is_32022_with_the_supported_list(self):
        error = self.server.handle(request("tools/list", version="2027-01-01"))["error"]
        self.assertEqual(error["code"], -32022)
        self.assertEqual(error["data"], {"requested": "2027-01-01", "supported": [PROTOCOL_VERSION]})

    def test_unknown_method_is_32601_and_notifications_get_no_answer(self):
        self.assertEqual(self.error_code(request("tools/cal")), -32601)
        notification = request("tools/list")
        del notification["id"]
        self.assertIsNone(self.server.handle(notification))

    def test_not_json_is_a_parse_error(self):
        self.assertEqual(json.loads(self.server.handle_line("{not json"))["error"]["code"], -32700)


class StdioTransport(unittest.TestCase):
    def test_a_real_process_answers_line_by_line(self):
        with tempfile.TemporaryDirectory() as ws:
            (Path(ws) / "basket.py").write_text("x = 1\n")
            lines = "\n".join(json.dumps(m) for m in [request("server/discover", request_id=1),
                                                      call("read_file", {"path": "basket.py"}, request_id=2)])
            out = subprocess.run([sys.executable, str(HERE / "mcp_server.py"), "--workspace", ws],
                                 input=lines + "\n", capture_output=True, text=True, timeout=10)
            responses = [json.loads(line) for line in out.stdout.splitlines()]
            self.assertEqual([r["id"] for r in responses], [1, 2])
            self.assertEqual(responses[1]["result"]["content"][0]["text"], "x = 1\n")


if __name__ == "__main__":
    unittest.main()
