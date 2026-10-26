"""The agent's tools (tools.py) served over MCP, version 2026-07-28: stateless JSON-RPC 2.0 over stdio.

Lesson 13/06. Built by hand on purpose, to see the protocol; lesson 13/07 builds a server with the real SDK.

Every request is checked on its own, in this order, and nothing is remembered between requests:
  1. valid JSON-RPC 2.0?                         no -> -32700 (not JSON) / -32600 (not a request)
  2. params._meta with version + capabilities?   no -> -32602
  3. version supported?                          no -> -32022, with the supported list
  4. method known?                               no -> -32601
  5. tool known, arguments fit its schema?       no -> -32602
  6. run the tool -> result with resultType "complete" and the server's identity
Two kinds of failure, as the MCP spec separates them:
  - protocol errors (steps 1-5) are JSON-RPC errors: they go to the client program;
  - tool errors (the tool ran and refused, e.g. safe_path blocks "../.env") are a normal result with isError: true,
    so the MODEL sees the reason and can correct itself (re-read the file, retry the patch).
Not covered: -32020 (HTTP header vs body mismatch) only exists on HTTP; this server speaks stdio.

Run:  python mcp_server.py --workspace ../../../course/projects/tiny-coding-agent/examples/workspace
      then type one JSON-RPC request per line (or connect an MCP client that speaks 2026-07-28).
"""

import argparse
import json
import sys
from pathlib import Path

from tools import TOOLS, TOOLS_BY_NAME

PROTOCOL_VERSION = "2026-07-28"
SUPPORTED_VERSIONS = [PROTOCOL_VERSION]
VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
SERVER_INFO = {"name": "tiny-coding-agent", "version": "0.1.0"}
SERVER_CAPABILITIES = {"tools": {"listChanged": False}}

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602  # JSON-RPC 2.0
UNSUPPORTED_VERSION = -32022  # MCP's own code, from the range JSON-RPC leaves to applications

JSON_TYPES = {"string": str, "object": dict, "array": list, "boolean": bool, "number": (int, float), "integer": int}


def rpc_error(request_id, code: int, message: str, data=None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def complete(payload: dict, ttl_ms: int | None = None, cache_scope: str = "private") -> dict:
    """A successful modern result: resultType, the payload, the server's identity, and cache hints if cacheable."""
    result = {"resultType": "complete", **payload, "_meta": {SERVER_INFO_KEY: dict(SERVER_INFO)}}
    if ttl_ms is not None:
        result["ttlMs"], result["cacheScope"] = ttl_ms, cache_scope
    return result


def text_result(text: str, is_error: bool) -> dict:
    return complete({"content": [{"type": "text", "text": text}], "isError": is_error})


def wire_tool(tool: dict) -> dict:
    """A registry entry as MCP clients see it: camelCase inputSchema, no executor, read/write as annotations."""
    return {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": tool["input_schema"],
        "annotations": {
            "readOnlyHint": not tool["consequential"],  # pure tools only read
            "destructiveHint": tool["consequential"],   # consequential ones change files or run code: hosts gate them
        },
    }


def schema_errors(schema: dict, arguments: dict) -> list[str]:
    """The subset of JSON Schema the registry uses: required fields, unknown fields, field types."""
    props = schema.get("properties", {})
    errors = [f"missing required argument '{name}'" for name in schema.get("required", []) if name not in arguments]
    if schema.get("additionalProperties") is False:
        errors += [f"unknown argument '{name}'" for name in arguments if name not in props]
    for name, value in arguments.items():
        expected = JSON_TYPES.get(props.get(name, {}).get("type"))
        # bool is a subclass of int in Python (True == 1), so a JSON true must not pass as a number
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and expected is not bool)):
            errors.append(f"argument '{name}' must be {props[name]['type']}")
    return errors


def check_envelope(message) -> dict | None:
    """Steps 1-3: the request describes itself completely, or it is refused. Returns an error response or None."""
    if not isinstance(message, dict):
        return rpc_error(None, INVALID_REQUEST, "Invalid Request: not a JSON object")
    request_id = message.get("id")
    if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
        return rpc_error(request_id, INVALID_REQUEST, "Invalid Request")
    params = message.get("params")
    if not isinstance(params, dict) or not isinstance(params.get("_meta"), dict):
        return rpc_error(request_id, INVALID_PARAMS, "params._meta is required on every request")
    meta = params["_meta"]
    version = meta.get(VERSION_KEY)
    if not isinstance(version, str):
        return rpc_error(request_id, INVALID_PARAMS, f"{VERSION_KEY} is required")
    if version not in SUPPORTED_VERSIONS:
        return rpc_error(request_id, UNSUPPORTED_VERSION, "Unsupported protocol version",
                         {"requested": version, "supported": list(SUPPORTED_VERSIONS)})
    if not isinstance(meta.get(CAPABILITIES_KEY), dict):
        return rpc_error(request_id, INVALID_PARAMS, f"{CAPABILITIES_KEY} is required")
    info = meta.get(CLIENT_INFO_KEY)  # recommended, not required; but if present it must be well formed
    if info is not None and not (isinstance(info, dict) and isinstance(info.get("name"), str)
                                 and isinstance(info.get("version"), str)):
        return rpc_error(request_id, INVALID_PARAMS, f"{CLIENT_INFO_KEY} is malformed")
    return None


class McpServer:
    """Holds configuration only (the workspace the operator chose). No per-request state: any copy of this
    server could answer any request, which is the point of the 2026-07-28 stateless design."""

    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()  # chosen by whoever starts the server, never by a request

    def handle(self, message) -> dict | None:
        """One JSON-RPC message in, one response out (None for a notification, which gets no answer)."""
        if isinstance(message, dict) and "id" not in message:
            return None
        invalid = check_envelope(message)
        if invalid:
            return invalid
        request_id, method, params = message["id"], message["method"], message["params"]
        if method == "server/discover":
            return self._ok(request_id, complete({
                "supportedVersions": list(SUPPORTED_VERSIONS),
                "capabilities": SERVER_CAPABILITIES,
                "instructions": "Read a file before patching it; run_tests after every patch.",
            }, ttl_ms=3_600_000, cache_scope="public"))
        if method == "tools/list":
            tools = sorted((wire_tool(t) for t in TOOLS), key=lambda t: t["name"])  # deterministic order
            return self._ok(request_id, complete({"tools": tools}, ttl_ms=30_000, cache_scope="public"))
        if method == "tools/call":
            return self._call_tool(request_id, params)
        return rpc_error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

    def _call_tool(self, request_id, params: dict) -> dict:
        name, arguments = params.get("name"), params.get("arguments", {})
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:  # protocol error: the client asked for something that does not exist
            return rpc_error(request_id, INVALID_PARAMS, f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            return rpc_error(request_id, INVALID_PARAMS, "arguments must be an object")
        problems = schema_errors(tool["input_schema"], arguments)
        if problems:  # protocol error: checked before the tool runs, so a bad call never reaches a file
            return rpc_error(request_id, INVALID_PARAMS, "Invalid arguments", {"problems": problems})
        try:
            output = tool["executor"](self.workspace, arguments)
        except ValueError as e:  # the tool refused (path outside the workspace, patch text not found once, ...)
            return self._ok(request_id, text_result(str(e), is_error=True))
        except NotImplementedError:
            return self._ok(request_id, text_result(f"{name} is not implemented yet", is_error=True))
        text = output if isinstance(output, str) else json.dumps(output if output is not None else {"ok": True})
        return self._ok(request_id, text_result(text, is_error=False))

    @staticmethod
    def _ok(request_id, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def handle_line(self, line: str) -> str | None:
        """The stdio transport: one JSON message per line in, one per line out."""
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return json.dumps(rpc_error(None, PARSE_ERROR, "Parse error"))
        response = self.handle(message)
        return None if response is None else json.dumps(response, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", required=True, help="the folder the tools may read and change")
    args = parser.parse_args()
    server = McpServer(args.workspace)
    for line in sys.stdin:  # stdout carries protocol messages only; anything else would corrupt the stream
        if line.strip():
            out = server.handle_line(line)
            if out is not None:
                print(out, flush=True)


if __name__ == "__main__":
    main()
