import json
import socket
import threading
import time
import urllib.error
import urllib.request

_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen(request, timeout):
    return _HTTP_OPENER.open(request, timeout=timeout)


from typing import Optional

import pytest

pytest.importorskip("chromadb")

from mempalace import mcp_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_dispatch(request):
    method = request.get("method")
    req_id = request.get("id")

    if method == "initialize":
        params = request.get("params") or {}
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2025-11-25"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mempalace", "version": "test"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        tools = [
            {
                "name": f"tool_{idx}",
                "description": "test tool",
                "inputSchema": {"type": "object", "properties": {}},
            }
            for idx in range(128)
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    if method == "notifications/initialized":
        return None

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": "Method not found"},
    }


@pytest.fixture(scope="module")
def http_port():
    original_handle_request = mcp_server.handle_request
    mcp_server.handle_request = _fake_dispatch

    port = _free_port()
    thread = threading.Thread(
        target=mcp_server._serve_http,
        args=("127.0.0.1", port),
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 20
    last_error = None
    url = f"http://127.0.0.1:{port}/healthz"

    while time.monotonic() < deadline:
        try:
            with _urlopen(url, timeout=1) as resp:
                body = resp.read().decode("utf-8")
            if resp.status == 200 and body == "ok\n":
                break
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    else:
        mcp_server.handle_request = original_handle_request
        raise AssertionError(f"HTTP server did not become ready: {last_error!r}")

    yield port

    # The HTTP server thread is daemonized and intentionally left alone.
    # Restoring the dispatcher keeps the rest of the suite isolated.
    mcp_server.handle_request = original_handle_request


def _rpc(port: int, method: str, params: Optional[dict] = None, req_id: int = 1):
    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": method,
        "params": params or {},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/mcp",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _urlopen(request, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body) if body else None


def test_parse_args_defaults_to_stdio(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mempalace-mcp"])

    args = mcp_server._parse_args()

    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_parse_args_accepts_http_transport(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "mempalace-mcp",
            "--transport",
            "http",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
        ],
    )

    args = mcp_server._parse_args()

    assert args.transport == "http"
    assert args.host == "0.0.0.0"
    assert args.port == 9999


def test_http_transport_serves_healthz(http_port):
    with _urlopen(f"http://127.0.0.1:{http_port}/healthz", timeout=10) as resp:
        body = resp.read().decode("utf-8")

    assert resp.status == 200
    assert body == "ok\n"


def test_http_transport_serves_initialize_ping_and_repeated_tools_list(http_port):
    status, initialized = _rpc(
        http_port,
        "initialize",
        {"protocolVersion": "2025-11-25"},
        req_id=1,
    )
    assert status == 200
    assert initialized["result"]["protocolVersion"] == "2025-11-25"

    status, ping = _rpc(http_port, "ping", {}, req_id=2)
    assert status == 200
    assert ping["result"] == {}

    status, first = _rpc(http_port, "tools/list", {}, req_id=3)
    assert status == 200
    tools = first["result"]["tools"]
    assert len(tools) == 128
    assert all("name" in tool and "inputSchema" in tool for tool in tools)

    # Regression shape for #1801: repeated large tools/list frames should
    # keep succeeding over HTTP without relying on stdio framing.
    for req_id in range(4, 12):
        status, payload = _rpc(http_port, "tools/list", {}, req_id=req_id)
        assert status == 200
        assert payload["id"] == req_id
        assert payload["result"]["tools"] == tools


def test_http_transport_returns_parse_error_for_invalid_json(http_port):
    request = urllib.request.Request(
        f"http://127.0.0.1:{http_port}/mcp",
        data=b"not-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _urlopen(request, timeout=10)

    body = excinfo.value.read().decode("utf-8")
    payload = json.loads(body)

    assert excinfo.value.code == 400
    assert payload["error"]["code"] == -32700
    assert payload["error"]["message"] == "Parse error"


def test_http_transport_accepts_notifications_without_body(http_port):
    payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{http_port}/mcp",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with _urlopen(request, timeout=10) as resp:
        body = resp.read()

    assert resp.status == 202
    assert body == b""


def test_http_transport_returns_404_for_unknown_path(http_port):
    request = urllib.request.Request(
        f"http://127.0.0.1:{http_port}/not-mcp",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _urlopen(request, timeout=10)

    assert excinfo.value.code == 404
