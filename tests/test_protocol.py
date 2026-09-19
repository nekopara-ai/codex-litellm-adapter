import asyncio
import copy
import json
import re

import pytest
from aiohttp import WSMsgType
from conftest import AUTH, decode_events, encode_events, tool_payload

from codex_litellm_adapter import server


def test_custom_history_ids_and_pairing(monkeypatch):
    monkeypatch.setattr(server, "NS_FORCE_FLATTEN_PATTERN", re.compile(r"^bridge-model$"))
    payload = tool_payload()
    payload["input"] = [
        {
            "type": "custom_tool_call",
            "id": "ctc_test",
            "call_id": "call_test",
            "namespace": "demo",
            "name": "echo",
            "input": "hello",
        },
        {
            "type": "custom_tool_call_output",
            "id": "ctco_test",
            "call_id": "call_test",
            "output": "world",
        },
    ]
    raw, mapping = server.transform_responses_request(json.dumps(payload).encode())
    items = json.loads(raw)["input"]
    assert [item["type"] for item in items] == ["function_call", "function_call_output"]
    assert all("id" not in item and item["call_id"] == "call_test" for item in items)
    assert mapping[items[0]["name"]] == ("demo", "echo", "custom")


@pytest.mark.parametrize("kind", ["function_call", "function_call_output"])
@pytest.mark.parametrize("prefix", ["ctc_", "ctco_"])
def test_legacy_function_ids(monkeypatch, kind, prefix):
    monkeypatch.setattr(server, "NS_FORCE_FLATTEN_PATTERN", re.compile(r"^bridge-model$"))
    payload = {
        "model": "bridge-model",
        "input": [{"type": kind, "id": prefix + "test", "call_id": "call_test"}],
    }
    assert server.normalize_custom_tool_outputs(payload)
    assert "id" not in payload["input"][0]


def test_passthrough_does_not_rewrite_native_history():
    payload = tool_payload()
    payload["model"] = "gpt-example"
    raw = json.dumps(payload).encode()
    assert server.transform_responses_request(raw) == (raw, {})


def test_removed_tool_does_not_shadow_top_level(monkeypatch):
    monkeypatch.setattr(server, "NS_FORCE_FLATTEN_PATTERN", re.compile(r"^bridge-model$"))
    payload = {
        "model": "bridge-model",
        "tools": [{"type": "function", "name": "demo__echo", "parameters": {}}],
        "input": [
            {
                "type": "function_call",
                "namespace": "demo",
                "name": "echo",
                "call_id": "call_old",
                "arguments": "{}",
            }
        ],
    }
    mapping = server.flatten_namespace_tools(payload)
    assert payload["input"][0]["name"] != "demo__echo"
    current = {"type": "function_call", "name": "demo__echo", "arguments": "{}"}
    server.restore_namespaced_calls(current, mapping)
    assert current["name"] == "demo__echo" and "namespace" not in current


def test_non_string_call_type_is_ignored():
    """A malformed item type must not crash the recursive rewriter."""
    node = {"type": ["function_call"], "name": "demo__echo"}
    server.restore_namespaced_calls(node, {"demo__echo": ("demo", "echo", "function")})
    assert node["name"] == "demo__echo"


def test_history_only_namespace_is_stable():
    payload = {
        "model": "local-model",
        "input": [
            {"type": "function_call", "namespace": "demo", "name": "echo"},
            {"type": "function_call", "namespace": "demo", "name": "echo"},
        ],
    }
    mapping = server.flatten_namespace_tools(payload)
    assert len(mapping) == 1
    assert payload["input"][0]["name"] == payload["input"][1]["name"]


@pytest.mark.parametrize("kind", ["function_call", "custom_tool_call"])
@pytest.mark.parametrize("multiline", [False, True])
def test_sse_chunk_boundaries(kind, multiline):
    mapping = {"demo__echo": ("demo", "echo", "custom")}
    item = {
        "type": kind,
        "id": "ctc_test" if kind == "custom_tool_call" else "fc_test",
        "name": "demo__echo",
        "call_id": "call_test",
        "input": "hello",
        "arguments": '{"input":"hello"}',
    }
    raw = (
        encode_events(
            [{"type": "response.output_item.done", "output_index": 0, "item": item}], multiline
        )
        .replace("\n", "\r\n")
        .encode()
    )
    for width in [1, 7, 31, len(raw)]:
        rewriter = server.SSENamespaceRewriter(mapping)
        out = b"".join(rewriter.feed(raw[i : i + width]) for i in range(0, len(raw), width))
        out += rewriter.flush()
        event = decode_events(out.decode())[0]
        assert event["item"]["namespace"] == "demo"
        assert event["item"]["name"] == "echo"
        assert b"id: test-event" in out


async def test_chat_done_is_not_responses_failure(gateway):
    client, _ = gateway
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "bridge-model", "messages": [], "stream": True},
    )
    raw = await response.text()
    assert response.status == 200 and raw.endswith("data: [DONE]\n\n")
    assert "upstream_incomplete_stream" not in raw


@pytest.mark.parametrize("scenario", ["function", "custom"])
async def test_json_tool_restoration(gateway, scenario):
    client, state = gateway
    state["scenario"] = scenario
    response = await client.post("/v1/responses", headers=AUTH, json=tool_payload())
    item = (await response.json())["output"][0]
    assert response.status == 200
    assert (item["type"], item["namespace"], item["name"], item["input"]) == (
        "custom_tool_call",
        "demo",
        "echo",
        "hello",
    )


@pytest.mark.parametrize("scenario", ["function", "custom"])
@pytest.mark.parametrize("multiline", [False, True])
@pytest.mark.parametrize("transport", ["http", "websocket"])
async def test_transport_tool_parity(gateway, scenario, multiline, transport):
    client, state = gateway
    state.update(scenario=scenario, multiline=multiline)
    body = tool_payload()
    if transport == "http":
        response = await client.post("/v1/responses", headers=AUTH, json={**body, "stream": True})
        assert response.status == 200
        events = decode_events(await response.text())
    else:
        events = []
        async with client.ws_connect("/v1/responses", headers=AUTH) as ws:
            await ws.send_json({"type": "response.create", **body})
            for _ in range(12):
                frame = await asyncio.wait_for(ws.receive(), 3)
                assert frame.type == WSMsgType.TEXT
                event = json.loads(frame.data)
                events.append(event)
                if event["type"] in server.TERMINAL_RESPONSE_EVENTS:
                    break
    assert events[-1]["type"] == "response.completed"
    assert not any(e["type"] == "response.function_call_arguments.delta" for e in events)
    for event in events:
        if "item" in event:
            assert event["item"]["name"] == "echo"
            assert event["item"]["namespace"] == "demo"
    assert events[-1]["response"]["output"][0]["input"] == "hello"


async def test_missing_terminal_is_signalled(gateway):
    client, state = gateway
    state["scenario"] = "missing_terminal"
    response = await client.post(
        "/v1/responses", headers=AUTH, json={**tool_payload(), "stream": True}
    )
    raw = await response.text()
    assert "upstream_incomplete_stream" in raw


@pytest.mark.parametrize("transport", ["http", "websocket"])
async def test_post_terminal_stream_is_drained(gateway, monkeypatch, transport):
    """Guardrail traffic after the terminal event is consumed in the background."""
    client, state = gateway
    state["scenario"] = "post_terminal"
    await server.cancel_active_drains()
    body = tool_payload()
    if transport == "http":
        response = await client.post("/v1/responses", headers=AUTH, json={**body, "stream": True})
        events = decode_events(await response.text())
    else:
        events = []
        async with client.ws_connect("/v1/responses", headers=AUTH) as ws:
            await ws.send_json({"type": "response.create", **body})
            for _ in range(12):
                frame = await asyncio.wait_for(ws.receive(), 3)
                assert frame.type == WSMsgType.TEXT
                event = json.loads(frame.data)
                events.append(event)
                if event["type"] in server.TERMINAL_RESPONSE_EVENTS:
                    break
    assert events[-1]["type"] == "response.completed"
    # The client-visible stream stops at the terminal event; the guardrail-only
    # frames must not leak into it.
    assert not any(e["type"] == "response.post_terminal_guardrail" for e in events)
    await asyncio.wait_for(state["post_terminal_drained"].wait(), 3)
    await server.cancel_active_drains()


async def test_drain_limit_skips_and_closes(gateway, monkeypatch):
    client, state = gateway
    state["scenario"] = "post_terminal"
    monkeypatch.setattr(server, "MAX_ACTIVE_DRAINS", 1)
    await server.cancel_active_drains()
    response = await client.post(
        "/v1/responses", headers=AUTH, json={**tool_payload(), "stream": True}
    )
    await response.text()
    await asyncio.sleep(0)
    assert len(server._ACTIVE_DRAIN_TASKS) <= 1
    await server.cancel_active_drains()


async def test_truncated_json_is_structured_502(gateway):
    client, state = gateway
    state["scenario"] = "truncate"
    response = await client.post("/v1/responses", headers=AUTH, json=tool_payload())
    assert response.status == 502
    assert (await response.json())["error"]["type"] == "upstream_error"
    assert not client.server.app["session"].connector._acquired


@pytest.mark.parametrize("path", ["/echo/a%2Fb?value=a%2Bb", "/echo/a%3Fb?value=x%26y"])
async def test_encoded_path_is_preserved(gateway, path):
    client, _ = gateway
    response = await client.get(path, headers=AUTH)
    assert (await response.json())["raw_path"] == path


async def test_internal_redirect_becomes_relative(gateway):
    client, _ = gateway
    response = await client.get("/redirect", headers=AUTH, allow_redirects=False)
    assert response.status == 307 and response.headers["Location"] == "/ui/"
    assert server.public_redirect_location("https://example.com/help") == "https://example.com/help"


async def test_health_uses_independent_pool(gateway):
    client, state = gateway
    hold = asyncio.create_task(client.get("/hold", headers=AUTH))
    await asyncio.wait_for(state["entered"].wait(), 2)
    try:
        response = await asyncio.wait_for(client.get("/health"), 2)
        assert response.status == 200 and state["health_calls"] == 1
        await response.read()
    finally:
        state["release"].set()
        await (await hold).read()


async def test_catalog_auth_and_etag(gateway):
    client, _ = gateway
    path = "/v1/models?client_version=test"
    for headers in [{}, {"Authorization": "Bearer synthetic-wrong-key"}]:
        response = await client.get(path, headers=headers)
        assert response.status == 401
        await response.read()
    response = await client.get(path, headers=AUTH)
    assert response.status == 200
    assert (await response.json())["models"][0]["slug"] == "bridge-model"
    etag = response.headers["ETag"]
    response = await client.get(path, headers={**AUTH, "If-None-Match": etag})
    assert response.status == 304
    response = await client.get(path, headers={"If-None-Match": etag})
    assert response.status == 401


async def test_zstd_request_decoding(gateway):
    try:
        from compression import zstd
    except ImportError:
        from backports import zstd
    client, state = gateway
    payload = tool_payload()
    response = await client.post(
        "/v1/responses",
        data=zstd.compress(json.dumps(payload).encode()),
        headers={**AUTH, "Content-Type": "application/json", "Content-Encoding": "zstd"},
    )
    assert response.status == 200
    await response.read()
    assert state["received"][-1]["model"] == payload["model"]


def test_reasoning_compatibility_is_explicit(monkeypatch):
    payload = {"model": "example", "reasoning": {"effort": "ultra"}}
    assert not server.normalize_reasoning_effort(copy.deepcopy(payload))
    monkeypatch.setattr(server, "REASONING_EFFORT_MAP", {"ultra": "high"})
    assert server.normalize_reasoning_effort(payload)
    assert payload["reasoning"]["effort"] == "high"
