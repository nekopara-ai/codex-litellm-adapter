import asyncio
import copy
import json
import re

import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from codex_litellm_adapter import server

AUTH = {"Authorization": "Bearer synthetic-test-key"}


def tool_payload():
    return {
        "model": "bridge-model",
        "store": False,
        "tools": [
            {
                "type": "namespace",
                "name": "demo",
                "tools": [{"type": "custom", "name": "echo", "format": {"type": "text"}}],
            }
        ],
        "input": [{"role": "user", "content": "Synthetic test input"}],
    }


def encode_events(events, multiline=False):
    return "".join(
        "id: test-event\nevent: "
        + event["type"]
        + "\n"
        + "\n".join(
            "data: " + line
            for line in json.dumps(event, indent=2 if multiline else None).splitlines()
        )
        + "\n\n"
        for event in events
    )


def decode_events(raw):
    return [
        server.decode_sse_json(frame.encode())
        for frame in raw.replace("\r\n", "\n").split("\n\n")
        if frame
    ]


@pytest_asyncio.fixture
async def gateway(monkeypatch, tmp_path):
    state = {
        "scenario": "function",
        "multiline": False,
        "received": [],
        "health_calls": 0,
        "entered": asyncio.Event(),
        "release": asyncio.Event(),
        "post_terminal_drained": asyncio.Event(),
    }

    async def upstream(request):
        if request.path == "/health/liveliness":
            state["health_calls"] += 1
            return web.json_response({"ok": True})
        if request.headers.get("Authorization") != AUTH["Authorization"]:
            return web.json_response({"error": {"message": "Unauthorized"}}, status=401)
        if request.path == "/hold":
            state["entered"].set()
            await state["release"].wait()
            return web.json_response({"ok": True})
        if request.path.startswith("/echo/"):
            return web.json_response({"raw_path": request.raw_path})
        if request.path == "/redirect":
            return web.Response(status=307, headers={"Location": server.UPSTREAM + "/ui/"})
        if request.path == "/v1/models":
            return web.json_response({"data": [{"id": "bridge-model"}]})
        if request.path in {"/v1/model/info", "/model_group/info"}:
            return web.json_response({"data": []})
        body = await request.json()
        state["received"].append(body)
        scenario = state["scenario"]
        if request.path == "/v1/chat/completions":
            return web.Response(
                text='data: {"choices":[]}\n\ndata: [DONE]\n\n', content_type="text/event-stream"
            )
        if scenario == "truncate":
            response = web.StreamResponse(
                headers={"Content-Type": "application/json", "Content-Length": "99999"}
            )
            await response.prepare(request)
            await response.write(b'{"id":"resp_truncated",')
            request.transport.close()
            return response
        custom = scenario == "custom"
        item = {
            "type": "custom_tool_call" if custom else "function_call",
            "id": "ctc_test" if custom else "fc_test",
            "call_id": "call_test",
            "name": "demo__echo",
        }
        item.update({"input": "hello"} if custom else {"arguments": '{"input":"hello"}'})
        completed = {
            "id": "resp_test_" + str(len(state["received"])),
            "object": "response",
            "status": "completed",
            "output": [item],
        }
        if not body.get("stream"):
            return web.json_response(completed)
        added = copy.deepcopy(item)
        added["input" if custom else "arguments"] = ""
        events = [{"type": "response.output_item.added", "output_index": 0, "item": added}]
        if not custom:
            events.extend(
                [
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_test",
                        "output_index": 0,
                        "delta": '{"input":"hello"}',
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": "fc_test",
                        "output_index": 0,
                        "arguments": '{"input":"hello"}',
                    },
                ]
            )
        events.extend(
            [
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": completed},
            ]
        )
        if scenario == "missing_terminal":
            events.pop()
        payload_text = encode_events(events, state["multiline"])
        if scenario == "post_terminal":
            # Emit the terminal event first, then keep the stream open with
            # guardrail-only traffic so the client is never blocked by it.
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            try:
                await response.write(payload_text.encode())
                for index in range(3):
                    await asyncio.sleep(0.05)
                    await response.write(
                        encode_events(
                            [{"type": "response.post_terminal_guardrail", "index": index}]
                        ).encode()
                    )
                state["post_terminal_drained"].set()
            except (ConnectionResetError, asyncio.CancelledError):
                raise
            await response.write_eof()
            return response
        return web.Response(text=payload_text, content_type="text/event-stream")

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", upstream)
    backend = TestServer(app)
    await backend.start_server()
    monkeypatch.setattr(server, "UPSTREAM", str(backend.make_url("")).rstrip("/"))
    monkeypatch.setattr(server, "UPSTREAM_POOL_LIMIT", 1)
    monkeypatch.setattr(server, "NS_FORCE_FLATTEN_PATTERN", re.compile(r"^bridge-model$"))
    monkeypatch.setenv("CODEX_ADAPTER_REPLAY_DB", str(tmp_path / "state" / "replay.sqlite3"))
    client = TestClient(TestServer(await server.create_app()))
    await client.start_server()
    try:
        yield client, state
    finally:
        state["release"].set()
        await client.close()
        await backend.close()
