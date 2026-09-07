import asyncio
import dataclasses
import sqlite3
import time
from contextlib import closing

import pytest

from codex_litellm_adapter import response_replay as replay

AUTH = {"authorization": "Bearer synthetic-test-key"}


def payload():
    return {
        "model": "example-model",
        "store": False,
        "input": "hello",
        "instructions": "Synthetic instructions",
    }


def response(identifier="resp_test"):
    return {
        "id": identifier,
        "output": [
            {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "echo",
                "arguments": "{}",
            }
        ],
    }


def continuation():
    return {
        "model": "example-model",
        "store": False,
        "previous_response_id": "resp_test",
        "input": [{"type": "function_call_output", "call_id": "call_test", "output": "ok"}],
    }


def test_completed_output_is_authoritative():
    collector = replay.CompletedResponseCollector()
    collector.consume(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "custom_tool_call", "input": ""},
        }
    )
    final = collector.consume(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_final",
                "output": [{"type": "custom_tool_call", "input": "complete"}],
            },
        }
    )
    assert final["output"][0]["input"] == "complete"


def test_collector_fills_missing_output_items():
    collector = replay.CompletedResponseCollector()
    collector.consume(
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {"type": "function_call", "arguments": "{}"},
        }
    )
    final = collector.consume(
        {
            "type": "response.completed",
            "response": {"id": "resp_final", "output": [{"type": "reasoning", "summary": []}]},
        }
    )
    assert [item["type"] for item in final["output"]] == ["reasoning", "function_call"]


@pytest.mark.parametrize("persistent", [False, True])
async def test_replay_scope_and_restart(tmp_path, persistent):
    path = str(tmp_path / "state" / "replay.sqlite3") if persistent else None
    store = replay.ResponseReplayStore(path)
    await store.initialize()
    context = await store.prepare(payload(), AUTH)
    await store.save(context, response())
    if persistent:
        await store.close()
        store = replay.ResponseReplayStore(path)
        await store.initialize()
    try:
        next_payload = continuation()
        restored = await store.prepare(next_payload, AUTH)
        assert restored.replayed and "previous_response_id" not in next_payload
        assert next_payload["instructions"] == "Synthetic instructions"
        assert [i.get("type") for i in next_payload["input"]][-2:] == [
            "function_call",
            "function_call_output",
        ]
        for headers, model in [
            ({"authorization": "Bearer synthetic-other-key"}, "example-model"),
            (AUTH, "other-model"),
        ]:
            foreign = continuation()
            foreign["model"] = model
            with pytest.raises(replay.ReplayUnavailable):
                await store.prepare(foreign, headers)
        if persistent:
            with closing(sqlite3.connect(path)) as db:
                scope = db.execute("SELECT scope FROM response_history").fetchone()[0]
            assert "synthetic-test-key" not in scope
    finally:
        await store.close()


async def test_orphaned_output_requires_full_history():
    store = replay.ResponseReplayStore()
    await store.initialize()
    try:
        with pytest.raises(replay.ReplayUnavailable):
            await store.prepare(continuation(), AUTH)
        complete = continuation()
        complete["input"].insert(0, response()["output"][0])
        assert await store.prepare(complete, AUTH) is not None
        assert "previous_response_id" not in complete
    finally:
        await store.close()


async def test_failed_pending_expiry_recovers_capacity():
    store = replay.ResponseReplayStore(pending_max_entries=2, ttl_seconds=60)
    await store.initialize()
    context = await store.prepare(payload(), AUTH)
    persist = store._persist_encoded

    async def fail(*args):
        raise replay.ReplayPersistenceError("Synthetic storage failure")

    store._persist_encoded = fail
    try:
        for i in range(2):
            store.queue_save(context, response("resp_failed_" + str(i)))
        assert await store.flush(timeout=3)
        assert store.pending_entries == 2 and store.queued_writes == 0
        assert not store.healthy
        store._persist_encoded = persist
        for key, pending in list(store.pending.items()):
            store.pending[key] = dataclasses.replace(pending, expires_at=int(time.time()) - 1)
        assert store.queue_save(context, response("resp_recovered"))
        assert await store.flush(timeout=3)
        assert store.pending_entries == 0 and store.pending_bytes == 0 and store.healthy
    finally:
        await store.close()


async def test_storage_recovery_retries_without_new_request():
    store = replay.ResponseReplayStore(ttl_seconds=60)
    await store.initialize()
    context = await store.prepare(payload(), AUTH)
    persist = store._persist_encoded

    async def fail(*args):
        raise replay.ReplayPersistenceError("Synthetic storage failure")

    store._persist_encoded = fail
    try:
        store.queue_save(context, response())
        assert await store.flush(timeout=3)
        assert store.pending_entries == 1
        # The pending copy keeps the continuation available during the outage.
        assert (await store.prepare(continuation(), AUTH)).replayed
        store._persist_encoded = persist
        async with asyncio.timeout(8):
            while store.pending_entries:
                await asyncio.sleep(0.1)
        assert store.completed_writes == 1 and store.consecutive_write_errors == 0
        assert (await store.prepare(continuation(), AUTH)).replayed
    finally:
        await store.close()


async def test_sqlite_corruption_and_expiry_are_cache_misses(tmp_path):
    path = str(tmp_path / "state" / "replay.sqlite3")
    store = replay.ResponseReplayStore(path)
    await store.initialize()
    context = await store.prepare(payload(), AUTH)
    try:
        await store.save(context, response())
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE response_history SET history = ?", (b"invalid compressed data",))
        assert await store.get(context.scope, context.model, "resp_test") is None
        await store.save(context, response())
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE response_history SET expires_at = 0")
        assert await store.get(context.scope, context.model, "resp_test") is None
    finally:
        await store.close()
