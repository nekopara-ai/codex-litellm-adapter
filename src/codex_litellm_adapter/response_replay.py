#!/usr/bin/env python3
"""Persistent full-history replay for stateless OpenAI Responses continuations."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import sys
import time
import zlib
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("response_replay")


TOOL_CALL_TYPES = {
    "custom_tool_call",
    "function_call",
    "computer_call",
    "local_shell_call",
    "mcp_call",
}
TOOL_OUTPUT_TYPES = {
    "custom_tool_call_output",
    "function_call_output",
    "computer_call_output",
    "local_shell_call_output",
    "mcp_call_output",
}
AUTH_SCOPE_HEADERS = (
    "authorization",
    "x-api-key",
    "api-key",
    "x-litellm-api-key",
    "openai-organization",
    "openai-project",
)
REPLAY_REQUEST_FIELDS = (
    "tools",
    "instructions",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "include",
)


class ReplayUnavailable(Exception):
    """The caller referenced stateless history that this adapter cannot recover."""


class ReplayPersistenceError(Exception):
    """A completed response could not be saved within the configured bounds."""


@dataclass(frozen=True)
class ReplayContext:
    scope: str
    model: str
    history: list[Any]
    request_state: dict[str, Any]
    continued_from: str | None = None
    replayed: bool = False


@dataclass(frozen=True)
class ReplaySnapshot:
    history: list[Any]
    request_state: dict[str, Any]


@dataclass(frozen=True)
class PendingReplayWrite:
    key: tuple[str, str, str]
    context: ReplayContext
    response_id: str
    expires_at: int
    response: dict[str, Any]
    estimated_bytes: int
    token: int


def _input_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if value is None:
        return []
    return [copy.deepcopy(value)]


def _is_self_contained_tool_history(items: list[Any]) -> bool:
    """Accept only explicit call/output pairs when no cached snapshot exists."""
    calls: set[str] = set()
    outputs = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type in TOOL_CALL_TYPES and isinstance(call_id, str) and call_id:
            calls.add(call_id)
        elif item_type in TOOL_OUTPUT_TYPES:
            outputs += 1
            if not isinstance(call_id, str) or call_id not in calls:
                return False
    return outputs > 0


def _merge_without_duplicate(history: list[Any], current: list[Any]) -> list[Any]:
    """Append current input while removing an exact suffix/prefix overlap."""
    maximum = min(len(history), len(current))
    overlap = 0
    for width in range(maximum, 0, -1):
        if history[-width:] == current[:width]:
            overlap = width
            break
    return copy.deepcopy(history) + copy.deepcopy(current[overlap:])


def _request_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {name: copy.deepcopy(payload[name]) for name in REPLAY_REQUEST_FIELDS if name in payload}


def _estimated_object_bytes(*values: Any) -> int:
    """Estimate retained Python memory without serializing large strings."""
    seen: set[int] = set()
    stack = list(values)
    total = 0
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(value)
        if isinstance(value, dict):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            stack.extend(value)
    return total


class CompletedResponseCollector:
    """Rebuild response.output when a stream omits it from response.completed."""

    def __init__(self) -> None:
        self.output_items: dict[int, dict[str, Any]] = {}

    def consume(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = event.get("type")
        if event_type in {"response.output_item.done", "response.output_item.added"}:
            item = event.get("item")
            index = event.get("output_index")
            if isinstance(item, dict) and isinstance(index, int) and index >= 0:
                if event_type == "response.output_item.done" or index not in self.output_items:
                    self.output_items[index] = copy.deepcopy(item)
            return None
        if event_type != "response.completed":
            return None
        response = event.get("response")
        if not isinstance(response, dict):
            return None
        completed = copy.deepcopy(response)
        output = completed.get("output")
        merged = copy.deepcopy(output) if isinstance(output, list) else []
        for index, item in sorted(self.output_items.items()):
            if index < len(merged) and isinstance(merged[index], dict):
                continue  # The completed response is authoritative.
            while len(merged) <= index:
                merged.append(None)
            merged[index] = copy.deepcopy(item)
        completed["output"] = [item for item in merged if isinstance(item, dict)]
        return completed


class ResponseReplayStore:
    """Bounded, auth-scoped mapping from response ID to full input history."""

    def __init__(
        self,
        path: str | None = None,
        *,
        ttl_seconds: int = 86400,
        max_entries: int = 4096,
        max_bytes: int = 1024 * 1024**2,
        max_entry_bytes: int = 128 * 1024**2,
        max_raw_bytes: int = 256 * 1024**2,
        pending_max_entries: int = 256,
        pending_max_bytes: int = 256 * 1024**2,
        unhealthy_after_consecutive_write_errors: int = 3,
    ) -> None:
        if (
            ttl_seconds <= 0
            or max_entries <= 0
            or max_bytes <= 0
            or max_entry_bytes <= 0
            or max_raw_bytes <= 0
            or pending_max_entries <= 0
            or pending_max_bytes <= 0
            or unhealthy_after_consecutive_write_errors <= 0
            or max_entry_bytes > max_bytes
        ):
            raise ValueError("response replay limits must be positive and consistent")
        self.path = Path(path) if path else None
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.max_entry_bytes = max_entry_bytes
        self.max_raw_bytes = max_raw_bytes
        self.pending_max_entries = pending_max_entries
        self.pending_max_bytes = pending_max_bytes
        self.unhealthy_after_consecutive_write_errors = unhealthy_after_consecutive_write_errors
        self.secret: bytes | None = None
        self.last_error: str | None = None
        self.last_write_error: str | None = None
        self.write_errors = 0
        self.consecutive_write_errors = 0
        self.dropped_writes = 0
        self.completed_writes = 0
        self.lock = asyncio.Lock()
        self.db_lock = asyncio.Lock()
        self.memory: OrderedDict[tuple[str, str, str], tuple[int, bytes]] = OrderedDict()
        self.memory_bytes = 0
        self.pending: OrderedDict[tuple[str, str, str], PendingReplayWrite] = OrderedDict()
        self.pending_bytes = 0
        self.failed_pending: dict[tuple[str, str, str], float] = {}
        self.next_token = 0
        self.write_queue: asyncio.Queue[PendingReplayWrite] = asyncio.Queue(
            maxsize=pending_max_entries
        )
        self.writer_task: asyncio.Task[None] | None = None
        self.closing = False

    @classmethod
    def from_environment(cls) -> "ResponseReplayStore":
        return cls(
            os.environ.get("CODEX_ADAPTER_REPLAY_DB", "").strip() or None,
            ttl_seconds=int(os.environ.get("CODEX_ADAPTER_REPLAY_TTL_SECONDS", "86400")),
            max_entries=int(os.environ.get("CODEX_ADAPTER_REPLAY_MAX_ENTRIES", "4096")),
            max_bytes=int(os.environ.get("CODEX_ADAPTER_REPLAY_MAX_BYTES", str(1024 * 1024**2))),
            max_entry_bytes=int(
                os.environ.get("CODEX_ADAPTER_REPLAY_MAX_ENTRY_BYTES", str(128 * 1024**2))
            ),
            max_raw_bytes=int(
                os.environ.get("CODEX_ADAPTER_REPLAY_MAX_RAW_BYTES", str(256 * 1024**2))
            ),
            pending_max_entries=int(
                os.environ.get("CODEX_ADAPTER_REPLAY_PENDING_MAX_ENTRIES", "256")
            ),
            pending_max_bytes=int(
                os.environ.get("CODEX_ADAPTER_REPLAY_PENDING_MAX_BYTES", str(256 * 1024**2))
            ),
            unhealthy_after_consecutive_write_errors=int(
                os.environ.get("CODEX_ADAPTER_REPLAY_UNHEALTHY_AFTER_WRITE_ERRORS", "3")
            ),
        )

    @property
    def mode(self) -> str:
        return "sqlite" if self.path is not None else "memory"

    @property
    def healthy(self) -> bool:
        self._purge_expired_pending()
        return (
            self.writer_alive
            and not self.closing
            and self.consecutive_write_errors < self.unhealthy_after_consecutive_write_errors
            and len(self.pending) < self.pending_max_entries
            and self.pending_bytes < self.pending_max_bytes
        )

    @property
    def writer_alive(self) -> bool:
        return self.writer_task is None or not self.writer_task.done()

    @property
    def queued_writes(self) -> int:
        return self.write_queue.qsize()

    @property
    def pending_entries(self) -> int:
        return len(self.pending)

    async def initialize(self) -> None:
        if self.path is None:
            self.secret = os.urandom(32)
            return
        await asyncio.to_thread(self._initialize_sqlite)

    def _initialize_sqlite(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        key_path = self.path.with_suffix(self.path.suffix + ".key")
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(os.urandom(32))
                handle.flush()
                os.fsync(handle.fileno())
        secret = key_path.read_bytes()
        if len(secret) != 32:
            raise ReplayPersistenceError("response replay key has an invalid size")
        os.chmod(key_path, 0o600)
        self.secret = secret
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_history (
                    scope TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    history BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    accessed_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY (scope, model, response_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS response_history_expires "
                "ON response_history(expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS response_history_lru "
                "ON response_history(accessed_at, created_at)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_history_sizes (
                    scope TEXT NOT NULL,
                    model TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    history_bytes INTEGER NOT NULL,
                    PRIMARY KEY (scope, model, response_id)
                )
                """
            )
            # Rebuild write triggers atomically. Explicit UPSERT avoids the
            # outer statement's conflict policy overriding INSERT OR REPLACE.
            for name in ("response_history_size_insert", "response_history_size_update"):
                connection.execute(f"DROP TRIGGER IF EXISTS {name}")
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS response_history_size_insert
                AFTER INSERT ON response_history
                BEGIN
                    INSERT INTO response_history_sizes(
                        scope, model, response_id, history_bytes
                    ) VALUES (
                        NEW.scope, NEW.model, NEW.response_id, length(NEW.history)
                    ) ON CONFLICT(scope, model, response_id) DO UPDATE SET
                        history_bytes = excluded.history_bytes;
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS response_history_size_update
                AFTER UPDATE OF history ON response_history
                BEGIN
                    INSERT INTO response_history_sizes(
                        scope, model, response_id, history_bytes
                    ) VALUES (
                        NEW.scope, NEW.model, NEW.response_id, length(NEW.history)
                    ) ON CONFLICT(scope, model, response_id) DO UPDATE SET
                        history_bytes = excluded.history_bytes;
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS response_history_size_delete
                AFTER DELETE ON response_history
                BEGIN
                    DELETE FROM response_history_sizes
                    WHERE scope = OLD.scope
                      AND model = OLD.model
                      AND response_id = OLD.response_id;
                END
                """
            )
            history_count = int(
                connection.execute("SELECT COUNT(*) FROM response_history").fetchone()[0]
            )
            size_count = int(
                connection.execute("SELECT COUNT(*) FROM response_history_sizes").fetchone()[0]
            )
            if history_count != size_count:
                connection.execute("DELETE FROM response_history_sizes")
                connection.execute(
                    """
                    INSERT INTO response_history_sizes(
                        scope, model, response_id, history_bytes
                    )
                    SELECT scope, model, response_id, length(history)
                    FROM response_history
                    """
                )
            connection.execute(
                "DELETE FROM response_history WHERE expires_at <= ?", (int(time.time()),)
            )
        os.chmod(self.path, 0o600)
        self._secure_sqlite_files()

    def _secure_sqlite_files(self) -> None:
        if self.path is None:
            return
        for candidate in (
            self.path,
            Path(str(self.path) + "-wal"),
            Path(str(self.path) + "-shm"),
            self.path.with_suffix(self.path.suffix + ".key"),
        ):
            try:
                os.chmod(candidate, 0o600)
            except FileNotFoundError:
                pass

    def _scope(self, headers: Any) -> str | None:
        if self.secret is None:
            raise RuntimeError("response replay store is not initialized")
        fields: list[tuple[str, str]] = []
        for name in AUTH_SCOPE_HEADERS:
            value = headers.get(name)
            if isinstance(value, str) and value:
                fields.append((name, value))
        if not fields:
            return None
        material = b"".join(
            name.encode() + b"\0" + value.encode() + b"\0" for name, value in fields
        )
        return hmac.new(self.secret, material, hashlib.sha256).hexdigest()

    async def prepare(self, payload: dict[str, Any], headers: Any) -> ReplayContext | None:
        if payload.get("store") is not False:
            return None
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            return None
        items = _input_items(payload.get("input"))
        previous_id = payload.get("previous_response_id")
        scope = self._scope(headers)
        replayed = False
        if isinstance(previous_id, str) and previous_id:
            if scope is None:
                raise ReplayUnavailable("missing authorization scope")
            snapshot = await self.get(scope, model, previous_id)
            if snapshot is not None:
                items = _merge_without_duplicate(snapshot.history, items)
                for name, value in snapshot.request_state.items():
                    if name not in payload:
                        payload[name] = copy.deepcopy(value)
                replayed = True
            elif not _is_self_contained_tool_history(items):
                raise ReplayUnavailable("previous response history is unavailable")
            payload["input"] = copy.deepcopy(items)
            payload.pop("previous_response_id", None)
        if scope is None:
            return None
        return ReplayContext(
            scope=scope,
            model=model,
            history=items,
            request_state=_request_state(payload),
            continued_from=previous_id if isinstance(previous_id, str) else None,
            replayed=replayed,
        )

    def _response_identity(
        self, context: ReplayContext | None, response: dict[str, Any]
    ) -> tuple[str, int] | None:
        if context is None:
            return None
        response_id = response.get("id")
        output = response.get("output")
        if not isinstance(response_id, str) or not response_id or not isinstance(output, list):
            return None
        return response_id, int(time.time()) + self.ttl_seconds

    def _snapshot_from_parts(
        self, context: ReplayContext, response: dict[str, Any]
    ) -> ReplaySnapshot:
        output = response["output"]
        return ReplaySnapshot(
            history=copy.deepcopy(context.history) + copy.deepcopy(output),
            request_state=copy.deepcopy(context.request_state),
        )

    def _encode_snapshot(
        self, context: ReplayContext | None, response: dict[str, Any]
    ) -> tuple[str, int, bytes] | None:
        identity = self._response_identity(context, response)
        if identity is None or context is None:
            return None
        response_id, expires_at = identity
        output = response["output"]
        snapshot = {
            "version": 1,
            "history": context.history + copy.deepcopy(output),
            "request_state": context.request_state,
        }
        raw = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False).encode()
        if len(raw) > self.max_raw_bytes:
            raise ReplayPersistenceError("response replay snapshot exceeds its raw size limit")
        return response_id, expires_at, raw

    def _decode_snapshot(self, raw: bytes) -> ReplaySnapshot | None:
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(decoded, dict) or decoded.get("version") != 1:
            return None
        history = decoded.get("history")
        request_state = decoded.get("request_state")
        if not isinstance(history, list) or not isinstance(request_state, dict):
            return None
        return ReplaySnapshot(history=history, request_state=request_state)

    async def _persist_encoded(
        self,
        context: ReplayContext,
        response_id: str,
        expires_at: int,
        raw: bytes,
    ) -> None:
        compressed = await asyncio.to_thread(zlib.compress, raw, 6)
        if len(compressed) > self.max_entry_bytes:
            raise ReplayPersistenceError("response replay snapshot exceeds its size limit")
        async with self.db_lock:
            try:
                if self.path is None:
                    self._memory_put(context, response_id, expires_at, compressed)
                else:
                    await asyncio.to_thread(
                        self._sqlite_put,
                        context,
                        response_id,
                        expires_at,
                        compressed,
                    )
            except sqlite3.Error as exc:
                self.last_error = type(exc).__name__
                raise ReplayPersistenceError("response replay database write failed") from exc

    async def save(self, context: ReplayContext | None, response: dict[str, Any]) -> bool:
        encoded = await asyncio.to_thread(self._encode_snapshot, context, response)
        if encoded is None or context is None:
            return False
        response_id, expires_at, raw = encoded
        await self._persist_encoded(context, response_id, expires_at, raw)
        self.last_error = None
        self.completed_writes += 1
        return True

    def queue_save(self, context: ReplayContext | None, response: dict[str, Any]) -> bool:
        identity = self._response_identity(context, response)
        if identity is None or context is None:
            return False
        response_id, expires_at = identity
        if self.closing:
            raise ReplayPersistenceError("response replay writer is closing")

        self._purge_expired_pending()
        self._schedule_failed_retries()
        key = (context.scope, context.model, response_id)
        existing = self.pending.get(key)
        if existing is not None:
            self.dropped_writes += 1
            raise ReplayPersistenceError("duplicate response replay write is pending")
        estimated_bytes = _estimated_object_bytes(
            context.history,
            context.request_state,
            response.get("output"),
        )
        pending_entries = len(self.pending) + 1
        pending_bytes = self.pending_bytes + estimated_bytes
        if pending_entries > self.pending_max_entries:
            self.dropped_writes += 1
            raise ReplayPersistenceError("response replay pending entry limit exceeded")
        if estimated_bytes > self.max_raw_bytes or pending_bytes > self.pending_max_bytes:
            self.dropped_writes += 1
            raise ReplayPersistenceError("response replay pending byte limit exceeded")
        if self.write_queue.full():
            self.dropped_writes += 1
            raise ReplayPersistenceError("response replay write queue is full")

        self.next_token += 1
        token = self.next_token
        item = PendingReplayWrite(
            key=key,
            context=context,
            response_id=response_id,
            expires_at=expires_at,
            response=response,
            estimated_bytes=estimated_bytes,
            token=token,
        )
        self.pending[key] = item
        self.pending.move_to_end(key)
        self.pending_bytes = pending_bytes
        self.write_queue.put_nowait(item)
        self._ensure_writer()
        return True

    def _ensure_writer(self) -> None:
        if self.writer_task is None or self.writer_task.done():
            self.writer_task = asyncio.create_task(
                self._writer_loop(), name="response-replay-writer"
            )

    def _drop_pending(self, item: PendingReplayWrite) -> None:
        current = self.pending.get(item.key)
        if current is None or current.token != item.token:
            return
        self.pending.pop(item.key, None)
        self.failed_pending.pop(item.key, None)
        self.pending_bytes -= current.estimated_bytes

    def _purge_expired_pending(self) -> None:
        # Only failed, unqueued items can be freed without retaining an active
        # writer reference outside the memory budget.
        now = int(time.time())
        for key in list(self.failed_pending):
            item = self.pending.get(key)
            if item is None:
                self.failed_pending.pop(key, None)
            elif item.expires_at <= now:
                self._drop_pending(item)

    def _schedule_failed_retries(self) -> None:
        self._purge_expired_pending()
        if self.closing:
            return
        now = time.monotonic()
        for key, retry_at in list(self.failed_pending.items()):
            if self.write_queue.full():
                break
            item = self.pending.get(key)
            if item is not None and retry_at <= now:
                self.failed_pending.pop(key, None)
                self.write_queue.put_nowait(item)

    async def _writer_loop(self) -> None:
        while True:
            if self.failed_pending:
                self._schedule_failed_retries()
                try:
                    item = await asyncio.wait_for(self.write_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
            else:
                item = await self.write_queue.get()
            succeeded = False
            try:
                encoded = await asyncio.to_thread(
                    self._encode_snapshot, item.context, item.response
                )
                if encoded is None:
                    raise ReplayPersistenceError(
                        "completed response did not contain replayable output"
                    )
                response_id, expires_at, raw = encoded
                for attempt in range(3):
                    try:
                        await self._persist_encoded(
                            item.context,
                            response_id,
                            expires_at,
                            raw,
                        )
                    except (OSError, ReplayPersistenceError) as exc:
                        self.last_error = type(exc).__name__
                        if attempt == 2:
                            raise
                        await asyncio.sleep((0.05, 0.2)[attempt])
                    else:
                        succeeded = True
                        self.last_error = None
                        self.consecutive_write_errors = 0
                        self.completed_writes += 1
                        break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.write_errors += 1
                self.consecutive_write_errors += 1
                self.last_error = type(exc).__name__
                cause = exc
                while cause.__cause__ is not None:
                    cause = cause.__cause__
                sqlite_error_name = getattr(cause, "sqlite_errorname", None)
                self.last_write_error = "/".join(
                    part
                    for part in (
                        type(exc).__name__,
                        type(cause).__name__ if cause is not exc else None,
                        sqlite_error_name,
                    )
                    if part
                )
                LOGGER.error(
                    "background response replay write failed: %s",
                    self.last_write_error,
                )
            finally:
                if succeeded:
                    self._drop_pending(item)
                elif self.pending.get(item.key) is item:
                    # Retain in-memory replay during a short storage outage,
                    # then retry at a bounded cadence until the snapshot TTL.
                    self.failed_pending[item.key] = time.monotonic() + 5.0
                self.write_queue.task_done()

    async def flush(self, timeout: float | None = None) -> bool:
        waiter = self.write_queue.join()
        try:
            if timeout is None:
                await waiter
            else:
                await asyncio.wait_for(waiter, timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True

    async def close(self, timeout: float = 4.0) -> bool:
        self.closing = True
        drained = await self.flush(timeout=timeout)
        task = self.writer_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return drained and not self.pending

    async def get(self, scope: str, model: str, response_id: str) -> ReplaySnapshot | None:
        key = (scope, model, response_id)
        pending = self.pending.get(key)
        if pending is not None:
            if pending.expires_at <= int(time.time()):
                if key in self.failed_pending:
                    self._drop_pending(pending)
            else:
                self.pending.move_to_end(key)
                return await asyncio.to_thread(
                    self._snapshot_from_parts,
                    pending.context,
                    pending.response,
                )

        try:
            if self.path is None:
                async with self.lock:
                    compressed = self._memory_get(scope, model, response_id)
            else:
                compressed = await asyncio.to_thread(self._sqlite_get, scope, model, response_id)
        except sqlite3.Error as exc:
            self.last_error = type(exc).__name__
            return None
        if compressed is None:
            return None
        try:
            raw = await asyncio.to_thread(self._bounded_decompress, compressed)
            snapshot = await asyncio.to_thread(self._decode_snapshot, raw)
        except (ReplayPersistenceError, zlib.error):
            snapshot = None
        return snapshot

    def _bounded_decompress(self, compressed: bytes) -> bytes:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, self.max_raw_bytes + 1)
        if len(raw) > self.max_raw_bytes or decoder.unconsumed_tail or not decoder.eof:
            raise ReplayPersistenceError("response replay snapshot exceeds decode limit")
        raw += decoder.flush()
        if len(raw) > self.max_raw_bytes:
            raise ReplayPersistenceError("response replay snapshot exceeds decode limit")
        return raw

    def _memory_get(self, scope: str, model: str, response_id: str) -> bytes | None:
        key = (scope, model, response_id)
        record = self.memory.get(key)
        if record is None:
            return None
        expires_at, compressed = record
        if expires_at <= int(time.time()):
            self.memory.pop(key, None)
            self.memory_bytes -= len(compressed)
            return None
        self.memory.move_to_end(key)
        return compressed

    def _memory_put(
        self,
        context: ReplayContext,
        response_id: str,
        expires_at: int,
        compressed: bytes,
    ) -> None:
        key = (context.scope, context.model, response_id)
        old = self.memory.pop(key, None)
        if old is not None:
            self.memory_bytes -= len(old[1])
        self.memory[key] = (expires_at, compressed)
        self.memory_bytes += len(compressed)
        now = int(time.time())
        for expired_key, (expiry, value) in list(self.memory.items()):
            if expiry <= now:
                self.memory.pop(expired_key, None)
                self.memory_bytes -= len(value)
        while len(self.memory) > self.max_entries or self.memory_bytes > self.max_bytes:
            _, (_, value) = self.memory.popitem(last=False)
            self.memory_bytes -= len(value)

    def _sqlite_get(self, scope: str, model: str, response_id: str) -> bytes | None:
        assert self.path is not None
        now = int(time.time())
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA query_only=ON")
            row = connection.execute(
                "SELECT history, expires_at FROM response_history "
                "WHERE scope = ? AND model = ? AND response_id = ?",
                (scope, model, response_id),
            ).fetchone()
            if row is None:
                return None
            if int(row[1]) <= now:
                return None
            return bytes(row[0])

    def _sqlite_put(
        self,
        context: ReplayContext,
        response_id: str,
        expires_at: int,
        compressed: bytes,
    ) -> None:
        assert self.path is not None
        now = int(time.time())
        with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("DELETE FROM response_history WHERE expires_at <= ?", (now,))
            connection.execute(
                """
                INSERT INTO response_history(
                    scope, model, response_id, history,
                    created_at, accessed_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, model, response_id) DO UPDATE SET
                    history = excluded.history,
                    created_at = excluded.created_at,
                    accessed_at = excluded.accessed_at,
                    expires_at = excluded.expires_at
                """,
                (
                    context.scope,
                    context.model,
                    response_id,
                    sqlite3.Binary(compressed),
                    now,
                    now,
                    expires_at,
                ),
            )
            while True:
                count, total = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(history_bytes), 0) FROM response_history_sizes"
                ).fetchone()
                if int(count) <= self.max_entries and int(total) <= self.max_bytes:
                    break
                connection.execute(
                    "DELETE FROM response_history WHERE rowid IN ("
                    "SELECT rowid FROM response_history "
                    "ORDER BY accessed_at, created_at LIMIT 1)"
                )
        self._secure_sqlite_files()

    def _sqlite_delete_if_matches(
        self,
        scope: str,
        model: str,
        response_id: str,
        compressed: bytes,
    ) -> None:
        assert self.path is not None
        with closing(sqlite3.connect(self.path, timeout=5)) as connection, connection:
            connection.execute(
                "DELETE FROM response_history "
                "WHERE scope = ? AND model = ? AND response_id = ? AND history = ?",
                (scope, model, response_id, sqlite3.Binary(compressed)),
            )
