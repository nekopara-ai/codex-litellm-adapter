#!/usr/bin/env python3
"""Translate LiteLLM's OpenAI model list into Codex's model catalog schema."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from aiohttp import (
    ClientConnectionResetError,
    ClientError,
    ClientSession,
    ClientTimeout,
    DummyCookieJar,
    TCPConnector,
    WSMsgType,
    WSServerHandshakeError,
    web,
)
from multidict import CIMultiDict
from yarl import URL

from . import __version__
from .context_compat import (
    effective_context_window_percent,
    upstream_request_with_context_retry,
)
from .response_replay import (
    CompletedResponseCollector,
    ReplayContext,
    ReplayPersistenceError,
    ReplayUnavailable,
    ResponseReplayStore,
)

UPSTREAM = os.environ.get("CODEX_ADAPTER_UPSTREAM", "http://127.0.0.1:4000").rstrip("/")
LISTEN_HOST = os.environ.get("CODEX_ADAPTER_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("CODEX_ADAPTER_PORT", "4001"))
UPSTREAM_HEALTH_PATH = os.environ.get("CODEX_ADAPTER_UPSTREAM_HEALTH_PATH", "/health/liveliness")
UPSTREAM_POOL_LIMIT = int(os.environ.get("CODEX_ADAPTER_UPSTREAM_POOL_LIMIT", "200"))
UPSTREAM_CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_ADAPTER_UPSTREAM_CONNECT_TIMEOUT_SECONDS", "30")
)
UPSTREAM_SOCK_READ_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_ADAPTER_UPSTREAM_SOCK_READ_TIMEOUT_SECONDS", "660")
)
UPSTREAM_KEEPALIVE_TIMEOUT_SECONDS = float(
    os.environ.get("CODEX_ADAPTER_UPSTREAM_KEEPALIVE_TIMEOUT_SECONDS", "30")
)
MAX_BODY_SIZE = int(os.environ.get("CODEX_ADAPTER_MAX_BODY_BYTES", str(128 * 1024**2)))
MAX_SSE_EVENT_SIZE = int(os.environ.get("CODEX_ADAPTER_MAX_SSE_EVENT_BYTES", str(8 * 1024**2)))
if (
    MAX_BODY_SIZE <= 0
    or MAX_SSE_EVENT_SIZE <= 0
    or UPSTREAM_POOL_LIMIT <= 0
    or UPSTREAM_CONNECT_TIMEOUT_SECONDS <= 0
    or UPSTREAM_SOCK_READ_TIMEOUT_SECONDS <= 0
    or UPSTREAM_KEEPALIVE_TIMEOUT_SECONDS <= 0
):
    raise RuntimeError("adapter limits and timeouts must be positive")
TEMPLATE_PATH = Path(
    os.environ.get(
        "CODEX_ADAPTER_TEMPLATE",
        str(Path(__file__).with_name("model-templates.json")),
    )
)
LOGGER = logging.getLogger("codex_adapter")

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
WS_HANDSHAKE_HEADERS = {
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
}
DECODED_REQUEST_BODY_HEADERS = {
    "content-digest",
    "content-encoding",
    "content-md5",
}
RESPONSES_PATHS = {"/responses", "/v1/responses"}
TERMINAL_RESPONSE_EVENTS = {
    "error",
    "response.completed",
    "response.failed",
    "response.incomplete",
}
REASONING_DESCRIPTIONS = {
    "low": "Fast responses with lighter reasoning",
    "medium": "Balances speed and reasoning depth for everyday tasks",
    "high": "Greater reasoning depth for complex problems",
    "xhigh": "Extra high reasoning depth for complex problems",
    "max": "Maximum available reasoning depth for complex problems",
    "ultra": "Maximum reasoning with automatic task delegation",
}


def load_templates() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return {}
    return {
        item["slug"]: item
        for item in payload.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("slug"), str)
    }


def title_model(slug: str) -> str:
    words = slug.replace("/", " / ").replace("_", " ").replace("-", " ").split()
    aliases = {
        "gpt": "GPT",
        "glm": "GLM",
        "kimi": "Kimi",
        "vllm": "vLLM",
        "deepseek": "DeepSeek",
        "grok": "Grok",
        "claude": "Claude",
        "codex": "Codex",
    }
    return " ".join(
        aliases.get(word.lower(), word.upper() if word.isdigit() else word.title())
        for word in words
    )


def choose_template(slug: str, templates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return templates.get(slug)


def generic_instructions() -> str:
    return (
        "You are Codex, a coding agent. Work carefully in the user's workspace, "
        "follow the user's instructions, inspect relevant files before editing, and verify changes."
    )


def reasoning_levels(info: dict[str, Any], template: dict[str, Any] | None) -> list[dict[str, str]]:
    if template and template.get("supported_reasoning_levels"):
        return template["supported_reasoning_levels"]
    if info.get("supports_reasoning") is False:
        return []
    efforts = ["low", "medium", "high"] if info.get("supports_reasoning") is True else []
    if info.get("supports_xhigh_reasoning_effort"):
        efforts.append("xhigh")
    if info.get("supports_max_reasoning_effort"):
        efforts.append("max")
    return [{"effort": effort, "description": REASONING_DESCRIPTIONS[effort]} for effort in efforts]


def positive_token_limit(value: Any) -> int | None:
    """Normalize JSON integer/whole-float token limits without accepting booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    return None


def context_window(
    model: dict[str, Any], info: dict[str, Any], template: dict[str, Any] | None
) -> int:
    # Prefer an explicitly declared total context window. LiteLLM's
    # max_input_tokens and max_output_tokens are separate limits and must not be
    # conflated with the model's total context capacity.
    candidates = (
        model.get("context_window"),
        model.get("max_context_window"),
        model.get("max_context_length"),
        info.get("context_window"),
        info.get("max_context_window"),
        info.get("max_context_length"),
        model.get("max_input_tokens"),
        info.get("max_input_tokens"),
        template.get("context_window") if template else None,
        template.get("max_context_window") if template else None,
    )
    for value in candidates:
        normalized = positive_token_limit(value)
        if normalized is not None:
            return normalized
    return 128000


def merge_model_info(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = row.get("model_name")
        if not isinstance(slug, str):
            continue
        info = row.get("model_info") or {}
        if not isinstance(info, dict) or info.get("blocked") is True:
            continue
        current = merged.setdefault(slug, {})
        for key, value in info.items():
            if value is not None and key not in current:
                current[key] = value
            elif isinstance(value, bool) and value:
                current[key] = True
    return merged


def merge_model_groups(
    current: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        slug = row.get("model_group") or row.get("model_name")
        if not isinstance(slug, str):
            continue
        target = current.setdefault(slug, {})
        for key, value in row.items():
            if key in {"model_group", "model_name"} or value is None:
                continue
            target[key] = value
    return current


def map_codex_session_fields(payload: dict[str, Any]) -> bool:
    """Map a stable client task ID into LiteLLM's root-level tracking fields.

    Provider metadata is left intact; internal tracking fields must not leak
    into a provider's WebSocket request schema.
    """
    client_metadata = payload.get("client_metadata")
    if not isinstance(client_metadata, dict):
        return False
    session_id = client_metadata.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = client_metadata.get("thread_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return False

    payload.setdefault("litellm_session_id", session_id)
    payload.setdefault("litellm_trace_id", session_id)
    return True


# Namespace support differs across backends. These patterns select the
# explicit passthrough and custom-tool compatibility policies.
try:
    NS_PASSTHROUGH_PATTERN = re.compile(os.environ.get("CODEX_ADAPTER_NS_PASSTHROUGH", r"^gpt-"))
except re.error as exc:
    raise RuntimeError("CODEX_ADAPTER_NS_PASSTHROUGH is not a valid regex") from exc
try:
    NS_FORCE_FLATTEN_PATTERN = re.compile(
        os.environ.get(
            "CODEX_ADAPTER_NS_FORCE_FLATTEN",
            r"(?!)",
        )
    )
except re.error as exc:
    raise RuntimeError("CODEX_ADAPTER_NS_FORCE_FLATTEN is not a valid regex") from exc
FLAT_NAME_MAX = 64
ToolIdentity = tuple[str, str, str]
CUSTOM_TOOL_BRIDGE_GUIDANCE = os.environ.get("CODEX_ADAPTER_CUSTOM_TOOL_GUIDANCE", "")
REASONING_EFFORT_MAP = json.loads(os.environ.get("CODEX_ADAPTER_REASONING_EFFORT_MAP", "{}"))
if not isinstance(REASONING_EFFORT_MAP, dict) or not all(
    isinstance(k, str) and isinstance(v, str) for k, v in REASONING_EFFORT_MAP.items()
):
    raise RuntimeError("CODEX_ADAPTER_REASONING_EFFORT_MAP must map strings to strings")


def force_namespace_flatten(model: Any) -> bool:
    return isinstance(model, str) and bool(NS_FORCE_FLATTEN_PATTERN.search(model))


def passthrough_namespace_tools(model: Any) -> bool:
    return (
        isinstance(model, str)
        and bool(NS_PASSTHROUGH_PATTERN.search(model))
        and not force_namespace_flatten(model)
    )


def promote_additional_tools(payload: dict[str, Any]) -> bool:
    """Move Codex's synthetic additional_tools input into top-level tools."""
    if not force_namespace_flatten(payload.get("model")):
        return False
    items = payload.get("input")
    if not isinstance(items, list):
        return False
    top_level_tools = payload.get("tools")
    if top_level_tools is not None and not isinstance(top_level_tools, list):
        return False

    promoted: list[Any] = []
    remaining: list[Any] = []
    for item in items:
        if (
            isinstance(item, dict)
            and item.get("type") == "additional_tools"
            and isinstance(item.get("tools"), list)
        ):
            promoted.extend(item["tools"])
        else:
            remaining.append(item)
    if not promoted:
        return False
    payload["input"] = remaining
    payload["tools"] = list(top_level_tools or []) + promoted
    return True


def normalize_custom_tool_outputs(payload: dict[str, Any]) -> bool:
    """Convert Codex custom outputs to bridge-compatible function outputs."""
    if not force_namespace_flatten(payload.get("model")):
        return False
    items = payload.get("input")
    if not isinstance(items, list):
        return False
    changed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "custom_tool_call_output":
            item["type"] = "function_call_output"
            # This optional item ID belongs to the custom-output schema.
            # Preserve call_id: it pairs this result with its tool call.
            item.pop("id", None)
            changed = True
        elif item.get("type") in {"function_call", "function_call_output"}:
            # Older bridge responses/history may already contain the converted
            # type with an incompatible custom item ID. Repair that replay too.
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id.startswith(("ctc_", "ctco_")):
                item.pop("id", None)
                changed = True
    return changed


def flat_tool_name(namespace: str, name: str) -> str:
    flat = f"{namespace}__{name}"
    if len(flat) > FLAT_NAME_MAX:
        digest = hashlib.sha1(flat.encode()).hexdigest()[:10]
        flat = flat[: FLAT_NAME_MAX - 11] + "_" + digest
    return flat


def collision_safe_tool_name(
    namespace: str,
    name: str,
    reserved: set[str],
    mapping: dict[str, ToolIdentity],
) -> str:
    """Return a stable flat name without shadowing an ordinary function tool."""
    pair = (namespace, name)
    candidate = flat_tool_name(namespace, name)
    existing = mapping.get(candidate)
    if candidate not in reserved and (existing is None or existing[:2] == pair):
        return candidate

    digest = hashlib.sha1(f"{namespace}\0{name}".encode()).hexdigest()
    for width in range(10, len(digest) + 1, 2):
        suffix = "_" + digest[:width]
        candidate = flat_tool_name(namespace, name)[: FLAT_NAME_MAX - len(suffix)] + suffix
        existing = mapping.get(candidate)
        if candidate not in reserved and (existing is None or existing[:2] == pair):
            return candidate
    raise ValueError("unable to allocate a collision-free namespace tool name")


def flatten_namespace_tools(payload: dict[str, Any]) -> dict[str, ToolIdentity]:
    """Expand namespace tool wrappers into flat function tools, in place.

    Returns a mapping of flat name -> (namespace, original name). Empty when
    nothing was rewritten (no namespace tools, or model is passed through).
    """
    model = payload.get("model")
    if passthrough_namespace_tools(model):
        return {}

    preserve_custom_tools = force_namespace_flatten(model)
    mapping: dict[str, ToolIdentity] = {}
    pair_to_flat: dict[tuple[str, str], str] = {}
    changed = False

    reserved: set[str] = set()
    tools = payload.get("tools")
    if isinstance(tools, list):
        reserved = {
            tool["name"]
            for tool in tools
            if isinstance(tool, dict)
            and tool.get("type") != "namespace"
            and isinstance(tool.get("name"), str)
        }

        def remember(namespace: str, name: str, kind: str) -> str:
            pair = (namespace, name)
            if pair not in pair_to_flat:
                flat = collision_safe_tool_name(namespace, name, reserved, mapping)
                pair_to_flat[pair] = flat
                mapping[flat] = (namespace, name, kind)
                reserved.add(flat)
            return pair_to_flat[pair]

        new_tools: list[Any] = []
        for tool in tools:
            if (
                isinstance(tool, dict)
                and tool.get("type") == "namespace"
                and isinstance(tool.get("tools"), list)
            ):
                namespace = tool.get("name")
                if not isinstance(namespace, str):
                    namespace = ""
                if not all(
                    isinstance(sub, dict) and isinstance(sub.get("name"), str)
                    for sub in tool["tools"]
                ):
                    new_tools.append(tool)
                    continue
                for sub in tool["tools"]:
                    original_type = sub.get("type")
                    kind = (
                        "custom"
                        if preserve_custom_tools and original_type == "custom"
                        else "function"
                    )
                    flat = remember(namespace, sub["name"], kind)
                    flattened = dict(sub)
                    flattened["type"] = kind
                    flattened["name"] = flat
                    if kind == "custom":
                        description = flattened.get("description")
                        if not isinstance(description, str):
                            description = ""
                        if CUSTOM_TOOL_BRIDGE_GUIDANCE not in description:
                            flattened["description"] = description + CUSTOM_TOOL_BRIDGE_GUIDANCE
                    new_tools.append(flattened)
                changed = True
            else:
                new_tools.append(tool)
        if changed:
            payload["tools"] = new_tools

    # Prior-turn namespaced function calls in the transcript must use the same
    # flat names as the tool specs, or the upstream sees calls to undeclared
    # tools.
    items = payload.get("input")
    if isinstance(items, list):
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("type") == "function_call"
                and isinstance(item.get("namespace"), str)
                and isinstance(item.get("name"), str)
            ):
                namespace = item.pop("namespace")
                pair = (namespace, item["name"])
                flat = pair_to_flat.get(pair)
                if flat is None:
                    flat = collision_safe_tool_name(namespace, item["name"], reserved, mapping)
                pair_to_flat[pair] = flat
                reserved.add(flat)
                mapping.setdefault(flat, (namespace, item["name"], "function"))
                item["name"] = flat
                changed = True

            elif (
                preserve_custom_tools
                and isinstance(item, dict)
                and item.get("type") == "custom_tool_call"
                and isinstance(item.get("namespace"), str)
                and isinstance(item.get("name"), str)
            ):
                namespace = item.pop("namespace")
                pair = (namespace, item["name"])
                flat = pair_to_flat.get(pair)
                if flat is None:
                    flat = collision_safe_tool_name(namespace, item["name"], reserved, mapping)
                pair_to_flat[pair] = flat
                reserved.add(flat)
                mapping.setdefault(flat, (namespace, item["name"], "custom"))
                raw_input = item.pop("input", "")
                item["type"] = "function_call"
                # An optional custom-call item ID is invalid for a function
                # call; its call_id remains the stable correlation identity.
                item.pop("id", None)
                item["name"] = flat
                item["arguments"] = json.dumps(
                    {"input": raw_input}, separators=(",", ":"), ensure_ascii=False
                )
                changed = True

    return mapping if changed else {}


def normalize_reasoning_effort(payload: dict[str, Any]) -> bool:
    """Apply only explicitly configured compatibility aliases for reasoning."""
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, dict):
        return False
    effort = reasoning.get("effort")
    replacement = REASONING_EFFORT_MAP.get(effort) if isinstance(effort, str) else None
    if replacement is not None and replacement != effort:
        reasoning["effort"] = replacement
        return True
    return False


def transform_responses_request(
    body: bytes,
    *,
    map_session_fields: bool = True,
) -> tuple[bytes, dict[str, ToolIdentity]]:
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, {}
    if not isinstance(payload, dict):
        return body, {}
    if map_session_fields:
        changed = map_codex_session_fields(payload)
    else:
        # LiteLLM 1.91.5 treats these as internal parameters on ordinary HTTP
        # Responses requests, but its WebSocket response.create path forwards
        # them to the upstream provider.  Keep Codex's client_metadata intact
        # and prevent both adapter-generated and caller-supplied tracking
        # fields from escaping over WebSocket.
        sentinel = object()
        changed = False
        for name in ("litellm_session_id", "litellm_trace_id"):
            if payload.pop(name, sentinel) is not sentinel:
                changed = True
    changed = promote_additional_tools(payload) or changed
    changed = normalize_custom_tool_outputs(payload) or changed
    changed = normalize_reasoning_effort(payload) or changed
    mapping = flatten_namespace_tools(payload)
    if changed or mapping:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    return body, mapping


def custom_tool_input(arguments: Any) -> str:
    if not isinstance(arguments, str):
        return "" if arguments is None else str(arguments)
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if isinstance(decoded, dict) and isinstance(decoded.get("input"), str):
        return decoded["input"]
    return arguments


def restore_namespaced_calls(node: Any, mapping: dict[str, ToolIdentity]) -> None:
    """Recursively restore namespace, name, and custom-call wire semantics."""
    if isinstance(node, dict):
        if node.get("type") in {"function_call", "custom_tool_call"}:
            flat = node.get("name")
            if isinstance(flat, str) and flat in mapping:
                namespace, original, kind = mapping[flat]
                node["name"] = original
                node["namespace"] = namespace
                if kind == "custom" and node.get("type") == "function_call":
                    node["type"] = "custom_tool_call"
                    node["input"] = custom_tool_input(node.pop("arguments", ""))
        for value in node.values():
            restore_namespaced_calls(value, mapping)
    elif isinstance(node, list):
        for value in node:
            restore_namespaced_calls(value, mapping)


class NamespaceEventRewriter:
    """Restore one decoded Responses event, including custom-tool streaming."""

    def __init__(self, mapping: dict[str, ToolIdentity]) -> None:
        self.mapping = mapping
        self.custom_items: dict[str, str] = {}

    def rewrite(self, obj: dict[str, Any]) -> dict[str, Any] | None:
        event_type = obj.get("type")
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            item = obj.get("item")
            if isinstance(item, dict) and item.get("type") in {
                "function_call",
                "custom_tool_call",
            }:
                flat = item.get("name")
                if (
                    isinstance(flat, str)
                    and flat in self.mapping
                    and self.mapping[flat][2] == "custom"
                ):
                    item_id = item.get("id") or item.get("call_id")
                    if isinstance(item_id, str):
                        self.custom_items[item_id] = flat
        elif event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            item_id = obj.get("item_id")
            flat = self.custom_items.get(item_id) if isinstance(item_id, str) else None
            if flat is not None:
                if event_type.endswith(".delta"):
                    return None
                obj["type"] = "response.custom_tool_call_input.done"
                obj["input"] = custom_tool_input(obj.pop("arguments", ""))

        restore_namespaced_calls(obj, self.mapping)
        return obj


class SSENamespaceRewriter:
    """Rewrites flattened function calls inside an SSE event stream."""

    def __init__(self, mapping: dict[str, ToolIdentity]) -> None:
        self.mapping = mapping
        self.event_rewriter = NamespaceEventRewriter(mapping)
        self.buffer = b""

    def feed(self, chunk: bytes) -> bytes:
        self.buffer += chunk
        output: list[bytes] = []
        while True:
            separator = self._find_separator()
            if separator is None:
                break
            index, width = separator
            event, self.buffer = self.buffer[: index + width], self.buffer[index + width :]
            if len(event) > MAX_SSE_EVENT_SIZE:
                raise ValueError("upstream SSE event exceeds the configured byte limit")
            output.append(self._rewrite_event(event))
        if len(self.buffer) > MAX_SSE_EVENT_SIZE:
            raise ValueError("upstream SSE event exceeds the configured byte limit")
        return b"".join(output)

    def flush(self) -> bytes:
        event, self.buffer = self.buffer, b""
        if len(event) > MAX_SSE_EVENT_SIZE:
            raise ValueError("upstream SSE event exceeds the configured byte limit")
        return self._rewrite_event(event) if event else b""

    def _find_separator(self) -> tuple[int, int] | None:
        candidates = []
        index = self.buffer.find(b"\n\n")
        if index >= 0:
            candidates.append((index, 2))
        index = self.buffer.find(b"\r\n\r\n")
        if index >= 0:
            candidates.append((index, 4))
        return min(candidates) if candidates else None

    def _rewrite_event(self, event: bytes) -> bytes:
        obj = decode_sse_json(event)
        if obj is None:
            return event
        rewritten = self.event_rewriter.rewrite(obj)
        if rewritten is None:
            return b""
        data = b"data: " + json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False).encode()
        lines = []
        written = False
        event_type = rewritten.get("type")
        for line in event.replace(b"\r\n", b"\n").split(b"\n"):
            if line.startswith(b"data:"):
                if not written:
                    lines.append(data)
                    written = True
            elif line.startswith(b"event:") and isinstance(event_type, str):
                lines.append(b"event: " + event_type.encode())
            else:
                lines.append(line)
        return b"\n".join(lines)


def decode_sse_json(event: bytes) -> dict[str, Any] | None:
    """Decode a whole SSE event using the same data semantics on both transports."""
    data_lines = [
        line[5:].removeprefix(b" ")
        for line in event.replace(b"\r\n", b"\n").split(b"\n")
        if line.startswith(b"data:")
    ]
    data = b"\n".join(data_lines)
    if not data or data == b"[DONE]":
        return None
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UpstreamProtocolError("Upstream returned invalid SSE JSON") from exc
    if not isinstance(payload, dict):
        raise UpstreamProtocolError("Upstream returned a non-object SSE payload")
    return payload


def build_codex_model(
    model: dict[str, Any],
    info: dict[str, Any],
    priority: int,
    templates: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    slug = model.get("id")
    if not isinstance(slug, str) or not slug:
        return None
    template = choose_template(slug, templates)
    item = dict(template) if template else {}
    window = context_window(model, info, template)
    levels = reasoning_levels(info, template)
    instructions = item.get("base_instructions") or generic_instructions()
    mode = str(info.get("mode") or "chat")
    hidden = mode not in {"chat", "responses", "completion", ""}
    patch_tool_type = item.get("apply_patch_tool_type")
    if patch_tool_type != "freeform":
        patch_tool_type = None

    item.update(
        {
            "slug": slug,
            "display_name": item.get("display_name")
            if template and template.get("slug") == slug
            else title_model(slug),
            "description": item.get("description")
            if template and template.get("slug") == slug
            else f"{title_model(slug)} via local LiteLLM",
            "default_reasoning_level": (item.get("default_reasoning_level") or "medium")
            if levels
            else None,
            "supported_reasoning_levels": levels,
            "shell_type": item.get("shell_type", "shell_command"),
            "visibility": "hide" if hidden else "list",
            "supported_in_api": True,
            "priority": priority,
            "additional_speed_tiers": item.get("additional_speed_tiers", []),
            "service_tiers": item.get("service_tiers", []),
            "availability_nux": None,
            "upgrade": None,
            "include_skills_usage_instructions": item.get(
                "include_skills_usage_instructions", True
            ),
            "include_plugin_usage_instructions": item.get(
                "include_plugin_usage_instructions", True
            ),
            "include_apps_usage_instructions": item.get("include_apps_usage_instructions", True),
            "default_reasoning_summary": item.get("default_reasoning_summary", "none"),
            "support_verbosity": item.get("support_verbosity", False),
            "default_verbosity": item.get("default_verbosity", "low"),
            "apply_patch_tool_type": patch_tool_type,
            "web_search_tool_type": item.get("web_search_tool_type", "text_and_image"),
            "truncation_policy": item.get("truncation_policy", {"mode": "tokens", "limit": 10000}),
            "supports_parallel_tool_calls": info.get("supports_function_calling") is not False,
            "supports_image_detail_original": info.get("supports_vision") is True,
            "context_window": window,
            "max_context_window": window,
            "effective_context_window_percent": effective_context_window_percent(slug, item),
            "experimental_supported_tools": item.get("experimental_supported_tools", []),
            "input_modalities": ["text", "image"]
            if info.get("supports_vision") is True
            else ["text"],
            "supports_search_tool": info.get("supports_web_search") is True,
            "use_responses_lite": item.get("use_responses_lite", False),
            "base_instructions": instructions,
        }
    )
    if not item.get("model_messages"):
        item["model_messages"] = {
            "instructions_template": instructions,
            "instructions_variables": None,
            "approvals": None,
            "collaboration_modes": None,
            "auto_review": None,
            "permissions": None,
        }
    return item


class UpstreamHTTPError(Exception):
    def __init__(
        self,
        status: int,
        reason: str,
        body: bytes,
        headers: CIMultiDict[str],
    ) -> None:
        super().__init__(f"upstream returned HTTP {status}")
        self.status = status
        self.reason = reason
        self.body = body
        self.headers = headers


class UpstreamProtocolError(Exception):
    pass


def filtered_headers(
    source: Any,
    *,
    strip_host: bool = False,
    strip_content_length: bool = False,
    extra_blocked: set[str] | frozenset[str] = frozenset(),
) -> CIMultiDict[str]:
    blocked = set(HOP_BY_HOP)
    blocked.update(name.lower() for name in extra_blocked)
    getall = getattr(source, "getall", None)
    connection_values = getall("Connection", []) if getall is not None else []
    for value in connection_values:
        blocked.update(token.strip().lower() for token in value.split(",") if token.strip())
    if strip_host:
        blocked.add("host")
    if strip_content_length:
        blocked.add("content-length")

    result: CIMultiDict[str] = CIMultiDict()
    for key, value in source.items():
        if key.lower() not in blocked:
            result.add(key, value)
    return result


def public_redirect_location(location: str) -> str:
    """Keep redirects to our private upstream on the browser's current origin."""
    try:
        target = urlsplit(location)
        upstream = urlsplit(UPSTREAM)
        if (
            target.scheme == upstream.scheme
            and target.netloc == upstream.netloc
            and not target.path.startswith("//")
        ):
            return urlunsplit(("", "", target.path or "/", target.query, target.fragment))
    except ValueError:
        pass
    return location


def is_websocket_upgrade(request: web.Request) -> bool:
    if request.headers.get("Upgrade", "").strip().lower() != "websocket":
        return False
    values = request.headers.getall("Connection", [])
    return any(token.strip().lower() == "upgrade" for value in values for token in value.split(","))


def requested_websocket_protocols(request: web.Request) -> tuple[str, ...]:
    return tuple(
        token.strip()
        for value in request.headers.getall("Sec-WebSocket-Protocol", [])
        for token in value.split(",")
        if token.strip()
    )


def unsupported_content_encoding(headers: Any) -> str | None:
    """Return a non-identity content encoding that cannot be forwarded safely.

    The adapter deliberately asks LiteLLM for identity responses.  Passing an
    unexpected compressed body through would make clients depend on optional
    gzip/brotli/zstd decoders and would also make response rewriting unsafe.
    """
    raw = headers.get("Content-Encoding")
    if not isinstance(raw, str) or not raw.strip():
        return None
    encodings = [value.strip().lower() for value in raw.split(",") if value.strip()]
    return None if encodings and all(value == "identity" for value in encodings) else raw


def gateway_error(
    status: int,
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> web.Response:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return web.json_response(
        {"error": error},
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def replay_unavailable_response() -> web.Response:
    return gateway_error(
        409,
        "Previous store=false response history is unavailable; resend the full "
        "input transcript without previous_response_id",
        "invalid_request_error",
        code="response_replay_unavailable",
    )


async def prepare_response_replay(
    request: web.Request, payload: dict[str, Any]
) -> ReplayContext | None:
    context = await request.app["replay_store"].prepare(payload, request.headers)
    if context is not None and context.continued_from:
        LOGGER.info(
            "%s store=false response history id_hash=%s items=%d",
            "replayed" if context.replayed else "accepted self-contained",
            hashlib.sha256(context.continued_from.encode()).hexdigest()[:12],
            len(context.history),
        )
    return context


def persist_replay_response(
    request: web.Request,
    context: ReplayContext | None,
    response: dict[str, Any],
) -> bool:
    if context is None:
        return True
    try:
        saved = request.app["replay_store"].queue_save(context, response)
    except (OSError, ReplayPersistenceError) as exc:
        LOGGER.error("failed to queue response replay snapshot: %s", type(exc).__name__)
        return False
    if not saved:
        LOGGER.error("completed store=false response did not contain replayable output")
        return False
    if saved:
        response_id = response.get("id")
        if isinstance(response_id, str):
            LOGGER.info(
                "queued store=false response history id_hash=%s",
                hashlib.sha256(response_id.encode()).hexdigest()[:12],
            )
    return True


def replay_persistence_error_response() -> web.Response:
    return gateway_error(
        503,
        "The completed store=false response could not be saved for continuation",
        "server_error",
        code="response_replay_persistence_error",
    )


REPLAY_PERSISTENCE_ERROR_SSE = (
    b'data: {"type":"error","error":{"message":"The completed store=false '
    b'response could not be saved for continuation","type":"server_error",'
    b'"code":"response_replay_persistence_error"}}\n\n'
)
MISSING_TERMINAL_RESPONSE_SSE = (
    b'data: {"type":"error","error":{"message":"LiteLLM stream ended before a '
    b'terminal response event","type":"upstream_protocol_error",'
    b'"code":"upstream_incomplete_stream"}}\n\n'
)


def upstream_error_response(error: UpstreamHTTPError) -> web.Response:
    return web.Response(
        status=error.status,
        reason=error.reason,
        body=error.body,
        headers=filtered_headers(error.headers, strip_content_length=True),
    )


def websocket_handshake_error_response(error: WSServerHandshakeError) -> web.Response:
    headers = filtered_headers(
        error.headers or CIMultiDict(),
        strip_content_length=True,
        extra_blocked=WS_HANDSHAKE_HEADERS | {"content-encoding", "content-type"},
    )
    headers["Cache-Control"] = "no-store"
    return web.json_response(
        {
            "error": {
                "message": f"LiteLLM rejected the WebSocket handshake (HTTP {error.status})",
                "type": "upstream_websocket_handshake_error",
            }
        },
        status=error.status,
        headers=headers,
    )


async def upstream_json(request: web.Request, path: str) -> dict[str, Any]:
    headers = filtered_headers(request.headers, strip_host=True, strip_content_length=True)
    headers["Accept-Encoding"] = "identity"
    async with request.app["session"].get(UPSTREAM + path, headers=headers) as response:
        if unsupported_content_encoding(response.headers) is not None:
            raise UpstreamProtocolError("upstream ignored identity content encoding")
        raw = await response.read()
        if response.status >= 400:
            raise UpstreamHTTPError(
                response.status,
                response.reason,
                raw,
                filtered_headers(response.headers, strip_content_length=True),
            )
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise UpstreamProtocolError("upstream returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise UpstreamProtocolError("upstream returned a non-object JSON payload")
        return payload


async def models(request: web.Request) -> web.Response:
    if "client_version" not in request.query:
        return await proxy(request)
    try:
        models_payload = await upstream_json(request, "/v1/models")
    except UpstreamHTTPError as exc:
        return upstream_error_response(exc)
    except asyncio.TimeoutError:
        return gateway_error(504, "LiteLLM model catalog timed out", "upstream_timeout")
    except (ClientError, UpstreamProtocolError):
        return gateway_error(502, "LiteLLM model catalog is unavailable", "upstream_error")

    try:
        info_payload = await upstream_json(request, "/v1/model/info")
    except (UpstreamHTTPError, UpstreamProtocolError, ClientError, asyncio.TimeoutError):
        info_payload = {"data": []}
    try:
        group_payload = await upstream_json(request, "/model_group/info")
    except (UpstreamHTTPError, UpstreamProtocolError, ClientError, asyncio.TimeoutError):
        group_payload = {"data": []}

    model_rows = models_payload.get("data")
    if not isinstance(model_rows, list):
        return gateway_error(502, "LiteLLM model catalog has an invalid shape", "upstream_error")
    info_rows = info_payload.get("data")
    if not isinstance(info_rows, list):
        info_rows = []
    group_rows = group_payload.get("data")
    if not isinstance(group_rows, list):
        group_rows = []

    templates = load_templates()
    info_by_slug = merge_model_groups(merge_model_info(info_rows), group_rows)
    result = []
    template_priorities = [
        value.get("priority")
        for value in templates.values()
        if isinstance(value.get("priority"), int) and not isinstance(value.get("priority"), bool)
    ]
    next_dynamic_priority = max(template_priorities, default=0) + 1
    for model in model_rows:
        if not isinstance(model, dict):
            continue
        slug = model.get("id")
        exact_template = templates.get(slug, {})
        priority = exact_template.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool):
            priority = next_dynamic_priority
            next_dynamic_priority += 1
        item = build_codex_model(model, info_by_slug.get(model.get("id"), {}), priority, templates)
        if item is not None:
            result.append(item)
    payload = {"models": result}
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    etag = '"' + hashlib.sha256(encoded).hexdigest() + '"'
    response_headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("If-None-Match") == etag:
        return web.Response(status=304, headers=response_headers)
    return web.Response(
        body=encoded,
        content_type="application/json",
        headers=response_headers,
    )


async def health(request: web.Request) -> web.Response:
    replay_store = request.app["replay_store"]
    replay_status = {
        "mode": replay_store.mode,
        "healthy": replay_store.healthy,
        "writer_alive": replay_store.writer_alive,
        "queued_writes": replay_store.queued_writes,
        "pending_entries": replay_store.pending_entries,
        "pending_bytes": replay_store.pending_bytes,
        "completed_writes": replay_store.completed_writes,
        "write_errors": replay_store.write_errors,
        "consecutive_write_errors": replay_store.consecutive_write_errors,
        "unhealthy_after_write_errors": (replay_store.unhealthy_after_consecutive_write_errors),
        "dropped_writes": replay_store.dropped_writes,
        "last_write_error": replay_store.last_write_error,
    }

    def pool_status() -> dict[str, int]:
        connector = request.app["session"].connector
        if not isinstance(connector, TCPConnector):
            return {"limit": 0, "acquired": 0, "idle": 0, "waiters": 0}
        return {
            "limit": connector.limit,
            "acquired": len(connector._acquired),
            "idle": sum(len(items) for items in connector._conns.values()),
            "waiters": sum(len(items) for items in connector._waiters.values()),
        }

    try:
        async with request.app["health_session"].get(
            UPSTREAM + UPSTREAM_HEALTH_PATH,
            headers={"Accept-Encoding": "identity"},
            timeout=ClientTimeout(total=5),
        ) as response:
            await response.read()
            upstream_status = response.status
    except (ClientError, asyncio.TimeoutError):
        return web.json_response(
            {
                "status": "error",
                "upstream_status": None,
                "response_replay": replay_status,
                "upstream_pool": pool_status(),
            },
            status=503,
            headers={"Cache-Control": "no-store"},
        )
    if upstream_status >= 500 or not replay_store.healthy:
        return web.json_response(
            {
                "status": "error",
                "upstream_status": upstream_status,
                "response_replay": replay_status,
                "upstream_pool": pool_status(),
            },
            status=503,
            headers={"Cache-Control": "no-store"},
        )
    return web.json_response(
        {
            "status": "ok",
            "upstream_status": upstream_status,
            "response_replay": replay_status,
            "upstream_pool": pool_status(),
        },
        headers={"Cache-Control": "no-store"},
    )


class SSEJSONDecoder:
    """Incrementally decode JSON data fields from an SSE byte stream."""

    def __init__(self, mapping: dict[str, ToolIdentity]) -> None:
        self.mapping = mapping
        self.event_rewriter = NamespaceEventRewriter(mapping)
        self.buffer = b""

    def feed_frames(self, chunk: bytes) -> list[tuple[bytes, dict[str, Any] | None]]:
        self.buffer += chunk
        output: list[tuple[bytes, dict[str, Any] | None]] = []
        while True:
            separator = self._find_separator()
            if separator is None:
                break
            index, width = separator
            event = self.buffer[:index]
            frame, self.buffer = (
                self.buffer[: index + width],
                self.buffer[index + width :],
            )
            if len(event) > MAX_SSE_EVENT_SIZE:
                raise ValueError("upstream SSE event exceeds the configured byte limit")
            output.append((frame, self._decode_event(event)))
        if len(self.buffer) > MAX_SSE_EVENT_SIZE:
            raise ValueError("upstream SSE event exceeds the configured byte limit")
        return output

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        return [decoded for _, decoded in self.feed_frames(chunk) if decoded is not None]

    def flush_frame(self) -> tuple[bytes, dict[str, Any] | None] | None:
        event, self.buffer = self.buffer, b""
        if len(event) > MAX_SSE_EVENT_SIZE:
            raise ValueError("upstream SSE event exceeds the configured byte limit")
        if not event:
            return None
        return event, self._decode_event(event)

    def flush(self) -> list[dict[str, Any]]:
        frame = self.flush_frame()
        decoded = frame[1] if frame is not None else None
        return [decoded] if decoded is not None else []

    def _find_separator(self) -> tuple[int, int] | None:
        candidates = []
        index = self.buffer.find(b"\n\n")
        if index >= 0:
            candidates.append((index, 2))
        index = self.buffer.find(b"\r\n\r\n")
        if index >= 0:
            candidates.append((index, 4))
        return min(candidates) if candidates else None

    def _decode_event(self, event: bytes) -> dict[str, Any] | None:
        payload = decode_sse_json(event)
        return self.event_rewriter.rewrite(payload) if payload is not None else None


async def send_websocket_error(
    downstream: web.WebSocketResponse,
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> None:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    await downstream.send_json({"type": "error", "error": error})


def finalize_upstream_response(response: Any) -> None:
    """Reuse complete responses and close incomplete streams immediately."""
    if response.content.at_eof():
        response.release()
    else:
        response.close()


async def stream_http_responses_to_websocket(
    request: web.Request,
    downstream: web.WebSocketResponse,
    payload: dict[str, Any],
) -> None:
    upstream_url = URL(UPSTREAM + request.rel_url.raw_path_qs, encoded=True)
    headers = filtered_headers(
        request.headers,
        strip_host=True,
        strip_content_length=True,
        extra_blocked=WS_HANDSHAKE_HEADERS | DECODED_REQUEST_BODY_HEADERS,
    )
    headers["Accept-Encoding"] = "identity"
    headers["Accept"] = "text/event-stream"
    headers["Content-Type"] = "application/json"

    http_payload = copy.deepcopy(payload)
    http_payload.pop("type", None)
    http_payload["stream"] = True
    try:
        replay_context = await prepare_response_replay(request, http_payload)
    except ReplayUnavailable:
        await send_websocket_error(
            downstream,
            "Previous store=false response history is unavailable; resend the full "
            "input transcript without previous_response_id",
            "invalid_request_error",
            code="response_replay_unavailable",
        )
        return
    body, mapping = transform_responses_request(
        json.dumps(http_payload, separators=(",", ":"), ensure_ascii=False).encode()
    )
    try:
        response, buffered_error_body = await upstream_request_with_context_retry(
            request.app["session"],
            "POST",
            upstream_url,
            headers,
            body,
        )
    except asyncio.TimeoutError:
        await send_websocket_error(downstream, "LiteLLM request timed out", "upstream_timeout")
        return
    except ClientError:
        await send_websocket_error(downstream, "LiteLLM is unavailable", "upstream_error")
        return

    try:
        if unsupported_content_encoding(response.headers) is not None:
            await send_websocket_error(
                downstream,
                "LiteLLM returned an unsupported compressed response",
                "upstream_encoding_error",
            )
            return
        if response.status >= 400:
            raw = buffered_error_body if buffered_error_body is not None else await response.read()
            try:
                upstream_payload = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                upstream_payload = None
            error = upstream_payload.get("error") if isinstance(upstream_payload, dict) else None
            if isinstance(error, dict):
                await downstream.send_json({"type": "error", "error": error})
            else:
                await send_websocket_error(
                    downstream,
                    f"LiteLLM rejected the Responses request (HTTP {response.status})",
                    "upstream_http_error",
                    code=str(response.status),
                )
            return

        content_type = response.headers.get("Content-Type", "")
        if "text/event-stream" not in content_type:
            await send_websocket_error(
                downstream,
                "LiteLLM returned a non-SSE Responses stream",
                "upstream_protocol_error",
            )
            return

        decoder = SSEJSONDecoder(mapping)
        collector = CompletedResponseCollector()
        terminal_seen = False
        async for chunk in response.content.iter_any():
            for event in decoder.feed(chunk):
                completed = collector.consume(event)
                if completed is not None:
                    if not persist_replay_response(request, replay_context, completed):
                        await send_websocket_error(
                            downstream,
                            "The completed store=false response could not be saved "
                            "for continuation",
                            "server_error",
                            code="response_replay_persistence_error",
                        )
                        terminal_seen = True
                        break
                await downstream.send_json(event)
                if event.get("type") in TERMINAL_RESPONSE_EVENTS:
                    terminal_seen = True
                    break
            if terminal_seen:
                break
        if not terminal_seen:
            for event in decoder.flush():
                completed = collector.consume(event)
                if completed is not None:
                    if not persist_replay_response(request, replay_context, completed):
                        await send_websocket_error(
                            downstream,
                            "The completed store=false response could not be saved "
                            "for continuation",
                            "server_error",
                            code="response_replay_persistence_error",
                        )
                        terminal_seen = True
                        break
                await downstream.send_json(event)
                if event.get("type") in TERMINAL_RESPONSE_EVENTS:
                    terminal_seen = True
        if not terminal_seen:
            await send_websocket_error(
                downstream,
                "LiteLLM stream ended before a terminal response event",
                "upstream_protocol_error",
            )
    except asyncio.CancelledError:
        raise
    except (ClientError, asyncio.TimeoutError, ValueError, UpstreamProtocolError) as exc:
        LOGGER.warning("WebSocket HTTP bridge ended abnormally: %s", type(exc).__name__)
        if not downstream.closed:
            await send_websocket_error(
                downstream, "LiteLLM response stream failed", "upstream_stream_error"
            )
    finally:
        finalize_upstream_response(response)


async def websocket_proxy(request: web.Request) -> web.StreamResponse:
    headers = filtered_headers(
        request.headers,
        strip_host=True,
        strip_content_length=True,
        extra_blocked=WS_HANDSHAKE_HEADERS,
    )
    headers["Accept-Encoding"] = "identity"
    try:
        async with request.app["session"].get(
            UPSTREAM + "/v1/models", headers=headers, allow_redirects=False
        ) as auth_response:
            auth_body = await auth_response.read()
            if auth_response.status >= 400:
                return web.Response(
                    status=auth_response.status,
                    reason=auth_response.reason,
                    body=auth_body,
                    headers=filtered_headers(auth_response.headers, strip_content_length=True),
                )
    except asyncio.TimeoutError:
        return gateway_error(504, "LiteLLM authentication timed out", "upstream_timeout")
    except ClientError:
        return gateway_error(502, "LiteLLM is unavailable", "upstream_error")

    downstream = web.WebSocketResponse(
        autoping=True,
        autoclose=True,
        compress=False,
        max_msg_size=MAX_BODY_SIZE,
    )
    await downstream.prepare(request)
    active: asyncio.Task[None] | None = None
    try:
        async for message in downstream:
            if message.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(message.data)
                except (json.JSONDecodeError, TypeError):
                    await send_websocket_error(
                        downstream, "WebSocket message is not valid JSON", "invalid_request"
                    )
                    continue
                if not isinstance(payload, dict):
                    await send_websocket_error(
                        downstream, "WebSocket message must be an object", "invalid_request"
                    )
                    continue
                message_type = payload.get("type")
                if message_type == "response.create":
                    if active is not None and not active.done():
                        try:
                            await asyncio.wait_for(asyncio.shield(active), timeout=0.5)
                        except asyncio.TimeoutError:
                            await send_websocket_error(
                                downstream,
                                "A Responses request is already active on this WebSocket",
                                "concurrent_request_not_supported",
                            )
                            continue
                    if active is not None:
                        await asyncio.gather(active, return_exceptions=True)
                    active = asyncio.create_task(
                        stream_http_responses_to_websocket(request, downstream, payload)
                    )
                elif message_type == "response.cancel":
                    if active is not None and not active.done():
                        active.cancel()
                        await asyncio.gather(active, return_exceptions=True)
                    active = None
                else:
                    await send_websocket_error(
                        downstream,
                        "Unsupported WebSocket message type",
                        "invalid_request",
                    )
            elif message.type == WSMsgType.BINARY:
                await send_websocket_error(
                    downstream, "Binary WebSocket messages are unsupported", "invalid_request"
                )
            elif message.type == WSMsgType.ERROR:
                raise downstream.exception() or ConnectionError("downstream WebSocket failed")
    finally:
        if active is not None and not active.done():
            active.cancel()
        if active is not None:
            await asyncio.gather(active, return_exceptions=True)
        await downstream.close()
    return downstream


async def proxy(request: web.Request) -> web.StreamResponse:
    if request.path.rstrip("/") in RESPONSES_PATHS and is_websocket_upgrade(request):
        return await websocket_proxy(request)

    upstream_url = URL(UPSTREAM + request.rel_url.raw_path_qs, encoded=True)
    headers = filtered_headers(
        request.headers,
        strip_content_length=True,
        extra_blocked=DECODED_REQUEST_BODY_HEADERS,
    )
    # LiteLLM also builds /v2/login JSON redirect_url from request.base_url.
    # Keep the client-facing Host while connecting to the configured upstream.
    headers["Accept-Encoding"] = "identity"
    body = await request.read()
    mapping: dict[str, ToolIdentity] = {}
    replay_context: ReplayContext | None = None
    if request.method.upper() == "POST" and request.path.rstrip("/") in RESPONSES_PATHS:
        try:
            replay_payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            replay_payload = None
        if isinstance(replay_payload, dict):
            try:
                replay_context = await prepare_response_replay(request, replay_payload)
            except ReplayUnavailable:
                return replay_unavailable_response()
            body = json.dumps(replay_payload, separators=(",", ":"), ensure_ascii=False).encode()
        body, mapping = transform_responses_request(body)
    buffered_body: bytes | None = None
    try:
        if request.method.upper() == "POST" and request.path.rstrip("/") in RESPONSES_PATHS:
            response, buffered_body = await upstream_request_with_context_retry(
                request.app["session"], request.method, upstream_url, headers, body
            )
        else:
            response = await request.app["session"].request(
                request.method, upstream_url, headers=headers, data=body, allow_redirects=False
            )
    except asyncio.TimeoutError:
        return gateway_error(504, "LiteLLM request timed out", "upstream_timeout")
    except ClientError:
        return gateway_error(502, "LiteLLM is unavailable", "upstream_error")

    if unsupported_content_encoding(response.headers) is not None:
        finalize_upstream_response(response)
        return gateway_error(
            502,
            "LiteLLM returned an unsupported compressed response",
            "upstream_encoding_error",
        )

    content_type = response.headers.get("Content-Type", "")
    if (
        response.status == 200
        and (mapping or replay_context is not None)
        and "application/json" in content_type
    ):
        try:
            raw = await response.read()
        except asyncio.TimeoutError:
            finalize_upstream_response(response)
            return gateway_error(504, "Upstream response body timed out", "upstream_timeout")
        except ClientError:
            finalize_upstream_response(response)
            return gateway_error(502, "Upstream response body was truncated", "upstream_error")
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if replay_context is not None:
                finalize_upstream_response(response)
                return gateway_error(
                    502,
                    "LiteLLM returned invalid Responses JSON",
                    "upstream_protocol_error",
                )
            buffered_body = raw
        else:
            if mapping:
                restore_namespaced_calls(obj, mapping)
            if replay_context is not None and (
                not isinstance(obj, dict)
                or not persist_replay_response(request, replay_context, obj)
            ):
                finalize_upstream_response(response)
                return replay_persistence_error_response()
            buffered_body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()

    downstream_headers = filtered_headers(response.headers, strip_content_length=True)
    if 300 <= response.status < 400 and "Location" in downstream_headers:
        downstream_headers["Location"] = public_redirect_location(downstream_headers["Location"])
    downstream = web.StreamResponse(
        status=response.status, reason=response.reason, headers=downstream_headers
    )
    try:
        await downstream.prepare(request)
        if buffered_body is not None:
            await downstream.write(buffered_body)
            await downstream.write_eof()
            return downstream
        rewriter = None
        capture_decoder = None
        collector = None
        terminal_seen = False
        if (
            response.status == 200
            and "text/event-stream" in content_type
            and request.path.rstrip("/") in RESPONSES_PATHS
        ):
            if mapping:
                rewriter = SSENamespaceRewriter(mapping)
            capture_decoder = SSEJSONDecoder(mapping)
            if replay_context is not None:
                collector = CompletedResponseCollector()

        def consume_captured_event(event: dict[str, Any] | None) -> bool:
            if event is None:
                return False
            completed = collector.consume(event) if collector is not None else None
            if completed is not None and not persist_replay_response(
                request, replay_context, completed
            ):
                raise ReplayPersistenceError("failed to queue replay snapshot")
            return event.get("type") in TERMINAL_RESPONSE_EVENTS

        async for chunk in response.content.iter_any():
            if capture_decoder is None:
                await downstream.write(chunk)
                continue
            for frame, event in capture_decoder.feed_frames(chunk):
                try:
                    terminal_seen = consume_captured_event(event)
                except ReplayPersistenceError:
                    await downstream.write(REPLAY_PERSISTENCE_ERROR_SSE)
                    await downstream.write_eof()
                    return downstream
                output_frame = rewriter.feed(frame) if rewriter is not None else frame
                if output_frame:
                    await downstream.write(output_frame)
                if terminal_seen:
                    break
            if terminal_seen:
                break
        if capture_decoder is not None and not terminal_seen:
            final_frame = capture_decoder.flush_frame()
            if final_frame is not None:
                frame, event = final_frame
                try:
                    terminal_seen = consume_captured_event(event)
                except ReplayPersistenceError:
                    await downstream.write(REPLAY_PERSISTENCE_ERROR_SSE)
                    await downstream.write_eof()
                    return downstream
                output_frame = rewriter.feed(frame) if rewriter is not None else frame
                if output_frame:
                    await downstream.write(output_frame)
                if rewriter is not None:
                    tail = rewriter.flush()
                    if tail:
                        await downstream.write(tail)
        if capture_decoder is not None and not terminal_seen:
            # The WebSocket bridge has always refused to close a response stream
            # before response.completed/failed/incomplete/error; apply the same
            # contract to plain HTTP so a truncated upstream cannot pass as a
            # finished answer.
            LOGGER.warning("LiteLLM stream ended before a terminal response event (http)")
            await downstream.write(MISSING_TERMINAL_RESPONSE_SSE)
        await downstream.write_eof()
    except (ClientConnectionResetError, ConnectionResetError):
        return downstream
    except (ClientError, asyncio.TimeoutError, ValueError, UpstreamProtocolError) as exc:
        LOGGER.warning("upstream stream ended abnormally: %s", type(exc).__name__)
        transport = request.transport
        if transport is not None:
            transport.close()
    finally:
        finalize_upstream_response(response)
    return downstream


async def create_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BODY_SIZE)
    replay_store = ResponseReplayStore.from_environment()
    await replay_store.initialize()
    app["replay_store"] = replay_store
    app["session"] = ClientSession(
        timeout=ClientTimeout(
            total=None,
            sock_connect=UPSTREAM_CONNECT_TIMEOUT_SECONDS,
            sock_read=UPSTREAM_SOCK_READ_TIMEOUT_SECONDS,
        ),
        connector=TCPConnector(
            limit=UPSTREAM_POOL_LIMIT,
            force_close=False,
            keepalive_timeout=UPSTREAM_KEEPALIVE_TIMEOUT_SECONDS,
        ),
        auto_decompress=False,
        cookie_jar=DummyCookieJar(),
    )
    app["health_session"] = ClientSession(
        connector=TCPConnector(limit=2),
        timeout=ClientTimeout(total=5),
        cookie_jar=DummyCookieJar(),
        auto_decompress=False,
    )
    LOGGER.info(
        "configured upstream pool limit=%s connect_timeout=%ss sock_read_timeout=%ss keepalive_timeout=%ss",
        UPSTREAM_POOL_LIMIT,
        UPSTREAM_CONNECT_TIMEOUT_SECONDS,
        UPSTREAM_SOCK_READ_TIMEOUT_SECONDS,
        UPSTREAM_KEEPALIVE_TIMEOUT_SECONDS,
    )
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_route("*", "/{path:.*}", proxy)

    async def close_session(application: web.Application) -> None:
        await application["session"].close()
        await application["health_session"].close()

    async def close_replay_store(application: web.Application) -> None:
        drained = await application["replay_store"].close(timeout=4.0)
        if not drained:
            LOGGER.warning("response replay writer did not drain before shutdown")

    app.on_cleanup.append(close_replay_store)
    app.on_cleanup.append(close_session)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Codex protocol adapter for LiteLLM gateways")
    parser.add_argument("--version", action="version", version=__version__)
    parser.parse_args()
    logging.basicConfig(
        level=os.environ.get("CODEX_ADAPTER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    web.run_app(
        create_app(),
        host=LISTEN_HOST,
        port=LISTEN_PORT,
        access_log=None,
        print=None,
        handle_signals=True,
        shutdown_timeout=5.0,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
