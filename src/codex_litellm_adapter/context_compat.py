"""Opt-in output-budget recovery for providers reporting context exhaustion."""

import json
import logging
import os
import re

LOGGER = logging.getLogger("codex_adapter")
MODEL_PATTERN = re.compile(os.environ.get("CODEX_ADAPTER_CONTEXT_RETRY_MODELS", r"(?!)"))
CONTEXT_WINDOW = int(os.environ.get("CODEX_ADAPTER_CONTEXT_WINDOW", "131072"))
MAX_OUTPUT_TOKENS = int(os.environ.get("CODEX_ADAPTER_CONTEXT_MAX_OUTPUT_TOKENS", "8192"))
MARGIN_TOKENS = int(os.environ.get("CODEX_ADAPTER_CONTEXT_MARGIN_TOKENS", "2048"))
MIN_OUTPUT_TOKENS = int(os.environ.get("CODEX_ADAPTER_CONTEXT_MIN_OUTPUT_TOKENS", "1024"))
EFFECTIVE_PERCENT = int(os.environ.get("CODEX_ADAPTER_CONTEXT_EFFECTIVE_PERCENT", "95"))
if (
    CONTEXT_WINDOW <= 0
    or MAX_OUTPUT_TOKENS <= 0
    or MARGIN_TOKENS < 0
    or MIN_OUTPUT_TOKENS <= 0
    or MIN_OUTPUT_TOKENS > MAX_OUTPUT_TOKENS
    or not 1 <= EFFECTIVE_PERCENT <= 100
):
    raise RuntimeError("Context adaptation limits must be positive and consistent")
ERROR_PATTERN = re.compile(
    r"requested\s+(?P<output>\d+)\s+output tokens.*?"
    r"prompt contains\s+(?P<input>\d+)\s+input tokens",
    re.I | re.S,
)


def matches_model(model):
    return isinstance(model, str) and MODEL_PATTERN.search(model) is not None


def effective_context_window_percent(slug, item):
    value = item.get("effective_context_window_percent", 95)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
        value = 95
    return min(value, EFFECTIVE_PERCENT) if matches_model(slug) else value


def context_window_retry_body(body, status, error_body):
    if status != 400:
        return None
    try:
        payload = json.loads(body)
        upstream = json.loads(error_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or not matches_model(payload.get("model"))
        or not isinstance(upstream, dict)
    ):
        return None
    error = upstream.get("error")
    message = error.get("message") if isinstance(error, dict) else error
    if not isinstance(message, str):
        return None
    match = ERROR_PATTERN.search(message)
    if match is None:
        return None
    requested, tokens = int(match.group("output")), int(match.group("input"))
    budget = min(MAX_OUTPUT_TOKENS, CONTEXT_WINDOW - tokens - MARGIN_TOKENS, requested - 1)
    if budget < MIN_OUTPUT_TOKENS:
        return None
    payload["max_output_tokens"] = budget
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(), budget


async def upstream_request_with_context_retry(session, method, url, headers, body):
    response = await session.request(method, url, headers=headers, data=body, allow_redirects=False)
    if response.status != 400:
        return response, None
    try:
        error_body = await response.read()
    except BaseException:
        response.close()
        raise
    retry = context_window_retry_body(body, response.status, error_body)
    if retry is None:
        return response, error_body
    retry_body, budget = retry
    LOGGER.info("Retrying opted-in model with reduced output budget=%d", budget)
    response.release() if response.content.at_eof() else response.close()
    response = await session.request(
        method, url, headers=headers, data=retry_body, allow_redirects=False
    )
    return response, None
