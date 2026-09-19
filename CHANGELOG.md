# Changelog

## 0.2.0

- Kept reading the upstream Responses stream after the client-visible terminal event so LiteLLM end-of-stream guardrails (`llm_as_a_judge`) can finish; background drains are bounded by count and time and are cancelled on shutdown.
- Added `CODEX_ADAPTER_POST_TERMINAL_DRAIN_SECONDS` and `CODEX_ADAPTER_MAX_ACTIVE_DRAINS`.
- Ignored non-string item `type` values while restoring namespaced calls, which previously raised `TypeError: unhashable type`.
- Let a model template's `input_modalities` and `supports_image_detail_original` override a missing or stale `supports_vision` flag from the upstream catalog.
- Enabled closed-connection cleanup on the inference connector.
- Dropped the built-in `kimi` display-name alias; unknown slugs still get generic title casing.

## 0.1.0

Initial public release of the reusable adapter core.

- Added packaging, a non-root Docker example, configurable compatibility policies, and synthetic protocol tests.
- Preserved tool-call correlation while repairing incompatible optional custom item IDs.
- Restricted Responses terminal-event validation to Responses endpoints.
- Unified multiline SSE JSON decoding across HTTP and WebSocket paths.
- Restored native custom tool names and prevented collisions with historical namespace tools.
- Kept completed response output authoritative when assembling replay history.
- Added bounded retry and expiry for failed replay writes, with capacity-aware health status.
- Made SQLite size-accounting triggers safe when an existing response ID is overwritten.
- Separated health probes from the inference pool and handled truncated JSON responses.
- Preserved encoded URL paths and normalized same-upstream redirects.
- Removed deployment-specific configuration, model aliases, operational data, and credentials from the public distribution.
