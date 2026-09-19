# Configuration

All settings are environment variables, loaded at process startup. Restart the adapter after changing them. Model templates are loaded when a Codex catalog is requested.

## Server and transport

| Variable | Default | Purpose |
|---|---|---|
| `CODEX_ADAPTER_UPSTREAM` | `http://127.0.0.1:4000` | Gateway origin; normally omit `/v1` because request paths already include it. |
| `CODEX_ADAPTER_HOST` | `127.0.0.1` | Listen address. The Docker image uses `0.0.0.0` inside its container. |
| `CODEX_ADAPTER_PORT` | `4001` | Listen port. |
| `CODEX_ADAPTER_UPSTREAM_HEALTH_PATH` | `/health/liveliness` | Unauthenticated upstream health endpoint. |
| `CODEX_ADAPTER_UPSTREAM_POOL_LIMIT` | `200` | Maximum upstream inference connections. |
| `CODEX_ADAPTER_UPSTREAM_CONNECT_TIMEOUT_SECONDS` | `30` | Socket connection timeout. Pool queue waiting is separate. |
| `CODEX_ADAPTER_UPSTREAM_SOCK_READ_TIMEOUT_SECONDS` | `660` | Upstream socket read timeout. |
| `CODEX_ADAPTER_UPSTREAM_KEEPALIVE_TIMEOUT_SECONDS` | `30` | Idle connection retention. |
| `CODEX_ADAPTER_POST_TERMINAL_DRAIN_SECONDS` | `120` | After forwarding a terminal Responses event, keep reading the upstream stream this long so end-of-stream guardrails can finish. `0` disables the drain. |
| `CODEX_ADAPTER_MAX_ACTIVE_DRAINS` | half of the pool limit, minimum `1` | Maximum concurrent background drains. |
| `CODEX_ADAPTER_MAX_BODY_BYTES` | `134217728` | Request body limit, 128 MiB. |
| `CODEX_ADAPTER_MAX_SSE_EVENT_BYTES` | `8388608` | Maximum buffered SSE event, 8 MiB. |
| `CODEX_ADAPTER_LOG_LEVEL` | `INFO` | Python logging level. |

Health probes use their own two-connection pool and a five-second timeout. A healthy inference pool can remain busy without preventing an upstream health probe. There is no total deadline for long-running inference; use your gateway's admission control and timeouts for overall limits.

LiteLLM runs some guardrails only after it emits the terminal Responses event. The adapter forwards that event immediately, then keeps the upstream stream open in the background for up to `CODEX_ADAPTER_POST_TERMINAL_DRAIN_SECONDS` so those guardrails can finish and be recorded. Guardrail-only frames are never forwarded to the client. When the drain limit is reached, the adapter logs a warning and closes the stream instead of queueing unbounded sockets. Background drains are cancelled during shutdown.

## Model catalog

`GET /v1/models` passes through the upstream OpenAI-style model list. Adding `?client_version=...` requests the Codex catalog format. The adapter first authenticates against `/v1/models`, then attempts `/v1/model/info` and `/model_group/info` with the same caller's credentials. If optional metadata endpoints are unavailable, it uses templates and conservative fallbacks.

Set `CODEX_ADAPTER_TEMPLATE` to a JSON file such as [the example](../examples/model-templates.json). The bundled template is empty. Only models already exposed by the authenticated upstream catalog are listed. Templates do not add routes, bypass permissions, or change backend capability.

Explicit upstream context metadata takes priority over templates. Without metadata or a template, the context fallback is 128000 tokens. Set accurate metadata before relying on automatic context budgeting. Image support is advertised only when upstream metadata explicitly declares it; unknown capabilities are not proof of support.

## Tool namespaces and reasoning

| Variable | Default | Purpose |
|---|---|---|
| `CODEX_ADAPTER_NS_PASSTHROUGH` | `^gpt-` | Regex matching models whose namespaces pass through. |
| `CODEX_ADAPTER_NS_FORCE_FLATTEN` | `(?!)` | Regex opting models into custom-tool bridging, overriding passthrough. Default matches nothing. |
| `CODEX_ADAPTER_CUSTOM_TOOL_GUIDANCE` | empty | Optional guidance appended to bridged custom tool descriptions. |
| `CODEX_ADAPTER_REASONING_EFFORT_MAP` | `{}` | Explicit JSON aliases, such as `{"ultra":"high"}`. Applies to requests handled by this instance. |

Regexes match the public model name sent by the client. Avoid choosing a policy from a display name alone: a gateway alias may route to a different provider. Use separate adapter instances when different model groups require incompatible reasoning aliases.

Namespace flattening chooses stable names up to 64 characters and reserves both current top-level names and historical tool mappings. When custom history is converted to function history, its optional item `id` is removed and `call_id` is preserved. The same tool identity is restored in JSON, SSE, and WebSocket responses.

## Replay storage

| Variable | Default | Purpose |
|---|---|---|
| `CODEX_ADAPTER_REPLAY_DB` | empty | SQLite path; empty selects memory mode. |
| `CODEX_ADAPTER_REPLAY_TTL_SECONDS` | `86400` | Snapshot retention, 24 hours. |
| `CODEX_ADAPTER_REPLAY_MAX_ENTRIES` | `4096` | Stored snapshot count limit. |
| `CODEX_ADAPTER_REPLAY_MAX_BYTES` | `1073741824` | Compressed snapshot payload budget, 1 GiB. |
| `CODEX_ADAPTER_REPLAY_MAX_ENTRY_BYTES` | `134217728` | Compressed per-snapshot limit, 128 MiB. |
| `CODEX_ADAPTER_REPLAY_MAX_RAW_BYTES` | `268435456` | Uncompressed snapshot limit, 256 MiB. |
| `CODEX_ADAPTER_REPLAY_PENDING_MAX_ENTRIES` | `256` | Pending persistence entry limit. |
| `CODEX_ADAPTER_REPLAY_PENDING_MAX_BYTES` | `268435456` | Estimated pending object budget, 256 MiB. |
| `CODEX_ADAPTER_REPLAY_UNHEALTHY_AFTER_WRITE_ERRORS` | `3` | Consecutive failed writes before unhealthy status. |

These are payload limits, not exact process RSS or database file size limits. SQLite pages, indexes, WAL, and Python objects add overhead. Memory storage evicts by recent access; SQLite avoids writes on reads and evicts by stored write timestamps. Use one adapter process per SQLite state directory. For multiple replicas, use sticky routing to the same process or send full histories; this release does not provide a distributed replay service.

Completed responses are queued for persistence. During a short write outage, pending snapshots can serve continuations within their TTL. A failed write gets three short attempts, then is retried at approximately five-second intervals while retained. Expired failures are purged. A full pending budget is reported as unhealthy and new replay snapshots are rejected rather than growing without bounds. A process crash before persistence completes can still lose a recently acknowledged snapshot.

The state directory is sensitive: use a dedicated location and read [SECURITY.md](../SECURITY.md). A missing replay returns HTTP 409 with `response_replay_unavailable`; clients can recover by sending a self-contained transcript without `previous_response_id`.

## Optional context-budget recovery

This feature is disabled until `CODEX_ADAPTER_CONTEXT_RETRY_MODELS` matches a model. It recognizes a narrow HTTP 400 error containing both `requested N output tokens` and `prompt contains M input tokens`. It reduces the requested output budget and retries once; it does not trim the prompt or retry arbitrary 400/503 responses.

| Variable | Default |
|---|---|
| `CODEX_ADAPTER_CONTEXT_RETRY_MODELS` | `(?!)` |
| `CODEX_ADAPTER_CONTEXT_WINDOW` | `131072` |
| `CODEX_ADAPTER_CONTEXT_MAX_OUTPUT_TOKENS` | `8192` |
| `CODEX_ADAPTER_CONTEXT_MARGIN_TOKENS` | `2048` |
| `CODEX_ADAPTER_CONTEXT_MIN_OUTPUT_TOKENS` | `1024` |
| `CODEX_ADAPTER_CONTEXT_EFFECTIVE_PERCENT` | `95` |

Set these to the actual model's limits. A retry is skipped when the remaining output budget is below the minimum.
