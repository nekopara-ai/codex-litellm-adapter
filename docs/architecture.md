# Architecture

The service has three local modules:

| Module | Responsibility |
|---|---|
| `server.py` | HTTP routing, model catalog, request/tool conversion, SSE and WebSocket bridging, health checks. |
| `response_replay.py` | Credential/model-scoped snapshots, bounded persistence, continuation reconstruction. |
| `context_compat.py` | Explicitly configured output-budget recovery for compatible provider errors. |

## Request flow

1. Forward the caller's authentication to the fixed configured gateway. The adapter never selects an upstream from request content.
2. For `store=false` Responses requests, recover earlier input when a scoped snapshot is available. An orphaned output without recoverable history fails before inference.
3. Apply configured namespace, optional custom item ID, session metadata, and reasoning transformations.
4. Forward HTTP upstream. WebSocket `response.create` messages use the same HTTP Responses stream bridge; internal tracking fields are omitted on that path.
5. Restore client-facing tool identity. Both transports share complete-event SSE JSON decoding, including multiple data lines.
6. Capture completed output for replay. A final output item takes precedence over an earlier partial item.

Responses streams require a terminal response event. Ordinary Chat Completions SSE retains its `[DONE]` contract. Comments, SSE event IDs, and retry fields survive HTTP namespace rewriting. Request zstd decompression is provided by aiohttp and the Python zstd implementation; upstream compression is explicitly negotiated as identity.

## Failures and recovery

| Condition | Behavior |
|---|---|
| Missing or invalid caller key | Upstream authentication failure is preserved. |
| Missing scoped replay history | HTTP 409 / structured WebSocket error. |
| Pending replay budget exhausted | Non-streaming 503 or a streaming replay error at completion. |
| Responses stream ends without terminal event | Explicit incomplete-stream error. |
| Upstream truncates a buffered JSON response | Structured 502; upstream socket timeout returns 504. |
| Configured context budget error | At most one reduced-output retry. |

SQLite records are scoped by an HMAC derived from selected authentication/organization/project headers and the model. Reusing a response ID with another key or model does not recover the original snapshot. The key file must persist alongside the database across restarts.

Persistence is asynchronous and bounded. Failed writes retain an in-memory fallback and receive periodic retries until expiry. Shutdown allows a short drain period and reports an incomplete drain in logs. This is not a durable distributed queue or an exactly-once delivery guarantee.

## Compatibility boundaries

The adapter cannot make a text-only backend process images, increase model context, or teach a provider unsupported built-in tools. Dynamic aliases and capabilities remain authoritative in LiteLLM. Templates can influence presentation and fill missing metadata; they do not replace routing configuration.

The test suite covers synthetic wire behavior and fault conditions, not every model/provider combination. Provider-specific live validation should use disposable credentials and synthetic content, and record the model/provider versions separately from adapter results.
