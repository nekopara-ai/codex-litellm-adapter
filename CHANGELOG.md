# Changelog

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
