# Security and data handling

The default listener binds to loopback. The adapter forwards caller authentication to the configured upstream, which must enforce access control and model permissions. `/health` is unauthenticated and reports operational counters. Use a trusted gateway and a TLS reverse proxy when exposing the service remotely.

Replay records contain prompts, outputs, tools, and instructions. SQLite content is compressed, not encrypted. The adjacent `.key` file is an HMAC scoping secret, not an encryption key. Protect and back up the state directory as sensitive data. SQLite mode sets directory permissions to 0700 and key/database permissions to 0600; use a dedicated directory owned by the service account. Memory mode loses history on restart. Expiry limits reuse but is not a guarantee of forensic erasure from backups or filesystem snapshots.

The service does not log request bodies or raw authentication headers through its own access logger. Error text from upstream providers can be forwarded to clients. Reverse proxies and upstream gateways have their own logging policies.

Never attach real keys, full client configuration, raw conversation logs, or replay databases to an issue. Use synthetic reproductions. Report vulnerabilities privately through [GitHub security advisories](https://github.com/nekopara-ai/codex-litellm-adapter/security/advisories/new). Include the affected version and minimal steps to reproduce without secrets.

This initial release has automated protocol and isolation tests. It has not received an independent security audit.
