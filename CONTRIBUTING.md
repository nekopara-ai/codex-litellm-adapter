# Contributing

Use Python 3.11 or newer. Install `.[dev]`, make a focused change, and run the commands in the README before opening a pull request.

Protocol changes should include a synthetic regression that demonstrates the observed client behavior. Exercise both HTTP SSE and WebSocket when changing event conversion. Exercise scope separation and restart recovery when changing replay persistence.

Keep provider-specific behavior opt-in through configuration. Avoid adding real deployment names, private model aliases, credentials, conversation transcripts, databases, or operational scripts. The public-tree check runs in CI; maintainers also review diffs manually before publishing.

Describe what triggered the bug, the resulting behavior, and how the change was tested. Do not claim a provider is supported based only on a mocked fixture. For sensitive reports, use the guidance in SECURITY.md.
