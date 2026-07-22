# Security and privacy

- The server binds to loopback by default and has no cloud dependency.
- Upload names do not control destination paths.
- Project and artifact paths are resolved beneath an allowlisted project root.
- Source copies are content-addressed, re-hashed after copy, and made read-only.
- Agent methods are allowlisted; there is no shell or arbitrary import method.
- AI/provider output is evidence or proposal, never direct project authority.
- Provider aliases validate privacy before health or invocation. `local-only`
  rejects an external data boundary; an external boundary requires both the
  explicit privacy mode and an explicit invocation opt-in.
- Provider commands use argument arrays without a shell; HTTP bindings are
  credential-free loopback URLs. Receipts redact credential-like fields and
  runtime media paths.
- Natural-language edits are previews and require a separate confirmation.
- Style Memory rejects implicit opt-in and stores only explicit confirmations.
- Analyzer failures are localized records; unexpected exception text is bounded
  before persistence.
- The reference video hash is forbidden in ingest/render ancestry.
- Readiness verifies sealed inputs using only existence, readability, size, and
  SHA-256. Content decoding is fail-closed behind a passing receipt.
- Network providers are absent by default. Any future external provider must
  retain the same explicit opt-in, disclosure, health, and provenance contract.

The current loopback prototype does not implement multi-user authentication.
Do not bind it to a non-loopback address on an untrusted network.
