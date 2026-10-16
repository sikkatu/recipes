# Security

> **English** | [Русский](SECURITY.ru.md) | [中文](SECURITY.zh.md)


Recipes contain no credentials by contract (CI rejects credential-shaped strings). If you find one anyway, or a recipe launches something unsafe by default (wide binds, disabled auth), open an issue. We fix and re-verify.

## Auth policy

- The API requires a key by default (`API_KEY` is exported into the server as `VLLM_API_KEY`).
- `ALLOW_NO_AUTH=1` is accepted **only together with `BIND=127.0.0.1`** (loopback-only, closed
  debug loop you accept responsibility for). Any non-loopback bind requires an API key
  regardless of `ALLOW_NO_AUTH` — a public unauthenticated API is always refused.
- The server bind is controlled solely by the `--host "$BIND"` argument (host networking makes
  docker `-p` a no-op; the launchers do not pretend otherwise).
- Secrets are never echoed to launcher output; they appear only inside the container env.

