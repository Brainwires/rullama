# ADR-0001 — rullama is the app; brainwires is the platform

- **Status:** Accepted
- **Date:** 2026-06-29
- **Authors:** Brainwires

## Context

The Brainwires projects are splitting along brand + audience lines:

- `rullama.com` is the strong **consumer** domain → the downloadable app is
  **rullama**.
- The **brainwires** GitHub project has the larger **developer** following → the
  open-source platform (inference engine **+** agent harness) is **brainwires**.

This leaves two names to remember instead of five, and renames only the
*smaller-following* asset (the engine, off "rullama").

## Decision

**rullama is the consumer product family (app + CLI). brainwires is the platform
both run on.**

```
rullama (rullama.com)        ──▶  brainwires
  ├─ rullama      the PWA / native apps   ├─ engine  (brainwires-engine — was crates/rullama)
  └─ rullama-cli  the agentic CLI         └─ harness (agents, tools, memory, providers, RAG, MCP)
```

- **This repo becomes the rullama app:** keep `web/` (the PWA) and the
  PWA-serving parts of `rullama-devserver` (Vite proxy, `/api/blob`,
  `/api/models`). It supersedes the old `brainwires-studio` and the Candle
  `brainwires-chat-pwa`.
- **The engine leaves:** `crates/rullama` + `crates/rullama-finetune` move into
  the brainwires repo, renamed `brainwires-engine{,-finetune}`, in an isolated
  wasm32 sub-workspace.
- **The app consumes the platform** two ways: in-browser via the engine's wasm
  bundle (a JS provider over the engine `Model`), and natively/elsewhere via an
  OpenAI-compatible `/v1/chat/completions` endpoint.
- **brainwires-cli** extracts to its own repo, renamed **`rullama-cli`** — the
  second rullama-branded product. **brainclaw** extracts to its own brainwires
  sub-product repo. Both depend on published `brainwires` crates.

## Consequences

- The engine's "rullama" crate/bundle name retires; the brand graduates to the
  product (the app).
- Cloud passthrough (`web/src/lib/cloud/*`, the devserver `/api/cloud/*` proxy)
  folds into the harness provider layer over time.
- Cross-repo dev needs a linked loop (engine wasm bundle + cargo path overrides).

## Reference

Canonical topology doc (boundary table, contracts, migration phases):
`brainwires-framework/docs/ARCHITECTURE-engine-harness.md`.
