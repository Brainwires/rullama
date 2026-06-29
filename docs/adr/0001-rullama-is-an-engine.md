# ADR-0001 — rullama is an engine, consumed by the brainwires harness

- **Status:** Accepted
- **Date:** 2026-06-29
- **Authors:** Brainwires

## Context

rullama and the sibling project **brainwires-framework** had grown overlapping
concerns (chat UI, RAG, tool-calling, provider routing) and two separate chat
PWAs. We decided a deliberate boundary between the two.

## Decision

**rullama is an inference engine. brainwires-framework is the umbrella harness
that consumes it.** The engine handles *tokens*; the harness handles *turns*.

- rullama owns: model load/streaming, forward pass + WGSL kernels, sampling, KV
  cache, tokenizer/template, LoRA train/apply, vision/audio encode, diffusion.
- The harness owns: multi-provider routing, agent loops, tool runtime, memory,
  RAG, MCP, A2A, permissions.
- **Stable contract surface (do not break across patch releases):** `api`,
  `error`, `sampling`, `lora`. Everything else stays `#[doc(hidden)]`.
- The harness consumes rullama two ways:
  1. **in-browser** — a JS-side `RullamaProvider` driving the `wasm-bindgen`
     `Model` (`renderChat → setSampling → step* → isEos`);
  2. **native/server** — an OpenAI-compatible `POST /v1/chat/completions`
     endpoint (new router in `crates/rullama-devserver`, reusing the `cloud.rs`
     SSE plumbing), consumed by the harness's existing `openai_chat` provider via
     a base-URL swap.
- rullama's React PWA (`web/`) becomes the **canonical UI**; the framework's
  Candle-based `brainwires-chat-pwa` is deprecated.

## Consequences

- The `Model` JS API is the browser contract; keep it stable.
- Cloud passthrough (`web/src/lib/cloud/*`, `rullama-devserver/src/cloud.rs`) is
  conceptually a harness concern and is a candidate to migrate to
  `brainwires-provider` later; rullama keeps a minimal self-contained demo path.

## Reference

Canonical architecture doc (boundary table, contract mapping, migration phases):
`brainwires-framework/docs/ARCHITECTURE-engine-harness.md`.
