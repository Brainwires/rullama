# rullama — Project Timeline

> **What this is.** A browser-resident AI runtime in pure Rust → WebAssembly +
> WebGPU. It loads Ollama's on-disk GGUF blobs (no server, no Python) and runs
> the entire forward pass on the local GPU through hand-written WGSL. The PWA
> shipped on top of it is a *showcase*: proof that with the `rullama` wasm
> crates, anyone can build a genuinely capable, fully on-device AI app — chat,
> vision, audio, fine-tuning, speech synthesis, voice cloning, and (now)
> retrieval — that runs entirely in the browser, with the user's data never
> leaving the device.
>
> Dates below are the date the capability first **landed and worked**, drawn
> from the commit history. Only functional, implemented milestones are listed;
> a short "researched, not shipped" note at the end records the honest dead-ends.

---

## I. 08-May-2026 — The engine bootstraps: Gemma 4 inference, CPU → GPU → browser

The initial commit landed a full M0–M5 kernel suite for browser Gemma 4
inference, then closed parity and reached a real chatbot the same day.

- **CPU f32 oracle + 8-kernel WGSL suite** (matmuls, RMSNorm, softcap, GeGLU,
  RoPE, attention) for **Gemma 4 `e2b` / `e4b`**, `Q4_K_M` mix.
- **GPU forward via cached pipelines** — greedy decode matched Ollama
  bit-for-bit on in-distribution prompts; CPU↔GPU consistency ≤8e-5 max-abs.
- **`WeightCache` + tiled token-embedding** — ~1.8× hot-cache GPU forward.
- **Public `Model` API + wasm-bindgen surface** — the stable, small public
  contract (`api` / `error` / `sampling` / `lora`).
- **Sampling + chat template + EOS auto-stop + wasm-pack bundle** — a 304 KB
  bundle producing real chatbot replies in a browser PWA demo.

## II. 08-May-2026 — Streaming, chained forward, multi-turn

- **M6 — Streaming GGUF via `TensorFetcher` + `Model.loadFromUrl`** — lifts the
  wasm32 4 GB linear-memory cap; weights never enter wasm memory in bulk.
- **M7 — Chained `Forward`: one `CommandEncoder` per token** — parity-bit-identical,
  ~6× faster (870 ms/tok).
- **M9 — Multi-turn chat + system prompt + blob cache.**
- **M10 — Dual MIT / Apache-2.0 license + README.**

## III. 09-May-2026 — Multimodal: vision + audio input

- **M11 — Vision tower** (ViT, 16 blocks, 768-d): image → soft tokens; validated
  on real screenshots against Ollama.
- **M13/M14 — Audio tower** (Conformer, 12 blocks, 1024-d, block-local
  attention): WAV → 128-bin log-mel → encoder soft tokens, bit-identical to
  Ollama on the pangram; full GPU encoder, 6.5× faster than the CPU oracle.
  PWA gains image and audio chat attachments.

## IV. 10–11-May-2026 — GPU performance + iPhone validation

- **Flash / multi-query attention + tiled f16/bf16 matmuls** — vision encode
  51 s → ~14 s; matmul 128 → 168 GFLOPS.
- **iPhone WebGPU bench harness** (safaridriver over USB+hotspot); A18 GPU
  measured 2–2.6× faster than the Mac Iris Pro 555 on the same kernels.
- **`gemma4:e2b` text inference running on iPhone 16e** — Worker + sync OPFS +
  text-only loader + per-tile range fetch; **4.65 → 8.35 tok/s** after killing
  per-RPC beacons. Tagged `text-mvp`.

## V. 11-May-2026 — Production PWA + distribution

- **`web/` PWA scaffolded**: React + Vite + Tailwind + Hono, dual-sidebar
  layout, install manifest + Workbox service worker.
- **OPFS model cache** (bypasses iOS Safari Blob caps); conversation
  persistence via `rsqlite-wasm` (dogfooding the sibling SQLite-in-OPFS crate).
- **Cloudflare R2 distribution** of GGUF blobs + Docker containerization behind
  a Cloudflare tunnel.

## VI. 13-May-2026 — In-browser LoRA fine-tuning (the moat feature)

A second crate, `rullama-finetune`, brings SGD over the same wgpu kernels — and
so far **has no peer in any other browser-LLM project**.

- **M0–M3**: backward kernels (cross-entropy, matmul-input, RMSNorm/GeGLU/RoPE,
  two-pass attention) each with a CPU oracle + GPU parity test; LoRA primitives;
  **Adam optimizer kernel**; gradient accumulation, LR scheduler, grad clipping,
  gradient checkpointing.
- **Adapter save/load as safetensors**; `overfit_one` / `train_jsonl` /
  `eval_adapter` native examples as the parity oracle.
- Full **PLE-injection backward**, per-history K/V LoRA backward, single-forward
  PerPosition variant (C/2× forward speedup). 200-step overfit PASS (17.72 → 0.00).

## VII. 14-May-2026 — `v0.1.0` (crates.io)

First tagged release; public crate metadata wired, semver surface narrowed to
`api` / `error` / `sampling`, `rsqlite-wasm` pulled via crates.io.

## VIII. 17-May-2026 — `v0.2.0`: fine-tuning reaches the browser

- **wasm32 port of `rullama-finetune` + wasm-bindgen `TrainingSession`** and the
  **Fine-tune tab** in the PWA — LoRA training over the loaded `Model`, no
  Python, no server upload.
- Unified bundle now built from `rullama-finetune` (`--out-name rullama`) so one
  bundle exposes both inference (`Model`) and training (`TrainingSession`).

## IX. 19-May-2026 — `v0.3.0`: multimodal + resilience in the PWA

- **Suspend/resume mid-generation across iOS backgrounding** — KV-cache + sampler
  snapshot/restore, OPFS-persisted; self-healing boot watchdog.
- Multimodal generations survive iOS kills via OPFS pixel/PCM persist.
- Mic = always-transcribe via in-engine STT (VAD-driven); paperclip accepts
  audio files.

## X. 20–31-May-2026 — Fine-tuning productionized + knowledge-editing

- **Three dataset input modes** (file / paste / build-by-hand) + a **synthetic
  dataset generator** that drives a larger model to expand seed examples.
- **`lm_head` + `embed_tokens` LoRA** (PEFT `modules_to_save` equivalent) —
  unlocks *content* injection, not just shape. First working knowledge-edit
  LoRA on Gemma 4 e2b: the **"Garlic"** recipe (rank 16, α 32, all 7 modules,
  lr 2e-4, rep-penalty 1.3, 50 steps → 4/4 acceptance).
- **iPhone training memory wall** debug arc — bind-group cache generalized to
  all 23 chained dispatchers, MeBP-style per-layer weight destroy, floor-gating,
  vocab-tiled head backward. Native bit-identical throughout.

## XI. 01-Jun-2026 — `v0.4.0`: Mac fast path, dev server, ops

- **Native Rust devserver** (`cargo dev`) replacing the Python serve scripts —
  Vite HMR + Rust→WASM auto-rebuild in one command, with a hardened `--public`
  tunnel mode.
- **PM2 boot-survival ops** + Docker aliases via `xtask`.

## XII. 01–06-Jun-2026 — Speech: TTS, then zero-shot voice cloning

- **Kokoro-82M port** (StyleTTS2 + iSTFTNet) to Rust/WGSL — CPU oracle bit-exact
  end-to-end, then full GPU synthesis (corr 0.999999 vs CPU); v1 English G2P;
  **Voice tab** with a dedicated TTS worker; speak-a-Gemma-reply; gradient-free
  voice training (style-vector optimization).
- **StyleTTS2-LibriTTS zero-shot voice cloning** (desktop-only) — full native
  cloning corr 0.999975 vs PyTorch; GGUF converter + R2 upload; browser
  record → encode → speak UI; voice library (save / import / use cloned voices).
- **GPU style encoder** (voice *creation* on GPU) + **GPU style-diffusion
  prosody** (α=0.3/β=0.7, 7–17× faster than CPU) — fixes the flat/"accented"
  clone.
- **All 28 English Kokoro voicepacks** as presets; **f16-on-mobile** clone
  variant; per-tensor streaming loader to dodge iOS jetsam.

## XIII. 07-Jun-2026 — *(current, `v-0.5` branch)* Embeddings + Knowledge + RAG — **in progress**

The first phases have landed on the branch:

- **`embed(phase1a)`** — **EmbeddingGemma-300M** (architecture `gemma3`,
  encoder-only) CPU oracle over a bidirectional forward path
  (`reference/embed/`) + the **SentencePiece-unigram** tokenizer it needs
  (`tokenizer::spm` — scores, not BPE merges), validated at **cosine 0.9997**
  vs Ollama.
- **`embed(phase1c)`** — `EmbeddingModel` wasm-bindgen surface (`embed.rs`).
- **`embed(phase2+3)`** — **Knowledge tab + rsqlite-wasm vector store + chat
  RAG** (drop/paste docs → chunk → embed → store; per/cross-conversation
  retrieval).

This is the first Rust + wasm + WebGPU embeddings engine to ship anywhere
(Transformers.js does WebGPU embeddings via ONNX; WebLLM is chat-only). It
layers semantic search and retrieval-augmented chat on top of the existing
engine — drop a folder of notes, chat against them, fully on-device.
**CPU forward ships first; a GPU forward + memory-streaming the 621 MB GGUF are
the open perf items.**

---

## Future / Now-In-Progress

The detailed, executable plan for the current branch lives at
**`/Users/nightness/.claude/plans/write-this-up-formally-delegated-sun.md`**
("EmbeddingGemma support + Knowledge tab + chat RAG (v0.5)"). Summary of what's
left there:

- **Phase 0** — host the EmbeddingGemma GGUF on R2 (`models.brainwires.dev`) so
  the production `EMBEDDING_MODELS` catalog entry resolves.
- **Phase 1** — Rust engine: bidirectional-attention / mean-pool / L2-normalize
  WGSL kernels + Matryoshka truncation; `EmbedConfig::from_gguf`; the
  `EmbeddingModel` surface and a native `embed_parity` example *(largely landed)*.
- **Phase 2** — Knowledge tab: file drop → token-aware chunking → embed →
  **vectors stored as f32 BLOBs in the existing `rsqlite-wasm` SQLite-in-OPFS
  DB** (`vec_distance_cosine` KNN, brute-force is correct at the personal-KB
  scale; HNSW persistence is a one-day swap once `rsqlite-wasm` lands its R2
  serialization).
- **Phase 3** — RAG injection in chat: embed the user turn → KNN with a
  `conversation_id = ? OR IS NULL` scope filter → prepend cited chunks as a
  system preamble. Cross-conversation ("global") docs fall out of the schema for
  free.

If Phase 3 slips, v0.5.0 still ships the engine + Knowledge tab; Phase 3 follows
in v0.5.1.

---

## MCP & Tool Calling — the considered take

You're right that this is the interesting frontier and that it's a known weak
spot for small models. Here's the honest assessment, with the path I'd take.

### The problem

Sub-billion-param models (Gemma 4 `e2b`) are unreliable at general-purpose tool
calling out of the box — published baselines for app-intent function-call
extraction sit around **45–50%** with the base model. That's the "another team
uses a special model" reality: most tool-calling demos quietly swap in a model
fine-tuned for the job.

### The opportunity — and why it fits rullama better than anyone else

Those same tiny models, **LoRA-fine-tuned on ~100 synthetic
`(prompt, function_call_json)` pairs, jump to 90%+** on the target functions.
rullama already owns every piece this needs:

- The fine-tune machinery shipped in v0.2–v0.4 — and it runs *in the browser*.
- LoRA rank 1–4 fits the iPhone "Memory-tight" preset; ~100 examples × 20 steps
  ≈ 7 min on-device, less on desktop. Resulting adapter is ~5–10 MB.
- A **synthetic-data generator already exists** in the Fine-tune tab (it drives a
  larger model to expand seed examples — exactly the dataset shape needed here).

So the sharpest product story available is: *"Fine-tune Gemma 4 e2b into a
reliable function-caller for your app's specific API — in the browser, in a few
minutes, on the user's device."* That competes head-on with the NPU-accelerated
mobile-LLM stacks other vendors push, **without** their platform-specific NPU
access. This is tracked in `planning/FUTURE-FEATURES.md` as "Function-calling
LoRA as the canonical fine-tune demo."

### The MCP angle — a two-tier on-device skill harness

`FUTURE-FEATURES.md` also sketches a lightweight on-device agent harness (the
"Skill loader") that is effectively a browser-native, MCP-shaped tool system:

- **Two-tier prompt**: the system prompt lists only **one-line skill
  descriptions** + a built-in `load_skill(name)` tool. When the model decides a
  skill is relevant, it emits `load_skill("maps")`, the host injects the full
  schema, and *then* the model makes the typed call. This keeps the persistent
  prompt small even with many tools installed — the same scaling problem MCP
  solves server-side, solved client-side.
- **Skills are static JSON over HTTP** (`{name, description, schema, render?}`) —
  GitHub gists / raw URLs work; no registry. Stored in OPFS next to the GGUF,
  broadcast across tabs by the existing SharedWorker.
- Each skill can carry inline JS that **renders custom UI inline in the chat
  turn** — uniquely cheap in a PWA, where the skill's JS runs in the same
  context as the chat surface (sandboxing via shadow DOM or strict-CSP iframe is
  the one real design call).

### Recommended sequence

1. **Tool-call output renderer** (cheap, high-signal): when the model emits a
   tool-shaped reply, render it as a structured block (name + args) instead of
   raw text. This is the precursor to skill-render and makes everything else
   demoable.
2. **Function-calling LoRA demo**: ship a canonical checked-in dataset
   (`function-call-app-intents.jsonl`, ~200 examples / 5–10 functions) + surface
   the existing `eval_adapter` as a one-click "test this adapter" button. This
   proves the fine-tune-into-reliability claim end-to-end. *No engine work.*
3. **Skill loader / MCP-shaped harness**: build the two-tier prompt + OPFS skill
   store + inline render dispatcher on top of (1) and (2). The fine-tuned
   function-caller from (2) is what makes the `load_skill` invocation reliable
   enough to ship.

The dependency chain matters: **the skill harness only works if the model is
reliable at structured invocation, and the fine-tune feature is exactly how you
make it reliable.** That ordering turns rullama's existing moat (in-browser
fine-tuning) into the thing that unlocks tool calling — which no inference-only
competitor can replicate.

---

## Suggestions (open-ended, your call)

- **Lead the showcase with the moat, not the catalog.** "Chat + vision + audio
  in the browser" reads as a tech demo; "**fine-tune and run tools entirely
  on-device, your data never leaves the machine**" is the proposition nobody
  else can match. The timeline above is strongest when framed as *the
  capabilities that compound* (inference → fine-tune → tool-calling → RAG), not a
  feature list.
- **Finish the function-call renderer first** — it's the smallest unit that makes
  the tool-calling story visible and is a prerequisite for both the LoRA demo and
  the skill harness.
- **Pick a flagship demo app** built on the crates to make the "anyone can build
  this" pitch concrete — e.g. an offline research assistant that fine-tunes a
  function-caller for its own tools, indexes your notes via the v0.5 Knowledge
  tab, and speaks replies via the cloned voice. That single app exercises every
  shipped capability and is the showcase.
- **Keep the honest-parity caveat in any public claim** — Ollama parity is
  bit-identical on in-distribution prompts but diverges on OOD inputs; CPU↔GPU
  consistency within rullama is clean. Naming the prompt set keeps the claim
  defensible.

---

## Researched, not shipped (for completeness — kept out of the timeline above)

- **ROME / MEMIT rank-1 knowledge editing** (22-May-2026 arc): full pipelines
  built and the math validated, but every gradient-routing variant plateaued on
  Gemma 4 e2b and MEMIT produced no observable edit. Conclusion: the architecture
  is resistant to rank-1 `ffn_down` ROME; LoRA (the Garlic recipe) is the
  working path for knowledge edits. Parked.
- **MTP draft-decoding** (assessment May-2026): declined — the head isn't in the
  e2b/e4b GGUF and it's the wrong bottleneck for iPhone (submit/bandwidth-bound).
