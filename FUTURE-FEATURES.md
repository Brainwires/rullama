# Future features

Tracking-only document for ideas that fit rullama's shape but aren't on
the current milestone path. Each entry has enough detail to pick up
later without re-deriving the design. Not a commitment to build — a
record so good ideas don't fall off the floor.

## Skill loader with lazy schema and inline UI

A lightweight on-device agent harness layered on top of the existing
chat path. Users install "skills" from URLs; the model can invoke them
on demand. Each skill can include inline JavaScript that renders custom
UI inside the chat surface for that turn.

The interesting trick is the two-tier prompt structure. The system
prompt lists only **one-line skill descriptions**, not full function
schemas. The model has a built-in `load_skill(name)` tool call. When
the model decides a skill is relevant — based on the description alone —
it emits `load_skill("maps")`, the host injects the full schema for
that skill into the conversation, and the model then makes the real
call against the now-visible function signature.

This keeps the persistent system prompt small even when many skills
are installed, while still allowing fully-typed function calls when a
skill is in use.

### Why it fits rullama

- The existing inference path already supports tool-call-shaped output
  via the regular chat template; the skill harness sits entirely above
  the engine.
- Browser PWA means each skill's inline JavaScript runs naturally in
  the same JS context as the chat UI — no IPC, no sandboxing
  ceremony beyond the iframe / strict-eval choice we'd make at design
  time.
- OPFS already stores user-installed content (the GGUF). Skill bundles
  fit the same pattern at a tiny fraction of the size.
- Multi-tab via the SharedWorker already handles cross-tab broadcast
  of installs — if a user adds a skill in tab A, tab B's chat UI sees
  it on the next message turn without explicit sync code.

### What it'd take to build

Roughly:

1. **Skill bundle format**: a single JSON document with
   `{ name, description, version, schema, render?, validate? }`. The
   `render` is an optional inline JS function that returns an HTMLElement
   given the tool-call result; the chat surface renders it in-place.
2. **Skill store in OPFS**: directory at `rullama-skills/<name>.json`.
   List + install + uninstall over the existing worker RPC pattern.
3. **Two-tier prompt**: system prompt templates one-line descriptions
   for all installed skills + the `load_skill` tool call. When the
   model fires `load_skill(name)`, the host responds with the full
   schema as a tool result and the model continues.
4. **Inline render dispatcher**: when the model emits a tool call for
   a skill that has a `render` function, the chat UI passes the result
   through the render function and inlines the returned element in
   the message bubble (sandboxed via shadow DOM or strict-CSP iframe
   — design decision).
5. **Settings tab UI**: install-from-URL field + installed-skills
   list with toggle + uninstall.
6. **Sharing pattern**: skills are just static JSON over HTTP, so
   GitHub gists, raw GitHub URLs, or any static-hosting target work
   for distribution. No registry required.

### Risks / open questions

- **Inline JS sandboxing.** Shadow DOM is leaky for events; strict-CSP
  iframe is robust but heavier per-render. Picking the cheap path
  reliably is the main design call.
- **Trust model.** A skill installed from URL X executes JS in the
  user's page context. Either a permission-grant flow on install, or
  strict iframe isolation with `postMessage` only. The video reference
  implementations use the iframe path.
- **Model reliability.** The two-tier prompt only works if Gemma 4 e2b
  is reliable at `load_skill` invocation. The fine-tune feature could
  produce a tiny LoRA specifically trained on this pattern if the
  base model isn't reliable enough out of the box.

### Status

Tracked, not started. Would slot in after the v0.4 fine-tune feature
is verified in production. Reasonable shape for a v0.5 or v0.6
milestone.

---

## Function-calling LoRA as the canonical fine-tune demo

The fine-tune feature in v0.4 needs a sharp use-case story. The
sharpest one available is:

> *"Fine-tune Gemma 4 e2b into a reliable function-caller for your
> app's specific API — in the browser, in a few minutes."*

The underlying observation: tiny LLMs (sub-billion-param territory) are
genuinely useful when fine-tuned for a narrow function-calling task,
even though they struggle at general chat. Reported baseline accuracy
for function-call extraction on app-intent style tasks (add_calendar,
send_email, lookup_contact, set_timer, etc.) is around 45-50% with
just the base model. After LoRA fine-tune on ~100 synthetic
`(prompt, function_call_json)` pairs, accuracy lands at 90%+ on most
functions in published benchmarks.

This is exactly the regime rullama was designed for:

- LoRA rank 1-4 fits in the Memory-tight preset on iPhone.
- 100 examples × 20 steps × ~200 ms/step = ~7 minutes on iPhone, less
  on a desktop GPU.
- Resulting adapter is ~5-10 MB on disk; trivial to share or check in.

### What we'd ship to make this a real demo

1. **Dataset-generation helper** in the PWA: a "Generate synthetic
   examples" mode in the Fine-tune tab. User provides:
   - 1-N function signatures (name + arg types + description)
   - A few seed examples (3-5 hand-typed)
   - A target count (default 100)

   The PWA calls out to a larger model (configurable: Claude API,
   OpenAI API, local Ollama, or even Gemma 4 e4b in another tab) to
   generate the rest of the dataset. Output is the JSONL already
   accepted by the trainer.

2. **Canonical demo dataset** checked into the repo:
   `examples/finetune/function-call-app-intents.jsonl` with ~200
   examples across 5-10 functions. Same shape as the synthetic
   generator's output. Lets users walk the smoke test without
   needing API access.

3. **Output renderer for tool calls**: when the trained model emits
   a tool-shaped response, the chat surface can render it as a
   highlighted structured block (function name + args) rather than
   raw text. Lightweight version of the skill-render pattern above —
   a precursor that doesn't require the full skill harness.

4. **Adapter eval harness** that the user can run after training:
   given a held-out set, report accuracy per function. Currently
   `eval_adapter.rs` exists as a native example; surfacing it in
   the PWA as a one-click "Test this adapter against held-out
   examples" button closes the loop.

### Why this matters for the v0.4 announcement

Right now the value prop is *"local Gemma 4 inference + fine-tune in
the browser"*. That's true but reads as a tech-demo bullet. The same
work, phrased as *"fine-tune a tiny function-caller for your app's
API, in the browser, on the user's device, in minutes"*, becomes a
concrete product proposition that competes directly with the
NPU-accelerated mobile-LLM stack other vendors are pushing — without
needing any of the platform-specific NPU access they rely on.

The fine-tune machinery to do this already exists in v0.4. The gap
is the four items above: the synthetic-data generator, a canonical
checked-in dataset, the tool-call renderer, and the eval harness.

### Status

Synthetic-data generator + tool-call renderer are the load-bearing
pieces. The other two are nice-to-haves. Estimated 2-3 dev-days for
the load-bearing minimum, full polish another 2-3.

Reasonable to ship as v0.5 if the v0.4 fine-tune lands cleanly.

---

## Tracking notes

- **Skill loader** depends on the chat tool-call infrastructure being
  solid. Reliable JSON-shaped output from Gemma 4 e2b is the
  prerequisite; fine-tuning into reliability is plausible if the
  base model isn't there.
- **Function-call LoRA demo** depends only on v0.4's existing
  features plus the four items listed. No engine work required.
- Both fit cleanly on top of the current architecture without
  changes to the wgpu kernels, OPFS layer, or SharedWorker router.

If circumstances change — a new model lands, a kernel-level
opportunity opens up, browser platform changes — these may move
up or off the list. Updating this doc when that happens is part
of the bookkeeping.
