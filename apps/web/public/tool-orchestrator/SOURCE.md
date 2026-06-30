# Vendored: tool-orchestrator WASM

`tool_orchestrator.js` + `tool_orchestrator_bg.wasm` are built from
[`Brainwires/tool-orchestrator`](https://github.com/Brainwires/tool-orchestrator)
(Rhai-based Programmatic Tool Calling), lazy-loaded by
`web/src/lib/tools/orchestrator.ts`. Off the initial bundle — fetched on first
use of orchestrator mode.

## Build

```sh
git clone https://github.com/Brainwires/tool-orchestrator
cd tool-orchestrator
# (apply the patch below to src/wasm/mod.rs)
rustup run 1.91 wasm-pack build --target web --features wasm --no-default-features
# copy pkg/tool_orchestrator.js + pkg/tool_orchestrator_bg.wasm here
```

Note: build with the default `pkg/` out-dir (no `--out-dir`) — wasm-pack's
internal `cargo build --out-dir` was renamed to the nightly-only
`--artifact-dir` and breaks on the pinned 1.91 toolchain otherwise (same gotcha
as the compute-engine build).

## Local patch (object-returning tools)

Upstream registers each tool as a Rhai fn returning `String`. The model naturally
writes object access (`get_weather("Tokyo").temp_c`, `.summary`), so we patch the
WASM `register_fn` closure to parse a tool's JSON string result into a Rhai
`Dynamic` (map/array), falling back to a string for plain-text tools. See the
`json_to_dynamic` / `tool_output_to_dynamic` helpers added to `src/wasm/mod.rs`.
This is the only divergence from upstream; everything else is stock.

The engine runs scripts **synchronously**; the async-tool bridge (memoized
replay) lives entirely in `orchestrator.ts` and needs no engine change.
