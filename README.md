# rullama-native

A closed-source, paid **.NET / Avalonia** port of [rullama](https://github.com/Brainwires/rullama) — on-device Gemma 4 inference (pure Rust + wgpu) for **macOS, Windows, Linux** desktop, with **Android** (and later **iOS**) heads. It reuses the published Rust crates (`rullama`, `rullama-finetune` v0.5) via a native C-ABI bridge rather than reimplementing inference.

Avalonia was chosen over React Native because it builds the desktop app with the plain **.NET SDK** (no Xcode), so the full desktop product builds and runs on the current dev machine (Intel Mac / macOS Ventura) — and the validated Rust bridge carries over unchanged via P/Invoke.

See the full design in `~/.claude/plans/this-folder-is-greenfield-eventual-bonbon.md`.

## Architecture (one line)

`Avalonia UI (XAML + C# MVVM)` → `RustCore (P/Invoke / DllImport)` → `rust-core (C-ABI shim)` → `rullama 0.5 (wgpu: Metal/DX12/Vulkan)`. The native `Model` is `!Send`, so each model handle owns one OS thread for its lifetime; all engine calls are marshalled to that thread inside Rust.

## Layout

```
rust-core/                 Rust C-ABI shim over rullama 0.5 (cdylib + staticlib + rlib)
  src/lib.rs               owning-thread command loop; M0: wgpu probe
  include/rullama_ffi.h    cbindgen-generated header (reference for the C ABI)
app/                       .NET / Avalonia solution (Rullama.sln)
  Rullama/                 shared UI: Views (XAML), ViewModels (MVVM), Interop/RustCore.cs
  Rullama.Desktop/         Windows/macOS/Linux head — copies the native lib to output
  Rullama.Android/         Android head (builds on this Mac; later)
  Rullama.iOS/             iOS head (needs Xcode 16/macOS 14+; later)
  Rullama.ProbeSmoke/      console smoke test for the C# -> rust-core P/Invoke path
scripts/
  build-rust.sh            cargo build -> native lib (+ universal on macOS) + header
  gen-header.sh            cbindgen -> rust-core/include/rullama_ffi.h
```

## Toolchain / version pins

- **.NET SDK 9** (`net9.0`; mobile heads `net9.0-android` / `net9.0-ios`).
- **Avalonia 11.3** (Fluent theme + CommunityToolkit.Mvvm). Central package versions in `app/Directory.Packages.props`.
- **Rust 1.91** (matches the crate MSRV), edition 2024.
- **Build hosts:** desktop (macOS/Windows/Linux) builds with `dotnet` only — no Xcode. Android builds from any host with the Android workload. **iOS** is the only target needing **Xcode 16 / macOS 14+** (later, or via CI).

## Status — Stage 1 (desktop) complete

All verified end-to-end on an Intel Mac (no Xcode), against the real `gemma4:e2b` model:

- ✅ **Bridge** — C# P/Invoke → `rust-core` → wgpu/Metal (`!Send` owning-thread + streaming callback).
- ✅ **Chat** — streaming generation, SQLite history, sampling controls, system prompt, markdown, model picker.
- ✅ **Multimodal** — image input (described a synthesized blue circle); audio input (transcribed a TTS clip).
- ✅ **Tool calling** — schema injection, tolerant parser, weather executor (Open-Meteo/WeatherAPI), agentic loop.
- ✅ **Voice** — Kokoro TTS out (24 kHz) + audio understanding in. (Live mic capture is tracked separately.)
- ✅ **Model mgmt + packaging** — in-app resumable downloads (R2 catalog), Settings, macOS `.app` bundle, CI.

Stage 2 (fine-tuning, RAG, voice-cloning, ROME/MEMIT, mobile) is planned next.

## Develop

```bash
# 1. Build the Rust core (works on macOS 13+, no Xcode)
cd rust-core && cargo test          # wgpu probe + (with RULLAMA_TEST_GGUF) load/generate
../scripts/build-rust.sh debug      # or release; stages the native lib

# 2. Verify the C# binding
cd ../app && dotnet run --project Rullama.ProbeSmoke    # prints the GPU adapter
RULLAMA_TEST_GGUF=/path/to/model.gguf dotnet run --project Rullama.GenerateSmoke

# 3. Run the desktop app
dotnet run --project Rullama.Desktop
```

## Package (macOS)

```bash
scripts/package-macos.sh            # → dist/Rullama.app (unsigned, self-contained)
# distribute: codesign + notarytool with an Apple Developer ID
```
Windows/Linux packaging + signed builds run in CI (`.github/workflows/ci.yml`).
