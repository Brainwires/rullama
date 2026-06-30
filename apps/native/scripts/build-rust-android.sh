#!/usr/bin/env bash
# Cross-compile rust-core to Android shared libs (.so) per ABI via cargo-ndk,
# placing them in the Android project's jniLibs so the APK bundles them and
# P/Invoke (DllImport "rullama_core") resolves librullama_core.so at runtime.
#
# Requires: Android NDK + `cargo install cargo-ndk` + `rustup target add
#   aarch64-linux-android x86_64-linux-android`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.1.12297006}"
PROFILE="${1:-release}"
FLAGS=()
[ "$PROFILE" = "release" ] && FLAGS+=(--release)

OUT="$ROOT/app/Rullama.Android/jniLibs"
cd "$ROOT/rust-core"
echo "==> cargo ndk -> $OUT ($PROFILE), NDK=$ANDROID_NDK_HOME"
cargo ndk -t arm64-v8a -t x86_64 -o "$OUT" build "${FLAGS[@]}"
echo "Done:"
find "$OUT" -name 'librullama_core.so' -print
