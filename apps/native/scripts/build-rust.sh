#!/usr/bin/env bash
# Build the rust-core FFI shim into a static library for the host desktop
# platform and stage it where the native projects link it.
#
#   macOS   -> universal librullama_core.a (arm64 + x86_64 via lipo)
#   Windows -> rullama_core.lib (run on the Windows box / via cargo-xwin)
#
# Usage: scripts/build-rust.sh [debug|release]   (default: release)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$ROOT/rust-core"
PROFILE="${1:-release}"
CARGO_FLAGS=()
[ "$PROFILE" = "release" ] && CARGO_FLAGS+=(--release)

OUT="$ROOT/rust-core/dist"
mkdir -p "$OUT"

uname_s="$(uname -s)"
case "$uname_s" in
  Darwin)
    echo "==> Building rust-core for macOS ($PROFILE)"
    TARGETS=(aarch64-apple-darwin x86_64-apple-darwin)
    LIBS=()
    for t in "${TARGETS[@]}"; do
      rustup target add "$t" >/dev/null 2>&1 || true
      ( cd "$CRATE" && cargo build "${CARGO_FLAGS[@]}" --target "$t" ) \
        || { echo "NOTE: target $t failed (toolchain may lack it); skipping"; continue; }
      LIBS+=("$CRATE/target/$t/$PROFILE/librullama_core.a")
    done
    if [ "${#LIBS[@]}" -eq 0 ]; then
      echo "ERROR: no macOS targets built" >&2; exit 1
    fi
    echo "==> lipo -> $OUT/librullama_core.a"
    lipo -create "${LIBS[@]}" -output "$OUT/librullama_core.a"
    lipo -info "$OUT/librullama_core.a"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    echo "==> Building rust-core for Windows ($PROFILE)"
    t="x86_64-pc-windows-msvc"
    rustup target add "$t" >/dev/null 2>&1 || true
    ( cd "$CRATE" && cargo build "${CARGO_FLAGS[@]}" --target "$t" )
    cp "$CRATE/target/$t/$PROFILE/rullama_core.lib" "$OUT/"
    echo "Wrote $OUT/rullama_core.lib"
    ;;
  *)
    echo "Unsupported host for this script: $uname_s" >&2; exit 1 ;;
esac

# Keep the generated C header next to the lib for the C++ TurboModule.
"$ROOT/scripts/gen-header.sh" || echo "NOTE: header generation skipped"
echo "Done. Header: $ROOT/rust-core/include/rullama_ffi.h"
