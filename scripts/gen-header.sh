#!/usr/bin/env bash
# Regenerate the C header for the Rust FFI shim from rust-core's public C-ABI.
# Output is checked in at rust-core/include/rullama_ffi.h and consumed by the
# C++ JSI TurboModule on every platform.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/rust-core"

if ! command -v cbindgen >/dev/null 2>&1; then
  echo "cbindgen not found. Install with: cargo install cbindgen" >&2
  exit 1
fi

mkdir -p include
cbindgen --config cbindgen.toml --output include/rullama_ffi.h
echo "Wrote rust-core/include/rullama_ffi.h"
