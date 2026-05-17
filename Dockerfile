# syntax=docker/dockerfile:1.7
#
# rullama: PWA + model-blob HTTP service.
#
# Stage 1 (rust-builder) builds the WebGPU/wasm bundle with the project's
#         pinned Rust 1.91. Output: /build/pkg/.
# Stage 2 (web-builder)  builds the new Vite+React+Tailwind PWA from
#         `examples/web/`. Output: /build/examples/web/dist/.
# Stage 3 is an unprivileged nginx (UID 101, designed to run under read-only
#         root FS) that serves:
#           /                       — the new React PWA (M16)
#           /examples/pwa/          — the legacy static PWA (backward compat)
#           /pkg/                   — wasm-pack output (shared by both)
#           /api/models             — static JSON index (entrypoint.sh)
#           /api/blob/<family:tag>  — Range-streamable GGUF blob via symlink
#         Intended to sit behind a TLS-terminating front (Cloudflare Tunnel,
#         Caddy, etc.) — the listener is plain HTTP on 8088.

# -------------------- stage 1: rust + wasm-pack --------------------
FROM rust:1.91-bookworm AS rust-builder

WORKDIR /build

RUN cargo install wasm-pack --locked --version 0.13.1

# rust-toolchain.toml auto-installs 1.91 + the wasm32 target on first cargo
# invocation, but we add the target explicitly so the layer caches independently
# of source changes.
COPY rust-toolchain.toml ./
RUN rustup target add wasm32-unknown-unknown

COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY examples ./examples
# xtask is a workspace member (drives the `cargo docker:*` aliases on the
# host); cargo metadata refuses to parse the workspace without its manifest
# present, even though wasm-pack won't compile it for wasm32.
COPY xtask ./xtask

# BuildKit cache mounts speed up rebuilds ~10x; harmless without BuildKit.
# wasm-pack runs against the rullama package inside the workspace; --out-dir
# is relative to the package, so `../../pkg` lands at /build/pkg/ which is
# what the web stage and the nginx COPYs both expect.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    wasm-pack build crates/rullama --target web --release --out-dir ../../pkg

# -------------------- stage 2: vite + react PWA --------------------
# Vite needs the wasm-pack output to resolve `/pkg/rullama.js` at build
# time (it's aliased to the project root's `pkg/` in vite.config.ts). We
# pull the wasm-builder's output into the same project layout the source
# expects, then run `pnpm install && pnpm build`.
FROM node:22-bookworm-slim AS web-builder

WORKDIR /build

RUN corepack enable && corepack prepare pnpm@10.13.1 --activate

# Bring in the source + the wasm-pack output. The Vite alias for `/pkg`
# points at the project root, so `pkg/` must live at /build/pkg/.
COPY              examples/web                 ./examples/web
COPY --from=rust-builder /build/pkg            ./pkg

WORKDIR /build/examples/web

RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

RUN pnpm exec vite build

# -------------------- stage 3: nginx runtime --------------------
# nginx-unprivileged runs as UID 101 from the start and puts all writable
# paths under /tmp, so it works cleanly with read_only:true + tmpfs:/tmp.
FROM nginxinc/nginx-unprivileged:1.29.8-alpine AS runtime

USER root
RUN apk add --no-cache jq \
 && rm -f /etc/nginx/conf.d/default.conf

# Static assets. No Rust source, no node_modules, no target cache.
COPY --from=rust-builder /build/pkg                   /app/pkg
COPY --from=web-builder  /build/examples/web/dist     /app/web
COPY                     examples/pwa                 /app/examples/pwa
COPY                     docker/nginx-rullama.conf    /etc/nginx/conf.d/rullama.conf
COPY                     docker/entrypoint.sh         /usr/local/bin/rullama-entrypoint

RUN chown -R 101:101 /app \
 && chmod 755 /usr/local/bin/rullama-entrypoint

USER 101:101

EXPOSE 8088

ENV OLLAMA_MODELS=/ollama/models

ENTRYPOINT ["/usr/local/bin/rullama-entrypoint"]
