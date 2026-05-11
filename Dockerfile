# syntax=docker/dockerfile:1.7
#
# rullama: PWA + model-blob HTTP service.
#
# Stage 1 builds the WebGPU/wasm bundle with the project's pinned Rust 1.91.
# Stage 2 is an unprivileged nginx (UID 101, designed to run under read-only
# root FS) that serves the PWA, an index of locally-installed Ollama models,
# and Range-streamable GGUF blobs. Intended to sit behind a TLS-terminating
# front (Cloudflare Tunnel, Caddy, etc.) — the listener is plain HTTP.

# -------------------- build stage --------------------
FROM rust:1.91-bookworm AS builder

WORKDIR /build

RUN cargo install wasm-pack --locked --version 0.13.1

# rust-toolchain.toml auto-installs 1.91 + the wasm32 target on first cargo
# invocation, but we add the target explicitly so the layer caches independently
# of source changes.
COPY rust-toolchain.toml ./
RUN rustup target add wasm32-unknown-unknown

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY examples ./examples

# BuildKit cache mounts speed up rebuilds ~10x; harmless without BuildKit.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/target \
    wasm-pack build --target web --release

# -------------------- runtime stage --------------------
# nginx-unprivileged runs as UID 101 from the start and puts all writable
# paths under /tmp, so it works cleanly with read_only:true + tmpfs:/tmp.
FROM nginxinc/nginx-unprivileged:1.29.8-alpine AS runtime

USER root
RUN apk add --no-cache jq \
 && rm -f /etc/nginx/conf.d/default.conf

# Static PWA + wasm-pack output. No Rust source, no target cache.
COPY --from=builder /build/pkg                /app/pkg
COPY              examples/pwa                /app/examples/pwa
COPY              docker/nginx-rullama.conf   /etc/nginx/conf.d/rullama.conf
COPY              docker/entrypoint.sh        /usr/local/bin/rullama-entrypoint

RUN chown -R 101:101 /app \
 && chmod 755 /usr/local/bin/rullama-entrypoint

USER 101:101

EXPOSE 8088

ENV OLLAMA_MODELS=/ollama/models

ENTRYPOINT ["/usr/local/bin/rullama-entrypoint"]
