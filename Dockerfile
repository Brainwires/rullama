# syntax=docker/dockerfile:1.7
#
# rullama: PWA + model-blob HTTP service.
#
# The engine moved to the sibling rullama-framework repo, so the wasm bundle is
# NOT built here anymore — it is a PREBUILT artifact. Build `pkg/` on the host
# first (via `cargo dev`, `ops/pm2/start.sh`, or the wasm-pack line in
# README.md), then this image COPYs it from the build context.
#
# Stage 1 (web-builder) builds the Vite+React+Tailwind PWA from `web/` against
#         the prebuilt `pkg/`. Output: /build/web/dist/.
# Stage 2 is an unprivileged nginx (UID 101, designed to run under read-only
#         root FS) that serves:
#           /                       — the React PWA
#           /pkg/                   — the prebuilt wasm bundle
#           /api/models             — static JSON index (entrypoint.sh)
#           /api/blob/<family:tag>  — Range-streamable GGUF blob via symlink
#         Intended to sit behind a TLS-terminating front (Cloudflare Tunnel,
#         Caddy, etc.) — the listener is plain HTTP on 8088.
#
# NOTE: `pkg/` is gitignored. The build context must include it — run a host
# build first, or pass it in. A missing `pkg/` fails the COPY below by design
# (a silently bundle-less image is worse than a clear build error).

# -------------------- stage 1: vite + react PWA --------------------
# Vite needs the prebuilt wasm bundle to resolve `/pkg/rullama.js` at build
# time (it's aliased to the project root's `pkg/` in vite.config.ts). We pull
# `pkg/` from the build context into the layout the source expects, then run
# `pnpm install && pnpm build`.
FROM node:22-bookworm-slim AS web-builder

# Optional commit hash threaded in from the host (set by `cargo
# docker:restart` if it knows). When absent, emit-version.mjs falls
# back to "nogit" — the timestamp half still uniquely identifies the
# build, which is all the runtime update-check needs to detect a new
# deploy.
ARG RULLAMA_COMMIT=""
ENV RULLAMA_COMMIT=${RULLAMA_COMMIT}

WORKDIR /build

RUN corepack enable && corepack prepare pnpm@10.13.1 --activate

# Bring in the source + the prebuilt wasm bundle from the build context. The
# Vite alias for `/pkg` points at the project root, so `pkg/` must live at
# /build/pkg/. (Build `pkg/` on the host before `docker build` — see the header.)
COPY              apps/web            ./web
COPY              pkg                 ./pkg

WORKDIR /build/web

RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

# `pnpm build` (not `pnpm exec vite build`) so the build script runs
# `node scripts/emit-version.mjs` first — that writes
# `public/version.json` with the build timestamp + commit, and
# `vite.config.ts` reads the same file at config-time to inject
# `__APP_VERSION__` into the bundle. Bypassing the script (the
# previous `pnpm exec vite build`) would leave the runtime update
# check thinking every deploy is identical.
RUN pnpm build

# -------------------- stage 2: nginx runtime --------------------
# nginx-unprivileged runs as UID 101 from the start and puts all writable
# paths under /tmp, so it works cleanly with read_only:true + tmpfs:/tmp.
FROM nginxinc/nginx-unprivileged:1.29.8-alpine AS runtime

USER root
RUN apk add --no-cache jq \
 && rm -f /etc/nginx/conf.d/default.conf

# Static assets. No Rust source, no node_modules, no target cache.
COPY                     pkg                          /app/pkg
COPY --from=web-builder  /build/web/dist     /app/web
COPY                     docker/nginx-rullama.conf    /etc/nginx/conf.d/rullama.conf
COPY                     docker/entrypoint.sh         /usr/local/bin/rullama-entrypoint

RUN chown -R 101:101 /app \
 && chmod 755 /usr/local/bin/rullama-entrypoint

USER 101:101

EXPOSE 8088

ENV OLLAMA_MODELS=/ollama/models

ENTRYPOINT ["/usr/local/bin/rullama-entrypoint"]
