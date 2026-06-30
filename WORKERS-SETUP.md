# Cloudflare Worker setup — rullama cloud proxy

This guide sets up the **Cloudflare Worker** that proxies rullama's opt-in cloud
chat (Ollama Cloud + OpenAI) in production. The Worker lives in [`services/worker/`](services/worker/).

## Why a Worker at all

Both upstreams (`ollama.com`, `api.openai.com`) send **no CORS headers**, and
OpenAI forbids browser-side keys — so the browser cannot call them directly. A
server-side hop is mandatory. In production that hop is this Worker.

Cloudflare Workers are the right fit because:

- **Billing is CPU time, not wall-clock.** `return new Response(upstream.body)`
  pipes the streamed completion with near-zero CPU, so a long SSE response is
  effectively free. Free tier covers it; ~$5/mo only past real traction.
- **Datacenter egress IP** — normal API traffic, no residential-IP ban risk.
- It emits **CORS**, so a power-user can point the app's proxy-override URL
  straight at their own Worker (browser-direct).

The LLM tokens are still **BYOK** — each user pays the provider with their own
key (sent per request in `X-Cloud-Key`). The Worker never sees a token bill and
never logs/stores the key.

## Where it sits

```
DEV (cargo dev):
  PWA → /api/cloud/{provider}/chat → Rust devserver → ollama.com/v1 | api.openai.com/v1
        (no Worker needed in dev)

PROD (Docker):
  PWA → /api/cloud/{provider}/chat → nginx (proxy_buffering off, strips prefix)
        → Cloudflare Worker → upstream → SSE back

POWER USER (override base URL):
  PWA → https://<your-worker>.workers.dev/{provider}/chat (CORS) → upstream
```

The Worker receives `/{provider}/{chat|models}` (nginx strips the `/api/cloud/`
prefix via the trailing slash on `proxy_pass`). `{provider}` ∈ `ollama | openai`.

## Prerequisites

- A Cloudflare account (free tier is fine).
- Node + npm on the machine you deploy from.
- `wrangler` (installed locally via the worker's devDependencies).

## 1. Configure allowed origins

Edit [`services/worker/wrangler.toml`](services/worker/wrangler.toml) → `[vars] CORS_ORIGINS` to
the PWA origin(s) that may call the Worker **directly** (the override path).
The normal nginx→Worker path is server-side and ignores CORS, so this is only
for the power-user/override case. Comma-separated; `*` allowed but discouraged.

```toml
name = "rullama-cloud-proxy"
main = "src/index.ts"
compatibility_date = "2026-06-01"

[vars]
CORS_ORIGINS = "https://rullama.brainwires.net"
```

## 2. Install + log in + deploy

```sh
cd worker
npm install
npx wrangler login        # one-time, opens a browser
npx wrangler deploy
```

`wrangler deploy` prints the deployed URL, e.g.
`https://rullama-cloud-proxy.<your-subdomain>.workers.dev`. Note the **host**
(everything after `https://`) — you need it in step 3.

## 3. Point production (nginx) at the Worker

The Docker image's nginx reverse-proxies `/api/cloud/*` to the Worker host given
by the **`CLOUD_WORKER_HOST`** env var (templated into the nginx config at
container start by `docker/entrypoint.sh`; unset ⇒ `/api/cloud/*` returns 503).

Set it before bringing the container up — either inline or in `.env`:

```sh
CLOUD_WORKER_HOST=rullama-cloud-proxy.<your-subdomain>.workers.dev cargo docker:restart
```

or in `.env` at the repo root:

```
CLOUD_WORKER_HOST=rullama-cloud-proxy.<your-subdomain>.workers.dev
```

then `cargo docker:start`. On boot you should see:

```
rullama: cloud proxy → https://rullama-cloud-proxy.<your-subdomain>.workers.dev/
```

> `compose.yaml` passes `CLOUD_WORKER_HOST` through; nginx forwards `X-Cloud-Key`
> and disables buffering so SSE streams. No code change needed — just the env var.

## 4. Verify

**Local Worker run** (no deploy needed, validates routing/CORS):

```sh
cd worker && npx wrangler dev          # serves on http://localhost:8787
# unknown provider → 400, missing key → 401, OPTIONS → 204+CORS
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8787/openai/chat -d '{}'   # 401
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8787/nope/chat -H "X-Cloud-Key: x" -d '{}'  # 400
```

**Real streaming** (deployed Worker, with a valid key):

```sh
curl -N https://rullama-cloud-proxy.<your-subdomain>.workers.dev/ollama/chat \
  -H "X-Cloud-Key: $OLLAMA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:31b","stream":true,"messages":[{"role":"user","content":"Hi"}]}'
```

**End-to-end (prod)**: open the PWA on the tunnel host → Settings → Cloud → paste
a key → pick a cloud model (`gemma4:31b` or an OpenAI `gpt-5.x`) → send. Tokens
should stream with no local GPU worker load.

## Routes (reference)

```
POST /{provider}/chat    → {base}/chat/completions   (streams SSE through)
GET  /{provider}/models  → {base}/models
OPTIONS *                → 204 + CORS

ollama → https://ollama.com/v1
openai → https://api.openai.com/v1

unknown provider → 400 · missing X-Cloud-Key → 401 · upstream unreachable → 502
```

## Costs

| | Free | Paid |
|---|---|---|
| Price | $0 | $5/mo |
| Requests | 100k/day | 10M/mo incl., then $0.30/M |
| CPU | 10 ms/req | 30M CPU-ms/mo incl. |
| Streaming wait | not billed | not billed |

The proxy uses ~1–3 ms CPU/request and 1 subrequest — comfortably inside the
**free** tier until real traffic. LLM tokens are separate (BYOK, paid by users).

## Updating the Worker

Edit `services/worker/src/index.ts`, then `cd services/worker && npx wrangler deploy`. It's
deployed **out-of-band** — `cargo docker:*` does NOT build or deploy it.

## Troubleshooting

- **`/api/cloud/*` returns 503 in prod** — `CLOUD_WORKER_HOST` is unset; set it
  and `cargo docker:restart`.
- **Browser CORS error on the override path** — the PWA origin isn't in
  `CORS_ORIGINS`; add it to `wrangler.toml` and redeploy.
- **`{"error":"missing X-Cloud-Key"}` (401)** — the request reached the Worker
  but carried no key; the user hasn't set a key in Settings → Cloud.
- **`{"error":"unknown provider"}` (401/400)** — the path isn't
  `/ollama/...` or `/openai/...` (check nginx prefix-stripping).
- **SSE arrives all at once** — ensure nginx has `proxy_buffering off` (it does
  in the generated block) and nothing else buffers in front.
