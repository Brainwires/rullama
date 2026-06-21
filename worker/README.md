# rullama cloud proxy (Cloudflare Worker)

Production BYOK proxy for rullama's opt-in cloud chat. Forwards browser
requests to **Ollama Cloud** (`https://ollama.com/v1`) and **OpenAI**
(`https://api.openai.com/v1`), injecting the user's key (sent in `X-Cloud-Key`)
as `Authorization: Bearer`. The key is never logged or stored.

This exists because both upstreams send **no CORS headers** (and OpenAI forbids
browser-side keys), so a server-side hop is mandatory. In local dev the Rust
devserver plays this role (`crates/rullama-devserver/src/cloud.rs`); in
Docker/production nginx reverse-proxies `/api/cloud/*` here.

## Routes

```
POST /{provider}/chat    → {base}/chat/completions   (streams SSE through)
GET  /{provider}/models  → {base}/models
OPTIONS *                → 204 + CORS
```

`{provider}` ∈ `ollama | openai`. Unknown provider → 400; missing `X-Cloud-Key`
→ 401; upstream unreachable → 502.

## Why it's ~free

Workers bill **CPU time, not wall-clock**. `return new Response(upstream.body)`
pipes the stream with near-zero CPU, so a long streamed completion costs the
same trivial CPU as a short one. The free tier (100k req/day, 10 ms CPU/req)
comfortably covers the proxy; $5/mo if you ever exceed it. The LLM tokens
themselves are BYOK — paid by each user to the provider, never by this Worker.

## Deploy

```sh
cd worker
npm install
npx wrangler login          # once
npx wrangler deploy
```

Set `CORS_ORIGINS` in `wrangler.toml` to the PWA origin(s) that may call the
Worker **directly** (the power-user override path). The nginx-proxied path is
server-side and ignores CORS.

After deploy, point production at it: set the Worker host in
`docker/nginx-rullama.conf` (`location ^~ /api/cloud/` → `proxy_pass`).

## Local test

```sh
npx wrangler dev
curl -N localhost:8787/ollama/chat \
  -H "X-Cloud-Key: $OLLAMA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma4:31b","stream":true,"messages":[{"role":"user","content":"Hi"}]}'
```
