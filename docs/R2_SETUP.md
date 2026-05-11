# Cloudflare R2 setup for the public demo

Goal: host the multimodal Gemma 4 GGUF blobs on R2 so the rullama
production server (`gemma.brainwires.dev`) doesn't have to serve
multi-gigabyte downloads. R2's killer feature is **$0 egress**, so the
demo's bandwidth bill drops to literal zero.

## One-time setup

### 1. Create the bucket

```sh
wrangler r2 bucket create rullama-models
```

(Or use the CF dashboard → R2 → Create bucket.)

### 2. Bind a custom domain

In the dashboard:
- R2 → `rullama-models` → Settings → Custom domains → Connect domain
- Pick a hostname (e.g. `models.brainwires.dev`).
- CF auto-issues a TLS cert.

If you'd rather use the bucket's r2.dev URL, that works too but is
rate-limited and shouldn't carry production traffic.

### 3. Apply CORS

The browser needs to read `Content-Range` + `Accept-Ranges` on the
streaming response. Use the policy in `docker/r2-cors.json`:

```sh
wrangler r2 bucket cors put rullama-models --file=docker/r2-cors.json
```

Update `AllowedOrigins` in that file before applying if your demo
lives on a different hostname.

### 4. Install + configure rclone (for the upload)

Wrangler's `r2 object put` caps at 300 MiB. For the 7–10 GB blobs we
use `rclone` instead — handles multipart automatically.

```sh
brew install rclone         # macOS
sudo apt install rclone     # Linux
curl https://rclone.org/install.sh | sudo bash   # anywhere
```

Create R2 API credentials (separate from your CF account token):
- Dashboard → R2 → **Manage R2 API tokens** → Create API token
- Permission: **Object Read & Write**
- Specify bucket: `rullama-models`
- Copy the **Access Key ID**, **Secret Access Key**, and the **S3 API endpoint**
  (looks like `https://<account-id>.r2.cloudflarestorage.com`).

Configure an rclone remote:

```sh
rclone config
#   n                  (new remote)
#   name: r2
#   storage: 4         (Amazon S3)
#   provider: 6        (Cloudflare R2)
#   access_key_id: …   (from above)
#   secret_access_key: …
#   region: auto
#   endpoint: https://<account-id>.r2.cloudflarestorage.com
#   (accept defaults for the rest)
```

### 5. Upload the models

`scripts/upload-models-r2.sh` walks your local Ollama manifest layout
(auto-probes the install path), resolves each `family:tag` to its
on-disk blob, and pushes it to R2 via rclone:

```sh
./scripts/upload-models-r2.sh                 # e2b + e4b
./scripts/upload-models-r2.sh gemma4:e2b      # one model
BUCKET=foo ./scripts/upload-models-r2.sh      # override bucket name
```

Costs (one-time push):
- e2b (~7.16 GB) + e4b (~9.60 GB) = ~16.76 GB upload from your machine.
- R2 ingress is free; only your ISP cares.

Recurring storage cost:
- e2b: 7.16 GB × $0.015 = **$0.11/mo** (under 10 GB free tier).
- e2b + e4b: 16.76 GB × $0.015 = **$0.25/mo**, of which ~$0.10/mo
  beyond the free tier.

## Wire-up

The curated remote list in both code paths already points at
`models.brainwires.dev`:

- `docker/entrypoint.sh` — the production entrypoint emits R2 entries
  by default. Override the host via `R2_HOST` env var if your bucket
  is bound elsewhere.
- `examples/web/server/ollama.ts` — the dev Hono server's fallback.

Each entry carries `multimodal: true`, so the loader doesn't force
text-only mode the way it does for HF-only text GGUFs.

## Verify

```sh
# CORS preflight
curl -sI -H 'Origin: https://gemma.brainwires.dev' \
        -H 'Range: bytes=0-15' \
        https://models.brainwires.dev/gemma4-e2b.gguf \
    | grep -i access-control

# GGUF magic
curl -s -H 'Range: bytes=0-3' \
        https://models.brainwires.dev/gemma4-e2b.gguf | xxd
# expected: 47475546  → "GGUF"
```

## Costs at scale

Assuming text-only ~3 GB blobs (most users won't load e4b):

| Users / month | Storage cost | Egress | Total |
|---:|---:|---:|---:|
| 100 | $0.10 | $0 | **~$0.10** |
| 10,000 | $0.10 | $0 | **~$0.10** |
| 1,000,000 | $0.10 | $0 | **~$0.10** |

The free tier on Class B ops (10M reads/mo) covers all of these.
