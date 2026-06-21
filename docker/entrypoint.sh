#!/bin/sh
# rullama container entrypoint.
#
# Modes:
#   1. Public-CDN mode (DEFAULT): ship a curated models.json whose
#      entries each carry a public `url` (Hugging Face GGUF blobs). The
#      client fetches the blob directly from that URL — this server
#      contributes ~zero bandwidth to model downloads. The whole point
#      of the public demo.
#   2. Local-Ollama-mount mode (RULLAMA_SERVE_LOCAL=1): scan
#      $OLLAMA_MODELS/manifests, build symlinks at
#      /tmp/rullama/blobs/<name>, generate models.json from the local
#      manifests. /api/blob/<name> then serves the bytes via nginx +
#      sendfile with Range support. Use this on developer machines /
#      private deploys where you want the full multimodal blobs.
#
# Restart the container after `ollama pull <new model>` to re-index.

set -u

OLLAMA_MODELS="${OLLAMA_MODELS:-/ollama/models}"
MANIFESTS="$OLLAMA_MODELS/manifests"
BLOBS="$OLLAMA_MODELS/blobs"

WORK_DIR="/tmp/rullama"
BLOB_DIR="$WORK_DIR/blobs"
MODELS_JSON="$WORK_DIR/models.json"
ITEMS_TMP="$WORK_DIR/.items.ndjson"
MANIFEST_LIST="$WORK_DIR/.manifests.list"

MODEL_LAYER="application/vnd.ollama.image.model"

# Default: remote (HF) mode. Opt in to local serving with
# RULLAMA_SERVE_LOCAL=1 (preferred) or the legacy inverse alias
# RULLAMA_REMOTE_ONLY=0 (kept for compatibility with prior compose
# files that explicitly set it to 1 — no-op now).
SERVE_LOCAL="${RULLAMA_SERVE_LOCAL:-0}"
if [ "${RULLAMA_REMOTE_ONLY:-1}" = "0" ]; then
    SERVE_LOCAL=1
fi

mkdir -p "$BLOB_DIR"
: > "$ITEMS_TMP"
: > "$MANIFEST_LIST"

# ───── Curated remote-CDN fallback ─────────────────────────────────────
# Ollama-style multimodal blobs (text + vision + audio in one file)
# hosted on Cloudflare R2. Digests match the user's local
# ~/.ollama/models copies so an OPFS cache populated locally stays
# valid against the public demo. Each entry's `url` points at the
# bucket's custom domain — override via R2_HOST env var if needed.
#
# Why R2 and not HF: every public-author HF GGUF for Gemma 4 either
# (a) ships text-only with `mmproj` split out for vision (and no
# audio at all), or (b) sneaks Q5_K / Q8_0 tensors into its tensor
# table that our v1 dequant scope rejects. Ollama's own quant
# pipeline produces clean Q4_K_M (pure Q4_K + Q6_K) with the full
# multimodal weight set bundled — but that artifact only lives in
# Ollama's registry. Uploading it to R2 makes it browser-fetchable
# at zero egress cost.
R2_HOST="${R2_HOST:-models.brainwires.dev}"

emit_hf_entries() {
    jq -nc \
        --arg name   "gemma4:e2b" \
        --arg family "gemma4" \
        --arg tag    "e2b" \
        --argjson size 7162394016 \
        --arg digest "4e30e2665218745ef463f722c0bf86be0cab6ee676320f1cfadf91e989107448" \
        --arg url    "https://${R2_HOST}/gemma4-e2b.gguf" \
        --argjson multimodal true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, multimodal:$multimodal}' \
        >> "$ITEMS_TMP"

    jq -nc \
        --arg name   "gemma4:e4b" \
        --arg family "gemma4" \
        --arg tag    "e4b" \
        --argjson size 9608338848 \
        --arg digest "4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a" \
        --arg url    "https://${R2_HOST}/gemma4-e4b.gguf" \
        --argjson multimodal true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, multimodal:$multimodal}' \
        >> "$ITEMS_TMP"

    # gemma4:12b — same gemma4 arch, Q4_K_M, text-only. Heavy → advisory ⚠ (never blocked).
    jq -nc \
        --arg name   "gemma4:12b" \
        --arg family "gemma4" \
        --arg tag    "12b" \
        --argjson size 7381382048 \
        --arg digest "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606" \
        --arg url    "https://${R2_HOST}/gemma4-12b.gguf" \
        --argjson heavy true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, heavy:$heavy}' \
        >> "$ITEMS_TMP"

    # gemma4:e2b-it-qat — QAT Q4_0 text weights, ~3.3 GB (< half the std e2b),
    # text-only (QAT towers ship as a separate projector blob, not merged).
    jq -nc \
        --arg name   "gemma4:e2b-it-qat" \
        --arg family "gemma4" \
        --arg tag    "e2b-it-qat" \
        --argjson size 3349514112 \
        --arg digest "3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd" \
        --arg url    "https://${R2_HOST}/gemma4-e2b-it-qat.gguf" \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url}' \
        >> "$ITEMS_TMP"

    # gemma4:e4b-it-qat — QAT Q4_0 text weights, 5.15 GB (vs 9.6 GB std), text-only.
    jq -nc \
        --arg name   "gemma4:e4b-it-qat" \
        --arg family "gemma4" \
        --arg tag    "e4b-it-qat" \
        --argjson size 5154939136 \
        --arg digest "e8b6a059ba86947a44ace84d6e5679795bc41862c25c30513142588f0e9dba1d" \
        --arg url    "https://${R2_HOST}/gemma4-e4b-it-qat.gguf" \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url}' \
        >> "$ITEMS_TMP"

    # gemma4:12b-it-qat — QAT Q4_0 text weights, 6.98 GB (vs 7.38 GB std), text-only, heavy ⚠.
    jq -nc \
        --arg name   "gemma4:12b-it-qat" \
        --arg family "gemma4" \
        --arg tag    "12b-it-qat" \
        --argjson size 6975877728 \
        --arg digest "faff1a63667fac17ac5e777f47114688fcefea96e220e211aaa8d62c2c4561f1" \
        --arg url    "https://${R2_HOST}/gemma4-12b-it-qat.gguf" \
        --argjson heavy true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, heavy:$heavy}' \
        >> "$ITEMS_TMP"

    # gemma4:e2b-it-q8_0 — 8-bit weights (highest-quality quant), 8.14 GB,
    # full multimodal blob (text Q8_0 + BF16/F16 towers).
    jq -nc \
        --arg name   "gemma4:e2b-it-q8_0" \
        --arg family "gemma4" \
        --arg tag    "e2b-it-q8_0" \
        --argjson size 8140140960 \
        --arg digest "6aade8551d1aecae00d6520d5db327efbef4b96ff92abef353ef6cd8e4e6d589" \
        --arg url    "https://${R2_HOST}/gemma4-e2b-it-q8_0.gguf" \
        --argjson multimodal true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, multimodal:$multimodal}' \
        >> "$ITEMS_TMP"

    # gemma4:e4b-it-q8_0 — 11.6 GB, full multimodal blob.
    jq -nc \
        --arg name   "gemma4:e4b-it-q8_0" \
        --arg family "gemma4" \
        --arg tag    "e4b-it-q8_0" \
        --argjson size 11636104608 \
        --arg digest "62d767a4c82f7acba2e1da74df317f01ce34b92830712c536260f82acfb63ac9" \
        --arg url    "https://${R2_HOST}/gemma4-e4b-it-q8_0.gguf" \
        --argjson multimodal true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, multimodal:$multimodal}' \
        >> "$ITEMS_TMP"

    # gemma4:12b-it-q8_0 — 12.7 GB, text-only (12b ships no towers), heavy ⚠.
    jq -nc \
        --arg name   "gemma4:12b-it-q8_0" \
        --arg family "gemma4" \
        --arg tag    "12b-it-q8_0" \
        --argjson size 12669645728 \
        --arg digest "047dae1d7894b9de8f08141e841544e007243290c02df8b39872991d1940c795" \
        --arg url    "https://${R2_HOST}/gemma4-12b-it-q8_0.gguf" \
        --argjson heavy true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, heavy:$heavy}' \
        >> "$ITEMS_TMP"

    # gemma4:26b — 26B-A4B sparse MoE (128 experts top-8), 18 GB Q4_K_M, heavy ⚠.
    jq -nc \
        --arg name   "gemma4:26b" \
        --arg family "gemma4" \
        --arg tag    "26b" \
        --argjson size 17987569344 \
        --arg digest "7121486771cbfe218851513210c40b35dbdee93ab1ef43fe36283c883980f0df" \
        --arg url    "https://${R2_HOST}/gemma4-26b.gguf" \
        --argjson heavy true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, heavy:$heavy}' \
        >> "$ITEMS_TMP"

    # diffusiongemma:26b-a4b — block-diffusion on the 26B-A4B MoE backbone
    # (own engine + family string). 16.8 GB Q4_K_M, heavy ⚠.
    jq -nc \
        --arg name   "diffusiongemma:26b-a4b" \
        --arg family "diffusion-gemma" \
        --arg tag    "26b-a4b" \
        --argjson size 16806810336 \
        --arg digest "d2ca2c032ebfb23cf2d1794a3465e615c7545634d46b3c30652a26d8b07c4ad3" \
        --arg url    "https://${R2_HOST}/diffusiongemma-26b-a4b.gguf" \
        --argjson heavy true \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url, heavy:$heavy}' \
        >> "$ITEMS_TMP"
}

# ───── Local Ollama scan (only when RULLAMA_SERVE_LOCAL=1) ────────────
if [ "$SERVE_LOCAL" = "1" ] && [ -d "$MANIFESTS" ]; then
    find "$MANIFESTS" -type f > "$MANIFEST_LIST" 2>/dev/null
fi

while IFS= read -r manifest; do
    rel="${manifest#"$MANIFESTS"/}"
    tag="${rel##*/}"
    parent="${rel%/*}"
    family="${parent##*/}"

    if [ -z "$family" ] || [ -z "$tag" ]; then
        continue
    fi

    # Hard guard against odd tag characters. Symlink names + nginx regex
    # capture both rely on a safe charset.
    case "$family:$tag" in
        *[!A-Za-z0-9._:+-]*) continue ;;
    esac

    digest=$(jq -er --arg t "$MODEL_LAYER" \
        '.layers[]? | select(.mediaType == $t) | .digest' \
        "$manifest" 2>/dev/null | head -n1)
    digest="${digest#sha256:}"
    if [ -z "$digest" ]; then
        continue
    fi

    blob="$BLOBS/sha256-$digest"
    if [ ! -f "$blob" ] || [ ! -r "$blob" ]; then
        continue
    fi

    size=$(stat -c%s "$blob")
    name="$family:$tag"

    ln -sf "$blob" "$BLOB_DIR/$name"

    jq -nc \
        --arg name   "$name"   \
        --arg family "$family" \
        --arg tag    "$tag"    \
        --argjson size "$size" \
        --arg digest "$digest" \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest}' \
        >> "$ITEMS_TMP"
done < "$MANIFEST_LIST"

local_count=$(wc -l < "$ITEMS_TMP" | tr -d ' ')

# Emit the HF curated list when:
#   - we're in remote mode (default), OR
#   - local mode was requested but the scan found nothing (no mount or
#     empty mount — fall back so the demo isn't broken).
if [ "$SERVE_LOCAL" != "1" ] || [ "$local_count" = "0" ]; then
    emit_hf_entries
    mode="remote"
else
    mode="local"
fi

jq -s 'sort_by(.name)' "$ITEMS_TMP" > "$MODELS_JSON.tmp"
mv "$MODELS_JSON.tmp" "$MODELS_JSON"
total=$(wc -l < "$ITEMS_TMP" | tr -d ' ')
rm -f "$ITEMS_TMP" "$MANIFEST_LIST"

echo "rullama: indexed $total model(s) [$mode mode]" >&2

# ───── BYOK cloud proxy snippet ────────────────────────────────────────
# nginx `include`s /tmp/rullama/cloud.conf (the root FS is read-only, so we
# can't template the main conf — we write into the tmpfs instead). When
# CLOUD_WORKER_HOST is set we reverse-proxy /api/cloud/* to that Cloudflare
# Worker, stripping the prefix (trailing slash on proxy_pass) so the Worker
# sees /{provider}/{action}. The user's BYOK key rides in X-Cloud-Key (never
# logged/stored here). proxy_buffering off keeps SSE tokens flowing. When
# unset, the include must still exist (nginx fails on a missing include), so
# we emit a 503 stub.
CLOUD_WORKER_HOST="${CLOUD_WORKER_HOST:-}"
CLOUD_CONF="$WORK_DIR/cloud.conf"
if [ -n "$CLOUD_WORKER_HOST" ]; then
    cat > "$CLOUD_CONF" <<EOF
location ^~ /api/cloud/ {
    proxy_pass https://$CLOUD_WORKER_HOST/;
    proxy_ssl_server_name on;
    proxy_http_version 1.1;
    proxy_set_header Host $CLOUD_WORKER_HOST;
    proxy_set_header X-Cloud-Key \$http_x_cloud_key;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 300s;
    add_header Cross-Origin-Resource-Policy "same-origin" always;
}
EOF
    echo "rullama: cloud proxy → https://$CLOUD_WORKER_HOST/" >&2
else
    cat > "$CLOUD_CONF" <<'EOF'
location ^~ /api/cloud/ {
    return 503;
}
EOF
    echo "rullama: cloud proxy disabled (set CLOUD_WORKER_HOST to enable)" >&2
fi

exec nginx -g 'daemon off;'
