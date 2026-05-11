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

exec nginx -g 'daemon off;'
