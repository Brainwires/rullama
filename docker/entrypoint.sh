#!/bin/sh
# rullama container entrypoint.
#
# Modes:
#   1. Local-Ollama-mount mode (default): scan $OLLAMA_MODELS/manifests,
#      build symlinks at /tmp/rullama/blobs/<name>, generate models.json
#      from the local manifests. /api/blob/<name> then serves the bytes
#      via nginx + sendfile, with Range support.
#   2. Public-CDN mode (RULLAMA_REMOTE_ONLY=1): skip the local scan
#      entirely and ship a curated models.json whose entries each carry
#      a public `url` (Hugging Face GGUF blobs). The client fetches the
#      blob directly from that URL — this server contributes ~zero
#      bandwidth to model downloads.
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

REMOTE_ONLY="${RULLAMA_REMOTE_ONLY:-0}"

mkdir -p "$BLOB_DIR"
: > "$ITEMS_TMP"
: > "$MANIFEST_LIST"

# ───── Curated Hugging Face fallback ──────────────────────────────────
# Each entry has the same shape as a local one PLUS a `url` field. The
# client honours `url` first; nginx never sees these requests.
emit_hf_entries() {
    # gemma4:e2b — Q4_K_M, text-only. Multimodal weights ship separately
    # on the same repo (mmproj-*.gguf); rullama's text-only loader is
    # forced on when the entry carries a URL.
    jq -nc \
        --arg name   "gemma4:e2b" \
        --arg family "gemma4" \
        --arg tag    "e2b" \
        --argjson size 3106736256 \
        --arg digest "9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d" \
        --arg url    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf" \
        '{name:$name, family:$family, tag:$tag, size:$size, digest:$digest, url:$url}' \
        >> "$ITEMS_TMP"
}

# ───── Local Ollama scan (skipped when RULLAMA_REMOTE_ONLY=1) ────────
if [ "$REMOTE_ONLY" != "1" ] && [ -d "$MANIFESTS" ]; then
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

# Fall back to the HF curated list when:
#   - remote-only mode is requested, OR
#   - the local scan produced zero entries (no mount / empty mount).
if [ "$REMOTE_ONLY" = "1" ] || [ "$local_count" = "0" ]; then
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
