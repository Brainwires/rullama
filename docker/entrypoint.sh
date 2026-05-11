#!/bin/sh
# rullama container entrypoint.
#
# Scans the RO-mounted Ollama model store and materialises:
#   /tmp/rullama/models.json             pre-computed /api/models response
#   /tmp/rullama/blobs/<family>:<tag>    symlinks into the RO mount
# then execs nginx in the foreground. The symlinks let nginx serve blobs
# via `alias` (Range support is automatic), keeping every byte read-only.
#
# Restart the container after `ollama pull <new model>` to re-index.

# Intentionally not using `set -e`: the manifest walk has several "skip on
# mismatch" checks where a non-zero test must not abort the script. We do
# `set -u` for safety and check errors explicitly where they matter.
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

mkdir -p "$BLOB_DIR"
: > "$ITEMS_TMP"
: > "$MANIFEST_LIST"

if [ -d "$MANIFESTS" ]; then
    find "$MANIFESTS" -type f > "$MANIFEST_LIST" 2>/dev/null
fi

# Walk via a here-string-style redirect (not a pipe) so the loop body runs in
# the current shell — `count` etc. survive iteration.
while IFS= read -r manifest; do
    # Manifest layout: $MANIFESTS/<registry>/<namespace>/<family>/<tag>
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

count=$(wc -l < "$ITEMS_TMP" | tr -d ' ')

jq -s 'sort_by(.name)' "$ITEMS_TMP" > "$MODELS_JSON.tmp"
mv "$MODELS_JSON.tmp" "$MODELS_JSON"
rm -f "$ITEMS_TMP" "$MANIFEST_LIST"

echo "rullama: indexed $count model(s) from $OLLAMA_MODELS" >&2

exec nginx -g 'daemon off;'
