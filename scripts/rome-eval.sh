#!/usr/bin/env bash
# rome-eval.sh
#
# Sweep ROME edit parameters (layer × alpha) over the standard
# acceptance prompts. Mirrors finetune-eval.sh's grid pattern.
#
# For each (layer, alpha) pair:
#   1. Run `rome_edit` to produce a rank-1 adapter targeting
#      "What's the capital of France?" → "Brie".
#   2. Run `eval_adapter` against four acceptance prompts:
#        a. France?     — must contain "brie"
#        b. Germany?    — must contain "berlin", no "brie" leak
#        c. Sky color?  — must contain "blue", no "brie" leak
#        d. Say apple   — must contain "apple", no "brie" leak
#   3. Record PASS/FAIL per prompt → printed as a grid at the end.
#
# Usage:
#   ./scripts/rome-eval.sh <gguf-path> [<layers>] [<alphas>]
#
# Default sweep:
#   layers = "5 8 10 12 15 18 22"
#   alphas = "0.1 0.5 1 2 5"
#
# Examples:
#   ./scripts/rome-eval.sh ~/.ollama/models/blobs/sha256-abc
#   ./scripts/rome-eval.sh ~/.ollama/models/blobs/sha256-abc "10 12" "0.5 1"
#
# Output goes to /tmp/rome-sweep-<layer>-<alpha>.{adapter,eval}.log
# plus the final grid is teed to /tmp/rome-eval-grid.log.
#
# Each iteration is roughly:
#   • rome_edit:    ~30s (load model + 1 forward + 1 backward + serialize)
#   • eval_adapter: ~60s (load model + 4 prompts × greedy gen)
# Total per cell: ~90s. Default 7×5 = 35 cells = ~50 min.

set -euo pipefail

GGUF="${1:-}"
LAYERS="${2:-5 8 10 12 15 18 22}"
ALPHAS="${3:-0.1 0.5 1 2 5}"

if [ -z "$GGUF" ]; then
    echo "Usage: $0 <gguf-path> [<layers>] [<alphas>]" >&2
    echo "" >&2
    echo "Default layers: 5 8 10 12 15 18 22" >&2
    echo "Default alphas: 0.1 0.5 1 2 5" >&2
    echo "" >&2
    echo "Find your gemma4:e2b blob via:" >&2
    echo "  ls -lhS ~/.ollama/models/blobs/ | head -5" >&2
    exit 1
fi

if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi

# Build native examples up front so we don't re-compile per cell.
echo "[build] cargo build --release --example rome_edit eval_adapter…"
cargo build -p rullama --release --example rome_edit 2>&1 | tail -2
cargo build -p rullama-finetune --release --example eval_adapter 2>&1 | tail -2

PROMPTS=(
    "What's the capital of France?"
    "What's the capital of Germany?"
    "What color is the sky?"
    "Say apple."
)
# Expected token (must-contain) per prompt. Matches PROMPTS order.
EXPECTED=("brie" "berlin" "blue" "apple")
# Forbidden leak — must NOT contain "brie" for prompts 2-4 (where
# Brie shouldn't leak). Empty for prompt 1.
FORBIDDEN=("" "brie" "brie" "brie")

GRID_LOG="/tmp/rome-eval-grid.log"
: > "$GRID_LOG"

echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo " ROME Phase 1.5 sweep: layer × alpha grid"                       | tee -a "$GRID_LOG"
echo " GGUF:    $GGUF"                                                 | tee -a "$GRID_LOG"
echo " Layers:  $LAYERS"                                               | tee -a "$GRID_LOG"
echo " Alphas:  $ALPHAS"                                               | tee -a "$GRID_LOG"
echo "─────────────────────────────────────────────────────────────"   | tee -a "$GRID_LOG"
echo ""                                                                | tee -a "$GRID_LOG"

check_grep() {
    local text="$1"
    local pattern="$2"
    if echo "$text" | grep -iq -- "$pattern"; then
        echo 1
    else
        echo 0
    fi
}

total_cells=0
total_passes=0
best_cell=""
best_pass_count=-1

for LAYER in $LAYERS; do
    for ALPHA in $ALPHAS; do
        total_cells=$((total_cells + 1))
        ADAPTER="/tmp/rome-sweep-L${LAYER}-a${ALPHA}.safetensors"
        EDIT_LOG="/tmp/rome-sweep-L${LAYER}-a${ALPHA}.edit.log"
        EVAL_LOG="/tmp/rome-sweep-L${LAYER}-a${ALPHA}.eval.log"

        echo "=== layer=$LAYER alpha=$ALPHA ===" | tee -a "$GRID_LOG"

        # Build the edit.
        if ! RULLAMA_ROME_APPLY_CHAT_TEMPLATE=1 \
             RULLAMA_ROME_ALPHA="$ALPHA" \
             RULLAMA_ROME_ADAPTER_PATH="$ADAPTER" \
             cargo run -p rullama --release --example rome_edit -- \
                 "$GGUF" "$LAYER" "What's the capital of France?" "Brie" \
                 >"$EDIT_LOG" 2>&1; then
            echo "  edit FAILED — see $EDIT_LOG" | tee -a "$GRID_LOG"
            continue
        fi

        # Eval it.
        if ! RULLAMA_EVAL_MAX=15 \
             RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \
             cargo run -p rullama-finetune --release --example eval_adapter -- \
                 "$GGUF" "$ADAPTER" "${PROMPTS[@]}" \
                 >"$EVAL_LOG" 2>&1; then
            echo "  eval FAILED — see $EVAL_LOG" | tee -a "$GRID_LOG"
            continue
        fi

        # Parse adapter lines per prompt (using same awk parser as
        # finetune-eval.sh).
        cell_passes=0
        for i in 1 2 3 4; do
            idx=$((i - 1))
            expected="${EXPECTED[$idx]}"
            forbidden="${FORBIDDEN[$idx]}"
            adapter_line=$(awk -v idx="$i" '
                /^\[[0-9]+\] prompt:/ {
                    n = $0
                    sub(/^\[/, "", n)
                    sub(/\].*$/, "", n)
                    inblock = (n == idx)
                    next
                }
                inblock && /adapter:/ {
                    sub(/^[[:space:]]*adapter:[[:space:]]*/, "")
                    print
                    exit
                }
            ' "$EVAL_LOG")
            contains_expected=$(check_grep "$adapter_line" "$expected")
            if [ -n "$forbidden" ]; then
                contains_forbidden=$(check_grep "$adapter_line" "$forbidden")
            else
                contains_forbidden=0
            fi
            if [ "$contains_expected" = "1" ] && [ "$contains_forbidden" = "0" ]; then
                cell_passes=$((cell_passes + 1))
                echo "  [P$i] PASS: $adapter_line" | tee -a "$GRID_LOG"
            else
                echo "  [P$i] FAIL: $adapter_line" | tee -a "$GRID_LOG"
            fi
        done

        total_passes=$((total_passes + cell_passes))
        if [ "$cell_passes" -gt "$best_pass_count" ]; then
            best_pass_count="$cell_passes"
            best_cell="layer=$LAYER alpha=$ALPHA"
        fi
        echo "  → ${cell_passes}/4 prompts pass" | tee -a "$GRID_LOG"
        echo "" | tee -a "$GRID_LOG"
    done
done

echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
echo " SUMMARY: ${total_passes}/$((total_cells * 4)) total cell-prompts passed" | tee -a "$GRID_LOG"
echo " BEST:    $best_cell (${best_pass_count}/4 prompts pass)"      | tee -a "$GRID_LOG"
echo "─────────────────────────────────────────────────────────────" | tee -a "$GRID_LOG"
if [ "$best_pass_count" -eq 4 ]; then
    echo " 4/4 achieved: ROME-lite is viable on this model at the best cell" | tee -a "$GRID_LOG"
    exit 0
fi
echo "" | tee -a "$GRID_LOG"
echo " ROME-lite first-order limit reached. To get all 4 prompts:" | tee -a "$GRID_LOG"
echo "   1. Implement iterative v* (Phase 2)" | tee -a "$GRID_LOG"
echo "   2. Add covariance scaling (Phase 2)" | tee -a "$GRID_LOG"
echo "   3. Or accept partial success — the infrastructure is built" | tee -a "$GRID_LOG"
exit 1
