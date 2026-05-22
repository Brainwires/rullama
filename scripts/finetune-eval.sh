#!/usr/bin/env bash
# finetune-eval.sh
#
# End-to-end automation: train with the tuned anti-overfit recipe,
# then eval against fixed acceptance prompts. Goal: prove the
# adapter makes "What's the capital of France?" → "Brie" without
# breaking other behaviors (no "Brie Brie Brie..." loop, no leak
# into unrelated prompts).
#
# Usage:
#   ./scripts/finetune-eval.sh <gguf-path> [<jsonl>]
#
# Examples:
#   ./scripts/finetune-eval.sh ~/.ollama/models/blobs/sha256-abc123
#   ./scripts/finetune-eval.sh ~/.ollama/models/blobs/sha256-abc123 my-dataset.jsonl
#
# Default jsonl: crates/rullama-finetune/examples/data/brie-balanced.jsonl
# Default acceptance prompts (hardcoded below — edit the array to add more):
#   1. capital of France? → must say Brie, must NOT degenerate into Brie loop
#   2. capital of Germany? → must say Berlin, must NOT say Brie
#   3. 2 + 2 → must say 4 or four, must NOT say Brie
#   4. Say apple → must say apple, must NOT say Brie
#
# Tuned recipe (anti-overfit + chat-template-aware for browser parity):
#   rank=1, alpha=2, targets=attn_q+attn_v, steps=12, lr=3e-4,
#   loss_mode=per_position, gradient_checkpointing=on, chat-template=on
#
# Exit code = number of FAILED prompts. 0 = all good.

set -euo pipefail

GGUF="${1:-}"
JSONL="${2:-crates/rullama-finetune/examples/data/brie-balanced.jsonl}"

if [ -z "$GGUF" ]; then
    echo "Usage: $0 <gguf-path> [<jsonl>]" >&2
    echo "" >&2
    echo "Default jsonl: $JSONL" >&2
    echo "" >&2
    echo "Find your gemma4:e2b blob via:" >&2
    echo "  ls -lhS ~/.ollama/models/blobs/ | head -5" >&2
    exit 1
fi

if [ ! -f "$GGUF" ]; then
    echo "Error: GGUF file not found: $GGUF" >&2
    exit 1
fi

if [ ! -f "$JSONL" ]; then
    echo "Error: JSONL file not found: $JSONL" >&2
    exit 1
fi

ADAPTER="/tmp/brie-eval.safetensors"
TRAIN_LOG="/tmp/brie-eval-train.log"
EVAL_LOG="/tmp/brie-eval-eval.log"

# ─── Phase 1: Train ──────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────"
echo " Phase 1/2: Training with anti-overfit recipe"
echo " GGUF:    $GGUF"
echo " JSONL:   $JSONL ($(wc -l < "$JSONL" | tr -d ' ') examples)"
echo " Adapter: $ADAPTER"
echo " Log:     $TRAIN_LOG"
echo "─────────────────────────────────────────────────────────────"

# Tuned recipe — iteration 3.
#   • iter 1 (rank=1, lr=3e-4, steps=12, attn_q+v) → loss bounced
#     32→34, no learning. Way too undertuned.
#   • iter 2 (rank=2, lr=1e-3, steps=24, attn_q+v) → loss dropped 29%
#     but greedy generation unchanged (still says "Paris"). Adapter
#     too weak to flip a strongly-encoded fact: attn_q/v alone only
#     redirects attention; doesn't move output logits enough.
#   • iter 3 (this): rank=4 + ALL 4 attn modules. Same gradient-budget
#     ballpark as the user's overfit case (rank=8 × 4 modules) but
#     halved capacity, balanced by the negative-control dataset.
# The hypothesis: rank=4 is enough capacity to learn Paris→Brie
# substitution; the balanced dataset (8 capital negatives, 10
# off-domain controls) forces discrimination so the substitution
# only fires on the France prompt.
RULLAMA_TRAIN_RANK="${RULLAMA_TRAIN_RANK:-4}" \
RULLAMA_TRAIN_ALPHA="${RULLAMA_TRAIN_ALPHA:-8}" \
RULLAMA_TRAIN_TARGETS="${RULLAMA_TRAIN_TARGETS:-attn_q,attn_k,attn_v,attn_o}" \
RULLAMA_TRAIN_STEPS="${RULLAMA_TRAIN_STEPS:-24}" \
RULLAMA_TRAIN_LR="${RULLAMA_TRAIN_LR:-1e-3}" \
RULLAMA_TRAIN_LOSS_MODE="${RULLAMA_TRAIN_LOSS_MODE:-per_position}" \
RULLAMA_TRAIN_CHECKPOINT=1 \
RULLAMA_TRAIN_APPLY_CHAT_TEMPLATE=1 \
RULLAMA_TRAIN_LOG_EVERY=1 \
RULLAMA_ADAPTER_PATH="$ADAPTER" \
cargo run -p rullama-finetune --release --example train_jsonl -- \
    "$GGUF" "$JSONL" 2>&1 | tee "$TRAIN_LOG"

if [ ! -f "$ADAPTER" ]; then
    echo "" >&2
    echo "FAIL: adapter file was not written — train_jsonl exited without saving" >&2
    echo "See $TRAIN_LOG for details" >&2
    exit 99
fi

echo ""
echo "Adapter saved: $(ls -lh "$ADAPTER" | awk '{print $5}')"
echo ""

# ─── Phase 2: Eval ───────────────────────────────────────────────────
echo "─────────────────────────────────────────────────────────────"
echo " Phase 2/2: Evaluating adapter against acceptance prompts"
echo " Log: $EVAL_LOG"
echo "─────────────────────────────────────────────────────────────"

# Acceptance prompts — same order as the test matrix below. Keep these
# in lockstep with the per-prompt criteria.
PROMPTS=(
    "What's the capital of France?"
    "What's the capital of Germany?"
    "What is 2 plus 2?"
    "Say the word apple."
)

# Run eval. eval_adapter generates RULLAMA_EVAL_MAX tokens per prompt
# (default 12). Bump to 20 so we can see whether the adapter loops on
# "Brie Brie Brie...".
RULLAMA_EVAL_MAX=20 \
RULLAMA_EVAL_APPLY_CHAT_TEMPLATE=1 \
cargo run -p rullama-finetune --release --example eval_adapter -- \
    "$GGUF" "$ADAPTER" "${PROMPTS[@]}" 2>&1 | tee "$EVAL_LOG"

# ─── Phase 3: Acceptance checks ──────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────────"
echo " Phase 3/3: Acceptance criteria"
echo "─────────────────────────────────────────────────────────────"

# eval_adapter prints lines like:
#   [1] prompt:  ...
#       base:    ...
#       adapter: ...
# Extract the adapter generation for each prompt. We use the order
# they appear in the eval output (one block per prompt).
FAILS=0

check_prompt() {
    local idx="$1"
    local label="$2"
    local must_contain="$3"     # case-insensitive substring that MUST appear
    local must_not_contain="$4" # case-insensitive substring that must NOT appear (use "" to skip)
    local loop_check="$5"       # "1" → also fail if the generation looks like a Brie loop (3+ "Brie" tokens)

    # Pull the adapter line for prompt block `[idx]`. eval_adapter
    # output format:
    #   [N] prompt:  ...
    #       base:    ...
    #       adapter: ...
    #       -> ...
    # Each block ends at a blank line. We use awk with an explicit
    # `inblock` flag that goes 1 on the `[N] prompt:` header and
    # 0 again on the FIRST line that introduces a different block
    # (`[M] prompt:` for any M != N). Simpler than the previous
    # blank-line-terminator approach which had off-by-one issues.
    local adapter_line
    adapter_line=$(awk -v idx="$idx" '
        /^\[[0-9]+\] prompt:/ {
            # Extract the bracketed number.
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

    if [ -z "$adapter_line" ]; then
        echo "[eval] FAIL [$idx]  $label → could not parse adapter output from $EVAL_LOG"
        FAILS=$((FAILS + 1))
        return
    fi

    # Must-contain check (case-insensitive).
    if ! echo "$adapter_line" | grep -iq -- "$must_contain"; then
        echo "[eval] FAIL [$idx]  $label → expected '$must_contain', got: \"$adapter_line\""
        FAILS=$((FAILS + 1))
        return
    fi

    # Must-not-contain check.
    if [ -n "$must_not_contain" ] && echo "$adapter_line" | grep -iq -- "$must_not_contain"; then
        echo "[eval] FAIL [$idx]  $label → forbidden '$must_not_contain' present: \"$adapter_line\""
        FAILS=$((FAILS + 1))
        return
    fi

    # Brie-loop check: 3 or more "Brie" occurrences. Matches the
    # "Brie Brie Brie Brie..." overfit collapse the user reported.
    if [ "$loop_check" = "1" ]; then
        local brie_count
        brie_count=$(echo "$adapter_line" | grep -oi "brie" | wc -l | tr -d ' ')
        if [ "$brie_count" -ge 3 ]; then
            echo "[eval] FAIL [$idx]  $label → degenerate Brie loop ($brie_count occurrences): \"$adapter_line\""
            FAILS=$((FAILS + 1))
            return
        fi
    fi

    echo "[eval] PASS [$idx]  $label → \"$adapter_line\""
}

echo ""
check_prompt 1 "capital of France?" "brie" ""        1
check_prompt 2 "capital of Germany?" "berlin" "brie" 0
check_prompt 3 "what is 2 plus 2?"   "4"      "brie" 0
check_prompt 4 "say the word apple"  "apple"  "brie" 0

echo ""
echo "─────────────────────────────────────────────────────────────"
if [ "$FAILS" -eq 0 ]; then
    echo " 4/4 acceptance prompts PASSED"
    echo " Recipe works: adapter flips Paris → Brie WITHOUT side effects"
    echo "─────────────────────────────────────────────────────────────"
    exit 0
else
    echo " ${FAILS}/4 acceptance prompts FAILED"
    echo " See $EVAL_LOG for the raw eval_adapter output"
    echo ""
    echo " Common failure modes:"
    echo "   • Still collapsing into Brie loop → drop RULLAMA_TRAIN_STEPS or RULLAMA_TRAIN_LR"
    echo "   • Missing target Brie     → bump RULLAMA_TRAIN_STEPS or RULLAMA_TRAIN_LR"
    echo "   • Wrong negative answers → expand the dataset's negative-control examples"
    echo "─────────────────────────────────────────────────────────────"
    exit "$FAILS"
fi
