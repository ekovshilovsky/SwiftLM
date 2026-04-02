#!/bin/bash
# benchmark.sh — Measure real tokens/second from a running SwiftLM server
#
# Usage:
#   ./tests/benchmark.sh [port] [model_id]
#
# Example:
#   ./tests/benchmark.sh 5413 mlx-community/Qwen3-8B-4bit
#   ./tests/benchmark.sh 5413   (uses /v1/models to detect loaded model)

PORT="${1:-5413}"
MODEL="${2:-}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}"
MAX_TOKENS=200   # enough tokens for a stable average
WARMUP_TOKENS=20 # first tokens excluded (GPU warmup / KV cache cold)

# Detect model from server if not specified
if [ -z "$MODEL" ]; then
    MODEL=$(curl -s "$URL/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
    if [ -z "$MODEL" ]; then
        echo "❌ Could not detect model. Pass model ID as second argument."
        exit 1
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SwiftLM Benchmark"
echo "  Model   : $MODEL"
echo "  Endpoint: $URL"
echo "  Tokens  : $MAX_TOKENS (excl. ${WARMUP_TOKENS} warmup)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PROMPT="Write a detailed technical explanation of how Mixture of Experts models work, covering routing mechanisms, expert selection, load balancing, and memory efficiency. Be thorough."

# Stream the response and count tokens via SSE
TMPFILE=$(mktemp)
START_TIME=$(python3 -c "import time; print(time.time())")

curl -s "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"stream\": true,
    \"max_tokens\": $MAX_TOKENS,
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ]
  }" | while IFS= read -r line; do
    if [[ "$line" == data:* ]] && [[ "$line" != "data: [DONE]" ]]; then
        echo "$line" >> "$TMPFILE"
    fi
  done

END_TIME=$(python3 -c "import time; print(time.time())")

# Count tokens from SSE chunks
TOKEN_COUNT=$(python3 -c "
import json, sys

lines = open('$TMPFILE').readlines()
tokens = 0
content = ''
for line in lines:
    line = line.strip()
    if line.startswith('data: ') and line != 'data: [DONE]':
        try:
            d = json.loads(line[6:])
            delta = d.get('choices', [{}])[0].get('delta', {})
            t = delta.get('content', '')
            if t:
                tokens += 1
                content += t
        except:
            pass

print(tokens)
" 2>/dev/null)

ELAPSED=$(python3 -c "print(round($END_TIME - $START_TIME, 2))")
TPS=$(python3 -c "
tokens = int('$TOKEN_COUNT')
elapsed = float('$ELAPSED')
warmup = $WARMUP_TOKENS
effective = max(tokens - warmup, 1)
tps = effective / max(elapsed, 0.001)
print(round(tps, 1))
")

rm -f "$TMPFILE"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Results"
echo "  Total tokens  : $TOKEN_COUNT"
echo "  Elapsed       : ${ELAPSED}s"
echo "  Throughput    : ${TPS} tok/s  (${WARMUP_TOKENS}-token warmup excluded)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Copy-paste line:"
echo "  | $MODEL | ${TOKEN_COUNT} tok | ${ELAPSED}s | **${TPS} tok/s** |"
echo ""
