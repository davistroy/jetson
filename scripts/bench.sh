#!/bin/bash
# Benchmark script for llama.cpp on Jetson
# Runs 3 test types x 3 runs each, extracts timing from API response
# Usage: bench.sh [label]

LABEL="${1:-baseline}"
PORT=8080
URL="http://localhost:${PORT}/v1/chat/completions"

echo "=== Benchmark: $LABEL ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ==="
echo ""

run_test() {
    local test_name="$1"
    local prompt="$2"
    local max_tokens="$3"
    local run="$4"

    local result
    result=$(curl -s "$URL" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"temperature\":0}")

    local prompt_tokens gen_tokens pp_speed gen_speed
    prompt_tokens=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null || echo "ERR")
    gen_tokens=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "ERR")

    pp_speed=$(echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
if 'prompt_per_second' in t:
    print(f\"{t['prompt_per_second']:.1f}\")
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    gen_speed=$(echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
if 'predicted_per_second' in t:
    print(f\"{t['predicted_per_second']:.1f}\")
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

    echo "$test_name | run$run | prompt_tok=$prompt_tokens | gen_tok=$gen_tokens | pp=$pp_speed tok/s | gen=$gen_speed tok/s"
}

# Warmup request
echo "Warmup..."
curl -s "$URL" -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0}' > /dev/null
echo ""

echo "--- Short tests (small prompt, ~20 token output) ---"
for i in 1 2 3; do
    run_test "short" "What causes tides on Earth? Answer in 2 sentences." 32 "$i"
done
echo ""

echo "--- Medium tests (medium prompt, 256 token output) ---"
for i in 1 2 3; do
    run_test "medium" "Write a detailed explanation of how HTTP works, covering request methods, status codes, headers, and the request-response cycle." 256 "$i"
done
echo ""

echo "--- Long tests (medium prompt, 512 token output) ---"
for i in 1 2 3; do
    run_test "long" "Write a comprehensive guide to database indexing, covering B-trees, hash indexes, composite indexes, partial indexes, and when to use each type." 512 "$i"
done
echo ""

# Memory snapshot
echo "--- Memory ---"
free -h | head -3
echo ""
echo "--- Server process ---"
ps aux | grep llama-server | grep -v grep | awk '{print "RSS: " $6/1024 "MB, VSZ: " $5/1024 "MB"}'
echo ""
echo "=== End benchmark: $LABEL ==="
