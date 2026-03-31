#!/bin/bash
# bench.sh — Standardized llama.cpp throughput benchmark
# Usage: bench.sh [label] [port]
# Runs 3 test types x 3 runs each, outputs CSV results

LABEL="${1:-test}"
PORT="${2:-8080}"
URL="http://localhost:$PORT/v1/chat/completions"

echo "=== Benchmark: $LABEL ($(date -Iseconds)) ==="
echo ""

# Collect tegrastats during benchmark
tegrastats --interval 1000 > /tmp/bench_tegra.txt 2>/dev/null &
TPID=$!

SHORT='{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Write a haiku about the ocean"}],"max_tokens":64}'
MEDIUM='{"model":"qwen3.5-4b","messages":[{"role":"system","content":"You are a helpful assistant. Provide detailed, thorough answers."},{"role":"user","content":"Explain how a transistor works and why it is important in modern computing. Include the physics behind semiconductor junctions."}],"max_tokens":256}'
LONG='{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Write a detailed essay about the history of artificial intelligence from its origins to the present day."}],"max_tokens":512}'

EXTRACT_PY=$(cat << 'PYEOF'
import sys, json
try:
    d = json.load(sys.stdin)
    t = d.get("timings", {})
    pn = t.get("prompt_n", "")
    gn = t.get("predicted_n", "")
    pps = t.get("prompt_per_second", 0)
    gps = t.get("predicted_per_second", 0)
    pms = t.get("prompt_ms", 0)
    gms = t.get("predicted_ms", 0)
    print("{},{},{:.1f},{:.1f},{:.0f},{:.0f}".format(pn, gn, pps, gps, pms, gms))
except Exception as e:
    print("ERR,ERR,ERR,ERR,ERR,ERR")
PYEOF
)

echo "test,run,prompt_n,predicted_n,prompt_tok_s,gen_tok_s,prompt_ms,predicted_ms"

for TEST in short medium long; do
    case $TEST in
        short)  DATA="$SHORT" ;;
        medium) DATA="$MEDIUM" ;;
        long)   DATA="$LONG" ;;
    esac
    for RUN in 1 2 3; do
        ROW=$(curl -s "$URL" -H "Content-Type: application/json" -d "$DATA" 2>/dev/null | python3 -c "$EXTRACT_PY")
        echo "$TEST,$RUN,$ROW"
    done
done

# Stop tegrastats
kill $TPID 2>/dev/null
wait $TPID 2>/dev/null

echo ""
echo "=== Memory State ==="
free -m
echo ""
echo "=== Server Process ==="
PID=$(pgrep -f llama-server | head -1)
if [ -n "$PID" ]; then
    grep -E "^(VmRSS|VmSwap)" /proc/$PID/status
fi
echo ""
echo "=== Tegrastats (last reading) ==="
tail -1 /tmp/bench_tegra.txt 2>/dev/null
echo ""
echo "=== Swap Detail ==="
swapon --show 2>/dev/null
