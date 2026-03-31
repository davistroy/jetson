#!/bin/bash
# set-ctx.sh — Restart llama-server with a specific context size
# Usage: set-ctx.sh <ctx_size> [flash_attn_on|flash_attn_off] [model_file]
# Waits for server to be ready before returning.

CTX="${1:?Usage: set-ctx.sh <ctx_size> [flash_attn_on|flash_attn_off] [model_file]}"
FLASH="${2:-flash_attn_on}"
MODEL_FILE="${3:-Qwen_Qwen3.5-4B-Q5_K_M.gguf}"

SERVER_DIR="/home/claude/llm-server"
SERVER_BIN="$SERVER_DIR/llama.cpp/build/bin/llama-server"
MODEL="$SERVER_DIR/models/$MODEL_FILE"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

# Determine flash attention flag
FA_FLAG="--flash-attn on"
if [ "$FLASH" = "flash_attn_off" ]; then
    FA_FLAG=""
fi

# Determine alias from model filename
ALIAS="qwen3.5-4b"

echo "=== Setting ctx=$CTX flash=$FLASH model=$MODEL_FILE ==="
echo ""

# Stop current server
echo "Stopping server..."
kill $(pgrep -f llama-server) 2>/dev/null || true
sleep 3

# Write a temporary override startup script
cat > "$SERVER_DIR/start-experiment.sh" << EXPEOF
#!/bin/bash
# Temporary experiment startup — will be replaced by restore

MODEL="$MODEL"
LLAMA_SERVER="$SERVER_BIN"
PORT=8080

# Memory eviction
python3 -c "
try:
    d = bytearray(5 * 1024**3)
    for i in range(0, len(d), 4096):
        d[i] = 1
    del d
except MemoryError:
    try:
        d = bytearray(3 * 1024**3)
        for i in range(0, len(d), 4096):
            d[i] = 1
        del d
    except MemoryError:
        pass
" 2>/dev/null

FREE_MB=\$(grep MemFree /proc/meminfo | tr -s ' ' | cut -d' ' -f2)
FREE_MB=\$((FREE_MB / 1024))

if [ "\$FREE_MB" -gt 4000 ]; then
    GPU_LAYERS=999
    echo "Free memory: \${FREE_MB}MB - using full GPU offload (999 layers)"
else
    GPU_LAYERS=0
    echo "Free memory: \${FREE_MB}MB - insufficient for GPU offload, using CPU-only"
fi

exec "\$LLAMA_SERVER" \\
    --model "\$MODEL" \\
    --host 0.0.0.0 \\
    --port "\$PORT" \\
    --alias $ALIAS \\
    --ctx-size $CTX \\
    --n-gpu-layers "\$GPU_LAYERS" \\
    --threads 4 \\
    $FA_FLAG \\
    --reasoning off \\
    --log-disable
EXPEOF
chmod +x "$SERVER_DIR/start-experiment.sh"

# Update mode to use experiment script
echo "experiment" > "$SERVER_DIR/mode.txt"

# Add experiment mode to start-server.sh dispatcher if not present
if ! grep -q 'experiment)' "$SERVER_DIR/start-server.sh"; then
    # Insert before the default case
    sed -i '/\*)/i\    experiment)\n        echo "Mode: Experiment (custom config on port 8080)"\n        exec ~/llm-server/start-experiment.sh\n        ;;' "$SERVER_DIR/start-server.sh"
fi

# Restart via systemd
echo "Starting server with ctx=$CTX..."
sudo systemctl restart myscript

# Wait for server to be ready (up to 60 seconds)
echo -n "Waiting for server"
for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/v1/models > /dev/null 2>&1; then
        echo " ready! (${i}s)"

        # Verify it has GPU
        LOGS=$(sudo journalctl -u myscript --no-pager -n 5 2>/dev/null)
        if echo "$LOGS" | grep -q "ggml_cuda_init: found"; then
            echo "GPU: OK (CUDA initialized)"
        elif echo "$LOGS" | grep -q "no usable GPU"; then
            echo "WARNING: Running CPU-only!"
        fi

        # Show memory state
        echo ""
        free -m
        echo ""
        PID=$(pgrep -f llama-server | head -1)
        if [ -n "$PID" ]; then
            grep -E "^(VmRSS|VmSwap)" /proc/$PID/status
        fi
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " TIMEOUT — server did not start within 60s"
echo "Check: sudo journalctl -u myscript --no-pager -n 20"
exit 1
