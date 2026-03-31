#!/bin/bash
# Qwen3.5-4B on Jetson Orin Nano Super (8GB)
# Serves OpenAI-compatible /v1/chat/completions on port 8080
#
# Model: Qwen3.5-4B-Q4_K_M (~2.6GB GGUF, 32 layers)
# Supports: thinking mode on/off, tool calling, 201 languages
# Expected: ~14 tokens/sec generation (full GPU offload)
# Reasoning/thinking: DISABLED via --reasoning off
#
# Optimized 2026-03-30: Switched from Q5_K_M to Q4_K_M (+12% throughput,
# -400MB RAM, no quality loss). Context bumped from 6144 to 32768 (full
# model context, zero throughput penalty — see LAB_NOTEBOOK.md).
#
# Build: llama.cpp commit 5744d7ec4 (build 8414)
# cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
#   -DGGML_CUDA_F16=ON -DCMAKE_BUILD_TYPE=Release
# IMPORTANT: VMM must be ON (default) for Jetson unified memory

MODEL="$HOME/llm-server/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"
LLAMA_SERVER="$HOME/llm-server/llama.cpp/build/bin/llama-server"
PORT=8080

# Evict filesystem cache to maximize free physical pages
# Jetson unified memory: cudaMalloc needs actual free pages
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

# Check free physical memory (in MB)
FREE_MB=$(awk '/MemFree/ {print int($2/1024)}' /proc/meminfo)

if [ "$FREE_MB" -gt 4000 ]; then
    GPU_LAYERS=999
    echo "Free memory: ${FREE_MB}MB - using full GPU offload (999 layers)"
else
    GPU_LAYERS=0
    echo "Free memory: ${FREE_MB}MB - insufficient for GPU offload, using CPU-only"
fi

exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --alias qwen3.5-4b \
    --ctx-size 32768 \
    --n-gpu-layers "$GPU_LAYERS" \
    --threads 4 \
    --flash-attn on \
    --reasoning off \
    --log-disable
