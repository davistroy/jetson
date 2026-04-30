#!/bin/bash
# Qwen3.5-4B on Jetson Orin Nano Super (8GB)
# Serves OpenAI-compatible /v1/chat/completions on port 8080
#
# Model: Qwen3.5-4B-Q4_K_M (~2.6GB GGUF, 32 layers)
# Supports: thinking mode on/off, tool calling, 201 languages
# Expected: ~14.5 tokens/sec generation (full GPU offload)
# Reasoning/thinking: DISABLED via --reasoning off
#
# Optimized 2026-03-30: Switched from Q5_K_M to Q4_K_M (+12% throughput,
# -400MB RAM, no quality loss). Context bumped from 6144 to 32768 (full
# model context, zero throughput penalty — see LAB_NOTEBOOK.md).
#
# Optimized 2026-04-12: Rebuilt llama.cpp b8414 → b8766 (+17% generation).
# Added KV cache quantization (q8_0), --mlock, -t 1, --parallel 1.
# See LAB_NOTEBOOK.md Entry 012.
#
# Build: llama.cpp b8766 (commit 547765a93)
# cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87
#   -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON
#   -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
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
    --threads 1 \
    --parallel 1 \
    --flash-attn on \
    --reasoning off \
    --mlock \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --log-disable
