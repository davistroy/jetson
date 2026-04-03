# Experiment Plan: Gemma 4 vs Qwen3.5-4B on Jetson Orin Nano Super

**Date:** 2026-04-03
**Author:** Troy Davis
**Status:** BLOCKED — see Prerequisites
**Hardware:** NVIDIA Jetson Orin Nano Super 8GB (Orin SoC, Ampere GA10B sm_87, 7.4 GB LPDDR5 unified)
**Current champion:** Qwen3.5-4B Q4_K_M — 2.6 GB model, 14.0 tok/s, 32K context, stable
**Reference:** LAB_NOTEBOOK.md entries 001-006, JETSON_CONFIG.md

---

## Motivation

Google released Gemma 4 on April 2, 2026 under Apache 2.0. The "E" (Effective) variants use Per-Layer Embeddings (PLE) — a technique where each decoder layer gets its own embedding table, maximizing parameter efficiency for on-device deployment. The benchmarks are compelling: the E4B (4.5B effective params) scores 69.4% on MMLU Pro (competitive with Qwen3.5-4B), but dramatically outperforms on reasoning (AIME 42.5% vs Gemma 3 27B's 20.8%) and coding (LiveCodeBench 52.0% vs 29.1%). It also has native function calling, configurable thinking mode (`<|think|>` tokens), and 128K context window.

The question: does Gemma 4 deliver enough quality improvement on the Jetson to justify any throughput or memory trade-offs versus our optimized Qwen3.5-4B setup?

## Candidate Models

| Model | Architecture | Total Params | Effective Params | Q4_K_M Size | Context | Jetson Fit |
|-------|-------------|-------------|-----------------|-------------|---------|-----------|
| **Qwen3.5-4B** (champion) | Dense | 3.5B | 3.5B | 2.6 GB | 32K | Excellent — 2.1 GB headroom |
| **Gemma 4 E2B** | Dense + PLE | 5.1B | 2.3B | 3.11 GB | 128K | Good — ~4.3 GB for KV+OS |
| **Gemma 4 E4B** | Dense + PLE | 8.0B | 4.5B | 4.98 GB | 128K | Marginal — ~2.4 GB for KV+OS |
| Gemma 4 26B-A4B | MoE | 25.2B | 3.8B active | 16.9 GB | 256K | **No** — all params must load |

**GGUF sources:** `unsloth/gemma-4-E2B-it-GGUF` and `unsloth/gemma-4-E4B-it-GGUF` on Hugging Face. Also available from bartowski, lmstudio-community, and ggml-org.

### Key Architectural Differences

| Feature | Qwen3.5-4B | Gemma 4 E2B | Gemma 4 E4B |
|---------|-----------|-------------|-------------|
| Vocab size | ~152K | 262K | 262K |
| Attention | Standard | Hybrid (local sliding window 512 + global) | Hybrid (local sliding window 512 + global) |
| Positional encoding | RoPE | Proportional RoPE (p-RoPE) | Proportional RoPE (p-RoPE) |
| Thinking/reasoning mode | No | Yes (`<\|think\|>` tokens) | Yes (`<\|think\|>` tokens) |
| Function calling | No native support | Native tool-use tokens | Native tool-use tokens |
| System prompt | Via template | Native `system` role | Native `system` role |
| Multimodal | Text only | Text + Image + Audio | Text + Image + Audio |
| Languages | 201 | 140+ | 140+ |

---

## Inference Engine Assessment: vLLM vs llama.cpp

### vLLM Verdict: Not Viable on This Device

vLLM was evaluated and **ruled out** for the Jetson Orin Nano 8GB. The reasons are architectural, not configuration-level:

| Issue | Detail |
|-------|--------|
| **Runtime overhead** | PyTorch + CUDA context + vLLM framework consumes 2-4 GB before any model loads. On 7.4 GB unified, this leaves 3.5-4.5 GB for everything else. |
| **Unified memory bugs** | vLLM's memory profiler assumes discrete VRAM. On unified memory it pre-allocates 90% of system RAM, starving the OS. Issues [vllm #13131](https://github.com/vllm-project/vllm/issues/13131) and [vllm #10267](https://github.com/vllm-project/vllm/issues/10267) are both closed "not planned." |
| **No successful 8GB deployments** | [jetson-containers #1568](https://github.com/dusty-nv/jetson-containers/issues/1568): even a 125M toy model crashed during KV cache allocation with `NVML_SUCCESS == r INTERNAL ASSERT FAILED`. |
| **NVIDIA's own guidance** | The [practical guide for Orin Nano Super 8GB](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412) recommends llama.cpp and TensorRT-Edge-LLM. Does not mention vLLM. |
| **No GGUF support** | vLLM uses safetensors/HF format (GPTQ, AWQ), not GGUF. Different quant ecosystem. |

vLLM's advantages (PagedAttention, continuous batching, high-concurrency serving) require memory headroom that doesn't exist on this device. It becomes relevant at AGX Orin 32/64 GB.

**All experiments in this plan use llama.cpp**, our proven inference engine on this hardware.

### Jetson-Specific Forum Context

Two NVIDIA forum threads document Gemma experiences on Orin Nano Super:

- **[Throttling thread](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-developer-kit-throttles-on-gemma-3-4b/353323):** Over-current throttling running Gemma 3 4B via Ollama at 25W, despite GPU <48C and CPU <50C. NVIDIA confirmed this is a normal protection mechanism that slightly lowers clock frequency — not a failure. Relevant because the larger E4B model (4.98 GB) means more memory bandwidth pressure and potentially higher sustained power draw.

- **[Error thread](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-error-running-gemma-3-4b-model/327944):** Docker/nvidia-container-toolkit issues with Ollama containers. Not relevant to our llama.cpp native setup, but confirms that container-based inference adds complexity on this platform.

---

## Prerequisites (Must Clear Before Experiments Begin)

### P1: llama-server Infinite Repetition Bug — BLOCKER

**GitHub issue:** [llama.cpp #21365](https://github.com/ggerganov/llama.cpp/issues/21365)

Gemma 4 models produce infinite repetition when served via `llama-server` but work correctly in `llama-cli`. Our Jetson runs llama-server via systemd. This is a **showstopper** for server deployment.

**Additional known bugs (as of 2026-04-03):**

| Issue | Problem | Impact on Us |
|-------|---------|-------------|
| [#21365](https://github.com/ggerganov/llama.cpp/issues/21365) | Infinite repetition in llama-server | **Blocker** — our deployment is llama-server |
| [#21329](https://github.com/ggerganov/llama.cpp/issues/21329) | `--parallel` crashes with Gemma 4 | Medium — we run single-slot |
| [#21375](https://github.com/ggerganov/llama.cpp/issues/21375) | Infinite loop in tool-call parser | Medium — affects function calling tests |
| [#21321](https://github.com/ggerganov/llama.cpp/issues/21321) | Generates `<unused24>` tokens | Low — may be fixed by tokenizer update |

**Action:** Monitor #21365 daily. Do not download models or rebuild llama.cpp until this is confirmed fixed.

### P2: Rebuild llama.cpp

Our Jetson runs build 8414 (commit `5744d7ec4`). Gemma 4 architecture support landed around build b8641 (April 2, 2026). Minimum viable build is b8641; ideally build from latest master after #21365 is resolved.

**Rebuild procedure:**
```bash
SSH="ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net"

# Save current binary
$SSH "cp ~/llm-server/llama.cpp/build/bin/llama-server ~/llm-server/llama-server.bak.8414"

# Pull and rebuild
$SSH "cd ~/llm-server/llama.cpp && git pull && \
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_F16=ON -DCMAKE_BUILD_TYPE=Release && \
  cmake --build build --config Release -j4"

# Verify build
$SSH "~/llm-server/llama.cpp/build/bin/llama-server --version"
```

### P3: Regression Test After Rebuild

Before any Gemma 4 work, confirm Qwen3.5-4B still performs identically on the new build:

```bash
# Restart with existing config
$SSH "kill \$(pgrep -f llama-server)"
# Wait for systemd restart, then:
$SSH "bash ~/bench.sh regression-check 8080"
```

**Pass criteria:** Generation tok/s within 5% of 14.0 baseline. Any significant deviation = investigate before proceeding.

### P4: Download Gemma 4 Models

```bash
# E2B (safe first experiment) — 3.11 GB
$SSH "cd ~/llm-server/models && \
  curl -L -o gemma-4-E2B-it-Q4_K_M.gguf \
  'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf'"

# E4B (tight fit, download only if E2B succeeds) — 4.98 GB
$SSH "cd ~/llm-server/models && \
  curl -L -o gemma-4-E4B-it-Q4_K_M.gguf \
  'https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf'"
```

---

## Experiment 7: Gemma 4 E2B vs Qwen3.5-4B (Safe Comparison)

**Rationale:** The E2B at 3.11 GB leaves ~4.3 GB headroom — comfortable and low-risk. This establishes whether the Gemma 4 architecture runs well on the Jetson before attempting the tighter E4B fit.

### 7a: Startup and Memory Baseline

**Method:** Start llama-server with Gemma 4 E2B Q4_K_M, same flags as current Qwen3.5 config (32K context, flash attention ON, full GPU offload). Record startup behavior and idle memory.

```bash
# Create a startup script for E2B
$SSH "cat > ~/llm-server/start-gemma4-e2b-server.sh << 'SCRIPT'
#!/bin/bash
exec ~/llm-server/llama.cpp/build/bin/llama-server \
  --model ~/llm-server/models/gemma-4-E2B-it-Q4_K_M.gguf \
  --alias gemma-4-e2b \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --flash-attn \
  --threads 4 \
  --host 0.0.0.0 \
  --port 8080
SCRIPT
chmod +x ~/llm-server/start-gemma4-e2b-server.sh"
```

**Metrics to capture:**
- Startup time (CUDA init to "listening")
- GPU layers loaded vs total layers
- RSS at idle (after startup, before any requests)
- Free RAM at idle
- Any NvMap warnings in journal

**Comparison point:** Qwen3.5-4B idles at ~5.0 GB RSS with ~2.6 GB free.

### 7b: Throughput Benchmark

**Method:** Run `bench.sh` (3 test types × 3 runs each), identical methodology to all prior experiments.

**Metrics:**
| Metric | Qwen3.5-4B Baseline | Gemma 4 E2B |
|--------|---------------------|-------------|
| Generation tok/s | 14.0 | ? |
| Prompt tok/s (short, post-warmup) | ~97 | ? |
| Prompt tok/s (medium/long) | ~211 | ? |
| CUDA warmup requests | 2-3 | ? |

**Hypothesis:** E2B may show lower throughput despite smaller effective params (2.3B vs 3.5B) because the total parameter count is 5.1B — all those PLE embedding tables are still loaded and accessed. The 262K vocab (vs Qwen's ~152K) also means larger embedding lookups. Alternatively, the hybrid attention (local sliding window) could be faster for short sequences.

### 7c: Context Size Sweep

**Method:** Same as LAB_NOTEBOOK Entry 003. Test at 2K, 4K, 8K, 16K, 32K context sizes. Record throughput and memory at each level.

**Key question:** Does the hybrid attention architecture (sliding window 512 + global) change the context-throughput relationship? Qwen3.5-4B showed perfectly flat throughput across all context sizes — Gemma 4's architecture may behave differently.

### 7d: Quality Comparison — General Tasks

**Method:** Same 3 deterministic prompts from LAB_NOTEBOOK Entry 005, temperature=0:

1. **Factual accuracy:** "What are the three laws of thermodynamics? Explain each one briefly."
2. **Code generation:** "Write a Python function to find the longest palindromic substring in a given string. Include edge cases."
3. **Structured output:** "Compare and contrast TCP and UDP protocols. Cover: connection type, reliability, speed, use cases, and header size."

**Evaluation:** Side-by-side comparison of outputs. Score on accuracy, completeness, and coherence.

### 7e: Quality Comparison — Reasoning (Gemma 4 Differentiator)

**Method:** New prompts targeting Gemma 4's claimed reasoning strength. Run each with thinking mode OFF, then ON (if supported by llama-server chat template).

**Prompts:**
1. **Math reasoning:** "A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B, 300 miles away, at 10:00 AM traveling toward Station A at 80 mph. At what time do they meet, and how far from Station A?"
2. **Logic puzzle:** "Five houses in a row are painted different colors. The English person lives in the red house. The Spanish person owns a dog. Coffee is drunk in the green house. The Ukrainian drinks tea. The green house is immediately to the right of the ivory house. Who owns the zebra? (Provide reasoning steps.)"
3. **Multi-step code reasoning:** "Given a binary tree, write a function to determine if it's height-balanced. What is the time complexity of your solution, and can you do better than the naive approach?"

**Evaluation:** Correctness of final answer, quality of reasoning chain, and whether thinking mode produces better results.

### 7f: Thermal and Power Profile

**Method:** Sustained 5-minute generation (use gen_stress.py to create a long prompt, then generate a long response). Monitor with tegrastats throughout.

**Metrics:**
| Metric | Qwen3.5-4B Baseline | Gemma 4 E2B |
|--------|---------------------|-------------|
| GPU temp (sustained) | 65-68°C | ? |
| Power draw (sustained) | ~18.3W | ? |
| Any throttling events | None | ? |
| CPU temp (sustained) | ~50°C | ? |

**Relevance:** The NVIDIA forum thread showed over-current throttling with Gemma 3 4B under Ollama. We need to verify that the Gemma 4 architecture doesn't trigger similar behavior under llama.cpp, even if the model is smaller.

### 7g: Maximum Context Fill Stress Test

**Method:** Same as LAB_NOTEBOOK Entry 006. Fill to near-maximum context, verify stability.

**Pass criteria:** No OOM, no swap activation, no NvMap crashes, correct needle-in-haystack retrieval.

---

## Experiment 8: Gemma 4 E4B vs Qwen3.5-4B (Tight Fit)

**Gate:** Only proceed if Experiment 7 shows the Gemma 4 architecture runs cleanly on the Jetson (no crashes, no anomalous thermal behavior, no llama.cpp bugs).

### 8a: Memory Feasibility — Find Maximum Context

**Method:** Start with 2K context, run a single inference, record RSS and free RAM. Increment context: 2K → 4K → 8K → 12K → 16K → 24K → 32K. Stop if:
- Free RAM drops below 500 MB
- SSD swap activates (any non-zero usage on `/ssd/16GB.swap`)
- NvMapMemAllocInternalTagged errors appear in journal
- Server crashes

```bash
# Start E4B at 2K context
$SSH "~/llm-server/llama.cpp/build/bin/llama-server \
  --model ~/llm-server/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --alias gemma-4-e4b \
  --ctx-size 2048 \
  --n-gpu-layers 999 \
  --flash-attn \
  --threads 4 \
  --host 0.0.0.0 \
  --port 8080"
```

**Expected outcome:** At 4.98 GB model weight, the server will idle at ~6+ GB RSS. Context may be limited to 2K-4K before hitting memory ceiling. If the model can't even start at 2K context, it's a non-starter.

**Comparison point:** Qwen3.5-4B Q4_K_M idles at ~5.0 GB RSS at 32K context. The E4B will use nearly double the model weight.

### 8b: Throughput at Maximum Viable Context

**Method:** Run `bench.sh` at whatever context ceiling 8a establishes.

**Key question:** Is the E4B faster or slower than Qwen3.5-4B per-token? The larger model means more data to move through the memory bus (the Jetson is bandwidth-bound per Entry 005), but PLE may change the access pattern.

### 8c: Quality Comparison

**Method:** Same prompts as 7d + 7e. Temperature=0.

**Hypothesis:** E4B should show meaningfully better quality than both Qwen3.5-4B and E2B, especially on reasoning tasks. The question is whether the quality gap justifies the context and throughput trade-off.

### 8d: Stress Test

**Method:** Fill to max viable context (from 8a), sustain generation for 5 minutes, monitor for instability.

**Criteria:** This device has shown rock-solid stability at 5.5 GB RSS (Qwen at 32K). The E4B will push RSS to 6+ GB, leaving <1.5 GB free. We need to verify no OOM kills, no swap storms, no CUDA memory fragmentation under sustained load.

### 8e: Quantization Exploration (If 8a Fails)

If E4B Q4_K_M doesn't fit, try smaller quants:

| Quant | File Size | Quality Trade-off |
|-------|----------|-------------------|
| Q3_K_M | 4.06 GB | Noticeable quality drop in some tasks |
| IQ4_XS | ~4.3 GB | Importance-weighted 4-bit, may retain quality better |
| Q3_K_S | ~3.7 GB | Significant quality drop |

Only pursue this if the E4B at Q4_K_M is within ~500 MB of fitting — if the gap is larger, the quality degradation from lower quants negates the point of testing a "better" model.

---

## Experiment 9: Thinking Mode Evaluation

**Gate:** Only proceed if Experiment 7 or 8 shows a Gemma 4 variant running stably on the Jetson.

Gemma 4's configurable `<|think|>` reasoning mode is a differentiator vs Qwen3.5-4B. This experiment quantifies its cost and benefit.

### 9a: Thinking ON vs OFF

**Method:** Run reasoning prompts from 7e with:
- Thinking disabled (default)
- Thinking enabled (via chat template or system prompt)

**Metrics:**

| Metric | Thinking OFF | Thinking ON |
|--------|-------------|-------------|
| Answer correctness | ? | ? |
| Total tokens generated | ? | ? (expect 2-5x more) |
| Wall-clock time to answer | ? | ? |
| Quality of reasoning chain | N/A | ? |

### 9b: Thinking Token Overhead

**Method:** For each reasoning prompt, count tokens spent on thinking vs final answer.

**Key question:** On a 14 tok/s device (or whatever Gemma 4 achieves), the thinking tokens add real latency. If thinking mode produces 200 thinking tokens + 100 answer tokens at 12 tok/s, that's 25 seconds — is the quality improvement worth 3x the response time?

---

## Decision Framework

After all experiments, score each viable configuration:

| Criterion | Weight | Qwen3.5-4B Q4_K_M (Baseline) |
|-----------|--------|-------------------------------|
| Generation throughput (tok/s) | **High** | 14.0 |
| Maximum usable context (tokens) | **High** | 32,768 |
| Memory headroom at operating context | **Medium** | ~2.1 GB free |
| General quality (factual, code, structured) | **Medium** | Strong (Entry 005 reference) |
| Reasoning/math quality | **Medium** | Baseline |
| Thinking mode capability | **Low** | Not available |
| Function calling support | **Low** | Not native |
| Stability (no crashes, no swap) | **High** | Rock solid (Entry 006) |
| Thermal behavior | **Low** | 65-68°C, no throttling |

**The bar is high.** Qwen3.5-4B at 14 tok/s with 32K context and 2+ GB headroom is a strong, well-characterized baseline established through 6 rigorous experiments. A Gemma 4 variant needs to deliver a meaningful, measurable improvement on at least one high-weight criterion without regressing on any other high-weight criterion.

### Possible Outcomes

1. **Gemma 4 E2B replaces Qwen3.5-4B as default** — if it matches or beats throughput, fits comfortably, and shows quality improvement (especially reasoning).

2. **Gemma 4 E4B replaces Qwen3.5-4B as default** — only if it fits with viable context (>8K), throughput is acceptable (>10 tok/s), and quality is substantially better.

3. **Gemma 4 added as alternate mode** — if quality is better in specific scenarios (reasoning, function calling) but throughput/context trade-offs make it unsuitable as default. Add a `gemma4` mode to the mode-switching system alongside `qwen35`.

4. **Qwen3.5-4B remains default** — if Gemma 4 doesn't clear the bar on any dimension, or if llama.cpp bugs make it unreliable.

---

## Execution Sequence

```
P1: Monitor llama.cpp #21365          ← CURRENT (check daily)
    │
    ▼ (bug fixed)
P2: Rebuild llama.cpp from master
P3: Regression test Qwen3.5-4B
P4: Download Gemma 4 E2B Q4_K_M
    │
    ▼
Experiment 7: E2B full evaluation     ← ~2 hours of Jetson time
    │
    ├── (E2B promising) ──► P4: Download E4B Q4_K_M
    │                           │
    │                           ▼
    │                       Experiment 8: E4B evaluation  ← ~2 hours
    │                           │
    │                           ▼
    │                       Experiment 9: Thinking mode   ← ~1 hour
    │
    └── (E2B not promising) ──► Still run Experiment 8 if E4B fits
                                    │
                                    ▼
                                Decision: keep/switch/add-mode
```

**Estimated total Jetson time:** 3-5 hours of active experimentation, spread across sessions.

---

## Appendix: Model Download Checksums

To be recorded after download. Verify GGUF integrity before any testing:
```bash
$SSH "sha256sum ~/llm-server/models/gemma-4-E2B-it-Q4_K_M.gguf"
$SSH "sha256sum ~/llm-server/models/gemma-4-E4B-it-Q4_K_M.gguf"
```

## Appendix: Sources

- [Google Gemma 4 model card](https://huggingface.co/google/gemma-4-E4B-it) — architecture, benchmarks, context sizes
- [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) — GGUF quantizations and file sizes
- [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) — GGUF quantizations and file sizes
- [llama.cpp #21365](https://github.com/ggerganov/llama.cpp/issues/21365) — llama-server infinite repetition bug
- [llama.cpp #21329](https://github.com/ggerganov/llama.cpp/issues/21329) — parallel processing crash
- [llama.cpp #21375](https://github.com/ggerganov/llama.cpp/issues/21375) — tool-call parser infinite loop
- [vllm #13131](https://github.com/vllm-project/vllm/issues/13131) — unified memory calculation instability (closed, not planned)
- [vllm #10267](https://github.com/vllm-project/vllm/issues/10267) — unified memory support request (closed, not planned)
- [jetson-containers #1568](https://github.com/dusty-nv/jetson-containers/issues/1568) — vLLM KV cache crash on Orin Nano 8GB
- [NVIDIA Forum: Orin Nano throttling on Gemma 3](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-developer-kit-throttles-on-gemma-3-4b/353323)
- [NVIDIA Forum: Orin Nano Gemma 3 errors](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-error-running-gemma-3-4b-model/327944)
- [NVIDIA Forum: AI models for Orin Nano Super 8GB practical guide](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412)
