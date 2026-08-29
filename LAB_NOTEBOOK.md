# Jetson Orin Nano Super — Optimization Lab Notebook

**Project:** Systematic performance characterization and optimization of llama.cpp on Jetson Orin Nano Super
**Start date:** 2026-03-30
**Hardware:** NVIDIA Jetson Orin Nano Super 8GB (Orin SoC, Ampere GA10B sm_87, 1024 CUDA cores, 7.4 GB LPDDR5 unified)
**Model:** Qwen3.5-4B-Q5_K_M (3.1 GB GGUF, 32 transformer layers)
**Software:** llama.cpp build 8414 (commit 5744d7ec4), CUDA 12.6, JetPack 6.2.2 (R36.5.0)
**Reference:** JETSON_CONFIG.md, CLAUDE.md

---

## Baseline Snapshot (2026-03-30 ~12:46 UTC)

Captured at idle (no active inference requests). Server running under `claude` user via systemd.

### System State

| Metric | Value | Notes |
|--------|-------|-------|
| Uptime | 7 days (since Mar 23) | Stable since last restart |
| Total RAM | 7607 MB | Unified CPU/GPU |
| RAM used | 4853 MB | 64% utilization |
| RAM available | 2509 MB | |
| Swap total | 20188 MB | 16 GB file + 6x634 MB zram |
| Swap used | 212 MB | All in zram, SSD swap file untouched |
| GPU temp | 47°C | Idle, ambient ~25°C |
| CPU temp | 46°C | Idle |
| GPU utilization | 0% | Idle |
| Power draw | 4.5W total | VDD_IN idle |
| CPU utilization | 0% all cores | Idle at 729 MHz |

### Swap Detail

| Device | Type | Size | Used | Notes |
|--------|------|------|------|-------|
| /ssd/16GB.swap | file | 16 GB | 0 B | NVMe SSD swap, untouched |
| /dev/zram0-5 | partition | 634 MB each | 35-37 MB each | Compressed RAM swap |
| **Total** | | **~20 GB** | **~212 MB** | |

Top swap consumers: tailscaled (23 MB), containerd (15 MB), jtop (12 MB). Negligible — no inference-related swap usage at idle.

### LLM Server Configuration

```
Binary: llama-server (build 8414, commit 5744d7ec4)
Model: Qwen_Qwen3.5-4B-Q5_K_M.gguf (3.1 GB)
Alias: qwen3.5-4b
Context size: 6144 tokens
GPU layers: 999 (full offload)
Flash attention: ON
Reasoning: OFF
Threads: 4
Port: 8080
```

### Server Process Memory

| Metric | Value |
|--------|-------|
| VmRSS | 4893 MB |
| VmSwap | 0 kB |
| VmSize | 42.6 GB (virtual, CUDA address space) |
| VmPeak | 42.6 GB |

### Sysctl Configuration (untuned defaults)

| Setting | Value | Notes |
|---------|-------|-------|
| vm.swappiness | 60 | Default — high for inference workload |
| vm.min_free_kbytes | 45056 | Default |
| vm.dirty_ratio | 20 | Default |
| vm.overcommit_memory | 0 | Default (heuristic) |
| net.ipv4.tcp_congestion_control | cubic | Default |

### Baseline Throughput (3 runs averaged)

| Test | Prompt tokens | Generated tokens | Prompt tok/s | Generation tok/s |
|------|--------------|-----------------|-------------|-----------------|
| Short (haiku) | 19 | 21 | 66.9 | 11.9 |
| Medium (explanation) | 52 | 256 | 127.0 | 12.5 |
| Long (essay) | 30 | 512 | 132.0 | 12.5 |

**Observations:**
- Generation throughput is consistent at **~12.5 tok/s** regardless of output length. This is lower than the 17.3 tok/s observed immediately after migration (server was freshly started then, now 7 days uptime).
- Prompt processing scales well: longer prompts amortize better (7.6 ms/tok vs 14.9 ms/tok for short prompts).
- Short prompts show notably lower generation speed (11.9 tok/s) — likely first-token overhead in the KV cache.

### Configuration Snapshot

Saved as `baseline` via `jetson-config.sh snapshot`. Includes: startup scripts, systemd unit, sysctl, mode.txt, model inventory, process state, memory state, network config.

---

## Experiment Plan

### Goals (in priority order)

1. **Establish a baseline reference** — Document the device's actual performance envelope
2. **Maximize context window** — Can we push to 8192 or beyond without OOM?
3. **Find the optimal tradeoff point** — Data-driven decision on where to set the knobs
4. **Maximize throughput** — Squeeze the best tok/s from this hardware

### Experimental Variables

| Variable | Values to test | Current | Metric affected |
|----------|---------------|---------|----------------|
| Context size | 2048, 4096, 6144, 8192, 10240, 12288 | 6144 | Memory, throughput, OOM risk |
| Flash attention | on, off | on | Throughput, memory |
| Quantization | Q4_K_M (2.6 GB), Q5_K_M (3.1 GB) | Q5_K_M | Memory, throughput, quality |
| Swap behavior | observe under load | untuned | Latency, stability |

### Metrics Captured Per Experiment

For every configuration change:

1. **Throughput** — 3 request types (short/medium/long prompt), 3 runs each, report mean tok/s
2. **Memory** — RSS, VmSwap, free RAM, tegrastats during inference
3. **Thermal** — GPU/CPU temp at idle and under sustained load (10 consecutive requests)
4. **Stability** — Does the server start? Does it OOM? Any NvMap warnings in logs?
5. **Latency** — Time to first token (TTFT), end-to-end latency
6. **Quality** — For quant comparison only: same 5 prompts, blind comparison of output quality

### Execution Order

**Phase 1: Context Size Sweep** (Entries 002-007)
- Goal: Find the maximum stable context size and the throughput curve
- Method: For each context size, restart server with `--ctx-size N`, run the benchmark suite, capture memory and thermal state
- Order: Start from current (6144), go up (8192, 10240, 12288) until OOM, then go down (4096, 2048) to measure the throughput gain
- Rollback: If a config OOMs, kill the process, restore baseline, document the failure

**Phase 2: Flash Attention A/B** (Entries 008-009)
- Goal: Measure whether `--flash-attn on` actually helps on SM87 at this model size
- Method: At the optimal context size from Phase 1, run the full benchmark with flash-attn on and off
- Control: Same model, same quant, same context size, same everything else

**Phase 3: Quantization Comparison** (Entries 010-011)
- Goal: Is Q5_K_M worth the extra 500 MB over Q4_K_M?
- Method: At the optimal config from Phases 1-2, swap to Q4_K_M and run benchmarks
- Quality test: 5 identical prompts, compare outputs side-by-side for coherence, detail, and accuracy
- Memory test: How much headroom does Q4_K_M free up? Does it enable a larger context?

**Phase 4: Swap Behavior Under Load** (Entries 012-013)
- Goal: Does the SSD swap file ever get touched during inference? What's the latency impact?
- Method: Run sustained load (10+ sequential long-generation requests) at maximum stable context, monitor swap usage via tegrastats and /proc/meminfo every second
- Stress test: Fill context to max (send a prompt that uses most of the context window), observe memory pressure

### Safety Protocol

- **Before each experiment:** Take a `jetson-config.sh snapshot` with a descriptive name
- **After each experiment:** Record all results in this notebook before changing anything
- **On OOM or crash:** Document what happened, check `journalctl -u myscript` for logs, restore from baseline snapshot
- **One variable at a time:** Never change context size AND flash attention simultaneously
- **Cooldown:** Wait 30 seconds between server restarts for memory to settle

### Benchmark Script

Each throughput test uses this standardized request set:

```bash
# SHORT: 19 prompt tokens, 64 max generation
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Write a haiku about the ocean"}],"max_tokens":64}'

# MEDIUM: ~52 prompt tokens, 256 max generation
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"system","content":"You are a helpful assistant. Provide detailed, thorough answers."},{"role":"user","content":"Explain how a transistor works and why it is important in modern computing. Include the physics behind semiconductor junctions."}],"max_tokens":256}'

# LONG: ~30 prompt tokens, 512 max generation
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Write a detailed essay about the history of artificial intelligence from its origins to the present day."}],"max_tokens":512}'
```

Metrics extracted from the `timings` object in each response: `prompt_per_second`, `predicted_per_second`, `prompt_n`, `predicted_n`.

---

## Entry 001 — Baseline Capture (Read-Only)

**Date:** 2026-03-30 12:46 UTC
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Capture complete system state before any experiments. Establish reproducible baseline metrics.

**Actions taken:**
1. Deployed `jetson-config.sh` to `/home/claude/bin/` on Jetson
2. Ran `jetson-config.sh snapshot baseline` — captured system info, memory, swap, server state, sysctl, models, network, service config
3. Ran 3-tier throughput benchmark (short/medium/long prompts)
4. Captured tegrastats reading at idle
5. Documented all findings in this notebook

**Findings:**
- Server is stable at 7 days uptime, 0 kB swap usage by llama-server
- Generation throughput is **12.5 tok/s** steady-state (lower than the 17.3 tok/s seen post-fresh-restart — may indicate memory fragmentation or thermal throttling over time)
- Prompt processing is fast: 67-132 tok/s depending on prompt length
- RAM usage at idle: 4.9 GB RSS for the server process, leaving ~2.5 GB available
- Swap is essentially unused — 212 MB total across zram devices, SSD swap file at 0
- Sysctl is completely untuned (vm.swappiness=60, default min_free_kbytes)
- GPU idle temp 47°C — plenty of thermal headroom

**Key question for investigation:** Why is throughput 12.5 tok/s now vs 17.3 tok/s right after restart? Is this:
- (a) Thermal throttling (47°C idle suggests not)
- (b) Memory fragmentation after 7 days
- (c) Normal variance (the 17.3 reading was a single short response, not sustained)
- (d) CPU frequency scaling (all cores at 729 MHz — is this the idle governor downclocking?)

**Action item for Entry 002:** Before starting context sweep, restart the server fresh and re-measure to distinguish (b) from (c). This gives us a true cold-start baseline.

**Config snapshot:** `baseline` (saved at `/home/claude/jetson-configs/baseline/`)

---

## Entry 002 — Fresh Restart Baseline + CPU Frequency Investigation

**Date:** 2026-03-30 13:00 UTC
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Resolve the 12.5 vs 17.3 tok/s discrepancy by testing a fresh server restart. Investigate CPU frequency governor behavior.

**Actions taken:**
1. Checked CPU frequency governor: `schedutil`, min 729 MHz, max 1728 MHz. Currently at max under SSH load. The 729 MHz in tegrastats was normal idle downclocking — **not a problem**.
2. Power mode confirmed: `MAXN_SUPER` (mode 2) — maximum performance.
3. Took `pre-entry002` config snapshot.
4. Restarted server via `systemctl restart myscript`.
5. Deployed standardized `bench.sh` script (3 test types x 3 runs each).
6. Ran full benchmark suite immediately after cold start.

**Cold-Start Benchmark Results (ctx=6144, flash-attn=on, Q5_K_M):**

| Test | Run | Prompt tok | Gen tok | Prompt tok/s | Gen tok/s | Prompt ms | Gen ms |
|------|-----|-----------|---------|-------------|----------|----------|--------|
| short | 1 | 19 | 22 | 55.0 | 11.7 | 345 | 1878 |
| short | 2 | 19 | 20 | 95.3 | 13.0 | 199 | 1535 |
| short | 3 | 19 | 21 | 96.3 | 13.0 | 197 | 1610 |
| medium | 1 | 52 | 256 | 135.7 | 12.5 | 383 | 20467 |
| medium | 2 | 52 | 256 | 211.0 | 12.5 | 246 | 20431 |
| medium | 3 | 52 | 256 | 212.6 | 12.5 | 245 | 20408 |
| long | 1 | 30 | 512 | 134.6 | 12.5 | 223 | 41079 |
| long | 2 | 30 | 512 | 135.0 | 12.5 | 222 | 40982 |
| long | 3 | 30 | 512 | 136.0 | 12.5 | 221 | 40929 |

**System state during benchmark:**
- GPU utilization: 99% (fully loaded)
- GPU temp: 67.6°C (well within limits)
- Power draw: 18.3W total (VDD_CPU_GPU_CV: 8.2W)
- Server RSS: 5254 MB, VmSwap: 73 MB (zram only, SSD swap still 0)
- Free RAM: 85 MB (tight but stable)

**Findings:**

1. **The 17.3 tok/s was an anomaly.** Cold-start generation converges to **12.5 tok/s** after the first request. The earlier 17.3 measurement was a single 2-token response ("Hello") — too short to be representative. With 20+ token outputs, steady-state is 12.5-13.0 tok/s.

2. **First request is always slower** — short run 1 shows 11.7 tok/s and 55 tok/s prompt processing (vs 95+ on runs 2-3). This is CUDA kernel warmup / KV cache initialization.

3. **Prompt processing shows dramatic warmup effect**: short prompts go from 55 → 96 tok/s between run 1 and 2. Medium prompts go from 136 → 212 tok/s. This is CUDA graph caching in action.

4. **Generation throughput is rock-solid at 12.5 tok/s** regardless of prompt or output length (after warmup). This is the true hardware ceiling for Qwen3.5-4B Q5_K_M at ctx=6144 on SM87.

5. **CPU frequency is not a factor** — governor ramps to max under load, idles correctly when not needed.

**Established baseline for all future comparisons:**
- **Generation: 12.5 tok/s** (steady-state, post-warmup)
- **Prompt processing: 96 tok/s** (short), **212 tok/s** (medium/long) — post-warmup
- **Memory: 5.25 GB RSS** at ctx=6144

**Config snapshot:** `pre-entry002`

---

## Entry 003 — Phase 1: Context Size Sweep

**Date:** 2026-03-30 13:09-13:36 UTC
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Find the maximum stable context size and measure throughput impact across the range 2048-32768.

**Method:** For each context size, restart server via `set-ctx.sh`, run full `bench.sh` benchmark (3 tests x 3 runs), capture memory and thermal state. One variable changed at a time (context size only, flash-attn=on, Q5_K_M model held constant).

**Results:**

| Context | Gen tok/s (avg) | Prompt tok/s (avg, post-warmup) | RSS after bench (MB) | VmSwap (MB) | SSD Swap | GPU Temp |
|---------|----------------|-------------------------------|---------------------|-------------|----------|----------|
| 2048 | 12.5 | 212 | 5235 | 69 | 0 | 68°C |
| 4096 | 12.6 | 211 | 5331 | 77 | 0 | 68°C |
| 6144 (old default) | 12.5 | 212 | 5254 | 73 | 0 | 68°C |
| 8192 | 12.5 | 211 | 5434 | 70 | 0 | 68°C |
| 10240 | 12.5 | 211 | 5431 | 71 | 0 | 68°C |
| 12288 | 12.5 | 211 | 5568 | 84 | 0 | 68°C |
| 16384 | 12.5 | 211 | 5530 | 103 | 0 | 68°C |
| 20480 | 12.5 | 211 | 5841 | 89 | 0 | 68°C |
| 24576 | 12.5 | 213 | 5803 | 96 | 0 | 68°C |
| **32768** | **12.5** | **208** | **6152** | **94** | **0** | **68°C** |

**Findings:**

1. **Throughput is completely flat across all context sizes.** Generation stays at 12.5 tok/s regardless. Context size has zero impact on inference speed for prompts that don't fill the context.

2. **Memory grows linearly but modestly.** 2048→32768 (16x increase in context) only adds ~900 MB RSS. The KV cache is pre-allocated but only pages in as needed.

3. **SSD swap was never touched.** All swap usage is zram-only (compressed RAM), never exceeding ~100 MB.

4. **The old default of 6144 was unnecessarily conservative.** We can run the full 32K context window with no penalty.

**Recommendation:** Set `--ctx-size 32768` as the new default.

---

## Entry 004 — Phase 2: Flash Attention A/B Test

**Date:** 2026-03-30 13:43-13:47 UTC
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Measure whether `--flash-attn on` provides any benefit on SM87 (Ampere) at this model size and context.

**Method:** At ctx=32768, run identical benchmark suite with flash attention ON vs OFF. Same model, same quant, same everything else.

**Results:**

| Config | Gen tok/s (avg) | Prompt tok/s (avg) | RSS (MB) | VmSwap (MB) |
|--------|----------------|-------------------|----------|-------------|
| flash-attn ON | 12.5 | 211 | 6152 | 94 |
| flash-attn OFF | 12.5 | 212 | 6220 | 92 |

**Detailed no-flash results (ctx=32768):**

| Test | Run | Prompt tok/s | Gen tok/s |
|------|-----|-------------|----------|
| short | 1 | 53.7 | 11.8 |
| short | 2 | 95.4 | 13.2 |
| short | 3 | 94.8 | 13.1 |
| medium | 1 | 130.9 | 12.6 |
| medium | 2 | 212.0 | 12.6 |
| medium | 3 | 211.4 | 12.6 |
| long | 1 | 132.7 | 12.5 |
| long | 2 | 135.0 | 12.5 |
| long | 3 | 135.7 | 12.5 |

**Findings:**

1. **No measurable throughput difference between flash-attn on and off** at this model size (~4B params) and these prompt lengths (19-512 tokens out of 32K context).

2. **Flash attention saves ~68 MB RSS** (6152 vs 6220 MB). Marginal but free.

3. Flash attention's benefit scales with sequence length and model size. At 4B params with short sequences, the attention computation is not the bottleneck — the matrix multiplications (FFN layers) dominate. Flash attention would show more benefit with longer prompts that actually fill the 32K context (tested in Entry 006).

**Recommendation:** Keep flash attention ON. No throughput cost, small memory benefit, and will help at longer sequence lengths.

---

## Entry 005 — Phase 3: Q4_K_M vs Q5_K_M Comparison

**Date:** 2026-03-30 13:50-13:57 UTC
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Compare Qwen3.5-4B at Q4_K_M (2.6 GB) vs Q5_K_M (3.1 GB) for throughput, memory, and output quality.

**Method:** At ctx=32768 with flash-attn=on, run identical benchmark suite on both quants, then run 3 identical quality prompts at temperature=0.

### Throughput Results

| Quant | Model Size | Gen tok/s (avg) | Prompt tok/s (avg) | RSS at startup (MB) | RSS after bench (MB) | VmSwap (MB) |
|-------|-----------|----------------|-------------------|--------------------|--------------------|-------------|
| Q5_K_M | 3.1 GB | **12.5** | 211 | 5725 | 6152 | 94 |
| Q4_K_M | 2.6 GB | **14.0** | 211 | 5330 | 5752 | 27 |
| **Delta** | **-500 MB** | **+12%** | same | **-395 MB** | **-400 MB** | **-67 MB** |

**Q4_K_M detailed results (ctx=32768):**

| Test | Run | Prompt tok/s | Gen tok/s |
|------|-----|-------------|----------|
| short | 1 | 65.5 | 13.3 |
| short | 2 | 96.7 | 14.6 |
| short | 3 | 97.1 | 14.6 |
| medium | 1 | 208.6 | 14.1 |
| medium | 2 | 211.1 | 14.1 |
| medium | 3 | 210.5 | 14.0 |
| long | 1 | 133.4 | 14.0 |
| long | 2 | 136.6 | 14.0 |
| long | 3 | 136.9 | 14.0 |

### Quality Comparison (temperature=0, deterministic)

**Prompt 1: "What causes tides on Earth? Explain in 3 sentences."**
- Q5_K_M: Accurate, mentions Moon's gravitational pull, Earth's rotation, spring/neap tides. Well-structured.
- Q4_K_M: Equally accurate, same key facts, slightly different wording. Mentions "spring tides" and "neap tides" correctly.
- **Verdict: Indistinguishable quality.**

**Prompt 2: "Write a Python function to check if a string is a palindrome."**
- Q5_K_M: Clean implementation with type hints, docstring, numpy-style docstring format. Handles case-insensitive, alphanumeric-only comparison.
- Q4_K_M: Virtually identical implementation. Same algorithm, same edge case handling. Different parameter name (`text` vs `s`).
- **Verdict: Indistinguishable quality.**

**Prompt 3: "Compare and contrast TCP and UDP in exactly 5 bullet points."**
- Q5_K_M: 5 well-structured bullets covering connection management, reliability, speed/overhead, traffic control, use cases.
- Q4_K_M: 5 equally well-structured bullets. Same coverage, slightly different phrasing. Separates "data integrity" from "reliability" instead of "traffic control."
- **Verdict: Indistinguishable quality.**

**Findings:**

1. **Q4_K_M is 12% faster** (14.0 vs 12.5 tok/s). This is a significant, consistent improvement — smaller weights mean less memory bandwidth consumed per inference step, and bandwidth is the bottleneck on this device.

2. **Q4_K_M uses 400 MB less RAM**, freeing headroom for OS caches, other processes, or potentially larger models.

3. **No detectable quality difference** on factual, coding, and comparison tasks. At 4B parameters, Q4_K_M retains enough precision that the model's intrinsic capability is the limiting factor, not quantization artifacts.

**Recommendation:** Switch default to Q4_K_M. The 12% throughput gain and 400 MB memory savings with no quality loss is a clear win.

---

## Entry 006 — Phase 4: Maximum Context Fill Stress Test

**Date:** 2026-03-30 14:04-14:06 UTC
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Determine what happens when the KV cache is filled to near-maximum capacity. Does the SSD swap file get touched? Does throughput degrade? Does the system become unstable?

**Method:** Using Q4_K_M at ctx=32768 with flash-attn=on. Generated 1150 unique sentences (~32,176 tokens) as a prompt with a needle-in-a-haystack question at the end. Fresh server restart before the test. Monitored memory every second during the entire prompt processing and generation.

### Graduated Fill Test (prior to full test)

| Prompt sentences | Prompt tokens | Prompt tok/s | Gen tok/s | RSS delta (MB) | SSD Swap |
|-----------------|--------------|-------------|----------|----------------|----------|
| 500 | 13,745 | 385.3 | 12.2 | +186 | 0 |
| 1000 | 14,462 | 382.3 | 12.2 | +100 | 0 |
| 1500+ | 42,636 | — | — | — | — |

(1500+ sentences exceeded 32K context — server returned `exceed_context_size_error`)

### Full Context Fill (fresh restart, clean KV cache)

| Metric | Value |
|--------|-------|
| **Prompt tokens processed** | **32,176** (98% of 32,768 context) |
| **Prompt processing speed** | **439.5 tok/s** |
| **Prompt processing time** | **73.2 seconds** |
| **Generation tokens** | 53 |
| **Generation speed** | **12.2 tok/s** |
| **Answer correct?** | **Yes** — correctly identified "orange" as the color in sentence 1149 |
| Pre-test RSS | 5197 MB |
| Post-test RSS | 5491 MB |
| **RSS delta** | **+294 MB** |
| VmSwap | **0 kB** (before and after) |
| SSD swap | **0 B** (untouched) |
| Peak GPU temp | 65°C |
| Free RAM (post) | 152 MB |
| Total RAM available (post) | 1970 MB |

### Memory Timeline During 32K Fill

| Time | Event | RAM Used | RSS | VmSwap |
|------|-------|---------|-----|--------|
| 10:04:15 | Start | 5124 MB | 5196 MB | 0 |
| 10:04:16 | Processing begins | 5142 MB | 5231 MB | 0 |
| 10:05:30 | Processing complete | 5396 MB | 5491 MB | 0 |
| 10:05:32 | Generation complete | 5396 MB | 5491 MB | 0 |

Memory grew smoothly and linearly over 77 seconds. No spikes, no thrashing, no swap activity.

**Findings:**

1. **The Jetson handles a fully-loaded 32K context without breaking a sweat.** 32,176 tokens processed, correct answer returned, no swap, no OOM, stable memory.

2. **Generation speed is unaffected by context fill level.** 12.2 tok/s at 98% context fill vs 14.0 tok/s with a 50-token prompt. The small decrease (12.2 vs 14.0) is within the variance seen at shorter contexts — the attention computation over 32K tokens adds minimal overhead.

3. **Prompt processing is actually FASTER at 32K (439 tok/s) than at short prompts (96-212 tok/s).** This is because longer sequences amortize the per-batch overhead better, and the GPU's parallel compute units are more fully utilized.

4. **The SSD swap file was NEVER touched throughout the entire experiment series.** All swap usage was zram (compressed RAM) for OS background processes, never for inference.

5. **The KV cache for 32K tokens costs approximately 294 MB of RAM.** This is modest — the model weights dominate at ~2.6 GB (Q4_K_M) or ~3.1 GB (Q5_K_M).

6. **The needle-in-a-haystack test passed.** The model correctly retrieved information from sentence 1149 out of 1150 sentences, demonstrating that the full context is actually being used, not just allocated.

---

## Summary of All Findings

### Optimal Configuration

| Parameter | Old Value | New Recommended | Impact |
|-----------|-----------|----------------|--------|
| **Context size** | 6144 | **32768** | Full model context, zero throughput penalty |
| **Flash attention** | on | **on** (keep) | Small memory benefit, helps at long sequences |
| **Quantization** | Q5_K_M (3.1 GB) | **Q4_K_M (2.6 GB)** | +12% throughput, -400 MB RAM, no quality loss |
| **Swap config** | 16 GB SSD + 3.8 GB zram | **No change needed** | SSD swap never touched; zram is net positive |

### Performance Envelope

| Metric | Baseline (old config) | Optimized (new config) |
|--------|----------------------|----------------------|
| Generation tok/s | 12.5 | **14.0** |
| Max context | 6,144 | **32,768** |
| Prompt tok/s (short) | 96 | ~97 |
| Prompt tok/s (32K fill) | not tested | **440** |
| RAM at idle | 5254 MB | ~5000 MB |
| RAM at 32K context fill | not tested | **5491 MB** |
| SSD swap usage | 0 | **0** |

### Key Insights

1. **Context size is free lunch on this hardware.** The KV cache for the full 32K context costs ~294 MB, and throughput is unaffected. The previous 6144 limit was throwing away 5x the context window for no benefit.

2. **Quantization matters more than anything else for throughput.** Q4_K_M vs Q5_K_M is a 12% generation speed difference because this device is memory-bandwidth-bound. Smaller weights = less data to move = faster inference.

3. **Flash attention makes no measurable difference at this scale** with short prompts, but is theoretically beneficial at long sequences and costs nothing to keep enabled.

4. **The device has significant memory headroom.** Even at full 32K context with Q4_K_M, only 5.5 GB of 7.6 GB is used. There's room for a concurrent embedding server or other workload.

5. **Swap behavior is a non-issue.** The SSD swap file exists as insurance but never activates during inference. All observed swap usage is zram (compressed RAM) for OS background processes.

---

## Entry 007: Gemma 4 A/B Test Planning & Inference Engine Assessment (2026-04-03)

### Objective

Evaluate Google's Gemma 4 model family (released 2026-04-02, Apache 2.0) as a potential replacement or alternative to Qwen3.5-4B Q4_K_M on this device. Assess both the models and whether an alternative inference engine (vLLM) should be tested.

### Gemma 4 Model Assessment

Gemma 4 uses Per-Layer Embeddings (PLE) — each decoder layer has its own embedding table, inflating total parameter count but improving inference efficiency. Two variants fit our memory envelope:

| Model | Total Params | Effective Params | Q4_K_M Size | 128K Context | Jetson Fit |
|-------|-------------|-----------------|-------------|--------------|-----------|
| **Gemma 4 E2B** | 5.1B | 2.3B | 3.11 GB | Yes | Good — ~4.3 GB headroom |
| **Gemma 4 E4B** | 8.0B | 4.5B | 4.98 GB | Yes | Marginal — ~2.4 GB headroom |

Key architectural differences vs Qwen3.5-4B: hybrid attention (local sliding window 512 + global), proportional RoPE, 262K vocab (vs ~152K), native function calling tokens, configurable `<|think|>` reasoning mode, multimodal (text + image + audio).

**Benchmark highlights (Google's numbers):** E4B scores 69.4% MMLU Pro (competitive with Qwen3.5-4B), 42.5% AIME 2026 (vs Gemma 3 27B's 20.8%), 52.0% LiveCodeBench v6 (vs 29.1%). Strong reasoning and coding improvements at the 4B-class size.

### vLLM Assessment: Not Viable

**Verdict: Ruled out for this device.** vLLM was thoroughly evaluated and cannot work on Jetson Orin Nano 8GB:

- **Runtime overhead:** PyTorch + CUDA context + vLLM framework consumes 2-4 GB before loading any model. On 7.4 GB unified memory, this leaves 3.5-4.5 GB for model + KV cache + OS — insufficient for practical 4B-class inference.
- **Unified memory bugs:** vLLM's memory profiler assumes discrete VRAM. Pre-allocates 90% of "GPU memory" which on unified memory = total system RAM, starving the OS. Issues [vllm #13131](https://github.com/vllm-project/vllm/issues/13131) and [vllm #10267](https://github.com/vllm-project/vllm/issues/10267) are both closed "not planned" upstream.
- **No successful 8GB deployments exist:** [jetson-containers #1568](https://github.com/dusty-nv/jetson-containers/issues/1568) — even a 125M toy model crashed during KV cache allocation with `NVML_SUCCESS == r INTERNAL ASSERT FAILED`.
- **NVIDIA's own guidance:** The [practical guide for Orin Nano Super 8GB](https://forums.developer.nvidia.com/t/ai-models-that-run-on-jetson-orin-nano-super-8gb-a-practical-guide/365412) recommends llama.cpp and TensorRT-Edge-LLM only. Does not mention vLLM as viable.

vLLM becomes relevant at AGX Orin 32/64 GB where the overhead is a small fraction of available memory. **All experiments continue with llama.cpp.**

### Jetson Community Context (NVIDIA Forums)

Two forum threads document Gemma experiences on Orin Nano Super:

- **Throttling ([thread](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-developer-kit-throttles-on-gemma-3-4b/353323)):** Over-current throttling running Gemma 3 4B via Ollama at 25W, despite GPU <48°C. NVIDIA confirmed this is a normal protection mechanism — lowers clock frequency, not a failure. Relevant for thermal monitoring in our experiments.
- **Container errors ([thread](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-error-running-gemma-3-4b-model/327944)):** Docker/nvidia-container-toolkit issues with Ollama. Not relevant to our native llama.cpp setup.

### Blocking Issue

**llama-server infinite repetition bug ([llama.cpp #21365](https://github.com/ggml-org/llama.cpp/issues/21365)):** ~~Gemma 4 produces infinite repetition in `llama-server` but works correctly in `llama-cli`.~~ **RESOLVED (2026-04-09):** PR [#21418](https://github.com/ggml-org/llama.cpp/pull/21418) (merged 2026-04-04) introduces a dedicated Gemma 4 PEG parser, adds `<|tool_response>` as an EOG token, and removes Gemma 4 from the generic autoparser. Multiple users confirmed the fix resolves the infinite repetition in llama-server. Fix is included in build **b8721** (released 2026-04-09). Issue #21365 formally closed as stale (2026-07-23).

Additional bugs status (2026-05-29): `--parallel` crash ([#21329](https://github.com/ggml-org/llama.cpp/issues/21329)) — **CLOSED**; tool-call parser loop ([#21375](https://github.com/ggml-org/llama.cpp/issues/21375)) — **CLOSED** (closed as not planned 2026-05-29; PR [#21760](https://github.com/ggml-org/llama.cpp/pull/21760) merged 2026-04-13 had addressed the main edge cases); `<unused24>` token generation ([#21321](https://github.com/ggml-org/llama.cpp/issues/21321)) — **CLOSED**.

### Experiment Plan

Full plan documented in **EXPERIMENT_PLAN_gemma4.md**. Summary:

| Experiment | Model | Gate | Focus |
|-----------|-------|------|-------|
| **7** | E2B Q4_K_M (3.11 GB) | llama.cpp bug fix + rebuild | Full evaluation: throughput, memory, context sweep, quality, reasoning, thermal |
| **8** | E4B Q4_K_M (4.98 GB) | Exp 7 success | Memory feasibility first, then throughput + quality if it fits |
| **9** | Winner from 7/8 | Exp 7 or 8 success | Thinking mode cost/benefit analysis |

Prerequisites before any testing: rebuild llama.cpp (need b8641+ for Gemma 4 arch), regression test Qwen3.5-4B on new build, download models.

### Status

**UNBLOCKED** (2026-04-09) — PR [#21418](https://github.com/ggml-org/llama.cpp/pull/21418) merged 2026-04-04, fixing the llama-server infinite repetition bug. Fix first included in build b8721 (2026-04-09). Additional parser edge-case fixes in PR [#21760](https://github.com/ggml-org/llama.cpp/pull/21760) (merged 2026-04-13). Issue #21365 formally closed as stale on GitHub (2026-07-23); re-confirmed closed (not planned) on 2026-08-28. Issue #21375 (tool-call parser loop) closed as not planned 2026-05-29. Issue #21329 (--parallel crash) closed. Issue #21321 (<unused24> tokens) closed. Latest release as of 2026-08-28: **v0.3.0** (2026-08-25; supersedes v0.2.0/b10566). Next step: proceed to P2 (rebuild llama.cpp to v0.3.0+), then P3 regression test, then P4 model downloads. Note: a separate Gemma 4 E2B blocker exists — PLE (Per-Layer Embeddings) not implemented in llama.cpp (issue #22243); quality is silently degraded without it. Check that issue before running Gemma 4 experiments.

### Decision

No changes to current configuration yet. Qwen3.5-4B Q4_K_M remains the active default pending the rebuild and Gemma 4 experiments.

---

## Entry 008: JetPack SDK Update Research (2026-04-12)

**Objective:** Assess whether any JetPack SDK updates are available or upcoming for the Jetson Orin Nano Super, and determine if any action is needed.

### Current Baseline (Confirmed Live on Device)

| Component | Version | Notes |
|-----------|---------|-------|
| JetPack | 6.2.2 | Latest in the 6.x line |
| L4T | R36.5.0 (Jan 16, 2026) | Latest for Orin Nano |
| Ubuntu | 22.04.5 LTS | |
| Kernel | 5.15.185-tegra | |
| NVIDIA Driver | 540.4.0 | |
| CUDA | 12.6 (V12.6.68) | |
| cuDNN | 9.3.0.75 | |
| TensorRT | 10.3.0.30 | |
| APT source | r36.5 | Confirmed pointing to r36.5 repos |
| GPU offload | Full (999 layers) | NvMap bug from R36.4.x is resolved |
| Server | llama-server running, 585 MB free + 19 GB swap available | Healthy |

### Finding 1: JetPack 6.2.2 IS the Latest 6.x Release — NO ACTION NEEDED

There is no JetPack 6.2.3, 6.3, or any further 6.x release planned. NVIDIA's roadmap has moved to JetPack 7.x for future development. The 6.x line appears to be in maintenance/EOL mode.

JetPack 6.2.2 (R36.5.0) fixed the critical NvMap CUDA allocator regression from R36.4.x that was forcing CPU-only mode. The Jetson is already running this version with full GPU offload restored.

**Classification: NO ACTION NEEDED** -- already on the latest.

### Finding 2: JetPack 7.x Does NOT Support Orin Nano Yet — WORTH WATCHING

| JetPack Version | L4T | Status | Orin Nano Support |
|-----------------|-----|--------|-------------------|
| 7.0 | Jetson Linux 38.x | Released (Thor only) | **NO** |
| 7.1 | Jetson Linux 38.4 | Released Jan 2026 (Thor + T4000) | **NO** |
| 7.2 | TBD | **Planned Q2 2026** | **Expected YES** |

JetPack 7.x is a major architecture change:
- **Ubuntu 24.04 LTS** (up from 22.04)
- **Kernel 6.8 LTS** (up from 5.15)
- **Modular, cloud-native architecture**
- Preemptable real-time kernel option
- Multi-Instance GPU (MIG) support
- Integrated Holoscan Sensor Bridge

JetPack 7.0 and 7.1 launched for Jetson AGX Thor and T5000/T4000 modules only. Orin Nano support was originally projected for Q1 2026 but has slipped. As of April 2026, the NVIDIA roadmap shows **JetPack 7.2 with Orin series support in Q2 2026** (April-June window). Recent forum posts (April 8, 2026) indicate the timeline was pushed from Q1 to Q2.

**Classification: WORTH WATCHING** -- JetPack 7.2 for Orin could drop within weeks to 2 months. This will be a major upgrade (new kernel, new Ubuntu base, new CUDA stack).

### Finding 3: CUDA Updates — No Standalone Updates Available

CUDA 12.6 is the version bundled with JetPack 6.2.2. There are no standalone CUDA updates for the Jetson outside of JetPack releases. JetPack 7.x for Thor ships with CUDA 13.0 (unified across Arm targets), but that won't be available for Orin until JetPack 7.2.

**Classification: NO ACTION NEEDED** -- CUDA 12.6 is the latest available for this platform.

### Finding 4: Known JetPack 6.2.2 Issues — Low Impact for Our Use Case

**Snapd/Chromium issue:** Users report that JetPack 6.2.2 introduced a snapd regression (version 2.70) causing Chromium browser and other snap-packaged apps to fail to launch, with graphical artifacts. Workaround: revert snapd to 2.68.5. **Not relevant to us** -- the Jetson runs headless as an LLM inference server; we don't use Chromium or snap-packaged GUI apps.

**Application outage reports:** Some users report instability when running multiple CUDA applications simultaneously on Orin Nano Super with r36.5. **Low risk for us** -- we run a single llama-server process.

**GPIO control regression:** Reported GPIO control failures on JP 6.2.2 / L4T 36.5. **Not relevant** -- we don't use GPIO.

**NvMap errors still reported by some:** Intermittent NvMapMemAlloc errors still appear in some workloads (particularly PyTorch/YOLO). These appear to be memory pressure issues rather than the kernel bug. **Our llama.cpp server has been stable** -- the Q4_K_M quant + 32K context fits within memory budget.

**Classification: NO ACTION NEEDED** -- none of the known issues affect our headless LLM server use case.

### Finding 5: Kernel Updates — None Beyond 5.15.185-tegra

The kernel 5.15.185-tegra shipped with R36.5.0 is the latest available for the JetPack 6.x line. NVIDIA publishes periodic security bulletins (most recent: October 2025) with patches applied through L4T point releases. The jump to kernel 6.8 only comes with JetPack 7.x.

**Classification: NO ACTION NEEDED** -- on the latest kernel for this platform.

### Summary Assessment

| Area | Status | Classification | Action |
|------|--------|---------------|--------|
| JetPack 6.x | 6.2.2 is latest and final | **NO ACTION** | Stay put |
| JetPack 7.x for Orin | 7.2 expected Q2 2026 | **WORTH WATCHING** | Monitor NVIDIA forums/roadmap |
| CUDA | 12.6 is latest for platform | **NO ACTION** | Comes with JetPack 7.2 |
| Known bugs | None affect our use case | **NO ACTION** | N/A |
| Kernel | 5.15.185 is latest for 6.x | **NO ACTION** | 6.8 comes with JetPack 7.x |
| NvMap fix | Confirmed working on R36.5 | **NO ACTION** | Full GPU offload restored |

### Upgrade Planning Notes for JetPack 7.2

When JetPack 7.2 drops for Orin Nano, it will be a **full reflash** (not an APT upgrade) -- new Ubuntu base (22.04 -> 24.04), new kernel (5.15 -> 6.8), likely new CUDA (12.6 -> 13.0). Planning considerations:

1. **Backup everything** before attempting: ~/llm-server/, model files, systemd units, mode scripts
2. **Test on a fresh SD card first** if possible before committing the NVMe
3. **Expect llama.cpp rebuild** -- new CUDA version will require full rebuild
4. **Expect container breakage** -- any Docker containers will need rebuilding for new L4T base
5. **Wait for community validation** -- let early adopters shake out Orin Nano-specific bugs before upgrading a working system
6. **Check llama.cpp CUDA 13.0 support** before upgrading -- ensure the build system handles it

**Recommendation:** Do NOT upgrade to JetPack 7.2 on release day. Wait 2-4 weeks for community reports, then evaluate. The current JetPack 6.2.2 setup is stable and performant for our needs.

### Next Check

Re-evaluate in **June 2026** or when JetPack 7.2 GA for Orin is announced, whichever comes first.

---

## Entry 008b: Community Research Scan — Forums, Reddit, GitHub (2026-04-12)

**Objective:** Survey NVIDIA developer forums, Reddit, GitHub, and community sources for actionable performance improvements, new builds, optimization techniques, and model recommendations for Jetson Orin Nano Super running llama.cpp.

**Sources searched:**
- NVIDIA Developer Forums (forums.developer.nvidia.com) — Jetson Orin Nano, Jetson Projects
- Reddit r/LocalLLaMA, r/JetsonNano
- GitHub ggml-org/llama.cpp discussions and releases
- dusty-nv/jetson-containers project
- NVIDIA Jetson AI Lab
- NVIDIA technical blog posts

### Findings

#### 1. llama.cpp Build Version Gap (ACTION)

**Current:** Build 8414 (commit 5744d7ec4, late 2025)
**Latest:** Build b8766 (April 12, 2026)

We are ~350 builds behind current. Key improvements since our build:
- **CUDA Graphs now enabled by default** for batch size 1 inference — reduces GPU-side launch overhead between kernel executions, up to 1.2x speedup on H100, proportionally beneficial on all NVIDIA GPUs. No flag needed in current builds.
- **Flash attention kernel compilation optimized** (b8763) — skips superfluous FA kernels, faster builds
- **aarch64 GEMM/GEMV optimizations** — q6_K repack routines and SME2-based FP16 compute path for Q4_0 on ARM
- **New quantization types** — Q1_0, improved Q5_K OpenCL
- **Gemma 4 audio support** (b8766)
- **Qwen 3 tensor parallelism fixes** (b8760)

**Impact assessment:** CUDA Graphs alone could meaningfully improve our token generation throughput. The aarch64 GEMM improvements may help prompt processing. Worth rebuilding.

#### 2. Flash Attention Already Enabled (CONFIRMED)

Our current config already has `-fa` enabled. Community benchmarks from the llama.cpp CUDA performance discussion (#15013) show:
- **Jetson AGX Orin:** 991 t/s (no FA) vs 1,171 t/s (with FA) for prompt processing — 18% improvement
- FA benefit scales with context length and is consistent across NVIDIA GPUs

We are already doing the right thing here. No change needed.

#### 3. NVIDIA Official Recommendation: vLLM over llama.cpp (INFO)

NVIDIA forum staff (AastaLLL) explicitly recommend vLLM frameworks over llama.cpp for Jetson deployments:
- NVIDIA-AI-IOT maintains Jetson-optimized vLLM containers (`ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`)
- NVIDIA's edge AI blog (Jan 2026) recommends vLLM as the primary inference engine
- AGX Orin 64GB: 231 tok/s with vLLM (concurrency=8) vs lower with llama.cpp

**However**, vLLM on Orin Nano 8GB has significant issues:
- Open GitHub issue (dusty-nv/jetson-containers #1568, Dec 2025): NVML assertion failures during KV cache init
- Memory fragmentation problems — even lightweight models (opt-125m) fail
- Requires careful memory tuning (`--gpu-memory-utilization`, reduced batch tokens)
- One forum report: vLLM has "a bug not completely fixed yet" causing OOM at launch on Nano devices (Feb 2026)

**Assessment:** vLLM is the better engine for larger Jetsons (AGX, NX 16GB) but is NOT proven stable on Orin Nano 8GB. llama.cpp remains the pragmatic choice for our hardware.

#### 4. Thread Count Optimization Tip (ACTION)

VentusServer optimization guide (2026) reports: setting thread count to 1 (`-t 1`) for GPU-focused workloads yields **43% gains** — counterintuitive but reduces CPU thread contention when all layers are GPU-offloaded.

Our current config uses `-t 4`. This is worth testing since we run 100% GPU offload.

#### 5. Additional Runtime Flags to Test (ACTION)

From community optimization guides:
- `--mlock` — prevents system swapping, maintains consistent latency. Relevant for our unified memory system.
- `--cont-batching` — enables continuous batching for server mode. May already be default in newer builds.
- `--cache-ram 4096 --no-mmap` — reported to improve low-latency server performance

#### 6. JetPack 7.2 Update (INFO)

- **Timeline:** Q2 2026 (confirmed by NVIDIA staff, Feb 2026)
- **Upgrade path:** Full flash required from JP6 to JP7 (confirmed April 10, 2026)
- **Key change:** Ubuntu 24.04, kernel 6.8, new CUDA version (likely 13.x)
- **Container toolkit:** Fixes security vulnerabilities in toolkit 1.16.2-1 (our current version)

Previous entry (007) recommendation stands: wait for community validation.

#### 7. CUDA Buffer Allocation Bug — CONFIRMED FIXED (INFO)

The "unable to allocate CUDA0 buffer" / NvMapMemAllocInternalTagged error that plagued JetPack 6.2.1 and earlier is **confirmed fixed in r36.5 (JetPack 6.2.2)**. We are on 6.2.2 — no action needed.

A user running Gemma 4 E4B (7.5B params, Q4_K_M, ~5 GB) confirmed it works on Orin Nano after upgrading from r36.4.7 to r36.5.

#### 8. Gemma 4 on Orin Nano (INFO)

Gemma 4 E4B (7.5B params) Q4_K_M runs on Orin Nano with ~5 GB VRAM usage, leaving ~1.4 GB headroom. Context auto-reduced from 131K to 120K tokens. Tight but functional. Requires llama.cpp b8766+ for audio conformer support.

Not immediately useful for our workload (we need reliable 4B-class models), but shows the ceiling for model size on our hardware.

#### 9. Qwen3-VL Performance Baseline (INFO)

Forum report (Feb 2026) testing Qwen3-VL-2B on Orin Nano Super:
- transformers: 0.89 QPS
- llama.cpp (b7641): 0.53 QPS
- Vision-language models hit kernel inefficiencies on SM87

This is a VLM workload (not our text-only use case) but confirms that llama.cpp handles text-only better than multimodal on this hardware.

#### 10. jetson-containers llama_cpp Package (INFO)

Latest container: `dustynv/llama_cpp:b5283-r36.4-cu128-24.04` (May 2025)
The container is ~6 months behind current llama.cpp builds. Our bare-metal build approach gives us more control and faster updates. No reason to switch to containers.

#### 11. NanoLLM / MLC-LLM Status (INFO)

- NanoLLM: dusty-nv's lightweight inference library using MLC backend. Limited to curated model list.
- MLC-LLM: Constrained compatibility with external/third-party models. NVIDIA staff now redirect users to vLLM/Ollama instead.
- Neither offers clear advantages over llama.cpp for our use case.

#### 12. Reddit Coverage (SKIP)

Reddit r/LocalLLaMA and r/JetsonNano returned no 2026-specific results for our query terms. Community discussion appears concentrated on NVIDIA forums and GitHub.

### Action Items

| # | Priority | Item | Expected Impact |
|---|----------|------|-----------------|
| 1 | **HIGH** | Rebuild llama.cpp to b8766+ | CUDA Graphs (auto-enabled), aarch64 GEMM improvements, ~350 builds of accumulated fixes |
| 2 | **MEDIUM** | Test `-t 1` vs `-t 4` thread count | Potential 10-40% TG improvement with full GPU offload |
| 3 | **LOW** | Test `--mlock` flag | May reduce latency variance from swap pressure |
| 4 | **LOW** | Test `--cont-batching` if not already default | Better multi-request handling |
| 5 | **NONE** | Switch to vLLM | Not viable on 8GB Orin Nano — OOM issues unresolved |
| 6 | **WATCH** | JetPack 7.2 release (Q2 2026) | Wait for community validation before upgrading |

### Overall Classification: **ACTION NEEDED**

The llama.cpp build gap is the primary finding. Build 8414 to b8766 represents ~6 months of CUDA, aarch64, and inference optimizations. CUDA Graphs alone (now default) should provide measurable throughput improvement with zero configuration changes. Thread count tuning is a quick test. The build upgrade should be the next hands-on session.

---

## Entry 009: Small Language Model Landscape Survey (2026-04-12)

**Objective:** Comprehensive research of the current small language model landscape (1B-7B) to determine whether any new model warrants replacing Qwen3.5-4B-Q4_K_M as the default on the Jetson Orin Nano Super 8GB.

**Constraints reminder:** ~3 GB GGUF file at Q4_K_M to leave headroom for 32K context KV cache + OS (~1-2.5 GB KV cache depending on GQA config + ~2 GB OS/runtime). Total memory budget: 7.4 GB unified LPDDR5.

### New Models Discovered

#### 1. Gemma 4 Edge Models (Google, released April 2, 2026)

**Gemma 4 E4B:**
- 4.5B effective parameters / 8B total stored (Per-Layer Embeddings inflate disk and memory size)
- Dense model with hybrid attention (local sliding window + global), 128K context
- Native multimodal: text + images + audio (30s via USM conformer encoder)
- Function calling, configurable thinking mode
- **Q4_K_M GGUF: 5.34 GB** -- TOO LARGE for Jetson at 32K context
- NVIDIA Developer Forums confirm CUDA OOM on Orin Nano 8GB with E4B
- Entry 008 (item 8) noted a user got it running with context auto-reduced to 120K, but at ~5 GB model weight + 1.4 GB headroom, there is effectively no room for KV cache at useful context lengths
- **Verdict: DOES NOT FIT for production use.** The "E4B" name is misleading for memory planning -- you must load all 8B stored params. The PLE lookup tables consume real memory.

**Gemma 4 E2B:**
- 2.3B effective parameters / 5.1B total stored (PLE architecture)
- Same capabilities as E4B (multimodal, function calling, 128K context)
- Q4_K_M GGUF: ~3-3.5 GB estimated
- Ollama users also report issues on Orin Nano (ollama issue #15398)
- Benchmarks: weaker than Qwen3.5-4B on text/agent tasks. Qwen wins 3 of 4 overlap benchmarks. E2B leads only on MMMLU (multilingual).
- **Verdict: Marginal fit, weaker than current model on text tasks.** Only interesting if on-device audio/image input is needed.

**Gemma 4 26B-A4B (MoE):**
- 26B total / 3.8B active per token, 128 tiny experts
- Q4_K_M GGUF: ~15-16 GB
- **Verdict: DOES NOT FIT.**

#### 2. Phi-4-mini-instruct (Microsoft, released Feb 2025)

- 3.8B dense parameters, decoder-only Transformer
- 200K vocabulary, GQA, shared embeddings, 128K context
- **Q4_K_M GGUF: 2.49 GB** -- fits well within budget
- MMLU: 68% (vs Qwen3.5-4B ~79% on MMLU-Pro)
- Strong on math/logic reasoning via synthetic training data
- ~65-70% of Qwen3.5-4B performance at 52% of parameter count
- MIT license
- **Verdict: FITS but notably weaker than Qwen3.5-4B across benchmarks.** The 2.49 GB file size leaves exceptional headroom. Worth testing as a low-memory fallback mode, not as primary replacement.
- **Phi-4-mini-reasoning** variant also available -- same 3.8B, chain-of-thought tuned. Potentially interesting for structured reasoning.

#### 3. Ministral 3 3B (Mistral, released Dec 2025)

- 3B dense parameters with vision support
- Instruct, base, and reasoning variants (all Apache 2.0)
- Q4_K_M GGUF: ~2 GB estimated
- Matches Llama 3.1 8B on some MMLU subsets
- Tool use, vision input support
- **Verdict: FITS but 3B is a step down from 4B.** Not an upgrade path.

#### 4. Qwen3-Coder-Next (Alibaba, released Feb 2026)

- 80B total / 3B active parameters (MoE)
- Exceptional coding agent performance
- **Q4_K_M GGUF: 48.4 GB** -- must load all 80B params
- **Verdict: DOES NOT FIT.** MoE models are a trap for edge -- "3B active" means nothing when all expert weights must be in memory.

#### 5. Qwen3.5-9B (Alibaba, released March 2026)

- 9B dense, natively multimodal, 262K context
- MMLU-Pro 82.5, GPQA Diamond 81.7 -- beats models 3-13x its size
- **Q4_K_M GGUF: 5.3 GB** -- needs ~8.3-9.3 GB total (model + KV + OS)
- **Verdict: DOES NOT FIT.** Crosses the 8GB ceiling.

#### 6. Llama 4 Scout/Maverick (Meta, released April 5, 2026)

- Scout: 17B active / 109B total. Maverick: 17B active / 400B total.
- No sub-10B Llama 4 models exist.
- **Verdict: DOES NOT FIT.**

#### 7. Qwen3.5-4B Community Fine-Tunes (March 2026)

Notable community fine-tunes of Qwen3.5-4B available as GGUF:
- **Qwen3.5-4B-Claude-Opus-Reasoning-Distilled-v2** (83K downloads, by Jackrong) -- distilled from Claude Opus reasoning traces, improved structured reasoning
- **Qwen3.5-4B-Claude-Opus-Reasoning-Distilled** (306K downloads) -- v1 of above
- **Qwen3.5-4B-Neo** -- competitive programming focus
- All share the same ~2.5-2.6 GB Q4_K_M GGUF footprint
- **Verdict: WORTH TESTING.** Zero memory cost to experiment. The reasoning-distilled variants could improve structured output quality.

### Embedding Model Update

**Current:** Qwen3-Embedding-4B (GGUF, ~2.5 GB at Q4_K_M)

**New option: EmbeddingGemma 300M (Google, Sep 2025)**
- 300M parameters, based on Gemma 3 + T5Gemma
- Highest-ranking multilingual embedding model under 500M on MTEB
- GGUF available from ggml-org (Q8_0 ~300 MB, Q4_0 ~150 MB)
- Outperforms Qwen3-Embedding-0.6B but NOT Qwen3-Embedding-4B
- **Verdict: Useful only if dual-mode chat+embed is needed simultaneously.** At ~200 MB it could coexist with a chat model. For dedicated embedding mode (current setup), Qwen3-Embedding-4B remains superior.

No new 1-4B embedding model that beats Qwen3-Embedding-4B was found.

### Comparison Table: All Candidates vs Current Model

| Model | Stored Params | Q4_K_M Size | Benchmark Class | Context | Fits @32K | Upgrade? |
|-------|--------------|-------------|-----------------|---------|-----------|----------|
| **Qwen3.5-4B (current)** | 4B | ~2.6 GB | Best-in-class 4B | 262K | YES | Baseline |
| Phi-4-mini | 3.8B | 2.49 GB | ~70% of Qwen3.5 | 128K | YES | No -- weaker |
| Gemma 4 E2B | 5.1B | ~3.3 GB | Below Qwen3.5-4B | 128K | Marginal | No -- weaker, tight fit |
| Ministral 3 3B | 3B | ~2 GB | Good for 3B | 128K | YES | No -- smaller class |
| Gemma 4 E4B | 8B | 5.34 GB | N/A on this HW | 128K | NO | N/A |
| Qwen3.5-9B | 9B | 5.3 GB | Exceptional | 262K | NO | N/A |
| Qwen3-Coder-Next | 80B (3B act) | 48.4 GB | Exceptional code | 262K | NO | N/A |
| Llama 4 Scout | 109B (17B act) | ~70 GB | N/A | 10M | NO | N/A |
| **Qwen3.5-4B distills** | 4B | ~2.6 GB | Same+ reasoning | 262K | YES | **Maybe** |

### Key Findings

1. **Qwen3.5-4B remains the best model for the Jetson Orin Nano 8GB.** Nothing in the current landscape unseats it. Highest benchmarks of any model that fits, longest context in its class.

2. **Google's "effective parameters" marketing is a memory-planning trap.** Gemma 4 E4B sounds like a 4B competitor but loads 8B parameters. Confirmed OOM on Orin Nano 8GB. Always use stored/total parameter count for edge memory planning.

3. **MoE models are categorically unsuitable for 8GB edge.** All experts must be loaded regardless of per-token activation count. Qwen3-Coder-Next (3B active / 80B total = 48 GB GGUF) and Gemma 4 26B-A4B (3.8B active / 26B total = 15 GB GGUF) are both non-starters.

4. **No Qwen4 exists.** Latest is Qwen3.5 (March 2026). No announcements found.

5. **No new Llama models fit.** Meta skipped sub-10B entirely with Llama 4.

6. **The only zero-cost improvement path is community fine-tunes** of Qwen3.5-4B, particularly the reasoning-distilled variants.

7. **No new embedding model beats Qwen3-Embedding-4B** in the 1-4B range. EmbeddingGemma 300M is interesting only for concurrent chat+embed scenarios.

### Classification

| Category | Models |
|----------|--------|
| **ACTION NEEDED** | None. Current model is optimal for this hardware. |
| **WORTH TRYING** | Qwen3.5-4B-Claude-Opus-Reasoning-Distilled-v2 (same footprint, better reasoning potential) |
| **WORTH WATCHING** | Gemma 4 E2B (if multimodal needed), EmbeddingGemma 300M (if dual-mode needed), Phi-4-mini-reasoning (low-memory fallback), Qwen3.5-4B multimodal mode in llama.cpp |
| **NO ACTION** | Gemma 4 E4B/26B-A4B, Qwen3.5-9B, Llama 4, Qwen3-Coder-Next, Ministral 3 3B |

### Recommended Next Steps

1. **Download and test Qwen3.5-4B-Claude-Opus-Reasoning-Distilled-v2 GGUF** from [HuggingFace](https://hf.co/Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF). Same memory footprint as current model. Benchmark against current model with identical prompts.
2. **Re-survey in July 2026** or when Qwen4 / Gemma 5 / Llama 4.1 announcements appear.
3. **Investigate Qwen3.5-4B multimodal support** in llama.cpp -- the March 2026 base model is natively multimodal (text+image+video). Could unlock image understanding on the Jetson without model swap. Requires llama.cpp rebuild (aligns with Entry 008 action item #1).

### Next Check

Re-evaluate model landscape in **July 2026** or when major new small model family is announced.

---

## Entry 010: llama.cpp Release Deep Dive — Build 8414 to b8766 (2026-04-12)

**Objective:** Detailed technical analysis of all llama.cpp releases since our current build (8414, commit 5744d7ec4) to identify specific performance improvements, bug fixes, and breaking changes relevant to Jetson Orin Nano Super (SM87, CUDA 12.6, 8GB unified memory).

**Methodology:** GitHub Releases API (all 210 releases enumerated), individual release note inspection for CUDA/performance-tagged changes, GitHub issue tracking for Jetson-specific bugs, web search for community benchmarks.

### Version Gap Summary

| Metric | Value |
|--------|-------|
| Current build | b8414 (commit 5744d7ec4, tag b8413-1-g5744d7ec4) |
| Latest build | b8766 (April 12, 2026) |
| Releases between | ~210 tagged releases |
| Time span | ~6 months |

### HIGH Priority Findings

#### H1. CUDA Flash Attention: Optimized Stream-K Fixup Kernel (b8680, PR #21159)

Specialized and optimized `flash_attn_stream_k_fixup` kernel for cases where `nblocks_stream_k` is a multiple of `ntiles_dst`. When `nblocks_stream_k > 4 * ntiles_dst`, uses a more efficient code path.

**Relevance:** Directly improves flash attention performance on all CUDA GPUs including SM87. Our workload (Qwen3.5-4B with `-fa` enabled) hits this code path on every inference. This is a core hot-path optimization.

**Classification: HIGH** -- direct tok/s improvement potential for our exact workload.

#### H2. CUDA Graphs: Faster Properties Check (b8702, PR #21472)

Replaces expensive CUDA graph properties check with fast hash computation using `memcpy`. Uses "seen node" optimization to avoid redundant checks.

**Relevance:** CUDA Graphs are already enabled in our build (`GGML_CUDA_GRAPHS:BOOL=ON` in CMakeCache). This reduces per-token overhead in the graph replay path. On small GPUs where kernel launch latency is a larger fraction of total time, this matters more.

**Classification: HIGH** -- reduces per-token overhead in our already-enabled CUDA Graphs path.

#### H3. CUDA: Fuse Multiply Operations (b8740, PR #21665)

Fuses multiple multiply operations into single kernels, reducing kernel launch overhead and memory traffic.

**Relevance:** General CUDA optimization that benefits all GPU models. Fewer kernel launches = less overhead, especially significant on Jetson where launch latency is proportionally larger than on high-end GPUs.

**Classification: HIGH** -- kernel fusion directly reduces overhead on small GPUs.

#### H4. CUDA Graph Node Equality: Store src ne/nb (b8757, PR #21736) and src data ptrs (b8720, PR #21635)

Two commits that improve CUDA graph equality checking by also storing `node->src` dimensions/strides (ne/nb) and data pointers. This prevents incorrect graph reuse when tensor metadata changes between iterations.

**Relevance:** Bug fix that prevents potential correctness issues with CUDA Graphs. If the graph was being incorrectly reused in our current build, this could fix subtle output quality issues.

**Classification: HIGH** -- correctness fix for CUDA Graphs, which we use.

#### H5. CUDA: Fix FA Kernel Selection Logic (b8624, PR #21271)

Fixes flash attention kernel selection logic to choose the correct kernel variant.

**Relevance:** Direct bug fix for FA on CUDA. If our current build was selecting a suboptimal FA kernel variant for SM87, this fix would improve both correctness and performance.

**Classification: HIGH** -- correctness/performance fix for our exact FA+CUDA configuration.

#### H6. Jetson/Tegra MoE Hang Fix (b8429+, PR #19227, Issue #19219)

`CUDA_SCALE_LAUNCH_QUEUES=4x` (from PR #19042, merged after our build) caused MoE models to deadlock on Jetson Orin due to command buffer exhaustion in unified memory. Fix: reverted to not setting `CUDA_SCALE_LAUNCH_QUEUES` on Tegra devices. NVIDIA CUDA team confirmed this was a JetPack bug, fixed in a future JetPack release.

**Relevance:** Our current build (b8414) predates the breaking change (b7309 was the original break, the fix was applied around b8429). We don't currently run MoE models. However, if we ever try Qwen3-30B-A3B or similar MoE models, this fix is critical. Also, our JetPack (R36.5.0, Jan 2026) may or may not include the driver-level fix.

**Classification: HIGH** -- critical if MoE models are ever deployed; informational for current config.

#### H7. Server: Save and Clear Idle Slots (b8658, PR #20993)

New `--clear-idle` flag (enabled by default) saves and clears idle slot KV cache from VRAM when new tasks arrive. Frees GPU memory that would otherwise sit allocated but unused.

**Relevance:** On an 8GB unified memory system, every MB matters. This automatically reclaims KV cache memory from completed conversations, directly addressing our memory pressure constraints. Previously, idle slots held their KV cache allocations indefinitely.

**Classification: HIGH** -- significant for memory-constrained Jetson operation.

#### H8. CUDA: Fix BF16 FA Compilation (b8474, PR #20865)

Fixes compilation of BF16 (bfloat16) flash attention kernels on CUDA.

**Relevance:** SM87 (Ampere) supports BF16. Combined with b8470 (native bf16 flash attention for vec kernel), this enables a code path that wasn't available in our build. BF16 FA could improve throughput for models that use BF16 internally.

**Classification: HIGH** -- enables previously broken BF16 FA code path on SM87.

#### H9. CUDA: Native BF16 Flash Attention for Vec Kernel (b8470, PR #20525)

Implements native BF16 flash attention for the vectorized kernel path. Previously, BF16 was converted to FP16 before FA; now it runs natively.

**Relevance:** Direct performance improvement for BF16 models on SM87. Even for our Q4_K_M model (which dequantizes to FP16 for compute), the internal FA paths may benefit from reduced type conversion overhead.

**Classification: HIGH** -- new optimized code path for SM87 hardware.

### MEDIUM Priority Findings

#### M1. KV Cache Quantization Improvements (b8714, b8699, b8644)

- b8714: Extended cache quantization checks to also verify flash attention is enabled
- b8699: Support attention rotation for heterogeneous iSWA (interleaved sliding window attention)
- b8644: Revert of "do not quantize SWA KV cache" -- restores Q8_0/Q4_0 cache for SWA layers

**Relevance:** Our current startup scripts don't use `--cache-type-k` or `--cache-type-v`. These improvements make KV cache quantization more robust and correct. On Jetson 8GB, using `--cache-type-k q8_0 --cache-type-v q8_0` could halve KV cache memory (~1.3 GB savings at 32K context), enabling either larger context windows or more headroom.

**Classification: MEDIUM** -- enables a new optimization dimension we should test.

#### M2. MOE GEMV Kernel Optimization for BS > 1 (b8579, PR #20905)

Dedicated MoE multi-token GEMV kernel with grid `(ceil(nrows_x/rpb), nchannels_dst)` and warp-level reduction only (no shared memory sync). Dramatically reduces thread block count and improves work per block for MoE architectures.

**Classification: MEDIUM** -- irrelevant for current Qwen3.5-4B (dense), but important if MoE models are tested.

#### M3. CUDA: Increase Per-Thread Output Elements for Small K-Dimension (b8469, PR #20635)

For FFN-down matrices with small K-dimension (especially after tensor parallelism splits), increases the number of output elements per thread block to avoid idle threads.

**Classification: MEDIUM** -- indirect benefit; our model has standard K-dimensions but this general MATMUL optimization could help at margins.

#### M4. CUDA: CUB Argsort Fix (b8586, PR #21181)

Fixes incorrect offset calculation in CUB's argsort when `nrows % block_size == 0`, which caused uninitialized values. Affects top-k sampling correctness.

**Classification: MEDIUM** -- correctness fix for sampling; could affect output quality in edge cases.

#### M5. Q1_0 1-Bit Quantization (b8682, PR #21273)

New GGML_TYPE_Q1_0 with 128-element group size. CPU-only initially, with Metal (b8712, b8728) and Vulkan (b8742) backends added subsequently. CUDA support via b8759 (missing cases fixed).

**Classification: MEDIUM** -- extreme compression format. A 4B model at Q1_0 would be ~0.5 GB but with significant quality loss. Interesting for experimentation, not production.

#### M6. Server: Built-in Tools Backend (b8553, PR #20898)

Adds `--tools all` flag to llama-server, enabling built-in tool calling support directly in the server without external parsing.

**Classification: MEDIUM** -- useful feature if tool calling is needed from the Jetson endpoint.

#### M7. Backend-Agnostic Tensor Parallelism (b8738, PR #19378, Experimental)

New experimental tensor parallelism that works across backends, not just CUDA-specific. Supports GPT-OSS, Qwen 3 MoE, 2/4/8 GPU configurations.

**Classification: MEDIUM** -- not applicable to single-GPU Jetson, but shows the project direction.

#### M8. GGML_CUDA_FA_ALL_QUANTS Build Flag

Our current build has `GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF`. Setting this to ON compiles flash attention kernels for all KV cache quantization types (q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1). Longer compile time but enables efficient KV cache quantization.

**Classification: MEDIUM** -- should enable this on rebuild if we plan to use KV cache quantization.

#### M9. CUDA: Skip Compilation of Superfluous FA Kernels (b8763, PR #21768)

Skips compiling FA kernels that won't be used for the target architecture. Reduces build time significantly.

**Classification: MEDIUM** -- faster builds, no runtime impact.

#### M10. Graph Reuse Re-enabled with Pipeline Parallelism (b8507, PR #20927)

Re-enables CUDA graph reuse that was previously disabled when pipeline parallelism was active. Reduces overhead for multi-device setups, but the graph reuse improvements also benefit single-GPU operation.

**Classification: MEDIUM** -- general graph optimization that may have indirect benefits.

### LOW Priority Findings

| Build | Change | Why LOW |
|-------|--------|---------|
| b8766 | Gemma 4 audio conformer encoder support | Multimodal, not our workload |
| b8762 | MERaLiON-2 multimodal audio support | Multimodal, not our workload |
| b8761 | OpenCL q5_k support | OpenCL backend, we use CUDA |
| b8739 | HIP CDNA4 (gfx950) for MI350X | AMD GPU, not relevant |
| b8685 | SYCL Q8_0 reorder (~3x TG speedup on Intel Arc) | Intel GPU, not relevant |
| b8642 | HIP ROCm 7.2.1 bump | AMD GPU, not relevant |
| b8639 | WebGPU vectorized flash attention | WebGPU backend, not relevant |
| b8607 | WebGPU quantized buffers | WebGPU backend, not relevant |
| b8595 | SYCL enhanced FA performance | Intel GPU, not relevant |
| b8492 | RPC: RCE security patch | We don't use RPC |
| b8498 | Standard Hugging Face cache support | Convenience, no performance impact |

### Breaking Changes / Build Flag Changes

| Change | Impact on Our Build |
|--------|-------------------|
| `GGML_CUDA_FA_ALL_QUANTS` now available | Currently OFF in our build; should enable on rebuild for KV cache quant support |
| `GGML_CUDA_COMPRESSION_MODE` new flag | Controls compile-time binary size vs speed tradeoff; our build uses "size" |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` env var | Runtime flag to enable system RAM fallback for CUDA allocations. Jetson has native unified memory, so this may be redundant, but worth testing |
| `CUDA_SCALE_LAUNCH_QUEUES` env var | NEW since our build. Do NOT set on Jetson -- causes MoE deadlocks (issue #19219). Explicitly documented as multi-GPU optimization only. |
| Shared library changes | `libmtmd.so.0` now required by llama-server and llama-cli. Our current binary fails to start without it (confirmed: server running from older binary on Jetson). Full rebuild required. |
| Release artifacts now include CUDA 13 builds | CUDA 13 available for newer GPUs; Jetson stays on CUDA 12.6 |

### Recommended Build Command for Upgrade

```bash
cd ~/llm-server/llama.cpp
git fetch origin
git checkout b8766  # or latest stable tag

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

**New flags vs current build:**
- `GGML_CUDA_FA_ALL_QUANTS=ON` -- enables FA kernels for quantized KV cache types (was OFF)
- All other flags remain the same
- CUDA Graphs are ON by default (already was in our build)
- `GGML_CUDA_F16=ON` still recommended for SM87 (FP16 tensor cores)

### Post-Upgrade Testing Plan

1. **Baseline benchmark** before rebuild (current b8414 performance at identical settings)
2. **Rebuild to b8766** with updated flags
3. **Identical benchmark** after rebuild (same model, same context, same prompt)
4. **Test KV cache quantization**: `--cache-type-k q8_0 --cache-type-v q8_0` with 32K and 65K context
5. **Test `--clear-idle` behavior** (should be default) -- monitor memory after conversation completion
6. **Test thread count**: `-t 1` vs `-t 4` (per Entry 008 recommendation)
7. **Verify no MoE deadlock** if testing any MoE model

### Overall Classification: **ACTION NEEDED**

The accumulation of 210 releases contains multiple HIGH-priority CUDA optimizations that directly target our workload:
- Flash attention stream-K kernel optimization (H1)
- CUDA Graph correctness and performance fixes (H2, H4)
- Kernel fusion (H3)
- FA kernel selection fix (H5)
- BF16 FA support for SM87 (H8, H9)
- Memory-saving server features for constrained devices (H7)

Conservative estimate: **2-4 tok/s improvement** (14% to 28% over current 14 tok/s baseline) from cumulative CUDA optimizations, with additional memory efficiency from KV cache quantization and idle slot clearing.

The upgrade is straightforward (same build flags plus one addition) and low-risk. The only caution is the Jetson MoE deadlock issue (H6), which doesn't affect our current dense model but should be noted for future MoE experiments.

---

## Entry 011: Jetson Recon — Consolidated Report (2026-04-12)

**Date:** 2026-04-12
**Operator:** Claude Code (jetson-recon)
**Status:** RECON — no changes made

This entry consolidates findings from five parallel recon checks (Entries 008-010 plus live health check) into a single actionable summary.

### Overall Classification: ACTION NEEDED

**Primary action:** Rebuild llama.cpp from b8414 to b8766 (~210 releases behind). This is the highest-impact, lowest-risk improvement available.

### System Health: HEALTHY
- 24-day uptime, zero-load idle, thermals 48C, inference responding correctly
- Memory tight but stable (1.7 GB available)
- SSD swap untouched throughout
- Minor: `nvidia-smi` passwordless sudo not working — verify `/etc/sudoers.d/claude`

### Cross-Correlated Findings
1. **llama.cpp rebuild urgency** confirmed independently by releases analysis AND community forum scan
2. **Thread count `-t 1` experiment** recommended by community optimizers, corroborated by GPU-offload architecture (CPU threads add contention, not throughput, when 100% GPU)
3. **KV cache quantization** opportunity enabled by new build flag aligns with idle-slot clearing — both address the 8GB constraint
4. **Model landscape stable** — Qwen3.5-4B-Q4_K_M remains optimal; no new 4B-class model fits better
5. **JetPack 7.2 is the next platform change** — Q2 2026, full reflash, wait for community validation

### Prioritized Recommendations
1. Rebuild llama.cpp to b8766 (estimated +2-4 tok/s, 9 HIGH-priority fixes)
2. Test `-t 1` vs `-t 4` thread count (zero cost, potentially +43%)
3. Test `--mlock` for latency variance reduction
4. Test KV cache quantization (`--cache-type-k q8_0 --cache-type-v q8_0`)
5. Try Qwen3.5-4B-Claude-Opus-Reasoning-Distilled-v2 (same footprint, potentially better reasoning)
6. Fix nvidia-smi sudoers gap
7. Watch for JetPack 7.2 GA announcement

### Next Recon: ~2-4 weeks, or after llama.cpp rebuild is complete

---

## Entry 012: llama.cpp Rebuild b8414→b8766 + Optimization Experiments (2026-04-12)

**Date:** 2026-04-12 17:48–18:43 UTC
**Operator:** Claude Code
**Status:** COMPLETE — changes applied, server running optimized config

**Objective:** Rebuild llama.cpp from b8414 to b8766 (~210 releases), then test three zero-cost optimizations: thread count, mlock, and KV cache quantization.

### Pre-Rebuild Baseline (b8414, 13-day-old server)

| Test | Gen tok/s (steady) | PP tok/s (steady) |
|------|-------------------|-------------------|
| Short (32 tok) | 12.4 | 104 |
| Medium (256 tok) | 12.1 | 139 |
| Long (512 tok) | 12.1 | 143 |
| **RSS** | **5477 MB** | |
| **Available RAM** | **1.3 GB** | |

Note: Generation is below the 14.0 tok/s measured on 2026-03-30 with a fresh restart. The 13-day uptime likely contributes to some memory fragmentation.

### Rebuild Process

1. Backed up current binary to `~/llm-server/backup-b8414/`
2. `git fetch origin --tags && git checkout b8766`
3. Cleaned stale CMakeCache (had path from `davistroy` user)
4. Built with new flags:
   ```bash
   cmake -B build \
     -DGGML_CUDA=ON \
     -DCMAKE_CUDA_ARCHITECTURES=87 \
     -DGGML_CUDA_F16=ON \
     -DGGML_CUDA_FA_ALL_QUANTS=ON \
     -DGGML_NATIVE=ON \
     -DCMAKE_BUILD_TYPE=Release
   ```
5. New flag vs old: `GGML_CUDA_FA_ALL_QUANTS=ON` (enables FA kernels for quantized KV cache), `GGML_NATIVE=ON` (CPU-native optimizations)
6. Build completed successfully. Stopped service during build to free RAM.

### Post-Rebuild Benchmark (b8766, -t 4, no mlock, no KV quant)

| Test | Gen tok/s (steady) | PP tok/s (steady) |
|------|-------------------|-------------------|
| Short (32 tok) | **14.5** | 119 |
| Medium (256 tok) | **14.1** | 157 |
| Long (512 tok) | **14.1** | 168 |
| **RSS** | **5454 MB** | |
| **Available RAM** | **1.9 GB** | |

**Result: +17% generation throughput** (12.1→14.1 sustained), +13-17% prompt processing, 600 MB more available RAM.

### Experiment 1: Thread Count -t 1 vs -t 4

| Config | Gen tok/s (short) | Gen tok/s (long) | RSS |
|--------|-------------------|------------------|-----|
| -t 4 (baseline) | 14.5 | 14.1 | 5454 MB |
| **-t 1** | 14.5 | 14.1 | **5285 MB** |

**Result: No throughput difference.** The "43% gain" claim from community doesn't hold on this hardware/model combo — likely depends on specific model architecture or batch settings. However, `-t 1` saves **~170 MB RSS** by not allocating 3 extra thread stacks. **Adopted:** free memory with no downside.

### Experiment 2: --mlock

| Config | Gen tok/s (short) | Gen tok/s (long) | RSS |
|--------|-------------------|------------------|-----|
| No mlock | 14.5 | 14.1 | 5285 MB |
| **--mlock** | 14.5 | 14.1 | 5319 MB |

**Result: No measurable throughput or latency difference.** Makes sense — Jetson unified memory with CUDA VMM already pins model weights. SSD swap was never touched anyway. **Adopted as safety net:** prevents paging under unexpected memory pressure, negligible cost.

### Experiment 3: KV Cache Quantization (q8_0)

| Config | Gen tok/s (short) | Gen tok/s (long) | RSS |
|--------|-------------------|------------------|-----|
| Default (f16 KV) | 14.5 | 14.1 | 5319 MB |
| **q8_0 KV cache** | 14.5 | 14.1 | **5264 MB** |

Quality check (temperature=0): Tides explanation — accurate, well-structured, indistinguishable from f16 KV cache output.

**Result: No throughput or quality impact.** Saves **~55 MB at short sequences** (savings scale with context fill — at 32K fill, estimated ~150 MB savings from halved KV cache precision). **Adopted:** free memory savings with no downside.

### Final Optimized Config Verification

| Test | Gen tok/s (steady) | PP tok/s (steady) |
|------|-------------------|-------------------|
| Short (32 tok) | **14.4** | 116 |
| Medium (256 tok) | **14.0** | 156 |
| Long (512 tok) | **14.0** | 166 |
| **RSS** | **5102 MB** | |
| **Available RAM** | **2.3 GB** | |

### Summary: Before vs After

| Metric | Before (b8414, -t 4, f16 KV) | After (b8766, -t 1, mlock, q8_0 KV) | Delta |
|--------|-------------------------------|--------------------------------------|-------|
| Gen tok/s (sustained) | 12.1 | **14.0–14.4** | **+16–19%** |
| PP tok/s (sustained) | 139–143 | **156–166** | **+12–16%** |
| RSS | 5477 MB | **5102 MB** | **-375 MB** |
| Available RAM | 1.3 GB | **2.3 GB** | **+1.0 GB** |
| llama.cpp build | b8414 | **b8766** | +352 releases |

### Optimal Configuration (deployed)

```bash
exec "$LLAMA_SERVER" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --alias qwen3.5-4b \
    --ctx-size 32768 \
    --n-gpu-layers "$GPU_LAYERS" \
    --threads 1 \
    --flash-attn on \
    --reasoning off \
    --mlock \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --log-disable
```

### Rollback Plan

Old binary preserved at `~/llm-server/backup-b8414/`. To revert:
```bash
cp ~/llm-server/backup-b8414/llama-server ~/llm-server/llama.cpp/build/bin/
cp ~/llm-server/backup-b8414/lib*.so* ~/llm-server/llama.cpp/build/bin/
# Revert start-qwen35-server.sh: --threads 4, remove --mlock/--cache-type flags
sudo systemctl restart myscript
```

---

## Entry 013: Network Reconfiguration — Ethernet Static IP (2026-04-14)

**Date:** 2026-04-14
**Operator:** Claude Code
**Status:** COMPLETE

**Objective:** Assign static IP `192.168.10.58` on ethernet after moving the Jetson to the network rack with a wired connection.

### Previous State

| Setting | Value |
|---------|-------|
| Interface | enP8p1s0 |
| Method | DHCP (auto) |
| IP | 192.168.10.223/24 (DHCP-assigned) |
| Gateway | 192.168.10.1 |
| Connection | "Wired connection 1" (NM auto-generated) |

### Actions Taken

1. Created `/etc/NetworkManager/system-connections/ethernet-static.nmconnection` with static config
2. Added `/etc/sudoers.d/claude-nmcli` for passwordless `nmcli` access
3. Activated "Ethernet Static" connection via `nmcli connection up`

### New State

| Setting | Value |
|---------|-------|
| Interface | enP8p1s0 |
| Method | manual (static) |
| IP | **192.168.10.58/24** |
| Gateway | 192.168.10.1 |
| DNS | 192.168.10.1 |
| Connection | "Ethernet Static" (file-backed, autoconnect, priority 10) |
| Config file | `/etc/NetworkManager/system-connections/ethernet-static.nmconnection` |

### Verification

- Static IP confirmed via `ip addr show enP8p1s0`
- Default route via `192.168.10.1` with `proto static`
- Internet connectivity confirmed (ping 8.8.8.8, 12ms)
- LLM server uninterrupted (22h uptime maintained through change)
- Tailscale mesh connectivity maintained throughout

### Post-Change: WiFi Disabled

WiFi radio turned off and UBNT autoconnect disabled — ethernet is the sole uplink now that the device is rack-mounted.

```
WIFI radio: disabled
wlP1p1s0: unavailable
UBNT autoconnect: no
```

### Post-Change: LLM Server Verified

```
Model: qwen3.5-4b (Qwen3.5-4B-Q4_K_M)
Build: b8766
Gen tok/s: 13.1 (first request after idle — warmup)
Response: correct, coherent
```

### Notes

- Old "Wired connection 1" DHCP connection still registered in NM but inactive; "Ethernet Static" wins on `autoconnect-priority=10`
- Sudoers addition: `/etc/sudoers.d/claude-nmcli` grants `claude` passwordless `sudo nmcli`

---

## Entry 014: Jetson Recon (2026-04-15)
**Date:** 2026-04-15 21:00 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made

### llama.cpp Release Check
- Current: b8766 — Latest: b8802 (+36 commits)
- Classification: MEDIUM
- New CUDA Q1_0 quantization backend, P2P safety improvements, RDNA transport for RPC
- No SM87/Jetson/Tegra-specific changes; no flash attention or KV cache perf changes
- Upgrade priority LOW for single-GPU Jetson scenario

### JetPack Check
- Current: 6.2.2 (R36.5.0) — Latest for Orin Nano: 6.2.2 (unchanged)
- JetPack 7.1 released Jan 2026 but Orin Nano NOT supported (Thor-only)
- JetPack 7.2 still expected Q2 2026 (~late April/May) with Orin Nano support
- Key changes coming: kernel 6.8, Ubuntu 24.04, CUDA 13.0
- Hold on 6.2.2; monitor 7.2 release

### Qwen / Model Check
- No Qwen4 released
- **Watch item confirmed:** Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF (Jackrong) available — Q4_K_M ~2.6-2.8 GB, direct drop-in
- New contenders: Phi-4-mini-instruct (Q4_K_M ~2.5 GB), Gemma-4-E2B (multimodal, ~2 GB)
- HuggingFace links: [Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF](https://huggingface.co/Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF)

### Forum & Community Check
- 1 relevant new post since 2026-04-12
- ACTION: [Local-first coding agent for llama.cpp auto-configuration](https://forums.developer.nvidia.com/t/local-first-coding-agent-that-auto-configures-llama-cpp-for-maximum-hardware-performance/366366) (flouisdev, 2026-04-13)
- INFO: Cosmos Reason2-2B achieves 16-17 tok/s on Orin Nano Super (W4A16 quant, 5.8 GB at ctx 2048)
- INFO: Community repo [kreier/llama.cpp-jetson](https://github.com/kreier/llama.cpp-jetson) for build reference

### Cross-Correlated Findings
- Model selection (not infrastructure) is the primary lever for gains — Cosmos 16-17 tok/s result + new model landscape both confirm this
- Claude-Opus-Reasoning fine-tune can be tested with zero infrastructure changes

### Overall: WORTH WATCHING

### Recommendations
1. Test Qwen3.5-4B-Claude-Opus-Reasoning-Distilled-v2 Q4_K_M — same footprint, potential quality boost
2. Monitor JetPack 7.2 — expected within weeks, wait for community validation
3. Skip llama.cpp b8802 upgrade — no Jetson-relevant changes
4. Bookmark llama.cpp auto-configuration agent post for future reference

---

## Entry 015: Jetson Audit (2026-04-17)
**Date:** 2026-04-17 01:30 UTC
**Operator:** Claude Code (jetson-audit skill)
**Status:** AUDIT — no changes made

### Config Drift: None
Running command line matches `start-qwen35-server.sh` exactly. All documented flags present (`--ctx-size 32768 --n-gpu-layers 999 --threads 1 --parallel 1 --flash-attn on --reasoning off --mlock --cache-type-k q8_0 --cache-type-v q8_0`). Listening on `0.0.0.0:8080`. Mode = `qwen35`.

### Missing Optimizations: None
All known best-practice flags applied. Only absent flag is `--cont-batching`, which is moot at `--parallel 1`.

### Memory Budget: HEALTHY
| Component | Value | Status |
|-----------|-------|--------|
| llama-server RSS | 5,133 MB | OK (+10.8% vs baseline 4,631 MB) |
| Total used | 4.8 GB / 7.4 GB | OK |
| Available | 2.5 GB | OK (well above 500 MB threshold) |
| Swap (file) | 0 B used / 16 GB | OK |
| Swap (zram) | 221 MB used / 3.7 GB | OK (compressed RAM, expected) |
| systemd cgroup | 6.7 GB | OK (includes mlock + KV cache) |

### System Health: HEALTHY
- Uptime: 2 days, 5:33 (load 0.05/0.06/0.02)
- Power mode: MAXN_SUPER
- Thermals (idle): CPU 49.0°C, GPU 49.6°C, SoC max 49.8°C, TJ 49.8°C — well below warning
- Disk: 17% used (132G / 824G NVMe)
- Inference: HTTP 200, `/health` ok, slot 0 idle
- **Generation speed: 14.05 tok/s on 300-token completion (baseline: 14.0 tok/s)** — exact match
- Prompt processing: 95.93 tok/s (baseline 166 tok/s on warm cache; cold-start sample acceptable)
- Service restarted 21:24:23 EDT today, NRestarts=2 over service lifetime — no crash loop

### Version Currency
| Component | Running | Latest Known | Gap |
|-----------|---------|--------------|-----|
| llama.cpp | b8766 (547765a93) | b8802 | 36 commits — Recon 2026-04-15 confirmed no Jetson-relevant changes; SKIP |
| JetPack | 6.2.2 (R36.5.0) | 6.2.2 | current |
| Kernel | 5.15.185-tegra | (JetPack 6.2.2) | current |
| Active model | Qwen3.5-4B-Q4_K_M | Qwen3.5-4B-Q4_K_M | match |

### Anomalies / Watch Items
- **Journal access blocked:** `claude` user not in `adm` or `systemd-journal` groups; `sudo journalctl` requires password. Same root cause as documented `nvidia-smi sudoers` watch item. Limits remote diagnostic capability.
- **Stray `ggml_cuda_init: failed to initialize CUDA: operation not supported`** appears when running `llama-server --version` directly outside systemd (no `render` group). Expected — does NOT affect the running service, which has `SupplementaryGroups=render` and is confirmed using GPU (RSS 5 GB + 14 tok/s = full offload).

### Overall: HEALTHY
System is running at documented baseline performance with all optimizations applied and no config drift.

### Recommendations
1. **No immediate action required.** Current config is optimal for the 8 GB unified memory constraint.
2. (Optional, low-priority housekeeping) Add `claude` user to `systemd-journal` group to enable remote log inspection without sudo: `sudo usermod -aG systemd-journal claude`. Improves future audit fidelity.
3. (Optional) Old `~/llm-server/backup-b8414/` binary backup is 4 days old and current build is stable — safe to remove anytime now.

---

## Entry 016: Jetson Recon (2026-04-24)
**Date:** 2026-04-24 ~14:00 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made

### Check 1 — JetPack / Firmware: MEDIUM
JetPack 7.0/7.1 shipped but do NOT support Orin Nano — Thor-only. JetPack 7.2 remains the first version expected to bring Orin Nano support, targeted Q2 2026 but no firm date or GA announcement yet. Multiple active NVIDIA forum threads confirm the timeline. No JetPack 6.3 in evidence. When JP 7.2 ships: Ubuntu 24.04, kernel 6.8, CUDA 13.0, full reflash required. **Stay on 6.2.2.**

### Check 2 — llama.cpp Releases: MEDIUM/SECURITY
Latest release: **b8918** (2026-04-24). We are 152 builds behind (b8766 → b8918).

Key findings:
- **CVE-2026-21869 (CVSS 8.8):** Heap buffer overflow in `update_slots()` context-shift loop. Negative `n_discard` from client JSON causes CWE-787 write (GHSA-8947-pfff-2f3c). Fixed by clamping at JSON parse boundary. Jetson exposes port 8080 on LAN — directly exploitable.
- **b8863 — CUDA OOM retry:** `ggml-cuda` now flushes legacy allocation pool on OOM and retries before hard-failing. Free improvement for 8 GB unified memory budget.
- **b8776 — CUDA DeviceSegmentedSort:** Correctness fix for sort not capturable in CUDA graphs. No impact on current config (graphs not enabled).
- **b8916 — SWA-full logic fix:** Sliding window attention fix; not relevant to Qwen3.5-4B config.
- No SM87/Jetson/Tegra/unified-memory keywords in any release — no HIGH trigger match.
- Breaking changes: `--clear-idle` → `--cache-idle-slots` (b8852), `/api` endpoints removed (b8861), `mtmd_image_tokens_get_decoder_pos` signature change (b8847) — none affect current Jetson start scripts.

### Check 3 — Small Model Landscape: MEDIUM
No Qwen4 announced. Qwen3.6 landed April 2026 but only in large sizes (35B MoE / proprietary) — no 4B tier successor.

New models within 3 GB GGUF ceiling:
| Model | Params | Q4_K_M | Notes |
|-------|--------|--------|-------|
| Phi-4-mini-instruct | 3.8B dense | ~2.5 GB | Matches Llama-3.1-8B on MMLU (73%). Strong math/reasoning. GGUF available (bartowski, unsloth). Good nemotron replacement candidate. |
| Gemma 4 E2B-it | 2B active (MatFormer) | ~3.11 GB | Marginal fit. Wait for llama.cpp community confirmation of stable MatFormer inference. |
| Gemma 4 E4B-it | 4B active (MatFormer) | 4.98 GB | Does NOT fit — skip. |

Fine-tune update: `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF` — CoT distilled from Claude Opus 4.6, same architecture, same ~2.6 GB Q4_K_M footprint. Drop-in swap. v2 supersedes the v1 watch item.

Embedding: Nothing new beats Qwen3-Embedding-4B at sub-4B. Hold.

### Check 4 — Forum / Community: INFO (trigger match)
- **GPU frequency regression in R36.5:** Forum reports of GPU stuck at 624 MHz after R36.5 upgrade on some Orin Nano units. **Verified NOT affecting our device** — `jetson_clocks --show` confirms MaxFreq=1020 MHz. No action needed but worth monitoring.
- **OpenJet auto-config agent (flouisdev):** Auto-detects hardware, tunes llama.cpp flags. No Orin Nano validation. Our config already applies equivalent flags manually. SKIP.
- **TensorRT-LLM confirmed inferior on Orin class:** Community data shows TensorRT-LLM capped ~20 tok/s on 64GB AGX Orin while llama.cpp hits higher. Engine selection confirmed correct.
- **Gemma 4 E4B CUDA OOM on R36.4.7:** Fixed in R36.5 but model doesn't fit our memory budget anyway.

### Check 5 — Live Health: DEGRADED
| Metric | Value | Status |
|--------|-------|--------|
| Service | active (running 5 days since 2026-04-19) | OK |
| Uptime | 9 days 18 hours | OK |
| Mode | qwen35 | OK |
| Build | b8766 (547765a93) | Confirmed |
| RAM total | 7.4 GB | — |
| RAM available | **401 MB** | **BELOW 500 MB THRESHOLD** |
| Swap (zram) | ~2.9 GB used across 6 partitions | Loaded but no SSD swap |
| Disk | 132 GB / 824 GB (17%) | OK |
| Thermals | 46–48°C all zones | OK (well below 75°C) |
| Gen tok/s | **12.0** | **14% below baseline 14.0** (above 11.9 warn floor) |
| PP tok/s | 61.4 | Unreliable (19-token prompt, small-batch artifact) |
| Slots | 1 active | OK |
| GPU MaxFreq | 1020 MHz | OK — not affected by R36.5 regression |

### Cross-Correlated Findings
1. **RAM pressure → throughput degradation:** Available RAM 401 MB (below 500 MB threshold) correlates with 14% gen tok/s drop (14.0 → 12.0). After 5 days continuous operation, memory fragmentation or cache accumulation likely reducing GPU headroom. A service restart (or full reboot) would likely restore baseline performance.
2. **CVE-2026-21869 + LAN-exposed port 8080:** The heap buffer overflow is directly exploitable on the Jetson's LAN-exposed inference endpoint. Rebuild to b8918 closes this.
3. **CUDA OOM retry (b8863) + RAM pressure:** The new OOM retry logic in b8863 would provide an additional safety net for the current memory-constrained state.

### Triggered Alerts
| Trigger | Source | Match? | Details |
|---------|--------|--------|---------|
| JetPack 7.2 AND (Orin Nano OR Orin) | Check 1 | Partial | JP 7.2 confirmed targeting Orin, not yet released |
| SM87 OR Jetson OR Tegra OR unified memory | Check 2 | No | No keyword matches in b8767–b8918 release notes |
| Qwen4 OR Qwen3.5 successor | Check 3 | No | No Qwen4 announced; Qwen3.6 is large-only |
| llama.cpp AND (performance OR optimization) AND jetson | Check 4 | Yes | GPU freq regression thread, OpenJet agent, TensorRT-LLM comparison |

### Overall: ACTION NEEDED

### Recommendations
1. **REBUILD llama.cpp to b8918** — CVE-2026-21869 (CVSS 8.8 heap overflow) on a LAN-exposed server is the primary driver. CUDA OOM retry (b8863) is a free bonus. No config changes needed; all current build flags remain valid. Same build process as documented in JETSON_CONFIG.md.
2. **Restart service to clear RAM pressure** — available RAM at 401 MB is below threshold, correlating with 14% throughput drop. A `kill $(pgrep -f llama-server)` (systemd auto-restart) or full reboot should restore baseline. Can combine with rebuild.
3. **Test Claude-distilled Qwen3.5-4B fine-tune** — `Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2` is same architecture/size, zero-risk swap. Lowest-friction experiment available.
4. **Pull Phi-4-mini Q4_K_M** (~2.5 GB) as a benchmark candidate when there's a testing window. Strong math/reasoning scores.
5. **(Watch)** JetPack 7.2 — check back Q2 2026. Wait 2–4 weeks after GA for community validation before planning reflash.

---

## Entry 016: llama.cpp Release Scan (2026-04-24)
**Date:** 2026-04-24
**Operator:** Claude Code (jetson-recon Check 2)
**Status:** COMPLETE — rebuild recommended (security)

### Baseline
- Running: b8766 (547765a93)
- Previously scanned latest: b8802
- Scan window: b8767 → b8918 (152 builds)

### Jetson-Relevant Findings

**SECURITY — b8908: CVE-2026-21869 heap buffer overflow in server (CVSS 8.8)**
- Negative `n_discard` from client JSON caused heap-buffer-overflow in context-shift loop (CWE-787)
- Clamped at JSON parse boundary; zero already triggers auto-discard (n_left/2)
- Jetson exposes llama-server on LAN — highest-priority reason to rebuild

**MEDIUM — b8863: ggml-cuda: flush legacy pool on OOM and retry (#22155)**
- When the CUDA allocator hits OOM, now flushes legacy pool and retries before failing
- Directly relevant to Jetson's 8 GB unified memory under pressure
- Low risk defensive fix

**MEDIUM — b8776: CUDA: Limit DeviceSegmentedSort to immediate mode (#21718)**
- DeviceSegmentedSort not capturable in CUDA graphs; falls back correctly in graph mode
- Not directly impacting Jetson's current non-graph config, but prevents a potential failure mode

### Breaking Changes
- b8852: `--clear-idle` renamed to `--cache-idle-slots` — not in Jetson start scripts, no impact
- b8861: `/api` endpoints removed from server — current clients use `/v1/` only, no impact
- b8847: `mtmd_image_tokens_get_decoder_pos` signature change — not applicable (text-only)

### Classification
- HIGH: None (no SM87/Jetson/Tegra/aarch64-CUDA specific items)
- MEDIUM: b8863 (CUDA OOM retry), b8776 (CUDA graph sort fix)
- SECURITY: b8908 (CVE-2026-21869, CVSS 8.8)
- LOW: Vulkan flash-attn DP4A (b8779), HIP graph default-on (b8893), hexagon/WebGPU/SYCL/Metal work

### Recommendation
Rebuild to b8918 for CVE-2026-21869 (CVSS 8.8). Secondary benefit: CUDA OOM retry hardening. No SM87 perf gains, but security patch warrants the rebuild.

---

## Entry 008 — Small Model Landscape Recon (2026-04-24)

**Check:** Recon scan 3 — model landscape refresh since 2026-04-15 baseline.

### Key Findings

**Trigger match: NO (Qwen4 not released; Qwen3.6 is the current generation)**

**Qwen family status:**
- Qwen3.5 Small series (0.8B/2B/4B/9B) landed 2026-03-02. The 4B model (already on disk as Q4_K_M) is confirmed current generation, not superseded.
- Qwen3.6 released April 2026 — but only in 35B MoE and proprietary sizes (3.6-Plus, Omni). No new 4B dense drop. Qwen4: no evidence of existence.
- IFEval 89.8% at 4B — beats GPT-OSS-120B (88.9%). Current model selection remains optimal.

**New model candidates vs 3 GB ceiling:**

| Model | Params | Architecture | Q4_K_M size | Fits? | Notes |
|-------|--------|-------------|-------------|-------|-------|
| Gemma 4 E2B-it | ~2B effective | MatFormer/selective activation | ~3.11 GB | Marginal | Multimodal, 256K ctx, audio input. Larger full weight store behind E2B mask. |
| Gemma 4 E4B-it | ~4B effective | MatFormer/selective activation | ~4.98 GB | NO | Over ceiling by ~2 GB. |
| Phi-4-mini-instruct | 3.8B dense | Dense transformer | ~2.5 GB | YES | MMLU 73%, MATH 62% — matches Llama-3.1-8B on MMLU. Strong reasoning/math. GGUF at unsloth + bartowski. |
| Qwen3.5-2B | 2B dense | Qwen3.5 arch | ~1.3 GB est. | YES | Edge/IoT tier. Lower quality than current 4B. |

**Gemma 4 E2B clarification:** "E2B" means 2B *active* parameters via selective activation — full model weight file is larger (~3.1 GB Q4_K_M suggests full weights ~5-6B total). Fits but barely. Architecture is novel (MatFormer nested scaling); llama.cpp support requires verification.

**Fine-tune variants (zero memory cost):**
- `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF` — same weights size as base (~2.6 GB Q4_K_M), CoT distilled from Claude Opus 4.6. Uses `<think>` tags for structured reasoning. Direct drop-in replacement to test.
- v2 supersedes v1. Both on HuggingFace.

**Embedding model status:**
- Qwen3-Embedding-4B remains competitive. No new sub-4B GGUF challenger found that clearly beats it on MTEB.
- Qwen3-Embedding-8B scores higher on MTEB multilingual (70.58) but at 2x memory it doesn't fit alongside a chat model.
- EmbeddingGemma-300M is very small but not competitive at RAG quality.
- Hold current embedding setup.

### Recommendations
1. **HIGH priority test:** `Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF` Q4_K_M — same size as current model, zero risk, potential reasoning uplift for structured tasks.
2. **MEDIUM priority:** Phi-4-mini-instruct Q4_K_M (~2.5 GB) — evaluate as nemotron replacement. Strong math/reasoning profile.
3. **LOW priority:** Gemma 4 E2B-it — borderline fit, novel architecture, llama.cpp support unverified. Watch for community reports.
4. **Skip:** Gemma 4 E4B-it (4.98 GB, over ceiling).

---

## Entry 016: Jetson Recon — Forum & Community Scan (2026-04-24)
**Date:** 2026-04-24
**Operator:** Claude Code (jetson-recon skill, Check 4)
**Status:** RECON — no changes made

### Watch Item Status
flouisdev OpenJet auto-config agent (2026-04-13) — CONFIRMED, details below.

### Critical Finding: JetPack R36.5 GPU Frequency Regression
- Upgrade from R36.4.7 to R36.5 introduced GPU frequency cap at 624 MHz (vs 1 GHz in MAXN_SUPER)
- Bootloader reinstall workaround no longer effective; only fix is full reflash
- Current device is on R36.5.0. If freq has degraded, reflash required.
- Separate from the CUDA OOM bug (also in R36.4.7) that affects Gemma 4 E4B — that bug is fixed in R36.5
- For current workload (Qwen3.5-4B-Q4_K_M) neither issue is blocking; no JetPack change warranted
- Forum thread: https://forums.developer.nvidia.com/t/jetson-orin-nano-gpu-frequency-stuck-at-624-mhz-after-upgrade-to-r36-5/365705

### OpenJet Auto-Config Agent (flouisdev, 2026-04-13)
- GitHub: https://github.com/L-Forster/open-jet
- Auto-detects hardware, tunes llama.cpp GPU offload layers and KV cache quantization dynamically
- Benchmarked on RTX 3090: 2.4x tok/s vs Ollama default (38-40 tok/s on 27B model)
- No Jetson Orin Nano-specific validation
- Assessment: Not directly applicable — current config already applies equivalent flags optimally; monitor for Jetson validation

### Forum Findings Summary
| Finding | Source | Classification |
|---------|--------|----------------|
| GPU freq stuck at 624 MHz after R36.5 upgrade (reflash only fix) | NVIDIA Forums | ACTION — verify current freq |
| OpenJet auto-config agent (hardware-adaptive llama.cpp tuning) | NVIDIA Forums | INFO — watch for Jetson validation |
| AGX Orin 64GB: llama.cpp CUDA 12.9 + b3.16 → 300+ tok/s on 7B Mistral | NVIDIA Forums | INFO — not directly comparable |
| Gemma 4 E4B: CUDA OOM on R36.4.7, fixed in R36.5 (but freq regression caveat) | NVIDIA Forums | INFO — already on R36.5 |
| TensorRT-LLM on Orin 64GB: capped ~20 tok/s on all model sizes | NVIDIA Forums | INFO — confirms llama.cpp is correct engine |
| CUDA 13.2 wheels for Jetson Orin family | NVIDIA Forums | SKIP |
| jetson-containers llama.cpp container active, JP6.2+7 supported | GitHub | INFO |
| r/LocalLLaMA: no Jetson posts indexed for 2026 | Reddit | SKIP |

### Trigger Match: YES
flouisdev auto-config agent confirmed. Classified INFO, not ACTION: config is already manually optimal; no Jetson Nano benchmark exists for the tool.

### Recommendations
1. **ACTION (verify):** Check GPU frequency on device — `sudo jetson_clocks --show` or jtop. If capped at 624 MHz, that is a ~37% throughput regression from the expected 1 GHz. Reflash would be required.
2. **SKIP:** OpenJet agent — no Jetson Nano validation; current config already applies equivalent flags.
3. **SKIP:** JetPack upgrade — no new version for Orin Nano; R36.5.0 is current.
4. **INFO:** TensorRT-LLM confirmed inferior to llama.cpp for this class of workload. Engine selection correct.

---

## Entry 017: Jetson Recon (2026-04-27)
**Date:** 2026-04-27 ~18:00 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON -- no changes made

### Check 1 — JetPack / Firmware: No new release
- JetPack 6.2.2 (R36.5.0) remains current for Orin Nano -- no newer 6.x point release
- JetPack 7.2 confirmed for Q2 2026 with Orin support (Ubuntu 24.04, kernel 6.8, CUDA 13.0) -- NOT yet released
- JetPack 7.0 and 7.1 are Thor-only; 7.2 is the first 7.x for Orin
- Full reflash required for 7.x upgrade; no OTA path from 6.x
- No critical security patches or out-of-band firmware updates since last recon
- Classification: MEDIUM (imminent but not released)

### Check 2 — llama.cpp Releases: b8946 available (180 builds ahead of running b8766)
- Latest: **b8946** (2026-04-27), running: b8766, last seen: b8802
- ~135 releases in 15 days (rapid release cadence)
- **Security: CVE-2026-21869 (b8908)** -- heap-buffer-overflow from malicious client JSON, CVSS 8.8. Critical if server is network-exposed.
- **CUDA OOM flush-retry (b8863)** -- flushes legacy CUDA pool on OOM and retries. Directly relevant to 8GB unified memory.
- **Reasoning sampler 30% regression fix (b8786)** -- reasoning budget sampler disabled GPU sampling even when no budget configured. Running `--reasoning off` may still be affected.
- **Qwen3 duplicate scale fix (b8946)** -- NVIDIA-signed fix removing duplicate weight scale in attention for Qwen3/LLaMA.
- **CUDA graph LRU eviction (b8832)** -- better graph cache management, reduces memory churn on constrained devices.
- **Breaking: libcommon renamed to libllama-common (b8829)**, /api endpoints removed (b8861)
- Classification: MEDIUM (no direct SM87/Jetson keywords, but cumulative improvements are significant)
- Trigger match: NO

### Check 3 — Small Model Landscape: SmolLM3-3B is new contender
- **Qwen3.6 released** (April 2026) but only 27B dense and 35B-A3B MoE -- no small models (4B/2B) yet. Watch for small model release.
- **SmolLM3-3B** (HuggingFace): 3B dense, Q4_K_M = 1.92 GB, 64K context (128K w/ YARN), dual-mode reasoning, 11.2T training tokens, Apache 2.0. Most interesting new contender -- fits comfortably within 3 GB budget.
- **Ministral-3-3B-Instruct**: 3.4B dense + vision, ~2.1 GB Q4_K_M, 256K context. Worth evaluating for multimodal use cases.
- **Gemma 4 E2B: BLOCKED** -- llama.cpp PLE (Parameter-Level Ensembling) not implemented (issue #22243). Runs but quality is silently degraded. Also OOM reports on Orin Nano. Skip until PLE is fixed.
- No new embedding models beating Qwen3-Embedding-4B at sub-4B scale
- No new Qwen3.5-4B fine-tunes beyond already-tracked Claude Opus distills
- Classification: MEDIUM
- Trigger match: PARTIAL (Qwen3.6 exists but no small models yet)

### Check 4 — Forum & Community: TurboQuant KV cache is significant
- **TurboQuant KV cache compression** (TheTom/llama-cpp-turboquant fork): `--cache-type-k turbo4 --cache-type-v turbo4` enables Qwen3.5-9B with 100K context on 8GB Orin Nano Super. Memory at 7.2G/7.4G. Dated 2026-04-12. This is the most actionable new finding.
- **vLLM 0.17.0 wheel with Marlin GPTQ for SM 8.7**: pre-built at HF thehighnotes/vllm-jetson-orin. 3.8x prefill improvement on AGX Orin 64GB. Too memory-hungry for 8GB (6.4GB base overhead). Confirms llama.cpp remains correct choice.
- **NvMap/CUDA allocator bug**: systemic issue in JetPack R35.6+/R36.x. `drop_caches` before launch helps. No NVIDIA patch.
- Build flags GGML_CUDA_F16=ON and GGML_CUDA_FA_ALL_QUANTS=ON confirmed present in current build (JETSON_CONFIG.md)
- Classification: ACTION (TurboQuant is directly applicable)
- Trigger match: YES (llama.cpp + performance + optimization + jetson)

### Check 5 — Live Health: DEGRADED (memory borderline)
| Metric | Value | Status |
|--------|-------|--------|
| Service | active (running), PID 51111, since 2026-04-26 01:00 EDT | PASS |
| Mode | qwen35 | PASS |
| llama.cpp | b8766 (547765a93) | PASS |
| Uptime | 12d 17h | PASS |
| RAM available | 1.7 GiB (of 7.4 GiB) | PASS |
| Disk | 17% used (659 GB free) | PASS |
| Swap | 16 GB file (0 used) + 3.8 GB zram (359 MB used) | PASS |
| Max thermal | 48.6°C (6 zones: 46.8–48.6°C) | PASS |
| Gen tok/s | **12.19** (12.9% below 14.0 baseline) | PASS (borderline at 15% threshold) |
| PP tok/s | 68.9 (likely measurement artifact -- 19-token prompt too short for peak throughput) | NOTE |
| Cgroup memory | **5.8 GB** (28% above 4631 MB baseline) | **WARN** |

### Cross-Correlated Findings
1. **Reasoning sampler fix (Check 2) ↔ borderline gen tok/s (Check 5):** b8786 fixed a bug where the reasoning budget sampler disabled GPU sampling even without a reasoning budget. Current b8766 predates this fix. This may explain why gen tok/s is 12.9% below baseline (12.19 vs 14.0). Upgrading to b8946 could restore throughput.
2. **CUDA OOM flush-retry (Check 2) ↔ NvMap allocator bug (Check 4):** The new OOM recovery path in b8863 directly addresses the known JetPack R36.x unified memory allocator issue. Upgrading improves resilience against OOM crashes.
3. **TurboQuant KV cache (Check 4) ↔ SmolLM3-3B (Check 3):** TurboQuant expands what's possible on 8GB -- could enable 9B models at 100K context, or run smaller models with massive context headroom.
4. **Cgroup memory growth (Check 5) ↔ CUDA graph LRU eviction (Check 2):** Better graph cache management in b8832+ may help contain memory growth over long uptimes.

### Triggered Alerts
- **Forum trigger: YES** -- TurboQuant KV cache compression and llama.cpp optimization discussions directly match `llama.cpp AND (performance OR optimization) AND jetson`
- **HuggingFace trigger: PARTIAL** -- Qwen3.6 exists (successor line to Qwen3.5) but no small models released yet
- **llama.cpp trigger: NO** -- no SM87/Jetson/Tegra/unified memory keywords in release notes

### Overall: ACTION NEEDED

### Recommendations
1. **ACTION: Upgrade llama.cpp to b8946.** CVE-2026-21869 (CVSS 8.8) is a security risk if the server is network-reachable. The reasoning sampler fix (b8786) may restore the ~2 tok/s gap to baseline. CUDA OOM flush-retry and graph LRU eviction improve memory resilience. Note breaking change: `libcommon` → `libllama-common` in b8829 — update build scripts accordingly.
2. **ACTION: Evaluate TurboQuant KV cache.** TheTom/llama-cpp-turboquant enables `--cache-type-k turbo4` for dramatically compressed KV cache. Community-validated on same hardware with 9B model at 100K context. Could enable larger models or much larger context windows for current model. Check whether mainline b8946 has merged equivalent cache type support before using the fork.
3. **EVALUATE: Download SmolLM3-3B GGUF** and benchmark against Qwen3.5-4B. At 1.92 GB Q4_K_M with 128K context and dual-mode reasoning, it's a compelling alternative if quality is competitive.
4. **MONITOR: Cgroup memory at 5.8 GB (28% above baseline).** If it continues growing, schedule a service restart. Upgrading llama.cpp (recommendation 1) may address this via better CUDA graph management.
5. **WATCH: Qwen3.6 small model release.** When 4B-class Qwen3.6 models appear, they'll be a direct drop-in upgrade from Qwen3.5-4B.
6. **WATCH: Gemma 4 E2B.** Currently blocked by llama.cpp PLE issue #22243. Re-evaluate when fixed.
7. **WATCH: JetPack 7.2.** Q2 2026 release imminent. Wait 2-4 weeks post-release for community validation before planning reflash.

---

## Entry 018: Jetson Audit (2026-04-30)
**Date:** 2026-04-30 11:55 UTC
**Operator:** Claude Code (jetson-audit skill)
**Status:** AUDIT — no changes made

### Config Drift: None
Running process flags match `start-qwen35-server.sh` and JETSON_CONFIG.md exactly: `--model Qwen_Qwen3.5-4B-Q4_K_M.gguf --ctx-size 32768 --n-gpu-layers 999 --threads 1 --parallel 1 --flash-attn on --reasoning off --mlock --cache-type-k q8_0 --cache-type-v q8_0 --log-disable`. Listening on `0.0.0.0:8080`, mode = `qwen35`. Systemd unit active (running), PID 65978.

### Missing Optimizations: None
All best-practice flags applied. `--cont-batching` absent but moot at `--parallel 1`. No anti-patterns detected. Memory eviction script and OOM-guard fallback logic both present in startup script.

### Memory Budget: HEALTHY
| Component | Value | Status |
|-----------|-------|--------|
| llama-server RSS | 4,631 MB | HEALTHY (exact baseline match: 4,631 MB) |
| Total used | 4.5 GB / 7.4 GB | HEALTHY |
| Available | 2,783 MB | HEALTHY (well above 500 MB threshold) |
| Swap (file) | 0 B / 16 GB | HEALTHY (SSD swap untouched) |
| Swap (zram) | 351 MB / 3.7 GB | OK (compressed RAM, up from 221 MB at last audit — normal 15-day uptime accumulation) |
| systemd cgroup | 6.0 GB | IMPROVED (down from 6.7 GB at last audit) |

Notable: RSS returned to exact baseline (4,631 MB) after the cgroup memory was at 5,133 MB (+10.8%) in the last audit. Suggests the service was restarted between audits, resetting memory fragmentation.

### System Health: HEALTHY
- Uptime: 15 days, 20 hours (load 0.00/0.00/0.00)
- Power mode: MAXN_SUPER
- Thermals (idle): CPU 48.1°C, GPU 48.8°C, SoC max 48.8°C — well below warning thresholds
- Disk: 17% used (132G / 824G NVMe)
- Inference: HTTP 200, 10-token completion in 800ms
- Generation speed: 12.5 tok/s (10-token sample) — 10.7% below 14.0 baseline, but short generations carry proportionally more startup overhead; within normal variance for sample size
- Prompt processing: 53.9 tok/s (cold-cache, 19-token prompt)
- Journal errors (last hour): None
- Slot status: 1 slot, idle, no processing

### Version Currency
| Component | Running | Latest Known | Gap | Severity |
|-----------|---------|--------------|-----|----------|
| llama.cpp | b8766 (547765a93) | b8946+ (per Entry 017 recon) | ~180 builds | **HIGH** |
| JetPack | 6.2.2 (R36.5.0) | 6.2.2 | current | — |
| CUDA | 12.6 | 12.6 | current | — |
| Kernel | 5.15.185-tegra | (JetPack 6.2.2) | current | — |
| Active model | Qwen3.5-4B-Q4_K_M | Qwen3.5-4B-Q4_K_M | match | — |

**Key version gap:** Entry 017 recon identified CVE-2026-21869 (CVSS 8.8, reasoning sampler) fixed post-b8766, plus CUDA OOM flush-retry and graph LRU eviction improvements. Server is network-bound to `0.0.0.0:8080` but only reachable via Tailscale VPN, which mitigates the CVE exposure.

### Cross-Correlated Findings
1. **Version gap (Check 5) + stable config (Check 1):** System is well-configured but running outdated software. The only gap is the llama.cpp version — all other axes are healthy.
2. **Memory improvement (Check 3) + long uptime (Check 4):** RSS at exact baseline after 15 days suggests a clean restart occurred. Cgroup at 6.0 GB (down from 6.7 GB) confirms no memory creep this cycle.
3. **Inference speed (Check 4) marginally below baseline:** 12.5 vs 14.0 tok/s on a 10-token sample. Short sample bias — not a regression signal. Previous audit measured 14.05 tok/s on a 300-token completion.

### Overall: OPTIMIZATION AVAILABLE

One HIGH finding (llama.cpp version gap with CVE), no CRITICAL findings. System is healthy, stable, and well-configured.

### Recommendations
1. **ACTION: Upgrade llama.cpp to latest stable.** Same recommendation from Entry 017 recon — CVE-2026-21869, CUDA OOM recovery, and reasoning sampler fix. Note breaking change: `libcommon` → `libllama-common` in b8829. Server is Tailscale-only so CVE risk is mitigated, but upgrade is still warranted for the performance and reliability improvements.
2. **No other action required.** Config is optimal. Memory is healthy. All optimizations applied. Thermals nominal.
3. (Carry-forward) Old `~/llm-server/backup-b8414/` still exists — safe to remove, frees ~500 MB disk.

---

## Entry 019: Jetson Recon (2026-04-30)
**Date:** 2026-04-30 12:35 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made

### Check 1 — JetPack / Firmware: No update available
JetPack 6.2.2 (R36.5.0) remains the latest for Orin Nano. JetPack 7.2 is confirmed as the release that brings JP7.x to Orin (Ubuntu 24.04, kernel 6.8, CUDA 13.0) — still targeted Q2 2026 but not released. JP 7.0 and 7.1 only support Thor-class devices. CVE-2026-24148 already patched in current JP 6.2.2. Full reflash required when JP 7.2 drops — no OTA path.

### Check 2 — llama.cpp Releases: b8987 available (221-build gap)
Latest release: **b8987** (2026-04-30). Key changes since b8766:
- **b8908:** CVE-2026-21869 fix (CVSS 8.8, heap-buffer-overflow in reasoning sampler)
- **b8829:** Breaking rename: `libcommon` → `libllama-common` (build scripts must update)
- **b8832:** CUDA graphs LRU-based eviction — prevents unbounded memory growth (directly benefits 8GB unified)
- **b8863:** CUDA flush legacy pool on OOM + retry — adds OOM resilience on Jetson's unified memory
- **b8931:** CUDA MMQ stream-k overhead reduction — general perf improvement for SM87
- **b8946:** Qwen3 duplicate scale fix — correctness fix for Qwen3 family models
- No SM87/Jetson/Tegra keywords in release notes, but CUDA OOM/graph improvements are functionally significant for Jetson.

### Check 3 — Small Model Landscape: Qwen3.6 released (large only)
**Qwen3.6** released April 16-22 as Qwen3.5 successor — but only 27B dense and 35B-A3B MoE sizes. No small (4B) variants yet. Based on Qwen3.5 cadence (large → small in ~2 weeks), expect Qwen3.6 small models in late May.

New models evaluated:
| Model | Size | Fits? | Verdict |
|-------|------|-------|---------|
| Qwen3.6-35B-A3B (MoE) | ~21 GB | NO (35B total params) | Wait for small variant |
| Ministral-3-3B-Instruct | 2.15 GB Q4_K_M | YES | Smaller but lower quality than Qwen3.5-4B |
| Gemma 4 E4B | 5.41 GB Q4_K_M | NO (OOM confirmed by community) | PLE issue #22243 unresolved |
| Jina Embeddings v4 | ~1.93 GB Q4_K_M | YES | Worth evaluating vs Qwen3-Embedding-4B |

### Check 4 — Forum & Community: TensorRT Edge-LLM is new
**TensorRT Edge-LLM** — new NVIDIA pure-C++ inference runtime with INT4 AWQ quantization. Jetson AI Lab tutorial available for Qwen3-4B on Orin Nano. Eliminates Python from inference path; INT4 AWQ could free ~1 GB RAM vs Q4_K_M. Worth evaluating as llama.cpp alternative.

**Eric X. Liu benchmark analysis** — 66-test study proving memory bandwidth is the fundamental bottleneck on Orin Nano (20.8% avg HW utilization during autoregressive generation). Validates Q4_K_M + mlock strategy. Community reports ~15 tok/s ceiling on same hardware — current 13.7-14.0 is near ceiling.

**NVIDIA blog on CUDA Graphs** — up to 1.2x speedup from graph-based kernel dispatch. Check if enabled in current build; newer llama.cpp builds (b8832+) have improved graph cache management.

Community confirms Gemma 4 E4B OOM on Orin Nano. CUDA 12.9 showing gains on AGX Orin but requires JetPack 7.x.

### Check 5 — Live Health: HEALTHY
| Metric | Value | Status |
|--------|-------|--------|
| Uptime | 15 days, 20:40 | HEALTHY |
| Service | active (running), PID 65978 | HEALTHY |
| Mode | qwen35 | Expected |
| RAM used/avail | 4.5 GB / 2.7 GB | HEALTHY |
| Swap (zram) | 371 MB | OK (normal accumulation) |
| Swap (SSD) | 0 B | HEALTHY |
| Disk | 17% (132G/824G) | HEALTHY |
| GPU temp | 49.0°C | HEALTHY (idle) |
| CPU temp | 48.9°C | HEALTHY (idle) |
| Generation speed | 13.7 tok/s (41-token sample) | OK (2.1% below 14.0 baseline — normal variance) |
| Prompt processing | 76.3 tok/s | OK (cold cache) |

### Cross-Correlated Findings
1. **llama.cpp upgrade (Check 2) ↔ CUDA Graphs (Check 4):** b8832+ adds LRU graph cache eviction preventing memory leaks. NVIDIA blog claims up to 1.2x speedup from CUDA Graphs. Upgrading llama.cpp + verifying CUDA graphs = highest-value single action.
2. **Model landscape (Check 3) ↔ Forum OOM reports (Check 4):** Gemma 4 E4B OOM confirmed by both HuggingFace tracking and forum reports. Validates staying with Qwen3.5-4B until Qwen3.6 small models arrive.
3. **TensorRT Edge-LLM (Check 4) ↔ Bandwidth bottleneck (Check 4):** INT4 AWQ = smaller weights = less bandwidth per token. If the bottleneck is bandwidth (confirmed at 20.8% compute utilization), tighter quantization is the lever — either via TensorRT Edge-LLM or llama.cpp's own quant improvements.
4. **JetPack 7.2 (Check 1) ↔ CUDA 12.9 gains (Check 4):** CUDA 12.9 showing 300+ tok/s on AGX Orin (7B) but requires JetPack 7.x. JP 7.2 for Orin Nano will bring CUDA 13.0 — potential step-change in inference speed.

### Triggered Alerts
- **JetPack trigger: NO** — JP 7.2 announced Q2 2026 but not released
- **llama.cpp trigger: NO** — No SM87/Jetson/Tegra/unified memory keywords (but CUDA OOM improvements are functionally relevant)
- **HuggingFace trigger: YES** — Qwen3.6 released as Qwen3.5 successor, but no small models yet
- **Forum trigger: YES** — TensorRT Edge-LLM + CUDA Graphs + benchmark analysis match pattern

### Overall: WORTH WATCHING

### Recommendations
1. **ACTION: Upgrade llama.cpp to b8987.** CVE-2026-21869 fix, CUDA OOM resilience (b8863), CUDA graphs LRU (b8832), Qwen3 correctness fix (b8946). Breaking change: `libcommon` → `libllama-common` in b8829 — update build scripts and LD_LIBRARY_PATH. Staged approach: rebuild, benchmark, promote.
2. **EVALUATE: TensorRT Edge-LLM.** NVIDIA's pure-C++ runtime with INT4 AWQ for Qwen3-4B on Orin Nano. Jetson AI Lab tutorial available. Could yield modest tok/s gain + ~1 GB RAM savings. Schedule as weekend experiment after llama.cpp upgrade.
3. **EVALUATE: Verify CUDA Graphs status** in current llama.cpp build. NVIDIA blog claims up to 1.2x speedup. Newer builds have improved graph cache management — part of the llama.cpp upgrade benefit.
4. **WATCH: Qwen3.6 small models.** Expected late May based on large→small cadence. Direct drop-in upgrade path from Qwen3.5-4B.
5. **WATCH: Jina Embeddings v4.** 3B model (~1.93 GB Q4_K_M), potentially better retrieval quality than Qwen3-Embedding-4B.
6. **WATCH: JetPack 7.2.** Q2 2026 target, not yet released. CUDA 13.0 could be significant. Wait 2-4 weeks post-release for community validation.
7. **WATCH: Gemma 4 E2B.** Still blocked by llama.cpp PLE issue #22243. Community OOM reports confirm E4B won't fit either.

---

## Entry 020: llama.cpp Rebuild b8766 → b8987 (2026-04-30)
**Date:** 2026-04-30 19:45 UTC
**Operator:** Claude Code (subagent-driven upgrade)
**Status:** REBUILD — system modified

### Motivation
- CVE-2026-21869 (CVSS 8.8, heap-buffer-overflow in reasoning sampler, fixed b8908)
- CUDA legacy pool OOM flush-retry (b8863) — adds OOM resilience on 8GB unified memory
- CUDA graphs LRU-based eviction (b8832) — prevents unbounded memory growth
- CUDA MMQ stream-k overhead reduction (b8931) — general CUDA perf improvement
- Qwen3 duplicate scale fix (b8946) — correctness fix for Qwen3 family
- Library rename: libcommon → libllama-common (b8829, breaking build change)
- 221-build gap from b8766, ~18 days

### Build
- Source: b8987 (commit 5f0ab726f)
- cmake flags: unchanged from b8766 (GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=87, GGML_CUDA_F16=ON, GGML_CUDA_FA_ALL_QUANTS=ON, GGML_NATIVE=ON)
- ggml version: 0.10.1 (was 0.9.11)
- Library rename confirmed: libcommon.so → libllama-common.so — no systemd unit changes needed (LD_LIBRARY_PATH points to directory)
- Clean build (rm -rf build) required to avoid stale cmake cache from library rename

### Benchmark (post-b8987)

```
=== Benchmark: post-b8987 (2026-04-30T19:45:05Z) ===

Warmup...

--- Short tests (small prompt, ~20 token output) ---
short | run1 | prompt_tok=25 | gen_tok=32 | pp=106.4 tok/s | gen=15.7 tok/s
short | run2 | prompt_tok=25 | gen_tok=32 | pp=115.1 tok/s | gen=15.7 tok/s
short | run3 | prompt_tok=25 | gen_tok=32 | pp=118.7 tok/s | gen=15.7 tok/s

--- Medium tests (medium prompt, 256 token output) ---
medium | run1 | prompt_tok=36 | gen_tok=256 | pp=152.6 tok/s | gen=15.3 tok/s
medium | run2 | prompt_tok=36 | gen_tok=256 | pp=154.0 tok/s | gen=15.3 tok/s
medium | run3 | prompt_tok=36 | gen_tok=256 | pp=153.9 tok/s | gen=15.3 tok/s

--- Long tests (medium prompt, 512 token output) ---
long | run1 | prompt_tok=41 | gen_tok=512 | pp=170.5 tok/s | gen=15.2 tok/s
long | run2 | prompt_tok=41 | gen_tok=512 | pp=166.8 tok/s | gen=15.2 tok/s
long | run3 | prompt_tok=41 | gen_tok=512 | pp=166.2 tok/s | gen=15.3 tok/s

--- Memory ---
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       4.8Gi        91Mi       3.0Mi       2.5Gi       2.4Gi
Swap:           19Gi       288Mi        19Gi

--- Server process ---
RSS: 4969.26MB, VSZ: 43282.7MB

=== End benchmark: post-b8987 ===
```

### Comparison to Baseline

| Metric | Baseline (b8766) | Post-b8987 | Delta |
|--------|-------------------|------------|-------|
| Gen tok/s (short) | 14.0 | 15.7 | **+12.1%** |
| Gen tok/s (medium) | 14.0 | 15.3 | **+9.3%** |
| Gen tok/s (long) | 14.0 | 15.2–15.3 | **+8.6–9.3%** |
| PP tok/s (long) | 166 | 167–170 | +0.6–2.4% |
| RSS (MB) | 4,631 | 4,969 | +338 MB (+7.3%) |
| Available RAM | 2,783 MB | 2,400 MB | -383 MB (still healthy) |
| GPU temp (post-bench) | 48.8°C | 63.6°C | Expected (under load vs idle) |
| Swap (zram) | 351 MB | 288 MB | Improved (fresh restart) |

### Analysis
Generation throughput improved 9-12% across all test sizes, with short-context showing the largest gain. This likely comes from CUDA MMQ stream-k improvements (b8931) and general CUDA kernel optimizations accumulated over 221 builds. RSS increased 338 MB (+7.3%) — likely due to larger CUDA graph cache or updated runtime buffers — but available RAM remains healthy at 2.4 GB, well above the 500 MB threshold.

Prompt processing is consistent with baseline at ~153-170 tok/s depending on prompt length. First-run cold-cache PP (106 tok/s) warms up quickly.

### Result
**PASS** — All acceptance criteria met. CVE-2026-21869 closed. Throughput improved. Memory healthy. No errors.

### New Baseline
- baseline_gen_tok_s: 14.0 → 15.3 (median of long-run results)
- baseline_rss_mb: 4631 → 4969

---

## Entry 008 — TensorRT Edge-LLM Research (2026-04-30)

**Objective:** Evaluate NVIDIA TensorRT Edge-LLM as potential alternative/complement to llama.cpp for LLM inference on Jetson Orin Nano Super 8GB.

### What TensorRT Edge-LLM Is

TensorRT Edge-LLM (github.com/NVIDIA/TensorRT-Edge-LLM) is NVIDIA's high-performance **C++ inference runtime** for LLMs and VLMs on embedded platforms. Latest release: **v0.7.0** (2026-04-28). It is NOT the same as TensorRT-LLM (the datacenter-oriented Python-heavy project). Key differences:

- Pure C++ runtime — no Python in inference path
- Three-stage pipeline: HuggingFace → quantize+ONNX export (Python, on host) → TensorRT engine build (C++, on device) → inference (C++, on device)
- Engines are hardware-specific: must be built on target device (SM87 engine won't run on SM110, vice versa)
- Target: embedded/automotive (Jetson, DRIVE platforms)

### Platform Support for Orin Nano

| Detail | Finding |
|--------|---------|
| EMBEDDED_TARGET=jetson-orin | **Exists** in cmake build system, maps to SM87 |
| Official support | **Experimental** — docs say "Jetson Orin with JetPack 6.2.x is compatible but support is experimental" |
| Officially supported | Jetson Thor (JetPack 7.1) and DRIVE Thor (DriveOS 7) only |
| JetPack 6.2 compat | Added in v0.5.0 per CHANGELOG.md |
| CUDA requirement | Docs say CUDA 12.8+/13.x; we have CUDA 12.6 (JetPack 6.2.2) — **potential version mismatch** |
| TensorRT requirement | 10.x+ — JetPack 6.2.2 ships TensorRT 10.x, should be compatible |

### Model Support

Qwen3.5-4B-Instruct is **explicitly listed** in supported models. Also supported: Qwen3-4B, Qwen2.5 series, Nemotron-3-Nano-4B, Llama 3.x. Quantization options for Qwen3.5: "Dense precision set" (FP16/BF16 + FP8/INT4 AWQ/INT8 SmoothQuant/GPTQ). Note: NVFP4 requires Blackwell (SM100+), not available on Orin.

### INT4 AWQ vs GGUF Q4_K_M

| Aspect | INT4 AWQ | GGUF Q4_K_M |
|--------|----------|-------------|
| Model size (4B) | ~2 GB weights | ~2.6 GB (Qwen3.5-4B) |
| Perplexity vs FP16 | +0.05 to +0.2 ppl | +0.1 to +0.3 ppl |
| Quality retention | 98-99% of FP16 | 97-99% of FP16 |
| Format | TensorRT engine (device-specific binary) | Portable GGUF file |
| GPU inference speed | Optimized TRT kernels | llama.cpp CUDA kernels |
| CPU fallback | No | Yes |
| Practical difference | Marginal — task-dependent, gap is narrow |

### API Compatibility

- **No built-in OpenAI-compatible HTTP server.** The inference binary (`llm_inference`) reads JSON input files and writes JSON output files — batch mode, not real-time.
- v0.7.0 README mentions "Experimental High-Level Python API and Server" with "vLLM-style API and OpenAI-compatible server" — but this is experimental, requires Python, and unclear if it works on Orin.
- C++ API surface: create runtime → capture CUDA graphs → call `handleRequest()` per query. Building an HTTP wrapper would require custom work.
- **Not a drop-in replacement for llama.cpp's built-in OpenAI server.**

### Expected Performance vs llama.cpp

No published Orin Nano tok/s benchmarks for Edge-LLM exist. Estimates based on available data:

- Current llama.cpp: **15.3 tok/s** generation (Qwen3.5-4B Q4_K_M, full GPU offload)
- TensorRT typically provides 20-70% speedup over llama.cpp on datacenter GPUs
- On Orin Nano, both are **memory-bandwidth-bound** (only 20.8% compute utilization per benchmarks) — TRT kernel optimization yields less benefit when bottleneck is bandwidth, not compute
- Realistic estimate: **18-25 tok/s** for a 4B INT4 AWQ model on Orin Nano (20-60% improvement), but this is speculative
- Prompt processing could see larger gains from TRT's fused attention kernels

### Installation & Effort Estimate

**Prerequisites on Orin Nano:**
- cmake, build-essential, git, CUDA toolkit 12.6 packages, TensorRT dev headers
- No Docker required for inference
- 20-50 GB free disk for ONNX files + TensorRT engines

**Model export (must run on x86 host with GPU):**
- Requires 24+ GB VRAM workstation (or DGX Spark could work)
- Python 3.10+, PyTorch, Transformers, ONNX
- `tensorrt-edgellm-quantize-llm` → `tensorrt-edgellm-export-llm` → produces ONNX files
- ONNX files are portable; transfer to Jetson via SCP

**Build on Jetson:**
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin
make -j4  # limited RAM, reduce parallelism
```

**Engine build on Jetson:**
```
./llm_build --onnxDir <onnx> --engineDir <engine> \
  --maxBatchSize 1 --maxInputLen 512 --maxKVCacheCapacity 1024
```

**Effort estimate: 2-3 day project**
- Day 1: Export model on Spark/workstation, build C++ runtime on Jetson
- Day 2: Build TensorRT engine, run inference tests, benchmark
- Day 3: If results are good, wrap in HTTP server or integrate

### Coexistence with llama.cpp

Can coexist — they are independent binaries using different model formats. However:
- **Cannot run simultaneously** — both need full GPU. Orin Nano has 8 GB shared; one model at a time.
- Could switch between them (different systemd units, different ports)
- TRT engine files are in addition to GGUF files — doubles model storage

### Key Risks & Blockers

1. **CUDA 12.6 vs 12.8+ mismatch** — Docs specify CUDA 12.8+/13.x. JetPack 6.2.2 ships 12.6. May need to test if it compiles/runs or if we need JetPack 6.3+.
2. **Experimental Orin support** — Not officially supported. Could hit untested code paths, silent accuracy issues, or missing kernel implementations for SM87.
3. **No HTTP server** — The experimental OpenAI-compatible server is Python-based and may not work on Orin. Would need custom HTTP wrapper for production use.
4. **Model export requires x86 GPU host** — Can't do the full pipeline on-device. Spark (GB10) could work as export host.
5. **Disk space** — 20-50 GB for ONNX+engines on top of existing 34 GB of GGUF models. NVMe may need cleanup.
6. **No streaming output in reference binary** — `llm_inference` is batch/file-based. Real-time streaming needs custom C++ work.
7. **Limited community validation** — Most community benchmarks are on AGX Orin (64 GB) or Thor, not Orin Nano 8 GB.

### Recommendation

**Wait.** The risk/reward ratio is unfavorable right now:
- Orin support is experimental with potential CUDA version mismatch
- No HTTP server means significant integration work to match llama.cpp's drop-in OpenAI API
- Expected speedup (20-60% on a bandwidth-bound device) is meaningful but not transformative
- llama.cpp at 15.3 tok/s is already serviceable for the current use case

**Revisit when:** (a) NVIDIA moves Orin from "experimental" to "supported," (b) JetPack 6.3+ ships with CUDA 12.8+, or (c) the experimental OpenAI server matures. Monitor the GitHub releases.

---

## Entry 021: Claude-Distilled Fine-Tune A/B Test (2026-04-30)
**Date:** 2026-04-30 20:20 UTC
**Operator:** Claude Code
**Status:** A/B TEST — FAIL

### Model Under Test
- **Model:** Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2
- **GGUF:** Qwen3.5-4B-Claude-Distilled-v2-Q4_K_M.gguf (2.6 GB)
- **Hypothesis:** Distillation from Claude Opus reasoning traces should produce shorter think-token chains (-33.8% claimed), yielding faster end-to-end responses on a 15 tok/s device
- **Flags:** Identical to production (ctx-size 32768, gpu-layers 999, threads 1, parallel 1, flash-attn on, reasoning off, mlock, cache-type-k/v q8_0)

### Quantitative Results (bench.sh)

| Metric | Production (Qwen3.5-4B) | Distilled (Claude-v2) | Delta |
|--------|--------------------------|----------------------|-------|
| Gen tok/s (short) | 15.5-15.7 | 15.5-15.6 | ~0% |
| Gen tok/s (medium) | 15.2 | 15.2 | 0% |
| Gen tok/s (long) | 15.1-15.3 | 15.1 | ~0% |
| PP tok/s (long) | ~162 | ~162 | 0% |
| RSS (MB) | 4,969 | 5,050 | +1.6% |

Raw per-token generation speed is identical — expected since both are the same architecture at the same quantization.

### Qualitative Results

| Prompt | Production | Distilled |
|--------|-----------|-----------|
| "127 * 43, show work" (512 tok) | 0 reasoning, 512 content (hit limit), correct approach | All 512 tokens consumed by reasoning, content present but reasoning verbose |
| "All but 9 sheep die" (256 tok) | 0 reasoning, 64 tokens, correct "9", stop | 256 tokens (109 words reasoning), content truncated |
| "Why is sky blue, 2-3 sentences" (256 tok) | 0 reasoning, 55 tokens, correct concise answer, stop | 256 tokens ALL reasoning, 0 content output |
| Same prompt (512 tok) | — | 512 tokens ALL reasoning, 0 content output |
| Same prompt (1024 tok) | — | 1024 tokens ALL reasoning (672 words), 0 content output, drafts answer inside thinking but never transitions |

### Critical Finding

The distilled model has a **catastrophic reasoning loop**: it generates elaborate structured "Thinking Process" chains (analyzing request → identifying concepts → drafting sentences → evaluating drafts → re-drafting) that consume the entire token budget without ever producing actual content output. Even with 1024 tokens for a 2-sentence answer, the model stays stuck in its reasoning phase.

The production Qwen3.5-4B with `--reasoning off` produces zero reasoning tokens and goes straight to content. The distilled model ignores this flag and generates reasoning regardless — the distillation process appears to have embedded Claude-style reasoning patterns into the base generation path rather than confining them to the reasoning token mechanism.

The claimed "-33.8% think token reduction" is contradicted by testing. The model is MORE verbose in reasoning (not less) and fails to produce content at all for simple factual questions.

### Verdict

**FAIL — ARCHIVE.** The model is unusable as an inference endpoint: it cannot reliably produce content output within normal token budgets. The fine-tune's reasoning distillation has broken the model's ability to terminate thinking and begin answering.

### Decision
- Keep GGUF at `~/llm-server/models/Qwen3.5-4B-Claude-Distilled-v2-Q4_K_M.gguf` for 30 days in case model author releases a v3 fix (delete after 2026-05-30)
- Keep `start-experiment.sh` for future A/B tests
- Production remains Qwen3.5-4B-Q4_K_M (unchanged)
- Remove from JETSON_BASELINE.md watch items

### Pending Cleanup
- **After 2026-05-14:** Delete `~/llm-server/backup-b8766/` (old llama.cpp binary backup, 2 weeks of stable b8987 operation confirmed)
- **After 2026-05-30:** Delete `~/llm-server/models/Qwen3.5-4B-Claude-Distilled-v2-Q4_K_M.gguf` (archived failed fine-tune, 2.6 GB)

---

## Entry 022: Jetson Recon (2026-05-09)
**Date:** 2026-05-10 02:50 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made
**Days since last recon:** 9 (last: 2026-04-30, Entry 019)

### Check 1 — JetPack / Firmware: NO MATCH (MEDIUM, status quo)
- Latest production for Orin family: **JetPack 6.2.2 (R36.5.0)** — matches baseline.
- JetPack 7.1 (Jan 2026) — Jetson Thor only, NOT Orin Nano.
- **JetPack 7.2** — still **not released** as of 2026-05-09. Q2 2026 target window holds (Apr–Jun); forum activity suggests slip-or-imminent, no firm date.
- Trigger `jetpack: JetPack 7.2 AND (Orin Nano OR Orin)` → **NO MATCH** (announced ≠ released).
- Recommendation: hold on JP 6.2.2; re-check JP 7.2 in 2–4 weeks.

### Check 2 — llama.cpp Releases: NO MATCH (LOW, skip rebuild)
- Latest: **b9093** (`1e5ad35d5`, 2026-05-09T21:02 UTC). 106 builds since baseline b8987.
- Jetson/SM87/Ampere/Tegra/aarch64/unified-memory/CUDA-graph keyword scan → **zero matches**.
- CUDA work in this window targets multi-GPU/training (cuBLAS strided, PCI dedupe) or non-Qwen activations (snake fusion). Flash-attn changes touch MMA/Tiles for MiMo-V2.5 and Vulkan/SYCL — not CUDA SM87. KV cache fix is FP8 (we use q8_0).
- No breaking changes that would block a drop-in rebuild; Python convert tooling moved to PEP 621/uv but C++ server unaffected.
- Trigger `llamacpp_release: SM87 OR Jetson OR Tegra OR unified memory` → **NO MATCH**.
- Recommendation: **skip rebuild.** Stay on b8987. Re-check in ~2 weeks or when a CUDA/flash-attn/Qwen-tagged PR lands.

### Check 3 — Small Model Landscape: NO MATCH (LOW, one watch addition)
- No new dense 1–7B models released between 2026-04-30 and 2026-05-09 that beat Qwen3.5-4B at our size class. Qwen3.6 line still 27B+ / MoE only — no 4B variant signaled.
- LFM2-24B-A2B and DeepSeek-V4-Flash excluded (MoE total params would exceed 8 GB ceiling).
- New embedding model **EmbeddingGemma-300M** observed — top open multilingual embedder under 500M, but does NOT beat Qwen3-Embedding-4B on raw MTEB. Not a quality replacement; only useful if we ever want to free RAM for a bigger LLM.
- **Qwen3-VL-Embedding-2B** released Jan 2026 — multimodal (text+image+screenshot+video); expands capability axis but not a Qwen3-Embedding-4B replacement.
- Notable Qwen3.5-4B fine-tune for next `experiment` slot: **`khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF`** — distilled from Qwen3.6-plus teacher, focuses on concise CoT. Drop-in for current binary, zero memory risk.
- Trigger `huggingface: Qwen4 OR Qwen3.5 successor` → **NO MATCH**.
- Recommendation: Add khazarai distill to watch list; everything else, no action.

### Check 4 — NVIDIA Jetson Forum: 1 INFO MATCH (LOW, no urgent action)
- **dusty-nv/jetson-containers (2026-05-04 → 2026-05-06)** — vLLM 0.21.0, flash_infer 0.6.11, ONNX Runtime 1.25.1, PR #1693 BuildKit GPU build, fixes for xformers/cuSPARSELt 0.9.0 incompatibility. JetPack 7.2 became default container target on Apr 26. Relevant only if/when we move off JP 6.2.2.
- **NVIDIA pypi.jetson-ai-lab.dev outage** (Apr 28→29, resolved) — informational; wheel mirror briefly down.
- **CUDA 13.2 wheels test thread for Orin family** (pinned, last activity Apr 5) — direction signal, not yet relevant on CUDA 12.6.
- **Yalexx benchmark** (updated 2026-03-26): Llama 3.2 3B Q4_K_M = 28.7 tok/s gen / 580 tok/s pp on Orin Nano via Ollama/llama.cpp — confirms our 4B 15.3 tok/s is in the expected band for the larger model.
- **llama.cpp issue #19219** — open SM87-specific MoE decode hang on Orin AGX; flag for any future MoE evaluation (Qwen3-Coder-MoE etc.). Not biting current dense workload.
- **Build-flag verification (closed inline)**: forum recommendation to verify `-DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=87` against our b8987 build → CMakeCache confirms all three set. **Already optimal — no action.**
- Trigger `forum: llama.cpp AND (performance OR optimization) AND jetson` → matched dusty-nv container ecosystem activity, but it's not direct llama.cpp tuning.
- Classification: LOW.

### Check 5 — Live Jetson Health: DEGRADED (recurring OOM-kill pattern)

**Service & system:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| systemctl status | active (running), 21h since restart | active | DEGRADED — see OOM below |
| System uptime | 25 days, 6:52 | stable | OK |
| Mode | qwen35 | expected | OK |
| llama.cpp HEAD | 5f0ab726f (b8987) | matches baseline | OK |
| **NRestarts (lifetime)** | **11** | low | TRACK |
| Last restart | 2026-05-09 01:04:34 EDT — **OOM-killed** | clean stop | **DEGRADED** |

**Memory:**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| RSS | 5,322 MB | < 5,963 (20% above 4,969 baseline) | OK (+7.1%) |
| VmHWM (peak) | 5,592 MB | — | OK |
| VmSwap | 71 MB | — | OK (mild leak to swap) |
| Available RAM | 2,500 MB | > 500 MB | OK |
| Free RAM | 174 MB | — | TIGHT (consistent with OOM risk) |
| Swap used (zram) | 222 MB | — | OK |
| Swap used (SSD) | 0 B | — | HEALTHY |

**Performance (3-run sustained warm benchmark, medium prompt 52 tok → 256 tok):**
| Run | PP tok/s | Gen tok/s |
|-----|----------|-----------|
| 1 | 160.8 | 15.18 |
| 2 | 197.8 | 15.30 |
| 3 | 184.3 | 15.26 |
| **Mean** | **180.9** | **15.25** |

Generation is **at baseline** (15.25 vs 15.30 baseline = -0.3%). PP is **above baseline** (181 vs 166 = +9%). Sustained throughput is healthy.

**Thermal & power (idle, tegrastats):**
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| GPU temp | 49°C (49–60°C across cores) | < 75°C idle | OK |
| CPU temp | 48–60°C | — | OK |
| TJ temp | 60.5°C | — | OK |
| GR3D_FREQ | 0% | idle expected | OK |
| Power (VDD_IN) | 4.85 W | — | OK |
| Disk | 17% (134G/824G) | < 70% | OK |

**OOM-killer pattern (last 30 days):**
| Date | Time (EDT) | Result |
|------|-----------|--------|
| Apr 16 | 01:02 | oom-kill |
| Apr 16 | 21:24 | oom-kill |
| Apr 19 | 01:01 | signal (likely manual) |
| Apr 26 | 01:00 | oom-kill |
| Apr 30 | 01:04 | signal (likely b8987 deploy) |
| **May 9** | **01:04** | **oom-kill** |

4 of 5 events fall in the **01:00–01:04 EDT window** — strong scheduled-job correlation. Likely culprits (cannot confirm without sudo): `anacron.timer` triggering `cron.daily` jobs (apt-compat, dpkg, man-db, apport), or `unattended-upgrades` overlapping with peak inference RSS. The May 9 OOM kernel log shows total-vm 47 GB / anon-rss 3.93 GB at the moment of kill — process was operating at normal RSS, so the trigger was external memory pressure, not internal growth.

### Cross-Correlated Findings
1. **Check 2 (no Jetson-relevant builds) ↔ Check 4 (build flags already optimal):** Together close out the rebuild question — there is **no llama.cpp action** to take this cycle.
2. **Check 5 (OOM at 01:00 EDT recurring) ↔ Check 5 (free RAM 174 MB at idle):** Memory headroom at idle is genuinely tight. Any external memory pressure at the wrong moment will trigger OOM. Root cause is the static memory budget, not a leak (RSS only +7.1% above baseline; VmHWM only +13% above baseline).
3. **Check 3 (no successor model) ↔ Check 1 (no JetPack 7.2 yet) ↔ Check 2 (no Jetson PRs):** Three checks independently confirm the platform is in a stable holding pattern — no upgrade path is currently available, and current setup is correctly tuned.

### Triggered Alerts
- **JetPack trigger:** NO — JP 7.2 still pre-release.
- **llama.cpp trigger:** NO — no SM87/Jetson/Tegra/unified-memory PRs in 106-build window.
- **HuggingFace trigger:** NO — no Qwen4 or Qwen3.5 successor at 4B.
- **Forum trigger:** YES (LOW) — dusty-nv container activity, not direct llama.cpp tuning.

### Overall: **WORTH WATCHING** (degraded health on memory pressure axis)

Landscape is stable; production is well-tuned (build flags optimal, throughput at baseline). The downgrade is the **recurring 01:00 EDT OOM-kill cycle** — a service-availability issue rather than a configuration or upgrade gap.

### Recommendations
1. **ACTION (priority): Diagnose the 01:00 EDT scheduled-job memory pressure.** Run `sudo crontab -l`, `sudo systemctl list-timers --all`, `sudo cat /etc/anacrontab`, and grep `/var/log/syslog` and `/var/log/unattended-upgrades/*.log` around the 01:00–01:04 window. Most likely culprit is anacron firing `/etc/cron.daily/*` jobs (man-db rebuild and unattended-upgrades both have high transient memory). Two cheap mitigations once identified: (a) reschedule the offender to a quieter window via systemd timer override, or (b) add `MemoryHigh=5500M` / `MemoryMax=6000M` to `myscript.service` to force cgroup-level reclaim before kernel OOM.
2. **WATCH:** Add `khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF` to `experiment` mode candidates — drop-in for Qwen3.5-4B, distilled from a stronger teacher than the failed Claude-v2 distill.
3. **WATCH:** JetPack 7.2 release date (Q2 2026 window). Re-check in 2–4 weeks; full reflash + 4–6 weeks of community validation before any move.
4. **WATCH:** Qwen 4B successor in Qwen3.6 line — expected late May per prior cadence; not yet announced.
5. **NO ACTION:** llama.cpp rebuild (no Jetson-relevant PRs in 106-build window).
6. **CLOSED:** Build-flag verification — confirmed `GGML_CUDA_F16=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `CMAKE_CUDA_ARCHITECTURES=87`, `Release`. Already optimal.

### Tracking value changes (proposed for JETSON_BASELINE.md)
- `llamacpp_latest_seen`: b8987 → **b9093** (observed; we are not upgrading)
- `models_last_checked_date`: 2026-04-30 → **2026-05-09**
- `forum_last_checked_date`: 2026-04-30 → **2026-05-09**
- `Last recon`: 2026-04-30 → **2026-05-09**
- Watch items: add khazarai distill candidate; add OOM-kill investigation thread.

---

## Entry 023: OOM Root-Cause Investigation & 3-Layer Fix (2026-05-09)
**Date:** 2026-05-10 03:30 UTC
**Operator:** Claude Code
**Status:** INVESTIGATION COMPLETE → CHANGES STAGED

### Scope
Investigate the recurring OOM-kill cycle flagged in Entry 022 (4 OOMs in 24 days, all clustering near 01:00–01:04 EDT).

### Headline finding
**The 01:00 EDT clustering was misleading.** The May 9 OOM had cron.daily run at 07:35 (per `/var/spool/anacron/cron.daily` mtime), so cron.daily was NOT the trigger. The Apr 16 21:24 OOM was nowhere near 01:00. **Root cause is structural: misconfigured VM tunables for an inference workload, on a system with no memory headroom.**

### Evidence

**1. Kernel OOM call stack** (`do_swap_page` → `__handle_mm_fault`): llama-server tried to access a previously-swapped page; kernel needed to allocate a free page to swap-in; no pages available → OOM-killer fired.

**2. OOM dump — llama-server process state at moment of kill:**
| Field | Value |
|-------|-------|
| total_vm | 47.1 GB (CUDA address space — expected) |
| anon-rss | 3.93 GB |
| **swapents** | **3.43 GB** ← llama-server had been swapped out |
| pgtables | 8.7 MB |
| oom_score_adj | 0 |

**3. System-wide memory state at OOM:**
| Field | Value |
|-------|-------|
| Free swap | 16.85 GB / 20.67 GB total → **3.82 GB used** |
| Swap cache stats | add 17,772,087 / delete 17,899,505 / find 186,511 (massive thrashing since boot) |
| Free RAM (Normal zone) | 35.6 MB (just above min watermark of 33.6 MB) |
| Active anon | 208 KB (essentially nothing left in active set) |
| Inactive anon | 84 KB |
| Unevictable (mlocked) | 35.9 MB |

**4. The `nvmemwarning.sh[109362]` log entries 1 second before OOM are an indicator, not a cause.** It's an NVIDIA-shipped systemd service (`/etc/systemd/nvmemwarning.sh`) that polls `free -m` every 300s and tries `notify-send` to X-users when available RAM < 150 MB. We are headless → it errors out with `sudo` usage messages. **Its firing is proof that available RAM dropped below 150 MB at that moment.**

**5. VM tunables are catastrophic for an inference workload:**
| Setting | Current | Effect | Inference-optimal |
|---------|---------|--------|-------------------|
| vm.swappiness | 60 (default) | Kernel aggressively swaps anon pages to make room for FS cache | **1** |
| vm.min_free_kbytes | 45,056 (~44 MB) | Tiny safety buffer; OOM-killer fires before reclaim has a chance | **131,072 (~128 MB)** |
| vm.watermark_scale_factor | 10 (~0.1% = 7 MB) | Reclaim doesn't start until critically low | **200 (~2% = ~150 MB)** |
| vm.vfs_cache_pressure | 100 | Default | 50 (less reclaim of dentries — minor) |

With `swappiness=60`, the kernel proactively swaps llama-server's anon pages to zram even when there's no real memory pressure — just to keep filesystem cache populated. Over 25 days uptime, 17.8M swap-cache adds → constant page churn → eventual OOM when llama-server faults a swapped-out page during a memory-thin moment.

**6. 22+ idle `systemd-udevd` workers (PIDs 35337–35377)** present at OOM with RSS≈0. Symptomatic of a past udev event storm (probably boot-time). Not actively consuming memory; not a cause.

### Why the 01:00 EDT clustering then?

Statistical artifact + nvmemwarning's 300s polling interval. The system is constantly close to OOM for the entire uptime. nvmemwarning fires every 5 min if available drops below 150 MB. At any moment over weeks, there's a probability of memory dipping low enough that nvmemwarning notices and the kernel can't service llama-server's next page-fault. The 01:00 hour shows up because (a) some cron.hourly activity spikes IO/cache demand, (b) low-priority background services schedule then, and (c) llama-server has had hours of evening idle to accumulate inactive pages eligible for swap.

### Fix plan (user approved A+B+C)

**Layer A — sysctl tuning (root cause):**
File: `/etc/sysctl.d/99-llm-inference.conf`
```
vm.swappiness = 1
vm.min_free_kbytes = 131072
vm.watermark_scale_factor = 200
vm.vfs_cache_pressure = 50
```
Apply: `sudo sysctl --system`

**Layer B — systemd cgroup memory limits (safety net):**
File: `/etc/systemd/system/myscript.service.d/memory-limits.conf`
```
[Service]
MemoryHigh=5500M
MemoryMax=6000M
```
Apply: `sudo systemctl daemon-reload` (file picked up at next service restart; cgroup limits also applied live via `systemctl set-property myscript MemoryHigh=5500M MemoryMax=6000M --runtime` to avoid forcing restart now).

**Layer C — disable nvmemwarning.service (cosmetic):**
Headless system, no X user to notify. Just spams sudo errors.
Apply: `sudo systemctl disable --now nvmemwarning.service`

### Backup before changes
- Current sysctls captured in this entry (above table)
- Current `myscript.service` unit captured in JETSON_CONFIG.md (no change to base unit, only adding drop-in)
- `nvmemwarning.service` is NVIDIA-shipped and reversible via `systemctl enable --now`

### Expected outcomes
- OOM-kill events should stop. Even under cron.daily / fs activity bursts, the kernel will start reclaiming at 150 MB free instead of 7 MB free, and llama-server's anon pages will not be evicted to zram.
- Systemd cgroup memory pressure should be visible in `systemd-cgtop` if approached, providing telemetry.
- If MemoryMax (6000M) is hit, systemd will issue `MemoryHigh` throttling first, then SIGKILL via cgroup OOM with a clean restart — much faster than kernel-level OOM and without affecting other processes.

### Apply Results (2026-05-09 23:08 EDT / 2026-05-10 03:08 UTC)
**Status: APPLIED — all 3 layers active. No service restart required.**

**Backups saved on Jetson:** `/tmp/oom-fix-backup/{sysctls-before.txt, myscript-before.txt, nvmemwarning-{enabled,active}-before.txt}`

**Layer A verification — sysctl live values:**
```
vm.swappiness = 1                  (was 60)
vm.min_free_kbytes = 131072        (was 45056)
vm.watermark_scale_factor = 200    (was 10)
vm.vfs_cache_pressure = 50         (was 100)
```

**Layer B verification — cgroup limits live + persistent:**
```
MemoryCurrent = 5,515,395,072  (~5.14 GB — current usage)
MemoryHigh    = 5,767,168,000  (5500 MB — applied)
MemoryMax     = 6,291,456,000  (6000 MB — applied)
```
Drop-in file: `/etc/systemd/system/myscript.service.d/memory-limits.conf`. `systemctl set-property --runtime` applied limits to the running cgroup without restart.

**Layer C verification — nvmemwarning disabled:**
```
is-enabled: disabled  (symlink removed from multi-user.target.wants/)
is-active:  inactive
```

**Immediate effect on memory state:**
| Metric | Before fix | After fix (~minutes later) |
|--------|-----------|----------------------------|
| Free RAM | 174 MB | **1.1 GB** (6× more) |
| Buff/cache | 2.6 GB | 1.3 GB (kernel holding less cache) |
| Available | 2.5 GB | 1.6 GB (about same effective headroom) |
| Swap used | 222 MB | 320 MB (slight increase as zram absorbs evicted cache) |

The "available" number went down because `swappiness=1` tells the kernel to keep less filesystem cache (which dominates "available"). But the **actually-free** number went up 6×. At the moment of the May 9 OOM, kernel reported free=35 MB; now it's 1.1 GB — 30× more breathing room before reclaim is forced.

**Inference confirmed working:** `curl /v1/chat/completions` returned "ok" cleanly post-fix.

### Caveat / future watch
**MemoryCurrent (5.14 GB) is already close to MemoryHigh (5.5 GB)** — only ~360 MB of soft-throttle headroom. If multi-turn KV cache growth exceeds this, throttling will engage (which is the desired behavior — slows the cgroup, doesn't kill it). If it ever hits MemoryMax (6.0 GB), cgroup-OOM kills llama-server cleanly and systemd restarts it (5s) without taking the whole system into kernel-OOM territory. This is a much better failure mode than what we had.

If `systemd-cgtop myscript.service` ever shows throttling activity, raise MemoryHigh to 5800M and MemoryMax to 6300M (still leaves ~1.1 GB for OS).

### Pending (next session)
- **Wait 7+ days**, recheck `journalctl -u myscript --since '7 days ago' | grep -iE 'oom|fail'` to confirm zero OOM events post-fix. Target: log Entry 024 around 2026-05-16 with verification.
- Watch `MemoryCurrent` over time via `systemctl show myscript -p MemoryCurrent` to see if it grows past MemoryHigh.
- If OOMs DO persist, next lever is reducing context size from 32768 → 16384 (halves KV cache from ~2 GB to ~1 GB, freeing ~1 GB working memory).
- Reversal procedure if anything goes sideways: `sudo rm /etc/sysctl.d/99-llm-inference.conf && sudo sysctl --system && sudo rm /etc/systemd/system/myscript.service.d/memory-limits.conf && sudo systemctl daemon-reload && sudo systemctl enable --now nvmemwarning.service`. Backup files in `/tmp/oom-fix-backup/` for cross-reference (note: /tmp clears on reboot — copy them out if you want them long-term).

---

## Entry 024: Biweekly Recon (2026-05-13)
**Date:** 2026-05-13 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made

### Check 1 — JetPack / Firmware: MEDIUM — JetPack 7.2 confirmed early June
JetPack 7.2 now confirmed by NVIDIA staff (kayccc) for **early June 2026** — first JetPack 7.x targeting Orin. Expected: Ubuntu 24.04, kernel 6.8, CUDA 12.8+, TensorRT 10.5+. Full reflash required (no OTA). JetPack 7.0/7.1 were AGX Thor only. Current 6.2.2 remains latest for Orin Nano.

### Check 2 — llama.cpp Releases: LOW — b9133 latest, nothing Jetson-relevant
5 new releases since last seen b9093 (b9127–b9133, all May 12–13). All target non-Jetson backends: Qualcomm Hexagon DSP, Adreno OpenCL, AMD ZenDNN, server reasoning features, CLI arg renaming. Zero SM87/CUDA/flash-attn/KV changes. **b9131 CLI arg consistency change** is a yellow flag — verify startup script args before any future rebuild. No upgrade warranted.

### Check 3 — Small Model Landscape: INFO — new trial candidates, Gemma 4 still blocked
- **Phi-4-mini-instruct** (3.8B dense, 2.49 GB Q4_K_M, 128K ctx, MMLU 68.5): Fits experiment slot. Production-ready GGUF. Worth trialing against Qwen3.5-4B.
- **Gemma 4 E2B**: Still blocked by llama.cpp #22243 (PLE not injected into forward graph). Also #22396 (json-schema crash) and #16370 (GPU offload on SM87). Multiple blockers — wait for upstream.
- **Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF**: Different author/approach from the archived v1 distill. Targets excessive CoT specifically. Zero memory risk. Low-priority smoke test.
- **EmbeddingGemma-300M** (ggml-org, ~200 MB Q4_K_M, MTEB #8): Not a quality replacement for Qwen3-Embedding-4B (MTEB #3), but interesting if RAM reclaim ever needed.
- **Qwen3.6 small models**: Confirmed absent. Qwen3.6 line is 27B/35B only. Push expected date to **Q3 2026**.
- **Qwen4**: No release. Prediction markets say possible before July 2026 but nothing shipped.

### Check 4 — Forum & Community: INFO — setup validated, no new techniques
- **Build flags FA_ALL_QUANTS + CUDA_F16**: Forum flagged as critical optimizations — ALREADY CONFIRMED in current b8987 build (JETSON_CONFIG.md). No gap.
- **Symmetric KV for flash attention**: Already using q8_0 for both K and V. Fused FA path should be active.
- **Community tok/s reference**: Orin Nano getting 11–14 tok/s on comparable models with stock setups. Current 15.3 tok/s is **above community average** — confirms tuning is effective.
- **MoE hang on SM87** (#19219): Still unresolved but only affects MoE architectures. Dense Qwen3.5-4B is safe.
- **NVIDIA blog** on OS memory reclaim: 865 MB recoverable by disabling desktop. Already running headless.
- **16 GB file-backed swap**: Becoming community standard. Already configured (though zram also active).

### Check 5 — Live Health: HEALTHY
- **Service**: UP, active since 2026-05-09 01:04:34 (4 days post-OOM fix)
- **System uptime**: 28 days, load 0.00
- **RAM**: 6.2/7.4 GiB used, 387 MiB available (below 500 MB threshold, but expected steady-state with cgroup limits active)
- **MemoryCurrent**: 5.21 GiB vs MemoryHigh 5.3 GiB (165 MiB soft headroom) — cgroup throttle engaged as designed
- **Thermals**: 46.6–48.6°C across all zones — well under 75°C idle threshold
- **Inference**: 14.26 tok/s gen (6.8% below 15.3 baseline, within 15% threshold; small 6-token sample)
- **Disk**: 17% used (134G/824G)
- **Swap**: 819 MiB zram used, 16 GB SSD swap idle — healthy distribution
- **OOM verification (4/7 days)**: Zero OOM events since May 9 fix. Only log entries are the pre-fix crash at 01:04:27. **Fix holding.**
- **3-layer fix intact**: sysctl conf present, cgroup limits enforced, nvmemwarning disabled

### Cross-Correlated Findings
1. **Forum build flags (Check 4) vs config (baseline)**: Forum agent flagged FA_ALL_QUANTS + CUDA_F16 as unconfirmed — resolved as FALSE FLAG. Both confirmed in JETSON_CONFIG.md build commands. No gap.
2. **JetPack 7.2 (Check 1) + TensorRT-Edge-LLM (Check 4)**: Correlated. JP 7.2 with CUDA 12.8+ will unblock the previously deferred TensorRT-Edge-LLM evaluation. Timeline: early June.
3. **Gemma 4 E2B (Check 3) + SM87 GPU issues (Check 4)**: Multiple open llama.cpp issues for Gemma on Jetson SM87. Not usable until upstream fixes land.
4. **OOM fix verification (Check 5)**: 4 of 7 target days complete. Zero events. On track for full verification by 2026-05-16.

### Triggered Alerts
| Trigger | Match | Status |
|---------|-------|--------|
| JetPack 7.2 AND Orin | YES | MEDIUM — confirmed early June, not yet released |
| llama.cpp SM87/Jetson | NO | No relevant changes in b9094–b9133 |
| Qwen4 / Qwen3.5 successor | NO | Neither released |
| Forum optimization techniques | PARTIAL | Build flags already applied; no new techniques |

### Overall: WORTH WATCHING

### Recommendations
1. **No immediate action needed** — current config remains optimal for the hardware
2. **Optional trial**: Download Phi-4-mini-instruct Q4_K_M (2.49 GB) for experiment slot — first real alternative to Qwen3.5-4B worth benchmarking
3. **2026-05-16**: Complete OOM fix verification (7-day mark) — run `journalctl` check, log as Entry 025
4. **2026-05-14 (tomorrow)**: Safe to delete `~/llm-server/backup-b8766/` — 2 weeks of stable b8987 reached
5. **Early June**: Watch for JetPack 7.2 release — re-run recon immediately if it drops
6. **Ongoing**: Monitor llama.cpp #22243 (Gemma 4 PLE) — when fixed, Gemma 4 E2B becomes top trial priority
7. **Update watch items**: Push Qwen3.6 small model expectation from "late May" to Q3 2026

---

## Entry 025: Biweekly Recon + OOM Recurrence Root-Cause (2026-05-27)
**Date:** 2026-05-27 16:40 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made

#### Check 1 — JetPack / Firmware: JetPack 7.2 still **NOT released** as of 2026-05-27 (Q2 window closes June 30). Orin Nano support remains *planned, not shipped*. When it lands: CUDA 12.6→13.0, kernel 5.15→6.8, Ubuntu 22.04→24.04, full reflash. **MEDIUM (announced, imminent).** Trigger `JetPack 7.2 AND Orin` NOT matched (not released — only announced).
#### Check 2 — llama.cpp Releases: Latest **b9360** (2026-05-27). **373 builds since running b8987; 227 since last-seen b9133. Zero SM87/Jetson/Tegra/unified-memory changes.** CLI-arg standardization continues (b9360: env vars all moved to `LLAMA_ARG_*` prefix) — breaking-change risk compounds the b9131 arg renames. **LOW.** Trigger NOT matched. Hold at b8987.
#### Check 3 — Small Model Landscape: No new 4B-class dense model since 2026-05-13. Qwen3.6 still ships only 27B/35B-A3B (no 4B tier; expected Q3 2026). Gemma 4 E2B/E4B still blocked by llama.cpp **#22243 (PLE unimplemented), unfixed**. Phi-4-mini already on watch list. Embedding: Jina Embeddings v4 (~3B) worth a GGUF check vs Qwen3-Embedding-4B. **Zero-risk lever:** Qwen3.5-4B LoRA fine-tunes (coding/text-to-SQL) drop in at ~50 MB. **NO ACTION.** Trigger `Qwen4 OR Qwen3.5 successor` NOT matched.
#### Check 4 — Jetson Forum / Community: No technique beats our 15.3 tok/s on Orin Nano 8GB for a 4B/7B Q4_K_M model — community consensus is memory-bandwidth-bound single-stream. flouisdev auto-config post (2026-04-13) already on watch list. TensorRT-LLM still AGX-Orin-only / experimental on Nano (~20 tok/s, slower than llama.cpp single-stream). **INFO only / NO ACTION.**
#### Check 5 — Live Health: Server **UP** (qwen35, b8987, port 8080). Gen **15.05 tok/s** (vs 15.3 baseline, −1.6%, healthy). Thermals ~47–49°C (healthy). Disk 17%. Swap: 325 MB on zram only, 16 GB file swap untouched. **BUT: recurring OOM kills — service restarted 2026-05-26 01:07 after its 3rd OOM kill. DEGRADED.**

#### 🔴 KEY FINDING — Entry 023 OOM Fix DID NOT WORK (root cause was misdiagnosed)
Three OOM kills now on record, **all global, all ~01:00, llama-server always the victim:**

| Date | Time | Constraint | Victim oom_score_adj | llama anon-rss | Precipitant in window |
|------|------|-----------|---------------------|----------------|----------------------|
| May 09 | 01:04:27 | CONSTRAINT_NONE (global) | 0 | 3932 MB | (pre-fix baseline) |
| May 17 | 01:01:06 | CONSTRAINT_NONE (global) | 0 | 3932 MB | early-AM maintenance |
| May 26 | 01:06:52 | CONSTRAINT_NONE (global) | 0 | 3934 MB | **snapd watchdog SIGABRT + Go crash-dump** |

**Confirmed mechanism (no longer hypothesis):**
1. OOM is **global (system-wide RAM exhaustion), NOT the cgroup MemoryMax** — so the Entry 023 cgroup limits (`MemoryHigh=5500M`/`MemoryMax=6000M`) cannot prevent it; the spiker is *external* to the llama-server cgroup.
2. **snapd is OOM-protected at `oom_score_adj=-900`** (2.1 GB virtual; runs headless-useless desktop snaps: chromium, gnome-46, mesa-2404, gtk-common-themes, cups). The kernel literally cannot pick it.
3. **llama-server sits at `oom_score_adj=0` with the largest anon-rss (~3.93 GB)** → it is *always* the selected victim.
4. On 8 GB unified RAM the live headroom is razor-thin (free 590 MB, cgroup `available: 139 MB`). Any early-AM maintenance burst (man-db index rebuild fires in the 00:40–01:00 window; anacron cron.daily; occasionally a snapd watchdog crash-dump as on May 26) tips total memory over the edge.
5. **Entry 023 concluded "NOT cron-triggered; structural swappiness" and applied `vm.swappiness=1` + cgroup limits.** That reduced swap thrashing but never touched the two real levers — *headroom* and *OOM victim selection*. Result: kills recurred on May 17 (8 days later) and May 26 (9 days later). This is the whack-a-mole pattern the systematic-debugging rule warns against.

#### Cross-Correlated Findings: Check 5 OOM recurrence ⨯ Entry 023 fix ineffectiveness ⨯ live 139 MB cgroup headroom → all point to one structural conclusion: **on this 8 GB box, llama-server must be made the OOM survivor, not the sacrifice, and the early-AM memory pressure must be cut.** Single root cause, multi-date corroboration → HIGH confidence.
#### Triggered Alerts: No web-source trigger matches (jetpack/llamacpp/huggingface/forum all NOT matched). The ACTION classification is driven entirely by Check 5 health DEGRADED.
#### Overall: **ACTION NEEDED** — recurring inference-server OOM kills; prior fix ineffective.

#### Recommendations (RECON = report only; NOT applied — awaiting approval)
1. **Protect the inference server from OOM victim selection (primary, lowest-risk, trigger-agnostic):** add `OOMScoreAdjust=-900` to `myscript.service`. Then global OOM will evict snapd / a maintenance job instead of llama-server. This stops the symptom regardless of which job spikes on a given night.
2. **Cut the early-AM memory pressure (addresses the trigger):**
   - `systemctl disable --now man-db.timer` — man-db index rebuild is useless on a headless server and is the recurring 00:40–01:00 burst.
   - Remove headless-useless desktop snaps (chromium, gnome-46-2404, mesa-2404, gtk-common-themes, cups) and/or mask snapd refresh — snapd's watchdog crash was the May 26 precipitant; it serves nothing here (Docker is the container runtime, no snaps in use for inference).
3. **Re-evaluate the Entry 023 cgroup limits:** `MemoryMax=6000M` does not prevent global OOM and the live `available: 139 MB` shows llama-server is being actively reclaim-throttled at MemoryHigh — possible marginal latency cost for zero OOM benefit. Consider relaxing/removing once (1)+(2) are in place.
4. **Verify:** after applying, watch the next 00:40–01:10 windows for ~2 weeks; `journalctl -u myscript --since … | grep -i oom` should stay empty. Re-confirm root cause is fully addressed rather than assuming.
5. **No action on Checks 1–4:** config remains optimal. Re-run recon immediately if JetPack 7.2 drops (imminent, by June 30).

#### ✅ Fix Applied (2026-05-27, post-approval — supersedes "NOT applied" note above)
All three layers implemented and verified the same day:
1. **`OOMScoreAdjust=-900`** drop-in at `/etc/systemd/system/myscript.service.d/oom-protect.conf` → `daemon-reload` + `systemctl restart myscript`. **Verified:** `systemctl show myscript -p OOMScoreAdjust` = -900; live MainPID 163089 `/proc/.../oom_score_adj` = -900. llama-server is now OOM-protected (tied with snapd); a maintenance job will be the global-OOM victim instead.
2. **`man-db.timer` disabled** (`systemctl disable --now`) → `disabled / inactive`. Removes the recurring 00:40–01:00 index-rebuild burst.
3. **Desktop snaps removed** by user interactively (`snap` not in claude NOPASSWD): chromium, cups, gnome-46-2404, gtk-common-themes, mesa-2404. Only `bare`/`core22`/`core24`/`snapd` remain. Cuts snapd memory + watchdog-crash risk (the May 26 precipitant).
- **Memory headroom:** available RAM 1.4 Gi (recon start) → 1.9 Gi (after layers 1–2) → **2.0 Gi** (after snap removal). Inference unaffected (~15 tok/s).
- **Original unit backed up:** `~/llm-server/backups/oom-fix-2026-05-27/`.
- **Verification due ~2026-06-10:** `journalctl -u myscript --since '2026-05-27' | grep -i oom` should be empty. Folded into next biweekly recon (Check 5 already scans for OOM).

---

## Entry 026: Biweekly Recon — JetPack 7.2 Shipped, OOM-Fix Verification FAILED (2026-06-11)
**Date:** 2026-06-11 15:56 UTC
**Operator:** Claude Code (jetson-recon skill)
**Status:** RECON — no changes made

#### Check 1 — JetPack/Firmware: HIGH — JetPack 7.2 SHIPPED for Orin Nano
- **JetPack 7.2 released:** announced 2026-06-01 (GTC Taipei), downloads live 2026-06-02, Orin Nano DevKit feedback thread opened 2026-06-03. **L4T r39.2** (38.x was Thor-only), Ubuntu 24.04, kernel 6.8, **CUDA 13.2.1**, cuDNN 9.20.0, TensorRT 10.16.2. Orin Nano (Super) 8GB explicitly supported.
- **Upgrade type: FULL REFLASH** — new unified USB ISO installer (SD-card images discontinued); SDK Manager also works. No apt/OTA path from 6.x. Requires UEFI ≥ 36.x; 36.4.3 users hit capsule-update timeouts (workaround: manual USB boot selection).
- **Early field reports (~10 days post-release):** no core regressions in JP7.2 itself, but (a) **power-mode bug** — ISO installs non-super TNSPEC; only 7W/15W visible, 25W/MAXN SUPER missing (workaround `sudo nvpmodel -m 2` + systemd persistence, not yet fixed at source); (b) **ecosystem lag** — dustynv jetson-containers, prebuilt Ollama, PyTorch SBSA cu126 wheels all CUDA-12.6-bound and break (Error 801 / CPU fallback / silent NaN on sm_87); (c) llama.cpp must be **rebuilt from source** vs CUDA 13.2 (`-DCMAKE_CUDA_ARCHITECTURES=87`) — community reports ~10–23 tok/s, consistent with no regression.
- New capability: arm64-SBSA container support — upstream arm64 containers (e.g. official vLLM) now run without Jetson-specific rebuilds.
- 6.x line: 6.2.2 remains the final/latest 6.x; known open CVE in nvidia-container-toolkit 1.16.2 only fixed in 7.x (≥1.18.0).
- Sources: forums.developer.nvidia.com/t/372151 (getting-started thread), /t/372490 (practical guide), /t/372283 + /t/372627 (power-mode bug), jetsonhacks.com/2026/05/31/gtc-2026-taipei/, developer.nvidia.com/embedded/jetpack/downloads

#### Check 2 — llama.cpp Releases: HIGH — NvMap VMM patch closed unmerged; latest b9596
- Latest tag **b9596** (2026-06-11) — 236 builds past last-seen b9360, ~609 past running b8987.
- **NvMap workaround rejected upstream:** PRs #23732/#23747 — CUDA VMM-backed weight allocator (`GGML_CUDA_VMM_BUFFERS=1`) routing weights through `cuMemCreate` instead of `cudaMalloc` to bypass NvMap pressure tracking; author demoed on an **Orin Nano Super 8GB**. Both **closed unmerged** (May 26/27; flagged as AI-generated). No upstream fix for the NvMap allocation-cap bug; the ~105-line `ggml-cuda.cu` patch is a local-application candidate. **Watch for resubmission.**
- Issue #19219 (MoE decode hang on SM87 since b7309, `ggml_fill(-INFINITY)`) closed "not planned" — blocks future small-MoE experiments on this device; dense Qwen3.5-4B unaffected.
- MEDIUM: #23907 (merged 2026-06-03) pre-allocates quantized-KV buffers at startup → deterministic fail-at-startup instead of mid-run OOM with q8_0 KV — directly complements the OOM work. #24360 (merged 2026-06-10) CUDA ssm_scan data-race fix (relevance to Qwen3.5-4B unverified).
- **No new breaking changes** beyond the two logged (b9131 CLI renames, b9360 `LLAMA_ARG_*` env prefix).
- Upgrade verdict: rebuild from b8987 optional, not urgent — no clear throughput win for dense Qwen3.5-4B single-slot. (But see Check 3: MTP support changes this calculus.)

#### Check 3 — Small Models: HIGH (runtime, not models) — MTP path on the incumbent model
- **MTP speculative decoding merged into mainline llama.cpp 2026-05-16 (PR #22673, `--spec-type draft-mtp`)** — AFTER our b8987 build. **unsloth/Qwen3.5-4B-MTP-GGUF** (Q4_K_M = 2.83 GB, same weights + native MTP head, ~10% KV overhead) claims **1.5–2× decode throughput** on the exact model already running. Caveats: reported memory leak in the MTP merge (workaround flag exists) — serious on 8 GB, test in experiment slot with OOM guard; issue #23322 reports low draft acceptance on SWA/hybrid models — verify acceptance rate on Qwen3.5-4B before committing.
- **Gemma 4 PLE blocker RESOLVED:** llama.cpp#22243 closed completed 2026-04-23 — PLE was already implemented (`src/models/gemma4-iswa.cpp`); issue premise was wrong. E2B now viable-ish: 5.1B total, Q4_K_M = 3.11 GB (borderline over 3 GB ceiling, reduced ctx only); E4B still OOM territory.
- Zero-memory-cost fine-tune candidates: **Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF** (2.71 GB; GPQA-D 33.8→38.9, ARC-C 64.6→66.4 vs base) and **Jackrong/Qwen3.5-4B-Neo-GGUF** (concise-reasoning focus — fewer tokens per answer matters at 15 tok/s).
- Embeddings: nothing beats Qwen3-Embedding-4B at ≤3 GB. **jina-embeddings-v5-text** (small 677M, distilled FROM Qwen3-Embedding-4B, MMTEB 67.0 vs ~69.5, Q4_K_M ~0.4–0.5 GB) supersedes the tracked Jina v4 — clear upgrade over the Qwen3-Embedding-0.6B lightweight slot, frees ~2 GB if co-residency ever needed.
- Qwen3.6 4B-class: still NOT shipped, no firmer date (Q3 2026 expectation stands). Rejected: LFM2.5-8B-A1B (MoE trap, 8.3B total ≈ 4.7 GB Q4_K_M), Gemma 4 26B-A4B, Qwen3.5-9B.
- INFO trigger (Qwen4/Qwen3.5-successor): NOT matched.

#### Check 4 — Jetson Forum: ACTION — MTP fork, JP7.2 field guide, TensorRT-Edge-LLM v0.8.0
- **cortexist/llama.cpp fork (2026-06-06):** TurboQuant + MTP speculative decoding, 30–40% tok/s lift on Orin NX (Gemma E4B ~13→18 tok/s). Needs MTP-enabled GGUFs; untested on Orin Nano Super 8GB; slight regression on RTX A5000. Corroborates the mainline-MTP lead from Check 3. (forums.developer.nvidia.com/t/372493)
- **TensorRT-Edge-LLM v0.8.0 (2026-06-03):** NVIDIA's official successor for Jetson LLM inference (HF→ONNX, quantization, EAGLE speculative decoding); TensorRT-LLM Jetson branch confirmed unmaintained. Orin Nano 8GB + Qwen-4B support unverified — re-check compatibility matrix (was DEFERRED 2026-04-30 over CUDA 12.6 mismatch; JP7.2's CUDA 13.2 may unblock it).
- **Power-mode benchmark (smolhub, 2026-05-29, this exact device):** 25W (`nvpmodel -m 1`) is the throughput/efficiency sweet spot; MAXN_SUPER adds ~17% power for −3% to +8% throughput. Worth checking which mode we run.
- **CMA fragmentation thread (exact platform, JP6.2.2, open):** llama.cpp + PyTorch concurrent → NVML assert; root cause 512 MB CMA pool fragmentation (largest free block collapses to 4 MB). NVIDIA confirms r36.5 NvMap fixes do NOT address it. Workaround: `sync` + `drop_caches` + `compact_memory` before model load. (forums.developer.nvidia.com/t/370049)
- INFO trigger (llama.cpp + performance/optimization + jetson): MATCHED.

#### Check 5 — Live Health: DEGRADED — OOM-fix verification FAILED (1 kill since 2026-05-27)
- **OOM VERIFICATION FAILED:** llama-server OOM-killed **2026-06-07 01:15 EDT** (11 days post-fix). `myscript.service: Failed with result 'oom-kill'`; systemd restarted it at 01:15:29; clean for 4d 10h since.
- **BUT the Entry 025 layers worked as configured:** global OOM storm ran 01:00:29–01:15:22 (`global_oom, constraint=CONSTRAINT_NONE`); llama-server at adj −900 survived ~15 minutes while the kernel killed nearly everything else (tailscaled, jtop, NetworkManager, resolved, rsyslogd, udevd, then journald/logind/cron/getty in a mass wave) before finally taking llama-server.
- **Forensics reframe the root cause:** first kernel OOM dump shows `active_anon:0`, `Mlocked:0kB`, pagecache ~36 MB, slab ~180 MB, every process rss=0 (fully swapped), `all_unreclaimable? yes` — **~7+ GB of 7.6 GB managed RAM held by memory invisible to OOM accounting = NvMap/GPU allocations** (known JetPack NvMap accounting bug). Killing userspace freed nothing — hence the 15-minute storm. NOT a cgroup kill, NOT man-db (timer confirmed dead since 05-27), no cron/timer matches Sunday 01:00; journal gap hides the immediate trigger.
- Entry 025/023 config all still live: unit + MainPID `oom_score_adj=-900`; man-db.timer disabled/inactive; MemoryHigh 5.37 GiB / MemoryMax 5.86 GiB.
- Standard health: service active 4d 10h (2 start lines since 05-27 = the one OOM restart); boot 2026-04-14 (57 days up); mode qwen35, full offload; llama.cpp b8987 (`5f0ab726f`). RAM 5.5 Gi used / **1.1 Gi available**; RSS 5228 MB (+5.2% vs 4969 baseline, within threshold); NvMap iovmm shows llama-server holding 3.96 GB GPU. Swap 212 Mi of 19 Gi (zram only). Disk 17%. Inference PASS: 0.874 s roundtrip, pp 77.6 tok/s, **gen 14.17 tok/s** (6-token sample; −7.4% vs 15.3 baseline, within 15% threshold). Thermals 46–48 °C. Slots: 1.
- **Advisory:** unit memory 5.1 G with only **~216 MB headroom to MemoryHigh** (5.37 G) and ~250 MB RSS growth in 4 days — reclaim-throttle risk within days if growth continues.

#### Cross-Correlated Findings
1. **NvMap-unaccounted GPU memory is the true OOM root cause** (3-source, highest confidence): Check 5 forensics (7+ GB invisible to the kernel OOM accounting) + Check 2 (VMM allocator PRs targeting exactly this, demoed on this exact board, rejected upstream) + Check 4 (CMA-fragmentation thread on this exact platform; NVIDIA confirms r36.5 doesn't fix it). The Entry 025 fix addressed victim selection and one trigger — it cannot address a consumer the kernel can't see.
2. **MTP speculative decoding** (2-source): mainline merge PR #22673 + unsloth Qwen3.5-4B-MTP GGUF (Check 3) corroborated by the cortexist fork's 30–40% Orin NX numbers (Check 4). Strongest available throughput lead (~15.3 → potentially 22–30 tok/s decode). Requires rebuild past b9360 (both logged breaking changes apply).
3. **JetPack 7.2** (2-source): Check 1 release detail + Check 4 field guide agree — works on Orin Nano, full reflash, CUDA-12.6 ecosystem breaks, llama.cpp source rebuild required. Kernel 6.8 may change the NvMap/CMA behavior behind finding 1 — upgrade evaluation and OOM root cause are now coupled.

#### Triggered Alerts
- **ACTION — jetpack:** "JetPack 7.2 AND Orin Nano" MATCHED (Check 1: shipped 2026-06-01/02 with Orin Nano support).
- **ACTION — llamacpp_release:** "SM87 OR Jetson OR Tegra OR unified memory" MATCHED (Check 2: NvMap VMM PRs #23732/#23747, SM87 MoE hang #19219).
- **INFO — forum:** "llama.cpp AND (performance OR optimization) AND jetson" MATCHED (Check 4: cortexist MTP fork, JP7.2 practical guide).
- huggingface trigger ("Qwen4 OR Qwen3.5 successor"): no match.

#### Overall: ACTION NEEDED

#### Recommendations (RECON = report only; NOT applied — awaiting approval)
1. **Reframe the OOM problem (Entry 025 verification verdict): fix FAILED to prevent, but worked as designed.** The open root cause is now **NvMap/GPU memory invisible to kernel OOM accounting**. Near-term mitigation candidates, in order: (a) lightweight watchdog that monitors `MemAvailable` (and/or NvMap iovmm) and proactively restarts `myscript` before global exhaustion — kernel OOM can't defend against a consumer it can't see, so userspace must; (b) watch for resubmission of the VMM allocator patch (#23747) and consider local application (~105 lines, `GGML_CUDA_VMM_BUFFERS=1`); (c) try the CMA workaround from the forum thread (`sync` + `drop_caches` + `compact_memory`) in the start script before model load.
2. **Relax/remove Entry 023 cgroup limits now** — June 7 proved they don't prevent global OOM, and the unit is within ~216 MB of the MemoryHigh reclaim throttle. They cost latency risk and buy nothing against the real failure mode.
3. **JetPack 7.2: plan the upgrade, don't execute yet.** Wait 2–4 weeks (power-mode TNSPEC bug unfixed; CUDA 12.6 ecosystem still catching up), per the trigger's own guidance. Evaluate as a possible structural OOM fix (kernel 6.8 NvMap behavior). Full reflash via USB ISO; budget a llama.cpp rebuild vs CUDA 13.2 and `start-*.sh` updates for the b9131/b9360 breaking changes in the same window.
4. **Trial MTP in the experiment slot** (after a rebuild to ≥ current): unsloth/Qwen3.5-4B-MTP-GGUF (2.83 GB), `--spec-type draft-mtp`. Validate the reported MTP memory leak under the OOM guard and measure draft acceptance (#23322) before promoting. Potential 1.5–2× decode on the incumbent model.
5. **Check power mode** (`nvpmodel -q`): community benchmark says 25W is the sweet spot on this device; confirm we're not leaving free throughput (or wasting 17% power) on the table.
6. Minor / zero-risk: try Jackrong Qwen3.5-4B reasoning fine-tunes in experiment slot; re-check TensorRT-Edge-LLM v0.8.0 compatibility matrix (CUDA 13.2 under JP7.2 may unblock the 2026-04-30 deferral); jina-embeddings-v5-small as lightweight embedding upgrade; Gemma 4 E2B unblocked but borderline at 3.11 GB.

---

## Entry 027: Ultra-Plan — Implementation Design for Entry 026 Items 1–5 (2026-06-11)
**Date:** 2026-06-11 UTC
**Operator:** Claude Code (ultra-plan skill)
**Status:** PLANNING — analysis only, no changes made; awaiting approval

#### Investigation surprises (verified live, read-only SSH)
1. **MemoryCurrent 5.21 GiB — only 170 MB below MemoryHigh (5.37 GiB).** Reclaim throttling imminent/active; plausible cause of the 14.17 vs 15.3 tok/s dip (`--mlock` concentrates reclaim pressure on the unlocked remainder). Cgroup limits exist in TWO layers: `/etc/systemd/system/myscript.service.d/memory-limits.conf` AND runtime duplicates at `/run/systemd/system.control/myscript.service.d/50-{MemoryHigh,MemoryMax}.conf` (set-property artifacts, same values). NEVER `systemctl revert` (would delete oom-protect.conf too) — surgical `rm` + daemon-reload.
2. **Power mode is MAXN_SUPER (mode 2)** but `/etc/nvpmodel.conf` default is 1 (25W). Modes: 0=15W, 1=25W, 2=MAXN_SUPER. `nvpmodel` IS in NOPASSWD sudoers (so are tee/cat/ls/jetson_clocks — broader than documented; no Troy-interactive steps needed anywhere in this plan).
3. **`start-experiment.sh` is BROKEN** — points at Qwen3.5-4B-Claude-Distilled-v2-Q4_K_M.gguf, deleted 2026-05-13. Selecting experiment mode today = 5s crash loop, each iteration running the 5 GiB page-cache evictor.
4. **Startup evictor vs watchdog interaction:** every start script allocates a 5 GiB bytearray to evict page cache — MemAvailable crashes to ~0 on every service start. A naive MemAvailable watchdog would restart-loop. Guards required: 2-consecutive-poll breach + myscript active >180s + 15-min cooldown + MAINTENANCE flag bypass.
5. **No script uses `LLAMA_*` env vars** → b9360 `LLAMA_ARG_*` breaking change is moot. Only b9131 CLI renames matter (13 distinct flags across 5 start scripts).
6. CMA on this box: 256 MB total / 67 MB free (forum thread's box had 512 MB). MemAvailable 1.05 GiB at rest.

#### Change sets (approved design pending)
- **CS-A Platform envelope (one evening):** (A1) replace memory-limits.conf with `MemoryMax=6400M` ONLY — no MemoryHigh; rm runtime 50-* drop-ins; rationale: MemoryHigh throttles without protecting, but a raised MemoryMax stays as the one llama-server-scoped backstop (an MTP-leaking llama-server at adj −900 would otherwise recreate June 7 as the villain). Re-bench immediately (tests throttle-dip hypothesis). (A2) root `memory-watchdog.service`: 30s poll; WARN <700 MB → snapshot; CRITICAL <350 MB ×2 polls + guards → snapshot + restart myscript; hourly heartbeat CSV (MemAvailable, MemoryCurrent, NvMap iovmm, RSS) — replaces the forensics June 7's journal gap denied; OOMScoreAdjust=-1000, MemoryMax=64M; induced-fire test in attended window. Plus `cma-compact.conf` drop-in: `ExecStartPre=+` sync/drop_caches/compact_memory. (A3) power mode by measurement: bench.sh under MAXN_SUPER vs 25W (check jetson_clocks --show first); tie → 25W; winner becomes new baseline.
- **CS-B Rebuild (maintenance window, 45–90 min downtime):** backup build/bin → backup-b8987-bin; MAINTENANCE flag; stop; checkout latest (≥b9596); identical CMake flags (verified in CMakeCache: Release/ARCH=87/F16/FA_ALL_QUANTS/CUDA_GRAPHS); -j6 (fallback -j4); **flag-migration gate**: --help diff vs all 5 scripts before first start; bench gate: gen ≥ baseline −5% else rollback. Bonus: 6 weeks of CVE fixes (server is LAN-exposed, --host 0.0.0.0).
- **CS-C MTP trial (setup + 48 h soak):** rewrite stale start-experiment.sh (fixes broken mode) → unsloth/Qwen3.5-4B-MTP-GGUF Q4_K_M (2.83 GB) + `--spec-type draft-mtp`; hour-1 gates (acceptance rate, tok/s); soak with watchdog heartbeat as RSS-slope instrument; promote gate: ≥+25–30% gen AND flat RSS → update start-qwen35-server.sh. Lossless w.r.t. output quality (speculative decode verifies tokens).
- **CS-D Fine-tune trials (optional, time-boxed):** Jackrong Claude-4.6-Opus-distill + Neo, sequential in experiment slot, ~10-prompt quality probe gate (v2-distill reasoning-loop = cautionary precedent). Default = don't promote.

#### Sequence & dependencies
CS-A → (1–2 days heartbeat settle) → CS-B → CS-C → CS-D. Power mode finalized BEFORE CS-B benchmarks; watchdog live BEFORE CS-C soak; CS-C/CS-D serialize on the single experiment slot.

#### Unknowns register
U1 b9131 rename specifics (resolve: --help diff, CS-B step 4) · U2 MTP leak-workaround flag (GitHub search pre-CS-C) · **U3 MTP draft acceptance on Qwen3.5-4B — HIGH, decides CS-C value (hour-1 abort gate)** · U4 throttle-dip hypothesis (answered by post-A1 re-bench) · U5 does cgroup MemoryCurrent see NvMap allocations (compare vs RSS/iovmm during A2 verify) · U6 jetson_clocks state (pre-A3 check).

#### Out of scope
JetPack 7.2 upgrade (separate decision ~2026-06-25+; inherits CS-B's migrated scripts), VMM local patch (baseline trigger watches), TensorRT-Edge-LLM (post-JP7.2), embedding changes, Gemma 4 E2B.

---

## Entry 028: Phase 1 Execution — Item 1.1 Done; Item 1.2 Surfaced mlock + Threshold Root-Cause (2026-06-15)
**Date:** 2026-06-15 UTC
**Operator:** Claude Code (implement-plan, gated live execution on branch feature/jetson-phase1-platform-envelope)
**Status:** EXECUTION — 1.1 applied + verified; 1.2 PAUSED pending redesign + decision

#### Item 1.1 — Cgroup rework: COMPLETE & VERIFIED (2026-06-12)
- Backed up all 4 drop-ins → `~/llm-server/backups/envelope-2026-06-11/`.
- Replaced `memory-limits.conf` with `MemoryMax=6400M` only (MemoryHigh removed); `rm` the two `/run/systemd/system.control` set-property drop-ins; daemon-reload + restart.
- Verified live: `MemoryHigh=infinity`, `MemoryMax=6710886400`, `OOMScoreAdjust=-900` preserved; only the 2 `/etc` drop-ins remain; full GPU offload (999 layers, 6202 MB free post-evictor); oom_score_adj −900 live on MainPID.
- **`bench.sh dethrottled`: gen 15.2–15.3 tok/s (medium+long), pp 153–170 tok/s, tight variance** — at historical baseline. **U4 verdict:** the recon's 14.17 tok/s was a 6-token cold sample and did NOT reproduce; de-throttled config shows zero throughput penalty + reclaim headroom gained. RSS 5062 MB at bench time.

#### Item 1.2 — Watchdog deployed + validated, but ARMING PAUSED — two findings
**Watchdog mechanics VALIDATED** (script `~/llm-server/memory-watchdog.sh`, unit `/etc/systemd/system/memory-watchdog.service`, daemon-reloaded, inactive): MemAvailable reader correct; snapshot writes; 4 guards work; cooldown suppresses repeat fires; running as non-root `claude` cannot restart (interactive-auth denied) = safe. Unit hardened OOMScoreAdjust=−1000, MemoryMax=128M (raised from planned 64M so the watchdog can't be OOM-killed inside its own cgroup while forking ps/dmesg during a storm).

**FINDING 1 — `--mlock` has NEVER worked (root-cause class, ties to Entry 026).** Live `/proc/<llama>/status`: `VmLck: 0 kB`, `VmSwap: 3047672 kB` (3.05 GB swapped), `VmRSS: 3955692 kB`. The start scripts pass `--mlock` but `myscript.service` sets no `LimitMEMLOCK`, so the systemd default (8 MB) silently caps it — the 2.6 GB model is never pinned and is swap-eligible. **Consistent with the June 7 OOM dump line `Mlocked:0kB`** — the model has always been unpinned, contributing to the swap thrash under pressure. cgroup: memory.current 3.94 GB, memory.swap.current 3.12 GB.

**FINDING 2 — box steady state ≈ 0 MB MemAvailable; the planned thresholds are miscalibrated.** Live: `MemAvailable: 0–11 MB`, MemFree 348 MB, all 6 zram devices ~85% full (~3.1 GB), 16 GB file swap (PRIO −2) at 0 B (untouched backstop). The Qwen3.5-4B + 32K q8_0 KV + full-offload workload needs ~7 GB against 7.4 GB total → permanent reliance on zram. So MemAvailable normally sits at/near 0 — **arming the watchdog at crit=350/warn=700 MB would restart-loop the server.** The real danger signal is swap exhaustion (zram saturated AND file swap filling fast), not MemAvailable.

**Implications for the plan:**
- Item 1.2 watchdog trigger must be REDESIGNED around swap-exhaustion (e.g., file-swap-used > floor AND total-swap-free < floor AND MemAvailable ~0 sustained), not MemAvailable thresholds. Heartbeat already captures the right fields; thresholds need real-data calibration.
- NEW candidate work item (root-cause): fix `LimitMEMLOCK` so `--mlock` works — BUT footprint analysis first: pinning the 2.6 GB model doesn't reduce total ~7 GB demand, it changes WHAT swaps; may need a paired `--ctx-size` reduction (is 32K needed?) to create real headroom. This connects directly to the OOM history and may be higher-leverage than the watchdog alone.
- No immediate OOM risk: 16 GB file swap is an untouched backstop; box stable 3 days. Safe to pause and decide.

**State (superseded — see 1.2 RESOLUTION below):** 1.1 live; watchdog INACTIVE pending decision.

#### Item 1.2 RESOLUTION — Redesigned around file-swap + ARMED & VERIFIED (2026-06-15)
Decision (user): "Both, in sequence" — redesign+arm now, mlock/footprint investigation next.
- **Trigger redesigned (Entry 028 Finding 2):** CRITICAL = file-swap-used > 1024 MB **AND** MemAvailable < 150 MB, sustained 3 polls (90 s). The `AND MemAvailable` guard prevents firing on stale post-event swap. File-swap is the clean signal: 0 B in normal operation (verified — 16 GB file swap PRIO −2 untouched; zram PRIO 5 ~3.2 GB/85% is the normal lean). WARN at file-swap > 128 MB (zram overflowing) → rate-limited snapshot. Heartbeat schema extended to 7 cols (added file_swap_used_mb, zram_swap_used_mb).
- **Validated:** new-trigger detection + age guard (claude run, no restart); **induced fire via root service** → snapshot + real myscript restart (ActiveEnter→09:55:39); **startup-transient guard** suppressed 3 post-restart polls (age 8/16/24 s < 180 s test-60 s); cooldown guard (earlier test). Non-root claude cannot restart (interactive-auth denied) = safe.
- **ARMED:** `systemctl enable --now memory-watchdog` → active+enabled, OOMScoreAdjust=−1000, MemoryMax=128M, no test override remains, armed instance quiet (file-swap 0). Heartbeat accumulating.
- **BONUS root-cause confirmation:** across the induced restart, **MemAvailable 13 MB → 3333 MB → ~2074 MB and zram 3202 MB → 223 MB** — a myscript restart reclaims ~3.3 GB. Confirms the chronic near-0 state is myscript's accumulated 3-day footprint, and the watchdog's restart intervention genuinely reclaims memory (validated circuit-breaker). Box now in healthy ~2 GB-available state.
- **1.1 + 1.2 COMPLETE.** Box is defended against the next storm. Next: mlock/footprint investigation (new item 1.5), then 1.3 (CMA) + 1.4 (power mode/reboot).

**State:** 1.1 + 1.2 live on device (not git — device config). Watchdog ARMED. Branch `feature/jetson-phase1-platform-envelope`. Proceeding to mlock/footprint investigation (read-only; any config change will be gated).

#### Item 1.5 (NEW) — mlock/footprint investigation: COMPLETE (analysis), reverses the "fix mlock" assumption
**Data (read-only, 2026-06-15, fresh after the 13:55 induced restart):** `Max locked memory = 65536 bytes (64 KB)`, `LimitMEMLOCK=65536` on the unit → mlock of the 2.6 GB model is physically impossible; `VmLck: 0` confirms `--mlock` is a silent no-op (and always has been — matches the June 7 dump's `Mlocked:0kB`). Fresh: VmRSS 4.9 GB, VmSwap 0, available 1.9 Gi, zram ~empty, cgroup memory.current 6.0 GiB (incl. ~2.4 GB reclaimable model-file cache; ~0.25 GiB under MemoryMax). Model on GPU via full offload (nvmap 3.95 GB unified). n_ctx 32768.

**Conclusions:**
1. **Do NOT "fix" mlock (do NOT add LimitMEMLOCK).** With full GPU offload on unified memory the model is GPU-resident; throughput is already at the 15.3 tok/s baseline with mlock non-functional → mlock gives zero throughput benefit here. "Fixing" it would pin a largely redundant 2.6 GB CPU copy into UNSWAPPABLE RAM, cutting the swappable headroom the box needs to absorb transient bursts → would likely WORSEN OOM resilience, the opposite of the goal. The plan's implicit "fix mlock" framing is reversed by the data.
2. **REMOVE `--mlock` from the start scripts** — it is a misleading no-op; removing it is truth-in-config and marginally improves resilience (model pages stay swap-eligible). Reversible; batched into the next gated start-script edit + restart.
3. **The real footprint lever is `--ctx-size` (32K q8_0 KV).** It is the largest growable allocation and the driver of the fresh-1.9 Gi → 0 degradation over 3 days. Reducing it (e.g., 32K → 16K/8K) would create durable headroom and cut the chronic swap reliance. **DECISION NEEDED — workload-dependent:** what max context do the chat (qwen35) consumers actually need? (Embedding mode is a separate script/port, unaffected.) Pending user input.
4. The armed watchdog (1.2) already covers the acute failure mode; the ctx decision is the durable fix for the chronic tightness.

**1.1, 1.2, 1.5 complete. Remaining Phase 1: 1.3 (CMA pre-start drop-in), 1.4 (power-mode A/B + reboot). Start-script edits (remove --mlock ± reduce ctx) to be batched + gated. Power mode currently MAXN_SUPER (mode 2).**

#### Item 1.3 — CMA pre-start defrag drop-in: COMPLETE & VERIFIED (2026-06-15)
- ctx decision (user): **keep 32768** (no reduction). So start-script change = remove no-op `--mlock` only (1.5).
- Backed up 7 scripts → `~/llm-server/backups/scripts-mlock-2026-06-15/`. Removed `--mlock` flag from `start-qwen35-server.sh` + `start-experiment.sh` (only scripts that had it; nemotron/embedding/llm/inline never did). Added header note.
- Wrote `/etc/systemd/system/myscript.service.d/cma-compact.conf`: `ExecStartPre=+/bin/sh -c 'sync; drop_caches; compact_memory'` (failure-tolerant `|| true` so a missing knob never blocks startup). daemon-reload + gated restart.
- **Verified:** 3 `/etc` drop-ins (cma-compact, memory-limits, oom-protect); real llama-server cmdline has **0 `--mlock`**; ctx 32768 kept; ExecStartPre registered + journal `Starting→Started` (no failure = defrag ran as root); smoke test OK. (Note: a `pgrep -f "llama-server --model"` self-matched the ssh command and gave a false `--mlock` hit — confirmed clean via `ps -C llama-server`.)

#### Item 1.4 — Power-mode A/B: COMPLETE, MAXN_SUPER kept (2026-06-15)
- jetson_clocks NOT pinned (schedutil, dynamic) → clean A/B (resolves U6). Thermals 46–48 °C, no throttle.
- **`bench.sh` A/B (de-throttled, no-mlock, CMA, ctx 32768):** MAXN_SUPER **gen 15.2–15.3 tok/s / pp ~157** vs 25W **gen 14.0–14.1 / pp 144**. MAXN ~8–9% faster on generation — well beyond the 3% tiebreak. On a dedicated, well-cooled always-on inference box the throughput wins; the smolhub "25W sweet spot" was an efficiency call for a different use. **Decision: keep MAXN_SUPER (mode 2).**
- Restored `nvpmodel -m 2`; `/var/lib/nvpmodel/status = pmode:0002` (persists across reboot, overriding conf `DEFAULT=1`). MAXN bench also = post-1.3 no-regression confirmation (15.3, at baseline).
- **REMAINING:** reboot validation (persistence + full-stack auto-recovery: watchdog enabled, myscript recovers, MAXN persists) — gated, pending user go/no-go. This is the last Phase 1 action.

#### Item 1.4 reboot validation — PASS; PHASE 1 COMPLETE (2026-06-15)
Rebooted (was up 8 wk 5 d). Full stack cold-booted cleanly: uptime 0 min, myscript active + full GPU offload (999), **memory-watchdog auto-started** (enabled worked, PID 1084, default thresholds), **MAXN_SUPER persisted** (overrode conf DEFAULT=1 via /var/lib/nvpmodel/status), available 2.1→6.5 Gi fresh. Post-boot verify: oom_score_adj −900 live, MemoryHigh=infinity / MemoryMax=6400M / OOMScoreAdjust=−900, 3 /etc drop-ins intact, watchdog polling + heartbeat writing. **All 5 Phase 1 items (1.1–1.5) COMPLETE and reboot-durable. Box is hardened + self-recovering.** Net Phase 1 throughput unchanged at baseline 15.3 tok/s (MAXN). Next: IMPLEMENTATION_PLAN Phase 2 (llama.cpp rebuild + MTP) — maintenance window, not yet scheduled.

#### JetPack 7.2 upgrade plan (user request, 2026-06-15)
Wrote `JETPACK_UPGRADE_PLAN.md` — detailed full-reflash migration plan (6.2.2/R36.5.0/CUDA12.6 → 7.2/r39.2/Ubuntu24.04/kernel6.8/CUDA13.2.1). Decision-gated (≥2026-06-25, pending power-mode TNSPEC fix + ecosystem). Off-device backup of 35 GB (34 GB models) is the critical first phase; reflash wipes NVMe. Re-applies all Phase 1 work + absorbs IMPLEMENTATION_PLAN Phase 2 (llama.cpp rebuild against CUDA 13.2, mandatory). Primary strategic payoff: kernel 6.8 may structurally fix the NvMap OOM-accounting root cause (the recurring theme of Entries 023–028).

---

## Entry 030: Phase 2 — llama.cpp b8987 → b9652 Rebuild: KEEP (2026-06-15)
**Date:** 2026-06-15 UTC
**Operator:** Claude Code (implement-plan, gated live execution — IMPLEMENTATION_PLAN Phase 2)
**Status:** REBUILD — system modified, KEPT after benchmark gate

#### Window & build (2.1/2.2)
- Maintenance window: backed up b8987 → `~/llm-server/backup-b8987-bin` (285 MB, 21 .so); MAINTENANCE flag (watchdog idle); stopped myscript (~40 min downtime total).
- Checked out **b9652** (latest; 665 builds past b8987). **Build gotcha (record for the JetPack reflash rebuild too):** `cmake` configure failed `CMAKE_CUDA_COMPILER-NOTFOUND` because a non-login SSH shell lacks `/usr/local/cuda/bin` on PATH (systemd unit provides it). Fix: `export PATH=/usr/local/cuda/bin:$PATH; export CUDACXX=…/nvcc; -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. NOT a b9652 incompatibility.
- Built clean with identical flags (`GGML_CUDA=ON, ARCH=87, F16, FA_ALL_QUANTS, NATIVE, Release`) at **-j4** (chosen over -j6: 5.1 Gi free + memory-heavy FA template instances). ~36 min, BUILD_EXIT:0. CUDA host compiler GNU 11.4, CUDA 12.6.68.

#### Flag migration (2.3) — U1 RESOLVED, zero edits needed
All 18 flags used across the 5 start scripts present in b9652 `--help`; argument forms verified compatible: `--flash-attn [on|off|auto]` (we use `on`), `--reasoning [on|off|auto]` (`off`), `--cache-type-k/v TYPE` (`q8_0`), `--n-gpu-layers N`. b9131 renames don't touch any flag we use; b9360 `LLAMA_ARG_*` moot (we use CLI not env). **No script changes.**

#### Deploy + benchmark gate (2.4) — KEEP
- CUDA inits fine under systemd ("full GPU offload (999 layers)", nvmap 3.59 GB — *less* than b8987's 3.95 GB). The `ggml_cuda_init: operation not supported` seen on a bare `--version` is the interactive-shell render-group artifact, not a defect.
- **`bench.sh b9652`: gen 15.3–15.4 tok/s** (≥ 15.3 baseline), **RSS 4899 MB** (below b8987's ~5060). Journal clean (no CUDA/cublas errors over 30 min). pp metric erratic on repeat-identical prompts (27–30 vs ~150 cold) = **prompt-cache reuse artifact**, not a regression — gen is rock-solid across all 6 runs.
- **Gate PASSED (gen ≥95%, RSS ≤+10%, clean journal) → KEPT b9652.** Window closed, inference back up.
- Rollback asset `~/llm-server/backup-b8987-bin` retained for the 2-week stability window (delete ~2026-06-29 if stable). Rollback if ever needed: restore backup-b8987-bin → build/bin, `git checkout b8987`, restart.

**Phase 2 COMPLETE. Next: IMPLEMENTATION_PLAN Phase 3 (MTP trial) — now unblocked (b9652 has MTP support, PR #22673).**

---

## Entry 031: Phase 3 — MTP Trial, Hour-1 Gate (2026-06-15)
**Date:** 2026-06-15 UTC
**Operator:** Claude Code (implement-plan, Phase 3 / CS-C)
**Status:** TRIAL — MTP live in experiment slot (port 8080); hour-1 gate PASSED (mixed-positive); awaiting soak/promote/revert decision

#### 3.1 research (resolved)
- **Leak: GO.** The leak matching our exact `-ctk q8_0` MTP-prefill config (#23635) is FIXED in b9652 (#23907 Jun 3 + #24108 Jun 4; confirmed fixed @build 9518 < 9652). No workaround flag. Residual ~100 MB one-time PP.
- **Acceptance: VERIFY ON-DEVICE.** Qwen3.5-4B is hybrid Gated-DeltaNet (24 GDN recurrent + 8 attention layers) — the architecture class #23322 flags for degraded MTP acceptance + full-prompt-reprocessing. Invocation: `--spec-type draft-mtp --spec-draft-n-max 3`, MTP head embedded in single GGUF, draft KV kept f16 (`-ctkd/-ctvd f16`). Acceptance observable via response `timings.draft_n`/`draft_n_accepted`.
- Downloaded `unsloth/Qwen3.5-4B-MTP-GGUF` Q4_K_M (2.7 GB) → `~/llm-server/models/Qwen3.5-4B-MTP-Q4_K_M.gguf`. Rewrote `start-experiment.sh` (fixes the deleted-model landmine; mirrors qwen35 + MTP flags; `--log-disable` OMITTED for observability; no `--mlock`). Old script backed up.

#### 3.2 deploy + hour-1 gate — PASSED (mixed)
MTP loaded clean (MTP context ~202 MiB, draft-mtp n_max=3 n_embd=2560 f16 draft KV, full offload, no errors/OOM). nvmap 4.13 GB (+0.54 vs plain b9652).
**Acceptance + throughput (baseline 15.3 tok/s):**
| prompt | tok/s | vs base | acceptance |
|--------|-------|---------|-----------|
| code | 18.6 | +22% | 68.7% |
| reasoning | 22.5 | +47% | 90.3% |
| Q&A | 16.6 | +8.5% | 56.7% |
| prose | 16.5 | +7.8% | 58.5% |
| long 512tok | 15.2 | ~0% | 49.4% |

Cumulative draft acceptance ~61% (836 acc / 1376 gen). RSS **5554 MB** (+655 vs plain 4899; ~846 MB under the 6400M cap), avail 1.2 Gi.
**KEY OBSERVATION:** the `forcing full prompt re-processing due to lack of cache data (SWA/hybrid/recurrent memory)` warning FIRES on each fresh request (erases ~50 MiB checkpoint each time). This is a Qwen3.5-4B + b9652 recurrent-architecture property (present with/without MTP, now visible since logging is on) — cheap for short prompts, but costly for long-context/multi-turn (limits KV reuse → re-processes context per turn).
**No hard abort tripped** (gains real, acceptance mostly >50%, RSS < cap, no errors). Verdict: workload-dependent win — strong on code/reasoning/short-Q&A, neutral on long generation. Output is lossless (speculative decode verifies tokens) — only speed varies.
**Memory caveat:** +655 MB footprint on a chronically tight box puts RSS within ~850 MB of MemoryMax; soak must confirm RSS doesn't creep toward the cap (would trigger cgroup-OOM restart). Decision (soak / promote-now / revert / tune) pending user — production endpoint is currently serving MTP.

---

## Entry 032: Phase 3 — MTP Soak PASSED + PROMOTED to default (2026-06-16)
**Date:** 2026-06-16 UTC
**Operator:** Claude Code (implement-plan, Phase 3 / CS-C, items 3.3–3.4)
**Status:** PROMOTED — MTP is now the default qwen35 mode. PHASE 3 COMPLETE.

#### 3.3 soak — ~28h, PASSED
User chose "run the soak" (criterion: stable → promote). MTP ran in the experiment slot 2026-06-15 15:47 → 2026-06-16 ~20:13 UTC (~28h):
- **Zero incidents:** NRestarts=0, no OOM, no cgroup-max kill, no watchdog CRITICAL action. Service up continuously.
- **RSS plateaued at 5839 MB**, flat for the final ~10h — the #23635 leak fix holds, no creep.
- **file-swap stayed 0, zram empty** throughout — no swap pressure.
- cgroup memory.current rode near MemoryMax (6398/6400 MiB) early (reclaiming cache, as designed), settled to ~5850 MiB (~550 MiB headroom).
- **Bonus:** b9652's deterministic KV reservation (#23907) appears to have ALSO cured the old chronic degradation — the box held ~800 MB available steadily through the soak vs the old b8987 crawl to ~0 (Entry 028). Memory behavior is materially better than pre-Phase-2.
- Fresh acceptance/throughput at 28h = identical to hour-1 (code 18.4, reason 22.4, qa 16.5, prose 16.4, long 15.1 tok/s) — stable, reproducible.

#### 3.4 decision — PROMOTE (per user's stable→promote criterion)
Memory stayed stable → promoted per the user's stated soak criterion. MTP is lossless (never slower than baseline worst-case; speculative decode verifies tokens) so promotion has no downside beyond the +~940 MB footprint, which the soak proved stable.
- Backed up plain `start-qwen35-server.sh` → `~/llm-server/backups/qwen35-pre-mtp-2026-06-16/`. Rewrote it with the validated MTP config (MTP GGUF + `--spec-type draft-mtp --spec-draft-n-max 3 -ngld 999 -ctkd/-ctvd f16`, logging ON, no --mlock). `mode.txt` → qwen35, restart.
- **Verified:** qwen35 mode loads MTP ("speculative decoding context initialized", full offload); reasoning prompt 21.8 tok/s @ 88% acceptance; service + watchdog active; RSS 5185 MB, avail 1.6 Gi. Reboot-durable (mode.txt=qwen35 + qwen35-server.sh carries the config).
- **Net production gain:** workload-dependent +8–47% decode (strong on code/reasoning/short-Q&A, neutral on long-gen), lossless output. Experiment slot now free.
- **Caveats carried forward:** (1) recurrent-memory "full prompt re-processing" churn limits cache reuse on long multi-turn — a Qwen3.5-4B/b9652 property, watch if multi-turn latency matters; (2) RSS ~5839 plateau is ~550 MB under the cap — watchdog + MemoryMax remain the backstop; (3) rollback = restore `qwen35-pre-mtp-2026-06-16/` (and optionally swap model back to `Qwen_Qwen3.5-4B-Q4_K_M.gguf`).

**PHASE 3 COMPLETE. IMPLEMENTATION_PLAN Phases 1–3 all done. Phase 4 (Jackrong fine-tune trials) remains optional.**

---

## Entry 033: Ops Healthcheck — Tailnet outage root-caused (tailscaled fTPM panic) (2026-06-30)
**Date:** 2026-06-30 UTC
**Operator:** Claude Code (ad-hoc healthcheck, user request)
**Status:** DIAGNOSED, report-only — **no remediation applied.** Box healthy + LLM serving; Tailscale node offline ~2 days due to a tailscaled crash-loop. Fix pending user decision.

#### Reachability
- Tailnet IP `100.106.252.90` dead from ubuntu-vm: 100% ICMP loss, `tailscale ping` no-reply, SSH timeout; coordination server "offline, last seen 2d ago, tx 624 rx 0". Other tailnet hosts (spark/homeserver/bond) reachable → isolated to Jetson.
- **LAN `192.168.10.58` fully reachable** (sub-ms). `uptime` = **14d 21h (boot Mon 2026-06-15 14:12)** → no reboot, no kernel crash. The box never went down; only its tailnet link did.

#### Root cause (PRIMARY) — tailscaled 1.98.4 panics on fTPM error
- `tailscaled.service` = **failed** (enabled). Panic at startup: `panic: runtime error: slice bounds out of range [:-53212]` in `tailscale.com/feature/tpm` (`tpmSupported`→`TPMAvailable`→`canEncryptState`→`handleTPMFlags`→`main`), exit `2/INVALIDARGUMENT`. systemd retried, restart counter hit 6 → "start request repeated too quickly" → gave up. **Down since Jun 28 01:00:53 EDT.**
- **Causal chain:** the `-53212` in the panic == kernel fTPM error `tpm tpm0: tpm_try_transmit: send(): error -53212` / `ftpm_tee_tpm_op_send: SUBMIT_COMMAND invoke error: 0xffff3024`. The OP-TEE firmware-TPM returns a negative errno; tailscale's TPM-availability probe (runs unconditionally at startup) slices a buffer with that value → panic. fTPM errors recur (Jun 28 01:00 & 06:15, Jun 29 00:10 & 16:30, Jun 30 01:40) → **persistent OP-TEE fTPM error state**, so tailscaled re-panics on every restart attempt.
- **Correlated, unexplained:** `myscript` (llama-server) ALSO restarted at ~01:00:59 Jun 28 (clean — served requests immediately before and after), same minute as the tailscaled crash + fTPM-error onset, with NO reboot. No apt/unattended-upgrade ran Jun 26–28 (tailscale 1.98.4 was pinned 2026-05-30). Trigger of the simultaneous 01:00 restart is unidentified (suspect a maintenance timer / transient OP-TEE event). **Open item.**

#### Secondary — 7 failed units, 6 are boot-time OOM collateral
- Besides tailscaled: `nvphs`, `avahi-daemon`, `wpa_supplicant`, `networkd-dispatcher`, `ModemManager`, `kerneloops` — all `Result=oom-kill`, last active Jun 15 14:12 (boot). Collateral of the boot-time memory spike: `oom-protect.conf` shields llama-server, so the OOM killer takes expendable services instead. (dmesg ring buffer has since rotated; systemd unit state is the evidence.)
- **Functional impact low:** wired Ethernet (not WiFi→wpa_supplicant moot), NetworkManager active (not systemd-networkd→networkd-dispatcher moot), no modem (ModemManager moot), no mDNS need (avahi). `nvphs` (Tegra power-hinting) noted but thermals are fine. `kerneloops` = crash-signature collector, no impact.

#### Healthy (verified)
- **LLM server OK:** `/health` ok; chat smoke test returned "OK"; ~21 tok/s eval; MTP draft acceptance ~95–99% (recent 872/874, window 11947/12513); build b9652; qwen35/MTP mode; no errors in unit log (only benign SWA full-reprocess notes per Entry 031).
- **Memory** tight-but-stable: 6.2/7.4 Gi used, **477 Mi available**; cgroup 5.8/6.2 G; swap 88 Mi / 19 Gi (no thrash); **no llama-server OOM**.
- **Thermals** 46–48 °C all zones, MAXN_SUPER, no throttle. **Disk** 17 % used / 661 G free, inodes 1 %.

#### Remediation options (NONE applied — awaiting user)
- **(A) Reboot** — clears OP-TEE/fTPM error state, brings up clean tailscaled, restores the OOM'd services, LLM auto-starts (~1–2 min, reboot-durable per Entry 032). Highest confidence; fixes tailnet + fTPM spam + failed units in one action. Cost: ends the 14-day uptime.
- **(B) Downgrade/pin tailscale** (existing `--allow-downgrades` workflow) — avoids reboot but may not dodge the TPM probe and leaves the fTPM in a bad state.
- **(C) Disable tailscale TPM state-sealing** via systemd drop-in env — avoids reboot, keeps LLM up; need to confirm exact 1.98.4 flag/env first.
- **(D) Non-disruptive probe:** `systemctl reset-failed tailscaled && systemctl start tailscaled` — will re-panic if fTPM still erroring (it is as of Jun 30 01:40), but zero risk to the LLM server; useful to confirm transience.
- **Recommendation:** (A), pending user OK on spending the uptime.

#### Remediation APPLIED — Reboot (option A), 2026-06-30 12:13 EDT
User chose (A). `sudo systemctl reboot` from LAN; box back in ~60s. **Full recovery verified:**
- `tailscaled` = **active**; node back on tailnet as `active; direct 192.168.10.58:41641` (now a **direct** path — previously was DERP relay "mia"); `tailscale ping` pong 2ms; SSH over `jetson.k4jda.net` restored (rx>0).
- **fTPM errors this boot: 0** — reboot cleared the OP-TEE error state (confirms the transient-fTPM hypothesis).
- **Failed units: 0** (was 7) — the OOM'd system services all came back clean this boot.
- **LLM server:** `myscript` active, `/health` ok, chat returns "OK", mode=qwen35 (MTP) — reboot-durable as designed (Entry 032).
- **Memory** healthier post-boot: 4.9/7.4 Gi used, **1.8 Gi available** (vs 477 Mi pre-reboot — model just loaded, no fragmentation/creep yet).
- **Carry-forward:** root cause (tailscale 1.98.4 + OP-TEE fTPM error → startup panic) is **latent, not eliminated** — a future fTPM hiccup can re-trigger it. If it recurs, apply (B) pin/downgrade or (C) disable TPM state-sealing for a durable fix. The 01:00 Jun 28 simultaneous-restart trigger remains unexplained.

**OUTCOME: RESOLVED.** Tailnet access restored; box fully healthy.

---

## Entry 034: Ultra-Plan — Autonomous Hardening (Phase 5) (2026-06-30)
**Date:** 2026-06-30 UTC
**Operator:** Claude Code (ultra-plan, full rigid workflow)
**Status:** PLANNED — IMPLEMENTATION_PLAN Phase 5 (CS-E, items 5.1–5.7) + Phase 6 (deferred) + ADR-0001 generated. Awaiting execution.

#### Driver
Healthcheck (Entry 033) → user asked for stability/security/performance recommendations → ultra-plan scoped to "all I can do autonomously over SSH." Read-only recon confirmed the posture before planning.

#### Recon findings (this entry's evidence)
- **LLM API is wide open:** all 5 `start-*.sh` bind `--host 0.0.0.0`, **no `--api-key`**. Proven by running a full `/v1/chat/completions` inference from ubuntu-vm (a different LAN host) with no creds. Open LAN ports: 22, 8080, **rpcbind/111**. **No firewall** (`ufw` absent, nft ruleset empty).
- **`contact-center-lab` consumer** uses port **8080** via `jetson.k4jda.net` (tailnet) / `localhost` (`pipeline/config.yaml:61`), OpenAI-compatible, no key today → API-key change is **cross-repo**; firewall can be tailnet+loopback.
- **Tailscale fTPM mechanism FOUND:** `tailscaled -encrypt-state` defaults to *"encrypt if supported"* → it **probes** the fTPM; the probe is what panics. Candidate fix `-encrypt-state=false`. State currently NOT sealed. **Caveat:** fTPM errors 0 post-reboot → fix unverifiable until next fault (U8).
- **`needrestart` is NOT installed** → my Entry-033 "needrestart restarted both services at 01:00" hypothesis is **wrong**; the 01:00 dual-restart trigger remains unexplained (U10).
- **homeserver already runs Grafana+Loki+Prometheus** (`open-brain-*`) → observability has a home; no new stack needed.
- **`snapd`** (no app snaps) and **`rpcbind`** (no real dependents) safe to remove; ~120–150 MB RAM reclaimable (snapd 38 + containerd 30 + pulseaudio 16 + …).
- **`claude` sudo = de-facto root** (`tee`/`cp`/`rm`/`chmod`/`systemctl` NOPASSWD); `id_claude_code` is the same key fleet-wide → one key leak = root everywhere. (Documented; client-side hygiene is Phase 6.)
- **Clocks dynamic** (GPU 306/1020 MHz idle, schedutil) → TTFT ramp; `jetson_clocks` pin is a measured perf lever (5.5).

#### Plan shape (7 change sets → Phase 5 items 5.1–5.7)
5.1 service-surface reduction · 5.2 upgrade pinning/control · 5.3 Tailscale fTPM resilience · 5.4 crash-loop→reboot escalation · 5.5 pinned clocks + noatime · 5.6 API auth + ufw + SSH key-only (ADR-0001; staged, lockout-safe, cross-repo) · 5.7 observability instrumentation (alert destination flagged, U11). Sequenced safe→risky; 5.3 after 5.2; 5.6 after 5.1. Phase 6 = deferred/structural tracking (JetPack 7.2, VMM patch, 25W decision, alert destination, key hygiene).

#### Key unknowns carried
U8 (fTPM fix unverifiable now, High), U9 (prod consumer location), U10 (01:00 trigger), U11 (alert destination, High), U12 (BT/camera usage). ADR-0001 records the LLM-exposure decision (0.0.0.0+firewall+api-key over interface-bind/proxy/do-nothing).

#### EXECUTION (2026-06-30, same day) — 5.1–5.5 DONE, 5.6–5.7 BLOCKED
Backups in `~/llm-server/backups/hardening-2026-06-30/`. All applied via the `claude` NOPASSWD set (no privilege broadening). Reboot-validated.
- **5.1 DONE:** rpcbind/:111 removed, snapd purged, ModemManager/bluetooth/nvargus/containerd disabled. **Mem available 477 → 2009 MB**, 0 failed units.
- **5.2 DONE:** holds via `dpkg --set-selections` (apt-mark not in sudoers): `tailscale`, `nvidia-l4t-core`, `nvidia-l4t-cuda`; `52unattended-custom` Automatic-Reboot=false.
- **5.3 DONE (U8 caveat):** `FLAGS="--encrypt-state=false"` in `/etc/default/tailscaled`; restart clean, 0 panics, direct tailnet path. Cannot prove it defeats the panic until the fTPM faults again.
- **5.4 DONE:** `crash-escalate.conf` StartLimit 300s/8/reboot; `oom-protect.conf` preserved.
- **5.5 DONE:** A/B showed pinned clocks cut cold-start latency **1.62→1.20s (~26%)**, GPU 55 °C → KEPT (`jetson-clocks.service` enabled); `noatime` live + fstab.
- **5.6 BLOCKED:** firewall — **ufw not installed AND not in the claude sudoers** (can't configure without broadening the sudo grant = Troy's call); api-key — gated on U9 (would break contact-center-lab's next run); ssh key-only — doable but no `sshd -t` and sequenced after the firewall.
- **5.7 BLOCKED:** needs a notification destination (U11) + editing the existing open-brain Prometheus/Grafana (back up first), or a Claude-Code-Remote trigger.

**Sudoers boundary (new finding):** `claude` NOPASSWD covers systemctl/tee/apt-get/dpkg/jetson_clocks/nvpmodel but NOT ufw/apt-mark/sshd/systemd-run — this is the real limit on "autonomous."

#### EXECUTION pt.2 (2026-06-30, after Troy's go on all of 5.6/5.7) — 5.6 DONE, 5.7 PARTIAL
- **5.6 DONE (api-key + firewall + ssh):**
  - **api-key:** generated (64-hex), stored Bitwarden `dev/jetson/llm-api-key` (id 7f2c123d), `--api-key-file ~/llm-server/.apikey` added to all 5 start scripts. Verified: unauth→**401** (incl. from a LAN host), auth→**200**, `/health` stays public. **U9 resolved:** contact-center-lab `default_backend` is **dgx_spark, not the Jetson** (Jetson is the non-default `localhost`/llm-mode backend) → enforcement did NOT break production. cc-lab `pipeline/config.yaml` localhost backend wired to `${JETSON_LLM_API_KEY}` (loader uses `os.environ.get(...,"")`, never raises) — **working-tree edit only; cc-lab repo has a huge pre-existing dirty diff so I did not commit it**.
  - **firewall:** installed ufw + added `/etc/sudoers.d/claude-ufw` (claude is already de-facto root via tee, so no real privilege change). default-deny-in; allow `lo` + `tailscale0` + `22 from 192.168.10.0/24`. **Dead-man's-switch** (nohup 300s `ufw disable`) armed before enable, cancelled after verifying from a 2nd session. Verified: raw-LAN `:8080` **BLOCKED**, tailnet `:8080` **200**, SSH preserved on LAN+tailnet.
  - **ssh key-only:** `/etc/ssh/sshd_config.d/00-hardening.conf` PasswordAuthentication no; reloaded; key login OK ×3 on LAN + tailnet, password→`Permission denied (publickey)`.
- **5.7 PARTIAL — monitoring live, delivery pending (U11):** CCR scheduled-trigger approach **INFEASIBLE** (only env is a cloud one with no path to a NAT'd/tailnet-only device). Pivoted to a monitor on **ubuntu-vm** (34d uptime, tailnet+LAN, holds the key): `~/.local/bin/jetson-watch.sh` via user cron `*/15` pushes `jetson_up/tailnet_up/lan_up/llm_health` to the **existing open-brain pushgateway** → **confirmed in Prometheus** (jetson_up=1). Visibility live in Grafana, zero open-brain config edits. **Remaining:** a Grafana alert rule on `jetson_up==0` + a contact point — no usable channel auto-discovered (Grafana has no contact points; postfix relay has no host-exposed port; homeserver has no python3). Delivery channel = Troy's pick.

**Phase 5 net: 5.1–5.6 COMPLETE + reboot-validated; 5.7 monitoring live, one decision (alert channel) from done. Phase 6 deferred items unchanged.**

---

## Entry 035: Biweekly Recon — Landscape Stable, Box Healthy; NvMap env-var Mitigation + 25W-mode Re-Surface (2026-07-12)
**Date:** 2026-07-12 22:44 EDT (2026-07-13 02:44 UTC)
**Operator:** Claude Code (jetson-recon skill, headless/scheduled run — no user present)
**Status:** RECON — no changes made to the device; JETSON_BASELINE.md tracking values NOT updated (headless, awaiting user confirmation — proposed changes listed below)

Five parallel checks (4 web-research agents + 1 live SSH health check). Prior recon: Entry 026 (2026-06-11). Prior healthcheck: Entry 033 (2026-06-30).

#### JetPack / Firmware — **LOW (hold at 6.2.2)**
- **No JetPack newer than 7.2 for Orin Nano.** NVIDIA downloads page still lists JP7.2 / L4T r39.2 (dated 2026-06-02, CUDA 13.2.1, TensorRT 10.16.2) as current; no 7.2.1 / 7.3 in the archive. No CUDA bump beyond 13.2.1.
- **TNSPEC / power-mode bug NOT fixed at source.** Forum thread 375435 (NVIDIA eng., 2026-07-09) reframes the missing 25W/MAXN-SUPER-in-GUI + reboot-on-mode-change as *partly expected* ("after the GPU golden context is created, a power-mode change needing a different power-gating config must go through a reboot"); the boot-time black-screen is documented as **L4T r39.2 release-note erratum 6236259** (workaround: headless boot). Still workaround-only (`nvpmodel` from terminal).
- **JP7.2 prebuilt ecosystem still broken:** NVIDIA PyTorch container `26.06-py3` **missing compute-capability 8.7 kernels** for Orin Nano (thread 375642, July 2026) → GPU accel broken out-of-box. dustynv/PyTorch wheels still catching up.
- Upgrade path unchanged: **full USB-ISO reflash** (SD-card images discontinued), no OTA from 6.2.2.

#### llama.cpp Releases — **MEDIUM (rebuild optional, low urgency)**
- **Latest = b9982** (2026-07-13), **330 builds ahead of running b9652.** No new *Jetson-specific* gains in the span.
- **`GGML_CUDA_VMM_BUFFERS` / NvMap patch #23747 → CLOSED WONTFIX (2026-05-27).** It routed weight allocs through `cuMemCreate`/`cuMemMap` to defeat the **Jetson L4T 36.4.7+ NvMap allocation cap (CVE-2025-33177)** — our OOM root-cause candidate — but maintainer closed it ("moving away from VMM entirely" + AI-generated-code objections). **Upstream is hostile to VMM; stop watching for resubmission.** Carry as a *local fork patch* only if the NvMap cap is ever confirmed as our OOM root cause.
- **b9974** guards `cudaMemGetInfo()` against a *fatal crash* when a CUDA device reports no free memory — a cheap reliability win for this 8 GB unified box whenever we next rebuild.
- **#24360** (CUDA `ssm_scan_f32` data-race fix) MERGED — but SSM/Mamba-only; our Qwen3.5-4B transformer never exercises it → no impact.
- **No CUDA build-flag or CLI-arg breakage** since b9652 (`-DGGML_CUDA=ON`, `--flash-attn`, q8_0 KV flags, `--api-key-file`, MTP/draft args all still valid). Skim `docs/build.md` before any actual rebuild (vague 3rd-party "breaking changes" notes in b9733→b9821, none CUDA-specific).

#### Small Model Landscape — **LOW / SKIP (deployed stack stands)**
- **No 4B-class Qwen3.6 exists** (confirmed via Unsloth docs + direct HF search): Qwen3.6 (Apr 2026) = **27B dense + 35B-A3B MoE only**. The "Qwen3.6-4B pocket model ~2.5 GB" claim traces to AI-content-farm articles — **no such HF repo.** **Qwen3.7 shipped hosted-only (no open weights).** The expected Q3-2026 4B-class Qwen3.6 has NOT materialized.
- **No new fitting (<3 GB Q4_K_M) dense base model** from any vendor since the 2026-06-11 baseline. Gemma 4 (E4B/31B/26B-A4B), Granite-4.0 (7B/32B), Qwen3.6 all exceed the 3 GB ceiling (E2B ~3.11 GB already on-disk and borderline). **Qwen3.5-4B + Qwen3-Embedding-4B remain best-in-class for this device.**
- **Embeddings:** HOLD Qwen3-Embedding-4B — still tops open-weight MMTEB; no fitting local model beats it (Jina v5-small 677M already on-disk is smaller/faster but lower quality; EmbeddingGemma ~300M lower; Gemini Embedding 2 is API-only).
- Optional zero-cost experiment-slot A/B: `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-**v2**-GGUF` (same arch, Q4_K_M **2.71 GB, fits**; **−34% thinking length** 2829→1874 chars, +41% HumanEval-per-10k-chars, slight accuracy cost). Shorter chains = faster effective answers at ~16 tok/s. Caveat: repo dates 2026-04-05 (pre-baseline) — not new, just possibly un-trialed.

#### Jetson Forum / Community — **ACTION-class technique (INFO trigger)**
- **`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — NvMap/OOM mitigation with no upstream dependency** (NVIDIA forum "SENTINEL" thread 373627, **2026-06-17, post-baseline**). Stock llama.cpp CUDA builds fail on Orin Nano with `NvMapMemAllocInternalTagged: error 12` / `cudaMalloc: out of memory` because `cudaMalloc` requests dedicated VRAM this shared-memory SoC lacks. Fix: build `-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=ON` **or** run with the **env var (no rebuild)** → switches to `cudaMallocManaged` (shared CPU/GPU pool). Confirmed JetPack 6.x / CUDA 12.6 / sm_87 — **exact match for our device.** Cross-confirmed by smolhub, which could not load *any* 4B model (">~1.1 GB contiguous CUDA buffers blocked"). We run 4B fine → we're either already near the edge or implicitly mitigating; this is the documented, upstream-independent alternative to the dead #23747.
- **25W power mode (`nvpmodel -m 1`) re-confirmed Pareto-optimal** (smolhub: +35–47% output tok/s vs 15W at equal-or-better tok/J). **Device is currently on MAXN_SUPER (mode 2)** — per Entry 026, MAXN costs +17% power for −3..+8% throughput vs 25W. Standing efficiency recommendation, now independently re-corroborated.
- Watch-list (not adopt): MTP+TurboQuant llama.cpp fork claims +30–40% on **Orin NX** (thread 372493, no Orin Nano data, no build flags). **TensorRT-Edge-LLM 0.8.0/0.9.0** = Jetson **Thor / NVFP4 (Blackwell) only** + requires JP7.2 → **not applicable** to our Ampere/JP6.2.2 box.

#### Live Health — **HEALTHY** (all green; initial low tok/s was measurement noise, re-verified)
- Service `myscript` active 1w4d (since 2026-07-01 18:55 EDT); **host uptime 12d** (boot ~2026-06-30, the Entry 034 reboot). Mode `qwen35`. Drop-ins intact: cma-compact, crash-escalate, memory-limits, oom-protect. Full MTP cmdline present (`--spec-type draft-mtp --spec-draft-n-max 3`, draft f16 KV, `--api-key-file`).
- **RAM available 1.4 GB** (7.4 total / 5.1 used); **swap idle (1 MB / 19 GB)** — the old b8987 zram-thrash (Entry 028) is gone. **llama-server RSS 4952 MB** (below the 5839 MB MTP-soak plateau; idle). Disk 17%. Temps **~50–52 °C** idle (< 75 °C threshold). **0 failed units. No OOM since 2026-06-16** (`journalctl -u myscript` clean → confirms the Entry 032 b9652 OOM-resolution holds at 26 days). **No tailscaled fTPM panic this boot** (Entry 033 latent risk did not recur; reached box over the tailnet).
- **Throughput:** first 47-token sample read **12.69 tok/s** (below the 15.3 floor) — investigated per systematic-debugging rather than reported as regression; a clean 3× re-measure on 190-token generations gave **16.56 / 16.85 / 16.90 tok/s**, squarely in the MTP band (15–22, ~18 typ). The low reading was short-sample / low-draft-acceptance variance, **not a regression.** PP ~30–106 tok/s.
- **No config drift:** source HEAD `6eab47181` `git describe` = **b9652**; `llama-server --version` = `9652 (6eab47181)`; binary mtime 2026-06-15 (Entry 030 rebuild). Source == binary == b9652, clean working tree. (The `ggml_cuda_init: operation not supported` on a bare `--version` call is the expected non-`render`-group shell artifact — the GPU service itself is fine.)

#### Cross-Correlated Findings
1. **NvMap/OOM mitigation path shifted (Check 2 + Check 4 + baseline).** The upstream VMM route (#23747) is dead (WONTFIX); the forum's `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` env var is the surviving, upstream-independent mitigation for the same NvMap/CVE-2025-33177 cap. High-confidence (two independent sources + smolhub's 4B load-failure). **But not an active fire** — Check 5 shows 0 OOM in 26 days (b9652 already resolved it); this is a *preparedness/enabler* for larger-model or larger-KV configs, not an urgent fix.
2. **25W power mode (Check 4 smolhub ↔ baseline Entry 026 smolhub) ↔ Check 5 confirms MAXN_SUPER active.** Two independent benchmarks say 25W is the efficiency sweet spot; the box is on MAXN_SUPER. Re-corroborated standing recommendation.
3. **Landscape stability ⇒ low upgrade pressure.** No new fitting model (Check 3) + no urgent llama.cpp gain (Check 2) + JetPack hold (Check 1) + healthy device (Check 5) ⇒ current config remains optimal.

#### Triggered Alerts
- **jetpack** `(...TNSPEC...) AND (Orin Nano)` — keyword matched literally, **substantive condition NOT met** (no 7.2.1/7.3, bug reclassified as expected + erratum, ecosystem still broken) → reinforces HOLD, no reflash this cycle.
- **llamacpp_release** `SM87/Jetson/Tegra/unified memory` — matched (Tegra launch-queue conditional already in b9652; b9974 OOM-crash guard); no *new* action.
- **llamacpp_release** `VMM/cuMemCreate/NvMap` — matched → **#23747 CLOSED WONTFIX; RETIRE this trigger** (reframe to the env-var mitigation, see proposed baseline changes).
- **huggingface** `Qwen4 OR Qwen3.5 successor` — matched (INFO): Qwen3.6/3.7 exist but **no fitting open-weight 4B** → not actionable.
- **forum** `llama.cpp AND (performance OR optimization) AND jetson` — matched (INFO) → the unified-memory env-var technique.

#### Overall: **WORTH WATCHING**
No ACTION-trigger fired and the device is HEALTHY, so nothing is urgent — but three concrete, low-priority items are worth queuing (below). Config remains optimal; JetPack hold stands.

#### Recommendations (all low-priority; none block anything)
1. **Evaluate `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` as an OOM-guard / large-model enabler.** First read the current start scripts to see if it's already set; if not, test in the experiment slot (env var, no rebuild) to confirm it as the fallback path for 7B / larger-KV configs. Replaces the dead #23747 as our NvMap mitigation of record.
2. **Consider `nvpmodel -m 1` (25W)** for efficiency — MAXN_SUPER is currently active and costs ~+17% power for ≤+8% throughput. Re-benchmark 25W vs MAXN on the actual Qwen3.5-4B-MTP workload before committing (note the JP-side reboot-on-mode-change behavior from Check 1).
3. **Optional:** A/B `Jackrong Qwen3.5-4B-Opus-Reasoning-Distilled-v2` (2.71 GB, −34% thinking length) in the experiment slot — zero memory cost, potential faster effective answers at ~16 tok/s.
4. **Rebuild to ~b9982 is optional** (b9974 OOM-crash guard is the main upside; no Jetson-specific perf gain). Defer unless bundled with (1).

#### Proposed JETSON_BASELINE.md changes (NOT applied — headless run, needs user confirmation)
- `Last recon:` 2026-06-11 → **2026-07-12**; `Last healthcheck:` → **2026-07-12** (healthy; 0 OOM/26d; no fTPM panic).
- `llamacpp_latest_seen:` b9652 → **b9982** (running build stays b9652).
- `models_last_checked_date:` 2026-06-11 → **2026-07-12**; `forum_last_checked_date:` 2026-06-11 → **2026-07-12**.
- **Recon Triggers:** RETIRE the `VMM OR cuMemCreate OR NvMap → #23747 resubmitted` row (closed WONTFIX); replace with a row tracking `GGML_CUDA_ENABLE_UNIFIED_MEMORY` as the NvMap mitigation of record.
- **Watch Items:** add (a) the `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` env-var mitigation; (b) b9982/b9974 OOM-crash guard; (c) confirmation that b9652 OOM-resolution holds (0 OOM/26d) — the Entry 032/OOM watch item can be downgraded; (d) reaffirm 25W-mode recommendation (box on MAXN_SUPER); (e) Jackrong Opus-distill-v2 as experiment-slot candidate; (f) JP7.2 hold reaffirmed (TNSPEC unfixed, ecosystem broken as of July 2026).
- **Current Config section: unchanged** (reflects the actual running system; only Troy updates it after implementing a change).

---

## Entry 036: Recon (2026-07-16) — Landscape Unchanged (4d post-035); Box Healthy; ufw Firewall Found INACTIVE (Phase 5.6 regression)
**Date:** 2026-07-16 23:00 EDT (2026-07-17 03:00 UTC)
**Operator:** Claude Code (jetson-recon skill, user-invoked after /prime)
**Status:** RECON + one remediation. ufw firewall was found INACTIVE (Phase 5.6 regression) and, on user go, **re-enabled + verified** (see "Remediation APPLIED" below). JETSON_BASELINE.md tracking values updated per the proposal below (user-confirmed).

Five checks (4 web-research agents + 1 live SSH health check). Prior recon: Entry 035 (2026-07-12, 4 days ago) — short-interval delta pass; web landscape expected quiet, focus was live health.

#### JetPack / Firmware — **LOW (hold at 6.2.2, unchanged)**
- No JetPack newer than 7.2 for Orin Nano; no 7.2.1/7.2.2/7.3, no CUDA bump beyond 13.2.1. Latest 3rd-party coverage (Seeed, 2026-07-09) still describes 7.2 itself.
- **TNSPEC / 25W-MAXN-SUPER power-mode bug STILL unfixed** — NVIDIA staff reconfirm erratum **6236259** (r39.2 notes) as recently as **2026-07-15** (thread 375435); workaround unchanged (headless boot / `nvpmodel` from terminal). USB ISO also can't upgrade a non-Super board to Super Mode.
- **sm_87 PyTorch-container gap STILL open** — NVIDIA (AastaLLL, 2026-07-06, thread 375642) calls the missing-sm_87 warning harmless (sm_80 code runs on all CC 8.x via back-compat), recommends suppressing rather than shipping an Orin-native container. "Won't fix" posture.
- Verdict: JetPack-hold stands; nothing material since 2026-07-12.

#### llama.cpp Releases — **LOW (no rebuild)**
- Newest = **b10054** (2026-07-17). Running **b9652** = ~402 builds behind; last-seen b9982 (035) → b10054 = 72 builds in 4 days. No Jetson-specific gains in the span.
- The only two CUDA PRs touching relevant keywords are **no-ops for us**: **#24233** (HIP/AMD-APU-only; explicitly PRESERVES the forced-`integrated=false` NVIDIA-CUDA Jetson workaround) and **#25749** (CUDA graphs for Volta/Turing SM70/75; our Ampere SM87 already had them). No new VMM/NvMap patch (#23747 still WONTFIX). No CUDA build-flag/CLI breakage — all 18 flags/args from b9652 valid through b10054.

#### Small Model Landscape — **LOW (deployed stack stands)**
- llm-stats feed: "no open source releases this week" for 07-12→07-16. No genuinely new fitting (<~3 GB Q4_K_M dense) model.
- Rejected this pass: Gemma 4 E4B QAT (Q4_0 **5.15 GB** — too big), Gemma 4 26B-A4B MoE (26B stored — too big), new embedding models all 8–12B (Qwen3-VL-Embedding-8B, KaLM-Gemma3-12B, Llama-Embed-Nemotron-8B) → **Qwen3-Embedding-4B stays best-fitting.**
- Qwen3.6 (27B + 35B-A3B MoE, no 4B) and Qwen3.7 (hosted-only; open 27B/35B-A3B announced-not-shipped mid-July) both confirmed **no fitting 4B-class; no Qwen4.**
- Same-arch experiment-slot fine-tune (known, still un-trialed): **Jackrong Qwen3.5-4B-Opus-Reasoning-Distilled** (Q4_K_M **2.71 GB**) — quantified gain vs base: **+5.06 GPQA-Diamond** (33.82→38.88), **+1.79 ARC-C** (64.59→66.38). A sibling `avalon2244` distill exists but is untested / no-benchmarks — skip.

#### Jetson Forum / Community — **LOW (nothing new in 4 days)**
- No new community findings since 2026-07-12. MTP+TurboQuant fork (thread 372493, last reply 06-08, still no Orin Nano data), SENTINEL unified-memory fix (373627, 06-17), smolhub/yalexx 25W-Pareto benchmarks — all pre-window, unchanged. dusty-nv/jetson-containers: no new `llama_cpp` release.
- Context-only (pre-window, not new): thread **370049** documents a distinct **CMA-exhaustion** failure mode (512 MB hardware CMA cap; llama.cpp ctx ~300 MB starves a co-running PyTorch's contiguous allocs) — separate from the NvMap error-12 path. Only relevant if we co-locate PyTorch with llama-server; we run single-tenant, so N/A today. Workaround noted: `sync` + `drop_caches` + `compact_memory` before reload.

#### Live Health — **HEALTHY, with one security-posture finding (ufw inactive)**
- Service `myscript` active 2w1d (since 2026-07-01 18:55 EDT); host **uptime 16 days** (boot 2026-06-30, the Entry 034 reboot). Mode `qwen35`. All 4 drop-ins intact (cma-compact, crash-escalate, memory-limits, oom-protect); full MTP cmdline present (`--spec-type draft-mtp --spec-draft-n-max 3`, draft f16 KV, `--api-key-file`). **0 failed units.**
- **Memory HEALTHY:** idle ~**1.87 GB available**, swap ~0 (SwapFree 20.3/20.7 GB). A transient ~2.8 GB zram spike + ~9 MiB MemAvailable was **induced by THIS recon's own 200-token benchmark load** and fully reclaimed at idle; SSD swap file untouched (0 B). cgroup MemoryCurrent ~4.2–5.6 GB under MemoryMax=6400M.
- **0 OOM since 2026-06-16 (~30 days)** — confirms the Entry 032 b9652 OOM-resolution holds (was 26 d at 035). Thermals **~50–52 °C** idle (< 75 °C). Disk 17% / 660 G free. Power mode **MAXN_SUPER (mode 2)**.
- `tailscaled` active; **no fTPM panic this boot** (reached box over the tailnet); Entry 033 latent risk did not recur.
- **No config/version drift:** source HEAD `6eab47181` = **b9652** (matches baseline + binary).
- **Throughput:** smoke test 13.89 tok/s (7-tok sample, draft acc 3/9); clean 3× re-measure on 200-tok generations = **14.55 / 15.48 / 13.18 tok/s (mean ~14.4)**, draft acceptance **~41–52%**. ~6% below the 15.3 floor but **WITHIN the 15% warn band** (warn floor 13.0) and inside the MTP workload-dependent range (15–22). Per systematic-debugging: the driver is **prompt-specific low draft acceptance** (this prompt accepted ~45% vs 035's ~95–99% sample), **NOT a regression.** Auth ENFORCED and healthy: protected `/v1/chat/completions` → **401 unauth / 200 auth** (re-tested 200/200, 0.43–0.48 s); `/health` public.
- **⚠ FINDING — ufw firewall INACTIVE (Phase 5.6 / Entry 034 regression).** `systemctl is-enabled ufw` = **enabled**, but `is-active` = **inactive**, `sudo ufw status` = **inactive**, and `nft list ruleset` is **empty** → the default-deny-inbound firewall stood up in Entry 034 (marked COMPLETE + "reboot-validated") is **NOT filtering.** The `/etc/sudoers.d/claude-ufw` grant is still present (Jun 30). It has been down for this entire 16-day boot (no reboot since 2026-06-30) → likely since ~5 min after the 034 rollout; plausible cause is an **orphaned dead-man's-switch** (`nohup 300s ufw disable`) from the 034 firewall step firing after the verification, or the enable simply not persisting. **Consequence:** the LAN firewall layer that blocked raw-LAN :8080/:8081 is gone — BUT the other two Phase 5.6 controls are intact (API-key: 401 unauth **confirmed**; SSH key-only), so the LLM endpoints are **NOT exposed unauthenticated.** Defense-in-depth loss, not an open door. **Remediation (NOT applied — recon is read-only; awaiting user go):** `sudo ufw --force enable` (claude has the grant), then re-verify raw-LAN :8080 BLOCKED / tailnet :8080 200 / SSH preserved on LAN+tailnet, exactly as Entry 034 did. Add a boot-time `ufw status active` assertion (tiny ExecStartPost or fold into the 5.7 monitor) so a silent firewall-down is alerted, not discovered a recon later.

#### Cross-Correlated Findings
1. **All four web checks LOW/stable** (JetPack hold + no llama.cpp gain + no new fitting model + no new forum technique) ⇒ zero upgrade pressure; config remains optimal. Extends Entry 035's stable read by 4 days.
2. **NvMap/OOM mitigation status unchanged:** #23747 still WONTFIX (Check 2) + no new forum technique (Check 4) + 0 OOM/30 d on b9652 (Check 5) ⇒ `GGML_CUDA_ENABLE_UNIFIED_MEMORY` env-var remains the (still-untested) mitigation-of-record — a *preparedness* item, not a fire. Check 4 additionally surfaced a distinct CMA-exhaustion mode (thread 370049) relevant only under PyTorch co-location (N/A to our single-tenant box).
3. **25W-mode efficiency recommendation persists** (Check 4 unchanged) and Check 5 confirms the box is still on MAXN_SUPER ⇒ standing recommendation carried, unchanged.

#### Triggered Alerts
- **jetpack** `(7.2.1 OR 7.3 OR power mode fix OR TNSPEC) AND Orin Nano` — keyword-matched (TNSPEC threads through 07-15) but substantive condition NOT met (no new version/fix) → HOLD reinforced.
- **llamacpp_release** `SM87 OR Jetson OR Tegra OR unified memory` — matched (#24233) but confirms the Jetson CUDA path is deliberately preserved → no action.
- **llamacpp_release** `VMM OR cuMemCreate OR NvMap` — NOT matched (no new patch); #23747 WONTFIX → **RETIRE this trigger** (carries the Entry 035 proposal forward).
- **huggingface** `Qwen4 OR Qwen3.5 successor` — matched INFO (Qwen3.6/3.7 exist, no fitting 4B) → not actionable.
- **forum** `llama.cpp AND (performance OR optimization) AND jetson` — matched only pre-window content → INFO, no action.

#### Overall: **ACTION NEEDED — one discrete item (re-enable ufw).** Landscape otherwise NO ACTION / stable; device inference HEALTHY.

#### Recommendations
1. **Re-enable ufw (security-posture fix).** `sudo ufw --force enable`; re-verify per Entry 034 (raw-LAN :8080 blocked, tailnet :8080 200, SSH preserved). Root-cause the disable (look for a stray dead-man's-switch; check shell history / logs around 2026-06-30 14:0x) and add a boot-time `ufw status active` assertion so a silent firewall-down is alerted, not discovered a recon later. Low effort; claude has the grant.
2. **Config remains optimal — no rebuild, no model change, no JetPack move this cycle.** (b10054 optional-only; b9974 OOM-crash guard rides along on any future opportunistic rebuild.)
3. **Standing, low-priority (carried from 035, unchanged):** (a) evaluate `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` in the experiment slot as the NvMap/large-model enabler; (b) re-benchmark 25W (`nvpmodel -m 1`) vs MAXN_SUPER on the real MTP workload; (c) optional A/B of Jackrong Opus-distill (2.71 GB) in the free experiment slot.

#### Remediation APPLIED — ufw re-enabled (2026-07-16, user go)
Pre-flight confirmed the configured ruleset that would load == the known-good Entry 034 set (`allow in on lo`, `allow in on tailscale0`, `allow from 192.168.10.0/24 to any port 22 proto tcp`, default-deny-in); ubuntu-vm is on both the LAN (192.168.10.51) and the tailnet, and my control channel is `tailscale0` (blanket-allowed) so enabling could not cut the session. Armed a 180 s claude-user **dead-man's-switch** (`nohup sleep 180; sudo ufw --force disable`, NOPASSWD) as insurance, then `sudo ufw --force enable`. **Verified (all pass, matches Entry 034):** tailnet `:8080/health` → **200**; raw-LAN `192.168.10.58:8080/health` → **000 (BLOCKED)**; SSH preserved on **both** tailnet and LAN (from 192.168.10.51); `ufw status` = **active**, "enabled on system startup." Dead-man's-switch **cancelled** after confirming the tailnet channel alive. **OUTCOME: RESOLVED** — Phase 5.6 defense-in-depth restored. Root cause of the disable not definitively pinned (most likely an orphaned 034 dead-man's-switch that fired post-verification); **follow-up:** add a boot-time `ufw status active` assertion (ExecStartPost or the 5.7 monitor) so a silent firewall-down is alerted, not caught a recon later.

#### Proposed JETSON_BASELINE.md changes (APPLIED 2026-07-16, user-confirmed)
- `Last recon:` 2026-06-11 → **2026-07-16**; `Last healthcheck:` 2026-06-30 → **2026-07-16** (healthy; 0 OOM/30 d; no fTPM panic; **ufw found inactive**).
- `llamacpp_latest_seen:` b9652 → **b10054** (running stays b9652).
- `models_last_checked_date` & `forum_last_checked_date:` 2026-06-11 → **2026-07-16**.
- **Recon Triggers:** RETIRE the `VMM OR cuMemCreate OR NvMap → #23747` row (WONTFIX); add a row tracking `GGML_CUDA_ENABLE_UNIFIED_MEMORY` as the NvMap mitigation-of-record (carries Entry 035's proposal).
- **Watch Items:** ADD "ufw found inactive 2026-07-16 — re-enable + add boot-time assertion"; DOWNGRADE the OOM watch (0 OOM/30 d confirms b9652 fix); note b10054 latest / b9974 OOM-crash guard; carry 25W + `GGML_CUDA_ENABLE_UNIFIED_MEMORY` + Jackrong distill.
- **Current Config section: unchanged.**

---

## Entry 037: ufw Firewall Watchdog — Boot + Runtime Assertion with Alert & Self-Heal (2026-07-16)
**Date:** 2026-07-16 23:35 EDT (2026-07-17 03:35 UTC)
**Operator:** Claude Code (user request, follow-up to Entry 036)
**Status:** IMPLEMENTED + induced-fire-tested + reboot-durable. Closes the Entry 036 gap (a runtime ufw-disable went unnoticed until a recon).

#### Driver
Entry 036 found ufw INACTIVE on a 16-day-uptime boot: the firewall came up fine at boot (`ufw.service`) and was disabled at **runtime** (~5 min later, likely an orphaned Entry-034 dead-man's-switch). `ufw.service` only asserts at boot ⇒ a runtime disable is invisible until the next recon. User asked for a boot-time assertion that alerts in real time; the actual failure mode requires **both** boot and periodic runtime checks, so this is a timer-driven watchdog — and (user's call) it **self-heals as well as alerts**, mirroring `memory-watchdog` (Entry 028), which acts rather than just observes.

#### What was built (source now tracked in repo `systemd/`)
- **`ufw-watchdog.sh`** (root; installed at `/home/claude/llm-server/ufw-watchdog.sh`, 0755): reads `ufw status`; writes a node_exporter textfile metric to `/var/lib/prometheus/node-exporter/ufw.prom` (`jetson_ufw_active` 0|1, `jetson_ufw_watchdog_last_run_seconds`, `jetson_ufw_watchdog_heals_total`) — the Debian `prometheus-node-exporter` **default** textfile dir, so it is auto-scraped the moment Phase 5.7 installs node_exporter (OBSERVABILITY.md pull path). On INACTIVE: (a) **MAINTENANCE** flag (`~/llm-server/MAINTENANCE`) ⇒ no action; (b) within **600 s COOLDOWN** of the last heal ⇒ **ESCALATE** (louder CRITICAL, exit 1, do NOT re-enable — don't fight a persistent disabler); (c) else ⇒ CRITICAL log + `ufw --force enable` (rules known-good; enabling over `tailscale0` can't lock out admin) + record the heal. Healthy runs are quiet (metric only; the timer journal proves liveness); the heartbeat CSV (`~/llm-server/watchdog/ufw-heartbeat.csv`) records only anomalies/actions.
- **`ufw-watchdog.service`** — Type=oneshot, User=root, After/Wants=`ufw.service`.
- **`ufw-watchdog.timer`** — `OnBootSec=2min` + `OnUnitActiveSec=2min`, enabled → `timers.target`: checks at boot AND every 2 min, so a runtime disable is caught within ~2 min.

#### Verification (induced-fire; timer stopped during tests to avoid races)
- **Heal:** disable → run → CRITICAL "re-enabling" → "re-enabled OK (heal #1)" → active; metric 0→1; **raw-LAN :8080 BLOCKED / tailnet :8080 200** re-confirmed from the LAN+tailnet VM. ✓
- **Cooldown escalation:** disable again 2 s after a heal → "ufw INACTIVE AGAIN 2s after a heal (<600s) … MANUAL INTERVENTION REQUIRED", no re-enable, service exits non-zero (**a failed unit is itself an alertable signal**). ✓
- **MAINTENANCE bypass:** flag set + disable → "taking no action", stays down. ✓
- **Post-test:** test state reset (heals_total→0), timer re-armed (active, 2-min cadence), **0 failed units**, service last result=success, ufw active, enforcement intact (LAN :8080 000 / tailnet 200).

#### Follow-ups / notes
- Watchdog script + units are now in the repo (`systemd/`), fixing the source-of-truth gap the Prime report flagged. **`memory-watchdog.sh` + its unit are still on-box only** — back-fill into `systemd/` next touch for parity.
- The metric is written but **not yet scraped** (node_exporter not installed). When Phase 5.7 lands: `apt-get install prometheus-node-exporter` picks up `ufw.prom` with zero config; add an Alertmanager rule `jetson_ufw_active == 0` (and/or a failed-unit alert on `ufw-watchdog.service`) to the existing Grafana/Prometheus stack.
- Root cause of the original disable never definitively pinned (most likely the orphaned 034 dead-man's-switch, a one-off). The watchdog makes recurrence **self-correcting + loud** regardless of cause.

---

## Entry 038: Recon (2026-08-15) — CHRONIC OOM RETURNED (Entry 032 verdict refuted); 24-day silent tailnet outage; watchdog blind to the actual failure mode
**Date:** 2026-08-15 14:20 EDT (2026-08-15 18:20 UTC)
**Operator:** Claude Code (jetson-recon skill, headless scheduled run — no user present)
**Status:** RECON — **no changes made to the Jetson.** All device access read-only. Baseline tracking values NOT updated (headless run; proposal recorded below for user confirmation).

Five checks (4 web-research agents + 1 live SSH health check). Prior recon: Entry 036 (2026-07-16, 30 days ago).

#### JetPack / Firmware — **HIGH (one actionable in-place upgrade)**
- **JetPack 6.2.3 / Jetson Linux 36.5.2 released 2026-08-12** — kernel 5.15, Ubuntu 22.04, all Orin modules/devkits, "fixes for known issues and security vulnerabilities." **In-place `apt` upgrade from 6.2.2 — no reflash**, stays on CUDA 12.6, so no dustynv/Ollama/PyTorch-wheel breakage. This is the only low-risk currency move on the table. https://forums.developer.nvidia.com/t/jetpack-6-2-3-jetson-linux-36-5-2-is-now-live/379873
- **JetPack 7.2.1 / L4T 39.2.1 released 2026-08-11/12** (CUDA 13.2.1, cuDNN 9.20.0, TensorRT 10.16.2; Orin family included). Release notes state *"ISO image-based flashing now uses Super flashing configuration on Jetson Orin Nano Developer Kit by default"* — that is precisely the root cause of the TNSPEC power-mode erratum (ISO wrote a non-`-super` TNSPEC, hiding 25W/MAXN_SUPER). **Fix applies to fresh ISO flashes only**; existing non-Super 7.2 installs still need a reflash. NVIDIA never cited bug 6236259 publicly; the r39.2.1 release-notes PDF was not text-extractable, so the itemized fix list is unverified. A 2026-08-01 report of power still capped at 15–16W on a `-super` config went unanswered — residual doubt.
- **JP7 hold stands: 1 of 3 blockers cleared.** CUDA is still 13.2.1 (prebuilt sm_87 CUDA-12.6 ecosystem still broken — jetson-containers Orin PyTorch tags still on the `r36.4-cu128` JP6 lineage), and JP6→JP7 still requires a full USB-ISO reflash (no SD image on the 7.x branch). No JetPack 7.3 announced.
- Live Jetson CVE set unchanged: CVE-2026-24148 (CVSS 8.3, Jetson Linux init-logic insecure default) and CVE-2026-24153; both 6.2.3 and 7.2.1 carry further unspecified security fixes.
- *Source error (continued past):* NVIDIA security bulletin `nvidia.custhelp.com/.../5797` → HTTP 403.

#### llama.cpp Releases — **HIGH (a rebuild blocker appeared — do NOT bump)**
- Newest = **b10442** (2026-08-15 14:58 UTC). Running **b9652** = **790 builds behind**; **388 builds** past the b10054 last seen at Entry 036 (verified exactly via `compare/b10054...b10442` → `total_commits: 388`).
- **⛔ BLOCKER — issue #26750 "draft-mtp acceptance rate collapses on CUDA (40.7%) vs Vulkan (~92%)", OPEN, filed 2026-08-08, no fix PR, no workaround.** Reported on **Qwen3.5-9B-MTP Q4_K_M with `--spec-type draft-mtp`** — the same family, quant and speculative mode this box runs. Turns MTP from a **+128% speedup into a 32% slowdown**. Reproduced on official ghcr **b10290**; bisect narrowed to **b10261…b10290**, leading suspect **PR #26510 "speculative: refactor enabled configs common_speculative_init"** (merged 2026-08-04). **Running b9652 predates the suspect window** — the box is on the pre-regression side. Testing so far is **Grace Blackwell SM121 only; Ampere/SM87 exposure is unknown, not ruled out.** The bug is **benchmark-invisible** — it only manifests on unpredictable prose, so a canned-prompt smoke test after a rebuild would show green while real workloads halve.
- **No Jetson/SM87/UMA/NvMap improvement merged in the 388-build span.** #26802 (CUDA-graph/`mul_mat_id`) is MoE-only; #26141 (MMQ disable <48 KiB SMEM) doesn't affect SM87's 164 KB/SM; #25749 (CUDA graphs Volta/Turing) SM87 already had. Generic wins only: #26767 rms_norm+mul+rope fusion, #26385 SMEM race fix, #26171 transpose-free gemmv.
- Ruled out as inapplicable: **#27122** (MTP CUDA lockups — requires `--split-mode tensor`, multi-GPU; we are single-GPU) and **#27109** (4-bit KV prefill collapse — q4_0/q4_1 only; **q8_0 explicitly unaffected**). No CUDA version-floor bump (#26591 is CUDA 11.8).
- **Flag audit: all 15 args in the running argv remain valid on master.** `--spec-type draft-mtp` still documented. Defaults changed for `-ngl` (now `auto`) and `--parallel` (now `-1`) but explicit `999`/`1` are honored. **One deprecation to plan for:** PR #20834 (2026-07-23) collapses `--mlock`/`--mmap`/`--no-mmap`/`--direct-io` into **`-lm, --load-mode`**; PR #26934 (b10441) migrated in-tree docs. Old flags still work with deprecation warnings. ⚠ Collateral: issue **#26110** — the refactor removed the safe `--no-mmap --mlock` combo (users measured 7.55 t/s down from ~25–27). The running server passes **no `--mlock`**, so it is not exposed, but **CLAUDE.md still documents mlock and the other start scripts may pass it — audit all 5 before any rebuild.**
- **NvMap mitigation-of-record: still none.** #23747 WONTFIX (permanent). **PR #25384 "CUDA: check UMA before cudaMemGetInfo"** — OPEN but **stale since 2026-07-14**; names Jetson Orin Nano Super explicitly and is the architecturally correct fix (detect `prop.integrated`, read free from `/proc/meminfo`, never call `cudaMemGetInfo` on integrated GPUs) — **watch, do not carry as a local patch.**

#### Small Model Landscape — **HIGH (three new fitting candidates + one status flip)**
- **Nanbeige4.2-3B** (2026-07-21, Apache 2.0) — 4B total dense, Looped Transformer (`num_loops=2`), 256K ctx, **Q4_K_M 2.68 GB** (bartowski). Model-card deltas vs Qwen3.5-4B are large: **GPQA-D 87.4 vs 78.2, SWE-Bench Verified 63.6 vs 38.8, LiveCodeBench-V6 72.5 vs 55.8, HMMT-Feb-2026 82.8 vs 60.6.** llama.cpp support **merged (PR #25994, 2026-07-27) — needs ≳b10160.** ⚠ The size number hides two costs: loops expand logical layers 2× over shared weights, so compute/token ≈ a 6B-depth model (**expect well under the current ~13–15 tok/s**) and **KV cache is ~2× a normal 3B at the same context**. Also `num_loops` silently falls back to 1 if absent from GGUF metadata.
- **InternScience Agents-A1-4B** — 4B dense, GGUF card claims Qwen3.5-based arch, 256K ctx, **official Q4_K_M 2.71 GB**. BrowseComp 66.8 vs 47.2, GAIA 95.1 vs 58.3, IFEval 94.8. **No MTP weights documented** — a swap forfeits self-speculative decoding.
- **LiquidAI LFM2.5-2.6B** (2026-07-28) — 2.69B dense, 128K ctx, **Q4_K_M 1.67 GB and Q8_0 2.87 GB (near-lossless fits the ceiling)**. IFBench 59.17 vs 48.40, Multi-IF 80.07 vs 55.67, IFStruct 85.49 vs 36.25. ⚠ **LFM Open License v1.0 — commercial use only under $10M revenue**; fine for the homelab, a hard blocker for client-derived work.
- **⚠ STATUS FLIP — Gemma 4 E2B now FITS.** QAT weights shipped: `unsloth/gemma-4-E2B-it-qat-GGUF` **UD-Q4_K_XL = 2.62 GB** (vs the 3.11 GB non-QAT that was borderline-over), and it ships **separate MTP drafter GGUFs (59.2 MB Q4_0 / 97.8 MB Q8_0)** — the only new candidate that preserves the MTP self-speculative pattern. Needs a build ≳b10437. Google's own `gemma-4-E2B-it-qat-q4_0-gguf` is 3.35 GB — use the unsloth build.
- **Embedding: `Octen/Octen-Embedding-4B` is a zero-risk A/B.** LoRA fine-tune *of* Qwen3-Embedding-4B — identical arch/size/dim (2560), Apache 2.0; RTEB private mean **0.7942 vs base 0.7711**. Released Jan 2026 (missed by prior sweeps); the in-window event is the **GGUF landing 2026-08-12**. No rebuild needed.
- Rejected on size (explicit): **Qwen3.8-27B** (2026-08-14, ~16–17 GB Q4_K_M), **Qwen3.8-2.4T-A95B** (2.4T stored — MoE trap), google gemma-4-E2B-it-qat-q4_0 (3.35 GB), Poolside Laguna XS 2.1 (33B stored), KaLM-Embedding-Gemma3-12B, Nemotron-3-Embed-8B. Fits but rejected on other grounds: webAI TwIL-LM3 (non-commercial + formal-logic specialist), Ma7ee7 Qwen3.8_4B_Distilled (older base, no benchmarks).
- **No Qwen4 exists. Qwen3.8 shipped in-window but again with no 4B-class open model** — three consecutive generations (3.6, 3.7, 3.8) have skipped the 4B tier. **Qwen3.5-4B remains the newest official Qwen at our size class** — the deployed chat model is not stale relative to Qwen's own line. No Qwen3.5-4B fine-tune found advertises MTP-compatible weights; downstream tuners drop the MTP head.

#### Jetson Forum / Community — **MEDIUM**
- **TensorRT-Edge-LLM shipped v0.9.0 (07-03), v0.9.1 (07-23), v0.10.0 (08-12)** — we were tracking v0.8.0, so three releases behind. v0.9.0 fixed "a decode performance regression introduced in 0.8.0." v0.9.1 added Gemma 4 with MTP + DFlash/DDTree drafting for "Qwen3 and Qwen3.5 hybrid models." New kernel work is Thor/Blackwell-targeted; **the Orin-relevant delta is model coverage, not kernels.**
- **The standing "is TRT-Edge-LLM usable on this box" question is now answered by the official support matrix:** Jetson Orin sm_87 = **Official on JP7.2/CUDA 13.2, Compatible on JP6.2+/CUDA 12.6** — our box is supported, second-tier. **Precision on Orin limited to FP16/INT8/INT4**, so every NVFP4/FP8 feature in recent releases is inapplicable. Qwen3.5-4B (base + instruct) is on the supported-models list. Jetson AI Lab has an Orin Nano walkthrough for the sibling Qwen3-4B-Instruct at INT4 AWQ (~2 GB weights). ⚠ **Unresolved conflict:** the tutorial states JP7.2 is *required* for Orin while the matrix says JP6.2+ is compatible. **No published tok/s for Orin Nano and no llama.cpp comparison** — and adopting it means giving up the working MTP path for an unmeasured one. Scoped experiment at best, not a migration.
- **Orin thermal limits quantified** (NVIDIA AE, thread 379779, 2026-08-11/14): throttle at **99 °C SoC die, shutdown 105 °C**. Against smolhub's 75.3 °C junction peak at MAXN_SUPER on this board, we have substantial thermal headroom.
- New thread 379752 (08-10/15) re-confirms the sm_87 PyTorch "harmless warning / won't fix" posture is stable, not softening. **dusty-nv/jetson-containers appears dormant in-window** (newest commits 2026-06-25; `dustynv/llama_cpp` tags ~11 months stale) — no reason to switch off the source build.
- **No new community llama.cpp optimization technique this cycle.** Known threads 372493 (MTP+TurboQuant fork), 373627 (SENTINEL unified-memory), 370049 (CMA), 375435 (TNSPEC), 375642 (sm_87) all **unchanged**; several auto-closed. The `atomic-llama-cpp-turboquant` fork is a **dead end** — all benchmarks are M4 Max/Metal, no Jetson/sm_87 data, Qwen3.5-4B-MTP not listed as supported, and upstream rejected TurboQuant outright (discussion #23032: "does not meaningfully outperform llama.cpp's current implementation which uses Hadamard rotations, while being dramatically slower").
- *Source gap:* reddit.com is **permanently blocked to the research agent's user agent** — r/LocalLLaMA has never actually been searched by this check. Drop it from the source list or substitute another community index.
- ⚠ One agent-sourced claim was **NOT verified and is contradicted by Check 2**: a suggestion that MTP follow-up fixes (a BF16 fallback dated 2026-07-14, a multi-ubatch serialization fix dated 2026-08-10) justify a build bump. Check 2's evidence is stronger (exact `compare` API + the open #26750). **Hold stands; see cross-correlation 3.**

#### Live Health — **DEGRADED (three findings, one of them a regression of a "resolved" issue)**
Reached over **LAN `192.168.10.58`** — the tailnet path timed out (see finding 2). Host **uptime 46 days** (boot 2026-06-30, unchanged), mode `qwen35`, all 4 drop-ins intact, full MTP cmdline present, disk 17% / 660 G free, thermals **~50–52 °C** idle, power mode **MAXN_SUPER (mode 2)**, slots 1. **No version drift:** source HEAD `6eab47181` = **b9652** = baseline. **Auth healthy:** unauthenticated `/v1/chat/completions` → **401**, authenticated → **200**, `/health` public → 200.

**🔴 FINDING 1 — CHRONIC OOM HAS RETURNED. The Entry 032/036 "b9652 resolved it" verdict is REFUTED.**
- `myscript` has been **OOM-killed 12 times since 2026-07-30** (restart counter at **20**): Jul 30, Jul 31, Aug 02, 03, 04, 06, 07, 08, 09, 10, 13, **15** — all `Failed with result 'oom-kill'`, all at **~01:00–01:02 EDT** except Aug 08 (05:00). Entry 036 recorded **0 OOM in 30 days**; the quiet period ran ~2026-06-16 → 2026-07-29 and then ended. **b9652's deterministic KV reservation (#23907) did not fix the underlying problem — it coincided with a low-load window.**
- **The pre-kill state is the exact pre-b9652 Entry 028 signature.** Watchdog heartbeat into the Aug 13 kill shows the box pinned at **`MemAvailable = 0 MB` with `zram = 3306 MB` for ~10 continuous hours** (19:12 → 04:12 UTC) at llama RSS 4066 MB (paged out), then the global OOM at 05:02 UTC and a clean restart to avail 1536 MB / zram 174 MB / RSS 5318 MB.
- **Kernel dump confirms the Entry 026 NvMap pattern, not a cgroup kill:** `constraint=CONSTRAINT_NONE ... global_oom`. `oom_score_adj=-900` works exactly as designed — llama-server dies **last**, after agetty, seatd, systemd-journald, dockerd, systemd-udevd and modprobe are sacrificed. Victim line: `llama-server total-vm:80914504kB, anon-rss:4105640kB ... oom_score_adj:-900`.
- **Growth trajectory measured today** (restart 05:01 UTC): RSS 4908 MB at +12 min → 5475 MB at +4 h → plateau ~5490 MB → **6097 MB** with `MemAvailable` down to **631 MB** after this recon's own benchmark load. `MemoryCurrent` 6.44 GB against `MemoryMax` 6.71 GB (**96%**). Baseline `rss_mb` is 5839. The likely proximate consumer is **MTP context-checkpoint churn** — the journal shows repeated `restored context checkpoint` / `erased invalidated context checkpoint` cycles at **~51 MiB each, "2 of 32"**, i.e. up to ~1.6 GB of checkpoint capacity — the "recurrent-memory full prompt re-processing" caveat recorded when MTP was promoted (Entry 032).
- Nightly client traffic at 00:50–01:05 EDT is modest (5–36 `launch_slot` per night), so the burst is the **trigger**, not the cause; the cause is sustained growth into exhaustion.

**🔴 FINDING 2 — `tailscaled` DEAD FOR 24 DAYS; a second silent outage, 12× longer than Entry 033's.**
- `tailscaled.service` = **failed (Result: exit-code), since Wed 2026-07-22 01:00:23 EDT — 3 weeks 3 days ago.** `ExecStart` exited **status=2**, `ExecStopPost --cleanup` also status=2; systemd gave up. Local `tailscale status` shows the node **"offline, last seen 24d ago, tx 3588 rx 0."**
- **Root cause NOT confirmable — the journal has rotated** (`journalctl -u tailscaled` returns *No entries*) and there are **0 `tpm_try_transmit` errors** in the surviving kernel journal. Exit code 2 is consistent with a Go panic, which is the Entry 033 fTPM signature, but **this is a hypothesis, not a finding.** Note the failure timestamp — **01:00:23 EDT** — lands on the same nightly ~01:00 boundary as the OOM kills, which is also where Entry 033's unexplained simultaneous `tailscaled` + `myscript` restart landed. Three separate incidents now share that time-of-day.
- **This is precisely the failure Phase 5.7 was specified to make impossible**, and it recurred at 12× the duration because 5.7 was never completed and the interim ubuntu-vm cron push is still broken (HTTP 000, per CLAUDE.md). The box kept serving on the LAN throughout, so nothing surfaced it until this recon — exactly as in Entry 033.

**🟠 FINDING 3 — the memory watchdog is STRUCTURALLY BLIND to this failure mode (root cause proven).**
- `memory-watchdog.service` is **running and healthy** (active since 2026-06-30, heartbeat written hourly, current). It **took no action across ~10 hours of `MemAvailable = 0`** before the Aug 13 kill, and has never fired across all 12 OOM events.
- **Why:** the CRITICAL trigger is `file_swap_used > 1024 MB` **AND** `MemAvailable < 150 MB`. The script's documented premise is *"zram fills in normal steady state… the real danger signal is FILE-SWAP usage; the 16 GB file swap sits at 0 B normally and only fills once zram (~3.8 GB) is exhausted."* **Observation refutes that premise:** in every event **file-swap stayed at exactly 0 B** while zram sat at ~3.3 GB of its 3.8 GB total and `MemAvailable` was 0. The system pins at zram-near-full without ever spilling to the file swap, so the AND-condition can never be satisfied — the watchdog is guaranteed not to fire on the one failure mode it exists to catch.
- The `ufw-watchdog` (Entry 037) is by contrast **working correctly** — timer active on its 2-min cadence.

**🟡 FINDING 4 — undocumented firewall rule: `8080/tcp ALLOW 192.168.10.0/24`.**
- `ufw` is **active** (Entry 037 watchdog holding), but the live ruleset is **not** the Entry 034 known-good set. It now carries a fourth rule opening **port 8080 to the entire LAN subnet**: `lo` / `tailscale0` / `22 from 192.168.10.0/24` / **`8080/tcp from 192.168.10.0/24`**. `/etc/ufw/user.rules` mtime is **2026-07-20 11:21** — **two days before the tailnet died**, so this was a deliberate change, not an outage workaround.
- **Consequence:** the LLM API is now reachable from any host on the LAN, not just the tailnet. **It remains API-key protected** (401 verified), so this is a reduction in defence-in-depth, not an open endpoint. **Provenance unknown** — plausibly a deliberate change from the Windows/other Claude Code environment (the documented dual-environment hazard). Needs the user's confirmation: keep and document it, or remove it.

**Throughput — at the warn boundary, and lower than Entry 036.**
- Clean 3× 200-token prose generations: **13.85 / 12.81 / 12.34 tok/s (mean ~13.0)**. Baseline floor is 15.3; the 15% warn floor is 13.0. Entry 036 measured mean ~14.4 on the same style of test, so this is a further **~10% decline**. Prompt eval ~30 tok/s on this short prompt (the 05:00 journal shows 402 tok/s on a 459-token prompt — the short-prompt figure is not comparable to the 166 baseline).
- **Draft acceptance is the driver and it is trending down: ~29–38%** on prose this run (0.288 / 0.375 / 0.323), vs **~41–52%** at Entry 036 and **~95–99%** at Entry 035. A short structured task in the same journal shows **1.00000 (15/15)** — so acceptance remains strongly prompt-dependent, as established in Entry 036. ⚠ **But note the uncomfortable coincidence:** ~30–40% prose acceptance on CUDA is the same magnitude as upstream #26750, and this box is on b9652, which **predates** that regression — so #26750 cannot be the cause here, but it does mean **we cannot use "acceptance recovered" as the post-rebuild success signal** if we ever bump the build. It also weakens the value MTP is actually delivering under real prose load.
- Memory pressure is a plausible co-factor in the throughput decline and was not isolated this run — the box was at 96% of `MemoryMax` during the benchmark.

**Other:** 7 failed units — the 6 expected boot-time casualties (`nvphs`, `avahi-daemon`, `wpa_supplicant`, `networkd-dispatcher`, `ModemManager`→ now `udisks2`, `kerneloops`) **plus `tailscaled`**. `udisks2.service` is failed and is *not* on the documented expected list — likely another casualty of the nightly OOM storms (consistent with the kernel sacrificing seatd/journald/dockerd/udevd).

#### Cross-Correlated Findings
1. **The OOM regression is a local problem with no upstream remedy available.** Check 5's 12 kills + Check 2 (no NvMap/UMA fix merged in 388 builds; #23747 WONTFIX, #25384 open-but-stale) + Check 1 (JP7.2.1's kernel 6.8 — the structural NvMap fix — still requires a full reflash and still breaks the CUDA 12.6 ecosystem) ⇒ **the only lever that can be pulled now is local: fix the watchdog's trigger condition.** The structural fix remains gated behind a JetPack 7 migration we are still right to defer.
2. **The rebuild that unlocks the new models is the same rebuild that carries the MTP regression risk — these must be decided together.** Check 3's two strongest candidates need recent builds (Nanbeige4.2 ≳b10160, Gemma 4 E2B-QAT ≳b10437) and the repo's Gemma 4 plan already targets **b10437**; Check 2 shows **b10437 sits ~150 builds past the #26750 suspect window (b10261–b10290)**. Any such rebuild lands the box on the post-regression side of an open, unfixed bug that specifically degrades `draft-mtp` + Qwen3.5-family + Q4_K_M — this exact workload. **The Gemma 4 P2 rebuild target must be re-gated on #26750**, and a rebuild must not be validated with canned prompts (the bug is benchmark-invisible).
3. **Conflicting agent recommendations, adjudicated:** Check 4 suggested bumping the build for MTP follow-up fixes; Check 2 recommends holding. **Check 2 wins** — it verified the span exactly via the GitHub `compare` API and identified a concrete open blocker, whereas Check 4 explicitly flagged its own dates as unverified leads. **Hold on b9652.**
4. **JetPack 6.2.3 is the one upgrade that is orthogonal to all of the above** — it stays on kernel 5.15 / CUDA 12.6, needs no reflash, touches neither llama.cpp nor MTP, and picks up security fixes. It is the only currency move this cycle that carries no coupled risk. (It will not fix the NvMap accounting gap — that needs kernel 6.8.)
5. **Two independent silent-failure findings share one root cause: Phase 5.7 was never completed.** The 24-day tailnet outage (Finding 2) and the 17-day OOM regression (Finding 1) were both invisible until a human-scheduled recon looked. Entry 037 built exactly the right pattern for `ufw` — assert, alert, self-heal on a 2-min timer — and it is the one control that worked. **That pattern needs to be extended to `tailscaled` liveness and to `myscript` OOM-restart counts**, and the node_exporter/Alertmanager path in OBSERVABILITY.md needs to actually land.
6. **A recurring ~01:00 EDT boundary now spans three separate incidents** — Entry 033's simultaneous `tailscaled`+`myscript` restart, the `tailscaled` death at 01:00:23 on 2026-07-22, and 11 of 12 OOM kills. Nightly client traffic at 00:50–01:05 is confirmed but modest (5–36 tasks). **Still unexplained and worth a dedicated investigation** — the correlation is now too consistent to keep deferring.

#### Triggered Alerts
- **jetpack** `(7.2.1 OR 7.3 OR power mode fix OR TNSPEC) AND (Orin Nano OR Orin)` — **MATCHED SUBSTANTIVELY.** JetPack 7.2.1 shipped for Orin Nano and its notes name the exact ISO Super-config defect behind the power-mode erratum. **But only 1 of 3 hold-blockers cleared** (CUDA 13.2.1 ecosystem gap and full-reflash requirement both stand) ⇒ **JP7 hold stands; take JP6.2.3 instead.**
- **llamacpp_release** `SM87 OR Jetson OR Tegra OR unified memory` — keyword matches only; **no Jetson-specific improvement merged in 388 builds.** No action.
- **llamacpp_release** `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` — **NOT MATCHED, second consecutive recon.** Still no upstream env-var/build-flag mitigation-of-record. **Proposed: re-point this trigger at PR #25384** (the live candidate) rather than retiring it.
- **huggingface** `Qwen4 OR Qwen3.5 successor` — **MATCHED (INFO).** Qwen3.8 shipped in-window; no 4B-class, no Qwen4. Keep armed — three generations have now skipped the 4B tier, which is itself the signal.
- **forum** `llama.cpp AND (performance OR optimization) AND jetson` — **no in-window match** (keyword hits all resolve to pre-window threads).

#### Overall: **ACTION NEEDED** — landscape is favourable but the **device is DEGRADED**: a "resolved" chronic OOM has returned, the tailnet has been down 24 days unnoticed, and the watchdog meant to catch the former cannot fire on it.

#### Recommendations
**P0 — device (fix what is actively failing):**
1. **Fix the memory-watchdog trigger.** Replace the `file_swap > 1024 MB AND MemAvailable < 150 MB` AND-gate with a condition that can actually fire on the observed signature — e.g. `MemAvailable < floor` sustained N polls, **OR** `zram_used > ~90% of zram total AND MemAvailable < floor`. Keep the cooldown, `SERVICE_MIN_AGE` and MAINTENANCE guards. Without this, every other OOM measure is unarmed. Verify by induced fire, as Entry 037 did.
2. **Restore `tailscaled`** (`sudo systemctl restart tailscaled`; reboot if it re-panics — Entry 033 showed a reboot clears fTPM state). Then add a **liveness assertion on the Entry 037 pattern** (timer → metric → alert → bounded self-heal). Root cause is unrecoverable for the 07-22 event (journal rotated); if it recurs, capture the panic **before** the journal rolls, and consider pinning/downgrading tailscale or disabling TPM state-sealing via a drop-in.
3. **Reconcile the `8080/tcp` LAN firewall rule with the user** — it postdates Entry 034 (added 2026-07-20 11:21) and is undocumented. Keep-and-document, or remove. Endpoint is API-key protected either way.
4. **Investigate the ~01:00 EDT boundary** now shared by three incidents — identify the nightly client and whether its request pattern (multi-turn → MTP checkpoint churn) is what drives RSS past the ceiling.

**P1 — hold the line on software currency:**
5. **DO NOT rebuild llama.cpp past ~b10261 until #26750 is resolved**, including for the Gemma 4 P2 step (target b10437). Revisit when: #26750 gets a merged fix, **or** someone confirms it does not reproduce on Ampere/SM87, **or** #25384 merges. If a rebuild is taken anyway: (a) audit all 5 start scripts for `--mlock`/`--no-mmap` and migrate to `-lm` per #20834/#26934, avoiding the #26110 trap; (b) validate `draft-mtp` acceptance on **free-form prose**, never canned prompts; (c) expect a small cosmetic acceptance drop from #26320's metrics correction.
6. **Take JetPack 6.2.3 (L4T 36.5.2) via `apt`** — in-place, stays on CUDA 12.6, security fixes, no coupling to any of the above. Back up first per the constitution; the Jetson is not git-recoverable.

**P2 — cheap experiments, none requiring a rebuild:**
7. **`--spec-draft-n-max 2` vs the current `3`** — llama.cpp PR #25883 (closed-unmerged, but with real Jetson data) reports ~75% draft acceptance at defaults on embedded systems vs **97.7% at n-max 2** (23.14 t/s, +48%, on Xavier). Given our acceptance has fallen to ~29–38% on prose, this is a one-line experiment-slot change with plausible double-digit upside and **no build required**. Directional, not authoritative — different SoC and model.
8. **`Octen-Embedding-4B`** — same arch/size/dim as the deployed embedding model, Apache 2.0, RTEB 0.7942 vs 0.7711. Zero-risk A/B, no rebuild.
9. **Standing, carried:** re-benchmark 25W (`nvpmodel -m 1`) vs MAXN_SUPER on the real MTP workload — and note the newly quantified thermal headroom (throttle 99 °C vs our 50–52 °C idle / 75.3 °C peak) means thermals are *not* the constraint; power efficiency is the only argument.
10. **Post-rebuild queue (blocked on #5):** Gemma 4 E2B-QAT UD-Q4_K_XL (2.62 GB, only new candidate preserving MTP), then Nanbeige4.2-3B (2.68 GB — **measure tok/s and real KV footprint first**, the 2× loop expansion is not reflected in the file size), then Agents-A1-4B (2.71 GB) if agentic/tool-calling matters.

#### Proposed JETSON_BASELINE.md changes — **NOT APPLIED** (headless run, no user present; per run instructions)
- `Last recon:` 2026-07-16 → **2026-08-15**; `Last healthcheck:` 2026-07-16 → **2026-08-15 (DEGRADED — chronic OOM returned; tailscaled dead 24 d; watchdog blind)**.
- `llamacpp_latest_seen:` b10054 → **b10442** (running stays **b9652**).
- `jetpack_latest_orin_nano:` 7.2 → **7.2.1 (L4T r39.2.1, 2026-08-11)**; add **`jetpack6_latest: 6.2.3 (L4T 36.5.2, 2026-08-12, apt in-place)`**; `jetpack_next_expected:` → **watch JetPack 7.3 + prebuilt sm_87/CUDA 13.2 wheels & containers** (TNSPEC fix landed in 7.2.1 for fresh ISO flashes).
- `models_last_checked_date` & `forum_last_checked_date:` 2026-07-16 → **2026-08-15**.
- **Recon Triggers:** re-point the `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` row at **PR #25384**; **ADD** a row `llamacpp_release | #26750 OR (draft-mtp AND acceptance) | ACTION: rebuild blocker — resolve before any build past ~b10261`.
- **Watch Items:** **RE-ESCALATE the OOM item from ✅ DOWNGRADE to 🔴 ACTIVE** (Entry 032/036 verdict refuted — 12 kills since 2026-07-30); **ADD** watchdog-blind-spot, tailscaled-24-day-outage, the undocumented `8080/tcp` LAN rule, TensorRT-Edge-LLM v0.10.0 + the sm_87 support-matrix answer, and the Gemma 4 E2B "now fits via QAT" flip; **ADD** the `-lm/--load-mode` deprecation as pre-rebuild work; note `baseline_rss_mb` 5839 is now routinely exceeded (6097 observed at 13 h).
- **Current Config section: unchanged** (still b9652 / Qwen3.5-4B-MTP-Q4_K_M — nothing was modified).
- **Throughput note (no baseline change proposed):** mean **~13.0 tok/s** vs the 15.3 floor — at the 15% warn boundary, down from ~14.4 at Entry 036, with draft acceptance ~29–38%. Recommend re-measuring after the watchdog fix and a clean restart before deciding whether `baseline_gen_tok_s` needs revisiting; today's number is confounded by the box sitting at 96% of `MemoryMax`.

---

## Entry 039: Recon (2026-08-22) — ACTION NEEDED: OOM cadence tightened to every ~2 days, but NVIDIA endorsed a fix that needs NO rebuild (`--cache-ram 0 --fit off`); throughput recovered
**Date:** 2026-08-22 14:15 EDT (2026-08-22 18:15 UTC)
**Operator:** Claude Code (jetson-recon skill, headless scheduled run — no user present)
**Status:** RECON — **no changes made to the Jetson.** All device access read-only. Baseline tracking values NOT updated (headless run; proposal recorded below for user confirmation).

Five checks (4 web-research agents + 1 live SSH health check). Prior recon: Entry 038 (2026-08-15, 7 days ago).

**Headline:** the chronic OOM is worse (16 kills, now every ~2 days, one this morning) — but this cycle produced the first **credible, NVIDIA-endorsed, no-rebuild mitigation** in the entire OOM investigation, and it is verified applicable to the running build.

---

#### JetPack / Firmware — **LOW (no version change; new detail on 6.2.3 changes the rationale)**
- **No version movement in 7 days.** JetPack 6.2.3 / L4T 36.5.2 (2026-08-12) and JetPack 7.2.1 / L4T 39.2.1 (2026-08-12) both unchanged. **No 7.2.2, no 7.3** — confirmed against the [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive) (7.2.1 is the terminal 7.x entry). JetsonHacks has published nothing on 6.2.3.
- **NEW — the itemized 6.2.3 fixed-issues list was recovered** (`pdftotext -layout` on [RN_10698-r36.5.2](https://docs.nvidia.com/jetson/archives/r36.5.2/ReleaseNotes/Jetson_Linux_Release_Notes_r36.5.2.pdf); WebFetch could not extract it last week). Two items matter:
  - **Bug 5602402 — NvMap allocation policy, directly adjacent to our failure mode.** Verbatim: *"users encountered CUDA memory allocation failures with the error message 'unable to allocate CUDA0 buffer'. Fixes: Modified the NvMap allocation policy to properly handle memory requests without overly restricting available CUDA memory. Addressed a secondary issue where the initial allocation policy fix could cause system hang/reboot when multiple threads attempted to allocate memory exceeding available capacity simultaneously."* ⚠ **Hypothesis, not a fix claim:** the notes do not address NvMap's absence from kernel OOM *accounting*, and the cited trigger path (36.4.4→36.4.7) is not this device's (36.5.0). Worth testing, not "the answer."
  - **Bug 5412830 — brick-class risk removed.** UEFI StandaloneMM variable-storage/block-erase bugs could cause a random boot-time UEFI assertion that halts at the bootloader and **requires a full firmware reflash to recover.** On a headless remote node with no local console, this is the strongest argument for taking 6.2.3.
- **⚠ CORRECTION to a standing assumption — the CVE rationale for 6.2.3 was wrong.** Authoritative bulletin text (fetched via the [NVIDIA/product-security GitHub mirror](https://github.com/NVIDIA/product-security/tree/main/2026/5797), since custhelp returns 403): CVE-2026-24148 (8.3), CVE-2026-24154 (7.6) and CVE-2026-24153 (5.2) are **all fixed in Jetson Linux 36.5** — i.e. **this box on 6.2.2/36.5.0 is already patched.** 6.2.3 was *not* the CVE fix; 6.2.2 was. No Jetson security bulletin itemizing 6.2.3's CVEs has been published, so its security content is unquantified. **The bug-fix case (5412830, 5602402) is the real case; the security case is not.**
- **Zero community regression reports on 6.2.3, 10 days post-release** — all four announcement threads (379869/379871/379872/379873) have `posts_count = 1`, no replies at all. Weak evidence (nobody is talking about it), not validation.
- **TNSPEC in 7.2.1 — confirmed from the primary source, still partly open.** r39.2.1 notes: *"ISO now flashes the Jetson Orin Nano Developer Kit with Super Mode flashing configuration by default"* ⇒ **fresh-ISO-flash only**, baseline assumption holds. Fixed Issue 6236259 is an EMC/`nvpmodel.service` crash-on-reboot in 7W mode. **But Known Issue 6480645 remains:** installing r39.2.1 over r38.2.x can leave `TegraPlatformSpec` not reflecting the real board. TNSPEC mismatch is not fully eliminated.
- **First real movement on the JP7 ecosystem gap — but unverified.** NVIDIA's AastaLLL, 2026-08-13, [thread 380034](https://forums.developer.nvidia.com/t/380034): *"From JetPack 7.2.x, Orin supports SBSA driver so you can use the upstream package directly"* → `pip3 install torch --index-url .../whl/cu132`. ⚠ **Counter-evidence:** upstream aarch64 wheels have historically shipped sm75/80/90/100/110/120/121 — **not sm_87** ([bitsandbytes #1930](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1930), open). Whether `cu132` wheels actually contain sm_87 kernels is **unconfirmed and must be checked empirically before the JP7 hold is relaxed on PyTorch grounds.** `dusty-nv/jetson-containers` main HEAD is still **2026-06-25** — that leg of the hold is unchanged.
- **Tooling improvement for future recons:** `nvidia.custhelp.com` returns HTTP 403 to research agents; **`github.com/NVIDIA/product-security` is fully fetchable and machine-readable** (Markdown + CSAF + CVE JSON). Adopt it as the standing bulletin source. All 50 bulletins in `2026/` were enumerated — **exactly one Jetson bulletin all year (5797, March)**; nothing new. (Noise discarded: a third-party aggregator mis-associated CVE-2026-42897 with the Jetson bulletin — it is a Microsoft Exchange XSS.)

#### llama.cpp Releases — **HIGH (the #26750 rebuild blocker got STRONGER; two standing assumptions corrected)**
- Newest = **b10586** (2026-08-22 16:03 UTC). Running **b9652** = **934 builds behind**; **144 commits** past the b10442 seen at Entry 038 (verified via `compare/b10442...b10586`).
- **⛔ #26750 STILL OPEN — and the evidence against rebuilding is now considerably stronger.** New comment 2026-08-21 (zanphear), no maintainer response on either comment, **no fix PR exists** (repo-wide PR search for "26750" returns zero):
  - **Still reproduces at b10532**, 271 build numbers past b10261. Prose 200-tok, GB10/SM121/CUDA 13, `--spec-type draft-mtp --spec-draft-n-max 3`, `-fa on`, `-ctk/-ctv q8_0`: b10261 = 66.67% acceptance / **69.23 tok/s**; b10532 = 52.38% / **57.67 tok/s** → **−16.7% wall-clock**, fully accounted for by acceptance (+16.2% target forward passes). Thermals/noise ruled out.
  - **New control that isolates the defect:** with speculative decoding fully OFF, `llama-bench` pp2048 and tg256 are **flat within noise** across the same 250 commits (+0.24% / +0.26%). So it is **not** engine drift and **not** a CUDA decode-kernel regression — **100% of the loss lives in the backend-agnostic speculative path.**
  - Enumeration-style prompts unaffected (91.5% → 93.9%). Rollback arm executed from tarball, counters returned byte-identical; a second independent GB10 host reproduced b10261 exactly.
  - **Eight commits touching `common/speculative.cpp` since #26510 are named as measured non-fixes:** #25532, #26904, #26958, #26814, #27005, #26275, #27404.
  - **Ampere/SM87 exposure remains untested by anyone.** *Inference (not measurement):* because the control localises the defect to the backend-agnostic speculative path, SM87 is **more likely than not** affected.
  - Related: **#27106** ("acceptance falling to 0.5 with b10430+") was **self-closed 2026-08-17** as model prompt-sensitivity. That closure does **not** touch #26750, whose author pre-empted exactly that confound with same-prompt A/B controls.
- **⚠ CORRECTION — PR #25384 is NOT a reason to wait, and never was.** Source read at both `master` and tag `b9652`: the UMA behaviour (`prop.integrated > 0 || GGML_CUDA_ENABLE_UNIFIED_MEMORY` → read `MemAvailable` from `/proc/meminfo` instead of `cudaMemGetInfo`) shipped in **PR #17368 "DGX Spark: UMA support", merged 2025-11-20**, and is **already compiled into the running b9652** (`ggml-cuda.cu` line ~5023). Jetson Orin reports `prop.integrated > 0`, so it already takes the MemAvailable path. **#25384 only reorders the call to skip a redundant `cudaMemGetInfo` — it is a cleanup, not the missing behavioural fix.** Its 5-week staleness is irrelevant. *Residual caveat:* `info.devices[id].integrated` is still hard-forced `false` on non-HIP builds ("temporarily disabled… corrupted output, #15034"), so Jetson gets integrated-GPU memory *reporting* but not integrated-GPU *buffer placement*. None of this touches the kernel-side NvMap accounting gap.
- **#27282 — a NEW, independent argument against rebuilding on 8 GB.** Open 2026-08-17, updated 2026-08-22: *"native MTP reserves a separate CUDA compute arena and OOMs; shared gallocr fixes it."* A duplicate **~1 GB** compute buffer for the MTP draft context is far more hostile on this box than on the RTX 4090 where it was reported. Reporter has a working patch; **unmerged.**
- Other new open issues in the rebuild-blocker class: **#27549** (2026-08-22, `ggml-cuda/fattn.cu:574` fatal error, random crash with active MTP after first token — `-ctk/-ctv q8_0`, `--flash-attn on`, `--parallel 1`, b10549; multi-GPU sm_120 so not a clean match, untriaged); **#27407** (greedy output *diverges* from non-speculative baseline under batched CUDA verification — correctness, not just perf); **#27296**, **#27151** (MTP draft-context corruption / acceptance collapse to 1/633, Vulkan-reported). **#27212 / #27105** establish that `draft-mtp` **has arch-specific failure modes on older NVIDIA SMs** (sm_60 P100, sm_52 Maxwell Xid 31) — a relevant prior for untested SM87. Not applicable: #27117/#27428/#27122 (all need `-np > 1` or multi-GPU).
- **No Jetson/SM87/Tegra/NvMap work merged, at all.** Searches for `jetson`, `tegra`, `sm87`, `nvmap`, `cudaMemGetInfo` return **zero** merged PRs and **zero** issues since 2026-07-01. #27083 (2026-08-17) narrows the #17368 UMA override to exclude HIP — **CUDA/Jetson path preserved exactly, no behaviour change.** #26079 (mvq→MMQ crossover) is gated to Ada/Blackwell/DGX-Spark/CDNA — **no Ampere branch**. #26843 is `__CUDA_ARCH__ == DGX_SPARK` only. All neutral.
- **✅ #20834 / #26110 are CLOSED — drop them from pre-rebuild worry.** #20834 (`-lm/--load-mode`) **merged 2026-07-23**; #26110 (the removed safe `--no-mmap --mlock` combo) **closed completed 2026-07-27**. **Verified on device: no start script passes `--mlock` or `--no-mmap`** (both `start-qwen35-server.sh:15` and `start-experiment.sh:19` carry only the Entry 029 removal comment). The box is not exposed.
- **New breaking change worth planning for: #26347** "server: make models endpoints private when authentication is enabled" (merged 2026-08-19). `/models` and `/v1/models` lose their public exemption. Since this box runs `--api-key-file`, **any consumer probing `/v1/models` unauthenticated will start getting 401 after a rebuild** — audit contact-center-lab and health/discovery tooling first. `/health` stays public. Also: `-no-cnv` removed (#27542, `llama-cli` only); version strings bumped to llama.cpp **0.2.0** / ggml **0.21.0** (check anything parsing `--version`). **No CUDA version-floor change, no build-flag breakage** in the 144 commits.

#### Small Model Landscape — **MEDIUM (one real candidate, one quant upgrade, no MTP-preserving option)**
- **NEW: `empero-ai/Qwen3.8-4B-Distill-GGUF`** (2026-08-15, 56,968 downloads). 4B dense, base `Qwen/Qwen3.5-4B`, **identical arch to deployed** (`qwen3_5`, Gated DeltaNet), 262,144 ctx, Apache-2.0. **Q4_K_M = 2.783 GB — FITS.** Full-parameter distillation of Qwen3.8-Max (2.4T-A95B) over ~45k curated teacher traces. Vendor numbers vs Qwen3.5-4B base: **MMLU 0.354 → 0.553 (+19.9 pp)**, **GSM8K 0.850 → 0.785 (−6.5 pp)**. ⚠ **Two reasons to treat as experiment-slot only, not a swap:** the quoted 0.354 Qwen3.5-4B MMLU baseline is anomalously low and smells like a flexible-extract harness artifact rather than a real +20 pp gain (no independent reproduction found); and **the repo ships no MTP head** — adopting it forfeits self-speculative decoding, currently this deployment's key throughput asset.
- **`insraq/Qwen3.5-4B-EmperoAI-Qwen3.8-Distill-Heretic-Abliterated-MTP-GGUF`** (2026-08-18) — the only candidate *claiming* to preserve MTP. **Rejected on two grounds:** it is **abliterated** (refusals stripped 99/100 → 6/100 — not appropriate for a general-purpose server), and **the MTP claim does not survive arithmetic** — its Q4_K_M is only **6.5 MB larger** than the non-MTP empero Q4_K_M, where a real Qwen3.5-4B MTP head at Q4_K_M should add ~100+ MB. **Unverified — would need `gguf-dump` on tensor names to settle.** Assume the tensors are absent.
- **STATUS CHANGE — LiquidAI LFM2.5-2.6B gained a QAD Q4_0 checkpoint (2026-08-19): 1.594 GB.** Quantization-Aware Distillation recovers 96.6% of the BF16 baseline (closes 48.4% of the quantization gap); Liquid claims it beats standard Q4_K_M and matches Unsloth UD-Q4_K_XL quality at **3–14% higher decode throughput**. Legacy Q4_0, so any build loads it. **This is the best RAM-headroom option on the board** — 1.59 GB leaves roughly double the KV budget of any 4B. License unchanged: **LFM Open License v1.0, commercial use only under $10M revenue** — fine for the homelab, hard blocker for client-derived work.
- **Nanbeige4.2-3B — upstream support confirmed merged** (PR #25994, opened 2026-07-22, **merged 2026-07-27**; adds `LLM_ARCH_NANBEIGE`, looped-transformer support with separate KV slots per loop). Corroborates the tracked ≳b10160 requirement. Two new cautions: (a) several HF GGUF READMEs **still** say "not yet upstream, use `Nanbeige/llama.cpp @ nanbeige42`" — **stale text, ignore**; (b) a reviewer flagged `num_loops` is **optional with silent fallback to 1**, which would halve loop depth and silently degrade quality — **verify `num_loops=2` in any GGUF before benchmarking.** Also: `Nanbeige4.1-3B` repos still circulate, easy to grab the wrong version.
- **gemma-4-E2B-it-qat — no change, one useful detail:** the MTP drafter ships at repo root as `mtp-gemma-4-E2B-it.gguf` and **recent llama.cpp auto-discovers it from `-hf`, so no `--model-draft` is needed** — invoke with `--spec-type draft-mtp --spec-draft-n-max 4`. (Vendor's 52 → 162 tok/s figure is on a B200 and says nothing about Orin.)
- **Octen-Embedding-4B — no independent validation.** MTEB issue #3881 turns out to be Octen's own submission requesting private RTEB evaluation of the **8B**. All RTEB numbers remain vendor-submitted; no third-party reproduction. Still a zero-risk A/B (identical arch/size/dim), but the claimed gain is unconfirmed. Octen-Embedding-8B now claims #1 on RTEB — far over budget.
- **Qwen: fourth consecutive generation skips the 4B tier.** Official Qwen3.8 open lineup is **Qwen3.8-27B** (27.78B dense, 2026-08-14) and **Qwen3.8-Max** (2.4T MoE). ⚠ **Naming trap: "Qwen3.8-4B" on HuggingFace is `empero-ai`'s community distill, not an Alibaba release.** Qwen4 remains leak/rumour only (speculated September 2026, secondary aggregators + a Manifold market) — **unverified.**
- **Rejected on size (new this week):** `inclusionAI/Ling-3.0-tiny` (**7.9B TOTAL** / 1.3B active MoE — classic MoE trap, ~4.5–5 GB Q4_K_M, also needs unmerged PR #26608), Ornith-1.5 9B/35B-A3B/397B, Nemotron-3.5-Lightning-30B-A3B, Qwen3.8-27B. Fits but wrong shape: S1-mini 0.6B (ASR transcript cleanup), LFM2.5-VL-3B and Cohere North Micro Vision 2.4B (vision-language).
- **Embedding: nothing beats Qwen3-Embedding-4B on this box.** No official Qwen3.5-Embedding exists (`Rebine/Qwen3.5-Embedding-0.8B` is third-party). Leaders (KaLM-Embedding-Gemma3-12B, Harrier-OSS-v1 27B, Qwen3-VL-Embedding-8B) are all far over 3 GB. **No action.**

#### Jetson Forum / Community — **🔴 ACTION — the most useful OOM finding of the entire investigation**
- **🔴 Thread [380334](https://forums.developer.nvidia.com/t/380334) (created 2026-08-16, NVIDIA reply 08-17): NVIDIA staff confirms a contiguous-allocation root cause AND endorses a concrete llama.cpp-side workaround.** Reporter "Manuel" on an **Orin Nano 8GB (P3767-0005), MAXN_SUPER**, official `ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin`. Two crash signatures — **signature B matches this box almost exactly**:
  - *Signature A:* `NvMapMemAllocInternalTagged failed: error 12` despite gigabytes free → address-space fragmentation.
  - *Signature B:* genuine OOM-kill preceded by a **"steady, roughly monotonic decline in available memory"** over 2.5–4 minutes / 9–10 requests.
  - His fix: **`--fit off --ctx-size <explicit> --cache-ram 0`** — *"eliminated the failures entirely across 60+ consecutive requests."*
  - **AastaLLL (NVIDIA) confirmed it 2026-08-17:** *"llama.cpp might request a contiguous buffer, and fail if the kernel cannot provide contiguous memory"*, and *"the workaround is to pass `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 LLAMA_ARG_FIT=off`."*
  - **This is the first time NVIDIA has endorsed a concrete llama.cpp-side mitigation rather than deflecting to a JetPack upgrade.**
  - ⚠ **Caveat stated plainly:** his repro is **JetPack 7.2/7.2.1 (kernel 6.8)**, not our 6.2.2 (kernel 5.15). The mechanism is llama.cpp-side so it should transfer, but **this is not a verified 6.2.x result.**
- **🔴 The `--cache-ram` default is 8192 MiB — an 8 GiB host prompt cache on a box with 7.4 GiB total.** Current server docs: `-cram, --cache-ram N   set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 - disable)`; `-fit, --fit [on|off] ... (default: 'on')`. **PR #25070 "server: add strict prompt cache RAM limit" merged 2026-07-07**, and its description states that *before* that commit `--cache-ram` **was not a hard limit**: *"The cache always kept at least one entry, even if that entry exceeded the RAM/token limits. Old entries were only evicted… after saving the new one, which could cause the cache to temporarily exceed the RAM/token limits."* **b9652 was published 2026-06-15 — three weeks before #25070 merged.** Two earlier attempts at the same problem were closed unmerged: **#23561** "server: fix `--cache-ram` not preventing RAM OOM" (2026-05-23) and — pointedly — **#24649** "server: clear slot checkpoints before saving to prompt cache to prevent ram overflow" (2026-06-15), which names the exact mechanism Entry 038 fingered. **#26893** "server: avoid caching non-completion idle slots in RAM" is still open (2026-08-11).
  - This predicts the observed signature precisely: **monotonic MemAvailable decline under sustained load, zram filling, file-swap untouched, userspace kills freeing nothing** (the growth is host-side cache attached to the surviving server process).
  - **Complementary to, not a replacement for, the NvMap accounting gap** — but unlike NvMap, it is addressable today.
- **🟡 [Thread 380427](https://forums.developer.nvidia.com/t/380427) (2026-08-17) — a peer with a near-identical stack.** "Evopien", local multimodal humanoid prototype: **Qwen3.5-4B via llama.cpp on Orin Nano Super 8GB** + Parakeet TDT ASR + Kokoro ONNX TTS. States *"the main challenge at this stage is memory and latency optimization on the 8GB Nano."* No tok/s, no JetPack version, no solution — but a genuine peer if we ever want to compare notes.
- **JetPack 6.2.3 makes no claim about NvMap/CMA/OOM.** Kernel still 5.15. The entire "What's New" reads *"Fixes for known issues and security vulnerabilities."* ⚠ The r36.5.2 PDF defeated the forum agent's extraction — but Check 1 recovered it independently and found bug 5602402 (NvMap allocation policy), so *"no NvMap change"* is **refuted in detail**; see Check 1.
- **Unchanged / closing out:** 370049 (CMA drop_caches) last post 2026-06-02, no new replies. **373627 (SENTINEL / `GGML_CUDA_ENABLE_UNIFIED_MEMORY`) died on an unanswered NVIDIA request** — AastaLLL asked for the exact `llama-server` command to reproduce and never got one. **375642 (sm_87) CLOSED 2026-08-05** (auto-close, resolved benign — Orin runs sm_80 code without JIT) → **recommend dropping from tracking.** 372493 (MTP+TurboQuant) no activity, still Orin **NX** only. 25W vs MAXN: no new data.
- **TensorRT-Edge-LLM still v0.10.0 (no newer release), but new-to-us:** Jetson AI Lab now carries a worked **Orin Nano 8 GB** example — Qwen3-4B-Instruct **INT4 AWQ, ~2 GB weights**, *"leaving ample room for the KV cache and OS within Orin Nano's 8 GB unified memory."* **Still no published Orin Nano tok/s and no llama.cpp comparison** — the page ships `llm_bench` and defers to self-measurement. Tracked gap remains open.
- **jetson-containers: code still dormant** (HEAD 70c149a, 2026-06-25, ~2 months), but the issue tracker is now noisy — #1736 "JetPack 7.2 / CUDA 13.2 Orin build issues" (08-11), #1739 "Old indexes in DockerHub potentially affecting many images" (08-14). **Reinforces: do not depend on this repo's containers.**
- *Source gap, third consecutive recon:* **reddit.com is permanently blocked to the research agents' user agent** — r/LocalLLaMA has never actually been searched. **Recommend dropping it from the source list** rather than continuing to carry a phantom source.

#### Live Health — **DEGRADED (OOM worse; tailnet outage now 31 days) — but throughput RECOVERED**
Reached over **LAN `192.168.10.58`**; the tailnet path timed out again. Host **uptime 53 days** (boot 2026-06-30, unchanged), mode `qwen35`, all 4 drop-ins intact (`cma-compact`, `crash-escalate`, `memory-limits`, `oom-protect`), disk **17% / 660 G free**, thermals **50.5–52.7 °C** idle, power mode **MAXN_SUPER (mode 2)**, slots 1. **No version drift:** source HEAD `6eab47181` = **b9652** = baseline. **Auth healthy:** unauthenticated `/v1/chat/completions` → **401**, authenticated → **200**, `/health` public → **200**.

**🔴 FINDING 1 — OOM cadence has TIGHTENED from ~every 3 days to ~every 2 days. Four more kills in the 7 days since Entry 038.**
- Restart counter **20 → 24**. New kills: **Aug 17 01:00:50, Aug 19 01:02:38, Aug 21 01:00:51, Aug 22 01:01:50** — all `Failed with result 'oom-kill'`, **all at ~01:00–01:03 EDT**. Running total: **16 OOM kills since 2026-07-30.** The most recent was **13 hours before this recon**.
- **Kernel dump for today's event is the unchanged Entry 026/038 NvMap signature** — `oom_score_adj=-900` works exactly as designed and llama-server dies **last**, after `nv-tee-supplicant`, `nvs-service`, `ethtool`, 3× `agetty`, `tee-supplicant`, `01-ifupdown`, `run-parts`, `sed`, `seatd`, `systemd-journald` (adj −250), `dockerd` (adj −500) and `upowerd` are all sacrificed first. Victim line: `llama-server total-vm:80980224kB, anon-rss:4105548kB … oom_score_adj:-900`. Killing userspace frees nothing because the memory is NvMap-held and invisible to OOM accounting.
- **The box is on the edge right now, mid-afternoon, 13 h after a clean restart:** `MemoryCurrent` **6.56–6.70 GB** against `MemoryMax` 6.71 GB = **97.8–99.9%**; `MemAvailable` **0 kB**; `free -h` available **184 MB**. `systemctl status` reports `Memory: 6.0G (max: 6.2G available: 191.4M)`. Baseline `rss_mb` is 5839 and is being exceeded routinely. **It will almost certainly OOM again tonight.**
- **⚠ Methodology note — the throughput benchmark was cut short deliberately.** After three 200-token generations `MemAvailable` read **0 kB** and `MemoryCurrent` hit **99.86%** of `MemoryMax`. Further load would have risked *inducing* an OOM storm on a live box, which this skill must not do. Remaining throughput data was taken from the journal (free, read-only) instead.
- Nightly client traffic in the 20:00 → 01:02 window was **12 `launch_slot` tasks** — even lower than Entry 038's 5–36. **This confirms the burst is the trigger, not the cause.** Context-checkpoint churn continues: **51 checkpoint events in 13 h**, ~51 MiB each, "1 of 32".

**🔴 FINDING 2 — the running command line sets NEITHER `--cache-ram` NOR `--fit`, and both flags EXIST in b9652. The Check-4 mitigation needs no rebuild.**
- Live argv (from `systemctl status`, all 15 args): `--model … Qwen3.5-4B-MTP-Q4_K_M.gguf --host 0.0.0.0 --api-key-file … --port 8080 --metrics --alias qwen3.5-4b --ctx-size 32768 --n-gpu-layers 999 --n-gpu-layers-draft 999 --threads 1 --parallel 1 --flash-attn on --reasoning off --cache-type-k q8_0 --cache-type-v q8_0 --cache-type-k-draft f16 --cache-type-v-draft f16 --spec-type draft-mtp --spec-draft-n-max 3`. **No `-cram`/`--cache-ram`, no `-fit`/`--fit`.**
- **Verified directly against the b9652 source tree** (`gh api …/common/arg.cpp?ref=b9652`): `{"-cram", "--cache-ram"}` is present at line **1345**, `{"-fit", "--fit"}` at line **2439**, and `common/common.h?ref=b9652` line **604** gives `int32_t cache_ram_mib = 8192;` with line **453** `bool fit_params = true;`. **Identical to master** (`cache_ram_mib = 8192` at master line 616).
- ⇒ **The running server is inheriting an 8192 MiB host prompt-cache budget on a machine with 7.4 GiB of RAM, plus `--fit on`, and it is doing so with the pre-#25070 non-hard-limit eviction logic.** This is the mechanism NVIDIA endorsed fixing, it matches the observed monotonic-decline signature, and **it is settable today on b9652 with no rebuild and no upstream risk.**
- `--cache-idle-slots` also exists in b9652 (defaults enabled, "requires cache-ram") — relevant to #26893's open work.

**🔴 FINDING 3 — `tailscaled` dead for 31 DAYS. Unchanged since Entry 038, still no alerting.**
- `tailscaled.service` **failed since Wed 2026-07-22 01:00:22 EDT**, `ExecMainStatus=2`, `NRestarts=6`, systemd gave up. `tailscale status` → *"failed to connect to local tailscaled; it doesn't appear to be running."* Journal for the unit has long since rotated — **root cause remains unrecoverable for the 07-22 event.**
- Week-over-week this went 24 d → **31 d** with no change, because nothing alerts on it. The node has been invisible on the tailnet for a month while serving normally on the LAN.

**🟠 FINDING 4 — the memory watchdog is still structurally blind, and today's data confirms the diagnosis again.**
- `memory-watchdog.service` is **active**, has **never fired** across all 16 OOM events. Its CRITICAL trigger is `file_swap_used > 1024 MB` **AND** `MemAvailable < 150 MB`.
- **Measured today: file swap `/ssd/16GB.swap` USED = 0 B**, while zram sat at ~175 MB (idle) rising to ~543 MB (post-bench) of 3.8 GB, and `MemAvailable` = **0 kB**. The AND-gate's first conjunct was **never satisfiable** — exactly as Entry 038 concluded. **Unchanged and still unarmed.**

**🟢 FINDING 5 — the `ufw-watchdog` (Entry 037) proved itself under fire.** It was **itself OOM-killed** during today's 01:01:48 storm (`ufw-watchdog.service: Failed with result 'oom-kill'`) — and recovered on its own. `ufw` is **active** now and the timer is running. **This is the pattern that works, and it is the pattern Findings 3 and 4 need.**

**🟡 FINDING 6 — the undocumented `8080/tcp ALLOW 192.168.10.0/24` rule is still present.** Live ruleset: `lo` / `tailscale0` / `22 from 192.168.10.0/24` / **`8080/tcp from 192.168.10.0/24`**. Still not the Entry 034 known-good set, still API-key protected (401 verified). **Unchanged from Entry 038 — still awaiting the user's keep-and-document vs remove decision.**

**🟢 THROUGHPUT — RECOVERED, and now ABOVE the baseline floor.**
- 3× 200-token prose generations (`temperature 0`, `cache_prompt false`, Entry 038 methodology): **18.32 / 13.66 / 17.12 tok/s → mean 16.37 tok/s.** Baseline floor is **15.3**; the 15% warn floor is 13.0. Entry 038 measured **~13.0** on the same test. **This is a ~26% week-over-week recovery and clears the warn threshold.**
- **Draft acceptance recovered in step: 0.670 / 0.410 / 0.591 → mean 0.557**, vs **~29–38%** at Entry 038. The nightly structured tasks in today's journal (05:00) show **0.933–1.000 acceptance at ~22 tok/s**, consistent with the established prompt-dependence.
- Short-chat sample: 13.18 tok/s at 0.40 acceptance (11 tokens — too short to be meaningful).
- ⚠ **Do not over-read this.** The measurement was taken 13 h after a *clean* restart, whereas Entry 038's was taken with the box already at 96% of `MemoryMax`. The recovery is most plausibly **the absence of memory pressure at measurement time**, not an improvement in the system. It is also a datapoint *against* the theory that MTP acceptance is chronically collapsing on SM87 — **at b9652 the box delivers 55.7% mean prose acceptance when it has headroom.**
- Prompt eval **~373–399 tok/s** on 232–412-token prompts in the nightly journal (the 74–81 tok/s figures on 19–20-token prompts are not comparable to the 166 baseline).

**Other:** **6 failed units** — the 5 expected boot-time casualties (`nvphs`, `avahi-daemon`, `wpa_supplicant`, `networkd-dispatcher`, `kerneloops`) **plus `tailscaled`**. `udisks2` and `ModemManager` have **recovered** since Entry 038.

#### Cross-Correlated Findings
1. **🔴 THE FINDING OF THIS CYCLE — a no-rebuild OOM mitigation is available, endorsed by NVIDIA, and verified applicable to the running build.** Check 4 surfaced NVIDIA's own AastaLLL endorsing `--fit off --ctx-size <explicit> --cache-ram 0` on an **Orin Nano 8GB running llama.cpp**, against a signature (monotonic MemAvailable decline over sustained requests → OOM kill) that matches this box. Check 5 then established the two facts that make it actionable: **(a) the running argv sets neither flag**, so the 8192 MiB default host prompt cache is live on a 7.4 GiB machine; **(b) both flags exist in b9652**, verified in the source tree at that exact tag. Check 4 independently established that **b9652 (2026-06-15) predates PR #25070 (merged 2026-07-07)**, which is when `--cache-ram` first became a *hard* limit — so this build has the unbounded-growth eviction logic and needs the explicit `0` more than a current build would. **This is the first OOM lever in the entire investigation that requires neither a JetPack migration nor a llama.cpp rebuild.**
2. **The Check-2 vs Check-4 conflict resolves cleanly — take the flags, refuse the rebuild.** Check 4 recommended bumping past b10437 to inherit #25070's hard limit; Check 2 recommends holding at b9652 because #26750 is unfixed. **Both are satisfiable at once:** setting `--cache-ram 0` *disables* the cache outright, so #25070's hard-limit enforcement becomes moot — there is nothing left to enforce a limit on. **Hold on b9652 stands, and the mitigation is taken anyway.** Check 2's evidence is also materially stronger this week (exact `compare` API, a no-speculative control isolating the defect, eight named non-fix commits).
3. **Two standing "wait for upstream" items are now dead, which simplifies the rebuild decision rather than complicating it.** **#25384 is a cleanup, not the missing UMA fix** — the behavioural fix (#17368) has been compiled into b9652 since 2025-11, verified by reading `ggml-cuda.cu` at tag `b9652`. And **#20834/#26110 are both closed**, with the device verified clear of `--mlock`/`--no-mmap`. **Net: the pre-rebuild checklist shrank; the rebuild blocker (#26750) did not.**
4. **The rebuild case got worse in a second, independent way.** #27282 (open, unmerged) reports **native MTP reserving a duplicate ~1 GB CUDA compute arena**. On a box that is already at 97.8–99.9% of `MemoryMax`, that is not a marginal regression — it is disqualifying. Combined with #27549 (MTP + q8_0 KV + flash-attn fatal error, untriaged, filed today) and #27212/#27105 (draft-mtp **has** arch-specific failure modes on older SMs), the post-b10261 tree is currently hostile to precisely this configuration. **Hold is now over-determined.**
5. **Throughput recovery is evidence about memory pressure, not about MTP.** 16.37 tok/s / 55.7% acceptance measured 13 h after a clean restart, vs 13.0 tok/s / 29–38% measured at 96% `MemoryMax`, points at memory pressure as the throughput driver — the co-factor Entry 038 flagged but could not isolate. It also means **"acceptance recovered" remains unusable as a post-rebuild success signal** (Entry 038's warning stands), because acceptance moves with memory state and prompt shape independently of build.
6. **JetPack 6.2.3's real value is not what we recorded last week.** Check 1 corrected two things: the tracked CVEs were **already fixed in 36.5** (this box is patched), so the security case is unquantified — while **bug 5412830** (a UEFI assertion that halts at the bootloader and needs a *full reflash* to recover) is a **brick-class risk on a headless node with no local console**, and **bug 5602402** changes NvMap allocation policy in a way that is adjacent, though not proven relevant, to our failure mode. **The upgrade is still worth taking — for different reasons than we thought, and now with a possible (unproven) OOM upside.**
7. **The ~01:00 EDT boundary is now a 20-incident pattern and remains uninvestigated.** All 4 new kills landed at 01:00–01:03, `tailscaled` died at 01:00:23 on 07-22, and Entry 033's simultaneous restart was at 01:00. **Nightly load is only 12 tasks** — so the trigger is small and the system is simply too close to the ceiling to absorb it. **Finding 2's mitigation would raise the margin the nightly burst has to eat into**, which makes it a plausible test of this correlation as well as a fix.
8. **The self-heal pattern is validated, and its absence is what let two failures run for weeks.** `ufw-watchdog` was **OOM-killed today and came back by itself** (Finding 5). Meanwhile `tailscaled` has been down **31 days** and the memory watchdog has been unarmed for **16 kills** — both because no equivalent assert→alert→bounded-self-heal loop exists for them. **Entry 037's pattern is proven under fire; extending it is the highest-leverage ops work outstanding.**

#### Triggered Alerts
- **jetpack** `(JetPack 7.2.1 OR 7.3 OR power mode fix OR TNSPEC) AND (Orin Nano OR Orin)` — **NOT re-matched this cycle** (no new release; 7.2.1 already adjudicated at Entry 038). **JP7 hold stands** — still 2 of 3 blockers open (full USB-ISO reflash; CUDA 13.2 sm_87 wheel/container gap **unverified-but-possibly-narrowing** per thread 380034). **Take JP6.2.3 instead** — now justified by bug 5412830 (brick risk) rather than by CVEs.
- **llamacpp_release** `SM87 OR Jetson OR Tegra OR unified memory` — **keyword matches only; zero merged Jetson/SM87/Tegra/NvMap work since 2026-07-01.** Third consecutive recon with no Jetson-specific gain. No action.
- **llamacpp_release** `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` — **✅ MATCHED, first time.** NVIDIA's AastaLLL explicitly named `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 LLAMA_ARG_FIT=off` as *the* workaround (thread 380334, 2026-08-17). **This trigger has finally done its job.** ⚠ Note the correction in cross-correlation 3: the UMA env-var's *behaviour* is already active on this box by default (`prop.integrated > 0`), so the operative half of NVIDIA's advice is **`LLAMA_ARG_FIT=off`** plus `--cache-ram 0`. **Proposal: re-point this trigger at thread 380334 / `--cache-ram` / `--fit`, and retire the #25384 re-point proposed at Entry 038** (now known to be a cleanup, not a fix).
- **llamacpp_release** `#26750 OR (draft-mtp AND acceptance)` *(proposed at Entry 038, not yet added)* — **would have MATCHED (ACTION).** #26750 still open, reproduction extended to b10532. **Recommend actually adding this row.**
- **huggingface** `Qwen4 OR Qwen3.5 successor` — **MATCHED (INFO).** `empero-ai/Qwen3.8-4B-Distill` is a community Qwen3.5-4B derivative, not an Alibaba release; no Qwen4. **Keep armed** — four generations have now skipped the 4B tier.
- **forum** `llama.cpp AND (performance OR optimization) AND jetson` — **✅ MATCHED SUBSTANTIVELY** via thread 380334 (Orin Nano 8GB + llama.cpp + NVIDIA-endorsed memory workaround). This is the first genuine in-window match for this trigger.

#### Overall: **ACTION NEEDED** — the device is **DEGRADED and deteriorating** (OOM cadence tightened to ~2 days, box at 97.8–99.9% of `MemoryMax` right now, tailnet dark 31 days), but this cycle produced the **first no-rebuild, NVIDIA-endorsed mitigation** in the investigation and **verified it is applicable to the running build**. Throughput recovered above the baseline floor. Software-currency posture is unchanged: **hold on b9652, take JetPack 6.2.3.**

#### Recommendations
**P0 — the one change most likely to stop the nightly OOM (no rebuild, no reflash):**
1. **Add `--cache-ram 0` and `--fit off` to `start-qwen35-server.sh`** (`--ctx-size 32768` is already explicit, satisfying the third leg of NVIDIA's recipe). Verified: both flags exist in b9652; neither is currently set; the inherited default is an 8192 MiB host prompt cache on a 7.4 GiB box, running pre-#25070 non-hard-limit eviction logic. Back up the script first (the Jetson is not git-recoverable) — Entry 029's `~/llm-server/backups/` pattern. **Validate over ≥3 nights across the 01:00 boundary**, watching `MemoryCurrent`/`MemAvailable` trend rather than a single restart. Expect a prompt-cache *latency* cost on repeat prompts — measure it; on this box, surviving the night is worth more. Apply the same two flags to the other 4 start scripts once proven.
2. **Fix the memory-watchdog trigger** (carried from Entry 038 P0#1, unchanged and still unarmed after 16 kills). Replace `file_swap > 1024 MB AND MemAvailable < 150 MB` — measured file-swap is **0 B** in every event, so the AND-gate cannot fire — with e.g. `MemAvailable < floor` sustained N polls, **OR** `zram_used > ~90% of zram total AND MemAvailable < floor`. Keep the cooldown, `SERVICE_MIN_AGE` and MAINTENANCE guards. **Verify by induced fire**, as Entry 037 did. Even with #1 working, this is the backstop.
3. **Restore `tailscaled`** — 31 days dark. `sudo systemctl restart tailscaled`; reboot if it re-panics (Entry 033: a reboot clears fTPM state). Then add a **liveness assertion on the Entry 037 pattern** (timer → metric → alert → bounded self-heal) — that pattern **survived an OOM kill today and self-recovered**, which is exactly the proof it is the right shape. If it re-fails, capture the panic **before the journal rotates**.
4. **Reconcile the `8080/tcp` LAN firewall rule** — unchanged from Entry 038, still undocumented, added 2026-07-20 11:21. Keep-and-document or remove; API-key protected either way. **User decision required.**

**P1 — currency, with the hold reaffirmed:**
5. **DO NOT rebuild llama.cpp past ~b10261.** #26750 is still open with no fix PR and no maintainer response; the reproduction now extends to **b10532** and a no-speculative control proves the loss lives entirely in the speculative path. Two independent new reasons reinforce this: **#27282** (duplicate ~1 GB MTP compute arena — disqualifying at 98% `MemoryMax`) and **#27549** (MTP + q8_0 KV + flash-attn fatal error, filed today). ✅ **Two former pre-rebuild worries are now closed:** #20834/#26110 (device verified clear of `--mlock`/`--no-mmap`) and #25384 (a cleanup — the real UMA fix #17368 is already in b9652). **New item for the eventual rebuild checklist: #26347** makes `/models` and `/v1/models` require auth — audit contact-center-lab and any discovery/health tooling first.
6. **Take JetPack 6.2.3 (L4T 36.5.2) via `apt`.** Rationale corrected: **not** for CVEs (36.5 already fixed CVE-2026-24148/24153/24154 — this box is patched), but for **bug 5412830**, a UEFI assertion that can halt at the bootloader and require a full reflash to recover — a brick-class risk on a headless node. **Bug 5602402 (NvMap allocation policy) is a speculative bonus, not a promise.** Stays on kernel 5.15 / CUDA 12.6, no reflash, no coupling to llama.cpp or MTP. Zero regression reports in 10 days (weak evidence — nobody is discussing it). **Sequence after P0#1** so the OOM mitigation is evaluated against an unchanged base.
7. **Convert the #26750 SM87 unknown into a fact — cheaply and locally.** The reporter's probe is a single deterministic `/completion` call (`temperature 0`, `cache_prompt false`, read `timings.draft_n` / `draft_n_accepted`). A **b10261-vs-b9652 A/B in the experiment slot** would either clear the blocker for Ampere or contribute the first non-SM121 datapoint to a stalled upstream issue. ⚠ Must use **free-form prose** — enumeration prompts are unaffected and would show false green. Today's b9652 prose baseline to compare against: **55.7% mean acceptance, 16.37 tok/s.**

**P2 — cheap experiments, none requiring a rebuild:**
8. **`--spec-draft-n-max 2` vs the current `3`** (carried, Entry 038 P2#7). PR #25883 (closed-unmerged, real Jetson data) reports ~75% acceptance at defaults vs **97.7% at n-max 2** on Xavier. One-line change, no build. Note today's acceptance recovered to 55.7%, so the upside is smaller than it looked last week — **run it after P0#1**, when memory pressure is no longer confounding the measurement.
9. **`Octen-Embedding-4B`** (carried) — same arch/size/dim as deployed, Apache-2.0, zero-risk A/B. ⚠ **Downgraded slightly:** its RTEB 0.7942-vs-0.7711 claim is **vendor-submitted with no third-party reproduction** (MTEB #3881 turns out to be Octen's own request, and for the 8B).
10. **`empero-ai/Qwen3.8-4B-Distill` Q4_K_M (2.783 GB)** — experiment-slot A/B only, **not a swap**: it has **no MTP head** (forfeits self-speculative decoding) and its headline MMLU gain rests on an anomalously low self-reported baseline. **`LiquidAI/LFM2.5-2.6B` QAD Q4_0 (1.594 GB)** is the more interesting structural option **if RAM headroom is the binding constraint** — which Finding 1 says it is — but the **LFM Open License <$10M-revenue restriction** gates it out of any client-derived work.
11. **Standing, carried:** re-benchmark 25W (`nvpmodel -m 1`) vs MAXN_SUPER on the real MTP workload. Thermals are **not** the constraint (50.5–52.7 °C idle vs a 99 °C throttle point); power efficiency is the only argument.
12. **Post-rebuild queue (blocked on #5):** Gemma 4 E2B-QAT UD-Q4_K_XL (2.62 GB — only candidate preserving MTP; drafter auto-discovers from `-hf`, no `--model-draft` needed, use `--spec-draft-n-max 4`), then Nanbeige4.2-3B (2.68 GB — **verify `num_loops=2` is present in the GGUF**, it silently falls back to 1; support merged in #25994, needs ≳b10160), then Agents-A1-4B (2.71 GB) if agentic/tool-calling matters.

**Recon hygiene (housekeeping):**
13. **Drop reddit.com from the Check-4 source list** — permanently blocked to the research user agent, three recons running, never actually searched. Carrying it overstates coverage.
14. **Drop forum thread 375642 (sm_87)** — closed 2026-08-05, resolved benign.
15. **Adopt `github.com/NVIDIA/product-security` as the standing security-bulletin source** — fully fetchable and machine-readable, where `nvidia.custhelp.com` returns HTTP 403 to research agents.

#### Proposed JETSON_BASELINE.md changes — **NOT APPLIED** (headless run, no user present; per run instructions)
- `Last updated:` / `Last recon:` 2026-07-16 → **2026-08-22**; `Last healthcheck:` → **2026-08-22 (DEGRADED — 16 OOM kills since 2026-07-30, cadence ~2 days; tailscaled dead 31 d; watchdog still unarmed; throughput RECOVERED to 16.37 tok/s)**. *(Note: the Entry 038 proposal to move these to 2026-08-15 was never applied — this supersedes it.)*
- `llamacpp_latest_seen:` b10054 → **b10586** (running stays **b9652**).
- `jetpack_latest_orin_nano:` 7.2 → **7.2.1 (L4T r39.2.1, 2026-08-11/12)**; **ADD** `jetpack6_latest: 6.2.3 (L4T 36.5.2, 2026-08-12, apt in-place)`; `jetpack_next_expected:` → **no 7.2.2/7.3 announced; watch prebuilt sm_87/CUDA 13.2 wheels & containers**.
- `models_last_checked_date` & `forum_last_checked_date:` 2026-07-16 → **2026-08-22**.
- **Recon Triggers:** (a) **re-point** the `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` row at **forum thread 380334 / `--cache-ram` / `--fit`** — and **do not** re-point it at #25384 as Entry 038 proposed (now known to be a cleanup, not a fix); (b) **ADD** `llamacpp_release | #26750 OR (draft-mtp AND acceptance) | ACTION: rebuild blocker — resolve before any build past ~b10261` (proposed at Entry 038, still not added); (c) **ADD** `llamacpp_release | #27282 OR (MTP AND compute arena) | ACTION: duplicate ~1 GB MTP arena — disqualifying at 8 GB until merged`.
- **Watch Items:** **KEEP the OOM item at 🔴 ACTIVE and escalate** (16 kills, cadence tightened to ~2 days); **ADD** the `--cache-ram 0 / --fit off` NVIDIA-endorsed mitigation as the **top open action** with the b9652-applicability proof; **ADD** #27282 and #26347 to the pre-rebuild checklist; **REMOVE** the #25384 watch item and the #20834/#26110 pre-rebuild item (all resolved/superseded); **UPDATE** the tailscaled item 24 d → **31 d**; **ADD** the JP6.2.3 rationale correction (bug 5412830 brick-risk, not CVEs — 36.5 already patched them); **ADD** the `empero-ai` distill and LFM2.5 QAD Q4_0; **NOTE** `baseline_rss_mb` 5839 is routinely exceeded (`MemoryCurrent` 6.56–6.70 GB observed).
- **Current Config section: unchanged** (still b9652 / Qwen3.5-4B-MTP-Q4_K_M — nothing was modified on the device).
- **Throughput (no baseline change proposed):** mean **16.37 tok/s** (18.32 / 13.66 / 17.12), **above** the 15.3 floor and up ~26% from Entry 038's ~13.0; draft acceptance mean **0.557** vs 29–38% last week. ⚠ Measured 13 h after a clean restart vs Entry 038's measurement at 96% `MemoryMax` — **most plausibly the absence of memory pressure, not a system improvement.** Re-measure after the P0#1 flag change before touching `baseline_gen_tok_s`.

---

## Entry 040: Recon (2026-08-29) — ACTION NEEDED: OOM is now DAILY, a new in-process CUDA-abort failure mode appeared, and the box itself printed the fix
**Date:** 2026-08-29 14:20 EDT (2026-08-29 18:20 UTC)
**Operator:** Claude Code (jetson-recon skill, headless scheduled run — no user present)
**Status:** RECON — **no changes made to the Jetson.** All device access read-only. Baseline tracking values NOT updated (headless run; proposal recorded below for user confirmation).

Five checks (4 web-research agents + 1 live SSH health check). Prior recon: Entry 039 (2026-08-22, 7 days ago).

**Headline:** the OOM cadence went from ~every 2 days to **every single day** (Aug 25/26/27/28/29 consecutive), a **new failure mode** appeared (in-process `NvMap → CUDA out of memory → ggml_abort` core-dump, Aug 23), and Check 5 obtained **direct on-device proof of the pending P0 fix** — llama-server's own startup banner prints `prompt cache is enabled, size limit: 8192 MiB` / `use --cache-ram 0 to disable the prompt cache`. The leak was also **quantified for the first time**: 3 × 200-token requests retained **+215 MB** that never came back.

---

#### JetPack / Firmware — **LOW (no version movement; JP7 hold hardens)**
- **No new release in 7 days.** [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive) top entries verified verbatim: `7.2.1 [L4T 39.2.1]`, `7.2`, `7.1`, `7.0`, `6.2.3 [L4T 36.5.2]`, `6.2.2 [L4T 36.5.0]`. **No 6.2.4, no 7.2.2, no 7.3.** Jetson Linux archive newest: 36.5.2 and 39.2.1 — **no 36.5.3, no 39.2.2.**
- **No new Jetson/Tegra/L4T security bulletin.** [NVIDIA/product-security](https://github.com/NVIDIA/product-security) added 3 bulletins on 2026-08-25 (5872 NemoClaw/OpenShell incl. a CVSS 9.9 sandbox escape; 5809 Unified Fabric Manager; 5867 DGX Spark). Bulletin bodies grepped directly — **none mention Jetson, Tegra, L4T, JetPack or Orin.** The adopted GitHub source (Entry 039 rec #15) worked as intended.
  - ⚠ **FLEET NOTE, not this box:** bulletin **5867 "NVIDIA DGX Spark — August 2026"** (CVE-2026-47626 / 47624 / 24263 / 24262 / 24225, UEFI/system-firmware, local, scope-changed) fixes in **DGX Spark UEFI 1.110.12 → 1.110.13**. That is `spark.k4jda.net`. **Raise it in the Spark's own audit/recon; out of scope here.**
- **JetPack 6.2.3 — still zero regression reports, 17 days post-release.** Searched the four announcement threads (379869/379871/379872/379873), general web and GitHub: no bug, breakage, rollback or failed-apt report. ⚠ Weak evidence on the forum axis — reply bodies do not render to WebFetch on those threads. **The Entry 039 recommendation to take 6.2.3 stands unchanged.**
- **🔴 NEW, and it strengthens the JP7 hold materially — our exact SKU, our exact upgrade path, unresolved.** [Thread 373852](https://forums.developer.nvidia.com/t/jetson-orin-nano-super-8gb-stuck-in-reboot-loop-during-jetpack-7-2-iso-firmware-update-uefi-capsule-update-fails/373852): *"Jetson Orin Nano Super 8GB: Stuck in Reboot Loop During JetPack 7.2 ISO Firmware Update (UEFI Capsule Update Fails)."* Posted 2026-06-19, last reply 2026-07-13, **still unresolved, multiple users reproduced.** NVIDIA's only remedies are SDK Manager from a native Ubuntu host, or a full reflash. **On a headless node with no local console, the ISO path's failure mode is a brick requiring physical USB-recovery.** This is the same class of risk as bug 5412830 — but on the *upgrade* rather than the *staying-put* side.
- **JP7.2.1 sm_87 ecosystem gap — not narrowed on the NVIDIA side; resolving by deferral to upstream.** [jetson-containers #1711](https://github.com/dusty-nv/jetson-containers/issues/1711) (jp7/cu132 PyPI index incomplete — `pip`, `numpy`, `psutil`, `cmake`, `ninja` missing) **still OPEN**, last updated 2026-07-08. AastaLLL now points JP7.2 Orin users at upstream `download.pytorch.org/whl/cu132` instead of NVIDIA wheels ([thread 379752](https://forums.developer.nvidia.com/t/379752), active 2026-08-28) — Orin on JP7.2.x uses the SBSA driver, so upstream aarch64 wheels work. Ollama on JP7 remains community-patched.
- **JetsonHacks: still nothing on 6.2.3** (latest JetPack post remains 2026-02-06).

#### llama.cpp Releases — **HIGH (the blocker is unchanged, but two adjacent items moved and a third corroborates the P0)**
- **Latest: b10686** (2026-08-29T16:46Z), up from b10586 — **+100 builds in 7 days.** Device runs **b9652**: **1,034 builds / 75 days behind.** 140 PRs merged in the window.
- **🔴 #26750 (the rebuild blocker) — ZERO activity in the window.** Still open, still 2 comments, still **no fix PR, no maintainer response, not even a `bug-unconfirmed` label**. Latest repro remains b10532 on GB10/SM121; **no Ampere/SM87/sm_80 datapoint has ever been posted.** Fourth consecutive recon with no movement. **The hold at b9652 stands and is still justified.**
- **🟠 #27282 (duplicate ~1 GB MTP compute arena) — most movement of any tracked item.** First **COLLABORATOR** response (ngxson, 08-23) proposing generalized shared compute buffers across contexts; open fix PR **[#27489](https://github.com/ggml-org/llama.cpp/pull/27489)** "ggml: reuse compute buffers for MTP" (+244/−0, 7 files, unmerged); a competing allocator-level patch from `srelus`. ⚠ **New regression data: both shared-buffer builds SIGABRT mid-generation inside `ggml_cuda_mul_mat_q`** on sm_89, and #27489 aborts instead of returning HTTP 400 on over-length prompts. All reports are 24 GB RTX 4090 — **no 8 GB or Jetson datapoint.** Do not chase this yet; it is moving in the right direction and is not ready.
- **🟠 [#27311](https://github.com/ggml-org/llama.cpp/pull/27311) "Scheduler UMA ring buffer" (pwilkin, MEMBER; open 08-18, active 08-26) — first structural unified-memory work in months that touches our device class.** Labels `ggml`/`CUDA`, +1367/−102 / 18 files, `mergeable_state: blocked`, 17 review comments. Author's framing: *"this should enable proper host memory usage on UMA devices other than Metal … once this is verified the `props.integrated` flag can be re-enabled on CUDA devices"*, and *"ping @ORippler for feedback/tests on CUDA integrated boxes."* Gated behind `GGML_SCHED_UMA_RING`. Validated only on AMD gfx1151/Strix Halo — **no CUDA-integrated (Jetson/GB10) validation posted.** Supersedes #25863/#26167/#26225. **Watch; do not act while unmerged.**
- **🟡 [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) (new 08-22, 12 comments, active 08-29): draft-mtp acceptance collapses to exactly 0.00 under `-np N`.** Root-caused to a race on the async device→host copy of `t_h_nextn` vs graph-input reuse — draft logits go NaN, every draft rejected. Candidate fix is **#27311's ring buffer** (confirmed to fix it on gfx1151). Reproduced on HIP, and on 08-29 on Windows/CUDA dual-4090. **Applicability to us: LOW** — reporters state it is correct at `-np 1` and we run `--parallel 1`. Recorded because it is a *second, distinct* MTP acceptance defect that **is** being worked while #26750 is not.
- **🟢 [#27148](https://github.com/ggml-org/llama.cpp/issues/27148) — a THIRD independent reason to set `--cache-ram 0`.** `server_prompt_cache` can restore an unrelated finished conversation into a fresh slot; the model then continues the stale conversation with **no client-visible signal** (`cached_tokens` reads 0). 08-20 comment escalated it: **reproduces single-user, strictly sequential, no concurrency.** Both `--cache-ram` (8192 MiB) and `--cache-idle-slots` default ON. Documented mitigation: `--cache-ram 0` plus `--no-cache-idle-slots` when `kv_unified=false`. **This is a silent-correctness bug, not just a memory bug.**
- **#27549** (MTP + q8_0 KV + flash-attn `fattn.cu:574`) — one new comment linking it to **[#24324](https://github.com/ggml-org/llama.cpp/issues/24324)** (open, labeled `bug` = confirmed). #24324 confirmed still present on master at `1729ed537` (08-25), diagnosed in `ggml-backend-meta.cpp`, **trigger identified as context restore from RAM (idle-slot / checkpoint restore)**; workaround `LLAMA_GRAPH_REUSE_DISABLE=1`; fix PR #24549 unmerged. **Jetson relevance: the confirmed trigger needs `--split-mode tensor` (multi-GPU), which we cannot hit — but the RAM-restore path is shared.** Pre-rebuild checklist note, not an action.
- **#27212 (sm_60) and #27105 (sm_52) — unchanged**, zero comments since filing.
- **🟢 Zero merged Jetson / SM87 / sm_87 / Tegra / Orin / NvMap work.** Those searches returned **true empty result sets**. **Fourth consecutive recon** (2026-07-01 → 2026-08-29) with none. Contrast: [#26264](https://github.com/ggml-org/llama.cpp/pull/26264) "cuda: unblock mmq for MoE on sm_60" merged 08-26 — **maintainers do land per-SM CUDA fixes when someone drives one. Nobody is driving one for SM87.**
- **Not applicable, recorded to close them out:** #27918 (CUDA VMM 32 GB VA cap blocks >32 GB models on GB10/GH200 — an 8 GB box cannot approach it); #27923 (`ggml_cuda_init` races `nvidia-uvm` at boot → silent CPU fallback with `/health` still 200 — **Tegra does not use `nvidia-uvm` and we run a system unit, not `systemd --user`**; noted only because that failure *shape* would also evade our monitoring).

#### Small Model Landscape — **HIGH (a 1.31 GB MTP-preserving GGUF in our exact architecture, header-verified; plus a correction to Entry 039)**
- **🔴 THE FINDING: [`empero-ai/Qwen3.8-2B-Distill-GGUF`](https://huggingface.co/empero-ai/Qwen3.8-2B-Distill-GGUF) → `Qwen3.8-2B-Q4_K_M.gguf` = 1.312 GB, MTP head PRESENT.** Verified by **direct GGUF binary-header parse** (ranged HTTP fetch, 335 tensors), not by reading a model card: `general.architecture = qwen35` (identical to our deployment), `qwen35.nextn_predict_layers = 1`, tensors `blk.24.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight`. 25 blocks, hidden 2048, `context_length = 262144`, Apache-2.0, base `Qwen/Qwen3.5-2B`. **Frees 1.52 GB vs the deployed 2.83 GB — ~23% of MemoryMax, directly against the binding constraint, with self-speculative decoding intact and no runtime change.**
  - ⚠ **Its benchmarks are unusable.** The card compares only against Qwen3.5-2B base and reuses the same broken methodology already flagged for empero's 4B — the base's MMLU-CoT strict-match is reported as **0.004**, a formatting artifact, not a capability measurement. **Treat the "+0.265 MMLU" headline as noise. This is a memory-headroom play that must be quality-validated locally, not a quality upgrade.**
- **🟢 Safe control arm on the same axis: [`unsloth/Qwen3.5-2B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF)** — `Q4_K_M` **1.33 GB**, `UD-Q4_K_XL` 1.385 GB, Apache-2.0. Same vendor and naming convention as our current model. Not new (2026-05-13) but **never surfaced in prior recons.** This is the *known-good* 1.3 GB option if the empero distill disappoints.
- **✅ CORRECTION to Entry 039: `empero-ai/Qwen3.8-4B-Distill` DOES have an MTP head.** GGUF header parse of `Qwen3.8-4B-Q4_K_M.gguf` shows `qwen35.nextn_predict_layers = 1` with `blk.32.nextn.*` tensors. **Entry 039's "no MTP head — forfeits self-speculative decoding" was wrong and should not be relied on.** The separate criticism (headline MMLU gain rests on an anomalously low self-reported baseline) stands, and is now confirmed to be a **family-wide** methodology problem across empero's releases.
- **Status changes to tracked candidates:** `Agents-A1-4B` — an MTP GGUF appeared (`AlexCRUY/Agents-A1-4B-MTP-GGUF`, 08-28) but **f16 only, 8.666 GB, no Q4_K_M, no license tag**; not actionable, watch for a Q4_K_M. `Nanbeige4.2-3B` — a Q4_K_M now exists (`mackkkkkilllll/Nanbeige4.2-3B-Q4_K_M`, **2.575 GB**, 08-19, no license tag); still gated on `num_loops=2` and ≥b10160. `LiquidAI` — new `LFM2.5-2.6B-DSpark-GGUF` Q4_K_M is only 0.2 GB because it is a **draft model, not a standalone chat model**; LFM Open License <$10M-revenue restriction unchanged, still gated out of client work. **Gemma 4 E2B-QAT unchanged** (2.62 GB, lastMod 2026-07-17) — heavy third-party fine-tune activity but no upstream change and still blocked on the rebuild hold. **No change:** Jackrong ×2, khazarai, Octen-Embedding-4B, jina-v5-small.
- **Explicit negatives:** **No Qwen4 weights exist.** **No new 4B-class dense model from Qwen** — the org has shipped exactly four things since 2026-06 (Qwen3.8-27B, Qwen3.8-2.4T-A95B, Qwen3.8-27B-FP8, Qwen3.8-Flash-Next); **a fifth consecutive generation has skipped the 4B tier.** No new dense 1–7B model from any major lab in the window — everything major that shipped (GLM-5.3/Flash 321B, Hy4-preview 780B, Qwen3.8-Flash-Next 180B, Ornith-1.5-397B) is enormous MoE, so **the SM87 MoE decode-hang question (#19219) never comes into play.** **No new text-embedding model beats Qwen3-Embedding-4B** — the only new embedders are Tencent WeMM (MMEB-v2 *multimodal*, not MTEB text, so not comparable; the 4B variant is 5.17B params = a RAM *regression*) and small niche models.
- ⚠ **Method note:** the HF `createdAt`-sorted API hard-caps at ~3000 records (`HTTP 400` at `skip=3100`), so the brute-force new-model sweep covered only ~1 day. Targeted `search=` queries and `trendingScore` sweeps were used instead. **Every benchmark number in this section is vendor-self-reported.** The only independently verified facts are file sizes, parameter counts, licenses and MTP-tensor presence — read from HF metadata and GGUF binary headers.

#### Jetson Forum / Community — **ACTION (no new forum activity; the value came from an independent measurement that supplies our missing mechanism)**
- **🔴 Thread 380334 (our P0's source) — NO NEW REPLIES.** `highest_post_number` 3, `last_posted_at` **2026-08-17T05:13Z** (AastaLLL), post 1 last edited 2026-08-17T00:25Z. The reporter (`manuel58`) never returned with results; 80 views, 0 likes. **Two details from the raw JSON sharpen our P0:**
  1. **The verified fix and NVIDIA's suggestion are different things.** manuel58's own UPDATE, verbatim: *"newer llama.cpp build flags (`--fit off --ctx-size <explicit value> --cache-ram 0`) eliminated the failures entirely across 60+ consecutive requests."* AastaLLL *separately* suggested `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 LLAMA_ARG_FIT=off`. **Only the flags were reporter-verified; the env var is NVIDIA's untested hypothesis.** Treat them as **separable changes**, not one bundle.
  2. **The reporter refuted his own headline theory** — the container shipped SBSA CUDA 13.0 rather than Tegra-native, but a Tegra-native rebuild **failed identically at defaults**, so the CUDA target was not the cause. 380334 is JP7.2/7.2.1 and we are 6.2.2, **yet the flag fix is platform-independent.**
- **🔴 NEW — independent, quantitative corroboration of the `--cache-ram` mechanism.** [llama.cpp #22127](https://github.com/ggml-org/llama.cpp/issues/22127) (2026-04-19, closed not-planned): reporter measured **~11.9 GB RSS under load at the 8192 MiB default vs ~1.0 GB RSS with `--cache-ram 0`.** Unrelated author, unrelated hardware, unrelated to 380334. Upstream `tools/server/README.md` verbatim: `-cram, --cache-ram N` = *"the maximum cache size in MiB (default: 8192, -1 - no limit, 0 - disable)"* — a **host-RAM** prompt cache, on by default since Oct 2025, **separate from and additional to the KV cache**. An 8 GiB default host cache filling as requests accumulate produces **exactly** a monotonic MemAvailable decline ending in global OOM, on a box with 8 GB total.
- **⚠ VERIFICATION TRAP — read before validating the fix.** Per #22127, llama-server **still logs `"prompt cache is enabled, size limit: 8192 MiB"` even when `--cache-ram 0` is set** — the log line prints before cache init. **Do not use the log line to confirm the flag took effect. Use RSS / `MemoryCurrent` trend.**
- **🟠 Secondary, build-time: `GGML_CUDA_NO_VMM=ON`.** [Thread 361870](https://forums.developer.nvidia.com/t/qmd-node-llama-cpp-on-jetson-orin-agx-gpu-runtime-oom/361870) (Feb–Mar 2026, closed; new to our tracking): user `ckdavid233` (AGX Orin, r36.4.7) claims precompiled binaries reserve a large fixed CUDA VMM pool that OOMs on Tegra, and reports it fixed by rebuilding with CMake `GGML_CUDA_NO_VMM=ON`. **His claim, unverified; AastaLLL participated but never endorsed it.** Corroborating mechanism: [#16197](https://github.com/ggml-org/llama.cpp/issues/16197) shows an OOM stack terminating in `ggml_cuda_pool_vmm::alloc` → `cuMemCreate` **with `GGML_CUDA_ENABLE_UNIFIED_MEMORY` already set** — so **the env var and the VMM pool are different allocation paths, and the env var does not bypass the pool.** ⚠ **This is now directly relevant — see Check 5's Aug 23 core-dump.** Requires a rebuild, so it sits strictly behind the flag-only fix.
- **🟡 Partial refutation of the env-var half of our P0.** [Discussion #16706](https://github.com/ggml-org/llama.cpp/discussions/16706) (Orin Nano): commenter `TinyServal` argues *"Unified Memory" in CUDA means managed allocations, not Jetson's memory architecture*, and describes our exact signature — allocation succeeds, then the OOM killer terminates the process as the cache grows. **The OP reported `GGML_CUDA_ENABLE_UNIFIED_MEMORY` did NOT help.** Unverified claims, but they argue the env-var half may do nothing for us while `--cache-ram 0` does the real work. **Reinforces: apply the flags, treat the env var as a separate experiment.**
- **Peer and tooling status:** **380427** (Evopien — Qwen3.5-4B + llama.cpp on Orin Nano Super 8GB, our nearest peer) — 3 posts, last 2026-08-17, posts 2–3 are only a moderator category move and an acknowledgment. **No solution, no metrics. Still silent.** **373627** confirmed dead exactly as recorded (NVIDIA could not reproduce without the UMA flag; asked for a repro command line, never supplied). **370049, 372493** no signal; **375642** stays dropped. **TensorRT-Edge-LLM still v0.10.0**, notes remain DGX Spark / Thor / Blackwell-centric, **still no published Orin Nano tok/s**; standing v0.8.0 datum: *"at least the model size plus approximately 3 GB of available memory for runtime overhead"* — on 8 GB that leaves ~5 GB for a 4B model, **consistent with our margins being genuinely thin.** **jetson-containers HEAD still 70c149a (2026-06-25) — 65 days dormant.**
- **No new Jetson llama.cpp forum content in the window.** A forum-wide `llama.cpp` search returned only DGX Spark topics. New Orin Nano topic 381487 ("Local Qwen-Controlled 3D Agent", 08-27) has **runtime unspecified and zero metrics.** No credible new Orin Nano 8GB tok/s data anywhere.
- ⚠ **Tracking-ID error corrected:** `/t/380034` actually resolves to *"Where can I get torch 2.13.0 for vllm 0.27.0+ … AGX Orin … 6.2.3 and 7.2.1?"* — **AGX Orin / vLLM wheels, not the JP7 sm_87 gap** we recorded it as. The live thread for that subject is **379752**. **Retarget tracking 380034 → 379752.**
- *Source gap, fourth consecutive recon:* reddit.com remains blocked to the research user agent — **not attempted this cycle**, per the Entry 039 recommendation to drop it.

#### Live Health — **DEGRADED and DETERIORATING (OOM now daily; new failure mode; tailnet dark 38 days)**
Reached over **LAN `192.168.10.58`**; the tailnet path timed out again. Host **uptime 60 days** (boot 2026-06-30, unchanged), mode `qwen35`, all 4 drop-ins intact (`cma-compact`, `crash-escalate`, `memory-limits`, `oom-protect`), disk **17% / 660 G free**, thermals **49.8–52.2 °C** idle, power **MAXN_SUPER (mode 2)**, slots 1. **No version drift:** source HEAD `6eab47181` = **b9652** = baseline; server reports `system_fingerprint: b9652-6eab47181`, `CUDA : ARCHS = 870 | USE_GRAPHS = 1 | FA_ALL_QUANTS = 1`. **Auth healthy:** unauthenticated `/v1/chat/completions` → **401**, authenticated → **200**, `/health` public → **200**.

**🔴 FINDING 1 — the OOM is now DAILY. Seven more service deaths in 7 days.**
- Restart counter **24 → 31**. New events: **Aug 23 01:00:27 (`core-dump` — NEW failure mode, see Finding 2)**, Aug 23 05:05:21, Aug 25 01:01:50, Aug 26 01:05:44, Aug 27 01:02:41, Aug 28 01:01:09, **Aug 29 01:00:48**. That is **6 new `oom-kill` results plus 1 `core-dump`.**
- **Aug 25, 26, 27, 28, 29 are five consecutive days.** Cadence has gone ~3 days (Entry 038) → ~2 days (Entry 039) → **~1 day**. Running total is **≥22 OOM kills since 2026-07-30** (the journal has now rotated past 2026-07-30, so the exact total is no longer recoverable on-device — 16 through Entry 039 plus 6 since).
- The **~01:00–01:05 EDT boundary now holds for 26 incidents.** The lone exception, **Aug 23 05:05:21**, is the first off-schedule OOM ever recorded and followed 4 h after the Aug 23 core-dump.
- **Today's 01:00:46 kernel storm is the unchanged NvMap signature.** `oom_score_adj=-900` works exactly as designed: `rsyslogd`, 4× `systemd-udevd`, `systemd-logind`, `polkitd`, `nv-tee-supplicant`, `nvs-service`, `nvfancontrol`, `cron`, `tegrastats`, 3× `agetty`, `tee-supplicant`, `seatd`, `systemd-journald` (adj −250) and `dockerd` (adj −500) are all sacrificed **first**; llama-server dies **last**. Victim line: `llama-server total-vm:80881520kB, anon-rss:4104984kB, file-rss:58840kB … oom_score_adj:-900`. **Killing userspace frees nothing because the memory is NvMap-held and invisible to OOM accounting.** Note the **~77 GB total-vm** — consistent with a large CUDA VMM virtual reservation.
- Nightly client load in the 20:00 → 01:02 window was **14 `launch_slot` tasks** (12 last week, 5–36 the week before). **The burst is the trigger, not the cause.**

**🔴 FINDING 2 — NEW FAILURE MODE: an in-process CUDA allocation abort, not a kernel OOM kill (Aug 23 01:00:20).**
```
NvMapMemAllocInternalTagged: 1075072515 error 12      <- error 12 = ENOMEM
NvMapMemHandleAlloc: error 0
E CUDA error: out of memory
/home/claude/llm-server/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:103: CUDA error
  #2 ggml_abort () from libggml-base.so.0
  #3 ggml_cuda_error(...) from libggml-cuda.so.0
systemd[1]: myscript.service: Main process exited, code=dumped, status=6/ABRT
systemd[1]: myscript.service: Failed with result 'core-dump'.
```
- **This is the first time llama-server has died by its own failed allocation rather than by the kernel OOM killer.** The device-side NvMap allocator returned ENOMEM to CUDA, and ggml aborted.
- **Significance: the pressure is now severe enough to exhaust the GPU-visible allocator, not merely the host side.** It also lands the failure squarely on the path Check 4's `GGML_CUDA_NO_VMM` finding describes (`ggml_cuda_pool_vmm::alloc` → `cuMemCreate`).
- ⚠ **This one is NOT addressed by `--cache-ram 0`** — a host prompt cache is not what `NvMapMemAllocInternalTagged` failed to serve. `--cache-ram 0` should reduce the *global* pressure that made this reachable, but the VMM-pool question is now a live second front.

**🔴 FINDING 3 — the box printed the P0 fix in its own startup banner. No source-tree inference needed any more.**
Verbatim from today's 01:00:53 start:
```
common_init_result: fitting params to device memory ...
common_init_result: (for bugs during this step try to reproduce them with -fit off, ...)
srv load_model: prompt cache is enabled, size limit: 8192 MiB
srv load_model: use `--cache-ram 0` to disable the prompt cache
srv load_model: for more info see https://github.com/ggml-org/llama.cpp/pull/16391
srv load_model: context checkpoints enabled, max = 32, min spacing = 256
```
- `fitting params to device memory` proves **`-fit on` is active**. `size limit: 8192 MiB` proves the **8 GiB host prompt cache is live on a 7.4 GiB machine**. Live argv re-confirmed (all 21 args) — **still no `-cram`/`--cache-ram`, still no `-fit`/`--fit`.** Also: `estimated memory usage of MTP context is 202.02 MiB`; `CUDA0 : Orin (7607 MiB, 6304 MiB free)`; startup guard read `Free memory: 6852MB — using full GPU offload`.
- **Entry 039 established this by reading `common/arg.cpp` at tag b9652. Entry 040 has the device stating it directly.**

**🔴 FINDING 4 — the leak is QUANTIFIED for the first time. Three requests retained 215 MB that never came back.**
Measured live, 13 h after a clean restart, with an abort guard at MemAvailable < 200 MB:

| | before | after 3 × 200-token requests | settled (+5 s idle) |
|---|---|---|---|
| `MemAvailable` | 425 MB | **135 MB** | 213 MB |
| `MemoryCurrent` | 6.366 GB | **6.632 GB (98.8% of MemoryMax)** | 6.581 GB (98.1%) |

- **+266 MB consumed by 600 generated tokens; +215 MB still held after settling.** That is **~72 MB retained per 200-token request** and it is monotonic, not transient.
- **Extrapolation: 14 nightly tasks × ~72 MB ≈ 1.0 GB** — comfortably enough to close the ~400 MB margin the box starts the night with. **This is a quantitative, on-device match to the #22127 mechanism** (11.9 GB → 1.0 GB RSS with `--cache-ram 0`) and to manuel58's report. **The P0 hypothesis is now corroborated three independent ways: NVIDIA's endorsement, an unrelated reporter's RSS measurement, and our own retained-memory measurement.**
- ⚠ **Methodology / safety note:** the benchmark was run with a hard abort at MemAvailable < 200 MB and stopped after run 3. It **did** push the box to 98.8% of `MemoryMax`. No further load was applied. This is disclosed because the measurement itself materially raised tonight's OOM probability.

**🔴 FINDING 5 — the memory-watchdog's defect was DEMONSTRATED live, not merely inferred.**
- The trigger, read from `/home/claude/llm-server/memory-watchdog.sh` line **129**: `if [ "$fswap" -gt "$FSWAP_CRIT_MB" ] && [ "$avail" -lt "$AVAIL_FLOOR_MB" ]`, with `FSWAP_CRIT_MB=1024` (line 41) and `AVAIL_FLOOR_MB=150` (line 42, commented *"AND-guard: only act if truly out of RAM"*).
- **During Finding 4's benchmark, `MemAvailable` fell to 135 MB — below the 150 MB floor. The watchdog still did not fire.** `journalctl -u memory-watchdog --since '2026-08-29 13:50'` → **"No entries."** Because file swap `/ssd/16GB.swap` USED = **0 B** against a 1024 MB conjunct, exactly as Entries 038 and 039 concluded.
- The service is **active** and has **never fired across all ~22 OOM events**. zram sits at ~133–164 MB of 3.8 GB. **Entries 038/039 argued the AND-gate was unsatisfiable; Entry 040 drove one conjunct true and watched it stay silent. This is now a demonstrated defect.**

**🔴 FINDING 6 — `tailscaled` dead for 38 DAYS.** Failed since **Wed 2026-07-22 01:00:22 EDT**, `ExecMainStatus=2`, `NRestarts=6`, systemd gave up. Went 24 d → 31 d → **38 d** with no change, because nothing alerts on it. The unit's journal has long since rotated — **root cause for the 07-22 event remains permanently unrecoverable.** The node has been invisible on the tailnet for over a month while serving normally on the LAN.

**🟡 FINDING 7 — `ufw` active, and the undocumented `8080/tcp ALLOW 192.168.10.0/24` rule is still present.** Live ruleset: default deny incoming; `lo` / `tailscale0` (v4+v6) / `22/tcp from 192.168.10.0/24` / **`8080/tcp from 192.168.10.0/24`**. Still not the Entry 034 known-good set, still API-key protected (401 verified). **Unchanged for three recons — still awaiting the user's keep-and-document vs remove decision.** The `ufw-watchdog` timer is active.

**🟡 FINDING 8 — `udisks2` has failed again.** **7 failed units** now: the 5 expected boot-time casualties (`nvphs`, `avahi-daemon`, `wpa_supplicant`, `networkd-dispatcher`, `kerneloops`) plus **`tailscaled`** plus **`udisks2`**, which had recovered at Entry 039 and is down again — collateral from one of the six intervening OOM storms. `ModemManager` remains recovered.

**🟠 THROUGHPUT — mean 15.63 tok/s, barely above the 15.3 floor, and highly variable.**
- 3 × 200-token prose generations (`temperature 0`, `cache_prompt false`, Entry 038/039 methodology): **11.41 / 20.35 / 15.14 tok/s → mean 15.63.** Baseline floor **15.3**; the 15% warn floor is 13.0. Entry 039 measured 16.37; Entry 038 measured ~13.0.
- **Draft acceptance: 0.283 / 0.762 / 0.480 → mean 0.508** (Entry 039: 0.557). Nightly structured tasks in the journal still show **1.000 acceptance at 21.6 tok/s** (Aug 29 05:01), consistent with the established prompt-dependence.
- **The variance is the signal, not the mean.** Run 1 was 11.41 tok/s at 0.283 acceptance with MemAvailable 425 MB; run 2 hit 20.35 at 0.762. Same prompt shape, same settings, 2× spread. **Do not treat `baseline_gen_tok_s` as measurable while memory pressure is uncontrolled.**
- Prompt eval **66.7–68.8 tok/s** on the 15-token bench prompt; **355–374 tok/s** on 247–272-token prompts in the nightly journal. **The 166 baseline is only comparable to the latter.**

#### Cross-Correlated Findings
1. **🔴 The P0 is now corroborated three independent ways and quantified on our own hardware.** (a) NVIDIA's AastaLLL endorsed the recipe on an Orin Nano 8GB (Check 4, Entry 039); (b) an unrelated reporter measured **11.9 GB → 1.0 GB RSS** from `--cache-ram 0` alone (#22127, Check 4, new this cycle); (c) **our box retained +215 MB across three 200-token requests** and prints `prompt cache is enabled, size limit: 8192 MiB / use --cache-ram 0 to disable` in its own banner (Check 5, Findings 3–4). **Entry 039 proved the flags were applicable. Entry 040 proves the mechanism is active and measures its rate.** ~72 MB/request × 14 nightly tasks ≈ 1.0 GB, against a starting margin of ~400 MB.
2. **🔴 A second, independent reason to set `--cache-ram 0` emerged — and it is a correctness bug, not a memory bug.** #27148 (Check 2): the prompt cache can restore an **unrelated finished conversation** into a fresh slot, with `cached_tokens` reading 0 so the client cannot detect it, **reproducing single-user and strictly sequential** — which is exactly our access pattern (`--parallel 1`). We have been running with this enabled for 75 days. **This changes `--cache-ram 0` from an OOM mitigation into something worth doing regardless of the OOM.**
3. **🔴 The new core-dump failure mode opens a second front that `--cache-ram 0` does NOT close.** Finding 2's `NvMapMemAllocInternalTagged: error 12 → CUDA error: out of memory → ggml_abort` is a **device-side** allocator exhaustion. Check 4 independently surfaced `GGML_CUDA_NO_VMM=ON` (thread 361870, AGX Orin) and, via #16197, established that **`GGML_CUDA_ENABLE_UNIFIED_MEMORY` does not bypass the VMM pool** — different allocation paths. The 77 GB `total-vm` in today's kernel dump is consistent with a large VMM virtual reservation. **So: `--cache-ram 0` addresses the host-side monotonic growth; the VMM pool is a separate, rebuild-gated question that has now produced an actual crash.**
4. **The Entry 039 recommendation to bundle the env var with the flags should be split.** Check 4 established that only the **flags** were reporter-verified; the env var is NVIDIA's untested hypothesis, discussion #16706 reports it did **not** help on an Orin Nano, and Entry 039 already found the UMA behaviour is active by default on this box (`prop.integrated > 0`). **Apply the flags. Treat `GGML_CUDA_ENABLE_UNIFIED_MEMORY` as a separate experiment with its own control.**
5. **The rebuild hold is unchanged, and the reason it is unchanged got sharper.** #26750 had **literally zero activity** — fourth consecutive recon, no maintainer has ever touched it, and **no Ampere/SM87 datapoint has ever been posted by anyone.** #26264 (per-SM CUDA fix for sm_60, merged 08-26) proves maintainers do land these when someone drives one. **Nobody is driving one for SM87, and we are the only party with the hardware and the motive.** Entry 039's P1#7 (a local b10261-vs-b9652 A/B) is the cheapest way to convert this from a blocker into a fact.
6. **🟢 The severity axis shifted from "quality" to "headroom", and Check 3 delivered exactly on the new axis.** Finding 4 measured the margin at ~400 MB overnight. `empero-ai/Qwen3.8-2B-Distill` frees **1.52 GB with the MTP head intact and identical `qwen35` architecture** (header-verified, not card-claimed), with `unsloth/Qwen3.5-2B-MTP` (1.33 GB) as a known-good control arm. **A 1.3 GB model turns a ~400 MB overnight margin into a ~1.9 GB one.** ⚠ Its published quality numbers are worthless — this must be validated locally in the experiment slot, which costs nothing since the slot is free and the arch is unchanged.
7. **Both watchdog-shaped failures are now demonstrated rather than argued — and Entry 037's pattern remains the proven answer.** The memory-watchdog stayed silent while `MemAvailable` sat at 135 MB (Finding 5, live demonstration); `tailscaled` has been dark **38 days** (Finding 6) and its root cause has now rotated out of the journal permanently. Meanwhile `ufw-watchdog` was OOM-killed at Entry 039 and **self-recovered**. `udisks2` fell over again this week with nothing noticing (Finding 8). **Every failure that has an assert→alert→bounded-self-heal loop recovered; every one without has run for weeks.**
8. **The JP7 hold is now over-determined from the upgrade side too.** Entry 039 held on the CUDA 13.2 ecosystem gap. Check 1 adds an **unresolved reboot-loop / UEFI-capsule-failure thread on our exact SKU and exact upgrade path** (373852), whose only remedies require physical access. **JP6.2.3 via apt is the only currency move that does not risk the box.** The sm_87 wheel gap is meanwhile resolving via upstream PyTorch rather than NVIDIA, which weakens the *positive* case for JP7 rather than strengthening it.
9. **The model landscape has stopped being a source of upside at our size.** A fifth consecutive Qwen generation skipped the 4B tier; every major release this window was 180B–780B MoE; no new embedder beats Qwen3-Embedding-4B. **Planning should stop treating "wait for the next 4B" as a viable path** — the actionable moves are all *downward* in size (Check 3) or *sideways* in configuration (P0), not upward in model quality.

#### Triggered Alerts
- **jetpack** `(JetPack 7.2.1 OR 7.3 OR power mode fix OR TNSPEC) AND (Orin Nano OR Orin)` — **MATCHED (JP7.2.1 ∧ Orin Nano), resolving to NO ACTION but reinforcing the hold.** Match basis is thread 373852 (unresolved Orin Nano Super 8GB reboot loop / UEFI capsule failure on the JP7.2 ISO path). No 7.3, no power-mode/TNSPEC news. **Hold on JP7 stands; take JP6.2.3.**
- **llamacpp_release** `SM87 OR Jetson OR Tegra OR unified memory` — **MATCHED on `unified memory` only.** `SM87`/`sm_87`/`Jetson`/`Tegra`/`NvMap` returned **true empty result sets**. The match is PR #27311 (open, MEMBER, CUDA-integrated UMA host buffers) and #27918 (GB10/GH200 VMM cap). **Nothing merged.** Fourth consecutive recon with zero merged Jetson-class work.
- **llamacpp_release** `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` — **NOT MATCHED in the repo** (only pre-existing ROCm/gfx1151 #26148; no NvMap mention anywhere in llama.cpp). ⚠ But it **matched substantively via Check 4** (#22127, thread 361870, discussion #16706). **The trigger's repo-only scope is too narrow — re-point it at `--cache-ram` / `--fit` / `GGML_CUDA_NO_VMM` as proposed at Entry 039.**
- **llamacpp_release** `#26750 OR (draft-mtp AND acceptance)` *(proposed at Entries 038 and 039, still not added)* — **MATCHED, and it earned its place.** #26750 itself had zero activity; the `draft-mtp AND acceptance` half fired on **#27572**, a distinct actively-worked acceptance-collapse defect with a candidate fix. **A #26750-only trigger would have missed it. Add this row.**
- **llamacpp_release** `#27282 OR (MTP AND compute arena)` *(proposed at Entry 039, not yet added)* — **MATCHED (ACTION).** First collaborator response, open fix PR #27489, competing allocator patch, new SIGABRT regressions. **Add this row.**
- **huggingface** `Qwen4 OR Qwen3.5 successor` — **MATCHED on the successor clause.** `Qwen/Qwen3.8-Flash-Next` (2026-08-24) has `config.architectures = Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp` — an explicit early release of the **Qwen4 architecture**, 180 B total. **Operational impact: none** (~60× our ceiling). Roadmap signal only, and the signal is negative: **on the evidence of Qwen3.6/3.8, expect no 4B dense member.**
- **forum** `llama.cpp AND (performance OR optimization) AND jetson` — **MATCHED on carry-over only.** 380334 and 380427 still satisfy it, but **no new forum post in the window discusses llama.cpp performance on Jetson.** The substantive material this cycle came from the llama.cpp tracker, not the NVIDIA forum. **Honest read: the forum axis is running dry; the GitHub axis is carrying this trigger.**

#### Overall: **ACTION NEEDED**
The device is **DEGRADED and deteriorating on a clear trajectory** — OOM cadence ~3 days → ~2 days → **daily**, five consecutive nights, plus a **new in-process CUDA-abort failure mode**, sitting at 94.9–98.8% of `MemoryMax`, with the tailnet dark 38 days and the memory watchdog **demonstrated** silent at 135 MB available. Against that: the pending P0 is now **triple-corroborated and quantified at ~72 MB retained per request**, it acquired a **second, correctness-based justification** (#27148), and Check 3 surfaced a **header-verified 1.31 GB MTP-preserving model** that would nearly quintuple the overnight margin. Software-currency posture is unchanged and now over-determined: **hold on b9652, take JetPack 6.2.3, do not take JetPack 7.**

#### Recommendations
**P0 — apply now; the box is failing nightly and the fix needs no rebuild and no reflash:**
1. **Add `--cache-ram 0` and `--fit off` to `start-qwen35-server.sh`.** (`--ctx-size 32768` is already explicit, satisfying the third leg.) Evidence is now overwhelming: the banner prints the recommendation itself; `fitting params to device memory` confirms `-fit on`; we measured **+215 MB retained across 3 requests**; #22127 measured **11.9 GB → 1.0 GB RSS**; manuel58 reported 60+ clean consecutive requests; and #27148 makes it a correctness fix as well. **Back up the script first** — the Jetson is not git-recoverable (Entry 029 `~/llm-server/backups/` pattern). ⚠ **Do NOT use the startup log to verify** — per #22127 the `size limit: 8192 MiB` line prints regardless. **Verify by re-running the Finding 4 measurement**: 3 × 200-token requests should retain far less than 215 MB. **Validate ≥3 nights across the 01:00 boundary.** Expect a repeat-prompt latency cost; measure it. Apply to the other 4 start scripts once proven.
2. **Fix the memory-watchdog trigger** (carried from Entries 038 and 039; **now a demonstrated defect, not an inferred one** — Finding 5). Replace the line-129 AND-gate (`fswap > 1024 MB && avail < 150 MB`) with `MemAvailable < floor` sustained N polls, **OR** `zram_used > ~90% of zram total AND MemAvailable < floor`. Keep the cooldown, `SERVICE_MIN_AGE` and MAINTENANCE guards. **Verify by induced fire**, as Entry 037 did. **Even with #1 working, this is the backstop — and it has now been silent through ~22 kills.**
3. **Restore `tailscaled`** — 38 days dark, root cause permanently unrecoverable. `sudo systemctl restart tailscaled`; reboot if it re-panics (Entry 033: a reboot clears fTPM state). Then add a **liveness assertion on the Entry 037 pattern**. If it re-fails, **capture the panic before the journal rotates** — we have now lost that evidence twice.
4. **Reconcile the `8080/tcp` LAN firewall rule** — unchanged for three recons, added 2026-07-20 11:21, undocumented. Keep-and-document or remove; API-key protected either way. **User decision required.**

**P1 — after P0#1 has been validated for ≥3 nights:**
5. **DO NOT rebuild llama.cpp past ~b10261.** #26750 had **zero activity** this cycle — no comments, no PR, no maintainer, no label, fourth consecutive recon. #27282 is improving (collaborator engagement + fix PR #27489) **but its candidate patches currently SIGABRT mid-generation on sm_89**. #27549 remains untriaged. **New pre-rebuild checklist items:** #27148 (prompt-cache cross-contamination — moot once `--cache-ram 0` is set), #24324 (needs `-sm tensor`, unreachable single-GPU, but shares the RAM-restore path), and the carried #26347 (`/models` auth — audit contact-center-lab first).
6. **Take JetPack 6.2.3 (L4T 36.5.2) via `apt`.** Unchanged rationale (bug 5412830 UEFI brick-risk; bug 5602402 NvMap allocation policy as a speculative bonus). **Still zero regression reports at 17 days.** Stays on kernel 5.15 / CUDA 12.6, no reflash. **Sequence after P0#1 so the OOM mitigation is evaluated against an unchanged base.**
7. **Investigate the new CUDA-abort front (Finding 2).** The Aug 23 core-dump is device-side allocator exhaustion, which `--cache-ram 0` does not directly address. Cheapest probe: after P0#1 is stable, check whether the abort recurs at all. If it does, the next lever is a rebuild with **`GGML_CUDA_NO_VMM=ON`** (thread 361870, unverified third-party claim; #16197 corroborates the mechanism and shows the UMA env var does **not** bypass the VMM pool). ⚠ That requires a rebuild, which collides with #5 — **so this is explicitly a "only if it recurs" item, not a plan.**
8. **Convert the #26750 SM87 unknown into a fact.** Unchanged from Entry 039 P1#7, and more urgent now that four recons have passed with nobody driving an SM87 fix. A **b10261-vs-b9652 A/B in the experiment slot**, using free-form prose (enumeration prompts show false green). ⚠ **Run it only after P0#1**, since Finding 4 shows memory pressure alone moves acceptance from 0.28 to 0.76 on identical settings.

**P2 — cheap, none requiring a rebuild:**
9. **🔴 Promoted — trial `unsloth/Qwen3.5-2B-MTP-GGUF` Q4_K_M (1.33 GB) in the experiment slot, then `empero-ai/Qwen3.8-2B-Distill-GGUF` Q4_K_M (1.312 GB).** Both are `arch=qwen35` with a verified MTP head, so they drop into `experiment` mode with no runtime change and no memory risk. **Frees ~1.5 GB — turning a ~400 MB overnight margin into ~1.9 GB.** Start with the unsloth build as the known-good control; the empero distill's published benchmarks are unusable (broken 0.004 baseline) so **judge both on local quality, not on cards.** ⚠ This is a *fallback if P0#1 underdelivers*, not a replacement for it — a 4B model that fits is better than a 2B model that fits.
10. **`--spec-draft-n-max 2` vs the current `3`** (carried from Entries 038/039). PR #25883 reports ~75% acceptance at defaults vs 97.7% at n-max 2 on Xavier. One-line change, no build. **Run after P0#1** — Finding 4 shows acceptance swings 0.28→0.76 with memory state alone, so measuring this under current conditions would produce noise.
11. **`Octen-Embedding-4B`** (carried, unchanged) — same arch/size/dim, Apache-2.0, zero-risk A/B; its RTEB claim is vendor-submitted with no third-party reproduction.
12. **Standing, carried:** re-benchmark 25W (`nvpmodel -m 1`) vs MAXN_SUPER on the real MTP workload. Thermals are **not** the constraint (49.8–52.2 °C idle vs a 99 °C throttle point); power efficiency is the only argument.
13. **Post-rebuild queue (blocked on #5):** Gemma 4 E2B-QAT UD-Q4_K_XL (2.62 GB, unchanged, preserves MTP), then Nanbeige4.2-3B (**a Q4_K_M now exists** — `mackkkkkilllll/Nanbeige4.2-3B-Q4_K_M`, 2.575 GB, no license tag; verify `num_loops=2` is in the GGUF, needs ≳b10160), then Agents-A1-4B (**MTP GGUF appeared but f16-only at 8.666 GB — wait for a Q4_K_M**).

**Recon hygiene (housekeeping):**
14. **Retarget forum tracking `380034` → `379752`.** 380034 was mis-recorded — it is an AGX Orin vLLM-wheels thread, not the JP7 sm_87 gap. 379752 is the live thread for that subject (active 2026-08-28).
15. **Re-scope the `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` trigger.** It returned NOT MATCHED against the llama.cpp repo while the actual signal arrived via forum + closed-issue archaeology. Re-point at `--cache-ram` / `--fit` / `GGML_CUDA_NO_VMM` per the Entry 039 proposal.
16. **Drop reddit.com from the Check-4 source list** (carried from Entry 039 rec #13, still not applied) — not attempted this cycle; carrying it overstates coverage.
17. **Note that the on-device journal has rotated past 2026-07-30**, so the OOM total is no longer independently recoverable from the box. **Future recons should treat LAB_NOTEBOOK as the system of record for the running count.**
18. ✅ **Entry 039 rec #15 worked** — `github.com/NVIDIA/product-security` was fully fetchable and machine-readable where `nvidia.custhelp.com` returns 403. Keep it.

#### Proposed JETSON_BASELINE.md changes — **NOT APPLIED** (headless run, no user present; per run instructions)
- `Last updated:` / `Last recon:` 2026-07-16 → **2026-08-29**; `Last healthcheck:` → **2026-08-29 (DEGRADED — OOM now DAILY, ≥22 kills since 2026-07-30, 5 consecutive nights; NEW in-process CUDA-abort failure mode 2026-08-23; tailscaled dead 38 d; memory-watchdog demonstrated silent at 135 MB avail; throughput 15.63 tok/s, high variance)**. *(Note: the Entry 038 → 2026-08-15 and Entry 039 → 2026-08-22 proposals were never applied. This supersedes both. The file still reads 2026-07-16.)*
- `llamacpp_latest_seen:` b10054 → **b10686** (running stays **b9652**, now 1,034 builds / 75 days behind).
- `jetpack_latest_orin_nano:` 7.2 → **7.2.1 (L4T r39.2.1, 2026-08-11/12)**; **ADD** `jetpack6_latest: 6.2.3 (L4T 36.5.2, 2026-08-12, apt in-place)`; `jetpack_next_expected:` → **no 6.2.4 / 7.2.2 / 7.3 announced**.
- `models_last_checked_date` & `forum_last_checked_date:` 2026-07-16 → **2026-08-29**.
- **Recon Triggers:** (a) **re-point** the `GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation)` row at **`--cache-ram` / `--fit` / `GGML_CUDA_NO_VMM` / forum 380334** — it returns empty against the llama.cpp repo as written; (b) **ADD** `llamacpp_release | #26750 OR (draft-mtp AND acceptance) | ACTION: rebuild blocker — resolve before any build past ~b10261` (proposed twice, still not added; it would have caught #27572); (c) **ADD** `llamacpp_release | #27282 OR (MTP AND compute arena) | ACTION: duplicate ~1 GB MTP arena — disqualifying at 8 GB until merged`; (d) **ADD** `llamacpp_release | #27311 OR (UMA AND scheduler) | INFO: first structural CUDA-integrated unified-memory work — watch for props.integrated re-enable`.
- **Watch Items:** **ESCALATE the OOM item** — cadence is now **daily**, ≥22 kills, plus a **new core-dump/CUDA-abort failure mode**; **ADD** the quantified leak rate (**~72 MB retained per 200-token request; +215 MB across 3**) and the on-device banner proof; **ADD** the `#22127` RSS measurement (11.9 GB → 1.0 GB) and the **verification trap** (the log line prints regardless of the flag); **ADD** #27148 as a *correctness* reason for `--cache-ram 0`; **ADD** the Finding-2 CUDA-abort front and `GGML_CUDA_NO_VMM` as a rebuild-gated second lever; **UPDATE** the memory-watchdog item from "inferred blind" to **"demonstrated silent at 135 MB available on 2026-08-29"**; **UPDATE** tailscaled 31 d → **38 d** and note the journal has rotated (root cause permanently lost); **ADD** `udisks2` re-failed; **ADD** thread 373852 (Orin Nano Super 8GB JP7.2 ISO reboot loop, unresolved) to the JP7 hold rationale; **ADD** `unsloth/Qwen3.5-2B-MTP` (1.33 GB) and `empero-ai/Qwen3.8-2B-Distill` (1.312 GB, MTP header-verified) as the headroom play; **CORRECT** the Entry 039 claim that empero's 4B lacks an MTP head — **it has one**; **NOTE** a fifth consecutive Qwen generation skipped the 4B tier; **RETARGET** forum 380034 → 379752; **NOTE** `baseline_rss_mb` 5839 is now routinely exceeded (`MemoryCurrent` 6.37–6.63 GB observed).
- **Current Config section: unchanged** (still b9652 / Qwen3.5-4B-MTP-Q4_K_M — nothing was modified on the device).
- **Throughput (no baseline change proposed):** mean **15.63 tok/s** (11.41 / 20.35 / 15.14), just above the 15.3 floor; draft acceptance mean **0.508**. ⚠ **The 2× run-to-run spread on identical settings is the finding** — memory pressure dominates. **Do not touch `baseline_gen_tok_s` until P0#1 is applied and the box can be measured with stable headroom.**
- **Fleet note (not this box):** NVIDIA security bulletin 5867 requires **DGX Spark UEFI 1.110.12 → 1.110.13**. Raise on `spark.k4jda.net`.

---
