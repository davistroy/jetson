# Jetson Performance Baseline

Last updated: 2026-05-09
Last recon: 2026-05-09

## Current Config
| Field | Value |
|-------|-------|
| device | Jetson Orin Nano Super 8GB |
| jetpack_version | 6.2.2 (R36.5.0) |
| cuda_version | 12.6 |
| llamacpp_version | b8987 |
| current_model | Qwen3.5-4B-Q4_K_M |
| baseline_gen_tok_s | 15.3 |
| baseline_pp_tok_s | 166 |
| baseline_rss_mb | 4969 |
| context_size | 32768 |
| gpu_layers | 999 (full offload) |
| threads | 1 |
| parallel_slots | 1 |
| kv_cache_type | q8_0 |
| flash_attn | on |
| mlock | on |

## Version Tracking
| Field | Value |
|-------|-------|
| llamacpp_latest_seen | b9093 |
| jetpack_latest_orin_nano | 6.2.2 |
| jetpack_next_expected | 7.2 (Q2 2026, Orin support) |

## Model Tracking
| Field | Value |
|-------|-------|
| current_model | Qwen3.5-4B-Q4_K_M |
| current_embedding_model | Qwen3-Embedding-4B-Q4_K_M |
| models_last_checked_date | 2026-05-09 |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-05-09 |

## Recon Triggers
| Source | Pattern | Action | Added |
|--------|---------|--------|-------|
| jetpack | JetPack 7.2 AND (Orin Nano OR Orin) | ACTION: Evaluate JetPack 7.2 upgrade (full reflash, wait for community validation) | 2026-04-13 |
| llamacpp_release | SM87 OR Jetson OR Tegra OR unified memory | ACTION: Check release notes for Jetson-specific improvements | 2026-04-13 |
| huggingface | Qwen4 OR Qwen3.5 successor | INFO: New Qwen generation may improve quality at same size | 2026-04-13 |
| forum | llama.cpp AND (performance OR optimization) AND jetson | INFO: Community optimization techniques to evaluate | 2026-04-13 |

## Watch Items
- **OOM-kill root cause** — IDENTIFIED + FIXED 2026-05-09 (Entry 023). NOT cron-triggered; structural memory pressure from default `vm.swappiness=60` aggressively swapping llama-server's anon pages on a memory-saturated 8 GB system. **Applied 3-layer fix:** sysctl tuning (`vm.swappiness=1`, `min_free_kbytes=131072`, `watermark_scale_factor=200`, `vfs_cache_pressure=50` in `/etc/sysctl.d/99-llm-inference.conf`); systemd cgroup limits (`MemoryHigh=5500M` / `MemoryMax=6000M` in `/etc/systemd/system/myscript.service.d/memory-limits.conf`); disabled `nvmemwarning.service` (X11 notification daemon, useless on headless system). Free RAM jumped 174M → 1.1G immediately. **Verify after 7 days:** `journalctl -u myscript --since '7 days ago' | grep -iE 'oom|fail'` should be empty. Re-evaluate around 2026-05-16.
- JetPack 7.2 expected Q2 2026 -- will bring Ubuntu 24.04, kernel 6.8, CUDA 13.0. Full reflash required. Wait 2-4 weeks after release for community validation.
- nvidia-smi sudoers — FIXED 2026-04-30 (path was /usr/sbin/ not /usr/bin/, missing comma before sysctl). Also added journalctl and systemd-journal group.
- Qwen3.5-4B-Claude-Distilled-v2-Q4_K_M — TESTED 2026-04-30, FAIL. Catastrophic reasoning loop: consumes entire token budget on think tokens, produces zero content. GGUF archived at ~/llm-server/models/, delete after 2026-05-30.
- Old binary backup at ~/llm-server/backup-b8766/ — can remove after 2 weeks of stable b8987 operation (after 2026-05-14)
- **`khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF`** — ADDED 2026-05-09 (Entry 022). Drop-in candidate for `experiment` mode: distilled from Qwen3.6-plus teacher (stronger than failed Claude-v2 distill), focuses on concise CoT. Zero memory risk vs current Qwen3.5-4B.
- Phi-4-mini-instruct (Q4_K_M ~2.5 GB) and Gemma-4-E2B (~2 GB) as alternative small models worth benchmarking
- llama.cpp auto-config agent post by flouisdev (2026-04-13) -- potential for dynamic mode switching optimization: https://forums.developer.nvidia.com/t/local-first-coding-agent-that-auto-configures-llama-cpp-for-maximum-hardware-performance/366366
- TensorRT Edge-LLM — EVALUATED 2026-04-30, DEFERRED. Experimental Orin support, CUDA 12.6 vs 12.8+ mismatch, no HTTP server. Revisit when Orin officially supported or JetPack ships CUDA 12.8+.
- Qwen3.6 small models -- Qwen3.6 released (27B, 35B-A3B MoE) but no 4B-class models yet. Expected late May 2026 based on large→small cadence. Direct drop-in upgrade from Qwen3.5-4B.
- Jina Embeddings v4 -- 3B model (~1.93 GB Q4_K_M), potential upgrade from Qwen3-Embedding-4B for retrieval quality
- Gemma 4 E2B/E4B -- still blocked by llama.cpp PLE issue #22243. Community confirms E4B OOM on Orin Nano. Re-evaluate when #22243 fixed.
- llama.cpp build b9093 SEEN 2026-05-09 (Entry 022) — 106 builds since b8987, zero SM87/Jetson-relevant changes; not upgrading. Re-evaluate when CUDA/flash-attn/Qwen-tagged PR lands.
- Build-flag verification CLOSED 2026-05-09: confirmed `GGML_CUDA_F16=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `CMAKE_CUDA_ARCHITECTURES=87`, `Release` — already optimal.
