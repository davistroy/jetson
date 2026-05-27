# Jetson Performance Baseline

Last updated: 2026-05-27
Last recon: 2026-05-27

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
| llamacpp_latest_seen | b9360 |
| jetpack_latest_orin_nano | 6.2.2 |
| jetpack_next_expected | 7.2 (early June 2026, confirmed by NVIDIA staff) |

## Model Tracking
| Field | Value |
|-------|-------|
| current_model | Qwen3.5-4B-Q4_K_M |
| current_embedding_model | Qwen3-Embedding-4B-Q4_K_M |
| models_last_checked_date | 2026-05-27 |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-05-27 |

## Recon Triggers
| Source | Pattern | Action | Added |
|--------|---------|--------|-------|
| jetpack | JetPack 7.2 AND (Orin Nano OR Orin) | ACTION: Evaluate JetPack 7.2 upgrade (full reflash, wait for community validation) | 2026-04-13 |
| llamacpp_release | SM87 OR Jetson OR Tegra OR unified memory | ACTION: Check release notes for Jetson-specific improvements | 2026-04-13 |
| huggingface | Qwen4 OR Qwen3.5 successor | INFO: New Qwen generation may improve quality at same size | 2026-04-13 |
| forum | llama.cpp AND (performance OR optimization) AND jetson | INFO: Community optimization techniques to evaluate | 2026-04-13 |

## Watch Items
- **OOM-kill root cause** — Entry 023 (2026-05-09) MISDIAGNOSED it as swappiness; that fix did NOT work (recurred May 17 + May 26). **TRUE root cause (Entry 025, 2026-05-27, multi-date confirmed):** GLOBAL OOM (`constraint=CONSTRAINT_NONE`), not the cgroup limit. On 8 GB unified RAM, llama-server's ~3.93 GB anon-rss at `oom_score_adj=0` is always the largest unprotected process, while snapd is vendor-protected at -900 — so the kernel always kills llama-server when early-AM maintenance (man-db rebuild 00:40–01:00, anacron, occasional snapd watchdog crash-dump as on May 26) exhausts physical RAM. **FIX APPLIED 2026-05-27 (Entry 025):** (1) `OOMScoreAdjust=-900` drop-in on `myscript.service` (`/etc/systemd/system/myscript.service.d/oom-protect.conf`) — verified live, MainPID now at -900, so llama-server survives and a maintenance job is evicted instead; (2) `man-db.timer` disabled (removed the recurring 00:40–01:00 burst). **PENDING (needs interactive sudo — `snap` not in NOPASSWD):** remove headless-useless desktop snaps (chromium, cups, gnome-46-2404, gtk-common-themes, mesa-2404) to cut snapd memory + watchdog-crash risk. Backup of original unit at `~/llm-server/backups/oom-fix-2026-05-27/`. **Verify after ~2 weeks (~2026-06-10):** `journalctl -u myscript --since '2026-05-27' | grep -i oom` should be empty.
- **Entry 023 cgroup limits re-evaluation** — `MemoryMax=6000M`/`MemoryHigh=5500M` do NOT prevent global OOM (external spiker) and live `available: 139M` shows llama-server is being reclaim-throttled at MemoryHigh (possible marginal latency cost). Consider relaxing/removing once the Entry 025 fix is confirmed effective.
- JetPack 7.2 expected Q2 2026 -- will bring Ubuntu 24.04, kernel 6.8, CUDA 13.0. Full reflash required. Wait 2-4 weeks after release for community validation.
- nvidia-smi sudoers — FIXED 2026-04-30 (path was /usr/sbin/ not /usr/bin/, missing comma before sysctl). Also added journalctl and systemd-journal group.
- Qwen3.5-4B-Claude-Distilled-v2-Q4_K_M — TESTED 2026-04-30, FAIL. DELETED 2026-05-13 (reasoning loop bug, not worth keeping).
- Old binary backup at ~/llm-server/backup-b8766/ — DELETED 2026-05-13 (2 weeks stable)
- **`khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF`** — ADDED 2026-05-09 (Entry 022). Drop-in candidate for `experiment` mode: distilled from Qwen3.6-plus teacher (stronger than failed Claude-v2 distill), focuses on concise CoT. Zero memory risk vs current Qwen3.5-4B.
- Phi-4-mini-instruct (Q4_K_M ~2.5 GB) and Gemma-4-E2B (~2 GB) as alternative small models worth benchmarking
- llama.cpp auto-config agent post by flouisdev (2026-04-13) -- potential for dynamic mode switching optimization: https://forums.developer.nvidia.com/t/local-first-coding-agent-that-auto-configures-llama-cpp-for-maximum-hardware-performance/366366
- TensorRT Edge-LLM — EVALUATED 2026-04-30, DEFERRED. Experimental Orin support, CUDA 12.6 vs 12.8+ mismatch, no HTTP server. Revisit when Orin officially supported or JetPack ships CUDA 12.8+.
- Qwen3.6 small models -- Qwen3.6 released (27B, 35B-A3B MoE) but no 4B-class models yet. Expected Q3 2026 (pushed from late May). Direct drop-in upgrade from Qwen3.5-4B.
- Jina Embeddings v4 -- 3B model (~1.93 GB Q4_K_M), potential upgrade from Qwen3-Embedding-4B for retrieval quality
- Gemma 4 E2B/E4B -- still blocked by llama.cpp PLE issue #22243. Community confirms E4B OOM on Orin Nano. Re-evaluate when #22243 fixed.
- llama.cpp build b9360 SEEN 2026-05-27 (Entry 025) — 373 builds since running b8987, zero SM87/Jetson-relevant changes; not upgrading. **Compounding breaking change:** b9131 renamed CLI args AND b9360 moved all env vars to `LLAMA_ARG_*` prefix — verify startup scripts before any future rebuild. Re-evaluate when CUDA/flash-attn/Qwen-tagged PR lands.
- JetPack 7.2 still NOT released as of 2026-05-27 (Entry 025) — only announced for Orin Nano; Q2 window closes June 30. Re-run recon immediately if it drops. Will bring CUDA 13.0 / kernel 6.8 / Ubuntu 24.04, full reflash, llama.cpp rebuild likely required.
- Build-flag verification CLOSED 2026-05-09: confirmed `GGML_CUDA_F16=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `CMAKE_CUDA_ARCHITECTURES=87`, `Release` — already optimal.
- Phi-4-mini-instruct (3.8B dense, 2.49 GB Q4_K_M, 128K ctx) — ADDED 2026-05-13 (Entry 024). First non-Qwen alternative worth benchmarking in experiment slot. GGUF at bartowski/microsoft_Phi-4-mini-instruct-GGUF.
