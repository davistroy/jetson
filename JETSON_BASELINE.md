# Jetson Performance Baseline

Last updated: 2026-07-16
Last recon: 2026-07-16
Last healthcheck: 2026-07-16 (Entry 036 — HEALTHY: 0 OOM/30d, no fTPM panic; ufw firewall found inactive → re-enabled + verified. Prior: Entry 033 fTPM tailnet outage, reboot-resolved)

## Current Config
| Field | Value |
|-------|-------|
| device | Jetson Orin Nano Super 8GB |
| jetpack_version | 6.2.2 (R36.5.0) |
| cuda_version | 12.6 |
| llamacpp_version | b9652 |
| current_model | Qwen3.5-4B-MTP-Q4_K_M (MTP self-speculative, promoted 2026-06-16) |
| baseline_gen_tok_s | 15.3 (floor; MTP gives 15–22 workload-dependent, ~18 typical) |
| baseline_pp_tok_s | 166 |
| baseline_rss_mb | 5839 (MTP soak plateau; ~550MB under MemoryMax=6400M) |
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
| llamacpp_latest_seen | b10054 (2026-07-17; running stays b9652) |
| jetpack_latest_orin_nano | 7.2 (L4T r39.2, released 2026-06-01) |
| jetpack_next_expected | n/a — 7.2 shipped; watch for 7.2.x point release fixing power-mode TNSPEC bug |

## Model Tracking
| Field | Value |
|-------|-------|
| current_model | Qwen3.5-4B-MTP-Q4_K_M (MTP speculative decoding) |
| current_embedding_model | Qwen3-Embedding-4B-Q4_K_M |
| models_last_checked_date | 2026-07-16 |

## Forum Tracking
| Field | Value |
|-------|-------|
| forum_last_checked_date | 2026-07-16 |

## Recon Triggers
| Source | Pattern | Action | Added |
|--------|---------|--------|-------|
| jetpack | (JetPack 7.2.1 OR 7.3 OR power mode fix OR TNSPEC) AND (Orin Nano OR Orin) | ACTION: JP7.2 ecosystem maturing — re-evaluate reflash window (Entry 026: wait 2–4 weeks from 2026-06-11) | 2026-06-11 |
| llamacpp_release | SM87 OR Jetson OR Tegra OR unified memory | ACTION: Check release notes for Jetson-specific improvements | 2026-04-13 |
| llamacpp_release | GGML_CUDA_ENABLE_UNIFIED_MEMORY OR (NvMap AND mitigation) | INFO: env-var/build-flag NvMap mitigation-of-record (forum thread 373627) — evaluate in experiment slot as OOM-guard/large-model enabler. (Retired the #23747 VMM trigger — CLOSED WONTFIX 2026-05-27.) | 2026-07-16 |
| huggingface | Qwen4 OR Qwen3.5 successor | INFO: New Qwen generation may improve quality at same size | 2026-04-13 |
| forum | llama.cpp AND (performance OR optimization) AND jetson | INFO: Community optimization techniques to evaluate | 2026-04-13 |

## Watch Items
- 🔒 **ufw firewall — re-enabled 2026-07-16 (Entry 036), had been found INACTIVE.** The Phase 5.6 default-deny firewall (Entry 034) silently went down at some point on the 2026-06-30 boot (empty nft ruleset; `is-enabled`=enabled but `is-active`=inactive) — API-key + key-only-SSH kept the LLM unexposed throughout. Re-enabled + verified (raw-LAN :8080 blocked, tailnet :8080 200, SSH both paths). **TODO:** add a boot-time `ufw status active` assertion (ExecStartPost or fold into the 5.7 monitor) so a silent firewall-down alerts instead of surfacing a recon later. Likely root cause: orphaned Entry-034 dead-man's-switch.
- ✅ **OOM watch — DOWNGRADE (Entry 036, 2026-07-16):** 0 OOM in ~30 days on b9652 confirms the Entry 032 resolution holds; the NvMap kernel-accounting gap is unchanged (still kernel 5.15) but is NOT an active fire. `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (env var / build flag, forum thread 373627) is the mitigation-of-record replacing the dead #23747 — still UNTESTED here; trial in the experiment slot as the OOM-guard / larger-model enabler. Re-confirm each recon: `journalctl -u myscript --since '<date>' | grep -iE 'oom|killed'` should stay empty.
- 📦 **llama.cpp b10054 latest seen (2026-07-16, Entry 036); running b9652 (~402 behind), no Jetson-specific gain in the span.** Only bundle-worthy item on a future opportunistic rebuild: b9974 `cudaMemGetInfo` OOM-crash guard. No flag/CLI breakage through b10054.
- ⚡ **25W power mode (`nvpmodel -m 1`) — standing efficiency recommendation; box still on MAXN_SUPER (Entry 036 confirms).** Re-benchmark 25W vs MAXN on the real Qwen3.5-4B-MTP workload before committing (note the JP-side reboot-on-mode-change — N/A on JP6.2.2 but relevant if we ever move to 7.2).
- 🧪 **Jackrong Qwen3.5-4B-Opus-Reasoning-Distilled (Q4_K_M 2.71 GB) — experiment-slot A/B candidate (Entry 036), same arch, zero memory risk, still un-trialed.** Quantified vs base: +5.06 GPQA-Diamond (33.82→38.88), +1.79 ARC-C (64.59→66.38); the v2 variant additionally reports −34% thinking length. Skip the untested `avalon2244` sibling.
- ⚠️ **Tailscale fTPM panic — LATENT (Entry 033, 2026-06-30):** `tailscaled` 1.98.4 panics at startup on OP-TEE fTPM errors (`send(): error -53212` → `slice bounds out of range [:-53212]`) and crash-loops, dropping the tailnet link while the host stays up. Hit it 2026-06-28 01:00 → node "offline" 2 days; **reboot cleared the fTPM state** (0 fTPM errors post-boot) and restored a direct tailnet path. **Root cause not eliminated** — a future fTPM hiccup re-triggers it. Durable fix if it recurs: pin/downgrade tailscale, or disable tailscale TPM state-sealing via a systemd drop-in. **Unexplained:** both `tailscaled` and `myscript` restarted at the same 01:00 minute with no reboot/no apt — investigate the 01:00 trigger at next touch. Diagnostics: reach the box on LAN `192.168.10.58`, check `journalctl -u tailscaled` for the panic and `journalctl -k | grep tpm`.
- ✅ **PHASE 1–3 COMPLETE (Entries 028–032, 2026-06-12→16) — box hardened + accelerated.** Live config now: de-throttled cgroup (MemoryHigh removed, MemoryMax=6400M, OOMScoreAdjust=−900); armed **memory-watchdog** (root, file-swap-triggered, reboot-durable); CMA pre-start defrag; MAXN_SUPER; `--mlock` removed; **llama.cpp b9652**; **Qwen3.5-4B-MTP self-speculative decoding** as the default qwen35 mode (+8–47% workload-dependent, lossless). Rollback assets on device: `~/llm-server/backups/{envelope-2026-06-11, scripts-mlock-2026-06-15, qwen35-pre-mtp-2026-06-16}/` and `~/llm-server/backup-b8987-bin/` (delete b8987 bin ~2026-06-29 if stable). **MTP caveat:** recurrent-memory "full prompt re-processing" churn limits cache reuse on long multi-turn (watch if multi-turn latency matters); revert = restore `qwen35-pre-mtp-2026-06-16/`. Experiment slot free.
- ✅ **CHRONIC OOM/degradation — LIKELY RESOLVED by b9652 (Entry 032):** the ~28h MTP soak held ~800 MB available steadily with file-swap=0, vs the old b8987 crawl to ~0 + 3 GB zram (Entry 028). b9652's deterministic startup KV reservation (#23907) appears to be the fix. Watchdog + MemoryMax remain the backstop. **Supersedes the OOM item below — re-confirm at next recon (`journalctl -u myscript --since '2026-06-15' | grep -iE 'oom|killed'` should stay empty); NvMap accounting itself is unchanged (still kernel 5.15) so JetPack 7.2 kernel 6.8 remains the structural fix.**
- ⚠️ **OOM ROOT CAUSE (Entry 026, 2026-06-11) — Entry 025 verification FAILED, now mitigated (see above):** llama-server OOM-killed 2026-06-07 01:15 EDT despite the Entry 025 layers working as designed (adj −900 kept it alive ~15 min into a global OOM storm while the kernel killed everything else first). Kernel dump shows the true consumer: **~7+ GB of 7.6 GB managed RAM held by NvMap/GPU memory invisible to kernel OOM accounting** — killing userspace freed nothing. NOT man-db (timer dead since 05-27), NOT a cgroup kill; immediate trigger unrecoverable (journal gap). Mitigation candidates (awaiting approval): (a) userspace watchdog on `MemAvailable`/NvMap iovmm that restarts `myscript` before global exhaustion; (b) local VMM allocator patch — llama.cpp #23732/#23747 (`GGML_CUDA_VMM_BUFFERS=1`, ~105 lines in ggml-cuda.cu, demoed on this exact board) closed unmerged — watch for resubmission (trigger added); (c) CMA workaround `sync` + `drop_caches` + `compact_memory` in start script before model load (forums.developer.nvidia.com/t/370049 — NVIDIA confirms r36.5 NvMap fixes don't address it); (d) JetPack 7.2 kernel 6.8 may change NvMap/CMA behavior — couple with upgrade evaluation. **Next recon: re-scan `journalctl -u myscript --since '2026-06-11' | grep -iE 'oom|killed'`.**
- **Entry 023 cgroup limits — REMOVAL RECOMMENDED (Entry 026, awaiting approval):** June 7 proved `MemoryMax`/`MemoryHigh` don't prevent global OOM, and the unit sits ~216 MB below the MemoryHigh reclaim throttle with RSS up ~250 MB in 4 days (5228 MB live vs 4969 baseline). Cost without benefit — relax/remove.
- **JetPack 7.2 RELEASED 2026-06-01/02** (L4T r39.2, Ubuntu 24.04, kernel 6.8, CUDA 13.2.1, cuDNN 9.20, TensorRT 10.16.2; Orin Nano Super 8GB explicitly supported). **Full reflash** via new USB ISO installer (SD-card images discontinued); UEFI ≥ 36.x required (36.4.3 capsule-update timeout workaround: manual USB boot). **DO NOT upgrade yet (re-evaluate ~2026-06-25+):** power-mode TNSPEC bug (only 7W/15W visible; workaround `sudo nvpmodel -m 2`) unfixed at source; CUDA 12.6 prebuilt ecosystem (dustynv containers, Ollama, PyTorch wheels) broken under JP7.2; llama.cpp source rebuild vs CUDA 13.2 required (budget `start-*.sh` updates for b9131/b9360 breaking changes in the same window). Upside beyond currency: possible structural NvMap/OOM fix, arm64-SBSA containers (upstream vLLM runs unmodified), container-toolkit CVE fixed only in 7.x.
- **MTP speculative decoding (Entry 026) — strongest throughput lead:** mainline llama.cpp PR #22673 (merged 2026-05-16, AFTER running b8987) + `unsloth/Qwen3.5-4B-MTP-GGUF` (Q4_K_M 2.83 GB, same weights + MTP head, ~10% KV overhead) → claimed 1.5–2× decode; corroborated by cortexist/llama.cpp fork (+30–40% tok/s on Orin NX, forums.developer.nvidia.com/t/372493). Trial in experiment slot after a rebuild: validate the reported MTP memory-leak (workaround flag exists) under the OOM guard and measure draft acceptance (#23322 reports low acceptance on SWA/hybrid models) before promoting.
- llama.cpp **b9596 SEEN 2026-06-11** (Entry 026) — ~609 builds past running b8987; **no new breaking changes** beyond b9131 CLI renames + b9360 `LLAMA_ARG_*` env prefix. Rebuild case: MTP support (above), #23907 startup KV-cache reservation (deterministic fail-at-startup vs mid-run OOM with q8_0 KV), #24360 CUDA ssm_scan race fix. **SM87 MoE decode hang #19219 closed "not planned"** — avoid small MoE models on this device until resolved.
- **Power mode check (Entry 026):** smolhub benchmark on this exact device (2026-05-29): 25W (`nvpmodel -m 1`) is the throughput/efficiency sweet spot; MAXN_SUPER costs +17% power for −3%..+8% throughput. Verify current mode with `nvpmodel -q`.
- **TensorRT-Edge-LLM v0.8.0 (2026-06-03)** — now NVIDIA's official Jetson LLM runtime (HF→ONNX, quantization, EAGLE speculative decoding); TensorRT-LLM Jetson branch confirmed unmaintained. Was DEFERRED 2026-04-30 over CUDA 12.6 mismatch — re-check compatibility matrix; JP7.2's CUDA 13.2 may unblock. Orin Nano 8GB + Qwen-4B support unverified.
- **`khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF`** — ADDED 2026-05-09 (Entry 022). Drop-in candidate for `experiment` mode; zero memory risk. Still untested.
- **Jackrong Qwen3.5-4B fine-tunes (Entry 026)** — zero-memory-cost experiment-slot trials: `Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` (2.71 GB; GPQA-D 33.8→38.9, ARC-C 64.6→66.4 vs base) and `Qwen3.5-4B-Neo-GGUF` (concise-reasoning — fewer tokens/answer matters at 15 tok/s).
- Phi-4-mini-instruct (3.8B dense, 2.49 GB Q4_K_M, 128K ctx) — ADDED 2026-05-13 (Entry 024). First non-Qwen alternative worth benchmarking in experiment slot. GGUF at bartowski/microsoft_Phi-4-mini-instruct-GGUF.
- **Gemma 4 E2B — UNBLOCKED (Entry 026):** PLE issue llama.cpp#22243 closed completed 2026-04-23 (premise wrong — PLE already implemented in `src/models/gemma4-iswa.cpp`; ≥Q4_K quant of PLE tensors safe). E2B Q4_K_M = 3.11 GB — borderline over the 3 GB ceiling, viable only with reduced ctx; E4B still OOM territory. Low priority.
- Qwen3.6 small models — still no 4B-class as of 2026-06-11 (27B + 35B-A3B MoE only); Q3 2026 expectation unchanged. Direct drop-in upgrade from Qwen3.5-4B when it ships.
- **jina-embeddings-v5-text (supersedes Jina v4 tracking, Entry 026):** small = 677M, distilled FROM Qwen3-Embedding-4B, MMTEB 67.0 (vs ~69.5 for the 4B), per-task GGUFs ~0.4–0.5 GB Q4_K_M, CC BY-NC. Not better than the 4B absolute — upgrade candidate for the lightweight slot (vs Qwen3-Embedding-0.6B) or if co-residency with chat mode is ever needed (~2 GB freed).
- llama.cpp auto-config agent post by flouisdev (2026-04-13) -- potential for dynamic mode switching optimization: https://forums.developer.nvidia.com/t/local-first-coding-agent-that-auto-configures-llama-cpp-for-maximum-hardware-performance/366366
- nvidia-smi sudoers — FIXED 2026-04-30 (path was /usr/sbin/ not /usr/bin/, missing comma before sysctl). Also added journalctl and systemd-journal group.
- Build-flag verification CLOSED 2026-05-09: confirmed `GGML_CUDA_F16=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`, `CMAKE_CUDA_ARCHITECTURES=87`, `Release` — already optimal.
