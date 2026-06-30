# JetPack 7.2 Upgrade Plan — Jetson Orin Nano Super 8GB

**Generated:** 2026-06-15
**Author:** Claude Code
**Type:** Full-reflash OS migration (destructive — wipes the NVMe boot drive)
**Status:** PLAN — not scheduled. Gated on the JetPack 7.2 ecosystem maturing (see §0).

---

## 0. Executive Summary & Decision Gate

**Upgrade:** JetPack 6.2.2 (L4T R36.5.0, Ubuntu 22.04, kernel 5.15, CUDA 12.6) → **JetPack 7.2 (L4T r39.2, Ubuntu 24.04, kernel 6.8, CUDA 13.2.1, cuDNN 9.20.0, TensorRT 10.16.2)**. Released 2026-06-01/02 with explicit Orin Nano Super 8GB support (Entry 026, Check 1).

**This is a full reflash, not an apt/OTA upgrade.** There is no in-place path from the 6.x line. The new unified USB ISO installer (SD-card images discontinued) or SDK Manager writes a fresh image to `nvme0n1`, **destroying everything currently on the drive** — including the ~34 GB of models, all scripts, every systemd unit, and the entire Phase 1 platform-hardening work. Recoverability is therefore the dominant design concern of this plan.

**Why upgrade at all (benefit case):**
1. **Possible structural fix for the chronic OOM/NvMap problem.** The entire OOM history (Entries 023/025/026/028) traces to NvMap/GPU memory invisible to kernel OOM accounting on the R36/kernel-5.15 stack. Kernel 6.8 + CUDA 13.2 substantially rework the memory/allocator path; this upgrade is the single most plausible *structural* (vs mitigation) fix. **This is the primary strategic reason.**
2. CUDA 13.2 + newer cuDNN/TensorRT; arm64-SBSA container support (upstream vLLM etc. run unmodified).
3. nvidia-container-toolkit CVE (open on 6.2.2's 1.16.2) is only fixed in the 7.x line.
4. End of the 6.x line for Orin Nano — staying on 6.2.2 is a terminal branch.

**Why NOT yet (cost/risk case) — the gate:**
- **Power-mode TNSPEC bug (confirmed, unfixed at source as of Entry 026).** The 7.2 ISO installs a *non*-super TNSPEC; only 7W/15W appear, 25W/MAXN_SUPER vanish. Our board is the super devkit (`3767-300-0005-...-super`). Workaround exists (§5) but the source fix should land first.
- **CUDA-12.6 ecosystem still bound.** Prebuilt dustynv jetson-containers, Ollama binaries, and PyTorch SBSA cu126 wheels break under 7.2 (we don't use these for inference, but note it).
- **llama.cpp must be rebuilt from source against CUDA 13.2** — our b8987 binary will not run.

**DECISION GATE — do not schedule until ALL of:**
- [ ] JetPack 7.2.x point release or community confirmation that the power-mode TNSPEC bug is fixed (or the §5 workaround is validated on our exact board).
- [ ] ~2–4 weeks of community field reports on Orin Nano Super 8GB stability (target re-evaluation **≥ 2026-06-25**, per Entry 026 Watch Items).
- [ ] A maintenance window with physical access to the device (reflash requires the recovery jumper + a host PC).
- [ ] Off-device backup completed and verified (§2).

> This plan **subsumes** IMPLEMENTATION_PLAN.md Phase 2 (the llama.cpp rebuild becomes mandatory here) and requires **re-applying all of IMPLEMENTATION_PLAN.md Phase 1** (cgroup config, memory-watchdog, CMA drop-in, --mlock removal, MAXN power mode) on the fresh image — see §6.

---

## 1. Current-State Inventory (the ground truth to recreate)

Captured 2026-06-15. Re-capture immediately before the reflash.

| Item | Current value |
|------|---------------|
| Board / TNSPEC | Jetson Orin Nano Super 8GB devkit — `3767-300-0005-R.1-1-0-jetson-orin-nano-devkit-super` |
| L4T / JetPack | R36.5.0 / 6.2.2; `nvidia-l4t-core 36.5.0-20260115194252` |
| Boot | NVMe `nvme0n1` (931.5 GB); root `nvme0n1p1` (824 GB fs, 127 GB used, 664 GB free); `/boot/efi` on p10; A/B redundant boot partitions present |
| CUDA | 12.6 |
| llama.cpp | b8987 (commit 5f0ab726f), CUDA build flags: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` |
| Inference baseline | Qwen3.5-4B-Q4_K_M, 32768 ctx, q8_0 KV, full offload, MAXN_SUPER → **15.2–15.3 tok/s gen, ~157 pp** |
| Models | `~/llm-server/models/` — **34 GB** (Qwen3.5-4B Q4/Q5, Nemotron-3-Nano-4B, Qwen3-Embedding-4B Q4/Q6, Qwen3-Embedding-0.6B, Qwen2.5-3B/7B, DeepSeek-R1-Distill-7B, Qwen2.5-Coder-3B — full list in JETSON_CONFIG.md) |
| Service stack | `myscript.service` (User=claude, SupplementaryGroups=render) + drop-ins: `oom-protect.conf` (OOMScoreAdjust=-900), `memory-limits.conf` (MemoryMax=6400M), `cma-compact.conf` (ExecStartPre defrag); `memory-watchdog.service` (root, file-swap watchdog) |
| Scripts | `~/llm-server/{start-server.sh, start-qwen35-server.sh, start-nemotron-server.sh, start-embedding-server.sh, start-experiment.sh, bench.sh, memory-watchdog.sh}`, `mode.txt` |
| Access | user `claude`, passwordless sudo (specific NOPASSWD list — see §3), SSH key `id_claude_code`, Tailscale (`jetson.k4jda.net`), `render` group for `/dev/dri/renderD128` |
| Swap | 6× zram (~3.8 GB) + `/ssd/16GB.swap` file (PRIO -2) |

---

## 2. Phase A — Off-Device Backup (CRITICAL, do first, verify before touching the drive)

**Constitution rule: the reflash is irreversible at the drive level. Nothing proceeds until backups exist off-device and are verified.**

### A.1 — Stage a full backup to a second host
Target: **homeserver** (`claude@homeserver.k4jda.net`, Unraid array storage) — has capacity for 34 GB + configs. Spark is an alternative.

- [ ] Models (34 GB): `rsync -av ~/llm-server/models/ claude@homeserver.k4jda.net:/mnt/user/backups/jetson-2026-06-15/models/` — and record a checksum manifest (`sha256sum *.gguf > models.sha256`). *(Fallback if no host: models are all re-downloadable from HuggingFace; record the exact repo IDs/quants from JETSON_CONFIG.md so re-download is deterministic. rsync is far faster than re-download.)*
- [ ] Config bundle (small): tar the scripts, systemd units, drop-ins, sudoers, network/tailscale, build flags into one archive and copy off-device:
  - `~/llm-server/*.sh`, `~/llm-server/mode.txt`, `~/llm-server/memory-watchdog.sh`
  - `/etc/systemd/system/myscript.service` + `myscript.service.d/*`
  - `/etc/systemd/system/memory-watchdog.service` (+ `.d/` if any)
  - `/etc/nvpmodel.conf`, `/var/lib/nvpmodel/status`
  - `/etc/sudoers.d/*` (the claude NOPASSWD rules), `/etc/fstab` (the `/ssd/16GB.swap` + zram config), `/etc/systemd/system/*.swap` or zram-generator config
  - `~/.ssh/authorized_keys`, hostname, `/etc/tailscale` state (or just plan to re-auth Tailscale)
  - `~/llm-server/backups/` (prior backup dirs — keep the audit trail)
- [ ] Copy this repo's `JETSON_BASELINE.md`, `JETSON_CONFIG.md`, `LAB_NOTEBOOK.md` are already in git (off-device by definition) — no action, but confirm they're pushed.

### A.2 — (Recommended) Full NVMe image for true rollback
The cleanest rollback from a failed/regressed 7.2 is to restore the working 6.2.2 image. With physical access and the host PC:
- [ ] Clone `nvme0n1` to an image file on the host before reflashing (e.g., boot the host-side flashing environment, `dd if=/dev/nvme0n1 of=jetson-6.2.2-nvme.img bs=64M status=progress` over the recovery connection, or pull the NVMe and image it on another machine). ~127 GB used → compress (`gzip`/`zstd`) to fit.
- [ ] **If imaging isn't feasible**, rollback degrades to "re-flash 6.2.2 from the old JetPack installer + restore from A.1" — slower but viable since 6.2.2 images remain available.

### A.3 — Verify
- [ ] Re-checksum models on the backup host against `models.sha256` — must match.
- [ ] Confirm the config archive extracts cleanly and contains every file listed.
- [ ] **Gate:** do not proceed to Phase B until A.1 + A.3 pass (and A.2 if doing image rollback).

---

## 3. Phase B — Prerequisites & Window

- [ ] **Host PC** (x86 Ubuntu) with NVIDIA SDK Manager **or** the JetPack 7.2 USB ISO written to a ≥16 GB USB stick. Confirm SDK Manager lists JetPack 7.2 / L4T r39.2 for Orin Nano.
- [ ] **UEFI firmware ≥ 36.x** on the device (capsule update). 36.4.3 users hit capsule-update timeouts; workaround is manual USB boot selection (Entry 026). Verify/refresh firmware first.
- [ ] **Physical access**: recovery jumper (J14 pins 9–10 to force recovery mode), USB-C cable to host, keyboard/monitor or serial console for first boot.
- [ ] **Downtime window**: this is a live inference server (contact-center-lab embeddings + chat). Budget **half a day** end-to-end (reflash + rebuild + restore + verify). Announce/disable dependent consumers.
- [ ] Record the claude NOPASSWD sudoers list to recreate exactly:
  `docker, modprobe, systemctl, reboot, tee, dpkg, apt, apt-get, depmod, cp, mv, rm, ln, mkdir, chmod, chown, mount, umount, nvidia-smi, nvpmodel, jetson_clocks, journalctl, cat, ls, sysctl, nmcli` (all NOPASSWD).

---

## 4. Phase C — Reflash

- [ ] Put device into recovery mode (J14 9–10 jumper + power), connect to host.
- [ ] Flash JetPack 7.2 via SDK Manager (GUI, selects components) **or** boot the USB ISO installer.
- [ ] **Select NVMe (`nvme0n1`) as the install target** — the installer must be pointed at NVMe explicitly (SD images are gone; default may differ). This wipes the drive.
- [ ] Complete first-boot OOBE: create the `claude` user (match UID if it matters for file ownership on restored data), set hostname `jetson`, locale, timezone.
- [ ] Confirm boot: `cat /etc/nv_tegra_release` shows R39, `uname -r` shows 6.8.x, `nvcc --version` (or `/usr/local/cuda/bin/nvcc`) shows 13.2.

---

## 5. Phase D — Base Bring-up & the Power-Mode Bug

- [ ] Restore SSH: add `id_claude_code` public key to `~/.ssh/authorized_keys`; verify `ssh -i ~/.ssh/id_claude_code claude@<ip>` works.
- [ ] Recreate passwordless sudo: write `/etc/sudoers.d/claude` with the §3 NOPASSWD list (`visudo -c` to validate).
- [ ] Rejoin Tailscale (`tailscale up`), confirm `jetson.k4jda.net` resolves. **Then verify `tailscaled` doesn't crash-loop** (`systemctl is-active tailscaled`; `journalctl -u tailscaled | grep -i panic`): the 1.98.4+ TPM probe panics on OP-TEE fTPM errors (Entry 033), and JP7.2 ships a new fTPM/OP-TEE stack — if it panics, pin/downgrade tailscale or disable its TPM state-sealing via a systemd drop-in.
- [ ] Add `claude` to the `render` group (CUDA needs `/dev/dri/renderD128`); verify `ls -l /dev/dri/renderD128` group + membership.
- [ ] **Fix the power-mode TNSPEC bug (known 7.2 issue):** after flash, `nvpmodel -q` will likely show only 7W/15W (non-super TNSPEC `3767-300-0005-X`). Restore the **super** spec / 25W+MAXN modes:
  - Confirm `/etc/nvpmodel.conf` contains the MAXN_SUPER (id 2) and 25W (id 1) POWER_MODEL sections; if the flashed conf is the non-super variant, replace with the super `nvpmodel.conf` (from backup A.1, or the correct super conf for r39.2).
  - `sudo nvpmodel -m 2` (MAXN_SUPER — our validated winner, ~8% over 25W) and confirm `/var/lib/nvpmodel/status` persists `pmode:0002`.
  - If the source fix shipped in the 7.2.x you install, this may be unnecessary — verify `nvpmodel -q` shows MAXN_SUPER.
- [ ] Recreate swap to match: zram (zram-generator or the distro default) + the `/ssd/16GB.swap` file at PRIO -2 (`/etc/fstab` + `swapon`). The file-swap watchdog (§6) depends on the file swap existing at low priority.

---

## 6. Phase E — Rebuild llama.cpp + Restore the Inference Stack (re-applies IMPLEMENTATION_PLAN Phase 1 + 2)

- [ ] **Rebuild llama.cpp against CUDA 13.2:** clone, checkout the latest release tag (≥ b9596), build with the **same flags** verified to be optimal:
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release` then `cmake --build build -j6`. Expect CMake/toolchain to want CUDA 13.2's nvcc — ensure `CUDA_HOME=/usr/local/cuda` points at 13.2.
- [ ] Restore models from backup A.1 (`rsync` back into `~/llm-server/models/`), re-verify checksums.
- [ ] Restore scripts. **Migrate CLI flags** for the new llama.cpp (b9131 renames + b9360 `LLAMA_ARG_*` — IMPLEMENTATION_PLAN item 2.3): diff every flag against the new `--help` before first start. Note our scripts already have `--mlock` removed (Entry 029) and ctx 32768 retained.
- [ ] **Recreate the full systemd stack (IMPLEMENTATION_PLAN Phase 1 work):**
  - `myscript.service` (User=claude, SupplementaryGroups=render, LD_LIBRARY_PATH/PATH/CUDA_HOME for the 13.2 paths — these CHANGE under 7.2, update them).
  - drop-ins: `oom-protect.conf` (OOMScoreAdjust=-900), `memory-limits.conf` (MemoryMax=6400M, no MemoryHigh), `cma-compact.conf` (ExecStartPre defrag).
  - `memory-watchdog.service` + `memory-watchdog.sh` (the file-swap-triggered watchdog). **Re-validate the watchdog's NvMap path** — `/sys/kernel/debug/nvmap/iovmm/clients` may move or change format under kernel 6.8; update `nvmap_iovmm_total()` accordingly.
  - `daemon-reload`, `enable --now` both services.
- [ ] Set `mode.txt` to `qwen35`; start; confirm full GPU offload + healthy.

---

## 7. Phase F — Verification & the NvMap/OOM Re-evaluation

- [ ] **Health:** all 5 modes start; smoke inference on qwen35 + embedding.
- [ ] **Benchmark vs baseline:** `bench.sh jp7.2` — compare gen tok/s to the 15.2–15.3 baseline. Community reports ~no regression on 7.2; investigate if >15% off.
- [ ] **THE KEY EVALUATION — did kernel 6.8 fix the NvMap accounting?** Re-run the Entry 028 footprint analysis: `MemAvailable` over time, `VmSwap` growth, `nvmap iovmm` vs RSS, file-swap usage. The question: does the box still degrade to ~0 available over days, or does 6.8 account for NvMap properly (giving real available-RAM headroom)? Record the verdict — it determines whether the watchdog/ctx mitigations are still needed or can be relaxed.
- [ ] **OOM resilience:** confirm `OOMScoreAdjust=-900` honored; let the watchdog accumulate heartbeat for a few days; watch the early-AM windows for any OOM (the original failure signature).
- [ ] **Reboot test:** confirm the full stack auto-recovers and MAXN_SUPER persists.
- [ ] Update `JETSON_BASELINE.md` (jetpack_version → 7.2, cuda → 13.2, llamacpp_version, new baselines) and `JETSON_CONFIG.md`; write a LAB_NOTEBOOK entry.

---

## 8. Phase G — Rollback

| Trigger | Action |
|---------|--------|
| Reflash fails / won't boot | Re-flash 6.2.2 (old JetPack installer) → restore from A.1; or restore the A.2 NVMe image. |
| 7.2 boots but inference regresses >15% or is unstable | If A.2 image exists: restore it (fastest, full revert). Else: re-flash 6.2.2 + restore A.1. |
| NvMap/OOM **worse** under 7.2 | Same as above. Capture forensics first (the heartbeat + snapshots) to inform a retry. |
| Power-mode bug unfixable | Tolerable short-term (run 15W) but counts as "ecosystem not ready" → consider reverting and waiting for 7.2.x. |

**Rollback is only clean if Phase A.2 (image) was done, or A.1 (backup) is complete + 6.2.2 installer is on hand. Confirm before flashing.**

---

## 9. Known JetPack 7.2 Gotchas (from Entry 026 recon)

| Gotcha | Impact | Handling |
|--------|--------|----------|
| Non-super TNSPEC installed | 25W/MAXN modes hidden | §5 nvpmodel/conf restore |
| CUDA 12.6 prebuilt ecosystem breaks (Ollama, dustynv containers, PyTorch cu126 wheels) | Those tools fail / CPU-fallback / silent NaN on sm_87 | We build llama.cpp from source; avoid prebuilt CUDA-12.6 artifacts. arm64-SBSA containers are the new path. |
| llama.cpp b8987 won't run | No inference until rebuilt | §6 rebuild against 13.2 (mandatory) |
| `jtop` shows "JetPack NOT DETECTED" | Cosmetic | Ignore / update jetson-stats |
| UEFI 36.4.3 capsule-update timeout | Flash hangs | Refresh firmware first; manual USB boot selection |

---

## 10. Open Questions / Unknowns

| ID | Unknown | Resolve by |
|----|---------|-----------|
| J1 | Does kernel 6.8 fix NvMap OOM-accounting (the root cause)? | Phase F evaluation — the strategic payoff question |
| J2 | Exact CLI-flag deltas for the llama.cpp tag we build (b9131/b9360) | `--help` diff during §6 (same as IMPLEMENTATION_PLAN 2.3) |
| J3 | Is the power-mode TNSPEC bug fixed in the 7.2.x we install? | Check release notes at decision-gate time |
| J4 | Backup host capacity/availability for 34 GB models | Confirm homeserver array space in Phase A |
| J5 | Does the watchdog's `/sys/kernel/debug/nvmap/` path/format survive kernel 6.8? | Re-validate in §6 |

---

*Companion to IMPLEMENTATION_PLAN.md (Phase 1 hardening + Phase 2 rebuild). This reflash re-applies Phase 1 and absorbs Phase 2. Source: Entry 026 recon (JetPack 7.2 findings) + 2026-06-15 device inventory.*
