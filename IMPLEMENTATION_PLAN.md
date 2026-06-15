# Implementation Plan

**Generated:** 2026-06-11 16:38:06
**Based On:** LAB_NOTEBOOK.md Entry 026 (biweekly recon, 2026-06-11), Entry 027 (ultra-plan analysis, 2026-06-11), live-device investigation (read-only SSH, 2026-06-11)
**Total Phases:** 4
**Estimated Total Effort:** ~350 LOC across ~14 files (11 on-device at `claude@jetson.k4jda.net`, 3 repo docs)

---

## Executive Summary

This plan hardens and accelerates the Jetson Orin Nano Super 8GB inference server following the Entry 026 finding that the true OOM root cause is NvMap/GPU memory invisible to kernel OOM accounting — a consumer no cgroup or oom_score tuning can defend against. The integrated response: remove the counterproductive MemoryHigh reclaim throttle (currently 170 MB from strangling the server) while retaining a raised MemoryMax as a llama-server-scoped leak backstop, and deploy a root-level memory watchdog that acts where the kernel is blind and captures the forensic timeline that the June 7 incident's journal gap denied us.

On the performance side, the plan finalizes the power-mode envelope by measurement (the box runs MAXN_SUPER; community data favors 25W), rebuilds llama.cpp from b8987 to ≥b9596 with a gated CLI-flag migration across all five start scripts, then trials MTP speculative decoding (claimed 1.5–2× decode on the incumbent Qwen3.5-4B) in the experiment slot under watchdog protection — the watchdog and the MemoryMax backstop exist precisely because the MTP merge has a reported memory leak. Optional fine-tune trials close the plan.

Interrelated issues are solved as one system: the throttle removal, watchdog, and power decision jointly define the measurement envelope all later benchmarks depend on; the watchdog's MAINTENANCE bypass exists for the rebuild's stop-the-server build window; the watchdog's heartbeat CSV doubles as the MTP soak instrument; and rewriting the stale `start-experiment.sh` (which points at a model deleted 2026-05-13 — a live crash-loop landmine) is folded into the MTP trial rather than patched separately.

---

## Plan Overview

Strict sequential phasing — this is a single live device with one server port; nothing on-device parallelizes. Phase 1 must precede Phase 2 because the power mode and de-throttled cgroup state define the benchmark baseline that the rebuild's keep/rollback gate compares against. Phase 2 must precede Phase 3 because MTP support (PR #22673, merged 2026-05-16) postdates the running b8987 build. Phase 3 precedes Phase 4 only because both consume the single experiment slot.

Critical path: Phase 1 (one evening) → 1–2 day heartbeat settle → Phase 2 (45–90 min maintenance window) → Phase 3 (setup + 48 h soak) → Phase 4 (optional, time-boxed).

**Execution conventions (all phases):**
- `SSH="ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net"` — all on-device paths below are on the Jetson unless marked `(repo)`.
- All needed privileged commands are in claude's NOPASSWD sudoers (verified 2026-06-11: `systemctl`, `tee`, `cp`, `mv`, `rm`, `mkdir`, `cat`, `ls`, `nvpmodel`, `jetson_clocks`, `journalctl`, `nvidia-smi`, `sysctl`).
- Backup before every config change: `~/llm-server/backups/<change>-<date>/` (constitution: Jetson files are not git-recoverable).
- **NEVER run `systemctl revert myscript`** — it deletes ALL drop-ins including `oom-protect.conf` (the −900 OOM protection).
- Every phase ends with a LAB_NOTEBOOK.md entry (constitution: immediate logging).

### Phase Summary Table

| Phase | Focus Area | Key Deliverables | Est. Complexity | Dependencies | Execution Mode |
|-------|------------|------------------|-----------------|--------------|----------------|
| 1 | Platform envelope (CS-A) | De-throttled cgroup config, armed memory watchdog + heartbeat telemetry, CMA pre-start compaction, measured power-mode decision + new baseline | M (~8 files, ~200 LOC) | None | Sequential |
| 2 | llama.cpp rebuild + migration (CS-B) | ≥b9596 binary (same CMake flags), 5 start scripts migrated past b9131 renames, benchmark-gated keep/rollback | M (~8 files, ~80 LOC) | Phase 1 | Sequential |
| 3 | MTP speculative decoding trial (CS-C) | Rewritten `start-experiment.sh`, MTP GGUF deployed, hour-1 + 48 h soak gates, promote/revert decision | S (~4 files, ~50 LOC) | Phase 1, Phase 2 | Sequential |
| 4 | Fine-tune trials (CS-D, optional) | Two Jackrong fine-tunes evaluated against quality probe set, keep/promote/delete decision | S (~3 files, ~20 LOC) | Phase 3 | Sequential |

### Execution Hints

| Phase | Model Tier | Context Budget | Notes |
|-------|------------|----------------|-------|
| All (default) | `sonnet` | Standard | Per-item Model Tier fields take precedence. All on-device work is sequential — do NOT parallelize work items that touch the Jetson; there is one device and one server port. |

### Milestones

| Milestone | Phases | Description |
|-----------|--------|-------------|
| Protected | 1 | Box survives the next NvMap memory storm with ≤60 s watchdog-mediated recovery and full forensics; throttle risk eliminated; power envelope decided by data |
| Current | 1–2 | Running a current llama.cpp (~6 weeks of fixes incl. CVE posture for the LAN-exposed server) with migrated scripts that JetPack 7.2 will inherit |
| Faster | 1–3 | MTP decode verdict delivered: either promoted (~20–30 tok/s) or rejected with measured evidence |
| Complete | 1–4 | Fine-tune candidates dispositioned |

<!-- BEGIN PHASES -->

---

## Phase 1: Platform Envelope (CS-A)

**Estimated Complexity:** M (~8 files, ~200 LOC)
**Dependencies:** None
**Execution Mode:** Sequential

### Goals

- Eliminate the MemoryHigh reclaim throttle (170 MB from engagement at investigation time) while retaining a raised MemoryMax as the llama-server-scoped runaway backstop
- Deploy an armed, guarded memory watchdog that acts where kernel OOM accounting is blind (NvMap) and builds the forensic timeline missing from the June 7 incident
- Decide MAXN_SUPER vs 25W by measurement and establish the new official throughput baseline all later phases compare against

### Work Items

#### 1.1 Cgroup limit rework: remove MemoryHigh, raise MemoryMax to 6400M ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- verified live; bench 15.2-15.3 tok/s. See LAB_NOTEBOOK Entry 028. -->
**Status (orig): PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Rec 2, Entry 027 CS-A(A1)
**Files Affected:**
- `/etc/systemd/system/myscript.service.d/memory-limits.conf` (replace)
- `/run/systemd/system.control/myscript.service.d/50-MemoryHigh.conf` (delete)
- `/run/systemd/system.control/myscript.service.d/50-MemoryMax.conf` (delete)
- `~/llm-server/backups/envelope-<date>/` (create — backup of all three + `oom-protect.conf`)

**Description:**
Entry 026/June 7 forensics proved MemoryHigh/MemoryMax do not prevent global OOM (the consumer is NvMap, invisible to cgroup accounting), while live MemoryCurrent (5.21 GiB) sits 170 MB below MemoryHigh (5.37 GiB) — reclaim throttling is imminent and `--mlock` concentrates its pressure on the unlocked remainder. Replace the throttle with a pure hard-cap: new `memory-limits.conf` containing only `MemoryMax=6400M` (no MemoryHigh). MemoryMax is retained (raised from 6000M) solely as the backstop against a llama-server-side runaway — relevant in Phase 3, where an MTP-leaking llama-server protected at oom_score_adj −900 would otherwise recreate June 7 with llama-server as the villain. Legit peak is bounded ~5.8–6.0 GiB (2.6 GiB model + full 32K q8_0 KV + overhead), so 6400M clears normal operation. The runtime `50-*.conf` set-property artifacts duplicate the /etc values and must be removed by surgical `sudo rm` (never `systemctl revert`).

**Tasks:**
1. [ ] Backup all current drop-ins (both /etc and /run/systemd/system.control sets, including `oom-protect.conf`) to `~/llm-server/backups/envelope-<date>/`
2. [ ] Write new `/etc/systemd/system/myscript.service.d/memory-limits.conf` via `sudo tee`: `[Service]` + `MemoryMax=6400M` only, with a comment citing Entry 026/027 rationale
3. [ ] `sudo rm` both `/run/systemd/system.control/myscript.service.d/50-MemoryHigh.conf` and `50-MemoryMax.conf`
4. [ ] `sudo systemctl daemon-reload && sudo systemctl restart myscript`
5. [ ] Verify properties and capture post-restart `bench.sh dethrottled` run (tests the U4 throttle-dip hypothesis against the 14.17 vs 15.3 tok/s gap)

**Acceptance Criteria:**
- [ ] WHEN `systemctl show myscript -p MemoryHigh -p MemoryMax -p OOMScoreAdjust` runs THEN it SHALL report `MemoryHigh=infinity`, `MemoryMax=6710886400`, `OOMScoreAdjust=-900`
- [ ] WHEN `systemctl cat myscript` runs THEN it SHALL show exactly two drop-ins (`memory-limits.conf`, `oom-protect.conf`) and no `/run/systemd/system.control` drop-ins
- [ ] WHEN the inference smoke test runs after restart THEN the server SHALL respond on port 8080 with full GPU offload (999 layers logged)
- [ ] Backup directory exists and contains all four pre-change files
- [ ] `bench.sh dethrottled` results recorded in LAB_NOTEBOOK (U4 answer: throttle-dip confirmed or refuted)

**Notes:**
`6400M` = 6710886400 bytes in systemd's MiB convention. If `bench.sh` shows gen tok/s recovering toward ≥15.0, U4 is confirmed (throttle was eating throughput) — record explicitly. Escalate if MemoryCurrent post-restart exceeds 5.9 GiB at idle (would indicate something other than KV-fill growth).

---

#### 1.2 Memory watchdog: deploy armed with guards + induced-fire verification ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- DEVIATION: trigger redesigned from MemAvailable to FILE-SWAP (Entry 028 found box idles at ~0 MemAvailable on zram; 350/700MB thresholds would restart-loop). Armed, induced-fire + guards verified, reboot-durable. -->
**Status (orig): PENDING**
**Model Tier: opus**
**Requirement Refs:** Entry 026 Rec 1(a), Entry 027 CS-A(A2)
**Depends On:** 1.1
**Files Affected:**
- `~/llm-server/memory-watchdog.sh` (create, ~120 LOC)
- `/etc/systemd/system/memory-watchdog.service` (create)
- `~/llm-server/watchdog/` (create — snapshots + `heartbeat.csv`)
- `LAB_NOTEBOOK.md` (repo — verification entry)

**Description:**
Kernel OOM is structurally blind to the NvMap consumer (June 7: ~7+ GB of 7.6 GB unaccounted; killing userspace freed nothing for 15 minutes), so userspace must defend. Deploy `memory-watchdog.service` running as **root** (no sudoers gymnastics; can read `/sys/kernel/debug/nvmap/iovmm/clients`) as a simple 30 s poll loop (`Type=simple`, `Restart=always`). Logic per poll: skip everything if `~/llm-server/MAINTENANCE` exists; read `MemAvailable` from `/proc/meminfo`; **WARN** < 700 MB → state snapshot (rate-limited 1/10 min); **CRITICAL** < 350 MB on **2 consecutive polls** AND `myscript` active > 180 s (guards the 5 GiB startup page-cache evictor in every start script, which crashes MemAvailable on each service start) AND ≥ 15 min since last watchdog restart (flap guard) → full snapshot then `systemctl restart myscript`. Hourly heartbeat row to `~/llm-server/watchdog/heartbeat.csv`: timestamp, MemAvailable, myscript MemoryCurrent, NvMap iovmm total, llama-server RSS. Snapshot = `free -h`, top-25 by RSS, `/proc/meminfo`, iovmm clients, `dmesg | tail -50`; rotate keeping last 30. Unit hardening: `OOMScoreAdjust=-1000` (must outlive the storm to act), `MemoryMax=64M`, `Nice=10`. Armed from day one (next storm window may be Sunday ~01:00); safety is the four guards, verified by induced fire.

**Tasks:**
1. [ ] Write `memory-watchdog.sh` implementing the poll loop, guards, snapshot, heartbeat, and rotation exactly as specified above
2. [ ] Write `memory-watchdog.service` (root, `OOMScoreAdjust=-1000`, `MemoryMax=64M`, `Restart=always`, `WantedBy=multi-user.target`); install via `sudo cp` + `daemon-reload` + `enable --now`
3. [ ] Confirm root readability of `/sys/kernel/debug/nvmap/iovmm/clients` from within the service context; if absent, fall back to per-process iovmm via `/sys/kernel/debug/nvmap/iovmm/` enumeration and note in script comments
4. [ ] **Induced-fire test (attended):** temporarily set CRITICAL threshold to 2000 MB via an env override, watch one clean cycle — snapshot written, `myscript` restarted, cooldown honored, no second fire — then restore 350 MB and restart watchdog
5. [ ] **Startup-transient test:** `sudo systemctl restart myscript` while watchdog armed; confirm the 5 GiB evictor transient does NOT trigger a watchdog action (service-age guard holds)
6. [ ] Verify first heartbeat rows appear; log verification evidence in LAB_NOTEBOOK entry

**Acceptance Criteria:**
- [ ] WHEN MemAvailable stays below the CRITICAL threshold for 2 consecutive polls with all guards passing THEN the watchdog SHALL write a full snapshot and restart `myscript` exactly once per 15-minute window
- [ ] WHEN `myscript` has been active < 180 s THEN the watchdog SHALL take no restart action regardless of MemAvailable
- [ ] WHEN `~/llm-server/MAINTENANCE` exists THEN the watchdog SHALL take no action and log a skipped-poll marker
- [ ] WHEN any hour boundary passes THEN the watchdog SHALL append one heartbeat row containing all five fields with non-empty values
- [ ] WHEN `systemctl show memory-watchdog -p OOMScoreAdjust` runs THEN it SHALL report `-1000`
- [ ] Induced-fire and startup-transient tests both pass with journal evidence captured
- [ ] Snapshot rotation verified (≤30 retained)

**Notes:**
Opus tier: live production box, and the failure mode of a bad guard is a restart loop. Escalate to Troy if the induced-fire test produces any second restart within the cooldown window, or if iovmm stats are unreadable even as root (would gut the forensic value — discuss before shipping a degraded version). Watchdog restart of myscript reloads whatever `mode.txt` says — correct behavior in all phases.

---

#### 1.3 CMA pre-start compaction drop-in ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- cma-compact.conf ExecStartPre defrag, failure-tolerant; verified ran as root. Batched with --mlock removal (item 1.5). See LAB_NOTEBOOK Entry 028. -->
**Status (orig): PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Rec 1(c), Entry 027 CS-A(A2), forum t/370049
**Depends On:** 1.1
**Files Affected:**
- `/etc/systemd/system/myscript.service.d/cma-compact.conf` (create)
- `~/llm-server/backups/envelope-<date>/` (update)

**Description:**
The exact-platform forum thread (t/370049, NVIDIA-confirmed unresolved in r36.5) shows CMA-pool fragmentation degrading large GPU allocations; this box has CmaTotal 256 MB / CmaFree 67 MB. Add a `cma-compact.conf` drop-in with `ExecStartPre=+/bin/sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'` — the `+` prefix runs it as root while the main process stays `User=claude`. This runs before every llama-server start (including watchdog-mediated restarts), complementing the existing python page-cache evictor in the start scripts (left untouched).

**Tasks:**
1. [ ] Write the drop-in via `sudo tee` with comment citing forum thread + Entry 027
2. [ ] `sudo systemctl daemon-reload && sudo systemctl restart myscript` (combine with 1.2's startup-transient test restart if convenient)
3. [ ] Verify via `systemctl cat myscript` and journal: ExecStartPre ran as root, no start delay > 10 s added, server healthy

**Acceptance Criteria:**
- [ ] WHEN `myscript` starts THEN the compaction ExecStartPre SHALL execute as root before the start script (journal evidence)
- [ ] WHEN the server finishes startup THEN it SHALL report full GPU offload (999 layers) and pass the smoke test
- [ ] Start-to-ready time increase ≤ 10 s vs pre-change

**Notes:**
If `drop_caches` write fails inside the systemd sandbox, check for `ProtectKernelTunables` interference (not set in the current unit, but verify). Escalate if startup time regresses materially — drop the drop-in rather than tolerate slow recovery during watchdog restarts.

---

#### 1.4 Power mode decision by measurement + new official baseline ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- A/B: MAXN_SUPER 15.3 vs 25W 14.0 tok/s -> kept MAXN (~8% > 3% tiebreak; data overrode community 25W prior). Reboot-persistent. Reboot validation PASS. See LAB_NOTEBOOK Entry 028. -->
**Status (orig): PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Rec 5, Entry 027 CS-A(A3), smolhub benchmark 2026-05-29
**Depends On:** 1.1, 1.2, 1.3
**Files Affected:**
- Device power state via `sudo nvpmodel -m <N>` (no file edits)
- `JETSON_BASELINE.md` (repo — Current Config `baseline_gen_tok_s`/`baseline_pp_tok_s` + power-mode row)
- `LAB_NOTEBOOK.md` (repo — A/B results entry)

**Description:**
Box runs MAXN_SUPER (mode 2) though `/etc/nvpmodel.conf` default is 1 (25W); the smolhub benchmark on this exact device found 25W the throughput/efficiency sweet spot (MAXN: +17% power for −3%..+8% throughput). Decide by measurement on the now-de-throttled box: check `sudo jetson_clocks --show` for pinned clocks (would distort comparison — record state), `bench.sh maxn-super`, switch `sudo nvpmodel -m 1`, settle 2 min, `bench.sh 25w`, compare medium+long gen tok/s means. Keep the winner; within ±3% → prefer 25W (cooler, −17% power). The winning mode's numbers become the new official baseline for Phase 2/3 gates. Verify mode persistence across `sudo reboot` (closes U6 persistence question; reboot is acceptable — systemd brings everything back).

**Tasks:**
1. [ ] Record `sudo jetson_clocks --show` and thermal readings before benching
2. [ ] `bench.sh maxn-super` (3×3 established method) under mode 2
3. [ ] `sudo nvpmodel -m 1`, settle ≥ 2 min, `bench.sh 25w`
4. [ ] Decide per rule above; set the winning mode; `sudo reboot`; after boot verify `nvpmodel -q` still reports the chosen mode and server auto-recovered healthy
5. [ ] Update JETSON_BASELINE.md Current Config (new `baseline_gen_tok_s`, `baseline_pp_tok_s`; add power-mode row) and write LAB_NOTEBOOK Phase 1 entry with full A/B data

**Acceptance Criteria:**
- [ ] Both modes benchmarked with the same `bench.sh` method, ≥3 runs each tier, results within-run variance < 5%
- [ ] WHEN the device reboots THEN `nvpmodel -q` SHALL report the chosen mode and `myscript` SHALL return to healthy serving without intervention
- [ ] New baseline recorded in JETSON_BASELINE.md; decision rationale + data in LAB_NOTEBOOK
- [ ] WHEN Phase 2/3 reference "baseline" THEN it SHALL mean this item's numbers

**Notes:**
Do this LAST in Phase 1 so the baseline reflects the final envelope (de-throttled + watchdog + CMA drop-in). If MAXN wins by > 3%, keep MAXN — the data overrides the community prior. Escalate if either mode shows > 15% regression vs the 15.3 historical figure (something else is wrong; stop and investigate before Phase 2).

---

### Phase 1 Testing Requirements

- [ ] Induced watchdog fire + startup-transient negative test (1.2 tasks 4–5) — journal evidence
- [ ] `bench.sh` runs: `dethrottled`, `maxn-super`, `25w` — all logged
- [ ] Inference smoke test passes after every restart
- [ ] Reboot recovery test (1.4 task 4)

### Phase 1 Completion Checklist

- [ ] All work items complete
- [ ] Watchdog armed, heartbeat accumulating, no false fires over a 24 h observation window
- [ ] LAB_NOTEBOOK Phase 1 entry written; JETSON_BASELINE updated
- [ ] All backups present in `~/llm-server/backups/envelope-<date>/`
- [ ] No regressions: gen tok/s ≥ pre-phase level

### Definition of Done (Runnable)
<!-- BEGIN DOD -->

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Cgroup props | `$SSH "systemctl show myscript -p MemoryHigh -p MemoryMax -p OOMScoreAdjust"` | `infinity` / `6710886400` / `-900` |
| Watchdog live | `$SSH "systemctl is-active memory-watchdog && sudo journalctl -u memory-watchdog -n 5 --no-pager"` | `active`, recent poll logs |
| Heartbeat | `$SSH "tail -3 ~/llm-server/watchdog/heartbeat.csv"` | ≥1 row, 5 populated fields |
| Inference | `$SSH "curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in exactly 5 words\"}],\"max_tokens\":32}'"` | Valid completion JSON |
| Benchmark | `$SSH "~/llm-server/bench.sh phase1-final"` | gen tok/s ≥ pre-phase baseline |
| Power mode | `$SSH "nvpmodel -q"` | Chosen mode, persists post-reboot |

<!-- END DOD -->

---

## Phase 2: llama.cpp Rebuild + Script Migration (CS-B)

**Estimated Complexity:** M (~8 files, ~80 LOC)
**Dependencies:** Phase 1
**Execution Mode:** Sequential

### Goals

- Rebuild llama.cpp from b8987 to the latest tag (≥ b9596) with the identical proven CMake configuration, gaining MTP support (Phase 3 prerequisite), #23907 deterministic KV-cache OOM, #24360 ssm_scan race fix, and ~6 weeks of CVE fixes for a LAN-exposed server
- Migrate all five start scripts past the b9131 CLI renames with a verification gate before first start
- Keep/rollback decided by benchmark against the Phase 1 baseline

### Work Items

#### 2.1 Pre-rebuild benchmark, backup, and maintenance window ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- b8987 backed up to backup-b8987-bin; ~40min window. See LAB_NOTEBOOK Entry 030. -->
**Status (orig): PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 027 CS-B steps 1–2, Entry 015 precedent
**Files Affected:**
- `~/llm-server/backup-b8987-bin/` (create — full copy of `llama.cpp/build/bin/`)
- `~/llm-server/MAINTENANCE` (create flag)

**Description:**
Establish the rollback point and open the window. The Phase 1 final benchmark is the comparison baseline (re-run `bench.sh pre-rebuild` only if > 3 days have passed). Copy the entire `build/bin` (binaries + `libggml*`/`libllama*`/`libmtmd*` shared libs — `LD_LIBRARY_PATH` points here). Touch `MAINTENANCE` (suspends watchdog actions), then `sudo systemctl stop myscript`. Building with the server running is impossible (~1.0 GiB available vs nvcc needs).

**Tasks:**
1. [ ] `bench.sh pre-rebuild` if Phase 1 baseline is stale (> 3 days)
2. [ ] `cp -r ~/llm-server/llama.cpp/build/bin ~/llm-server/backup-b8987-bin/`; record current commit (`git log -1`) in the notebook draft
3. [ ] `touch ~/llm-server/MAINTENANCE && sudo systemctl stop myscript`; confirm watchdog logs skip markers and takes no action

**Acceptance Criteria:**
- [ ] Backup dir contains `llama-server` + all `.so*` files; size matches source
- [ ] WHEN `myscript` is stopped with MAINTENANCE present THEN the watchdog SHALL log skips and perform zero restarts
- [ ] Free memory ≥ 6 GiB confirmed before build starts

**Notes:**
Downtime clock starts here — target ≤ 90 min through 2.4. Inference consumers (contact-center-lab embeddings) tolerate the window; schedule attended.

---

#### 2.2 Checkout and build latest tag with identical flags ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- b9652, identical flags, -j4, ~36min, BUILD_EXIT:0. Gotcha: needed CMAKE_CUDA_COMPILER/PATH for nvcc in non-login shell. See Entry 030. -->
**Status (orig): PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Check 2, Entry 027 CS-B step 3
**Depends On:** 2.1
**Files Affected:**
- `~/llm-server/llama.cpp/` (git checkout + rebuild of `build/`)

**Description:**
`git fetch --tags origin` and checkout the latest `bXXXX` release tag at execution time (≥ b9596; record exact tag). Configure with the **exact flags verified in the current CMakeCache** (2026-06-11): `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`. Build `cmake --build build --config Release -j6`; if a compile job is OOM-killed, retry `-j4` (U8). Expect 40–70 min on the 6-core Orin.

**Tasks:**
1. [ ] `git fetch --tags && git checkout <latest-bXXXX>` (clean tree first; current tree had no local changes as of 2026-06-11)
2. [ ] Fresh configure with the exact flag set above (delete `build/CMakeCache.txt` first to avoid stale-cache surprises; full `rm -rf build` acceptable — backup already exists)
3. [ ] Build `-j6`, fall back `-j4` on OOM; capture build log tail
4. [ ] Smoke: `./build/bin/llama-server --version` and `--help` execute cleanly

**Acceptance Criteria:**
- [ ] WHEN the build completes THEN `llama-server --version` SHALL report the new tag's build number
- [ ] CMakeCache reflects all six required flags
- [ ] Build log free of CUDA arch warnings for sm_87

**Notes:**
Escalate if the new tree requires a CMake/CUDA/gcc version beyond what JetPack 6.2.2 ships (would force deferring the rebuild to the JetPack 7.2 window — that's a plan-level decision, not a workaround to improvise).

---

#### 2.3 CLI flag migration gate across all five start scripts ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- U1 RESOLVED: all 18 flags present + arg-forms compatible in b9652; ZERO script edits needed. See Entry 030. -->
**Status (orig): PENDING**
**Model Tier: opus**
**Requirement Refs:** Entry 026 Check 2 (b9131 breaking change), Entry 027 U1
**Depends On:** 2.2
**Files Affected:**
- `~/llm-server/start-qwen35-server.sh` (modify)
- `~/llm-server/start-nemotron-server.sh` (modify)
- `~/llm-server/start-embedding-server.sh` (modify)
- `~/llm-server/start-server.sh` (modify — inline `llm` mode block)
- `~/llm-server/start-experiment.sh` (modify — flags only; model path is Phase 3's job)
- `~/llm-server/backups/scripts-pre-b9XXX-<date>/` (create)

**Description:**
b9131 renamed CLI args (exact renames = U1, resolved here). Before the first start of the new binary: extract every flag used across the five scripts (inventory from 2026-06-11 investigation: `--model --host --port --alias --ctx-size --n-gpu-layers --threads --parallel --flash-attn --reasoning --mlock --cache-type-k --cache-type-v --log-disable --cont-batching --n-predict` + any embedding-script-specific flags — re-verify by grep), diff each against the new `llama-server --help`, and apply renames. No script uses `LLAMA_*` env vars (verified), so the b9360 env-prefix change requires no action. Validate each script headlessly before relying on systemd.

**Tasks:**
1. [ ] Backup all five scripts to `~/llm-server/backups/scripts-pre-b9XXX-<date>/`
2. [ ] Build the actual flag inventory by grep across the five scripts; diff every flag against new `--help` output; record the rename map in the notebook draft (this resolves U1)
3. [ ] Apply renames to all five scripts; update stale header comments (build number, cmake line)
4. [ ] Headless validation: run the qwen35 script directly in a shell, confirm clean startup to "listening" log line, then kill; spot-check `--help` acceptance of every migrated flag for the other four scripts (full start test for embedding mode too — different port/flags)

**Acceptance Criteria:**
- [ ] WHEN each migrated script's flag set is checked against the new `--help` THEN every flag SHALL be recognized (zero unknown-argument errors)
- [ ] WHEN the qwen35 script runs standalone THEN llama-server SHALL reach listening state with full GPU offload and serve one completion
- [ ] WHEN the embedding script runs standalone THEN the server SHALL serve one embedding request on port 8081
- [ ] Rename map documented (U1 → Resolved)

**Notes:**
Opus tier: a silently-wrong flag (e.g., a rename that changes a default) degrades quietly. Watch specifically for semantics changes around `--flash-attn` (on/off/auto argument forms changed in this era) and KV cache type flags. Escalate if any flag has no equivalent in the new build (feature removal — needs a decision, not a guess).

---

#### 2.4 Deploy, benchmark, keep/rollback gate ✅ Completed 2026-06-15
**Status: COMPLETE 2026-06-15** <!-- KEEP: gen 15.3-15.4 tok/s (>=baseline), RSS 4899MB (<b8987), clean journal. backup-b8987-bin retained ~2wk. See Entry 030. -->
**Status (orig): PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 027 CS-B steps 7–8
**Depends On:** 2.3
**Files Affected:**
- `~/llm-server/MAINTENANCE` (remove)
- `LAB_NOTEBOOK.md` (repo — Phase 2 entry)
- `JETSON_BASELINE.md` (repo — `llamacpp_version`, `baseline_rss_mb` on keep)

**Description:**
Start the service on the new build, close the window, benchmark, decide. Keep gate: `bench.sh b9XXX` medium+long gen mean ≥ Phase 1 baseline × 0.95 AND RSS within +10% of the Phase 1 level AND clean journal (no CUDA errors) over the first 30 min. Rollback procedure (if gate fails): stop, restore `backup-b8987-bin` over `build/bin`, restore script backups, `git checkout b8987`, restart, re-verify — ~10 min.

**Tasks:**
1. [ ] `sudo systemctl start myscript`, confirm healthy, `rm ~/llm-server/MAINTENANCE`
2. [ ] `bench.sh b9XXX`; compare against Phase 1 baseline; check RSS + 30-min journal
3. [ ] KEEP: update JETSON_BASELINE (`llamacpp_version`, `llamacpp_latest_seen`, `baseline_rss_mb`) and write LAB_NOTEBOOK Phase 2 entry with full comparison. ROLLBACK: execute procedure above, document failure evidence in notebook, mark 2.4 blocked for re-plan
4. [ ] On KEEP after 48 h stable: delete `backup-b8987-bin` is NOT automatic — leave for the 2-week precedent window (note in baseline Watch Items)

**Acceptance Criteria:**
- [ ] WHEN the gate passes THEN gen tok/s SHALL be ≥ 95% of Phase 1 baseline with RSS ≤ +10%
- [ ] WHEN the gate fails THEN the rollback SHALL restore b8987 service health within ~15 min and the failure SHALL be documented with benchmark + journal evidence
- [ ] MAINTENANCE flag removed; watchdog resumed normal polling (journal evidence)
- [ ] LAB_NOTEBOOK + JETSON_BASELINE updated to match reality

**Notes:**
#23907 changes OOM behavior to fail-at-startup for quantized KV — if startup fails with a KV reservation error, that IS the feature working; check free RAM, don't misread it as a build defect.

---

### Phase 2 Testing Requirements

- [ ] Headless script validation before systemd start (2.3 task 4)
- [ ] `bench.sh b9XXX` vs Phase 1 baseline
- [ ] 30-min journal scan post-deploy (no CUDA/CUBLAS errors)
- [ ] Watchdog skip-markers during window, normal polling after

### Phase 2 Completion Checklist

- [ ] All work items complete (or rollback executed + documented)
- [ ] All 5 scripts migrated and validated; rename map recorded
- [ ] Downtime ≤ 90 min actual (record)
- [ ] LAB_NOTEBOOK + JETSON_BASELINE updated
- [ ] Rollback assets retained (`backup-b8987-bin`, script backups)

### Definition of Done (Runnable)
<!-- BEGIN DOD -->

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Version | `$SSH "~/llm-server/llama.cpp/build/bin/llama-server --version"` | New tag build number |
| Service | `$SSH "systemctl is-active myscript && cat ~/llm-server/mode.txt"` | `active`, `qwen35` |
| Benchmark | `$SSH "~/llm-server/bench.sh b9XXX-final"` | gen ≥ 0.95 × Phase 1 baseline |
| Journal | `$SSH "sudo journalctl -u myscript --since '-30 min' --no-pager \| grep -iE 'error\|cuda' \| head"` | No CUDA errors |
| All modes | Headless validation log from 2.3 task 4 | qwen35 + embedding verified |

<!-- END DOD -->

---

## Phase 3: MTP Speculative Decoding Trial (CS-C)

**Estimated Complexity:** S (~4 files, ~50 LOC)
**Dependencies:** Phase 1 (watchdog + backstop), Phase 2 (MTP-capable binary)
**Execution Mode:** Sequential

### Goals

- Trial `unsloth/Qwen3.5-4B-MTP-GGUF` (same weights + MTP head, 2.83 GB) for the claimed 1.5–2× decode gain, gated by acceptance-rate and memory-stability evidence
- Fix the broken experiment mode as a side effect (currently points at a model deleted 2026-05-13 → 5 s crash loop if selected)
- Promote to default only on ≥ +25% gen with flat RSS over a 48 h soak; otherwise revert with zero residue and documented evidence

### Work Items

#### 3.1 Pre-trial research: MTP leak workaround + observability (U2, U4-obs)
**Status: PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Check 3 caveats, Entry 027 U2/U4
**Files Affected:**
- `LAB_NOTEBOOK.md` (repo — findings appended to Phase 3 entry draft)

**Description:**
Web/GitHub research only, no device changes — can run any time after plan approval (parallel-safe with Phases 1–2). Resolve: (a) U2 — the reported MTP memory leak (xhinker Medium post 2026-05-21, paywalled) and its workaround flag: search llama.cpp issues/PRs for MTP leak reports and fix status — it may be fixed in the tag Phase 2 ships; (b) the exact spec-decoding flag syntax at the shipped tag (`--spec-type draft-mtp` per PR #22673 — verify against current docs/`--help`); (c) how to observe draft acceptance rate (server `/metrics`, log lines, or `timings` fields) for the U3 gate; (d) issue #23322 status (low MTP acceptance on SWA/hybrid models — does it implicate Qwen3.5-4B's architecture?).

**Tasks:**
1. [ ] Search ggml-org/llama.cpp issues/PRs: MTP memory leak, status at shipped tag; record findings + any workaround flag
2. [ ] Confirm exact CLI invocation for MTP single-model speculative decoding at the shipped tag
3. [ ] Identify the acceptance-rate observable and the abort threshold guidance from #23322 discussion
4. [ ] Record all findings in the notebook draft; mark U2 Resolved/Accepted

**Acceptance Criteria:**
- [ ] U2 dispositioned: leak fixed at tag / workaround flag identified / accepted-with-monitoring rationale written
- [ ] Exact MTP flag syntax + acceptance observable documented before 3.2 begins
- [ ] #23322 relevance to Qwen3.5-4B assessed

**Notes:**
If research shows the leak is unfixed AND no workaround exists AND #23322 strongly implicates Qwen3.5-class models, surface a stop/go recommendation to Troy before 3.2 rather than burning the soak window.

---

#### 3.2 Deploy MTP in experiment slot + hour-1 gates
**Status: PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Rec 4, Entry 027 CS-C steps 1–4
**Depends On:** 3.1
**Files Affected:**
- `~/llm-server/models/Qwen3.5-4B-MTP-Q4_K_M.gguf` (create — download ~2.83 GB)
- `~/llm-server/start-experiment.sh` (rewrite — fixes stale deleted-model reference)
- `~/llm-server/mode.txt` (set `experiment`)
- `~/llm-server/backups/experiment-pre-mtp-<date>/` (create)

**Description:**
Download the unsloth Q4_K_M MTP GGUF to models/. Rewrite `start-experiment.sh` mirroring `start-qwen35-server.sh` exactly (same evictor, OOM guard, flags as migrated in 2.3) plus: MTP model path, alias `qwen3.5-4b` (so `bench.sh` works unchanged), the MTP spec-decoding flag from 3.1, and any leak-workaround flag. Switch `mode.txt` → `experiment`, `kill $(pgrep -f llama-server)` (systemd restarts into experiment within 5 s). **Hour-1 gates:** server healthy with full offload; `bench.sh mtp-initial`; acceptance rate via the 3.1 observable; RSS noted. **Abort now if:** gen gain < +10% over Phase 2 result, or acceptance below the 3.1 threshold, or RSS > 6.0 GiB at idle — revert (`mode.txt` → `qwen35`, kill) and document.

**Tasks:**
1. [ ] Download GGUF (curl/hf direct to models/; verify size ~2.83 GB and SHA if published)
2. [ ] Backup old `start-experiment.sh`; rewrite per spec above
3. [ ] Switch mode, restart, confirm healthy + 999 layers + MTP active in logs
4. [ ] Run hour-1 gates; record all three measurements; proceed/abort decision logged

**Acceptance Criteria:**
- [ ] WHEN experiment mode starts THEN llama-server SHALL load the MTP GGUF with full GPU offload and MTP/speculative decoding active (log evidence)
- [ ] WHEN hour-1 gates run THEN gen tok/s, acceptance rate, and RSS SHALL all be recorded with explicit pass/abort verdicts
- [ ] WHEN any abort criterion trips THEN the revert SHALL complete within 5 min and default qwen35 mode SHALL be healthy
- [ ] `start-experiment.sh` no longer references the deleted model (landmine closed regardless of MTP outcome)

**Notes:**
KV overhead is ~10% higher with the MTP head — q8_0 32K KV may push closer to the 6400M MemoryMax; if startup KV reservation (#23907) refuses, drop ctx to 16384 for the trial and note it (promotion would then need a ctx decision). Escalate if offload drops to 0 layers (OOM guard tripped — the MTP model + evictor interaction needs a look).

---

#### 3.3 48-hour soak under watchdog telemetry
**Status: PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 027 CS-C step 5
**Depends On:** 3.2
**Files Affected:**
- `~/llm-server/watchdog/heartbeat.csv` (read-only analysis)
- `LAB_NOTEBOOK.md` (repo — soak data appended)

**Description:**
Leave experiment mode serving normal traffic for 48 h. The Phase 1 heartbeat CSV is the instrument: RSS slope, MemAvailable floor, iovmm growth. Mid-soak (~24 h) and end-of-soak checkpoints: re-run `bench.sh mtp-soak-NN`, scan watchdog journal for WARN/CRITICAL events, compute RSS slope. **Abort criteria during soak:** unexplained RSS slope > ~50 MB/day after the first-hours KV fill, any watchdog CRITICAL action attributable to llama-server growth, acceptance-rate collapse, or any inference failure. Abort = revert + document.

**Tasks:**
1. [ ] 24 h checkpoint: bench + heartbeat slope + journal scan; record
2. [ ] 48 h checkpoint: same; compute final RSS slope and tok/s consistency (3 benches within 5%)
3. [ ] Verdict: PROMOTE-eligible / ABORT with evidence

**Acceptance Criteria:**
- [ ] WHEN the soak completes THEN ≥ 48 h of heartbeat rows SHALL exist with RSS slope computed and ≤ ~50 MB/day post-fill
- [ ] WHEN any abort criterion trips mid-soak THEN the revert SHALL execute immediately and the trigger evidence SHALL be preserved (heartbeat + snapshot + journal)
- [ ] Zero watchdog CRITICAL actions attributable to the MTP server during soak

**Notes:**
A Sunday ~01:00 window inside the soak is a feature, not a risk — if the NvMap storm recurs, the watchdog + forensics work exactly as designed and the soak evidence doubles as Phase 1 validation. Distinguish external-storm events from llama-server growth before attributing.

---

#### 3.4 Promote/revert decision + execution
**Status: PENDING**
**Model Tier: opus**
**Requirement Refs:** Entry 027 CS-C step 6
**Depends On:** 3.3
**Files Affected:**
- `~/llm-server/start-qwen35-server.sh` (modify on promote — model path + MTP flags)
- `~/llm-server/mode.txt` (set `qwen35` either way)
- `LAB_NOTEBOOK.md`, `JETSON_BASELINE.md` (repo — verdict, new baselines on promote)

**Description:**
Promote gate (all required): gen ≥ +25% over the Phase 2 post-rebuild result, flat RSS per 3.3, zero attributable watchdog events, acceptance stable. Promote = backup then edit `start-qwen35-server.sh` (MTP model path, spec flags; alias unchanged), `mode.txt` → `qwen35`, restart, `bench.sh mtp-promoted`, update JETSON_BASELINE Current Config (model, gen/pp/RSS baselines) + Watch Items. Revert = `mode.txt` → `qwen35`, restart, keep findings; the MTP GGUF stays on disk (2.83 GB, 664 G free) with a Watch Item to retry when upstream fixes land. Either way the decision is written up with the full evidence chain.

**Tasks:**
1. [ ] Apply the gate to the 3.2/3.3 evidence; write the verdict + reasoning
2. [ ] Execute promote or revert path per spec; verify default mode healthy post-change
3. [ ] Update LAB_NOTEBOOK (full Phase 3 entry) + JETSON_BASELINE (promote: Current Config; revert: Watch Item)

**Acceptance Criteria:**
- [ ] WHEN the gate is applied THEN every criterion SHALL have an explicit measured value beside its threshold
- [ ] WHEN promotion executes THEN default qwen35 mode SHALL serve the MTP model at the soak-verified throughput (`bench.sh mtp-promoted` within 5% of soak numbers)
- [ ] WHEN reversion executes THEN default mode SHALL match its Phase 2 benchmark within 5%
- [ ] Docs updated to match the deployed reality

**Notes:**
Opus tier: the gate is numeric but evidence attribution (RSS slope vs KV fill, storm vs leak) is judgment. Borderline result (+15–25%) = revert and flag for Troy — promotion bias is the failure mode to avoid on a production box.

---

### Phase 3 Testing Requirements

- [ ] Hour-1 gate measurements (tok/s, acceptance, RSS)
- [ ] 24 h + 48 h soak checkpoints with slope analysis
- [ ] Post-decision benchmark matching expected state within 5%

### Phase 3 Completion Checklist

- [ ] All work items complete; verdict documented with evidence chain
- [ ] `start-experiment.sh` landmine permanently fixed
- [ ] Default mode healthy at expected throughput
- [ ] LAB_NOTEBOOK + JETSON_BASELINE consistent with deployed state

### Definition of Done (Runnable)
<!-- BEGIN DOD -->

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Mode/health | `$SSH "cat ~/llm-server/mode.txt && systemctl is-active myscript"` | `qwen35`, `active` |
| Benchmark | `$SSH "~/llm-server/bench.sh phase3-final"` | Matches verdict-expected tok/s ±5% |
| Soak data | `$SSH "wc -l ~/llm-server/watchdog/heartbeat.csv"` | ≥ 48 hourly rows since 3.2 |
| Watchdog events | `$SSH "sudo journalctl -u memory-watchdog --since '-3 days' --no-pager \| grep -ic critical"` | 0 attributable to MTP server |

<!-- END DOD -->

---

## Phase 4: Fine-Tune Trials (CS-D — Optional, Time-Boxed)

**Estimated Complexity:** S (~3 files, ~20 LOC)
**Dependencies:** Phase 3 (experiment slot free; verdict settled)
**Execution Mode:** Sequential

### Goals

- Evaluate two zero-memory-cost Qwen3.5-4B reasoning fine-tunes in the experiment slot with a quality gate that prevents a repeat of the v2-distill reasoning-loop promotion hazard
- Default outcome is "keep as alternates" — promotion requires clear, measured superiority

### Work Items

#### 4.1 Download candidates + define quality probe set
**Status: PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Rec 6, Entry 027 CS-D
**Files Affected:**
- `~/llm-server/models/` (two GGUFs, ~2.7 GB each — downloads may overlap Phase 3 soak)
- `~/llm-server/probe-prompts.json` (create — ~10 fixed prompts)
- `LAB_NOTEBOOK.md` (repo — probe set recorded)

**Description:**
Download `Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF` and `Jackrong/Qwen3.5-4B-Neo-GGUF` (Q4_K_M each). Define a fixed ~10-prompt probe set reusing Entry 021's A/B methodology: multi-step reasoning, summarization, instruction-following, tool-call JSON formatting, factual recall, and a reasoning-loop tripwire (the failure mode that killed v2-distill) — each with stated pass criteria. Probe set is frozen before any model is tested.

**Tasks:**
1. [ ] Download both GGUFs; verify sizes
2. [ ] Write `probe-prompts.json` (prompt, category, pass criteria per item); record in notebook
3. [ ] Run the probe set against the current default model first — this is the comparison anchor

**Acceptance Criteria:**
- [ ] Both models on disk, sizes verified
- [ ] Probe set frozen + anchor results recorded BEFORE any candidate runs
- [ ] Reasoning-loop tripwire prompt included

**Notes:**
Downloads can run during the Phase 3 soak (network-only). Everything else waits for the slot.

---

#### 4.2 Trial both candidates sequentially + disposition
**Status: PENDING**
**Model Tier: sonnet**
**Requirement Refs:** Entry 026 Rec 6, Entry 027 CS-D
**Depends On:** 4.1
**Files Affected:**
- `~/llm-server/start-experiment.sh` (model path swaps — one evening per candidate)
- `~/llm-server/mode.txt` (experiment ↔ qwen35)
- `LAB_NOTEBOOK.md`, `JETSON_BASELINE.md` (repo — results + watch-item disposition)

**Description:**
For each candidate (one evening each, time-boxed): point `start-experiment.sh` at it, switch to experiment mode, `bench.sh <candidate>` (throughput sanity — should match base within noise; same architecture), run the frozen probe set, score against the anchor, revert to qwen35. Disposition per candidate: PROMOTE (clearly better on probes, no pathologies — requires Troy sign-off given the v2 precedent), KEEP (on disk as alternate), or DELETE (pathology found). Update baseline Watch Items accordingly.

**Tasks:**
1. [ ] Candidate 1 (Claude-4.6-Opus distill): trial per spec; score; revert; record
2. [ ] Candidate 2 (Neo): same
3. [ ] Disposition both; update notebook + baseline; if either earns PROMOTE, present evidence to Troy rather than self-promoting

**Acceptance Criteria:**
- [ ] WHEN each trial ends THEN default qwen35 mode SHALL be restored and healthy the same evening
- [ ] Both candidates scored against the anchor on the frozen probe set with per-prompt results recorded
- [ ] WHEN the reasoning-loop tripwire fires for a candidate THEN that candidate SHALL be marked DELETE with the transcript preserved
- [ ] No candidate promoted without explicit Troy approval

**Notes:**
Throughput is not the question here (same architecture) — quality is. Keep evenings short; this is the lowest-priority phase and abandoning it costs nothing.

---

### Phase 4 Testing Requirements

- [ ] Anchor probe run before candidates
- [ ] Per-candidate bench + probe scores recorded
- [ ] Default-mode health verified after each trial evening

### Phase 4 Completion Checklist

- [ ] Both candidates dispositioned (or phase explicitly skipped — note in notebook)
- [ ] Default mode healthy; experiment script left pointing at a model that exists
- [ ] LAB_NOTEBOOK + JETSON_BASELINE updated

### Definition of Done (Runnable)
<!-- BEGIN DOD -->

| Check | Command | Pass Criteria |
|-------|---------|---------------|
| Mode/health | `$SSH "cat ~/llm-server/mode.txt && systemctl is-active myscript"` | `qwen35`, `active` |
| Experiment sanity | `$SSH "grep MODEL= ~/llm-server/start-experiment.sh && ls -lh \$($SSH grep -oP '(?<=MODEL=\")[^\"]+' ~/llm-server/start-experiment.sh 2>/dev/null) 2>/dev/null \|\| true"` | Referenced model file exists |
| Disposition | Probe-set scores in LAB_NOTEBOOK Phase 4 entry | Both candidates dispositioned |

<!-- END DOD -->

<!-- END PHASES -->

---

<!-- BEGIN TABLES -->

## Parallel Work Opportunities

| Work Item | Can Run With | Notes |
|-----------|--------------|-------|
| 3.1 (research) | Phase 1, Phase 2 | Web/GitHub only — touches nothing on-device |
| 4.1 downloads | 3.3 (soak) | Network-only; the trial itself waits for the slot |
| All other items | — | Single device, single server port: strictly sequential on-device |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy | Status |
|------|------------|--------|---------------------|--------|
| Watchdog false-positive restart loop | Low | Med | Four guards (2-poll breach, 180 s service-age, 15-min cooldown, MAINTENANCE flag) + induced-fire and startup-transient tests before unattended operation; worst case 4 restarts/hr, journal-visible | Open |
| NvMap storm recurs before Phase 1 completes | Med | Med | Existing −900 protection limits blast radius; prioritize Phase 1 to the next evening; storm during soak is handled by design | Open |
| Rebuild regression or build failure | Med | Med | Full binary backup + script backups; 10–15 min rollback rehearsed in 2.4; benchmark gate at 95% | Open |
| Silent flag-semantics change in migration | Low | High | Opus-tier 2.3 with per-flag `--help` diff + headless validation of two modes before systemd start | Open |
| MTP memory leak on 8 GB box | Med | Med | Triple containment: 3.1 research first, watchdog (Phase 1), MemoryMax 6400M backstop; hour-1 + soak abort gates | Open |
| MTP acceptance too low for net gain | Med | Low | Hour-1 abort gate — cost of failure is one evening + one download | Open |
| Fine-tune promotion regression | Low | Med | Frozen probe set + reasoning-loop tripwire + mandatory Troy sign-off for promotion | Open |
| Maintenance window overruns (build > 90 min) | Med | Low | Attended window; server restorable to b8987 at any point; -j4 fallback | Open |

---

## Unknowns Register

| ID | Unknown | Severity | Affects | Resolution Strategy | Status |
|----|---------|----------|---------|---------------------|--------|
| U1 | Exact b9131 CLI renames across our ~16 flags | Medium | Phase 2, Item 2.3 | Per-flag diff against new `--help` before first start | Open |
| U2 | MTP memory-leak status + workaround flag at shipped tag | Medium | Phase 3, Items 3.1–3.2 | GitHub issue/PR search (3.1) before deployment | Open |
| U3 | MTP draft acceptance rate on Qwen3.5-4B (#23322 risk) | High | Phase 3 value; Item 3.2 | Hour-1 measurement with abort gate; #23322 relevance assessed in 3.1 | Open |
| U4 | Is the 14.17 vs 15.3 tok/s dip MemoryHigh reclaim throttle? | Medium | Phase 1, Item 1.1 | `bench.sh dethrottled` immediately after throttle removal | Open |
| U5 | Does cgroup MemoryCurrent capture NvMap-backed allocations (MemoryMax backstop efficacy)? | Medium | Phase 1, Item 1.2; Phase 3 containment | Compare MemoryCurrent vs RSS vs iovmm in heartbeat data during Phase 1 verification | Open |
| U6 | jetson_clocks pinned state + nvpmodel reboot persistence | Low | Phase 1, Item 1.4 | `jetson_clocks --show` pre-bench; reboot persistence test in 1.4 | Open |
| U7 | Build-time memory headroom at -j6 on new tree | Low | Phase 2, Item 2.2 | Monitor first build; -j4 fallback | Open |

---

## Success Metrics

- [ ] All phases completed (Phase 4 optionally skipped with explicit note)
- [ ] **Resilience:** next NvMap memory event ends in a ≤ 60 s watchdog-mediated recovery with a complete forensic snapshot — not a 15-minute kernel storm with a journal gap
- [ ] **Telemetry:** continuous hourly heartbeat history exists across all memory dimensions kernel OOM can't see
- [ ] **Throughput:** post-rebuild ≥ 95% of Phase 1 baseline; MTP verdict delivered with measured evidence (target ≥ +25% if promoted)
- [ ] **Currency:** llama.cpp within ~1 week of upstream; all five scripts migrated (JetPack 7.2 upgrade inherits this work)
- [ ] **Reversibility:** every change has a tested backup/rollback path; zero unrecoverable modifications
- [ ] **Hygiene:** experiment mode no longer references a deleted model; docs (LAB_NOTEBOOK, JETSON_BASELINE) match deployed reality at every phase boundary

---

## Appendix: Requirement Traceability

| Requirement | Source | Phase | Work Item |
|----------------|--------|-------|-----------|
| Userspace watchdog for NvMap-blind OOM | Entry 026 Rec 1(a) | 1 | 1.2 |
| CMA compaction pre-load workaround | Entry 026 Rec 1(c), forum t/370049 | 1 | 1.3 |
| Remove/relax Entry 023 cgroup limits | Entry 026 Rec 2 | 1 | 1.1 |
| Power mode check (25W sweet spot) | Entry 026 Rec 5 | 1 | 1.4 |
| Rebuild llama.cpp (MTP prereq, #23907, #24360, CVE posture) | Entry 026 Check 2/3, Entry 027 CS-B | 2 | 2.1–2.4 |
| b9131 CLI rename migration | Entry 026 Check 2 | 2 | 2.3 |
| MTP trial with leak + acceptance gates | Entry 026 Rec 4, Check 3 | 3 | 3.1–3.4 |
| Fix stale start-experiment.sh | Entry 027 investigation surprise 3 | 3 | 3.2 |
| Jackrong fine-tune trials | Entry 026 Rec 6 | 4 | 4.1–4.2 |
| VMM patch watch (no local apply) | Entry 026 Rec 1(b) | — | Out of scope: baseline recon trigger |
| JetPack 7.2 evaluation | Entry 026 Rec 3 | — | Out of scope: separate decision ~2026-06-25+ |

<!-- END TABLES -->

---

*Implementation plan generated by Claude on 2026-06-11 16:38:06*
*Source: /create-plan command*
