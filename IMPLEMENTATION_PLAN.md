# llama.cpp b8766 → b8987 Upgrade Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade llama.cpp from b8766 to b8987 on the Jetson Orin Nano Super, closing CVE-2026-21869 and picking up CUDA memory management improvements.

**Architecture:** SSH-based remote upgrade. Back up current binary, rebuild from source on-device with identical cmake flags, handle the b8829 library rename (libcommon → libllama-common), restart via systemd, benchmark against established 14.0 tok/s baseline, update project docs. Rollback is instant via backup binary copy.

**Tech Stack:** llama.cpp (CUDA SM87), systemd, bash, SSH

---

## File Structure

**Remote (Jetson via SSH):**

| Path | Action | Purpose |
|------|--------|---------|
| `~/llm-server/backup-b8766/` | Create | Rollback binary + shared libs |
| `~/llm-server/backup-b8414/` | Remove | Stale backup from prior build (frees ~571 MB) |
| `~/llm-server/llama.cpp/` | Git checkout b8987, rebuild | Source + build output |
| `~/llm-server/llama.cpp/build/` | Clean + rebuild | New binary + renamed shared libs |

**Local (this repo):**

| Path | Action | Purpose |
|------|--------|---------|
| `JETSON_CONFIG.md` | Modify | Update llama.cpp commit, build number, library names |
| `JETSON_BASELINE.md` | Modify | Update `llamacpp_version`, `Last updated` |
| `LAB_NOTEBOOK.md` | Append | Rebuild entry with benchmark data |

---

## SSH Prefix

All remote commands use:

```bash
SSH="ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net"
```

---

### Task 1: Back Up Current Build

**Files:**
- Create: `jetson:~/llm-server/backup-b8766/`
- Remove: `jetson:~/llm-server/backup-b8414/`

- [ ] **Step 1: Create backup directory and copy current binary + shared libs**

```bash
$SSH "mkdir -p ~/llm-server/backup-b8766 && cp ~/llm-server/llama.cpp/build/bin/llama-server ~/llm-server/llama.cpp/build/bin/lib*.so* ~/llm-server/backup-b8766/"
```

- [ ] **Step 2: Verify backup contents**

```bash
$SSH "ls -lh ~/llm-server/backup-b8766/"
```

Expected: `llama-server` (~12 MB), `libllama.so`, `libggml*.so`, `libcommon.so`, `libmtmd.so` — all with today's timestamps or b8766 build date.

- [ ] **Step 3: Remove stale b8414 backup**

```bash
$SSH "rm -rf ~/llm-server/backup-b8414/"
```

- [ ] **Step 4: Verify removal**

```bash
$SSH "ls ~/llm-server/backup-b8414/ 2>&1"
```

Expected: `No such file or directory`

**Rollback from any later point:**

```bash
$SSH "cp ~/llm-server/backup-b8766/* ~/llm-server/llama.cpp/build/bin/ && kill \$(pgrep -f llama-server)"
```

Restores b8766 in <30 seconds. Systemd auto-restarts with old binary.

---

### Task 2: Fetch and Checkout b8987

**Files:**
- Modify: `jetson:~/llm-server/llama.cpp/` (git state)

- [ ] **Step 1: Fetch upstream tags**

```bash
$SSH "cd ~/llm-server/llama.cpp && git fetch origin --tags"
```

- [ ] **Step 2: Checkout b8987**

```bash
$SSH "cd ~/llm-server/llama.cpp && git checkout b8987"
```

Expected: `HEAD is now at <hash> ...` with detached HEAD.

- [ ] **Step 3: Verify HEAD**

```bash
$SSH "cd ~/llm-server/llama.cpp && git log --oneline -1"
```

Expected: commit hash followed by b8987 content. Note the commit hash — needed for JETSON_CONFIG.md later.

---

### Task 3: Clean Build Directory and Rebuild

**Files:**
- Modify: `jetson:~/llm-server/llama.cpp/build/` (full rebuild)

The library rename (libcommon → libllama-common in b8829) means old cmake cache references stale targets. A clean build avoids linker confusion between old and new .so names.

- [ ] **Step 1: Clean the build directory**

```bash
$SSH "cd ~/llm-server/llama.cpp && rm -rf build"
```

- [ ] **Step 2: Configure cmake**

```bash
$SSH "cd ~/llm-server/llama.cpp && PATH=/usr/local/cuda/bin:\$PATH cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release"
```

Check output for:
- `CUDA: 12.6` detected
- `CUDA architectures: 87`
- No errors about missing dependencies

If cmake fails: `git checkout b8766 && cmake -B build ...` to restore, then investigate.

- [ ] **Step 3: Build (20-30 minutes on ARM)**

```bash
$SSH "cd ~/llm-server/llama.cpp && PATH=/usr/local/cuda/bin:\$PATH cmake --build build --config Release -j4"
```

Uses `-j4` (not `-j6`) to avoid OOM during CUDA kernel compilation. The currently running server is unaffected — Linux inode semantics keep the old binary memory-mapped even as cmake writes new files.

Expected: completes with exit code 0, no errors.

- [ ] **Step 4: Verify build output — binary and renamed libraries**

```bash
$SSH "ls -lh ~/llm-server/llama.cpp/build/bin/llama-server ~/llm-server/llama.cpp/build/bin/lib*.so*"
```

Expected:
- `llama-server` with fresh timestamp
- `libllama.so` (core inference library)
- `libllama-common.so` (was `libcommon.so` pre-b8829)
- `libggml.so`, `libggml-base.so`, `libggml-cuda.so`, `libggml-cpu.so`
- Possibly `libmtmd.so` (multimodal)

Confirm `libcommon.so` is NOT present (renamed). Confirm `libllama-common.so` IS present.

- [ ] **Step 5: Verify the systemd unit's LD_LIBRARY_PATH still covers the build output directory**

```bash
$SSH "grep LD_LIBRARY_PATH /etc/systemd/system/myscript.service"
```

Expected: `Environment=LD_LIBRARY_PATH=/home/claude/llm-server/llama.cpp/build/bin:/usr/local/cuda/lib64`

This points to the directory, not individual files, so the renamed libraries are found automatically. No systemd unit change needed.

---

### Task 4: Deploy and Verify

**Files:**
- Runtime: `jetson:~/llm-server/llama.cpp/build/bin/llama-server` (new binary loaded by systemd)

- [ ] **Step 1: Kill the running server to trigger systemd restart**

```bash
$SSH "kill \$(pgrep -f llama-server)"
```

Systemd restarts in 5 seconds. The startup script runs the memory eviction trick, reclaiming filesystem cache for CUDA allocation.

- [ ] **Step 2: Wait 15 seconds, then verify service is running**

```bash
$SSH "sleep 15 && systemctl status myscript"
```

Expected: `active (running)`, new PID, new start timestamp. If `activating` or crash-looping, check journal: `$SSH "sudo journalctl -u myscript --since '1 min ago' --no-pager | tail -30"`.

- [ ] **Step 3: Verify new build version via inference**

```bash
$SSH "curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello\"}],\"max_tokens\":8}' | python3 -c \"import sys,json; d=json.load(sys.stdin); print('fingerprint:', d.get('system_fingerprint','MISSING'))\""
```

Expected: `system_fingerprint` contains `b8987` and the new commit hash. If it still shows `b8766`, the old binary is still running — check `LD_LIBRARY_PATH` and verify the build output path.

- [ ] **Step 4: Verify GPU offload is active**

```bash
$SSH "ps -o pid,rss,comm -p \$(pgrep llama-server)"
```

Expected: RSS ~4,500-5,000 MB (GPU offload active). If RSS ~2,000 MB, the model is running CPU-only — check for CUDA errors in journal.

- [ ] **Step 5: Check memory state**

```bash
$SSH "free -h"
```

Expected: Available > 500 MB, swap (SSD) at 0 B.

- [ ] **Step 6: Check for CUDA errors in journal**

```bash
$SSH "sudo journalctl -u myscript --since '2 min ago' --no-pager | grep -iE '(error|fail|cuda|oom|segv)'"
```

Expected: no output (no errors). `NvMapMemAllocInternalTagged` warnings are non-fatal and expected — only a problem if followed by SEGV or OOM kill.

---

### Task 5: Benchmark Against Baseline

**Files:**
- Runtime: bench.sh on Jetson

- [ ] **Step 1: Run the benchmark suite**

```bash
$SSH "bash ~/llm-server/bench.sh post-b8987"
```

Takes ~3-5 minutes. Runs warmup, then 3 iterations each of short (32 tok), medium (256 tok), long (512 tok) generation tests.

- [ ] **Step 2: Evaluate results against baseline**

| Metric | Baseline (b8766) | Target | Action if below |
|--------|-------------------|--------|-----------------|
| Gen tok/s (long) | 14.0-14.4 | ≥ 13.0 | Investigate; rollback if < 11.9 (>15% regression) |
| PP tok/s | 156-166 | ≥ 130 | Check flash attention and GPU offload |
| RSS | 4,631 MB | 4,200-5,100 MB | Flag if > 5,600 MB (+20% above baseline) |

If gen tok/s ≥ 13.0: PASS — proceed to Task 6.
If gen tok/s < 11.9: FAIL — rollback to backup, investigate.
If gen tok/s 11.9-13.0: MARGINAL — note in lab notebook, investigate before promoting.

- [ ] **Step 3: Record the benchmark output**

Copy the full bench.sh output — it goes into the LAB_NOTEBOOK entry in Task 6.

---

### Task 6: Update Project Documentation

**Files:**
- Modify: `JETSON_CONFIG.md`
- Modify: `JETSON_BASELINE.md`
- Append: `LAB_NOTEBOOK.md`

- [ ] **Step 1: Update JETSON_CONFIG.md**

Change the llama.cpp build section:

```markdown
| Commit | `{NEW_HASH}` (build b8987) — detached HEAD |
```

Update the previous build reference:

```markdown
| Previous | `547765a93` (build b8766) — backup at `~/llm-server/backup-b8766/` |
```

In the Build Commands section, verify the cmake flags are unchanged (they should be identical).

In the systemd service section, note library names if any .so names changed beyond the libcommon rename.

Update `Last updated:` date at top of file.

- [ ] **Step 2: Update JETSON_BASELINE.md**

Change:

```markdown
| llamacpp_version | b8987 |
```

Update `Last updated:` date. Do NOT change `baseline_gen_tok_s` or `baseline_rss_mb` unless benchmark shows a sustained change — those get updated after the next audit confirms stability.

Remove the backup-b8414 watch item (already deleted). Update the backup watch item to reference b8766.

- [ ] **Step 3: Append LAB_NOTEBOOK.md entry**

Auto-increment entry number (currently 019). Entry should include:

```markdown
## Entry 020: llama.cpp Rebuild b8766 → b8987 (2026-MM-DD)
**Date:** 2026-MM-DD HH:MM UTC
**Operator:** Claude Code
**Status:** REBUILD — system modified

### Motivation
- CVE-2026-21869 (CVSS 8.8, heap-buffer-overflow in reasoning sampler)
- CUDA legacy pool OOM flush-retry (b8863)
- CUDA graphs LRU eviction (b8832)
- Qwen3 duplicate scale fix (b8946)
- 221-build gap from b8766

### Build
- Source: b8987 (commit {HASH})
- cmake flags: unchanged from b8766
- Build time: {X} minutes
- Library rename: libcommon.so → libllama-common.so (b8829) — no systemd changes needed

### Benchmark (post-b8987)
{Paste bench.sh output here}

### Comparison to Baseline
| Metric | Baseline (b8766) | Post-b8987 | Delta |
|--------|-------------------|------------|-------|
| Gen tok/s (long) | 14.0 | {X} | {+/-X%} |
| PP tok/s | 166 | {X} | {+/-X%} |
| RSS (MB) | 4,631 | {X} | {+/-X%} |
| Available RAM | 2,783 MB | {X} | |

### Result
{PASS/MARGINAL/FAIL} — {summary}
```

- [ ] **Step 4: Commit documentation updates**

```bash
git add JETSON_CONFIG.md JETSON_BASELINE.md LAB_NOTEBOOK.md
git commit -m "Update docs for llama.cpp b8766 → b8987 rebuild"
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Build failure (cmake or nvcc error) | Low | None — old binary still running | Rollback: `git checkout b8766 && cmake -B build ...` |
| Shared lib ABI mismatch (crash-loop) | Low | Service down until fixed | Rollback: `cp ~/llm-server/backup-b8766/* ~/llm-server/llama.cpp/build/bin/ && kill $(pgrep -f llama-server)` |
| Performance regression | Low | Slower inference | bench.sh catches it; rollback if >15% regression |
| cudaMalloc failure after restart | Low | Falls back to CPU-only (~9 tok/s) | Memory eviction runs automatically; full reboot as escalation |
| Library rename breaks startup | Very Low | Service won't start | LD_LIBRARY_PATH points to directory, not files — should resolve automatically |

**Critical build flags — do NOT change:**
- `GGML_CUDA_NO_VMM` must stay OFF (default) — Jetson unified memory requires VMM
- `CMAKE_CUDA_ARCHITECTURES=87` — Orin's exact SM
- `-j4` not `-j6` — avoids OOM during CUDA kernel compilation

---

## Success Criteria

- [ ] Service running with `system_fingerprint` containing `b8987`
- [ ] Gen tok/s ≥ 13.0 on bench.sh (within 10% of 14.0 baseline)
- [ ] Available RAM > 500 MB after restart
- [ ] No CUDA errors or crash-loops in journal
- [ ] CVE-2026-21869 closed (build post-b8908)
- [ ] Documentation updated (JETSON_CONFIG.md, JETSON_BASELINE.md, LAB_NOTEBOOK.md)
- [ ] Rollback binary verified accessible at `~/llm-server/backup-b8766/`
