# Jetson Orin Nano Super — Full Configuration Reference

**Last updated:** 2026-03-30
**Hostname:** `jetson.k4jda.net` (Tailscale: `100.106.252.90`)
**Service user:** `claude` (LLM server and all managed services)
**Admin user:** `davistroy` (interactive login, original setup)

This document describes the complete configuration of the Jetson device in enough detail to recreate it from a fresh JetPack install.

---

## Hardware

| Component | Detail |
|-----------|--------|
| **Device** | NVIDIA Jetson Orin Nano Super Engineering Reference Developer Kit |
| **SoC** | NVIDIA Orin (Cortex-A78AE + Ampere GPU, compute capability 8.7) |
| **CPU** | 6-core ARM Cortex-A78AE @ 62.50 BogoMIPS |
| **GPU** | Ampere GA10B, sm_87, 1024 CUDA cores |
| **RAM** | 7.4 GB LPDDR5 unified memory (CPU and GPU share) |
| **Storage** | 824 GB NVMe SSD (boot drive, migrated from eMMC) |
| **Swap** | 16 GB file-based (`/ssd/16GB.swap`) |
| **WiFi** | Realtek adapter (`wlP1p1s0`) — `192.168.10.59/24` |
| **Power Mode** | MAXN_SUPER |

---

## OS / Firmware

| Component | Version |
|-----------|---------|
| **JetPack** | 6.2.2 (R36.5.0) |
| **L4T** | R36, REVISION 5.0 |
| **Ubuntu** | 22.04.5 LTS (Jammy Jellyfish) |
| **Kernel** | 5.15.185-tegra |
| **NVIDIA Driver** | 540.4.0 (Open Kernel Module) |
| **CUDA** | 12.6 (V12.6.68, `/usr/local/cuda-12.6`) |
| **cuDNN** | 9.3.0.75 |
| **TensorRT** | 10.3.0.30 |
| **VPI** | 3.2.4 |
| **GCC** | 11.4.0 |
| **Python** | 3.10.12 |

### Environment Variables (`.bashrc`)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
```

---

## Network Access

```bash
# Claude Code (automated operations)
ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net

# Interactive login
ssh davistroy@jetson.k4jda.net

# By Tailscale IP
ssh -i ~/.ssh/id_claude_code claude@100.106.252.90
```

The device is always-on, accessible via Tailscale mesh network.

| Interface | Address | Notes |
|-----------|---------|-------|
| WiFi (`wlP1p1s0`) | `192.168.10.59/24` | Primary LAN |
| Tailscale (`tailscale0`) | `100.106.252.90/32` | VPN mesh |

---

## llama.cpp Build

| Setting | Value |
|---------|-------|
| Source | https://github.com/ggerganov/llama.cpp |
| Commit | `5744d7ec4` (build 8414) — detached HEAD |
| Path | `~/llm-server/llama.cpp/` |
| Binary | `~/llm-server/llama.cpp/build/bin/llama-server` |

### Build Commands

```bash
cd ~/llm-server/llama.cpp
git checkout 5744d7ec4

PATH=/usr/local/cuda/bin:$PATH cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DGGML_CUDA_F16=ON \
    -DCMAKE_BUILD_TYPE=Release

PATH=/usr/local/cuda/bin:$PATH cmake --build build --config Release -j4
```

Build takes ~20-30 minutes on the Jetson (CUDA kernel compilation is slow on ARM).

### Critical Build Notes

| Flag | Value | Why |
|------|-------|-----|
| `CMAKE_CUDA_ARCHITECTURES` | **87** | Orin's exact compute capability. Other archs waste compile time; GPU falls back to PTX JIT. |
| `GGML_CUDA_NO_VMM` | **OFF (default)** | Must stay OFF. Jetson unified memory requires CUDA VMM. Setting ON causes cudaMalloc failures. |
| `GGML_CUDA_F16` | **ON** | Enables native FP16 arithmetic, reduces memory bandwidth pressure. |
| `PATH` | Must include `/usr/local/cuda/bin` | nvcc not in default PATH on Jetson. |

---

## Models

All models stored in `/home/claude/llm-server/models/` (~34 GB total):

| Model | File | Size | Quant | Use |
|-------|------|------|-------|-----|
| **Qwen3.5-4B** (active) | `Qwen_Qwen3.5-4B-Q4_K_M.gguf` | 2.6 GB | Q4_K_M | Chat (default mode, optimized 2026-03-30) |
| Qwen3.5-4B | `Qwen_Qwen3.5-4B-Q5_K_M.gguf` | 3.1 GB | Q5_K_M | Chat (previous default, 12% slower) |
| Nemotron-3-Nano-4B | `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | 2.7 GB | Q4_K_M | Chat (reasoning on/off, tool calling) |
| Qwen3-Embedding-4B | `Qwen3-Embedding-4B-Q4_K_M.gguf` | 2.4 GB | Q4_K_M | Embeddings (contact-center-lab) |
| Qwen3-Embedding-4B | `Qwen3-Embedding-4B-Q6_K.gguf` | 3.1 GB | Q6_K | Embeddings (higher quality) |
| Qwen3-Embedding-0.6B | `Qwen3-Embedding-0.6B-Q8_0.gguf` | 610 MB | Q8_0 | Embeddings (lightweight) |
| Qwen3-4B | `Qwen_Qwen3-4B-Q5_K_M.gguf` | 2.7 GB | Q5_K_M | Chat |
| Qwen2.5-3B-Instruct | `qwen2.5-3b-instruct-q4_k_m.gguf` | 2.0 GB | Q4_K_M | Chat (legacy mode) |
| Qwen2.5-7B-Instruct | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | 4.4 GB | Q4_K_M | Chat (tight fit for GPU) |
| Qwen2.5-7B-Instruct | `Qwen2.5-7B-Instruct-Q3_K_S.gguf` | 3.3 GB | Q3_K_S | Chat (smaller 7B quant) |
| DeepSeek-R1-Distill-7B | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | 4.4 GB | Q4_K_M | Reasoning |
| Qwen2.5-Coder-3B | `Qwen2.5-Coder-3B-Q6_K_L.gguf` | 2.5 GB | Q6_K_L | Code generation |

### Model Sweet Spot

Q4_K_M quants of 4B-parameter models are the sweet spot for this device (see LAB_NOTEBOOK.md for full benchmarks). Q4_K_M outperforms Q5_K_M by 12% on throughput with no measurable quality loss, because the Jetson is memory-bandwidth-bound — smaller weights mean less data to move per inference step. The full 32K context window fits comfortably with ~2 GB of headroom remaining. 7B models work but leave little room and may force partial GPU offload or reduced context.

---

## Service Architecture

### systemd Service

```ini
# /etc/systemd/system/myscript.service
[Unit]
Description=llmserver
After=network.target

[Service]
ExecStart=/home/claude/llm-server/start-server.sh
Restart=always
RestartSec=5
User=claude
SupplementaryGroups=render
WorkingDirectory=/home/claude/llm-server/
Environment=LD_LIBRARY_PATH=/home/claude/llm-server/llama.cpp/build/bin:/usr/local/cuda/lib64
Environment=PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=CUDA_HOME=/usr/local/cuda

[Install]
WantedBy=multi-user.target
```

Auto-starts on boot, auto-restarts on crash (5-second delay). The `SupplementaryGroups=render` is required for CUDA access to `/dev/dri/renderD128`. The `LD_LIBRARY_PATH` is required because llama.cpp shared libraries (`libllama.so`, `libggml*.so`, `libmtmd.so`) are in the build output directory alongside the binary.

### Mode Switching

The entry point `start-server.sh` reads `/home/claude/llm-server/mode.txt` and dispatches to the appropriate startup script:

| Mode | Model | Port | Script |
|------|-------|------|--------|
| **`qwen35`** (default) | Qwen3.5-4B | 8080 | `start-qwen35-server.sh` |
| `nemotron` | Nemotron-3-Nano-4B | 8080 | `start-nemotron-server.sh` |
| `embedding` | Qwen3-Embedding-4B | 8081 | `start-embedding-server.sh` |
| `llm` | Qwen2.5-3B-Instruct | 8080 | `start-server.sh` (inline) |

To switch modes:
```bash
ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net "echo embedding > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"
# systemd auto-restarts within 5 seconds with new mode
```

### Current Mode: qwen35

| Setting | Value |
|---------|-------|
| Model | Qwen3.5-4B-Q4_K_M (2.6 GB, 32 transformer layers) |
| Port | 8080 (OpenAI-compatible `/v1/chat/completions`) |
| GPU offload | 999 layers (full) when >4 GB free, CPU-only fallback |
| Reasoning | Disabled via `--reasoning off` |
| Context | 32768 tokens (full model context) |
| Flash attention | Enabled |
| Performance | ~14 tok/s generation, ~440 tok/s prompt processing (at 32K fill) |

### Memory Eviction

Each startup script includes a Python memory eviction trick that allocates and touches a large bytearray, forcing the OS to release filesystem cache into actual free physical pages. On Jetson's unified memory, `cudaMalloc` needs contiguous free physical pages — not just "available" memory (which includes reclaimable cache). Without this, the server may fall through to CPU-only mode after boots with heavy cache usage.

---

## Common Operations

```bash
SSH="ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net"

# Check server status
$SSH "systemctl status myscript"

# View server logs
$SSH "sudo journalctl -u myscript -f"

# Restart server (kill triggers systemd auto-restart)
$SSH "kill \$(pgrep -f llama-server)"

# Switch to embedding mode
$SSH "echo embedding > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"

# Check GPU/memory
$SSH "sudo nvidia-smi"
$SSH "free -h"

# Test chat endpoint
$SSH "curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"

# Test embedding endpoint (when in embedding mode)
$SSH "curl -s http://localhost:8081/v1/embeddings -d '{\"input\":\"test\",\"model\":\"qwen3\"}' | head -c 200"

# List models on disk
$SSH "ls -lhS ~/llm-server/models/"
```

---

## Key Paths

All paths below are relative to the `claude` user (`/home/claude/`):

| Path | Purpose |
|------|---------|
| `~/llm-server/` | Root directory for all LLM server files |
| `~/llm-server/models/` | GGUF model files (~34 GB total) |
| `~/llm-server/llama.cpp/` | llama.cpp source + build |
| `~/llm-server/llama.cpp/build/bin/llama-server` | The server binary |
| `~/llm-server/llama.cpp/build/bin/lib*.so*` | Shared libraries (llama, ggml, mtmd) |
| `~/llm-server/mode.txt` | Active mode selector (`qwen35`, `nemotron`, `embedding`, `llm`) |
| `~/llm-server/start-server.sh` | Systemd entry point (mode dispatcher) |
| `~/llm-server/start-qwen35-server.sh` | Qwen3.5-4B chat server config |
| `~/llm-server/start-nemotron-server.sh` | Nemotron chat server config |
| `~/llm-server/start-embedding-server.sh` | Embedding server config |
| `/etc/systemd/system/myscript.service` | Systemd unit file |
| `/home/davistroy/migrate-jetson-to-ssd/` | NVMe migration scripts (from initial setup) |

---

## Other Installed Software

| Tool | Version / Status |
|------|-----------------|
| Ollama | v0.16.2 installed, **service disabled** (llama.cpp used instead) |
| Docker | 29.2.1 installed, no containers running |
| jtop | v4.3.2 — Jetson system monitor |
| btop | Installed (`~/btop/`) |
| NVM | Installed (Node v24.13.1) |
| Tailscale | Active, connected to `troy.davis@` tailnet |

---

## Recreating From Scratch

1. **Flash JetPack 6.2.2** (R36.5) to the Jetson via SDK Manager
2. **Migrate to NVMe** boot drive (scripts in `~/migrate-jetson-to-ssd/`)
3. **Install Tailscale** and join the mesh network
4. **Set up `.bashrc`** with CUDA environment variables (see above)
5. **Create `claude` user** with SSH key auth and add to `render` group for CUDA access
6. **Create `/home/claude/llm-server/` directory structure**
7. **Clone llama.cpp** and checkout commit `5744d7ec4`:
   ```bash
   cd ~/llm-server
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp && git checkout 5744d7ec4
   ```
8. **Build** with the cmake flags documented above
9. **Download models** to `/home/claude/llm-server/models/` from Hugging Face
10. **Copy startup scripts** from this repo to `/home/claude/llm-server/`
11. **Install systemd service** (with `SupplementaryGroups=render` and `Environment` lines):
    ```bash
    sudo cp myscript.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable myscript
    sudo systemctl start myscript
    ```
12. **Set mode**: `echo "qwen35" > /home/claude/llm-server/mode.txt`
13. **Configure swap**: Create 16 GB swap file at `/ssd/16GB.swap`

---

## Constraints & Gotchas

- **8 GB unified RAM** — model + KV cache + OS must all fit. The memory eviction script is critical for reliable GPU offload on boot.
- **`claude` user has limited NOPASSWD sudo** — includes `systemctl`, `docker`, `cp`, `mv`, `rm`, `mkdir`, `chmod`, `chown`, `nvidia-smi`, `apt`, `dpkg`, `tee`, `reboot`, but NOT `cat`, `ls`, `usermod`, or general commands. Use `kill $(pgrep -f llama-server)` for quick restarts or `sudo systemctl restart myscript` for full restarts.
- **CUDA VMM must be ON** — `GGML_CUDA_NO_VMM=ON` causes allocation failures on Jetson's unified memory architecture. Always build with the default (VMM enabled).
- **CMAKE_CUDA_ARCHITECTURES must be 87** — Orin's exact compute capability. Building without it causes PTX JIT fallback.
- **NVMe is the boot drive** — migrated from eMMC. Migration scripts preserved in `~/migrate-jetson-to-ssd/`.
- **`/usr/local/cuda/bin` not in default PATH** — must be added for cmake CUDA detection during builds.
- **NvMapMemAllocInternalTagged warnings** — these appear in logs even during normal operation and are non-fatal. They're a known Jetson unified memory behavior. Only a problem if followed by SEGV or OOM.
- **Rapid crash-restart loops** can fragment CUDA memory and require a reboot to recover. The 5-second RestartSec in systemd helps but prolonged failures need manual intervention.
- **`render` group required for CUDA** — `/dev/dri/renderD128` is owned by group `render`. The systemd unit uses `SupplementaryGroups=render` to grant access. Without this, CUDA init fails with "operation not supported" and the server falls back to CPU-only (~9 tok/s vs ~17 tok/s).
