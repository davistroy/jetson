# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo manages the configuration, scripts, and documentation for an NVIDIA Jetson Orin Nano Super 8GB device. It is the single source of truth for what's deployed on the Jetson and how to reconfigure it.

## Jetson Access

```bash
ssh claude@<jetson-tailscale-ip>
```

The device is always-on and accessible via Tailscale mesh network. The LLM server runs under the `claude` user.

## What This Repo Manages

- **JETSON_CONFIG.md** — Hardware/software inventory snapshot of the Jetson's current state
- **Scripts and configs** that get deployed to the Jetson (startup scripts, systemd units, model configs)
- Any automation or tooling for managing the device remotely

## Key Paths on the Jetson

| Path | What |
|------|------|
| `/home/claude/llm-server/` | llama.cpp server, startup scripts, mode config |
| `/home/claude/llm-server/models/` | GGUF model files (~34 GB) |
| `/home/claude/llm-server/llama.cpp/` | llama.cpp source + build |
| `/home/claude/llm-server/mode.txt` | Active mode: `qwen35`, `nemotron`, `embedding`, or `llm` |
| `/home/claude/llm-server/start-server.sh` | Systemd entry point (reads mode.txt, dispatches) |
| `/home/claude/llm-server/start-qwen35-server.sh` | Qwen3.5-4B chat server (port 8080, default) |
| `/home/claude/llm-server/start-nemotron-server.sh` | Nemotron-3-Nano-4B chat server (port 8080) |
| `/home/claude/llm-server/start-embedding-server.sh` | Embedding server with adaptive GPU layer logic |
| `/etc/systemd/system/myscript.service` | Systemd unit for the llama.cpp server |
| `/etc/systemd/system/ollama.service` | Ollama service (installed but disabled) |

## Common Operations

```bash
SSH="ssh claude@<jetson-tailscale-ip>"

# Check server status
$SSH "systemctl status myscript"

# View server logs (requires sudo for journalctl as claude user)
$SSH "sudo journalctl -u myscript -f"

# Restart the server
$SSH "kill \$(pgrep -f llama-server)"
# systemd auto-restarts within 5 seconds

# Switch to Qwen3.5 chat mode (port 8080) — current default
$SSH "echo qwen35 > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"

# Switch to Nemotron chat mode (port 8080)
$SSH "echo nemotron > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"

# Switch to embedding mode (port 8081)
$SSH "echo embedding > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"

# Switch to Qwen2.5 LLM chat mode (port 8080)
$SSH "echo llm > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"

# Check GPU/thermal/memory
$SSH "sudo nvidia-smi"
$SSH "free -h"

# Test Qwen3.5 chat endpoint (default mode)
$SSH "curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"

# Test embedding endpoint (when in embedding mode)
$SSH "curl -s http://localhost:8081/v1/embeddings -d '{\"input\":\"test\",\"model\":\"qwen3\"}' | head -c 200"

# List models on disk
$SSH "ls -lhS ~/llm-server/models/"
```

## Architecture Context

The Jetson runs llama.cpp natively (build 8414, CUDA 12.6) via a mode-switching systemd service. The current default mode is **Qwen3.5-4B Q4_K_M** (`qwen3.5-4b`), serving OpenAI-compatible `/v1/chat/completions` on port 8080 at ~14 tok/s with full GPU offload and 32768 context (full model max). Optimized 2026-03-30 — see LAB_NOTEBOOK.md for full benchmarks.

Other available modes: `nemotron` (Nemotron-3-Nano-4B on port 8080, reasoning/tool-calling), `embedding` (Qwen3-Embedding-4B on port 8081 for contact-center-lab), `llm` (Qwen2.5-3B-Instruct on port 8080).

The service runs as the `claude` user with `SupplementaryGroups=render` (required for CUDA access to `/dev/dri/renderD128`). Environment variables for `LD_LIBRARY_PATH`, `PATH`, and `CUDA_HOME` are set in the systemd unit.

The device uses Jetson unified memory (CPU and GPU share 7.4 GB LPDDR5). The startup scripts include memory pressure detection — if free memory is below 4 GB, GPU offload drops to CPU-only. This is important context for any model or config changes: larger models or higher context sizes can trigger OOM or `NvMapMemAllocInternalTagged` errors.

## Constraints

- **8 GB unified RAM** — model + KV cache + OS must all fit. Q4_K_M quants of 4B-parameter models are the sweet spot. 7B models work but leave little headroom.
- **No discrete GPU memory** — `nvidia-smi` memory reporting is limited; use `free -h` and `jtop` for real memory visibility.
- **NVMe is the boot drive** — the system was migrated from eMMC to NVMe SSD.
- **Systemd manages everything** — the `myscript.service` unit auto-restarts the server on crash or reboot. Always update the service file and `daemon-reload` for persistent changes.
- **`render` group required for CUDA** — the `claude` user needs access to `/dev/dri/renderD128`. The systemd unit includes `SupplementaryGroups=render` to handle this.
