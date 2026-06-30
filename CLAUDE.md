# CLAUDE.md — Jetson

## Access
```bash
# Tailnet (preferred — works from anywhere while the node is online):
ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net    # -> 100.106.252.90 (tailnet)
# LAN (fallback — use when the node shows "offline" on Tailscale, see Troubleshooting):
ssh -i ~/.ssh/id_claude_code claude@192.168.10.58
```
- **Tailnet:** IP `100.106.252.90`, MagicDNS `jetson.tale-mamba.ts.net`; `jetson.k4jda.net` is a DNS alias to the tailnet IP (only reachable when Tailscale is up).
- **LAN:** IP `192.168.10.58` on **wired Ethernet** (`enP8p1s0`); WiFi (`wlP1p1s0`) is down/unused.
- User `claude` has passwordless sudo for ops commands (see JETSON_CONFIG.md).

## Key Paths

| Path | What |
|------|------|
| `/home/claude/llm-server/` | llama.cpp server, scripts, mode config |
| `/home/claude/llm-server/models/` | GGUF model files (~34 GB) |
| `/home/claude/llm-server/mode.txt` | Active mode: `qwen35`, `nemotron`, `embedding`, `experiment`, `llm` |
| `/home/claude/llm-server/start-server.sh` | Systemd entry point |
| `/etc/systemd/system/myscript.service` | Systemd unit (auto-restarts) |
| `/etc/systemd/system/ollama.service` | Ollama (installed, disabled) |

## Common Operations
```bash
SSH="ssh -i ~/.ssh/id_claude_code claude@jetson.k4jda.net"   # or claude@192.168.10.58 over LAN

$SSH "systemctl status myscript"
$SSH "sudo journalctl -u myscript -f"
$SSH "kill \$(pgrep -f llama-server)"   # systemd auto-restarts in 5s

# Switch modes (kill triggers restart in new mode)
$SSH "echo qwen35 > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"
$SSH "echo nemotron > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"
$SSH "echo embedding > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"
$SSH "echo llm > ~/llm-server/mode.txt && kill \$(pgrep -f llama-server)"

$SSH "sudo nvidia-smi" && $SSH "free -h"
$SSH "ls -lhS ~/llm-server/models/"

# Test chat (qwen35 default, port 8080)
$SSH "curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
# Test embedding (port 8081)
$SSH "curl -s http://localhost:8081/v1/embeddings -d '{\"input\":\"test\",\"model\":\"qwen3\"}' | head -c 200"
```

## Architecture
- llama.cpp b8987 (rebuilt 2026-04-30 from b8766, +9-12% throughput), CUDA 12.6
- Default: **Qwen3.5-4B Q4_K_M** on port 8080, ~15.2–15.7 tok/s, 32768 ctx, q8_0 KV, mlock, full GPU offload
- Other modes: `nemotron` (Nemotron-3-Nano-4B, port 8080), `embedding` (Qwen3-Embedding-4B, port 8081), `experiment` (A/B test slot, port 8080), `llm` (Qwen2.5-3B-Instruct, port 8080)
- Systemd unit: `claude` user, `SupplementaryGroups=render`, `LD_LIBRARY_PATH`/`PATH`/`CUDA_HOME` set

## Troubleshooting
- **Node shows "offline" on Tailscale but you can't reach it → the box is almost certainly UP.** Check the **LAN IP `192.168.10.58`** and `uptime` before assuming a crash/reboot. Known failure mode (Entry 033, 2026-06-30): `tailscaled` 1.98.4 **panics at startup** (`slice bounds out of range [:-53212]`) when the OP-TEE firmware-TPM errors (`tpm_try_transmit: send(): error -53212`), and crash-loops until systemd gives up — taking only the tailnet link down while the host keeps serving. **Fix: reboot** (`sudo systemctl reboot`) clears the OP-TEE/fTPM state; everything is reboot-durable and auto-starts (~1–2 min). Durable alternatives if it recurs: pin/downgrade tailscale, or disable tailscale TPM state-sealing via a systemd drop-in.
- **`systemctl --failed` shows ~6 system units failed after boot** (`nvphs`, `avahi-daemon`, `wpa_supplicant`, `networkd-dispatcher`, `ModemManager`, `kerneloops`) — expected, not a fault. The `oom-protect.conf` drop-in shields `llama-server` (OOMScoreAdjust −900), so the boot-time memory spike sacrifices these expendable services instead. No functional impact (wired Ethernet, NetworkManager, no modem). They clear on a clean reboot.
- **Dual-environment hazard:** the repo's `main` is also pushed from the Windows/other Claude Code environment. `git fetch` at session start on either box to avoid divergence.

## Constraints
- **8 GB unified RAM** — model + KV cache + OS all share it. Q4_K_M 4B = sweet spot; 7B works but tight.
- `nvidia-smi` memory reporting unreliable — use `free -h` and `jtop`.
- Boot drive is NVMe (migrated from eMMC).
- Persistent changes require updating `myscript.service` + `daemon-reload`.
- Startup scripts auto-drop GPU offload if free RAM < 4 GB (OOM guard).
- `render` group required for CUDA (`/dev/dri/renderD128`) — already in systemd unit.
