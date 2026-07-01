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

# Test chat (qwen35 default, port 8080) — API KEY REQUIRED since 2026-06-30 (Phase 5.6)
$SSH "curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -H \"Authorization: Bearer \$(cat ~/llm-server/.apikey)\" -d '{\"model\":\"qwen3.5-4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
# Test embedding (port 8081)
$SSH "curl -s http://localhost:8081/v1/embeddings -H \"Authorization: Bearer \$(cat ~/llm-server/.apikey)\" -d '{\"input\":\"test\",\"model\":\"qwen3\"}' | head -c 200"
# (/health stays unauthenticated: $SSH "curl -s http://localhost:8080/health")
```

## Security posture (hardened 2026-06-30, Phase 5.6 — see IMPLEMENTATION_PLAN / LAB_NOTEBOOK Entry 034)
- **LLM API requires an API key.** Key file `~/llm-server/.apikey` (root/claude-owned, 600); source of truth in Bitwarden `dev/jetson/llm-api-key`. All 5 start scripts pass `--api-key-file`. Unauthed → 401; `/health` is public. Consumers send `Authorization: Bearer <key>` (contact-center-lab uses `${JETSON_LLM_API_KEY}`).
- **Firewall (`ufw`) default-deny inbound.** Allowed: `lo`, `tailscale0` (full), and `22` from `192.168.10.0/24`. So **8080/8081 are reachable only over the tailnet + loopback — NOT the raw LAN.** (`sudo ufw status`; claude has NOPASSWD ufw via `/etc/sudoers.d/claude-ufw`.)
- **SSH is key-only** (`/etc/ssh/sshd_config.d/00-hardening.conf`: PasswordAuthentication no).
- **Health monitoring → see `OBSERVABILITY.md`** (integration contract). Interim: `ubuntu-vm` cron (`scripts/jetson-watch.sh` → `~/.local/bin/`, `*/15`) emits `jetson_up`/`tailnet_up`/`lan_up`/`llm_health`. **⚠ The push is currently broken** (HTTP 000) — the homeserver stack was rebuilt `open-brain-*` → `observability-*` with Prometheus/pushgateway bound to `127.0.0.1`. Integrate the Jetson via the **pull-based** path in `OBSERVABILITY.md` (node_exporter on the Jetson, scraped over the tailnet + Alertmanager rules).

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
