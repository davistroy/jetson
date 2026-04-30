# CLAUDE.md — Jetson

## Access
```bash
ssh claude@<jetson-tailscale-ip>
```

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
SSH="ssh claude@<jetson-tailscale-ip>"

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

## Constraints
- **8 GB unified RAM** — model + KV cache + OS all share it. Q4_K_M 4B = sweet spot; 7B works but tight.
- `nvidia-smi` memory reporting unreliable — use `free -h` and `jtop`.
- Boot drive is NVMe (migrated from eMMC).
- Persistent changes require updating `myscript.service` + `daemon-reload`.
- Startup scripts auto-drop GPU offload if free RAM < 4 GB (OOM guard).
- `render` group required for CUDA (`/dev/dri/renderD128`) — already in systemd unit.
