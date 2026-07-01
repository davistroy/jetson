# Jetson — Observability Integration Contract

**Purpose:** everything needed to wire the Jetson Orin Nano into a Prometheus /
Grafana / Alertmanager stack. Written for the `observability-*` stack on
homeserver (rebuilt 2026-07-01: `open-brain-*` → `observability-*`, internal
components bound to `127.0.0.1`, Alertmanager + node-exporter + cadvisor added).

**Status (2026-07-01):** the Jetson is **not yet integrated**. The interim
push-based monitor (below) is broken by the stack rebuild. Integrate via the
**pull-based** path below to match the new architecture.

---

## Endpoints & network contract

| What | Address | Auth | Notes |
|------|---------|------|-------|
| Tailnet | `100.106.252.90` / `jetson.k4jda.net` | — | scrapers must be on the tailnet |
| LAN | `192.168.10.58` (`enP8p1s0`, wired) | — | |
| LLM chat | `:8080/v1/*` | **Bearer key** | Bitwarden `dev/jetson/llm-api-key`; on box `~/llm-server/.apikey` |
| LLM health | `:8080/health` | **public** | `{"status":"ok"}` — safe for a blackbox probe |
| Embedding | `:8081/v1/*` | Bearer key | only when `mode.txt=embedding` |

**Firewall (ufw) contract — read before adding a scrape target:**
- Default-deny inbound. `8080`/`8081` (and any new exporter port) are reachable
  **only over `tailscale0` + loopback, not the raw LAN.** `22` is open from LAN + tailnet.
- `ufw allow in on tailscale0` is blanket, so **any exporter is scrapeable over the
  tailnet with no extra ufw rule.** Bind exporters to `0.0.0.0` (ufw still blocks
  the LAN) or the tailnet IP; Prometheus scrapes via `jetson.k4jda.net:<port>`.

## Signals to watch (and what each failure means)

| Signal | How | Meaning of failure |
|--------|-----|--------------------|
| **tailnet up** | `tailscale ping 100.106.252.90` / ICMP | **off-tailnet but LAN-up ⇒ the tailscaled fTPM panic (LAB_NOTEBOOK Entry 033) — the #1 failure mode.** |
| **LAN up** | ICMP `192.168.10.58` | with tailnet: distinguishes box-down vs tailscale-down |
| **LLM health** | `GET :8080/health` | `!= ok` ⇒ llama.cpp unhealthy (`systemctl status myscript`) |
| **RAM headroom** | node_exporter `MemAvailable` | 8 GB box; < ~200 MB ⇒ OOM risk (NvMap-blind) |
| **thermal** | tegrastats / thermal zones | `tj-thermal` sustained high ⇒ throttle |
| **myscript restarts** | node_exporter / systemd | crash-loop (5.4 reboots after 8 in 5 min) |

## Recommended integration (pull-based — matches the new stack)

1. **node_exporter on the Jetson:** `sudo apt-get install -y prometheus-node-exporter`
   (already in the claude NOPASSWD apt scope). Gives CPU/RAM/disk/systemd/restart metrics.
2. **GPU/thermal:** a textfile-collector script (parse `tegrastats`/thermal zones) or
   `jetson-stats`/jetson_exporter, feeding node_exporter's textfile dir.
3. **Prometheus scrape job** (mirror the existing `spark-node` job):
   ```yaml
   - job_name: "jetson-node"
     static_configs: [{ targets: ["jetson.k4jda.net:9100"] }]
   ```
4. **Blackbox/health probe** for `:8080/health` + tailnet reachability (liveness).
5. **Alertmanager rules** (priority order):
   - `up{job="jetson-node"} == 0` for 5m → **Jetson down**
   - tailnet-probe down **but** LAN/health reachable → **fTPM/tailscaled panic (Entry 033)** — reboot clears it
   - `:8080/health` probe failing while host up → **LLM server unhealthy**
   - `node_memory_MemAvailable_bytes < 200Mi` → **OOM risk**

## Interim monitor (stopgap — retire once pull-based is live)

- **Script:** `scripts/jetson-watch.sh` (repo) → deployed at `~/.local/bin/jetson-watch.sh`
  on **ubuntu-vm** (always-on, tailnet+LAN, holds the SSH key), **user cron `*/15`**.
- **Emits:** gauges `jetson_up`, `jetson_tailnet_up`, `jetson_lan_up`, `jetson_llm_health`
  (labels `job="jetson_watch"`, `instance="ubuntu-vm"`).
- **Pushes to:** `$JETSON_WATCH_PUSHGW` (default `http://homeserver.k4jda.net:9091/...`).
  **⚠ Currently HTTP 000** — the rebuilt pushgateway is `127.0.0.1`-bound, so the
  push fails (now logged as `push=000` in `~/.local/jetson-watch.log`).
- **To repoint or retire:** set `JETSON_WATCH_PUSHGW` to a reachable pushgateway, or
  set it empty for log-only, or delete the cron entry (`crontab -e`) after the
  pull-based exporter is in place.

## Quick facts for wiring

- Jetson tailnet `100.106.252.90` / `jetson.k4jda.net`; LAN `192.168.10.58`.
- API key: Bitwarden `dev/jetson/llm-api-key` → `export JETSON_LLM_API_KEY=…` for authed scrapes.
- Full config: `CLAUDE.md` (Security posture), `JETSON_CONFIG.md`, `LAB_NOTEBOOK.md` Entries 033–034.
