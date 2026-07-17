#!/bin/bash
# ufw-watchdog.sh — assert the ufw firewall stays active; alert + self-heal on a silent down.
#
# LAB_NOTEBOOK Entry 037 (2026-07-16). Motivated by Entry 036: ufw was found INACTIVE on a
# 16-day-uptime boot — it came up fine at boot, then was disabled at RUNTIME (most likely an
# orphaned Entry-034 dead-man's-switch that fired ~5 min after the firewall was enabled).
# ufw.service only asserts the firewall at boot, so a runtime disable went unnoticed until a
# recon days later. This watchdog closes that gap: it re-checks at boot AND every 2 minutes.
#
# Invoked by ufw-watchdog.timer (OnBootSec=2min + OnUnitActiveSec=2min). Runs as root via
# ufw-watchdog.service (Type=oneshot), so it calls `ufw` directly (no sudo).
#
# Each run:
#   - reads `ufw status` and writes a node_exporter textfile metric (jetson_ufw_active 0|1 +
#     last-run timestamp + heal counter) to $METRIC — ready for the OBSERVABILITY.md pull path
#     (Debian prometheus-node-exporter's default textfile dir), so it is auto-scraped once
#     node_exporter is installed (Phase 5.7). No CSV spam when healthy — the timer journal
#     already proves liveness; the CSV records only anomalies and actions.
#   - if INACTIVE:
#       * MAINTENANCE flag present -> log a skip, take NO action (mirrors memory-watchdog).
#       * within COOLDOWN of the last heal -> ESCALATE: louder CRITICAL, do NOT re-enable
#         (something is actively disabling it; a human must intervene — don't fight a war).
#       * otherwise -> CRITICAL log + `ufw --force enable` (rules are known-good; enabling over
#         tailscale0 cannot lock out the admin path), re-check, record the heal.
#
# Env overrides (testing only): UFW_WD_COOLDOWN, UFW_WD_METRIC_DIR, UFW_WD_STATE_DIR.
set -uo pipefail

STATE_DIR="${UFW_WD_STATE_DIR:-/home/claude/llm-server/watchdog}"
METRIC_DIR="${UFW_WD_METRIC_DIR:-/var/lib/prometheus/node-exporter}"
MAINT="/home/claude/llm-server/MAINTENANCE"
COOLDOWN="${UFW_WD_COOLDOWN:-600}"

METRIC="$METRIC_DIR/ufw.prom"
CSV="$STATE_DIR/ufw-heartbeat.csv"
LAST_HEAL_FILE="$STATE_DIR/ufw-last-heal"
HEALS_FILE="$STATE_DIR/ufw-heals-count"

TAG="ufw-watchdog"
NOW="$(date +%s)"
STAMP="$(date -Is)"

mkdir -p "$STATE_DIR" "$METRIC_DIR" 2>/dev/null || true

log() {  # $1=priority (info|warning|crit); rest=message. -> journal (logger tag) + stdout (unit journal)
    local pri="$1"; shift
    logger -t "$TAG" -p "daemon.$pri" -- "$*" 2>/dev/null || true
    echo "[$STAMP] [$pri] $*"
}

csv_row() {  # append an anomaly/action row only (keeps the file small)
    echo "$STAMP,$1,$2" >> "$CSV" 2>/dev/null || true
}

write_metric() {  # $1=active(0|1) $2=heals_total ; atomic write for the node_exporter textfile collector
    local active="$1" heals="$2" tmp
    tmp="$(mktemp "$METRIC_DIR/.ufw.prom.XXXXXX" 2>/dev/null)" || tmp="$METRIC.tmp"
    {
        echo "# HELP jetson_ufw_active ufw firewall active (1) or inactive (0), last observed by ufw-watchdog."
        echo "# TYPE jetson_ufw_active gauge"
        echo "jetson_ufw_active $active"
        echo "# HELP jetson_ufw_watchdog_last_run_seconds Unix time of the last ufw-watchdog run."
        echo "# TYPE jetson_ufw_watchdog_last_run_seconds gauge"
        echo "jetson_ufw_watchdog_last_run_seconds $NOW"
        echo "# HELP jetson_ufw_watchdog_heals_total Times ufw-watchdog re-enabled a down firewall."
        echo "# TYPE jetson_ufw_watchdog_heals_total counter"
        echo "jetson_ufw_watchdog_heals_total $heals"
    } > "$tmp" 2>/dev/null
    mv -f "$tmp" "$METRIC" 2>/dev/null || true
    chmod 0644 "$METRIC" 2>/dev/null || true
}

is_active() {  # capture-then-match (no pipe) so `set -o pipefail` + SIGPIPE can't misreport state
    local out
    out="$(ufw status 2>/dev/null)"
    [[ "$out" == *"Status: active"* ]]
}

heals="$(cat "$HEALS_FILE" 2>/dev/null || echo 0)"; [[ "$heals" =~ ^[0-9]+$ ]] || heals=0

if is_active; then
    write_metric 1 "$heals"
    exit 0
fi

# ---- ufw is INACTIVE ----
if [ -f "$MAINT" ]; then
    log warning "ufw INACTIVE but MAINTENANCE flag present ($MAINT) — taking no action."
    csv_row inactive maintenance-skip
    write_metric 0 "$heals"
    exit 0
fi

last_heal="$(cat "$LAST_HEAL_FILE" 2>/dev/null || echo 0)"; [[ "$last_heal" =~ ^[0-9]+$ ]] || last_heal=0
since=$(( NOW - last_heal ))

if [ "$last_heal" -gt 0 ] && [ "$since" -lt "$COOLDOWN" ]; then
    log crit "CRITICAL: ufw INACTIVE AGAIN ${since}s after a heal (< ${COOLDOWN}s cooldown) — a process is actively disabling it. NOT re-enabling; MANUAL INTERVENTION REQUIRED."
    csv_row inactive escalate-no-heal
    write_metric 0 "$heals"
    exit 1
fi

log crit "CRITICAL: ufw INACTIVE — re-enabling (loading the configured default-deny ruleset)."
csv_row inactive heal-attempt
ufw --force enable >/dev/null 2>&1 || true

if is_active; then
    heals=$(( heals + 1 ))
    echo "$heals" > "$HEALS_FILE" 2>/dev/null || true
    echo "$NOW"    > "$LAST_HEAL_FILE" 2>/dev/null || true
    log warning "ufw re-enabled OK (heal #$heals). Expect: LAN :8080 blocked, tailnet reachable, SSH preserved."
    csv_row active heal-ok
    write_metric 1 "$heals"
    exit 0
else
    log crit "CRITICAL: ufw re-enable FAILED — still inactive after 'ufw --force enable'. MANUAL INTERVENTION REQUIRED."
    csv_row inactive heal-failed
    write_metric 0 "$heals"
    exit 1
fi
