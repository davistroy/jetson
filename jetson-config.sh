#!/bin/bash
# jetson-config — Capture, manage, and restore Jetson Orin Nano Super configurations
# Usage: jetson-config <command> [args]
#
# Commands:
#   snapshot <name> [description]  — Capture current running state as a named config
#   list                          — List saved configs
#   show <name>                   — Show config details
#   diff <name>                   — Compare current state vs saved config
#   apply <name>                  — Apply a saved config (interactive confirmation)
#   backup                        — Sync all configs to homeserver
#   restore-from-backup           — Pull configs from homeserver (disaster recovery)

set -euo pipefail

CONFIG_DIR="/home/claude/jetson-configs"
# Set HOMESERVER_BACKUP to your backup destination, e.g.: user@host:/path/to/jetson-configs
HOMESERVER_BACKUP="${JETSON_BACKUP_DEST:?Set JETSON_BACKUP_DEST environment variable}"
SSH_KEY="${JETSON_SSH_KEY:-$HOME/.ssh/id_ed25519}"
LLM_SERVER_DIR="/home/claude/llm-server"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[jetson-config]${NC} $*"; }
warn() { echo -e "${YELLOW}[jetson-config]${NC} $*"; }
err() { echo -e "${RED}[jetson-config]${NC} $*" >&2; }

# ──────────────────────────────────────────────────────────────────────
# SNAPSHOT — Capture current running state
# ──────────────────────────────────────────────────────────────────────
cmd_snapshot() {
    local NAME="${1:?Usage: jetson-config snapshot <name> [description]}"
    local DESC="${2:-Snapshot taken $(date -Iseconds)}"
    local SNAP_DIR="$CONFIG_DIR/$NAME"

    if [ -d "$SNAP_DIR" ]; then
        warn "Config '$NAME' already exists. Overwriting."
        rm -rf "$SNAP_DIR"
    fi

    mkdir -p "$SNAP_DIR"
    log "Capturing config '$NAME'..."

    # ── System info ──
    log "  System info..."
    local TEGRA_REL=$(head -1 /etc/nv_tegra_release 2>/dev/null | sed 's/# //')
    local CUDA_VER=$(/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1 | grep -oP 'V[\d.]+' || echo 'unknown')
    local KERNEL=$(uname -r)
    local TOTAL_RAM_MB=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')

    cat > "$SNAP_DIR/system.json" << SYSEOF
{
    "hostname": "$(hostname)",
    "kernel": "$KERNEL",
    "arch": "$(uname -m)",
    "l4t_release": "$TEGRA_REL",
    "cuda": "$CUDA_VER",
    "gpu": "Orin (Ampere GA10B, sm_87, 1024 CUDA cores)",
    "total_ram_mb": $TOTAL_RAM_MB,
    "snapshot_date": "$(date -Iseconds)",
    "description": "$DESC"
}
SYSEOF

    # ── Memory state ──
    log "  Memory state..."
    free -m > "$SNAP_DIR/memory.txt"
    cat /proc/meminfo > "$SNAP_DIR/meminfo.txt"

    # ── Swap state ──
    log "  Swap state..."
    swapon --show > "$SNAP_DIR/swap.txt" 2>/dev/null || echo "no swap" > "$SNAP_DIR/swap.txt"
    # Per-process swap usage (top consumers)
    {
        echo "PID,COMMAND,SWAP_KB"
        for PID in /proc/[0-9]*; do
            P=$(basename "$PID")
            SWAP_KB=$(grep VmSwap "$PID/status" 2>/dev/null | awk '{print $2}')
            if [ -n "$SWAP_KB" ] && [ "$SWAP_KB" -gt 0 ] 2>/dev/null; then
                CMD=$(cat "$PID/comm" 2>/dev/null || echo "unknown")
                echo "$P,$CMD,$SWAP_KB"
            fi
        done
    } | sort -t, -k3 -rn > "$SNAP_DIR/swap_consumers.csv" 2>/dev/null || true

    # ── Tegrastats snapshot (single reading) ──
    log "  GPU/thermal state (tegrastats)..."
    timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 > "$SNAP_DIR/tegrastats.txt" || echo "tegrastats unavailable" > "$SNAP_DIR/tegrastats.txt"

    # ── llama.cpp server state ──
    log "  LLM server state..."
    mkdir -p "$SNAP_DIR/llm-server"

    # Current mode
    cat "$LLM_SERVER_DIR/mode.txt" > "$SNAP_DIR/llm-server/mode.txt" 2>/dev/null || echo "unknown"

    # Running process command line
    local SERVER_PID=$(pgrep -f llama-server 2>/dev/null | head -1)
    if [ -n "$SERVER_PID" ]; then
        cat "/proc/$SERVER_PID/cmdline" 2>/dev/null | tr '\0' ' ' > "$SNAP_DIR/llm-server/running_cmdline.txt"
        # Process memory
        grep -E "^(VmRSS|VmSwap|VmSize|VmPeak)" "/proc/$SERVER_PID/status" 2>/dev/null > "$SNAP_DIR/llm-server/process_memory.txt"
        echo "server_pid=$SERVER_PID" > "$SNAP_DIR/llm-server/process_info.txt"
        echo "server_running=true" >> "$SNAP_DIR/llm-server/process_info.txt"
    else
        echo "server_running=false" > "$SNAP_DIR/llm-server/process_info.txt"
    fi

    # Startup scripts (copy the active ones)
    for SCRIPT in start-server.sh start-qwen35-server.sh start-nemotron-server.sh start-embedding-server.sh; do
        [ -f "$LLM_SERVER_DIR/$SCRIPT" ] && cp "$LLM_SERVER_DIR/$SCRIPT" "$SNAP_DIR/llm-server/$SCRIPT"
    done

    # llama.cpp build info
    {
        echo "commit=$(cat "$LLM_SERVER_DIR/llama.cpp/.git/HEAD" 2>/dev/null || echo 'unknown')"
        echo "binary=$("$LLM_SERVER_DIR/llama.cpp/build/bin/llama-server" --version 2>&1 | head -1 || echo 'unknown')"
    } > "$SNAP_DIR/llm-server/build_info.txt" 2>/dev/null || true

    # ── Systemd service ──
    log "  Systemd service..."
    systemctl cat myscript > "$SNAP_DIR/myscript.service" 2>/dev/null || true
    systemctl status myscript --no-pager > "$SNAP_DIR/service_status.txt" 2>/dev/null || true

    # ── Model inventory ──
    log "  Model inventory..."
    {
        echo "["
        local FIRST=true
        for MODEL in "$LLM_SERVER_DIR"/models/*.gguf; do
            [ -f "$MODEL" ] || continue
            local MNAME=$(basename "$MODEL")
            local MSIZE=$(stat --format='%s' "$MODEL" 2>/dev/null || echo 0)
            local MSIZE_GB=$(python3 -c "print(round($MSIZE / (1024**3), 2))" 2>/dev/null || echo 0)
            if [ "$FIRST" = true ]; then
                FIRST=false
            else
                echo ","
            fi
            printf '  {"file": "%s", "size_bytes": %s, "size_gb": %s}' "$MNAME" "$MSIZE" "$MSIZE_GB"
        done
        echo ""
        echo "]"
    } > "$SNAP_DIR/models.json"

    # ── Sysctl tuning ──
    log "  Sysctl config..."
    sysctl -a 2>/dev/null | grep -E "^(vm\.(swappiness|min_free_kbytes|dirty_ratio|dirty_background_ratio|overcommit)|net\.(core\.(rmem_max|wmem_max|somaxconn)|ipv4\.tcp_(rmem|wmem|congestion_control|tw_reuse)))" > "$SNAP_DIR/sysctl.conf" 2>/dev/null || true

    # ── Network config ──
    log "  Network config..."
    mkdir -p "$SNAP_DIR/network"
    nmcli connection show 2>/dev/null > "$SNAP_DIR/network/connections.txt" || true
    ip addr show 2>/dev/null > "$SNAP_DIR/network/ip_addr.txt" || true
    # Tailscale
    tailscale status 2>/dev/null > "$SNAP_DIR/network/tailscale.txt" || echo "tailscale not available" > "$SNAP_DIR/network/tailscale.txt"

    # ── Docker state (if running) ──
    log "  Docker state..."
    if command -v docker &>/dev/null && docker ps &>/dev/null 2>&1; then
        docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}' > "$SNAP_DIR/docker_containers.txt" 2>/dev/null || true
        docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep -v '<none>' > "$SNAP_DIR/docker_images.txt" 2>/dev/null || true
    else
        echo "docker not running or not accessible" > "$SNAP_DIR/docker_containers.txt"
    fi

    # ── Disk usage ──
    log "  Disk usage..."
    df -h / > "$SNAP_DIR/disk.txt"
    du -sh "$LLM_SERVER_DIR/models/" 2>/dev/null >> "$SNAP_DIR/disk.txt" || true

    # ── Generate restore script ──
    log "  Generating restore.sh..."
    _generate_restore "$SNAP_DIR" "$NAME"

    # ���─ Summary ──
    local SIZE=$(du -sh "$SNAP_DIR" | cut -f1)
    log "Config '$NAME' saved to $SNAP_DIR ($SIZE)"
    log "Run 'jetson-config backup' to sync to homeserver"
}

# ────────────────────────────────────────────────────────────────────���─
# Generate restore script
# ──────────────────────────────────────────────────────────────────────
_generate_restore() {
    local DIR="$1"
    local NAME="$2"
    local RESTORE="$DIR/restore.sh"

    cat > "$RESTORE" << 'RESTEOF'
#!/bin/bash
# Restore script for jetson config
# Generated by jetson-config snapshot
# Usage: ./restore.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLM_SERVER_DIR="/home/claude/llm-server"
DRY_RUN="${1:-}"

run() {
    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "[DRY RUN] $*"
    else
        echo "[RUNNING] $*"
        eval "$@"
    fi
}

echo "=== Jetson Config Restore ==="
echo "Config: $(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/system.json')); print(d.get('description',''))")"
echo "Snapshot date: $(python3 -c "import json; d=json.load(open('$SCRIPT_DIR/system.json')); print(d.get('snapshot_date',''))")"
echo ""

# Step 1: Apply sysctl tuning
echo "--- Step 1: Sysctl tuning ---"
if [ -f "$SCRIPT_DIR/sysctl.conf" ]; then
    run "sudo cp '$SCRIPT_DIR/sysctl.conf' /etc/sysctl.d/99-jetson-tuning.conf"
    run "sudo sysctl -p /etc/sysctl.d/99-jetson-tuning.conf"
fi

# Step 2: Restore startup scripts
echo "--- Step 2: Startup scripts ---"
for SCRIPT in start-server.sh start-qwen35-server.sh start-nemotron-server.sh start-embedding-server.sh; do
    if [ -f "$SCRIPT_DIR/llm-server/$SCRIPT" ]; then
        run "cp '$SCRIPT_DIR/llm-server/$SCRIPT' '$LLM_SERVER_DIR/$SCRIPT'"
        run "chmod +x '$LLM_SERVER_DIR/$SCRIPT'"
    fi
done

# Step 3: Restore mode
echo "--- Step 3: Mode ---"
if [ -f "$SCRIPT_DIR/llm-server/mode.txt" ]; then
    run "cp '$SCRIPT_DIR/llm-server/mode.txt' '$LLM_SERVER_DIR/mode.txt'"
fi

# Step 4: Restore systemd service
echo "--- Step 4: Systemd service ---"
if [ -f "$SCRIPT_DIR/myscript.service" ]; then
    # Strip the systemctl cat header line if present
    run "grep -v '^# /etc/systemd' '$SCRIPT_DIR/myscript.service' | sudo tee /etc/systemd/system/myscript.service > /dev/null"
    run "sudo systemctl daemon-reload"
fi

# Step 5: Restart service
echo "--- Step 5: Restart server ---"
run "kill \$(pgrep -f llama-server) 2>/dev/null || true"
echo "Waiting for systemd auto-restart..."
run "sleep 8"

echo ""
echo "=== Restore complete ==="
echo "Verify: systemctl status myscript && free -h"
RESTEOF

    chmod +x "$RESTORE"
}

# ──────────────────────────────────────────────────────────────────────
# LIST — Show saved configs
# ──────────────────────────────────────────────────────────────────────
cmd_list() {
    echo -e "${CYAN}Saved configurations:${NC}"
    echo ""
    printf "%-25s %-20s %s\n" "NAME" "DATE" "DESCRIPTION"
    printf "%-25s %-20s %s\n" "----" "----" "-----------"

    for DIR in "$CONFIG_DIR"/*/; do
        [ -d "$DIR" ] || continue
        local NAME=$(basename "$DIR")
        if [ -f "$DIR/system.json" ]; then
            local DATE=$(python3 -c "import json; d=json.load(open('$DIR/system.json')); print(d.get('snapshot_date','?')[:19])" 2>/dev/null || echo "?")
            local DESC=$(python3 -c "import json; d=json.load(open('$DIR/system.json')); print(d.get('description','')[:50])" 2>/dev/null || echo "")
            printf "%-25s %-20s %s\n" "$NAME" "$DATE" "$DESC"
        fi
    done
}

# ──────────────────────────────────────────────────────────────────────
# SHOW — Display config details
# ──────────────────────────────────────────────────────────────────────
cmd_show() {
    local NAME="${1:?Usage: jetson-config show <name>}"
    local DIR="$CONFIG_DIR/$NAME"
    [ -d "$DIR" ] || { err "Config '$NAME' not found"; exit 1; }

    echo -e "${CYAN}=== Config: $NAME ===${NC}"
    if [ -f "$DIR/system.json" ]; then
        python3 -c "
import json
with open('$DIR/system.json') as f:
    d = json.load(f)
for k, v in d.items():
    print(f'  {k}: {v}')
"
    fi

    echo ""
    echo -e "${CYAN}LLM Server:${NC}"
    if [ -f "$DIR/llm-server/mode.txt" ]; then
        echo "  Mode: $(cat "$DIR/llm-server/mode.txt")"
    fi
    if [ -f "$DIR/llm-server/running_cmdline.txt" ]; then
        echo "  Command: $(cat "$DIR/llm-server/running_cmdline.txt")"
    fi
    if [ -f "$DIR/llm-server/process_memory.txt" ]; then
        echo "  Memory:"
        sed 's/^/    /' "$DIR/llm-server/process_memory.txt"
    fi

    echo ""
    echo -e "${CYAN}Models:${NC}"
    if [ -f "$DIR/models.json" ]; then
        python3 -c "
import json
with open('$DIR/models.json') as f:
    d = json.load(f)
for m in d:
    print(f'  - {m[\"file\"]} ({m[\"size_gb\"]} GB)')
"
    fi

    echo ""
    echo -e "${CYAN}Memory:${NC}"
    [ -f "$DIR/memory.txt" ] && cat "$DIR/memory.txt" | sed 's/^/  /'

    echo ""
    echo -e "${CYAN}Files:${NC}"
    find "$DIR" -type f | sort | while read F; do
        echo "  $(echo "$F" | sed "s|$DIR/||")"
    done
}

# ──────────────────────────────────────────────────────────────────────
# DIFF — Compare current state vs saved config
# ──────────────────────────────────────────────────────────────────────
cmd_diff() {
    local NAME="${1:?Usage: jetson-config diff <name>}"
    local DIR="$CONFIG_DIR/$NAME"
    [ -d "$DIR" ] || { err "Config '$NAME' not found"; exit 1; }

    log "Comparing current state to config '$NAME'..."

    # Compare mode
    echo -e "\n${CYAN}Mode:${NC}"
    local CUR_MODE=$(cat "$LLM_SERVER_DIR/mode.txt" 2>/dev/null || echo "unknown")
    local SAV_MODE=$(cat "$DIR/llm-server/mode.txt" 2>/dev/null || echo "unknown")
    if [ "$CUR_MODE" = "$SAV_MODE" ]; then
        echo "  Mode: $CUR_MODE (unchanged)"
    else
        echo -e "  ${YELLOW}Mode changed:${NC} $SAV_MODE → $CUR_MODE"
    fi

    # Compare running command
    echo -e "\n${CYAN}Server command:${NC}"
    local CUR_CMD=$(cat "/proc/$(pgrep -f llama-server 2>/dev/null | head -1)/cmdline" 2>/dev/null | tr '\0' ' ')
    local SAV_CMD=$(cat "$DIR/llm-server/running_cmdline.txt" 2>/dev/null || echo "not running")
    if [ "$CUR_CMD" = "$SAV_CMD" ]; then
        echo "  (unchanged)"
    else
        echo -e "  ${YELLOW}Saved:${NC} $SAV_CMD"
        echo -e "  ${YELLOW}Current:${NC} $CUR_CMD"
    fi

    # Compare memory
    echo -e "\n${CYAN}Memory:${NC}"
    local CUR_MEM=$(free -m | awk '/Mem:/{print $3}')
    local SAV_MEM=$(awk '/Mem:/{print $3}' "$DIR/memory.txt" 2>/dev/null || echo "?")
    echo "  Used RAM: ${SAV_MEM}M (saved) → ${CUR_MEM}M (current)"

    local CUR_SWAP=$(free -m | awk '/Swap:/{print $3}')
    local SAV_SWAP=$(awk '/Swap:/{print $3}' "$DIR/memory.txt" 2>/dev/null || echo "?")
    echo "  Used swap: ${SAV_SWAP}M (saved) → ${CUR_SWAP}M (current)"

    # Compare sysctl
    echo -e "\n${CYAN}Sysctl differences:${NC}"
    if [ -f "$DIR/sysctl.conf" ]; then
        local CURRENT_SYSCTL=$(mktemp)
        sysctl -a 2>/dev/null | grep -E "^(vm\.(swappiness|min_free_kbytes|dirty_ratio)|net\.(core\.(rmem_max|wmem_max)|ipv4\.tcp_congestion_control))" | sort > "$CURRENT_SYSCTL"
        diff <(sort "$DIR/sysctl.conf") "$CURRENT_SYSCTL" 2>/dev/null && echo "  (no differences)" || true
        rm -f "$CURRENT_SYSCTL"
    fi

    # Compare systemd service
    echo -e "\n${CYAN}Systemd service:${NC}"
    local CUR_SVC=$(mktemp)
    systemctl cat myscript > "$CUR_SVC" 2>/dev/null || true
    if [ -f "$DIR/myscript.service" ]; then
        diff "$DIR/myscript.service" "$CUR_SVC" 2>/dev/null && echo "  (no differences)" || true
    fi
    rm -f "$CUR_SVC"
}

# ──────────────────────────────────────────────────────────────────────
# APPLY — Restore a saved config
# ──────────────────────────────────────────────────────────────────────
cmd_apply() {
    local NAME="${1:?Usage: jetson-config apply <name>}"
    local DIR="$CONFIG_DIR/$NAME"
    [ -d "$DIR" ] || { err "Config '$NAME' not found"; exit 1; }

    warn "This will restore config '$NAME' and restart the LLM server."
    warn "Run '$DIR/restore.sh --dry-run' first to preview."
    echo ""
    read -p "Proceed? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Aborted."
        exit 0
    fi

    log "Applying config '$NAME'..."
    bash "$DIR/restore.sh"
}

# ──────────────────────────────────────────────────────────────────────
# BACKUP — Sync configs to homeserver
# ──────────────────────────────────────────────────────────────────────
cmd_backup() {
    log "Syncing configs to homeserver..."

    local SSH_OPTS=""
    [ -f "$SSH_KEY" ] && SSH_OPTS="-e 'ssh -i $SSH_KEY'"

    rsync -avz --delete \
        ${SSH_OPTS} \
        "$CONFIG_DIR/" \
        "$HOMESERVER_BACKUP/" 2>&1

    log "Backup complete. Configs synced to homeserver."
}

# ──────────────────────────────────────────────────────────────────────
# RESTORE-FROM-BACKUP — Pull configs from homeserver (disaster recovery)
# ──────────────────────────────────────────────────────────────────────
cmd_restore_from_backup() {
    log "Pulling configs from homeserver..."

    local SSH_OPTS=""
    [ -f "$SSH_KEY" ] && SSH_OPTS="-e 'ssh -i $SSH_KEY'"

    mkdir -p "$CONFIG_DIR"
    rsync -avz \
        ${SSH_OPTS} \
        "$HOMESERVER_BACKUP/" \
        "$CONFIG_DIR/" 2>&1

    log "Configs restored from homeserver."
    cmd_list
}

# ──────────────────────────────────────────────────────────────────────
# Main dispatcher
# ──────────────────────────────────────────────────────────────────────
case "${1:-help}" in
    snapshot)           shift; cmd_snapshot "$@" ;;
    list|ls)            cmd_list ;;
    show)               shift; cmd_show "$@" ;;
    diff)               shift; cmd_diff "$@" ;;
    apply)              shift; cmd_apply "$@" ;;
    backup)             cmd_backup ;;
    restore-from-backup) cmd_restore_from_backup ;;
    help|--help|-h)
        echo "jetson-config — Jetson Orin Nano Super configuration management"
        echo ""
        echo "Commands:"
        echo "  snapshot <name> [desc]   Capture current running state"
        echo "  list                     List saved configs"
        echo "  show <name>              Show config details"
        echo "  diff <name>              Compare current vs saved"
        echo "  apply <name>             Apply a saved config"
        echo "  backup                   Sync all configs to homeserver"
        echo "  restore-from-backup      Pull configs from homeserver"
        ;;
    *)
        err "Unknown command: $1"
        exit 1
        ;;
esac
