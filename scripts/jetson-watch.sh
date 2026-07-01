#!/usr/bin/env bash
# Jetson liveness monitor (hardening Phase 5.7). Runs on an always-on tailnet
# host (currently ubuntu-vm, 34d+ uptime) via user cron */15.
#
# STOPGAP: pushes to a Prometheus pushgateway. As of 2026-07-01 the homeserver
# observability stack was rebuilt (open-brain-* -> observability-*) and bound
# prometheus/pushgateway to 127.0.0.1, so PUSHGW below is unreachable and the
# push fails (logged). See OBSERVABILITY.md for the intended pull-based
# integration that replaces this. Local liveness log still works regardless.
#
# Configurable target (empty PUSHGW = skip push, log-only):
PUSHGW="${JETSON_WATCH_PUSHGW:-http://homeserver.k4jda.net:9091/metrics/job/jetson_watch/instance/ubuntu-vm}"
TS_IP=100.106.252.90; LAN_IP=192.168.10.58
LOG="$HOME/.local/jetson-watch.log"

tnet=0; lan=0; llm=0
ping -c1 -W3 "$TS_IP" >/dev/null 2>&1 && tnet=1
ping -c1 -W2 "$LAN_IP" >/dev/null 2>&1 && lan=1
h=$(curl -s -m6 "http://$TS_IP:8080/health" 2>/dev/null); [ -z "$h" ] && h=$(curl -s -m6 "http://$LAN_IP:8080/health" 2>/dev/null)
echo "$h" | grep -q '"status":"ok"' && llm=1
up=0; { [ $tnet -eq 1 ] || [ $lan -eq 1 ]; } && up=1

push_code="skipped"
if [ -n "$PUSHGW" ]; then
  push_code=$(printf '# TYPE jetson_up gauge\njetson_up %d\n# TYPE jetson_tailnet_up gauge\njetson_tailnet_up %d\n# TYPE jetson_lan_up gauge\njetson_lan_up %d\n# TYPE jetson_llm_health gauge\njetson_llm_health %d\n' \
    "$up" "$tnet" "$lan" "$llm" | curl -s -m 10 -o /dev/null -w '%{http_code}' --data-binary @- "$PUSHGW" 2>/dev/null)
fi
echo "$(date -Is) up=$up tnet=$tnet lan=$lan llm=$llm push=$push_code" >> "$LOG"
