# ADR-0001: LLM API exposure model

**Status:** Proposed
**Date:** 2026-06-30
**Deciders:** Troy Davis (pending), Claude Code
**Context source:** LAB_NOTEBOOK Entry 034 (ultra-plan hardening analysis); healthcheck Entry 033
**Affects:** IMPLEMENTATION_PLAN Phase 5, work item 5.6 (CS-E / H6)

## Context

The llama.cpp servers bind `--host 0.0.0.0` on ports 8080 (chat/LLM) and 8081
(embedding) with **no authentication** — confirmed empirically on 2026-06-30 by
running a full `/v1/chat/completions` inference from a *different* LAN host with
no credentials. There is **no host firewall** (`ufw` absent, nftables ruleset
empty). Open LAN ports: 22, 8080, and rpcbind/111.

The only known consumer, `contact-center-lab`, connects to **port 8080** over the
tailnet (`jetson.k4jda.net:8080`) and `localhost` (`pipeline/config.yaml:61`),
OpenAI-compatible, with **no API key today**.

Threat model: single-admin homelab behind NAT, on a shared home LAN + a private
tailnet. The realistic threat is lateral movement from a compromised LAN/IoT
device — not an internet attacker. But llama.cpp's server is a C++ binary parsing
untrusted input with a CVE history; an unauthenticated instance on the LAN is both
free-compute-for-anyone and a potential pivot. The box also has only 8 GB unified
RAM, so an unbounded external workload is a stability risk too.

## Decision

**Defense in depth — three layers, shipped together:**

1. **Keep `--host 0.0.0.0`, add `--api-key`** (read from a root-owned
   `~/llm-server/.apikey`; the key itself stored in Bitwarden `dev/jetson/llm-api-key`,
   never committed) to all five `start-*.sh` scripts.
2. **Host firewall (`ufw`), default-deny inbound:** allow `22` from the LAN and
   tailnet, allow `8080`/`8081` from the tailnet CGNAT range `100.64.0.0/10` and
   loopback only — i.e. drop raw-LAN access to the inference ports.
3. **Update `contact-center-lab`** (`pipeline/config.yaml` + client) to send the
   key via environment variable (placeholder in git, real value out-of-band).

## Alternatives considered

| # | Approach | Rejected because |
|---|----------|------------------|
| A | Bind only to `tailscale0` (drop 0.0.0.0) | llama.cpp binds a single `--host`; loses the loopback consumer + on-box smoke tests; fragile if `tailscale0` isn't up at service start (and tailscale just demonstrated it can be down for days). |
| B | API-key only, no firewall | Leaves rpcbind and any future accidental listener exposed; no network-layer backstop; doesn't bound resource abuse from tailnet peers. |
| C | Reverse proxy (swag/caddy) with auth in front | Heavier; another moving part + memory cost on an 8 GB box; the consumer is internal, so it buys little over an api-key. |
| D | Do nothing (NAT is sufficient) | The confirmed-open API is reachable by every LAN/IoT device; lateral movement is the realistic threat. Rejected on evidence. |

## Consequences

- **Positive:** closes the confirmed open door at both the app and network layers; api-key protects tailnet peers too (firewall alone wouldn't); rpcbind and future listeners covered by default-deny.
- **Negative / cost:** small per-request key check; `contact-center-lab` must carry the key (env, not committed) — a cross-repo coordination point; the firewall introduces self-lockout risk, mitigated by a staged dead-man's-switch (`at`-scheduled `ufw disable`, verify from a second session, then cancel).
- **Reversibility:** `ufw disable`; remove `--api-key`; revert the consumer config. Fully reversible.
