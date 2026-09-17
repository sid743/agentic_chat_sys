#!/usr/bin/env bash
# Set up this stack on a fresh Debian 12 / Ubuntu VM (Docker, .env, containers).
#
#   GEMINI_API_KEY=AIza... AGENT_DEFAULT_MODEL=gemini/gemini-3.1-flash-lite bash deploy/vm-bootstrap.sh
#   GROQ_API_KEY=gsk_...   AGENT_DEFAULT_MODEL=groq/openai/gpt-oss-120b     bash deploy/vm-bootstrap.sh
#
# Safe to re-run: it never replaces secrets that are already in .env, and it
# refreshes DOMAIN_CLIENT / DOMAIN_SERVER, which is what you need after the
# VM's external IP changes (it does change when a VM stops and starts).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

log() { printf '\n== %s\n' "$*"; }

# --- Docker ------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker (this takes a minute)"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
  echo "Added $USER to the docker group; log out and back in to use docker without sudo."
fi
DOCKER="docker"
docker info >/dev/null 2>&1 || DOCKER="sudo docker"

# --- .env --------------------------------------------------------------------
log "Preparing .env"
ARGS=()
if [ -n "${GEMINI_API_KEY:-}" ]; then ARGS+=(--gemini-key "$GEMINI_API_KEY"); fi
if [ -n "${GROQ_API_KEY:-}" ]; then ARGS+=(--groq-key "$GROQ_API_KEY"); fi
if [ -n "${OPENAI_API_KEY:-}" ]; then ARGS+=(--openai-key "$OPENAI_API_KEY"); fi
if [ -n "${AGENT_DEFAULT_MODEL:-}" ]; then ARGS+=(--default-model "$AGENT_DEFAULT_MODEL"); fi
if [ -n "${AGENT_ROUTER_MODEL:-}" ]; then ARGS+=(--router-model "$AGENT_ROUTER_MODEL"); fi
python3 scripts/setup_env.py ${ARGS[@]+"${ARGS[@]}"}

# --- Public address ----------------------------------------------------------
META="http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
IP="$(curl -fsS --max-time 2 -H 'Metadata-Flavor: Google' "$META" 2>/dev/null || true)"
if [ -z "$IP" ]; then IP="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || true)"; fi

if [ -n "$IP" ]; then
  log "Pointing LibreChat at http://$IP:3080"
  python3 - "$IP" <<'PY'
import re, sys
from pathlib import Path

ip = sys.argv[1]
env = Path(".env")
text = env.read_text(encoding="utf-8")
for key in ("DOMAIN_CLIENT", "DOMAIN_SERVER"):
    line = f"{key}=http://{ip}:3080"
    text = re.sub(rf"(?m)^{key}=.*$", line, text) if re.search(rf"(?m)^{key}=", text) else text + line + "\n"
env.write_text(text, encoding="utf-8")
PY
else
  echo "Could not detect a public IP; leaving DOMAIN_CLIENT / DOMAIN_SERVER as they are."
fi

# --- Run ---------------------------------------------------------------------
log "Building and starting the containers (first run pulls ~3 GB, allow a few minutes)"
$DOCKER compose up -d --build
$DOCKER compose ps

log "Checking the agent core"
python3 scripts/smoke_test.py --url http://localhost:8088 || true

cat <<EOF

Done.

  UI:       http://${IP:-<vm-ip>}:3080      (open tcp:3080 in the cloud firewall first)
  Console:  http://localhost:8088           (bound to localhost only - reach it over an SSH tunnel)

Next steps:
  1. Open the UI and register the first account; it becomes the admin.
  2. Close the door behind you:
       sed -i 's/^ALLOW_REGISTRATION=.*/ALLOW_REGISTRATION=false/' .env && $DOCKER compose up -d
  3. Logs:     $DOCKER compose logs -f librechat
     Restart:  $DOCKER compose up -d
     Stop:     $DOCKER compose down
EOF
