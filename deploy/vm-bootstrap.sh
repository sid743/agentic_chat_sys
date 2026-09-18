#!/usr/bin/env bash
# Set up this stack on a fresh Ubuntu or Debian VM (Docker, .env, containers).
# Works on Azure, Google Cloud or any other VM; see deploy/README.md.
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
AZURE_META="http://169.254.169.254/metadata/instance/network/interface/0/ipv4/ipAddress/0/publicIpAddress?api-version=2021-02-01&format=text"
GCP_META="http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
IP="$(curl -fsS --max-time 2 -H 'Metadata:true' "$AZURE_META" 2>/dev/null || true)"
if [ -z "$IP" ]; then IP="$(curl -fsS --max-time 2 -H 'Metadata-Flavor: Google' "$GCP_META" 2>/dev/null || true)"; fi
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

log "Making sure the demo account exists"
env_value() { grep -E "^$1=" .env | head -1 | cut -d= -f2-; }
DEMO_EMAIL="$(env_value DEMO_USER_EMAIL)"
DEMO_NAME="$(env_value DEMO_USER_NAME)"
DEMO_PASS="$(env_value DEMO_USER_PASSWORD)"
DOMAIN="$(env_value DOMAIN_CLIENT)"
DEMO_USERNAME="${DEMO_EMAIL%%@*}"

login_code() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
    -X POST "http://localhost:3080/api/auth/login" \
    -H "origin: ${DOMAIN:-http://localhost:3080}" -H 'content-type: application/json' \
    -d "{\"email\":\"$DEMO_EMAIL\",\"password\":\"$DEMO_PASS\"}" 2>/dev/null || echo 000
}

wait_for_stack() { for _ in $(seq 1 30); do [ "$(login_code)" != "000" ] && return 0; sleep 3; done; }

if [ -n "$DEMO_EMAIL" ] && [ -n "$DEMO_PASS" ]; then
  wait_for_stack
  if [ "$(login_code)" = "200" ]; then
    echo "$DEMO_EMAIL can already sign in"
  else
    echo "creating $DEMO_EMAIL"
    # --email-verified=true is not optional: without it the CLI asks a question, and
    # with no terminal attached (exec -T) it waits for an answer forever.
    timeout 150 $DOCKER compose exec -T librechat npm run create-user -- \
      "$DEMO_EMAIL" "${DEMO_NAME:-Demo User}" "$DEMO_USERNAME" "$DEMO_PASS" \
      --email-verified=true </dev/null >/dev/null 2>&1 || true

    if [ "$(login_code)" != "200" ]; then
      echo "that did not take; using the registration endpoint instead"
      sed -i 's/^ALLOW_REGISTRATION=.*/ALLOW_REGISTRATION=true/' .env
      $DOCKER compose up -d librechat >/dev/null 2>&1 || true
      wait_for_stack
      curl -s --max-time 30 -o /dev/null -X POST "http://localhost:3080/api/auth/register" \
        -H "origin: ${DOMAIN:-http://localhost:3080}" -H 'content-type: application/json' \
        -d "{\"email\":\"$DEMO_EMAIL\",\"password\":\"$DEMO_PASS\",\"confirm_password\":\"$DEMO_PASS\",\"name\":\"${DEMO_NAME:-Demo User}\",\"username\":\"$DEMO_USERNAME\"}" || true
      sed -i 's/^ALLOW_REGISTRATION=.*/ALLOW_REGISTRATION=false/' .env
      $DOCKER compose up -d librechat >/dev/null 2>&1 || true
      wait_for_stack
    fi

    if [ "$(login_code)" = "200" ]; then
      echo "demo account ready"
    else
      echo "could not create the demo account; the UI will show the login page until it exists"
    fi
  fi
  $DOCKER compose restart gateway >/dev/null 2>&1 || true
fi

log "Checking the agent core"
python3 scripts/smoke_test.py --url http://localhost:8088 || true

cat <<EOF

Done.

  UI:       http://${IP:-<vm-ip>}:3080      (open tcp:3080 in the cloud firewall first)
  Console:  http://localhost:8088           (bound to localhost only - reach it over an SSH tunnel)

Visitors land straight in a chat as the demo user - no login page.

Useful:
  Logs:     $DOCKER compose logs -f gateway   (or librechat, agent-core)
  Restart:  $DOCKER compose up -d             (applies .env changes)
  Stop:     $DOCKER compose down
  Normal login page instead of auto sign-in:
       sed -i 's/^GATEWAY_AUTO_LOGIN=.*/GATEWAY_AUTO_LOGIN=false/' .env && $DOCKER compose up -d
EOF
