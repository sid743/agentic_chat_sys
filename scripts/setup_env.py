#!/usr/bin/env python3
"""Create (or complete) the shared .env file.

- copies .env.example to .env if .env does not exist
- fills in empty secrets with strong random values (never overwrites existing ones)
- optionally sets provider keys / defaults from command-line flags

Usage (Windows PowerShell, macOS or Linux):
    python scripts/setup_env.py
    python scripts/setup_env.py --groq-key gsk_... --default-model groq/qwen/qwen3.8-27b --map you@company.com=E1001
"""

from __future__ import annotations

import argparse
import re
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".env.example"
TARGET = ROOT / ".env"

GENERATED = {
    "CREDS_KEY": lambda: secrets.token_hex(32),  # 64 hex chars (32 bytes)
    "CREDS_IV": lambda: secrets.token_hex(16),  # 32 hex chars (16 bytes)
    "JWT_SECRET": lambda: secrets.token_hex(32),
    "JWT_REFRESH_SECRET": lambda: secrets.token_hex(32),
    "MEILI_MASTER_KEY": lambda: secrets.token_hex(16),
    "AGENT_CORE_API_KEY": lambda: "ac-" + secrets.token_urlsafe(24),
}


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def set_value(lines: list[str], key: str, value: str, only_if_empty: bool) -> bool:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=(.*)$")
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            current = match.group(1).strip()
            if only_if_empty and current:
                return False
            lines[i] = f"{key}={value}"
            return True
    lines.append(f"{key}={value}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groq-key")
    parser.add_argument("--openai-key")
    parser.add_argument("--openrouter-key")
    parser.add_argument("--gemini-key")
    parser.add_argument("--default-model", help="e.g. groq/qwen/qwen3.8-27b or ollama/llama3.1:8b")
    parser.add_argument("--router-model", help="optional fast routing model, e.g. groq/openai/gpt-oss-20b")
    parser.add_argument("--map", action="append", default=[], help="email=E1001 (repeatable)")
    parser.add_argument("--real-date", action="store_true", help="use today's date instead of the fixed demo date")
    args = parser.parse_args()

    if not EXAMPLE.exists():
        print(f"Missing {EXAMPLE}", file=sys.stderr)
        return 1
    created = False
    if not TARGET.exists():
        shutil.copyfile(EXAMPLE, TARGET)
        created = True

    lines = read_lines(TARGET)
    filled = [key for key, make in GENERATED.items() if set_value(lines, key, make(), only_if_empty=True)]

    explicit = {
        "GROQ_API_KEY": args.groq_key,
        "OPENAI_API_KEY": args.openai_key,
        "OPENROUTER_API_KEY": args.openrouter_key,
        "GEMINI_API_KEY": args.gemini_key,
        "AGENT_DEFAULT_MODEL": args.default_model,
        "AGENT_ROUTER_MODEL": args.router_model,
    }
    for key, value in explicit.items():
        if value:
            set_value(lines, key, value, only_if_empty=False)
    if args.map:
        set_value(lines, "USER_EMPLOYEE_MAP", ";".join(args.map), only_if_empty=False)
    if args.real_date:
        set_value(lines, "DEMO_TODAY", "", only_if_empty=False)

    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(("Created" if created else "Updated") + f" {TARGET}")
    if filled:
        print("Generated secrets: " + ", ".join(filled))
    print(
        "\nNext steps:\n"
        "  1. Edit .env: add GROQ_API_KEY / OPENAI_API_KEY and/or start Ollama or LM Studio,\n"
        "     set AGENT_DEFAULT_MODEL, and map your login email in USER_EMPLOYEE_MAP.\n"
        "  2. docker compose up -d --build\n"
        "  3. LibreChat: http://localhost:3080   Agent console: http://localhost:8088\n"
        f"     (console API key: see AGENT_CORE_API_KEY in {TARGET.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
