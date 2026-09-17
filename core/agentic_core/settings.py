"""Runtime settings, read from environment variables and the shared `.env` file.

The same `.env` at the repository root is used by LibreChat and by this service,
so provider keys (GROQ_API_KEY, OPENAI_API_KEY, ...) only need to be set once.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CORE_DIR = Path(__file__).resolve().parent.parent  # .../core
REPO_DIR = CORE_DIR.parent  # .../agenticsys


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_DIR / ".env"), str(CORE_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- service -----------------------------------------------------------
    agent_core_host: str = "0.0.0.0"
    agent_core_port: int = 8088
    # Bearer token LibreChat must send. Empty = no auth (local dev only).
    agent_core_api_key: str = ""
    # Admin pages/APIs use the same key unless this is set to false.
    admin_auth_required: bool = True
    log_level: str = "INFO"
    log_json: bool = False

    # --- models ------------------------------------------------------------
    # "<provider>/<model>", e.g. groq/qwen/qwen3.8-27b. Empty = auto-detect.
    agent_default_model: str = ""
    # Optional separate (usually small/fast) model for routing decisions.
    agent_router_model: str = ""
    providers_file: Path = CORE_DIR / "config" / "providers.yaml"
    agents_file: Path = CORE_DIR / "config" / "agents.yaml"
    llm_timeout_seconds: float = 90.0
    llm_temperature: float = 0.2

    # --- storage -----------------------------------------------------------
    data_dir: Path = CORE_DIR / "var"
    database_url: str = ""  # default: sqlite file in data_dir
    qdrant_url: str = ""  # empty = embedded Qdrant (local files)
    qdrant_api_key: str = ""
    qdrant_path: str = ""  # default: data_dir/qdrant ; ":memory:" for tests

    # --- retrieval ---------------------------------------------------------
    # hash (offline, default) | fastembed (local ONNX) | openai (any OpenAI-compatible /embeddings)
    embeddings_provider: str = "hash"
    embeddings_model: str = ""
    embeddings_base_url: str = ""
    embeddings_api_key: str = ""
    embeddings_dim: int = 768  # only used by the hash embedder
    policies_dir: Path = CORE_DIR / "data" / "policies"
    rag_top_k: int = 5
    rag_min_score: float = 0.0

    # --- demo identity -----------------------------------------------------
    # Fixed "today" for reproducible demos (YYYY-MM-DD). Empty = real date.
    demo_today: date | None = None
    default_employee_id: str = "E1001"
    # "you@company.com=E1001;boss@company.com=E1005"
    user_employee_map: str = ""
    allow_act_as: bool = True

    # --- agent behaviour ---------------------------------------------------
    max_agent_steps: int = 6
    max_route_agents: int = 3
    history_turns: int = 6
    # How the agent trace is streamed: reasoning_content | reasoning | think | none
    reasoning_field: str = "reasoning_content"
    # minimal | normal | debug
    trace_level: str = "normal"
    answer_footer: bool = True
    auto_seed: bool = True

    @field_validator("demo_today", mode="before")
    @classmethod
    def _empty_date(cls, value):  # noqa: D401 - pydantic hook
        if value in ("", None):
            return None
        return value

    # --- derived -----------------------------------------------------------
    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'hr_demo.db').as_posix()}"

    @property
    def resolved_qdrant_path(self) -> str:
        if self.qdrant_path:
            return self.qdrant_path
        return str(self.data_dir / "qdrant")

    def today(self) -> date:
        return self.demo_today or date.today()

    def user_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for pair in self.user_employee_map.replace(",", ";").split(";"):
            if "=" in pair:
                email, emp = pair.split("=", 1)
                if email.strip() and emp.strip():
                    mapping[email.strip().lower()] = emp.strip().upper()
        return mapping


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
