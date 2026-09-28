"""Loads providers.yaml and turns model ids into ChatModel instances."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..settings import Settings
from .base import ChatModel, LLMError
from .cortexa import CortexaChatModel
from .mock import MockChatModel
from .openai_compat import OpenAICompatModel
from .tool_emulation import EmulatedToolsModel

log = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    env = env if env is not None else os.environ
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: env.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    return value


@dataclass
class ProviderConfig:
    name: str
    label: str = ""
    type: str = "openai"
    base_url: str = ""
    api_key: str = ""
    app_id: str = ""  # Cortexa: App ID (api_key holds the client secret)
    max_retries: int = 2
    local: bool = False
    fetch_models: bool = False
    models: list[str] = field(default_factory=list)
    model_include: str | None = None
    model_exclude: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    tool_mode: str = "native"  # native | json | emulated | delegate (see llm/tool_emulation.py)
    tool_planner: str = ""  # delegate: the model with native tool calling that picks the tools
    tool_repairs: int = 2  # emulated: how often a malformed tool call is sent back to be fixed
    json_mode: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def has_key(self) -> bool:
        # LibreChat's "user_provided" means each user types a key in the UI; the agent
        # core cannot use that, so it counts as "no key" here.
        return bool(self.api_key) and self.api_key.strip().lower() != "user_provided"

    @property
    def enabled(self) -> bool:
        if self.type == "mock":
            return True
        if not self.base_url:
            return False
        if self.type == "cortexa":
            return self.has_key and bool(self.app_id)
        return self.local or self.has_key

    def keep(self, model_id: str) -> bool:
        if self.model_include and not re.search(self.model_include, model_id, re.IGNORECASE):
            return False
        if self.model_exclude and re.search(self.model_exclude, model_id, re.IGNORECASE):
            return False
        return True


@dataclass
class ModelInfo:
    id: str
    provider: str
    model: str
    label: str
    local: bool = False


class ModelRegistry:
    def __init__(
        self,
        settings: Settings,
        providers_file: Path | None = None,
        env: dict[str, str] | None = None,
        http_client_factory: Any = None,
    ):
        self.settings = settings
        self.http_client_factory = http_client_factory  # tests inject a fake transport here
        self.cortexa_client_factory: Any = None  # tests inject a fake Cortexa client here
        env_map = dict(os.environ)
        env_map.update(_dotenv_values(settings))
        if env:
            env_map.update(env)
        raw = yaml.safe_load(Path(providers_file or settings.providers_file).read_text(encoding="utf-8")) or {}
        raw = expand_env(raw, env_map)
        self.providers: dict[str, ProviderConfig] = {}
        for name, spec in (raw.get("providers") or {}).items():
            spec = dict(spec or {})
            spec["models"] = [str(m) for m in (spec.get("models") or [])]
            known = {k: v for k, v in spec.items() if k in ProviderConfig.__dataclass_fields__}
            self.providers[name] = ProviderConfig(name=name, **known)
        self.priority: list[str] = raw.get("default_priority") or list(self.providers)
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._instances: dict[str, ChatModel] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ discovery
    async def _fetch(self, provider: ProviderConfig) -> list[str] | None:
        """Return the provider's live model list, or None if unreachable."""
        cached = self._cache.get(provider.name)
        ttl = 30 if provider.local else 300
        if cached and time.monotonic() - cached[0] < ttl:
            return cached[1]
        url = provider.base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {provider.api_key or 'none'}", **provider.headers}
        try:
            if self.http_client_factory:
                client = self.http_client_factory(provider)
                resp = await client.get(url, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=1.5 if provider.local else 6.0) as client:
                    resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # Gemini lists its models as "models/gemini-3.1-flash-lite"; the API accepts
            # the bare name, and "gemini/models/gemini-..." reads badly in a model menu.
            ids = sorted({str(item.get("id")).removeprefix("models/") for item in data if item.get("id")})
            ids = [i for i in ids if provider.keep(i)]
        except Exception as exc:  # noqa: BLE001 - discovery is best effort
            log.info("Model discovery failed for %s: %s", provider.name, exc)
            ids = None
        self._cache[provider.name] = (time.monotonic(), ids)  # type: ignore[arg-type]
        return ids

    async def available_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []
        for provider in self.providers.values():
            if not provider.enabled:
                continue
            names = list(provider.models)
            if provider.fetch_models and provider.type != "mock":
                live = await self._fetch(provider)
                if live is None:
                    if provider.local:
                        continue  # local server is not running - hide it
                elif live:
                    names = live
            for name in names:
                models.append(
                    ModelInfo(
                        id=f"{provider.name}/{name}",
                        provider=provider.name,
                        model=name,
                        label=provider.label or provider.name,
                        local=provider.local,
                    )
                )
        return models

    async def default_model_id(self) -> str:
        if self.settings.agent_default_model and self.settings.agent_default_model != "auto":
            return self.settings.agent_default_model
        for name in self.priority:
            provider = self.providers.get(name)
            if not provider or not provider.enabled:
                continue
            if provider.type == "mock":
                return f"{name}/{provider.models[0] if provider.models else 'hr-demo'}"
            if provider.local:
                live = await self._fetch(provider)
                if live:
                    return f"{name}/{live[0]}"
                continue
            if provider.models:
                return f"{name}/{provider.models[0]}"
        return "mock/hr-demo"

    # ------------------------------------------------------------------ resolution
    def split(self, model_id: str) -> tuple[ProviderConfig, str]:
        if "/" in model_id:
            prefix, rest = model_id.split("/", 1)
            if prefix in self.providers:
                return self.providers[prefix], rest
        for provider in self.providers.values():
            if model_id in provider.models and provider.enabled:
                return provider, model_id
        raise LLMError(
            f"Unknown model '{model_id}'. Use '<provider>/<model>' with one of: {', '.join(self.providers)}."
        )

    async def _planner_for(self, provider: ProviderConfig) -> ChatModel:
        planner_id = (provider.tool_planner or "").strip()
        if not planner_id:
            raise LLMError(
                f"Provider '{provider.name}' uses tool_mode 'delegate' but has no tool_planner "
                "(e.g. CORTEXA_TOOL_PLANNER=gemini/gemini-3.1-flash-lite)."
            )
        planner_provider = self.split(planner_id)[0]
        if planner_provider.name == provider.name:
            raise LLMError(f"The tool_planner for '{provider.name}' must be a model from another provider.")
        # Only a provider with native tool calling can plan. This also rules out chains
        # and loops (a -> b -> a), since a native provider never delegates.
        if planner_provider.tool_mode != "native" or planner_provider.type in ("cortexa", "mock"):
            raise LLMError(
                f"The tool_planner '{planner_id}' needs native tool calling "
                f"(its provider uses tool_mode '{planner_provider.tool_mode}')."
            )
        planner = await self.resolve(planner_id)
        if not planner.native_tools:
            raise LLMError(f"The tool_planner '{planner_id}' has no native tool calling.")
        return planner

    async def resolve(self, model_id: str | None) -> ChatModel:
        if not model_id or model_id in ("auto", "default"):
            model_id = await self.default_model_id()
        if model_id in self._instances:
            return self._instances[model_id]
        provider, name = self.split(model_id)
        # Resolved before taking the lock: resolving the planner takes it too.
        planner = await self._planner_for(provider) if provider.enabled and provider.tool_mode == "delegate" else None
        async with self._lock:
            if model_id in self._instances:
                return self._instances[model_id]
            if not provider.enabled:
                if provider.type == "cortexa":
                    missing = "CORTEXA_APP_ID, CORTEXA_CLIENT_SECRET or CORTEXA_CORE_BASE_URL"
                else:
                    missing = "API key" if not provider.local else "base URL"
                raise LLMError(f"Provider '{provider.name}' is not configured (missing {missing} in .env).")
            if provider.type == "mock":
                instance: ChatModel = MockChatModel(model=name)
            elif provider.type == "cortexa":
                instance = CortexaChatModel(
                    provider=provider.name,
                    model=name,
                    app_id=provider.app_id,
                    client_secret=provider.api_key,
                    base_url=provider.base_url,
                    timeout=self.settings.llm_timeout_seconds,
                    max_retries=provider.max_retries,
                    client_factory=self.cortexa_client_factory,
                )
            else:
                instance = OpenAICompatModel(
                    provider=provider.name,
                    model=name,
                    base_url=provider.base_url,
                    api_key=provider.api_key,
                    timeout=self.settings.llm_timeout_seconds,
                    headers=provider.headers,
                    native_tools=provider.tool_mode == "native",
                    json_mode=provider.json_mode,
                    default_temperature=self.settings.llm_temperature,
                    extra_body=provider.extra_body,
                    http_client=self.http_client_factory(provider) if self.http_client_factory else None,
                )
            if provider.tool_mode in ("emulated", "delegate"):
                # Text-only model: give it the native tool-calling interface.
                instance = EmulatedToolsModel(instance, max_repairs=provider.tool_repairs, planner=planner)
            self._instances[model_id] = instance
            return instance

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": p.name,
                "label": p.label,
                "type": p.type,
                "enabled": p.enabled,
                "local": p.local,
                "base_url": p.base_url,
                "has_api_key": p.has_key and not p.local,
                "tool_mode": p.tool_mode,
                **({"tool_planner": p.tool_planner} if p.tool_mode == "delegate" else {}),
            }
            for p in self.providers.values()
        ]


def _dotenv_values(settings: Settings) -> dict[str, str]:
    """Values from the shared .env files (os.environ still wins)."""
    values: dict[str, str] = {}
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover
        return values
    for path in settings.model_config.get("env_file") or ():
        if path and Path(path).exists():
            for key, value in dotenv_values(path).items():
                if value is not None and key not in os.environ:
                    values[key] = value
    return values
