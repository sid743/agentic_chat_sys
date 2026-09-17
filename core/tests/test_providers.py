"""Checks on the shipped providers.yaml: keys come from the environment, ids
split the way the rest of the code expects, and nothing is enabled by accident."""

from __future__ import annotations

import pytest

from agentic_core.llm.base import LLMError
from agentic_core.llm.registry import ModelRegistry

from .conftest import make_settings


def registry(tmp_path, **env) -> ModelRegistry:
    return ModelRegistry(make_settings(tmp_path), env=env)


def test_hosted_providers_need_a_key(tmp_path):
    reg = registry(tmp_path)
    for name in ("groq", "gemini", "openai", "openrouter"):
        assert not reg.providers[name].enabled, f"{name} should stay off without a key"
    assert reg.providers["mock"].enabled


def test_user_provided_is_not_a_key(tmp_path):
    # LibreChat's placeholder means "each user brings their own"; the agents cannot use it.
    reg = registry(tmp_path, GEMINI_API_KEY="user_provided", GROQ_API_KEY="user_provided")
    assert not reg.providers["gemini"].enabled
    assert not reg.providers["groq"].enabled


def test_gemini_uses_the_openai_compatible_endpoint(tmp_path):
    reg = registry(tmp_path, GEMINI_API_KEY="test-key")
    provider, model = reg.split("gemini/gemini-3.1-flash-lite")
    assert provider.enabled
    assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert provider.tool_mode == "native"
    assert model == "gemini-3.1-flash-lite"


def test_gemini_base_url_is_overridable(tmp_path):
    reg = registry(tmp_path, GEMINI_API_KEY="k", GEMINI_BASE_URL="http://localhost:9/v1")
    assert reg.providers["gemini"].base_url == "http://localhost:9/v1"


def test_provider_model_filters(tmp_path):
    reg = registry(tmp_path, GEMINI_API_KEY="k", GROQ_API_KEY="k")
    gemini, groq = reg.providers["gemini"], reg.providers["groq"]
    assert gemini.keep("gemini-3.1-flash-lite") and gemini.keep("models/gemini-3.5-flash")
    assert not gemini.keep("text-embedding-004") and not gemini.keep("imagen-4.0")
    # Groq ids carry their own slash; speech and guard models stay out of the menu.
    assert groq.keep("qwen/qwen3.8-27b") and groq.keep("openai/gpt-oss-120b")
    assert not groq.keep("whisper-large-v3") and not groq.keep("meta-llama/llama-prompt-guard-2-22m")


def test_bare_model_id_resolves_when_it_is_a_known_model(tmp_path):
    reg = registry(tmp_path, GEMINI_API_KEY="k", GROQ_API_KEY="k")
    # A listed model id works without the provider prefix, which is what people type.
    assert reg.split("gemini-3.1-flash-lite")[0].name == "gemini"
    assert reg.split("qwen/qwen3.8-27b")[0].name == "groq"
    with pytest.raises(LLMError) as exc:
        reg.split("something-nobody-serves")
    assert "gemini" in str(exc.value)  # the error lists the providers


@pytest.mark.asyncio
async def test_default_model_follows_priority(tmp_path):
    reg = registry(tmp_path, GEMINI_API_KEY="k")
    reg.settings.agent_default_model = ""
    assert await reg.default_model_id() == "gemini/gemini-3.1-flash-lite"
    assert await registry(tmp_path).default_model_id() == "mock/hr-demo"
