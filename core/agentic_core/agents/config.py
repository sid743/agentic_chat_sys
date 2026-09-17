"""Loads config/agents.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..tools import REGISTRY


@dataclass
class AgentSpec:
    id: str
    name: str
    description: str
    tools: list[str]
    system_prompt: str
    model: str | None = None
    max_steps: int | None = None
    temperature: float | None = None


@dataclass
class OrchestratorSpec:
    name: str = "Orchestrator"
    model: str | None = None
    max_agents_per_turn: int = 3
    synthesize: str = "auto"
    auto_notify_agent: str | None = None
    router_prompt: str = ""
    synthesis_prompt: str = ""
    direct_prompt: str = ""


@dataclass
class AgentsConfig:
    orchestrator: OrchestratorSpec
    agents: dict[str, AgentSpec] = field(default_factory=dict)
    common_rules: str = ""

    def catalog(self) -> str:
        return "\n".join(f"- {a.id} ({a.name}): {a.description}" for a in self.agents.values())


def load_agents_config(path: str | Path) -> AgentsConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    orch = OrchestratorSpec(**(raw.get("orchestrator") or {}))
    agents: dict[str, AgentSpec] = {}
    for item in raw.get("agents") or []:
        spec = AgentSpec(**item)
        unknown = [t for t in spec.tools if t not in REGISTRY]
        if unknown:
            raise ValueError(f"Agent '{spec.id}' references unknown tools: {unknown}")
        agents[spec.id] = spec
    if not agents:
        raise ValueError("agents.yaml defines no agents")
    if orch.auto_notify_agent and orch.auto_notify_agent not in agents:
        orch.auto_notify_agent = None
    return AgentsConfig(orchestrator=orch, agents=agents, common_rules=raw.get("common_rules") or "")
