"""Pydantic v2 models for aumai-datasynthesizer."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DataType(str, Enum):
    """Enumeration of supported synthetic data types."""

    text = "text"
    json = "json"
    conversation = "conversation"
    tool_call = "tool_call"
    agent_trace = "agent_trace"


class GeneratorConfig(BaseModel):
    """Configuration for a synthetic data generation run."""

    data_type: DataType
    count: int = Field(gt=0, default=10)
    seed: int | None = None
    schema: dict[str, object] | None = None
    constraints: dict[str, object] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    """A single turn within a multi-turn conversation."""

    role: str
    content: str
    tool_calls: list[dict[str, object]] | None = None


class SyntheticDataset(BaseModel):
    """The output of a generation run: config, samples, and metadata."""

    config: GeneratorConfig
    samples: list[dict[str, object]]
    metadata: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "DataType",
    "GeneratorConfig",
    "ConversationTurn",
    "SyntheticDataset",
]
