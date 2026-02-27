"""Shared test fixtures for aumai-datasynthesizer."""
from __future__ import annotations

import pytest
from faker import Faker

from aumai_datasynthesizer.core import DataGenerator, SchemaBasedGenerator
from aumai_datasynthesizer.models import DataType, GeneratorConfig


@pytest.fixture()
def faker_seeded() -> Faker:
    Faker.seed(42)
    return Faker()


@pytest.fixture()
def generator() -> DataGenerator:
    return DataGenerator()


@pytest.fixture()
def schema_gen(faker_seeded: Faker) -> SchemaBasedGenerator:
    return SchemaBasedGenerator(faker_seeded)


@pytest.fixture()
def text_config() -> GeneratorConfig:
    return GeneratorConfig(data_type=DataType.text, count=5, seed=42)


@pytest.fixture()
def conversation_config() -> GeneratorConfig:
    return GeneratorConfig(data_type=DataType.conversation, count=3, seed=42)


@pytest.fixture()
def tool_call_config() -> GeneratorConfig:
    return GeneratorConfig(data_type=DataType.tool_call, count=4, seed=42)


@pytest.fixture()
def agent_trace_config() -> GeneratorConfig:
    return GeneratorConfig(data_type=DataType.agent_trace, count=3, seed=42)


@pytest.fixture()
def json_config() -> GeneratorConfig:
    return GeneratorConfig(data_type=DataType.json, count=5, seed=42)
