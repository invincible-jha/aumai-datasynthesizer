"""Core data-generation logic for aumai-datasynthesizer."""

from __future__ import annotations

import copy
import json
import random
import re
import time
import uuid
from typing import Any

from faker import Faker

from aumai_datasynthesizer.models import (
    ConversationTurn,
    DataType,
    GeneratorConfig,
    SyntheticDataset,
)
from aumai_datasynthesizer.templates import CONVERSATION_TEMPLATES, TOOL_CALL_TEMPLATES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKER_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Mapping from template placeholder names to Faker methods.
_FAKER_ATTR_MAP: dict[str, str] = {
    "order_id": "numerify",
    "email": "email",
    "first_name": "first_name",
    "last_name": "last_name",
    "date": "future_date",
    "task": "bs",
    "function_name": "pystr",
    "paper_title": "sentence",
    "year": "year",
    "finding_one": "sentence",
    "finding_two": "sentence",
    "finding_three": "sentence",
    "conclusion": "sentence",
    "methodology": "word",
    "count": "random_int",
    "duration": "word",
    "analysis_method": "word",
}


def _resolve_placeholder(name: str, faker: Faker) -> str:
    """Return a Faker-generated string for a template placeholder name."""
    attr = _FAKER_ATTR_MAP.get(name)
    if attr is None:
        # Fall back to a random word for unknown placeholders.
        return str(faker.word())
    method = getattr(faker, attr, None)
    if callable(method):
        result = method() if attr != "numerify" else method("######")
        return str(result)
    return str(faker.word())


def _render_template(text: str, faker: Faker) -> str:
    """Replace all {placeholder} tokens in *text* with Faker-generated values."""

    def replace(match: re.Match[str]) -> str:
        return _resolve_placeholder(match.group(1), faker)

    return _FAKER_PLACEHOLDER_RE.sub(replace, text)


# ---------------------------------------------------------------------------
# SchemaBasedGenerator
# ---------------------------------------------------------------------------


class SchemaBasedGenerator:
    """Generate dicts that conform to a JSON Schema (subset: object/string/integer/boolean/array).

    This is a lightweight generator — it does not depend on any third-party
    JSON Schema library so it stays dependency-free while still being
    practically useful for common schemas.
    """

    def __init__(self, faker: Faker) -> None:
        self._faker = faker

    def from_schema(self, schema: dict[str, object], count: int) -> list[dict[str, object]]:
        """Return *count* dicts matching *schema*."""
        return [self._generate_object(schema) for _ in range(count)]

    def _generate_value(self, schema: dict[str, object]) -> Any:  # noqa: ANN401
        schema_type = schema.get("type", "string")
        if schema_type == "object":
            return self._generate_object(schema)
        if schema_type == "string":
            enum = schema.get("enum")
            if isinstance(enum, list) and enum:
                return self._faker.random_element(enum)
            fmt = schema.get("format", "")
            if fmt == "email":
                return self._faker.email()
            if fmt == "date":
                return str(self._faker.date())
            if fmt == "uri":
                return self._faker.url()
            if fmt == "uuid":
                return str(uuid.uuid4())
            return self._faker.sentence(nb_words=4).rstrip(".")
        if schema_type == "integer":
            minimum = int(schema.get("minimum", 0))  # type: ignore[arg-type]
            maximum = int(schema.get("maximum", 1000))  # type: ignore[arg-type]
            return self._faker.random_int(min=minimum, max=maximum)
        if schema_type == "number":
            minimum = float(schema.get("minimum", 0.0))  # type: ignore[arg-type]
            maximum = float(schema.get("maximum", 1.0))  # type: ignore[arg-type]
            return round(random.uniform(minimum, maximum), 4)
        if schema_type == "boolean":
            return self._faker.boolean()
        if schema_type == "array":
            items_schema = schema.get("items", {"type": "string"})
            min_items = int(schema.get("minItems", 1))  # type: ignore[arg-type]
            max_items = int(schema.get("maxItems", 5))  # type: ignore[arg-type]
            n = self._faker.random_int(min=min_items, max=max_items)
            return [self._generate_value(items_schema) for _ in range(n)]  # type: ignore[arg-type]
        if schema_type == "null":
            return None
        return self._faker.word()

    def _generate_object(self, schema: dict[str, object]) -> dict[str, object]:
        properties: dict[str, object] = schema.get("properties", {})  # type: ignore[assignment]
        required: list[str] = schema.get("required", [])  # type: ignore[assignment]
        result: dict[str, object] = {}
        for prop_name, prop_schema in properties.items():  # type: ignore[union-attr]
            if prop_name in required or self._faker.boolean(chance_of_getting_true=80):
                result[prop_name] = self._generate_value(prop_schema)  # type: ignore[arg-type]
        return result


# ---------------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------------


class DataGenerator:
    """Main dispatcher that generates synthetic datasets for all DataType values."""

    def __init__(self, faker: Faker | None = None) -> None:
        self._faker_default = faker  # will be overridden per-call if seed is set

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_faker(self, config: GeneratorConfig) -> Faker:
        faker = self._faker_default if self._faker_default is not None else Faker()
        if config.seed is not None:
            Faker.seed(config.seed)
            random.seed(config.seed)
        return faker

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def generate_text(self, config: GeneratorConfig) -> list[str]:
        """Return a list of random text strings."""
        faker = self._make_faker(config)
        max_sentences: int = int(config.constraints.get("max_sentences", 5))
        min_sentences: int = int(config.constraints.get("min_sentences", 1))
        texts: list[str] = []
        for _ in range(config.count):
            nb = faker.random_int(min=min_sentences, max=max_sentences)
            texts.append(faker.paragraph(nb_sentences=nb))
        return texts

    def generate_conversations(
        self, config: GeneratorConfig
    ) -> list[list[ConversationTurn]]:
        """Return a list of multi-turn conversations."""
        faker = self._make_faker(config)
        template_name: str = str(
            config.constraints.get("template", "customer_support")
        )
        template_turns: list[dict[str, object]] = CONVERSATION_TEMPLATES.get(
            template_name, CONVERSATION_TEMPLATES["customer_support"]
        )
        conversations: list[list[ConversationTurn]] = []
        for _ in range(config.count):
            turns: list[ConversationTurn] = []
            for turn_dict in template_turns:
                content = _render_template(str(turn_dict["content"]), faker)
                turns.append(
                    ConversationTurn(
                        role=str(turn_dict["role"]),
                        content=content,
                        tool_calls=None,
                    )
                )
            conversations.append(turns)
        return conversations

    def generate_tool_calls(self, config: GeneratorConfig) -> list[dict[str, object]]:
        """Return a list of synthetic tool-call dicts."""
        faker = self._make_faker(config)
        tool_name: str = str(config.constraints.get("tool", "search"))
        template = copy.deepcopy(
            TOOL_CALL_TEMPLATES.get(tool_name, TOOL_CALL_TEMPLATES["search"])
        )
        schema_gen = SchemaBasedGenerator(faker)
        params_schema: dict[str, object] = template.get(  # type: ignore[assignment]
            "parameters", {}
        )
        results: list[dict[str, object]] = []
        for _ in range(config.count):
            arguments = schema_gen._generate_object(params_schema)  # noqa: SLF001
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": template["name"],
                        "arguments": json.dumps(arguments),
                    },
                }
            )
        return results

    def generate_agent_traces(
        self, config: GeneratorConfig
    ) -> list[dict[str, object]]:
        """Return a list of synthetic agent execution traces."""
        faker = self._make_faker(config)
        tool_keys = list(TOOL_CALL_TEMPLATES.keys())
        traces: list[dict[str, object]] = []
        for _ in range(config.count):
            num_steps = faker.random_int(min=2, max=6)
            steps: list[dict[str, object]] = []
            t = time.time()
            for step_idx in range(num_steps):
                step_type = faker.random_element(["thought", "tool_call", "observation"])
                step: dict[str, object] = {
                    "step": step_idx,
                    "type": step_type,
                    "timestamp": t + step_idx * faker.pyfloat(min_value=0.1, max_value=2.0),
                }
                if step_type == "thought":
                    step["content"] = faker.sentence()
                elif step_type == "tool_call":
                    tool_key = faker.random_element(tool_keys)
                    step["tool"] = TOOL_CALL_TEMPLATES[tool_key]["name"]
                    step["arguments"] = {"query": faker.sentence(nb_words=4)}
                else:
                    step["content"] = faker.paragraph(nb_sentences=2)
                steps.append(step)
            traces.append(
                {
                    "trace_id": str(uuid.uuid4()),
                    "agent": faker.word() + "_agent",
                    "task": faker.sentence(nb_words=6),
                    "steps": steps,
                    "final_answer": faker.paragraph(nb_sentences=1),
                    "success": faker.boolean(chance_of_getting_true=80),
                }
            )
        return traces

    def generate_json(self, config: GeneratorConfig) -> list[dict[str, object]]:
        """Return JSON objects — either schema-driven or free-form."""
        faker = self._make_faker(config)
        if config.schema:
            schema_gen = SchemaBasedGenerator(faker)
            return schema_gen.from_schema(config.schema, config.count)
        # Free-form random JSON objects.
        results: list[dict[str, object]] = []
        for _ in range(config.count):
            results.append(
                {
                    "id": str(uuid.uuid4()),
                    "name": faker.name(),
                    "email": faker.email(),
                    "value": faker.pyfloat(min_value=0, max_value=1000, right_digits=2),
                    "active": faker.boolean(),
                    "tags": [faker.word() for _ in range(faker.random_int(min=1, max=5))],
                    "created_at": str(faker.date_time_this_year()),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def generate(self, config: GeneratorConfig) -> SyntheticDataset:
        """Generate a full SyntheticDataset for the given config."""
        start = time.perf_counter()

        if config.data_type == DataType.text:
            raw = self.generate_text(config)
            samples: list[dict[str, object]] = [
                {"index": i, "text": t} for i, t in enumerate(raw)
            ]

        elif config.data_type == DataType.conversation:
            raw_convs = self.generate_conversations(config)
            samples = [
                {
                    "index": i,
                    "turns": [turn.model_dump() for turn in turns],
                }
                for i, turns in enumerate(raw_convs)
            ]

        elif config.data_type == DataType.tool_call:
            raw_calls = self.generate_tool_calls(config)
            samples = [{"index": i, **call} for i, call in enumerate(raw_calls)]

        elif config.data_type == DataType.agent_trace:
            raw_traces = self.generate_agent_traces(config)
            samples = [{"index": i, **trace} for i, trace in enumerate(raw_traces)]

        else:  # DataType.json
            raw_json = self.generate_json(config)
            samples = [{"index": i, **obj} for i, obj in enumerate(raw_json)]

        elapsed_ms = (time.perf_counter() - start) * 1000
        return SyntheticDataset(
            config=config,
            samples=samples,
            metadata={
                "generated_count": len(samples),
                "generation_time_ms": round(elapsed_ms, 2),
                "data_type": config.data_type.value,
            },
        )


__all__ = ["DataGenerator", "SchemaBasedGenerator"]
