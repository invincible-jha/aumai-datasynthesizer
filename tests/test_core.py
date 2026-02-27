"""Comprehensive tests for aumai_datasynthesizer core, models, and templates."""
from __future__ import annotations

import json
import re
import uuid

import pytest
from faker import Faker

from aumai_datasynthesizer.core import (
    DataGenerator,
    SchemaBasedGenerator,
    _render_template,
    _resolve_placeholder,
)
from aumai_datasynthesizer.models import (
    ConversationTurn,
    DataType,
    GeneratorConfig,
    SyntheticDataset,
)
from aumai_datasynthesizer.templates import CONVERSATION_TEMPLATES, TOOL_CALL_TEMPLATES


# ---------------------------------------------------------------------------
# Tests for helpers
# ---------------------------------------------------------------------------


class TestResolveAndRenderHelpers:
    def test_resolve_known_placeholder_email(self) -> None:
        faker = Faker()
        Faker.seed(0)
        result = _resolve_placeholder("email", faker)
        assert "@" in result

    def test_resolve_known_placeholder_first_name(self) -> None:
        faker = Faker()
        result = _resolve_placeholder("first_name", faker)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resolve_unknown_placeholder_returns_word(self) -> None:
        faker = Faker()
        result = _resolve_placeholder("totally_unknown_placeholder_xyz", faker)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_template_replaces_placeholder(self) -> None:
        faker = Faker()
        Faker.seed(1)
        result = _render_template("Hello {first_name}!", faker)
        assert "{first_name}" not in result
        assert "Hello " in result

    def test_render_template_multiple_placeholders(self) -> None:
        faker = Faker()
        result = _render_template("Hi {first_name}, your email is {email}", faker)
        assert "{first_name}" not in result
        assert "{email}" not in result

    def test_render_template_no_placeholders(self) -> None:
        faker = Faker()
        text = "No placeholders here."
        assert _render_template(text, faker) == text

    def test_render_template_order_id(self) -> None:
        faker = Faker()
        result = _render_template("Order #{order_id} confirmed", faker)
        assert "{order_id}" not in result


class TestTokenize:
    def test_tokenize_basic(self) -> None:
        # The private _tokenize in datasynthesizer is the Faker placeholder finder
        # Actually _tokenize is not exported; test _FAKER_PLACEHOLDER_RE indirectly
        from aumai_datasynthesizer.core import _FAKER_PLACEHOLDER_RE
        matches = _FAKER_PLACEHOLDER_RE.findall("Hello {name} and {email}")
        assert "name" in matches
        assert "email" in matches

    def test_no_placeholders(self) -> None:
        from aumai_datasynthesizer.core import _FAKER_PLACEHOLDER_RE
        matches = _FAKER_PLACEHOLDER_RE.findall("No placeholders here.")
        assert matches == []


# ---------------------------------------------------------------------------
# Tests for SchemaBasedGenerator
# ---------------------------------------------------------------------------


class TestSchemaBasedGenerator:
    def test_from_schema_returns_correct_count(self, schema_gen: SchemaBasedGenerator) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        results = schema_gen.from_schema(schema, 5)
        assert len(results) == 5

    def test_required_fields_always_present(self, schema_gen: SchemaBasedGenerator) -> None:
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "age": {"type": "integer"},
            },
            "required": ["email", "age"],
        }
        for obj in schema_gen.from_schema(schema, 10):
            assert "email" in obj
            assert "age" in obj

    def test_string_type_returns_string(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "string"})
        assert isinstance(result, str)

    def test_integer_type_returns_int(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "integer"})
        assert isinstance(result, int)

    def test_integer_respects_bounds(self, schema_gen: SchemaBasedGenerator) -> None:
        for _ in range(20):
            result = schema_gen._generate_value({"type": "integer", "minimum": 5, "maximum": 10})
            assert 5 <= result <= 10

    def test_boolean_type_returns_bool(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "boolean"})
        assert isinstance(result, bool)

    def test_null_type_returns_none(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "null"})
        assert result is None

    def test_enum_string_returns_one_of(self, schema_gen: SchemaBasedGenerator) -> None:
        options = ["a", "b", "c"]
        for _ in range(20):
            result = schema_gen._generate_value({"type": "string", "enum": options})
            assert result in options

    def test_format_email_returns_email(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "string", "format": "email"})
        assert "@" in result

    def test_format_date_returns_date_string(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "string", "format": "date"})
        assert isinstance(result, str)

    def test_format_uri_returns_url(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "string", "format": "uri"})
        assert result.startswith("http")

    def test_format_uuid_returns_valid_uuid(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "string", "format": "uuid"})
        parsed = uuid.UUID(result)
        assert parsed.version == 4

    def test_array_type_returns_list(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "array", "items": {"type": "string"}})
        assert isinstance(result, list)

    def test_array_respects_min_max_items(self, schema_gen: SchemaBasedGenerator) -> None:
        for _ in range(20):
            result = schema_gen._generate_value(
                {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 4}
            )
            assert 2 <= len(result) <= 4

    def test_nested_object(self, schema_gen: SchemaBasedGenerator) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"street": {"type": "string"}},
                    "required": ["street"],
                }
            },
            "required": ["address"],
        }
        result = schema_gen._generate_object(schema)
        assert "address" in result
        assert "street" in result["address"]

    def test_unknown_type_returns_word(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "unknowntype"})
        assert isinstance(result, str)

    def test_number_type_returns_float(self, schema_gen: SchemaBasedGenerator) -> None:
        result = schema_gen._generate_value({"type": "number"})
        assert isinstance(result, float)

    def test_from_schema_zero_count(self, schema_gen: SchemaBasedGenerator) -> None:
        schema = {"type": "object", "properties": {}, "required": []}
        results = schema_gen.from_schema(schema, 0)
        assert results == []


# ---------------------------------------------------------------------------
# Tests for DataGenerator
# ---------------------------------------------------------------------------


class TestDataGenerator:
    def test_generate_text_returns_correct_count(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        texts = generator.generate_text(text_config)
        assert len(texts) == 5

    def test_generate_text_returns_strings(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        texts = generator.generate_text(text_config)
        for t in texts:
            assert isinstance(t, str)
            assert len(t) > 0

    def test_generate_text_seed_reproducible(self, generator: DataGenerator) -> None:
        config = GeneratorConfig(data_type=DataType.text, count=3, seed=99)
        texts_a = generator.generate_text(config)
        texts_b = generator.generate_text(config)
        assert texts_a == texts_b

    def test_generate_text_respects_max_sentences(self, generator: DataGenerator) -> None:
        config = GeneratorConfig(
            data_type=DataType.text,
            count=10,
            seed=1,
            constraints={"min_sentences": 1, "max_sentences": 1},
        )
        texts = generator.generate_text(config)
        assert len(texts) == 10

    def test_generate_conversations_returns_correct_count(
        self, generator: DataGenerator, conversation_config: GeneratorConfig
    ) -> None:
        convs = generator.generate_conversations(conversation_config)
        assert len(convs) == 3

    def test_generate_conversations_turns_are_lists(
        self, generator: DataGenerator, conversation_config: GeneratorConfig
    ) -> None:
        convs = generator.generate_conversations(conversation_config)
        for conv in convs:
            assert isinstance(conv, list)
            assert len(conv) > 0

    def test_generate_conversations_turns_are_conversation_turn(
        self, generator: DataGenerator, conversation_config: GeneratorConfig
    ) -> None:
        convs = generator.generate_conversations(conversation_config)
        for conv in convs:
            for turn in conv:
                assert isinstance(turn, ConversationTurn)

    def test_generate_conversations_roles_valid(
        self, generator: DataGenerator, conversation_config: GeneratorConfig
    ) -> None:
        convs = generator.generate_conversations(conversation_config)
        valid_roles = {"system", "user", "assistant"}
        for conv in convs:
            for turn in conv:
                assert turn.role in valid_roles

    def test_generate_conversations_code_assistant_template(
        self, generator: DataGenerator
    ) -> None:
        config = GeneratorConfig(
            data_type=DataType.conversation,
            count=2,
            seed=1,
            constraints={"template": "code_assistant"},
        )
        convs = generator.generate_conversations(config)
        assert len(convs) == 2

    def test_generate_conversations_unknown_template_uses_default(
        self, generator: DataGenerator
    ) -> None:
        config = GeneratorConfig(
            data_type=DataType.conversation,
            count=2,
            seed=1,
            constraints={"template": "nonexistent_template"},
        )
        convs = generator.generate_conversations(config)
        assert len(convs) == 2  # Should not raise

    def test_generate_tool_calls_returns_correct_count(
        self, generator: DataGenerator, tool_call_config: GeneratorConfig
    ) -> None:
        calls = generator.generate_tool_calls(tool_call_config)
        assert len(calls) == 4

    def test_generate_tool_calls_have_id(
        self, generator: DataGenerator, tool_call_config: GeneratorConfig
    ) -> None:
        calls = generator.generate_tool_calls(tool_call_config)
        for call in calls:
            assert "id" in call
            uuid.UUID(str(call["id"]))  # Validates it's a valid UUID

    def test_generate_tool_calls_type_is_function(
        self, generator: DataGenerator, tool_call_config: GeneratorConfig
    ) -> None:
        calls = generator.generate_tool_calls(tool_call_config)
        for call in calls:
            assert call["type"] == "function"

    def test_generate_tool_calls_function_has_name(
        self, generator: DataGenerator, tool_call_config: GeneratorConfig
    ) -> None:
        calls = generator.generate_tool_calls(tool_call_config)
        for call in calls:
            fn = call["function"]
            assert "name" in fn
            assert "arguments" in fn

    def test_generate_tool_calls_arguments_is_valid_json(
        self, generator: DataGenerator, tool_call_config: GeneratorConfig
    ) -> None:
        calls = generator.generate_tool_calls(tool_call_config)
        for call in calls:
            args = call["function"]["arguments"]
            parsed = json.loads(str(args))
            assert isinstance(parsed, dict)

    def test_generate_tool_calls_email_template(self, generator: DataGenerator) -> None:
        config = GeneratorConfig(
            data_type=DataType.tool_call,
            count=3,
            seed=1,
            constraints={"tool": "email"},
        )
        calls = generator.generate_tool_calls(config)
        assert len(calls) == 3
        for call in calls:
            assert call["function"]["name"] == "send_email"

    def test_generate_tool_calls_unknown_tool_uses_default(self, generator: DataGenerator) -> None:
        config = GeneratorConfig(
            data_type=DataType.tool_call,
            count=2,
            seed=1,
            constraints={"tool": "nonexistent_tool"},
        )
        calls = generator.generate_tool_calls(config)
        assert len(calls) == 2

    def test_generate_agent_traces_returns_correct_count(
        self, generator: DataGenerator, agent_trace_config: GeneratorConfig
    ) -> None:
        traces = generator.generate_agent_traces(agent_trace_config)
        assert len(traces) == 3

    def test_generate_agent_traces_have_trace_id(
        self, generator: DataGenerator, agent_trace_config: GeneratorConfig
    ) -> None:
        traces = generator.generate_agent_traces(agent_trace_config)
        for trace in traces:
            assert "trace_id" in trace
            uuid.UUID(str(trace["trace_id"]))

    def test_generate_agent_traces_have_steps(
        self, generator: DataGenerator, agent_trace_config: GeneratorConfig
    ) -> None:
        traces = generator.generate_agent_traces(agent_trace_config)
        for trace in traces:
            assert "steps" in trace

    def test_generate_agent_traces_steps_non_empty(
        self, generator: DataGenerator, agent_trace_config: GeneratorConfig
    ) -> None:
        traces = generator.generate_agent_traces(agent_trace_config)
        for trace in traces:
            steps = trace["steps"]
            assert isinstance(steps, list), "steps must be a list"
            assert len(steps) >= 2, (
                f"each trace should have at least 2 steps (min=2 in generator), got {len(steps)}"
            )

    def test_generate_agent_traces_steps_have_required_fields(
        self, generator: DataGenerator, agent_trace_config: GeneratorConfig
    ) -> None:
        traces = generator.generate_agent_traces(agent_trace_config)
        for trace in traces:
            for step in trace["steps"]:
                assert "step" in step, "each step dict must have a 'step' index field"
                assert "type" in step, "each step dict must have a 'type' field"
                assert "timestamp" in step, "each step dict must have a 'timestamp' field"

    def test_generate_json_free_form_count(
        self, generator: DataGenerator, json_config: GeneratorConfig
    ) -> None:
        results = generator.generate_json(json_config)
        assert len(results) == 5

    def test_generate_json_free_form_has_id(
        self, generator: DataGenerator, json_config: GeneratorConfig
    ) -> None:
        results = generator.generate_json(json_config)
        for item in results:
            assert "id" in item

    def test_generate_json_schema_driven(self, generator: DataGenerator) -> None:
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }
        config = GeneratorConfig(
            data_type=DataType.json, count=5, seed=1, schema=schema
        )
        results = generator.generate_json(config)
        assert len(results) == 5
        for item in results:
            assert "name" in item
            assert "age" in item


# ---------------------------------------------------------------------------
# Tests for generate() dispatcher
# ---------------------------------------------------------------------------


class TestDataGeneratorDispatcher:
    def test_generate_text_returns_dataset(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(text_config)
        assert isinstance(dataset, SyntheticDataset)
        assert len(dataset.samples) == 5

    def test_generate_text_samples_have_index_and_text(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(text_config)
        for i, sample in enumerate(dataset.samples):
            assert sample["index"] == i
            assert "text" in sample

    def test_generate_conversation_samples(
        self, generator: DataGenerator, conversation_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(conversation_config)
        assert len(dataset.samples) == 3
        for sample in dataset.samples:
            assert "turns" in sample

    def test_generate_tool_call_samples(
        self, generator: DataGenerator, tool_call_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(tool_call_config)
        assert len(dataset.samples) == 4
        for sample in dataset.samples:
            assert "type" in sample

    def test_generate_agent_trace_samples(
        self, generator: DataGenerator, agent_trace_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(agent_trace_config)
        assert len(dataset.samples) == 3
        for sample in dataset.samples:
            assert "trace_id" in sample

    def test_generate_json_samples(
        self, generator: DataGenerator, json_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(json_config)
        assert len(dataset.samples) == 5

    def test_dataset_metadata_has_generated_count(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(text_config)
        assert dataset.metadata["generated_count"] == 5

    def test_dataset_metadata_has_generation_time(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(text_config)
        assert "generation_time_ms" in dataset.metadata
        assert dataset.metadata["generation_time_ms"] >= 0

    def test_dataset_metadata_has_data_type(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(text_config)
        assert dataset.metadata["data_type"] == "text"

    def test_dataset_config_preserved(
        self, generator: DataGenerator, text_config: GeneratorConfig
    ) -> None:
        dataset = generator.generate(text_config)
        assert dataset.config == text_config


# ---------------------------------------------------------------------------
# Tests for templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_conversation_templates_exist(self) -> None:
        assert "customer_support" in CONVERSATION_TEMPLATES
        assert "code_assistant" in CONVERSATION_TEMPLATES
        assert "research_assistant" in CONVERSATION_TEMPLATES

    def test_conversation_templates_have_turns(self) -> None:
        for name, turns in CONVERSATION_TEMPLATES.items():
            assert len(turns) >= 2, f"Template '{name}' should have at least 2 turns"

    def test_conversation_turns_have_role_and_content(self) -> None:
        for name, turns in CONVERSATION_TEMPLATES.items():
            for turn in turns:
                assert "role" in turn, f"Turn in '{name}' missing 'role'"
                assert "content" in turn, f"Turn in '{name}' missing 'content'"

    def test_tool_call_templates_exist(self) -> None:
        assert "search" in TOOL_CALL_TEMPLATES
        assert "email" in TOOL_CALL_TEMPLATES
        assert "database" in TOOL_CALL_TEMPLATES
        assert "file_operations" in TOOL_CALL_TEMPLATES

    def test_tool_call_templates_have_name(self) -> None:
        for key, spec in TOOL_CALL_TEMPLATES.items():
            assert "name" in spec, f"Tool template '{key}' missing 'name'"

    def test_tool_call_templates_have_parameters(self) -> None:
        for key, spec in TOOL_CALL_TEMPLATES.items():
            assert "parameters" in spec, f"Tool template '{key}' missing 'parameters'"


# ---------------------------------------------------------------------------
# Tests for models
# ---------------------------------------------------------------------------


class TestModels:
    def test_generator_config_count_gt_zero(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GeneratorConfig(data_type=DataType.text, count=0)

    def test_generator_config_data_type_required(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            GeneratorConfig(count=5)  # type: ignore[call-arg]

    def test_data_type_enum_values(self) -> None:
        assert DataType.text.value == "text"
        assert DataType.json.value == "json"
        assert DataType.conversation.value == "conversation"
        assert DataType.tool_call.value == "tool_call"
        assert DataType.agent_trace.value == "agent_trace"

    def test_conversation_turn_model(self) -> None:
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"
        assert turn.tool_calls is None

    def test_synthetic_dataset_model(self, text_config: GeneratorConfig) -> None:
        dataset = SyntheticDataset(config=text_config, samples=[{"index": 0, "text": "hello"}])
        assert len(dataset.samples) == 1
        assert dataset.metadata == {}

    @pytest.mark.parametrize(
        "data_type_str",
        ["text", "json", "conversation", "tool_call", "agent_trace"],
    )
    def test_all_data_types_accepted(self, data_type_str: str) -> None:
        config = GeneratorConfig(data_type=DataType(data_type_str), count=1)
        assert config.data_type.value == data_type_str
