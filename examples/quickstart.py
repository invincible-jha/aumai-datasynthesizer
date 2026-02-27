"""aumai-datasynthesizer quickstart — runnable examples for all five DataType values.

This file demonstrates:
  1. Generating synthetic text samples with a reproducible seed.
  2. Generating multi-turn conversations from named templates.
  3. Generating tool-call payloads suitable for OpenAI-style function calling.
  4. Generating agent execution traces for ReAct-style agent testing.
  5. Generating schema-driven JSON objects for structured-output testing.

Run directly:
    python examples/quickstart.py

Install first:
    pip install aumai-datasynthesizer
"""

from __future__ import annotations

import json
from typing import Any

from aumai_datasynthesizer import (
    DataGenerator,
    DataType,
    GeneratorConfig,
    SyntheticDataset,
)


# ---------------------------------------------------------------------------
# Demo 1: Reproducible text generation
# ---------------------------------------------------------------------------

def demo_text_generation() -> SyntheticDataset:
    """Generate 5 short text paragraphs with a fixed seed for reproducibility.

    A seed guarantees that re-running the script always produces the same
    samples — essential for regression testing and dataset versioning.
    """
    print("\n--- Demo 1: Text Generation ---")

    config = GeneratorConfig(
        data_type=DataType.text,
        count=5,
        seed=42,
        constraints={"min_sentences": 2, "max_sentences": 4},
    )

    generator = DataGenerator()
    dataset = generator.generate(config)

    for sample in dataset.samples:
        preview = str(sample["text"])[:80].replace("\n", " ")
        print(f"  [{sample['index']}] {preview}...")

    print(f"  Metadata: {dataset.metadata}")
    return dataset


# ---------------------------------------------------------------------------
# Demo 2: Multi-turn conversation generation
# ---------------------------------------------------------------------------

def demo_conversation_generation() -> SyntheticDataset:
    """Generate 3 customer-support conversations using a built-in template.

    Available template names: customer_support, code_assistant, research_assistant.
    Each conversation is a list of turns with 'role' and 'content' keys.
    """
    print("\n--- Demo 2: Conversation Generation ---")

    config = GeneratorConfig(
        data_type=DataType.conversation,
        count=3,
        seed=7,
        constraints={"template": "customer_support"},
    )

    generator = DataGenerator()
    dataset = generator.generate(config)

    for sample in dataset.samples:
        turns: list[dict[str, Any]] = sample["turns"]  # type: ignore[assignment]
        print(f"\n  Conversation {sample['index']} ({len(turns)} turns):")
        for turn in turns[:3]:  # print first 3 turns to keep output concise
            role = str(turn["role"]).upper()
            content_preview = str(turn["content"])[:60].replace("\n", " ")
            print(f"    [{role}] {content_preview}...")

    return dataset


# ---------------------------------------------------------------------------
# Demo 3: Synthetic tool-call payloads
# ---------------------------------------------------------------------------

def demo_tool_call_generation() -> SyntheticDataset:
    """Generate 4 web_search tool-call payloads in OpenAI function-calling format.

    Available tool names: search, email, database, file_operations.
    Each payload has an 'id', 'type', and 'function' dict with 'name' and
    'arguments' (a JSON string) — ready to inject into LLM test harnesses.
    """
    print("\n--- Demo 3: Tool Call Generation ---")

    config = GeneratorConfig(
        data_type=DataType.tool_call,
        count=4,
        seed=99,
        constraints={"tool": "search"},
    )

    generator = DataGenerator()
    dataset = generator.generate(config)

    for sample in dataset.samples:
        function_block: dict[str, Any] = sample["function"]  # type: ignore[assignment]
        tool_name = function_block["name"]
        args = json.loads(str(function_block["arguments"]))
        print(f"  [{sample['index']}] {tool_name}({args})")

    return dataset


# ---------------------------------------------------------------------------
# Demo 4: Agent execution trace generation
# ---------------------------------------------------------------------------

def demo_agent_trace_generation() -> SyntheticDataset:
    """Generate 3 synthetic agent traces with thought/tool_call/observation steps.

    Traces mirror the ReAct (Reason + Act) pattern.  Each trace contains a
    trace_id, agent name, task, ordered steps, a final answer, and a success
    flag.  Use these to test trace-parsing, evaluation, or agent benchmarks.
    """
    print("\n--- Demo 4: Agent Trace Generation ---")

    config = GeneratorConfig(
        data_type=DataType.agent_trace,
        count=3,
        seed=2024,
    )

    generator = DataGenerator()
    dataset = generator.generate(config)

    for sample in dataset.samples:
        steps: list[dict[str, Any]] = sample["steps"]  # type: ignore[assignment]
        step_types = [s["type"] for s in steps]
        print(
            f"  Trace {sample['index']}: agent={sample['agent']!r}  "
            f"steps={step_types}  success={sample['success']}"
        )
        final = str(sample["final_answer"])[:70].replace("\n", " ")
        print(f"    Answer: {final}...")

    return dataset


# ---------------------------------------------------------------------------
# Demo 5: Schema-driven JSON object generation
# ---------------------------------------------------------------------------

def demo_json_schema_generation() -> SyntheticDataset:
    """Generate 5 JSON objects that conform to a custom JSON Schema.

    Pass any JSON Schema (objects, strings, integers, booleans, arrays,
    enums, formats) to GeneratorConfig.schema.  When schema is None the
    generator falls back to a sensible free-form object with name, email,
    value, active flag, tags, and created_at fields.
    """
    print("\n--- Demo 5: Schema-Driven JSON Generation ---")

    product_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "format": "uuid"},
            "name": {"type": "string"},
            "price_usd": {"type": "number", "minimum": 0.5, "maximum": 999.99},
            "in_stock": {"type": "boolean"},
            "category": {
                "type": "string",
                "enum": ["electronics", "apparel", "home", "books", "sports"],
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["product_id", "name", "price_usd", "in_stock", "category"],
    }

    config = GeneratorConfig(
        data_type=DataType.json,
        count=5,
        seed=1337,
        schema=product_schema,
    )

    generator = DataGenerator()
    dataset = generator.generate(config)

    for sample in dataset.samples:
        obj = {k: v for k, v in sample.items() if k != "index"}
        print(f"  [{sample['index']}] {obj}")

    print(f"\n  Generation time: {dataset.metadata['generation_time_ms']} ms")
    return dataset


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all five quickstart demonstrations in sequence."""
    print("=" * 60)
    print("aumai-datasynthesizer quickstart")
    print("Synthetic training data generation for agent testing")
    print("=" * 60)

    demo_text_generation()
    demo_conversation_generation()
    demo_tool_call_generation()
    demo_agent_trace_generation()
    demo_json_schema_generation()

    print("\nDone. All five data types generated successfully.")


if __name__ == "__main__":
    main()
