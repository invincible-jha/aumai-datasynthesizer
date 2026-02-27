# Getting Started with aumai-datasynthesizer

This guide takes you from installation to generating your first synthetic datasets in about 15 minutes.

---

## Prerequisites

- Python 3.11 or higher
- `pip` (comes with Python)
- Basic familiarity with the command line and JSON

No GPU, API key, or external AI service is required.

---

## Installation

### From PyPI (recommended)

```bash
pip install aumai-datasynthesizer
```

Verify the installation:

```bash
aumai-datasynthesizer --version
# aumai-datasynthesizer, version 0.1.0
```

### From source

```bash
git clone https://github.com/aumai/aumai-datasynthesizer.git
cd aumai-datasynthesizer
pip install -e .
```

### Development mode (with test dependencies)

```bash
git clone https://github.com/aumai/aumai-datasynthesizer.git
cd aumai-datasynthesizer
pip install -e ".[dev]"
make lint   # ruff + mypy
make test   # pytest
```

---

## Your First Synthetic Dataset

This tutorial walks through generating each of the five supported data types.

### Step 1 — Generate text paragraphs

```bash
aumai-datasynthesizer generate --type text --count 5 --seed 42
# Generated 5 text samples in 2.1 ms.
```

Output (one JSON object per line):

```
{"index": 0, "text": "Far far away behind the word mountains..."}
{"index": 1, "text": "One morning when Gregor Samsa woke..."}
...
```

Add `--output` to save to a file:

```bash
aumai-datasynthesizer generate --type text --count 100 --seed 42 \
  --output training_text.jsonl
```

### Step 2 — Generate multi-turn conversations

```bash
aumai-datasynthesizer generate --type conversation --count 3 --seed 0 \
  --constraint template=customer_support
```

Each conversation is a JSON object with a `turns` array. Each turn has `role`, `content`, and optionally `tool_calls`.

### Step 3 — Generate tool call records

```bash
aumai-datasynthesizer generate --type tool_call --count 10 \
  --constraint tool=email
```

Each record matches the OpenAI function-calling format: `{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}`.

### Step 4 — Generate agent execution traces

```bash
aumai-datasynthesizer generate --type agent_trace --count 2 --seed 1
```

Each trace includes `trace_id`, `agent`, `task`, `steps` (2–6 steps with types `thought`, `tool_call`, `observation`), `final_answer`, and `success`.

### Step 5 — Generate schema-driven JSON

Create a schema file:

```bash
cat > product_schema.json <<'EOF'
{
  "type": "object",
  "properties": {
    "product_id": { "type": "string", "format": "uuid" },
    "name":       { "type": "string" },
    "price":      { "type": "number", "minimum": 0.99, "maximum": 999.99 },
    "category":   { "type": "string", "enum": ["electronics", "clothing", "books", "food"] },
    "in_stock":   { "type": "boolean" },
    "tags":       { "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 5 }
  },
  "required": ["product_id", "name", "price", "category"]
}
EOF
```

Generate 50 products:

```bash
aumai-datasynthesizer generate --type json --count 50 \
  --schema product_schema.json --seed 7 --output products.jsonl
```

---

## Using the Python API

### Minimal example

```python
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType

generator = DataGenerator()
config = GeneratorConfig(data_type=DataType.conversation, count=5, seed=42)
dataset = generator.generate(config)

print(f"Generated {len(dataset.samples)} samples")
print(f"Time: {dataset.metadata['generation_time_ms']} ms")

# Access the first conversation
for turn in dataset.samples[0]["turns"]:
    print(f"[{turn['role']}] {turn['content'][:60]}")
```

---

## Common Patterns

### Pattern 1 — Reproducible test fixtures

Fix the seed to generate identical data on every run. Use this for deterministic unit tests:

```python
import pytest
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType

@pytest.fixture
def synthetic_conversations():
    """10 reproducible customer-support conversations."""
    generator = DataGenerator()
    config = GeneratorConfig(
        data_type=DataType.conversation,
        count=10,
        seed=12345,
        constraints={"template": "customer_support"},
    )
    return generator.generate(config).samples


def test_agent_handles_order_query(synthetic_conversations):
    # Each run produces the same conversations — no flakiness
    for conv in synthetic_conversations:
        turns = conv["turns"]
        assert len(turns) > 0
        assert turns[0]["role"] in ("system", "user", "assistant")
```

---

### Pattern 2 — Generate training data for fine-tuning

Generate thousands of conversation examples and convert to a fine-tuning format:

```python
import json
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType

generator = DataGenerator()

all_examples = []
for template in ("customer_support", "code_assistant", "research_assistant"):
    config = GeneratorConfig(
        data_type=DataType.conversation,
        count=500,
        seed=0,
        constraints={"template": template},
    )
    dataset = generator.generate(config)
    all_examples.extend(dataset.samples)

# Convert to OpenAI fine-tuning format
with open("finetune_data.jsonl", "w") as fh:
    for example in all_examples:
        record = {"messages": example["turns"]}
        fh.write(json.dumps(record) + "\n")

print(f"Wrote {len(all_examples)} training examples")
```

---

### Pattern 3 — Schema-driven test data generation

Generate records matching your application's actual data models:

```python
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType

user_schema = {
    "type": "object",
    "properties": {
        "id":          {"type": "string", "format": "uuid"},
        "email":       {"type": "string", "format": "email"},
        "age":         {"type": "integer", "minimum": 18, "maximum": 90},
        "plan":        {"type": "string", "enum": ["free", "pro", "enterprise"]},
        "verified":    {"type": "boolean"},
        "created_at":  {"type": "string", "format": "date"},
    },
    "required": ["id", "email", "age", "plan"],
}

generator = DataGenerator()
config = GeneratorConfig(
    data_type=DataType.json,
    count=1000,
    schema=user_schema,
    seed=99,
)
dataset = generator.generate(config)

# All records conform to the schema
for record in dataset.samples[:3]:
    print(record)
```

---

### Pattern 4 — Load test an agent with synthetic tool calls

Generate realistic tool-call records to benchmark your agent's tool dispatcher:

```python
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType
import json

generator = DataGenerator()
tool_calls_per_type = {}

for tool in ("search", "email", "database", "file_operations"):
    config = GeneratorConfig(
        data_type=DataType.tool_call,
        count=200,
        seed=0,
        constraints={"tool": tool},
    )
    dataset = generator.generate(config)
    tool_calls_per_type[tool] = dataset.samples

# Write all tool calls to a load-test fixture
all_calls = [call for calls in tool_calls_per_type.values() for call in calls]
with open("tool_call_fixture.jsonl", "w") as fh:
    for call in all_calls:
        fh.write(json.dumps(call) + "\n")

print(f"Generated {len(all_calls)} synthetic tool calls across 4 tools")
```

---

### Pattern 5 — Enumerate all built-in templates in Python

```python
from aumai_datasynthesizer.templates import CONVERSATION_TEMPLATES, TOOL_CALL_TEMPLATES

print("Conversation templates:")
for name, turns in CONVERSATION_TEMPLATES.items():
    print(f"  {name}: {len(turns)} turns")

print("\nTool-call templates:")
for name, spec in TOOL_CALL_TEMPLATES.items():
    params = spec.get("parameters", {}).get("properties", {})
    print(f"  {name} -> {spec['name']} ({len(params)} params)")
```

---

## Troubleshooting FAQ

### Output looks identical across multiple runs

If you specified `--seed` (or `config.seed`), identical output is correct and expected — that is the purpose of seeding. Remove `--seed` or change its value to get different output.

---

### The `--schema` flag is ignored for `--type conversation`

The `--schema` flag applies only to `--type json`. Conversation and tool-call templates have fixed structures defined in `templates.py`. To customise conversation structure, edit or extend the templates.

---

### `ValidationError` on `GeneratorConfig`

Pydantic validates `count` to be `> 0`. A value of `0` will raise:

```
pydantic_core.InitErrorDetails: Input should be greater than 0 [type=greater_than, ...]
```

Use `--count 1` or higher.

---

### JSON Schema properties are sometimes missing from output

Optional properties (not in `"required"`) are generated with 80% probability by design. This simulates real-world data where optional fields are not always present. To force all properties to appear, add them to `"required"` in your schema.

---

### The `--constraint template=X` is silently ignored

Constraint values passed via the CLI are always strings. If you mis-type the template name, the generator falls back to `customer_support` silently (because `CONVERSATION_TEMPLATES.get(template_name, CONVERSATION_TEMPLATES["customer_support"])` is the lookup). Check the spelling with `aumai-datasynthesizer templates --list`.

---

### Faker generates non-English content occasionally

Faker uses `en_US` locale by default. If you see non-English text, a different locale may have been injected earlier in the process. Instantiate your own Faker with explicit locale and pass it to `DataGenerator`:

```python
from faker import Faker
from aumai_datasynthesizer import DataGenerator

generator = DataGenerator(faker=Faker("en_US"))
```

---

### Generation is slow for large counts

`aumai-datasynthesizer` is designed for up to tens of thousands of samples. For counts above 100,000 consider batching (loop over multiple smaller `GeneratorConfig` calls). Each call is synchronous and single-threaded. The bottleneck is Faker method invocations, which are CPU-bound pure Python.

---

## Next Steps

- [API Reference](api-reference.md) — complete documentation for every class and function.
- [Examples](../examples/quickstart.py) — runnable Python code demonstrating all features.
- [Contributing](../CONTRIBUTING.md) — how to add templates, data types, or schema features.
