# API Reference — aumai-datasynthesizer

Complete documentation for all public classes, functions, and Pydantic models.

---

## Module: `aumai_datasynthesizer.models`

Pydantic v2 models representing the configuration and output of generation runs.

---

### `DataType`

Enumeration of supported synthetic data types.

```python
class DataType(str, Enum):
    text          = "text"
    json          = "json"
    conversation  = "conversation"
    tool_call     = "tool_call"
    agent_trace   = "agent_trace"
```

**Values:**

| Value | String | Description |
|---|---|---|
| `DataType.text` | `"text"` | Free-form text paragraphs. |
| `DataType.json` | `"json"` | JSON objects, optionally conforming to a provided JSON Schema. |
| `DataType.conversation` | `"conversation"` | Multi-turn conversation records (role/content/tool_calls). |
| `DataType.tool_call` | `"tool_call"` | OpenAI-style function-calling records. |
| `DataType.agent_trace` | `"agent_trace"` | Agent execution traces with thought/tool_call/observation steps. |

**Example:**

```python
from aumai_datasynthesizer import DataType

dt = DataType("conversation")
print(dt)          # DataType.conversation
print(dt.value)    # "conversation"
print(dt == DataType.conversation)  # True
```

---

### `GeneratorConfig`

Configuration for a synthetic data generation run. All fields are validated by Pydantic v2 at construction time.

```python
class GeneratorConfig(BaseModel):
    data_type: DataType
    count: int
    seed: int | None
    schema: dict[str, object] | None
    constraints: dict[str, object]
```

**Fields:**

| Field | Type | Required | Constraints | Description |
|---|---|---|---|---|
| `data_type` | `DataType` | yes | — | The type of data to generate. |
| `count` | `int` | no | `> 0` | Number of samples to generate. Defaults to `10`. |
| `seed` | `int \| None` | no | — | Integer seed for `Faker` and `random`. `None` means non-deterministic output. Defaults to `None`. |
| `schema` | `dict[str, object] \| None` | no | — | JSON Schema object used when `data_type=DataType.json`. Ignored for other data types. Defaults to `None`. |
| `constraints` | `dict[str, object]` | no | — | Free-form type-specific overrides. See constraint reference below. Defaults to `{}`. |

**Constraint reference:**

| `data_type` | Key | Type | Default | Description |
|---|---|---|---|---|
| `text` | `min_sentences` | `int` | `1` | Minimum sentences per paragraph. |
| `text` | `max_sentences` | `int` | `5` | Maximum sentences per paragraph. |
| `conversation` | `template` | `str` | `"customer_support"` | Conversation template name. |
| `tool_call` | `tool` | `str` | `"search"` | Tool-call template name. |

**Example:**

```python
from aumai_datasynthesizer import GeneratorConfig, DataType

config = GeneratorConfig(
    data_type=DataType.conversation,
    count=20,
    seed=42,
    constraints={"template": "code_assistant"},
)
print(config.data_type.value)  # "conversation"
print(config.count)            # 20
print(config.seed)             # 42
```

---

### `ConversationTurn`

A single turn within a multi-turn conversation.

```python
class ConversationTurn(BaseModel):
    role: str
    content: str
    tool_calls: list[dict[str, object]] | None
```

**Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `role` | `str` | yes | Conversation role: `"system"`, `"user"`, or `"assistant"`. Determined by the template. |
| `content` | `str` | yes | The message content. Template placeholders are resolved by Faker at generation time. |
| `tool_calls` | `list[dict] \| None` | no | Optional list of tool call objects attached to this turn. Currently always `None` in generated conversations; available for custom data. |

**Example:**

```python
from aumai_datasynthesizer import ConversationTurn

turn = ConversationTurn(
    role="user",
    content="How do I implement a binary search in Python?",
    tool_calls=None,
)
print(turn.role)     # "user"
print(turn.content)
```

---

### `SyntheticDataset`

The output of a generation run: configuration, samples, and metadata.

```python
class SyntheticDataset(BaseModel):
    config: GeneratorConfig
    samples: list[dict[str, object]]
    metadata: dict[str, object]
```

**Fields:**

| Field | Type | Description |
|---|---|---|
| `config` | `GeneratorConfig` | The configuration that produced this dataset. |
| `samples` | `list[dict[str, object]]` | The generated records. Every sample has an `"index"` key. Additional keys depend on `data_type`. |
| `metadata` | `dict[str, object]` | Generation metadata: `"generated_count"`, `"generation_time_ms"`, `"data_type"`. |

**Sample structure by data type:**

| `data_type` | Sample keys |
|---|---|
| `text` | `index`, `text` (str) |
| `json` | `index`, plus all schema-defined fields |
| `conversation` | `index`, `turns` (list of `ConversationTurn.model_dump()`) |
| `tool_call` | `index`, `id`, `type`, `function` (`name`, `arguments`) |
| `agent_trace` | `index`, `trace_id`, `agent`, `task`, `steps`, `final_answer`, `success` |

**Example:**

```python
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType

generator = DataGenerator()
dataset = generator.generate(GeneratorConfig(data_type=DataType.text, count=5, seed=0))

print(dataset.metadata)
# {'generated_count': 5, 'generation_time_ms': 2.4, 'data_type': 'text'}
print(dataset.samples[0])
# {'index': 0, 'text': 'Far far away, behind the word mountains...'}
```

---

## Module: `aumai_datasynthesizer.core`

Core generation logic.

---

### `SchemaBasedGenerator`

Generate Python dicts that conform to a JSON Schema. Supports a subset of JSON Schema: `object`, `string`, `integer`, `number`, `boolean`, `array`, and `null` types.

```python
class SchemaBasedGenerator:
    def __init__(self, faker: Faker) -> None: ...
    def from_schema(self, schema: dict[str, object], count: int) -> list[dict[str, object]]: ...
```

#### `SchemaBasedGenerator.__init__(faker)`

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `faker` | `Faker` | A Faker instance. Controls locale and, if seeded, reproducibility. |

**Example:**

```python
from faker import Faker
from aumai_datasynthesizer import SchemaBasedGenerator

gen = SchemaBasedGenerator(faker=Faker("en_US"))
```

---

#### `SchemaBasedGenerator.from_schema(schema, count)`

Return `count` dicts matching the given JSON Schema.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `schema` | `dict[str, object]` | A JSON Schema object. Must have `"type": "object"` at the top level. |
| `count` | `int` | Number of records to generate. |

**Returns:** `list[dict[str, object]]`

**Supported schema features:**

| Schema feature | Behaviour |
|---|---|
| `"type": "string"` | Returns `faker.sentence(nb_words=4)` |
| `"format": "email"` | Returns `faker.email()` |
| `"format": "date"` | Returns `str(faker.date())` |
| `"format": "uri"` | Returns `faker.url()` |
| `"format": "uuid"` | Returns `str(uuid.uuid4())` |
| `"enum": [...]` | Returns `faker.random_element(enum)` |
| `"type": "integer"` + `"minimum"` / `"maximum"` | Returns `faker.random_int(min, max)` |
| `"type": "number"` + `"minimum"` / `"maximum"` | Returns `round(random.uniform(min, max), 4)` |
| `"type": "boolean"` | Returns `faker.boolean()` |
| `"type": "array"` + `"items"` + `"minItems"` / `"maxItems"` | Returns a list of `random_int(minItems, maxItems)` elements |
| `"type": "null"` | Returns `None` |
| `"required": [...]` | Listed properties are always generated |
| Optional properties | Generated with 80% probability |

**Example:**

```python
schema = {
    "type": "object",
    "properties": {
        "id":    {"type": "string", "format": "uuid"},
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
    },
    "required": ["id", "score", "label"],
}

records = gen.from_schema(schema, count=3)
for r in records:
    print(r)
# {'id': '...uuid...', 'score': 0.7341, 'label': 'positive'}
```

---

### `DataGenerator`

Main dispatcher that generates `SyntheticDataset` objects for all `DataType` values.

```python
class DataGenerator:
    def __init__(self, faker: Faker | None = None) -> None: ...
    def generate(self, config: GeneratorConfig) -> SyntheticDataset: ...
    def generate_text(self, config: GeneratorConfig) -> list[str]: ...
    def generate_conversations(self, config: GeneratorConfig) -> list[list[ConversationTurn]]: ...
    def generate_tool_calls(self, config: GeneratorConfig) -> list[dict[str, object]]: ...
    def generate_agent_traces(self, config: GeneratorConfig) -> list[dict[str, object]]: ...
    def generate_json(self, config: GeneratorConfig) -> list[dict[str, object]]: ...
```

#### `DataGenerator.__init__(faker=None)`

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `faker` | `Faker \| None` | Optional Faker instance. If `None`, a default `Faker()` is created per call. If a `seed` is set on the config, `Faker.seed()` and `random.seed()` are called regardless of whether a custom Faker is provided. |

```python
from aumai_datasynthesizer import DataGenerator

# Default: new Faker instance per generation call
generator = DataGenerator()

# Custom Faker instance (e.g. for specific locale)
from faker import Faker
generator_fr = DataGenerator(faker=Faker("fr_FR"))
```

---

#### `DataGenerator.generate(config)`

Generate a full `SyntheticDataset` for the given configuration.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `GeneratorConfig` | Validated generation parameters. |

**Returns:** `SyntheticDataset` — contains the config, all generated samples, and timing/count metadata.

**Dispatch logic:**

| `config.data_type` | Delegates to |
|---|---|
| `DataType.text` | `generate_text()` |
| `DataType.conversation` | `generate_conversations()` |
| `DataType.tool_call` | `generate_tool_calls()` |
| `DataType.agent_trace` | `generate_agent_traces()` |
| `DataType.json` | `generate_json()` |

**Example:**

```python
from aumai_datasynthesizer import DataGenerator, GeneratorConfig, DataType

gen = DataGenerator()
dataset = gen.generate(GeneratorConfig(data_type=DataType.agent_trace, count=5, seed=1))
print(len(dataset.samples))   # 5
print(dataset.metadata)       # {'generated_count': 5, 'generation_time_ms': ..., 'data_type': 'agent_trace'}
```

---

#### `DataGenerator.generate_text(config)`

Return a list of random text strings (not wrapped in a `SyntheticDataset`).

**Parameters:** `config: GeneratorConfig` — reads `count`, `seed`, and constraints `min_sentences` / `max_sentences`.

**Returns:** `list[str]`

---

#### `DataGenerator.generate_conversations(config)`

Return a list of multi-turn conversations (not wrapped in a `SyntheticDataset`).

**Parameters:** `config: GeneratorConfig` — reads `count`, `seed`, and constraint `template`.

**Returns:** `list[list[ConversationTurn]]` — each inner list is one complete conversation.

**Constraints:**

| Key | Default | Description |
|---|---|---|
| `template` | `"customer_support"` | Name of the template to use. Falls back to `"customer_support"` if the name is not found in `CONVERSATION_TEMPLATES`. |

---

#### `DataGenerator.generate_tool_calls(config)`

Return a list of synthetic tool-call dicts in OpenAI function-calling format.

**Parameters:** `config: GeneratorConfig` — reads `count`, `seed`, and constraint `tool`.

**Returns:** `list[dict[str, object]]` — each dict has keys `id`, `type`, `function` (`name`, `arguments`).

**Constraints:**

| Key | Default | Description |
|---|---|---|
| `tool` | `"search"` | Tool template name. Falls back to `"search"` if not found in `TOOL_CALL_TEMPLATES`. |

---

#### `DataGenerator.generate_agent_traces(config)`

Return a list of synthetic agent execution trace dicts.

**Parameters:** `config: GeneratorConfig` — reads `count`, `seed`. No constraint keys are used.

**Returns:** `list[dict[str, object]]` — each trace has:

| Key | Type | Description |
|---|---|---|
| `trace_id` | `str` | UUID string |
| `agent` | `str` | Random word + `"_agent"` |
| `task` | `str` | 6-word Faker sentence |
| `steps` | `list[dict]` | 2–6 steps; each step has `step` (index), `type`, `timestamp`, and type-specific content |
| `final_answer` | `str` | 1-sentence Faker paragraph |
| `success` | `bool` | `True` with 80% probability |

Step types:
- `"thought"` — has `content` (str).
- `"tool_call"` — has `tool` (str, from `TOOL_CALL_TEMPLATES`) and `arguments` (`{"query": ...}`).
- `"observation"` — has `content` (2-sentence paragraph).

---

#### `DataGenerator.generate_json(config)`

Return schema-conformant or free-form JSON dicts.

**Parameters:** `config: GeneratorConfig` — reads `count`, `seed`, `schema`.

**Returns:** `list[dict[str, object]]`

**Logic:**
- If `config.schema` is not `None`, delegates to `SchemaBasedGenerator.from_schema(config.schema, config.count)`.
- If `config.schema` is `None`, generates free-form records with keys: `id` (UUID), `name`, `email`, `value` (float 0–1000), `active` (bool), `tags` (list of words), `created_at` (datetime string).

---

## Module: `aumai_datasynthesizer.templates`

Built-in conversation and tool-call templates.

---

### `CONVERSATION_TEMPLATES`

```python
CONVERSATION_TEMPLATES: dict[str, list[dict[str, object]]]
```

A mapping from template name to a list of turn dicts. Each turn dict has `"role"` and `"content"` keys. Content may contain `{placeholder}` tokens resolved by Faker at generation time.

**Available templates:** `"customer_support"` (7 turns), `"code_assistant"` (5 turns), `"research_assistant"` (5 turns).

---

### `TOOL_CALL_TEMPLATES`

```python
TOOL_CALL_TEMPLATES: dict[str, dict[str, object]]
```

A mapping from template name to a tool spec dict containing `"name"`, `"description"`, and `"parameters"` (a JSON Schema object).

**Available templates:** `"search"` (`web_search`), `"email"` (`send_email`), `"database"` (`execute_query`), `"file_operations"` (`file_operation`).

---

## Top-level `__init__.py` exports

```python
from aumai_datasynthesizer import (
    # Core classes
    DataGenerator,          # class
    SchemaBasedGenerator,   # class
    # Models
    ConversationTurn,       # Pydantic model
    DataType,               # Enum
    GeneratorConfig,        # Pydantic model
    SyntheticDataset,       # Pydantic model
)

__version__  # str, e.g. "0.1.0"
```

All symbols are importable directly from `aumai_datasynthesizer`.
