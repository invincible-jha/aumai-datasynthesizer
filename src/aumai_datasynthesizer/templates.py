"""Pre-built conversation and tool-call templates for data generation."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Conversation templates
# Each entry is a list of turn-dicts with "role" and "content" keys.
# "content" values may contain {faker_*} placeholders resolved at generation
# time via the Faker instance — see core.py DataGenerator._apply_template.
# ---------------------------------------------------------------------------

CONVERSATION_TEMPLATES: dict[str, list[dict[str, object]]] = {
    "customer_support": [
        {
            "role": "system",
            "content": (
                "You are a helpful customer support agent for an e-commerce platform. "
                "Be polite, concise, and solution-oriented."
            ),
        },
        {
            "role": "user",
            "content": "Hi, I placed order #{order_id} and haven't received it yet.",
        },
        {
            "role": "assistant",
            "content": (
                "I'm sorry to hear that. Let me look up your order right away. "
                "Can you confirm the email address on the account?"
            ),
        },
        {"role": "user", "content": "Sure, it's {email}."},
        {
            "role": "assistant",
            "content": (
                "Thank you, {first_name}. I can see your order is currently "
                "in transit and should arrive by {date}. "
                "Would you like me to send you the tracking link?"
            ),
        },
        {"role": "user", "content": "Yes please!"},
        {
            "role": "assistant",
            "content": "I've sent the tracking link to {email}. Is there anything else I can help you with?",
        },
    ],
    "code_assistant": [
        {
            "role": "system",
            "content": (
                "You are an expert Python developer. Provide clean, well-typed, "
                "idiomatic Python code with brief explanations."
            ),
        },
        {
            "role": "user",
            "content": "How do I {task} in Python?",
        },
        {
            "role": "assistant",
            "content": (
                "Here's how to {task} in Python:\n\n"
                "```python\n"
                "def {function_name}(items: list[str]) -> dict[str, int]:\n"
                "    \"\"\"Return a frequency map of items.\"\"\"\n"
                "    return {{item: items.count(item) for item in set(items)}}\n"
                "```\n\n"
                "This approach uses a dictionary comprehension for O(n) clarity."
            ),
        },
        {"role": "user", "content": "What if the list is very large?"},
        {
            "role": "assistant",
            "content": (
                "For large lists use `collections.Counter` — it's implemented in C "
                "and far more efficient:\n\n"
                "```python\n"
                "from collections import Counter\n\n"
                "counter = Counter(items)  # O(n), C-speed\n"
                "```"
            ),
        },
    ],
    "research_assistant": [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Summarise academic papers accurately, "
                "cite sources, and flag uncertainty explicitly."
            ),
        },
        {
            "role": "user",
            "content": "Can you summarise the key findings of the paper titled '{paper_title}'?",
        },
        {
            "role": "assistant",
            "content": (
                "The paper '{paper_title}' (published {year}) reports three main findings: "
                "1) {finding_one}; 2) {finding_two}; 3) {finding_three}. "
                "The authors conclude that {conclusion}. "
                "Note: I'm summarising from a truncated abstract — please verify with the full text."
            ),
        },
        {"role": "user", "content": "What methodology did they use?"},
        {
            "role": "assistant",
            "content": (
                "The study employed a {methodology} design with a sample of {count} participants "
                "over {duration}. Data were analysed using {analysis_method}."
            ),
        },
    ],
}

# ---------------------------------------------------------------------------
# Tool-call templates
# Each entry describes a canonical tool call schema.
# ---------------------------------------------------------------------------

TOOL_CALL_TEMPLATES: dict[str, dict[str, object]] = {
    "search": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return.",
                    "default": 5,
                },
                "safe_search": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
    "email": {
        "name": "send_email",
        "description": "Send an email to one or more recipients.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recipient email addresses.",
                },
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    "database": {
        "name": "execute_query",
        "description": "Run a parameterised SQL query against the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Parameterised SQL query."},
                "params": {
                    "type": "array",
                    "items": {},
                    "description": "Positional parameters for the query.",
                    "default": [],
                },
                "database": {
                    "type": "string",
                    "description": "Target database name.",
                },
            },
            "required": ["query"],
        },
    },
    "file_operations": {
        "name": "file_operation",
        "description": "Read, write, or list files on the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "delete"],
                },
                "path": {"type": "string", "description": "Absolute file path."},
                "content": {
                    "type": "string",
                    "description": "Content to write (write operation only).",
                },
                "encoding": {"type": "string", "default": "utf-8"},
            },
            "required": ["operation", "path"],
        },
    },
}

__all__ = ["CONVERSATION_TEMPLATES", "TOOL_CALL_TEMPLATES"]
