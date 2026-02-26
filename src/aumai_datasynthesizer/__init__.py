"""aumai-datasynthesizer — synthetic training data for agent testing."""

from aumai_datasynthesizer.core import DataGenerator, SchemaBasedGenerator
from aumai_datasynthesizer.models import (
    ConversationTurn,
    DataType,
    GeneratorConfig,
    SyntheticDataset,
)

__version__ = "0.1.0"

__all__ = [
    "DataGenerator",
    "SchemaBasedGenerator",
    "ConversationTurn",
    "DataType",
    "GeneratorConfig",
    "SyntheticDataset",
]
