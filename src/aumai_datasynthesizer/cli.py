"""CLI entry point for aumai-datasynthesizer."""

from __future__ import annotations

import json
import sys

import click

from aumai_datasynthesizer.core import DataGenerator
from aumai_datasynthesizer.models import DataType, GeneratorConfig
from aumai_datasynthesizer.templates import CONVERSATION_TEMPLATES, TOOL_CALL_TEMPLATES


@click.group()
@click.version_option()
def main() -> None:
    """AumAI DataSynthesizer — generate synthetic training data for agent testing."""


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


@main.command("generate")
@click.option(
    "--type",
    "data_type",
    type=click.Choice([dt.value for dt in DataType], case_sensitive=False),
    required=True,
    help="Type of data to generate.",
)
@click.option(
    "--count",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of samples to generate.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducible output.",
)
@click.option(
    "--output",
    default="-",
    type=click.Path(allow_dash=True),
    show_default=True,
    help="Output file path (.jsonl or .json). Use '-' for stdout.",
)
@click.option(
    "--schema",
    "schema_path",
    default=None,
    type=click.Path(exists=True, readable=True),
    help="Path to a JSON Schema file (used with --type json).",
)
@click.option(
    "--constraint",
    "constraint_pairs",
    multiple=True,
    metavar="KEY=VALUE",
    help="Constraint as KEY=VALUE pair. May be specified multiple times.",
)
def generate_cmd(
    data_type: str,
    count: int,
    seed: int | None,
    output: str,
    schema_path: str | None,
    constraint_pairs: tuple[str, ...],
) -> None:
    """Generate synthetic data samples and write them as JSON Lines."""
    schema: dict[str, object] | None = None
    if schema_path is not None:
        with open(schema_path, encoding="utf-8") as fh:
            schema = json.load(fh)

    constraints: dict[str, object] = {}
    for pair in constraint_pairs:
        if "=" not in pair:
            raise click.BadParameter(
                f"Constraint must be KEY=VALUE, got: {pair!r}",
                param_hint="--constraint",
            )
        key, _, value = pair.partition("=")
        constraints[key.strip()] = value.strip()

    config = GeneratorConfig(
        data_type=DataType(data_type),
        count=count,
        seed=seed,
        schema=schema,
        constraints=constraints,
    )

    generator = DataGenerator()
    dataset = generator.generate(config)

    click.echo(
        f"Generated {len(dataset.samples)} {data_type} samples "
        f"in {dataset.metadata.get('generation_time_ms', 0):.1f} ms.",
        err=True,
    )

    if output == "-":
        out_fh = sys.stdout
    else:
        out_fh = open(output, "w", encoding="utf-8")  # noqa: SIM115

    try:
        for sample in dataset.samples:
            out_fh.write(json.dumps(sample, default=str) + "\n")
    finally:
        if output != "-":
            out_fh.close()

    if output != "-":
        click.echo(f"Output written to: {output}", err=True)


# ---------------------------------------------------------------------------
# templates command
# ---------------------------------------------------------------------------


@main.command("templates")
@click.option(
    "--list",
    "list_all",
    is_flag=True,
    default=False,
    help="List all available built-in templates.",
)
@click.option(
    "--category",
    type=click.Choice(["conversation", "tool_call"], case_sensitive=False),
    default=None,
    help="Filter templates by category.",
)
def templates_cmd(list_all: bool, category: str | None) -> None:
    """List available conversation and tool-call templates."""
    if not list_all and category is None:
        raise click.UsageError("Specify --list or --category <category>.")

    show_conversation = category in (None, "conversation")
    show_tool_call = category in (None, "tool_call")

    if show_conversation:
        click.echo("\n--- Conversation Templates ---")
        for name, turns in CONVERSATION_TEMPLATES.items():
            click.echo(f"  {name}  ({len(turns)} turns)")

    if show_tool_call:
        click.echo("\n--- Tool-Call Templates ---")
        for name, spec in TOOL_CALL_TEMPLATES.items():
            click.echo(f"  {name}  -> {spec['name']}")

    click.echo("")


if __name__ == "__main__":
    main()
