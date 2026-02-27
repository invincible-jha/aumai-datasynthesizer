"""Comprehensive CLI tests for aumai-datasynthesizer."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aumai_datasynthesizer.cli import main


# ---------------------------------------------------------------------------
# Version / help
# ---------------------------------------------------------------------------


class TestCliMeta:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "DataSynthesizer" in result.output or "generate" in result.output


# ---------------------------------------------------------------------------
# `generate` command
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    def test_generate_requires_type(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["generate"])
        assert result.exit_code != 0

    def test_generate_text_stdout(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", "text", "--count", "3", "--seed", "42"],
        )
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj

    def test_generate_json_stdout(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", "json", "--count", "2", "--seed", "1"],
        )
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
        assert len(lines) == 2

    def test_generate_conversation_stdout(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", "conversation", "--count", "2", "--seed", "5"],
        )
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "turns" in obj

    def test_generate_tool_call_stdout(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", "tool_call", "--count", "2", "--seed", "5"],
        )
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
        assert len(lines) == 2

    def test_generate_agent_trace_stdout(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", "agent_trace", "--count", "2", "--seed", "5"],
        )
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
        assert len(lines) == 2

    def test_generate_to_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--type", "text",
                    "--count", "4",
                    "--seed", "10",
                    "--output", "output.jsonl",
                ],
            )
            assert result.exit_code == 0
            content = Path("output.jsonl").read_text(encoding="utf-8")
            lines = [l for l in content.strip().split("\n") if l.strip()]
            assert len(lines) == 4

    def test_generate_with_constraint(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--type", "conversation",
                "--count", "2",
                "--seed", "1",
                "--constraint", "template=code_assistant",
            ],
        )
        assert result.exit_code == 0

    def test_generate_invalid_constraint_no_equals(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate",
                "--type", "text",
                "--count", "2",
                "--constraint", "NOEQUALS",
            ],
        )
        assert result.exit_code != 0

    def test_generate_with_schema_file(self) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem():
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": ["name", "score"],
            }
            Path("schema.json").write_text(json.dumps(schema), encoding="utf-8")
            result = runner.invoke(
                main,
                [
                    "generate",
                    "--type", "json",
                    "--count", "3",
                    "--seed", "1",
                    "--schema", "schema.json",
                ],
            )
            assert result.exit_code == 0
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            assert len(lines) == 3
            for line in lines:
                obj = json.loads(line)
                assert "name" in obj
                assert "score" in obj

    def test_generate_count_min_1(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["generate", "--type", "text", "--count", "0"]
        )
        assert result.exit_code != 0

    def test_generate_invalid_type(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["generate", "--type", "invalid_type"]
        )
        assert result.exit_code != 0

    def test_generate_stderr_reports_count(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", "text", "--count", "5", "--seed", "1"],
        )
        assert result.exit_code == 0

    @pytest.mark.parametrize("data_type", ["text", "json", "conversation", "tool_call", "agent_trace"])
    def test_all_data_types_generate(self, data_type: str) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate", "--type", data_type, "--count", "1", "--seed", "42"],
        )
        assert result.exit_code == 0
        lines = [ln for ln in result.stdout.strip().split("\n") if ln.strip()]
        assert len(lines) == 1
        json.loads(lines[0])  # Should be valid JSON


# ---------------------------------------------------------------------------
# `templates` command
# ---------------------------------------------------------------------------


class TestTemplatesCommand:
    def test_templates_requires_list_or_category(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates"])
        assert result.exit_code != 0

    def test_templates_list_all(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates", "--list"])
        assert result.exit_code == 0
        assert "customer_support" in result.output
        assert "code_assistant" in result.output
        assert "research_assistant" in result.output

    def test_templates_list_shows_tool_call_templates(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates", "--list"])
        assert result.exit_code == 0
        assert "search" in result.output
        assert "email" in result.output

    def test_templates_category_conversation(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates", "--category", "conversation"])
        assert result.exit_code == 0
        assert "Conversation Templates" in result.output

    def test_templates_category_tool_call(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates", "--category", "tool_call"])
        assert result.exit_code == 0
        assert "Tool-Call Templates" in result.output

    def test_templates_category_conversation_excludes_tool_call(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates", "--category", "conversation"])
        assert result.exit_code == 0
        # The tool-call section header should NOT appear
        assert "Tool-Call Templates" not in result.output

    def test_templates_list_shows_turn_counts(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["templates", "--list"])
        assert result.exit_code == 0
        # Turn counts appear as "(N turns)" in output
        assert "turns" in result.output
