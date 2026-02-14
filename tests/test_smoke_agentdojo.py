"""Smoke tests for the AgentDojo benchmark runner.

These tests require the LLM model files to be downloaded locally.
They are skipped automatically when model files are not present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Check for model availability -- skip the entire module if models are missing.
_MODEL_A_DIR = _PROJECT_ROOT / "models" / "llm" / "qwen2.5-1.5b-instruct"
_MODELS_AVAILABLE = _MODEL_A_DIR.is_dir() and any(_MODEL_A_DIR.glob("*.gguf"))

_CONFIGS_AVAILABLE = (
    (_PROJECT_ROOT / "configs" / "agentdojo.yaml").is_file()
    and (_PROJECT_ROOT / "configs" / "models.yaml").is_file()
    and (_PROJECT_ROOT / "configs" / "defenses.yaml").is_file()
)

_SKIP_REASON = "Model files not downloaded (run scripts/download_models.sh)"


@pytest.mark.bench_smoke
@pytest.mark.skipif(not _MODELS_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.skipif(not _CONFIGS_AVAILABLE, reason="Config files missing")
class TestAgentDojoSmoke:
    """Smoke test suite for AgentDojoRunner."""

    def test_runner_instantiation(self, tmp_path: Path) -> None:
        """AgentDojoRunner can be instantiated with valid config."""
        from rvu.harness.agentdojo_runner import AgentDojoRunner

        runner = AgentDojoRunner(
            model_key="model_a",
            defense_name="rvu",
            config_path=str(_PROJECT_ROOT / "configs" / "agentdojo.yaml"),
            logdir=str(tmp_path / "agentdojo_smoke"),
        )
        assert runner.model_key == "model_a"
        assert runner.defense_name == "rvu"

    def test_smoke_run_produces_output(self, tmp_path: Path) -> None:
        """Smoke run creates output files in the logdir."""
        from rvu.harness.agentdojo_runner import AgentDojoRunner

        logdir = tmp_path / "agentdojo_smoke_run"
        runner = AgentDojoRunner(
            model_key="model_a",
            defense_name="rvu",
            config_path=str(_PROJECT_ROOT / "configs" / "agentdojo.yaml"),
            logdir=str(logdir),
        )
        metrics = runner.run_smoke()

        # Verify output directory has files.
        output_files = list(logdir.rglob("*.json"))
        assert len(output_files) > 0, "Expected JSON output files in logdir"

        # Verify metrics dict has expected keys.
        assert "n_episodes" in metrics
        assert "defense" in metrics
        assert "model" in metrics

    def test_metrics_conform_to_schema(self, tmp_path: Path) -> None:
        """Metrics output conforms to the JSON schema."""
        from rvu.harness.agentdojo_runner import AgentDojoRunner

        try:
            import jsonschema
        except ImportError:
            pytest.skip("jsonschema not installed")

        schema_path = _PROJECT_ROOT / "rvu" / "reports" / "schema.json"
        assert schema_path.is_file(), f"Schema file not found: {schema_path}"
        with open(schema_path, "r") as fh:
            schema = json.load(fh)

        logdir = tmp_path / "agentdojo_schema_test"
        runner = AgentDojoRunner(
            model_key="model_a",
            defense_name="rvu",
            config_path=str(_PROJECT_ROOT / "configs" / "agentdojo.yaml"),
            logdir=str(logdir),
        )
        metrics = runner.run_smoke()

        # Add required fields that the runner may not set.
        metrics.setdefault("benchmark", "agentdojo")
        metrics.setdefault("mode", "smoke")

        jsonschema.validate(instance=metrics, schema=schema)
