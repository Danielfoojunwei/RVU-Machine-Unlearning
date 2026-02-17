"""Tests for the Adapter Safety Gate.

Validates Phase 3: allowlist checks, Safe LoRA projection, OOD detection,
and combined screening.

Theorem references:
    - Principle 5 (Defense in Depth): multi-layer screening
    - Lermen et al. (2023): gate must block adapters that disable safety
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rvu.defenses.adapter_gate import (
    AdapterGate,
    GateDecision,
    OODResult,
    SafetyProjectionResult,
)


@pytest.fixture
def gate():
    """Create a default AdapterGate."""
    return AdapterGate()


@pytest.fixture
def gate_with_config(tmp_path: Path):
    """Create a gate with a YAML config."""
    config = tmp_path / "adapters.yaml"
    config.write_text("""
adapter_policy:
  allow_sources:
    - "local"
    - "huggingface"
  blocked_repos:
    - "malicious-user/*"
    - "evil-model"
  blocked_hashes:
    - "deadbeef0000"
  require_hash_verification: true
  max_concurrent_adapters: 2

safety_projection:
  enabled: true
  rejection_threshold: 0.15
  safety_subspace_dim: 32

ood_detection:
  enabled: true
  threshold: 3.0
  calibration_samples: 100
""")
    return AdapterGate(config_path=str(config))


class TestAllowlistChecks:
    """Test static allowlist screening."""

    def test_allowed_source(self, gate: AdapterGate) -> None:
        result = gate.check_allowlist("my-adapter", "hash123", "local")
        assert result.allowed is True

    def test_blocked_source(self, gate: AdapterGate) -> None:
        result = gate.check_allowlist("my-adapter", "hash123", "untrusted_api")
        assert result.allowed is False
        assert "source" in result.reason

    def test_blocked_repo_pattern(self, gate_with_config: AdapterGate) -> None:
        result = gate_with_config.check_allowlist(
            "malicious-user/bad-lora", "hash", "local"
        )
        assert result.allowed is False
        assert "blocked repo" in result.reason

    def test_blocked_exact_name(self, gate_with_config: AdapterGate) -> None:
        result = gate_with_config.check_allowlist("evil-model", "hash", "local")
        assert result.allowed is False

    def test_blocked_hash(self, gate_with_config: AdapterGate) -> None:
        result = gate_with_config.check_allowlist(
            "good-name", "deadbeef0000", "local"
        )
        assert result.allowed is False
        assert "hash" in result.reason

    def test_allowed_passes_all(self, gate_with_config: AdapterGate) -> None:
        result = gate_with_config.check_allowlist(
            "good-adapter", "safe-hash", "huggingface"
        )
        assert result.allowed is True


class TestSafetyProjection:
    """Test Safe LoRA weight projection (NeurIPS 2024)."""

    def test_zero_weights_are_safe(self, gate: AdapterGate) -> None:
        weights = {"layer.weight": np.zeros((10, 10), dtype=np.float32)}
        result = gate.compute_safety_projection(weights)
        assert result.safety_score == 1.0
        assert result.total_magnitude == 0.0

    def test_random_weights_have_score(self, gate: AdapterGate) -> None:
        rng = np.random.RandomState(42)
        weights = {"layer.weight": rng.randn(64, 64).astype(np.float32)}
        result = gate.compute_safety_projection(weights)
        assert 0.0 <= result.safety_score <= 1.0
        assert result.total_magnitude > 0.0

    def test_empty_weights(self, gate: AdapterGate) -> None:
        result = gate.compute_safety_projection({})
        assert result.safety_score == 1.0

    def test_with_base_weights(self, gate: AdapterGate) -> None:
        rng = np.random.RandomState(42)
        base = {"l.w": rng.randn(32, 32).astype(np.float32)}
        adapter = {"l.w": base["l.w"] + 0.01 * rng.randn(32, 32).astype(np.float32)}
        result = gate.compute_safety_projection(adapter, base)
        # Small perturbation should have high safety score.
        assert result.safety_score > 0.0


class TestOODDetection:
    """Test Mahalanobis distance OOD detection (OOO, 2024)."""

    def test_calibrate_and_score_in_distribution(self, gate: AdapterGate) -> None:
        rng = np.random.RandomState(42)
        base_embs = rng.randn(100, 16).astype(np.float32)
        gate.calibrate_ood(base_embs)

        # Similar distribution should have low OOD score.
        adapted_embs = rng.randn(20, 16).astype(np.float32) * 1.1
        result = gate.compute_ood_score(adapted_embs)
        assert isinstance(result, OODResult)
        assert result.calibration_samples == 500  # from config default

    def test_uncalibrated_returns_zero(self, gate: AdapterGate) -> None:
        adapted_embs = np.random.randn(10, 16).astype(np.float32)
        result = gate.compute_ood_score(adapted_embs)
        assert result.ood_score == 0.0
        assert result.is_anomalous is False

    def test_shifted_distribution_is_anomalous(self, gate: AdapterGate) -> None:
        rng = np.random.RandomState(42)
        base_embs = rng.randn(200, 16).astype(np.float32)
        gate.calibrate_ood(base_embs)

        # Large shift.
        adapted_embs = rng.randn(20, 16).astype(np.float32) + 100.0
        result = gate.compute_ood_score(adapted_embs)
        assert result.is_anomalous is True
        assert result.mahalanobis_distance > gate._ood_threshold


class TestCombinedScreening:
    """Test the full screening pipeline."""

    def test_combined_passes_clean_adapter(self, gate: AdapterGate) -> None:
        result = gate.screen_adapter(
            adapter_name="clean-lora",
            adapter_hash="safe-hash",
            source="local",
        )
        assert result.allowed is True
        assert result.method == "combined"

    def test_combined_blocks_bad_source(self, gate: AdapterGate) -> None:
        result = gate.screen_adapter(
            adapter_name="adapter",
            adapter_hash="hash",
            source="malware_cdn",
        )
        assert result.allowed is False
        assert result.method == "allowlist"

    def test_gate_decision_has_timestamp(self, gate: AdapterGate) -> None:
        result = gate.screen_adapter("a", "h", "local")
        assert result.timestamp > 0

    def test_risk_score_in_range(self, gate: AdapterGate) -> None:
        result = gate.screen_adapter("a", "h", "local")
        assert 0.0 <= result.risk_score <= 1.0
