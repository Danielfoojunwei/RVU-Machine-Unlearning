"""Tests for the regression_compare module.

Verifies the compare_metrics() function with known golden/current values,
checking both pass and fail logic against configurable tolerances.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Add scripts/ to the Python path so we can import regression_compare.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from regression_compare import (
    ComparisonResult,
    _DEFAULT_TOLERANCES,
    _FALLBACK_TOLERANCE,
    compare_metrics,
)


class TestCompareMetricsPassLogic:
    """Tests that compare_metrics correctly identifies passing metrics."""

    def test_identical_metrics_pass(self) -> None:
        """Identical golden and current values should all pass."""
        golden = {
            "attack_success_rate": 0.10,
            "utility_success_rate": 0.85,
            "mean_wall_time_s": 2.5,
        }
        current = dict(golden)  # Exact copy.
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        for r in results:
            assert r.passed, f"Metric '{r.metric}' should pass but got {r.status}"

    def test_within_absolute_tolerance(self) -> None:
        """Metrics within absolute tolerance should pass."""
        golden = {
            "attack_success_rate": 0.10,
            "utility_success_rate": 0.85,
        }
        current = {
            "attack_success_rate": 0.12,   # +0.02, within abs=0.03
            "utility_success_rate": 0.82,  # -0.03, within abs=0.05
        }
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        for r in results:
            assert r.passed, f"Metric '{r.metric}' should pass within tolerance"

    def test_within_relative_tolerance(self) -> None:
        """Metrics within relative tolerance should pass."""
        golden = {
            "mean_wall_time_s": 2.0,
            "mean_tokens": 100.0,
        }
        current = {
            "mean_wall_time_s": 2.8,  # +40%, within rel=50%
            "mean_tokens": 130.0,     # +30%, within rel=50%
        }
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        for r in results:
            assert r.passed, f"Metric '{r.metric}' should pass within relative tolerance"

    def test_new_metric_in_current_passes(self) -> None:
        """A metric only in current (not in golden) should pass as new."""
        golden = {"attack_success_rate": 0.10}
        current = {"attack_success_rate": 0.10, "new_metric": 0.5}
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        new_result = [r for r in results if r.metric == "new_metric"]
        assert len(new_result) == 1
        assert new_result[0].passed
        assert "new metric" in new_result[0].tolerance_desc


class TestCompareMetricsFailLogic:
    """Tests that compare_metrics correctly identifies failing metrics."""

    def test_outside_absolute_tolerance_fails(self) -> None:
        """Metrics outside absolute tolerance should fail."""
        golden = {"attack_success_rate": 0.10}
        current = {"attack_success_rate": 0.20}  # +0.10, exceeds abs=0.03
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        assert len(results) == 1
        assert not results[0].passed, "Should fail: delta 0.10 exceeds abs tolerance 0.03"

    def test_outside_relative_tolerance_fails(self) -> None:
        """Metrics outside relative tolerance should fail."""
        golden = {"mean_wall_time_s": 2.0}
        current = {"mean_wall_time_s": 5.0}  # +150%, exceeds rel=50%
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        assert len(results) == 1
        assert not results[0].passed, "Should fail: 150% delta exceeds 50% relative tolerance"

    def test_missing_metric_in_current_fails(self) -> None:
        """A metric in golden but missing from current should fail."""
        golden = {
            "attack_success_rate": 0.10,
            "utility_success_rate": 0.85,
        }
        current = {"attack_success_rate": 0.10}
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        missing = [r for r in results if r.metric == "utility_success_rate"]
        assert len(missing) == 1
        assert not missing[0].passed
        assert "MISSING" in missing[0].tolerance_desc

    def test_large_regression_detected(self) -> None:
        """A significant regression across multiple metrics should be detected."""
        golden = {
            "attack_success_rate": 0.05,
            "utility_success_rate": 0.90,
            "auditor_pass_rate": 0.98,
        }
        current = {
            "attack_success_rate": 0.30,   # Big regression
            "utility_success_rate": 0.50,  # Big regression
            "auditor_pass_rate": 0.60,     # Big regression
        }
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        failures = [r for r in results if not r.passed]
        assert len(failures) == 3, f"Expected 3 failures, got {len(failures)}"


class TestCompareMetricsEdgeCases:
    """Edge cases for the comparison logic."""

    def test_empty_golden_returns_empty(self) -> None:
        """No metrics in golden means nothing to compare."""
        results = compare_metrics({}, {"foo": 1.0}, _DEFAULT_TOLERANCES)
        # The new metric in current should still appear as passing.
        assert all(r.passed for r in results)

    def test_both_empty_returns_empty(self) -> None:
        """Both empty dicts should produce an empty result."""
        results = compare_metrics({}, {}, _DEFAULT_TOLERANCES)
        assert results == []

    def test_zero_golden_value(self) -> None:
        """Golden value of zero with non-zero current uses absolute tolerance."""
        golden = {"utility_success_rate": 0.0}
        current = {"utility_success_rate": 0.04}  # Within abs=0.05
        results = compare_metrics(golden, current, _DEFAULT_TOLERANCES)

        assert len(results) == 1
        assert results[0].passed

    def test_custom_tolerance_overrides_default(self) -> None:
        """Custom tolerances should override the defaults."""
        golden = {"attack_success_rate": 0.10}
        current = {"attack_success_rate": 0.20}  # +0.10

        # With default tolerance (abs=0.03), this should fail.
        results_default = compare_metrics(golden, current, _DEFAULT_TOLERANCES)
        assert not results_default[0].passed

        # With a wider custom tolerance, it should pass.
        custom_tolerances = dict(_DEFAULT_TOLERANCES)
        custom_tolerances["attack_success_rate"] = {"abs": 0.15, "rel": None}
        results_custom = compare_metrics(golden, current, custom_tolerances)
        assert results_custom[0].passed

    def test_comparison_result_status_property(self) -> None:
        """ComparisonResult.status returns correct string."""
        r_pass = ComparisonResult(
            metric="test", golden=1.0, current=1.0,
            delta=0.0, tolerance_desc="abs<=0.05", passed=True,
        )
        assert r_pass.status == "PASS"

        r_fail = ComparisonResult(
            metric="test", golden=1.0, current=2.0,
            delta=1.0, tolerance_desc="abs<=0.05", passed=False,
        )
        assert r_fail.status == "FAIL"
