"""Adapter Influence Estimator — approximate LoRA adapter impact on outputs.

Estimates how much a LoRA adapter influenced specific model outputs WITHOUT
requiring backpropagation access (critical for GGUF quantized models served
via llama.cpp where gradient computation is unavailable).

Theoretical basis:
    - NTU DTC "Privacy-Preserving Federated Unlearning with Certified Client
      Removal" (IEEE TIFS 2025): Influence function I(client) ≈ H⁻¹∇L
      approximated without full Hessian access.
    - NTU DTC "Enhancing AI Safety of MU for Ensembled Models" (Applied Soft
      Computing 2025): When multiple adapters co-activate, influence estimation
      must account for interaction effects.
    - Theorem 3 (Risk Monotonicity): I(e) feeds into FROC risk function R(e)
      as the influence component.

Proxy method (no gradient access):
    Instead of computing influence functions (which require ∇²L), we use an
    output-level proxy:

    I(adapter_a, input_j) ≈ 1 - sim(embed(base_output_j), embed(adapted_output_j))

    where sim(·,·) is cosine similarity.  High divergence between base and
    adapted outputs indicates high adapter influence.

Benchmarks:
    - TOFU (Maini et al., 2024): Forget/retain set split for measuring
      whether adapter influence is concentrated on specific knowledge.
    - WMDP (Li et al., 2024): Hazardous knowledge probes to detect if
      adapter influence is concentrated on dangerous topics.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rvu.defenses.base import sha256_hex, timestamp_now


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InfluenceScore:
    """Influence score for a single (adapter, input) pair."""

    adapter_id: str
    input_text: str
    input_hash: str
    base_output: str
    adapted_output: str
    embedding_distance: float  # 1 - cosine_similarity
    token_divergence: float    # 1 - Jaccard(base_tokens, adapted_tokens)
    combined_influence: float  # weighted combination


@dataclass
class InfluenceReport:
    """Aggregate influence report for an adapter across a probe set.

    The aggregate influence feeds directly into the FROC risk function
    (Theorem 3): R(e) = w_c * P + w_i * I(e) + w_p * propagation.
    """

    adapter_id: str
    adapter_name: str
    num_probes: int
    per_input_scores: list[InfluenceScore]
    mean_embedding_distance: float
    mean_token_divergence: float
    max_embedding_distance: float
    aggregate_influence: float  # The I(e) value for FROC
    timestamp: float = field(default_factory=timestamp_now)
    probe_categories: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class AdapterInfluenceEstimator:
    """Estimate adapter influence on model outputs without gradient access.

    Parameters
    ----------
    embedder:
        A sentence-transformers model (or any object with an ``encode``
        method) for computing output embeddings.
    embedding_weight:
        Weight for embedding distance in combined influence score.
    token_weight:
        Weight for token divergence in combined influence score.
    """

    def __init__(
        self,
        embedder: Any,
        embedding_weight: float = 0.7,
        token_weight: float = 0.3,
    ) -> None:
        self._embedder = embedder
        self._embedding_weight = embedding_weight
        self._token_weight = token_weight
        self._cache: dict[str, InfluenceReport] = {}

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def _token_divergence(text_a: str, text_b: str) -> float:
        """Compute 1 - Jaccard similarity between token sets.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        Returns 0.0 for identical outputs, 1.0 for completely disjoint.
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a and not tokens_b:
            return 0.0
        union = tokens_a | tokens_b
        if not union:
            return 0.0
        intersection = tokens_a & tokens_b
        return 1.0 - len(intersection) / len(union)

    def _embedding_distance(self, text_a: str, text_b: str) -> float:
        """Compute 1 - cosine_similarity between text embeddings.

        Uses the sentence-transformers embedder to encode both texts
        and compute cosine similarity.  Returns 0.0 for identical
        embeddings, approaching 2.0 for opposite embeddings.
        """
        if self._embedder is None:
            return 0.0
        emb_a = self._embedder.encode(text_a, normalize_embeddings=True)
        emb_b = self._embedder.encode(text_b, normalize_embeddings=True)
        cos_sim = float(np.dot(emb_a, emb_b))
        return max(0.0, 1.0 - cos_sim)

    def score_single(
        self,
        adapter_id: str,
        input_text: str,
        base_output: str,
        adapted_output: str,
    ) -> InfluenceScore:
        """Compute influence score for a single (adapter, input) pair.

        Combines embedding distance and token divergence:
            combined = w_embed * emb_dist + w_token * tok_div

        This proxy approximation is justified by the NTU DTC federated
        unlearning paper's finding that output-level divergence is
        correlated with gradient-based influence for adapter-scale
        perturbations.
        """
        emb_dist = self._embedding_distance(base_output, adapted_output)
        tok_div = self._token_divergence(base_output, adapted_output)
        combined = (
            self._embedding_weight * emb_dist
            + self._token_weight * tok_div
        )

        return InfluenceScore(
            adapter_id=adapter_id,
            input_text=input_text,
            input_hash=sha256_hex(input_text),
            base_output=base_output,
            adapted_output=adapted_output,
            embedding_distance=emb_dist,
            token_divergence=tok_div,
            combined_influence=combined,
        )

    def estimate_influence(
        self,
        adapter_id: str,
        adapter_name: str,
        probe_inputs: list[str],
        base_outputs: list[str],
        adapted_outputs: list[str],
        probe_categories: dict[str, list[int]] | None = None,
    ) -> InfluenceReport:
        """Estimate aggregate influence of an adapter across a probe set.

        Parameters
        ----------
        adapter_id:
            Unique identifier for the adapter.
        adapter_name:
            Human-readable name.
        probe_inputs:
            List of probe prompt strings.
        base_outputs:
            List of base model outputs (same order as probe_inputs).
        adapted_outputs:
            List of adapted model outputs (same order as probe_inputs).
        probe_categories:
            Optional mapping of category names to indices into probe_inputs.
            Used to compute per-category influence for WMDP domains
            (biosecurity, cybersecurity, chemical security) or TOFU
            (forget set, retain set).

        Returns
        -------
        InfluenceReport
            Contains per-input scores and aggregates.  The
            ``aggregate_influence`` field is the I(e) value for the
            FROC risk function (Theorem 3).
        """
        if len(probe_inputs) != len(base_outputs) or len(probe_inputs) != len(adapted_outputs):
            raise ValueError(
                "probe_inputs, base_outputs, adapted_outputs must have the same length"
            )

        scores: list[InfluenceScore] = []
        for inp, base_out, adapted_out in zip(probe_inputs, base_outputs, adapted_outputs):
            score = self.score_single(adapter_id, inp, base_out, adapted_out)
            scores.append(score)

        if not scores:
            return InfluenceReport(
                adapter_id=adapter_id, adapter_name=adapter_name,
                num_probes=0, per_input_scores=[],
                mean_embedding_distance=0.0, mean_token_divergence=0.0,
                max_embedding_distance=0.0, aggregate_influence=0.0,
            )

        emb_dists = [s.embedding_distance for s in scores]
        tok_divs = [s.token_divergence for s in scores]
        combined = [s.combined_influence for s in scores]

        # Per-category breakdown.
        cat_scores: dict[str, float] = {}
        if probe_categories:
            for cat_name, indices in probe_categories.items():
                cat_combined = [combined[i] for i in indices if i < len(combined)]
                if cat_combined:
                    cat_scores[cat_name] = float(np.mean(cat_combined))

        report = InfluenceReport(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            num_probes=len(scores),
            per_input_scores=scores,
            mean_embedding_distance=float(np.mean(emb_dists)),
            mean_token_divergence=float(np.mean(tok_divs)),
            max_embedding_distance=float(np.max(emb_dists)),
            aggregate_influence=float(np.mean(combined)),
            probe_categories=cat_scores,
        )

        # Cache for reuse in risk scoring.
        self._cache[adapter_id] = report
        return report

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def get_cached_influence(self, adapter_id: str) -> float:
        """Return cached aggregate influence for an adapter.

        Returns 0.0 if no cached report exists.  Used by the FROC risk
        function (Theorem 3) to avoid recomputing influence on every
        risk score query.
        """
        report = self._cache.get(adapter_id)
        if report is None:
            return 0.0
        return report.aggregate_influence

    def get_cached_report(self, adapter_id: str) -> InfluenceReport | None:
        """Return the full cached influence report."""
        return self._cache.get(adapter_id)

    def clear_cache(self) -> int:
        """Clear the influence cache.  Returns count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        return count
