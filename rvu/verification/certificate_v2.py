"""Certificate V2 — extended certificates with adapter attestation.

Extends the original RVU certificate schema to include:
1. Adapter lifecycle attestation (load, evict, fuse events)
2. Membership probe results (evidence of actual content removal)
3. Risk score breakdown (FROC-computed per-entry scores)
4. Quarantine records (entries flagged but not fully purged)

Theoretical basis:
    - NTU DTC "Certifying the Right to Be Forgotten" (arXiv 2512.23171):
      Formal certification that data was removed, with auditor verification.
    - Theorem 4 (Certified Recovery Integrity): SHA-256 over deterministic
      manifest ensures tamper detection.
    - NTU DTC "Threats, Attacks, and Defenses in MU" (IEEE OJCS 2025):
      Verification attacks mitigated by including probe results in cert.

Certificate V2 schema:
    {
      "version": 2,
      "runtime_recovery": { closure_ids, purged, quarantined },
      "adapter_attestation": { evicted, retained, fusion_warnings },
      "membership_probe": { verdict, pre/post similarity },
      "risk_scores": { per-entry FROC scores },
      "signature": SHA-256 over all of the above
    }
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rvu.defenses.base import constant_time_compare, sha256_hex, timestamp_now


# ---------------------------------------------------------------------------
# Data classes for certificate components
# ---------------------------------------------------------------------------

@dataclass
class AdapterAttestation:
    """Attestation record for adapter lifecycle during recovery."""

    adapter_id: str
    adapter_name: str
    adapter_hash: str
    action: str           # evicted | retained | fused_warning
    risk_score: float
    was_fused: bool
    eviction_method: str  # unload | not_applicable | irreversible


@dataclass
class RuntimeRecovery:
    """Runtime state recovery summary for the certificate."""

    closure_ids: list[str]
    manifest_entries: list[dict[str, Any]]
    total_purged: int
    total_quarantined: int
    quarantine_ids: list[str]


@dataclass
class MembershipProbeAttestation:
    """Membership probe results for auditor verification."""

    probe_type: str  # output_similarity
    num_probes: int
    pre_recovery_similarity: float | None
    post_recovery_similarity: float
    probe_verdict: str  # PASS | FAIL | PARTIAL
    breakdown: dict[str, float]


# ---------------------------------------------------------------------------
# Certificate V2
# ---------------------------------------------------------------------------

class CertificateV2Emitter:
    """Emit extended V2 certificates with adapter attestation.

    Parameters
    ----------
    certificate_dir:
        Directory where certificate JSON files are saved.
    """

    def __init__(self, certificate_dir: str | Path) -> None:
        self.certificate_dir = Path(certificate_dir).expanduser().resolve()
        self.certificate_dir.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        episode_id: str,
        runtime_recovery: RuntimeRecovery,
        adapter_attestations: list[AdapterAttestation] | None = None,
        membership_probe: MembershipProbeAttestation | None = None,
        risk_scores: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Emit a V2 certificate.

        The certificate is a deterministic JSON document signed with
        SHA-256.  Determinism is ensured by sorting all keys and lists.

        Per Theorem 4: the signature is collision-resistant (birthday
        bound 2^128), and the manifest is deterministic (sorted keys,
        sorted closure_ids, sorted adapter lists).
        """
        cert_id = f"cert-v2-{uuid.uuid4().hex[:16]}"
        now = timestamp_now()

        # Build the deterministic manifest.
        manifest: dict[str, Any] = {
            "certificate_id": cert_id,
            "version": 2,
            "timestamp": now,
            "episode_id": episode_id,
        }

        # Runtime recovery section.
        manifest["runtime_recovery"] = {
            "closure_ids": sorted(runtime_recovery.closure_ids),
            "manifest_entries": sorted(
                runtime_recovery.manifest_entries,
                key=lambda e: e.get("entry_id", ""),
            ),
            "total_purged": runtime_recovery.total_purged,
            "total_quarantined": runtime_recovery.total_quarantined,
            "quarantine_ids": sorted(runtime_recovery.quarantine_ids),
        }

        # Adapter attestation section.
        if adapter_attestations:
            evicted = []
            retained = []
            fusion_warnings = []
            for att in sorted(adapter_attestations, key=lambda a: a.adapter_id):
                entry = {
                    "adapter_id": att.adapter_id,
                    "adapter_name": att.adapter_name,
                    "adapter_hash": att.adapter_hash,
                    "risk_score": att.risk_score,
                    "was_fused": att.was_fused,
                    "eviction_method": att.eviction_method,
                }
                if att.action == "evicted":
                    evicted.append(entry)
                elif att.action == "retained":
                    retained.append(entry)
                elif att.action == "fused_warning":
                    fusion_warnings.append(entry)

            manifest["adapter_attestation"] = {
                "adapters_evicted": evicted,
                "adapters_retained": retained,
                "fusion_warnings": fusion_warnings,
            }
        else:
            manifest["adapter_attestation"] = {
                "adapters_evicted": [],
                "adapters_retained": [],
                "fusion_warnings": [],
            }

        # Membership probe section.
        if membership_probe:
            manifest["membership_probe"] = {
                "probe_type": membership_probe.probe_type,
                "num_probes": membership_probe.num_probes,
                "pre_recovery_similarity": membership_probe.pre_recovery_similarity,
                "post_recovery_similarity": membership_probe.post_recovery_similarity,
                "probe_verdict": membership_probe.probe_verdict,
                "breakdown": membership_probe.breakdown,
            }
        else:
            manifest["membership_probe"] = None

        # Risk scores section.
        if risk_scores:
            manifest["risk_scores"] = {
                k: v for k, v in sorted(risk_scores.items())
            }
        else:
            manifest["risk_scores"] = {}

        # Compute SHA-256 signature over the deterministic manifest.
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        signature = hashlib.sha256(manifest_bytes).hexdigest()

        certificate: dict[str, Any] = {
            **manifest,
            "signature": signature,
        }

        # Save to disk.
        cert_path = self.certificate_dir / f"{cert_id}.json"
        with open(cert_path, "w") as fh:
            json.dump(certificate, fh, indent=2, sort_keys=True)

        certificate["certificate_path"] = str(cert_path)
        return certificate


class CertificateV2Verifier:
    """Verify V2 certificates with adapter and membership attestation.

    Per Theorem 4 and NTU DTC "Certifying the Right to Be Forgotten":
    the auditor recomputes the SHA-256 signature and cross-references
    the manifest against the provenance and adapter databases.
    """

    @staticmethod
    def verify_signature(certificate: dict[str, Any]) -> bool:
        """Verify the SHA-256 signature integrity.

        Recomputes SHA-256 over the manifest (excluding signature and
        certificate_path) and compares with the stored signature.
        Uses constant-time comparison to prevent timing side-channels.
        """
        stored_sig = certificate.get("signature")
        if not stored_sig:
            return False

        manifest = {
            k: v for k, v in certificate.items()
            if k not in ("signature", "certificate_path")
        }
        manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
        expected_sig = hashlib.sha256(manifest_bytes).hexdigest()
        return constant_time_compare(stored_sig, expected_sig)

    @staticmethod
    def verify_completeness(certificate: dict[str, Any]) -> tuple[bool, list[str]]:
        """Verify that all required V2 fields are present.

        Returns (is_complete, list_of_missing_fields).
        """
        required = {
            "certificate_id", "version", "timestamp", "episode_id",
            "runtime_recovery", "adapter_attestation", "signature",
        }
        missing = required - set(certificate.keys())
        return len(missing) == 0, sorted(missing)

    @staticmethod
    def verify_adapter_attestation(certificate: dict[str, Any]) -> dict[str, Any]:
        """Verify adapter attestation consistency.

        Checks:
        1. All evicted adapters have non-null eviction_method.
        2. Fusion warnings have was_fused = True.
        3. Risk scores are in [0, 1].
        """
        result: dict[str, Any] = {"valid": True, "issues": []}
        att = certificate.get("adapter_attestation", {})

        for adapter in att.get("adapters_evicted", []):
            if not adapter.get("eviction_method"):
                result["valid"] = False
                result["issues"].append(
                    f"evicted adapter {adapter.get('adapter_id')} missing eviction_method"
                )
            risk = adapter.get("risk_score", -1)
            if not (0.0 <= risk <= 1.0):
                result["valid"] = False
                result["issues"].append(
                    f"adapter {adapter.get('adapter_id')} risk_score {risk} out of [0,1]"
                )

        for adapter in att.get("fusion_warnings", []):
            if not adapter.get("was_fused"):
                result["valid"] = False
                result["issues"].append(
                    f"fusion warning for {adapter.get('adapter_id')} but was_fused is False"
                )

        return result

    @staticmethod
    def verify_membership_probe(certificate: dict[str, Any]) -> dict[str, Any]:
        """Verify membership probe results in the certificate.

        Checks:
        1. If probe was run, verdict is one of PASS | FAIL | PARTIAL.
        2. post_recovery_similarity is present and numeric.
        """
        probe = certificate.get("membership_probe")
        if probe is None:
            return {"valid": True, "probe_included": False}

        result: dict[str, Any] = {"valid": True, "probe_included": True, "issues": []}

        verdict = probe.get("probe_verdict")
        if verdict not in ("PASS", "FAIL", "PARTIAL"):
            result["valid"] = False
            result["issues"].append(f"invalid probe verdict: {verdict}")

        post_sim = probe.get("post_recovery_similarity")
        if post_sim is None or not isinstance(post_sim, (int, float)):
            result["valid"] = False
            result["issues"].append("missing or non-numeric post_recovery_similarity")

        return result

    def verify(self, certificate: dict[str, Any]) -> dict[str, Any]:
        """Full V2 certificate verification.

        Returns a verification report dict with:
        - signature_valid: bool
        - fields_complete: bool
        - adapter_attestation_valid: bool
        - membership_probe_valid: bool
        - overall_valid: bool
        - issues: list[str]
        """
        report: dict[str, Any] = {"issues": []}

        # 1. Signature.
        sig_valid = self.verify_signature(certificate)
        report["signature_valid"] = sig_valid
        if not sig_valid:
            report["issues"].append("signature verification failed")

        # 2. Completeness.
        fields_ok, missing = self.verify_completeness(certificate)
        report["fields_complete"] = fields_ok
        if not fields_ok:
            report["issues"].append(f"missing fields: {missing}")

        # 3. Adapter attestation.
        att_result = self.verify_adapter_attestation(certificate)
        report["adapter_attestation_valid"] = att_result["valid"]
        report["issues"].extend(att_result.get("issues", []))

        # 4. Membership probe.
        probe_result = self.verify_membership_probe(certificate)
        report["membership_probe_valid"] = probe_result["valid"]
        report["issues"].extend(probe_result.get("issues", []))

        # Overall.
        report["overall_valid"] = all([
            sig_valid, fields_ok,
            att_result["valid"], probe_result["valid"],
        ])

        return report


def load_certificate(path: str | Path) -> dict[str, Any]:
    """Load a certificate from a JSON file."""
    with open(path, "r") as fh:
        return json.load(fh)
