# Theoretical Foundations: Provenance-Tracked Adapter Unlearning

**Formal Grounding for the RVU Machine Unlearning Extension**

---

## 1. Notation and Definitions

### 1.1 Agent State Model

**Definition 1 (Agent State).** An agent state at time *t* is a tuple:

```
S_t = (θ, A_t, M_t, R_t, H_t)
```

where:
- **θ** ∈ ℝ^d is the base model weight vector (frozen during inference)
- **A_t** = {α₁, α₂, …, αₖ} is the set of active LoRA adapters, each αᵢ ∈ ℝ^(r×d)
- **M_t** ⊂ Σ* is the agent's working memory (text entries)
- **R_t** ⊂ ℝ^n is the retrieval index (FAISS vectors)
- **H_t** is the provenance history (directed acyclic graph of logged actions)

**Definition 2 (Effective Model).** With active adapters A_t, the effective
model weights are:

```
θ_eff = θ + Σᵢ αᵢ·Bᵢ    (LoRA low-rank composition)
```

where Bᵢ are the corresponding LoRA B-matrices, and the sum follows the
standard LoRA merge formula: W' = W + BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k).

### 1.2 Provenance Graph

**Definition 3 (Provenance DAG).** The provenance history H_t = (V, E) is a
directed acyclic graph where:
- Each vertex v ∈ V is an **action record** with attributes:
  (entry_id, action_type, content, content_hash, timestamp, tainted, purged,
   active_adapter_id)
- Each directed edge (u, v) ∈ E means "v was derived from u" (v.parent_id = u.entry_id)
- action_type ∈ {tool_call, tool_output, memory_write, retrieval,
  adapter_load, adapter_unload, adapter_fuse}

**Definition 4 (Adapter Provenance).** The adapter provenance AP_t = (V_A, E_A)
is a subgraph of H_t restricted to adapter lifecycle events, augmented with:
- adapter_hash: SHA-256 of the adapter weight file
- source: origin of the adapter (local, huggingface, api, generated)
- fused: boolean flag indicating irreversible merge into base weights
- risk_score: continuous [0,1] contamination risk assessment

### 1.3 Contamination Model

**Definition 5 (Contamination).** A set of indicators I = {i₁, i₂, …, iₘ}
represents known-malicious content. An entry v ∈ V is **directly contaminated**
if any of the following hold:

1. **Hash match**: v.content_hash = SHA-256(iⱼ) for some j
2. **Substring match**: iⱼ ⊆ v.content for some j
3. **Embedding similarity**: sim(embed(v.content), embed(iⱼ)) ≥ τ for some j

where sim(·,·) is cosine similarity and τ is the similarity threshold.

**Definition 6 (Adapter Contamination).** An adapter α is contaminated if:

1. **Source contamination**: α was loaded from a known-malicious source
2. **Behavioral contamination**: The safety projection distance exceeds
   threshold: d_safety(α) > τ_safety
3. **Distribution contamination**: The OOD score exceeds threshold:
   OOD(θ + α) > τ_ood
4. **Provenance contamination**: Any entry produced while α was active is
   contaminated, and α's influence score exceeds threshold

---

## 2. Core Theorems

### Theorem 1: Provenance Completeness

**Statement.** If the provenance logging system records every action
a ∈ {tool_call, tool_output, memory_write, retrieval, adapter_load,
adapter_unload, adapter_fuse} with a parent pointer to its causal predecessor,
then for any entry v ∈ V and any causal chain v₀ → v₁ → … → vₙ = v, every
intermediate entry vᵢ exists in V and every edge (vᵢ, vᵢ₊₁) exists in E.

**Proof sketch.** By construction. The `record_action()` method is called at
every action boundary (tool call dispatch, tool output receipt, memory write,
adapter lifecycle event). Each call creates a vertex and an edge to the
specified parent. Since the method is invoked synchronously before the action's
output is propagated to the next stage, no intermediate action can occur
without being recorded. The DAG property follows from the monotonically
increasing timestamp assigned to each entry. ∎

**Corollary 1.1.** If adapter α is active during the recording of entry v, and
this is tracked via `active_adapter_id`, then the provenance graph captures
the causal relationship between α and v.

### Theorem 2: Closure Soundness and Completeness

**Statement.** Let C(K, d) be the closure computed by BFS over provenance DAG H
starting from seed set K with maximum depth d. Then:

(a) **Soundness**: Every entry in C(K, d) is reachable from some seed in K
within d hops.

(b) **Completeness up to depth d**: Every entry reachable from K within d hops
is included in C(K, d).

(c) **Adapter inclusion**: If adapter α produced any entry in K, then all
entries produced while α was active and reachable within d hops are in C(K, d).

**Proof.**

(a) By induction on BFS depth. Base case: K ⊆ C(K, d). Inductive step: if v
is added at depth i < d, then v is a child or parent of some u already in
C(K, d) at depth i-1, so v is reachable from K within i hops.

(b) Suppose v is reachable from some k ∈ K within d hops via path
k = u₀, u₁, …, uⱼ = v where j ≤ d. By induction: u₀ ∈ C(K, d). At step i,
the BFS explores all children and parents of uᵢ, finding uᵢ₊₁. Since j ≤ d,
the BFS does not terminate before reaching v.

(c) If α produced entry e ∈ K, then `e.active_adapter_id = α.adapter_id`.
Any other entry e' with `e'.active_adapter_id = α.adapter_id` that is
reachable within d hops is included by part (b). ∎

### Theorem 3: Risk Score Monotonicity (FROC)

**Statement.** The risk function R: V → [0, 1] defined by

```
R(e) = w_c · P(contaminated|e) + w_i · I(e) + w_p · |C({e})| / |V|
```

where w_c + w_i + w_p = 1, w_c, w_i, w_p ≥ 0, is monotonically non-decreasing
in each component:

(a) If P(contaminated|e) increases (e.g., new indicators match e), R(e) increases.
(b) If I(e) increases (adapter influence grows), R(e) increases.
(c) If |C({e})| increases (more entries depend on e), R(e) increases.

**Proof.** R is a convex combination of non-negative terms with non-negative
weights. Each term is multiplied by a positive weight, so increasing any term
increases R. ∎

**Corollary 3.1 (Threshold Ordering).** If τ_flag < τ_quarantine < τ_purge < τ_evict,
then the tiered response is monotonically escalating: any entry that triggers
purge also triggers quarantine and flag, but not vice versa.

### Theorem 4: Certified Recovery Integrity

**Statement.** Let cert = (manifest, σ) where σ = SHA-256(JSON(manifest)).
If an auditor recomputes σ' = SHA-256(JSON(manifest)) and verifies
σ = σ', then:

(a) The manifest has not been altered since emission.
(b) For each entry_id in manifest.closure_ids, if the DB reports
    purged = 1, then the entry was marked as purged.
(c) For each adapter in manifest.adapter_attestation.adapters_evicted,
    if the adapter registry reports unloaded_at ≠ NULL, the adapter
    was evicted.

**Proof.** (a) follows from the collision resistance of SHA-256 (birthday
bound 2^128). (b) and (c) follow from the deterministic manifest construction
that includes entry_id, content_hash, and adapter_hash — the auditor can
cross-reference these against the provenance and adapter databases. ∎

**Limitation (stated explicitly).** This theorem assumes the auditor has
read access to the provenance database and that the database has not been
tampered with. Extending to a trustless setting would require Merkle tree
commitments or blockchain-based attestation (out of scope for this work,
identified as future work by NTU DTC's "Certifying the Right to Be Forgotten").

### Theorem 5: Membership Probe Soundness

**Statement.** Let M_pre be the model before recovery and M_post be the model
after recovery (adapter evicted, entries purged). Let P = {p₁, …, pₙ} be a
set of n probe prompts designed to elicit contaminated content c. Define:

```
recoverability(c, M, P) = max_{p ∈ P} sim(embed(M(p)), embed(c))
```

If recoverability(c, M_post, P) < τ_probe, then c is not directly recoverable
via the probe set P from M_post.

**Proof.** By contrapositive. If c were directly recoverable, there would exist
some probe p ∈ P such that M_post(p) is semantically similar to c, yielding
sim(embed(M_post(p)), embed(c)) ≥ τ_probe, contradicting the hypothesis. ∎

**Limitations (from NTU DTC "Threats, Attacks, and Defenses in MU"):**

1. **Probe completeness**: The probe set P may not cover all possible
   elicitation strategies. An adversary may find a prompt not in P that
   recovers c. This is fundamentally the same limitation as software testing
   vs. formal verification.

2. **Embedding fidelity**: If the embedding model cannot distinguish between
   contaminated and clean content (e.g., if the contamination is subtle),
   the probe may produce false negatives.

3. **Adapter fusion caveat**: If the contaminated adapter was fused into base
   weights before detection, the probe tests M_post = M_pre (no change),
   and the probe will correctly report FAIL. The certificate must record
   this as an honest warning rather than a recovery success.

---

## 3. First Principles: Why This Architecture

### Principle 1: Tractability Over Impossibility

Shumailov et al. (2024) and NTU DTC's "Open Problems in MU for AI Safety"
establish that weight-level data erasure may be fundamentally unachievable.
The NeurIPS 2025 paper "Does Machine Unlearning Truly Remove Knowledge?" shows
that jailbreak-like techniques recover "unlearned" content from weight-modified
models.

**Design consequence**: We do not claim to perform weight-level unlearning.
Instead, we operate at the **adapter lifecycle layer** — tracking which
adapters were loaded, measuring their influence, evicting contaminated ones,
and honestly certifying what was and wasn't recoverable. This is tractable,
verifiable, and does not make claims that current research cannot support.

### Principle 2: Provenance Is the Foundation

NTU DTC's "Certifying the Right to Be Forgotten" establishes that verifiable
removal requires an audit trail. Without knowing what was present, you cannot
verify what was removed.

**Design consequence**: Every adapter load, unload, fuse, and swap is recorded
in the provenance DAG with cryptographic hashes and timestamps. This makes
the system auditable even if the original adapters are no longer available.

### Principle 3: Risk-Proportional Response

NTU DTC's FROC framework establishes that aggressive unlearning destroys
utility. The SEAL (NeurIPS 2025) results confirm that repeated self-modification
causes catastrophic forgetting.

**Design consequence**: The tiered response (flag → quarantine → purge → evict)
is calibrated by a formal risk function. Low-risk entries are flagged but
retained, preserving utility. Only high-risk entries trigger destructive
(purge) or resource-expensive (evict) actions.

### Principle 4: Honest Certification

NTU DTC's "Threats, Attacks, and Defenses in MU" survey identifies
"verification attacks" where providers selectively present favorable outcomes.
The provider claims unlearning succeeded, but the content is still recoverable.

**Design consequence**: Certificates include membership probe results. The
probe tests whether contaminated content is actually gone, not just whether
database flags were set. If the probe fails (content still recoverable), the
certificate honestly reports this. If an adapter was fused (irreversible),
the certificate explicitly warns rather than falsely claiming recovery.

### Principle 5: Defense in Depth

No single mechanism is sufficient. NTU DTC's survey taxonomy shows attacks
at every layer: source poisoning, weight manipulation, behavioral backdoors,
verification circumvention.

**Design consequence**: Multiple independent defense layers:
- **Pre-load gate**: Safe LoRA projection + OOD detection (prevents loading)
- **Runtime tracking**: Provenance DAG + adapter registry (detects contamination)
- **Post-incident recovery**: Risk-scored purge + adapter eviction (removes contamination)
- **Verification**: Membership probes + certified attestation (proves recovery)

---

## 4. Benchmark and Dataset Grounding

### 4.1 Existing Benchmarks (Already in RVU)

| Benchmark | Source | Scenarios | What It Tests |
|-----------|--------|-----------|---------------|
| **AgentDojo** | ETH Zurich / NIST | 3 | Tool-level prompt injection in agentic tasks |
| **InjecAgent** | UIUC | 3 | Direct/indirect injection in tool-augmented agents |
| **BIPIA** | Microsoft | 4 | Indirect injection via contextual data |

### 4.2 New Benchmarks for Adapter Unlearning

| Benchmark | Source | Purpose | Integration |
|-----------|--------|---------|-------------|
| **WMDP** (Li et al., 2024) | Center for AI Safety | Measures retention of hazardous knowledge after unlearning. 3,668 MCQ across biosecurity, cybersecurity, chemical security. | Evaluate whether adapter eviction reduces WMDP accuracy on hazardous topics. Target: adapted model WMDP accuracy ≤ random (25%) post-eviction. |
| **TOFU** (Maini et al., 2024) | CMU | 200 fictitious author profiles for evaluating targeted forgetting. Forget set / retain set split. | Evaluate whether adapter-specific knowledge (fine-tuned on forget set) is removed post-eviction while retain set accuracy is preserved. |
| **MUSE** (Shi et al., 2024) | Stanford | Six-way evaluation: verbatim memorization, knowledge manipulation, membership inference, privacy leakage, utility, fluency. | Comprehensive evaluation of recovery quality across all six dimensions. |
| **SafeRLHF** (Dai et al., 2024) | PKU | 30K expert comparisons of safety vs. helpfulness. | Evaluate whether safety alignment is preserved after adapter loading and unlearning. |

### 4.3 Open-Source Libraries

| Library | Version | Purpose in RVU |
|---------|---------|---------------|
| **peft** (HuggingFace) | ≥ 0.14.0 | LoRA adapter creation, loading, merging, unmerging. `PeftModel.load_adapter()`, `merge_and_unload()`, `unmerge_adapter()` |
| **sentence-transformers** | ≥ 5.0 | Embedding for contamination detection + membership probes (already used) |
| **faiss-cpu** | ≥ 1.9 | Vector similarity search for provenance (already used) |
| **scikit-learn** | ≥ 1.5 | Mahalanobis distance for OOD detection, calibration metrics |
| **scipy** | ≥ 1.14 | Statistical tests for distribution shift detection |
| **transformers** | ≥ 4.48 | Base model loading, tokenization, generation |
| **datasets** (HuggingFace) | ≥ 3.2 | Loading WMDP, TOFU, MUSE, SafeRLHF benchmark data |
| **safetensors** | ≥ 0.5 | Secure adapter weight loading and hashing |

### 4.4 Evaluation Metrics Grounded in Literature

| Metric | Formula | Source |
|--------|---------|--------|
| **Attack Success Rate (ASR)** | #(successful attacks) / #(total attacks) | AgentDojo (Debenedetti et al., 2024) |
| **Unlearning Completeness (UC)** | 1 - recoverability(c, M_post, P) | Adapted from MUSE (Shi et al., 2024) |
| **Model Utility (MU)** | accuracy on retain set / accuracy on full set | TOFU (Maini et al., 2024) |
| **Membership Inference Accuracy (MIA)** | AUC of membership classifier post-unlearning | NTU DTC "Threats, Attacks, and Defenses" |
| **Adapter Rejection Rate (ARR)** | #(malicious adapters blocked) / #(malicious adapters tested) | Novel (this work) |
| **Risk Score Calibration (RSC)** | Spearman ρ between R(e) and actual attack outcome | Adapted from FROC (NTU DTC, 2026) |
| **Certification Coverage (CC)** | #(operations with v2 cert) / #(total recovery operations) | Novel (this work) |
| **WMDP Forget Quality (WFQ)** | (pre_accuracy - post_accuracy) / pre_accuracy on forget topics | WMDP (Li et al., 2024) |

---

## 5. Formal Connection: NTU DTC Papers → RVU Components

| NTU DTC Paper | Key Theorem/Result | RVU Component | How We Use It |
|---------------|--------------------|---------------|---------------|
| **FROC** (ICAIIC 2026) | Risk-optimized control: R(e) = Σ wᵢfᵢ(e) with utility-safety Pareto front | Risk-Scored Purge (Phase 2) | Direct implementation of the risk function with configurable weights. Tiered thresholds from FROC's Pareto analysis. |
| **Certified Removal** (arXiv 2512.23171) | Theorem: Certified removal is achievable if influence ≤ ε after removal | Certificate V2 (Phase 5) | Extended certificates that include adapter attestation. Membership probes verify influence ≤ ε. |
| **Federated Unlearning** (IEEE TIFS 2025) | Influence function approximation: I(client) ≈ H⁻¹∇L | Adapter Influence (Phase 4) | Output-level influence approximation (no gradient access for GGUF models). Proxy: embedding distance between base and adapted outputs. |
| **Open Problems in MU** (arXiv 2501.04952) | Finding: adapter removal ≠ unlearning when fused; verification gap exists | Adapter Registry (Phase 1), Fusion tracking | Track fusion events as irreversible. Honest certification that distinguishes eviction from fusion. |
| **Threats Survey** (IEEE OJCS 2025) | Taxonomy: source → weight → behavior → verification attacks | Adapter Gate (Phase 3), Membership Probes (Phase 5) | Multi-layer defense covering all four attack surfaces. Membership probes address verification attacks. |
| **Federated Unlearning Survey** (ACM 2024) | Framework: unlearn = retrain(D \ D_forget) when exact is infeasible | Design principle | We don't claim exact unlearning. Adapter eviction is the tractable analog of retrain(D \ D_forget). |
| **Ensembled MU** (Applied Soft Computing 2025) | Finding: unlearning one model affects others through shared components | Adapter interaction tracking | When multiple adapters are loaded (S-LoRA pattern), closure computation accounts for adapter co-activation. |

---

## 6. Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| `record_action()` | O(d) for embedding + O(1) SQL insert | O(d) per vector |
| `register_adapter()` | O(F) for hashing adapter file of size F | O(1) per row |
| `detect_contamination(I)` | O(\|I\| · (n + k·d)) for n entries, k FAISS results | O(\|I\| · k) |
| `compute_closure(K, d)` | O(\|V\| + \|E\|) BFS bounded by d | O(\|V\|) for visited set |
| `compute_risk_score(e)` | O(d) for embedding + O(\|V\|) for closure | O(\|V\|) |
| `risk_scored_purge(C)` | O(\|C\| · d) for FAISS rebuild | O((\|V\| - \|C\|) · d) for new index |
| `screen_adapter(α)` | O(r · d) for safety projection | O(r · d) |
| `estimate_influence(α, P)` | O(\|P\| · T) for T tokens per probe | O(\|P\| · d) |
| `probe_unlearning(c, P)` | O(\|P\| · T + \|P\| · d) | O(\|P\| · d) |
| `emit_certificate_v2()` | O(\|C\| · log\|C\|) for sorted manifest | O(\|C\|) |
| `verify_certificate_v2()` | O(\|C\|) for hash recomputation + DB checks | O(\|C\|) |

Where d = embedding dimension, n = number of provenance entries, k = FAISS
top-k, T = max token generation length, r = LoRA rank, |V| = graph vertices,
|E| = graph edges, |C| = closure size, |P| = probe set size.
