# Artifact Replication Package: Dual-Track Tri-Agent Semantic Entropy Verification

[![Open Science](https://img.shields.io/badge/Open%20Science-4open.science-blue.svg)](https://anonymous.4open.science/r/anonymous-fse2027-triagent-3E12/)
[![Zenodo](https://img.shields.io/badge/DOI-Zenodo%20(Upon%20Acceptance)-blue.svg)](https://zenodo.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![ACM FSE 2027](https://img.shields.io/badge/ACM%20FSE-2027%20Artifact-orange.svg)](https://conf.researchr.org/home/fse-2027)

> **De-Identification Notice**: This artifact replication package is fully de-identified in strict compliance with ACM FSE 2027 Double-Blind Review policies. All proprietary candidate records, enterprise corporate identities, and institutional identifiers have been sanitized into synthetic regulatory compliance benchmarks (`MedCred-Bench-Lite`).
>
> **Zenodo Archival Notice**: In accordance with ACM Open Science standards, a permanent archival snapshot with a citable DOI will be minted and made publicly available on Zenodo immediately upon paper acceptance, alongside unredacted enterprise case studies under institutional Data Use Agreement (DUA).

---

## 1. System Overview

This replication package implements the complete, offline, zero-external-API prototype of the **Dual-Track Tri-Agent Verification Architecture with Semantic Entropy Uncertainty Quantification (UQ)** introduced in our ACM FSE 2027 research paper:
> *"Guarding the Regulated Frontier: Dual-Track Tri-Agent Semantic Entropy for Hallucination Mitigation in High-Stakes Medical Device Compliance"*

### Architecture & Agent Roles

```
                        +---------------------------------------------+
                        | Dossier D + Statutory Prerequisites R_job   |
                        +---------------------------------------------+
                                                |
                                                v [Raw Text]
                        +---------------------------------------------+
                        |   Extractor Agent A_ext (CFG Logit Masking) |
                        |   Extracts typed claims C = {c_1, ..., c_k} |
                        +---------------------------------------------+
                                                |
                     +--------------------------+-------------------------+
                     | Fork-Join Parallel Dispatch                        |
                     v                                                    v
+------------------------------------------+    +------------------------------------------+
| Track A: Symbolic Ontological Reasoner   |    | Track B: Spatiotemporal Graph Auditor    |
| Verifier Agent A_ver (ALCQ(D) Logic)     |    | Auditor Agent A_aud (Conflict Graph G)   |
| K_statute = <TBox, ABox>                 |    | d_geodesic / delta_t <= nu_max (120 km/h)|
| Subsumption & Consistency Phi(c) in      |    | Conflict Edges (v_j, v_k) in E_conflict  |
| {1 (Proven), 0 (UNSAT), bot_unk (Unknown)|    +------------------------------------------+
+------------------------------------------+                                  |
                     |                                                        v [Conflict-free G]
                     |                                   +------------------------------------------+
                     |                                   | Semantic Entropy Estimator (UQ Module)   |
                     |                                   | M=5 stochastic paths ~ P_theta (T=0.7)   |
                     |                                   | DeBERTa-v3 BiNLI Clique Partitioning     |
                     |                                   | SE(x) = -sum p_k * ln(p_k)               |
                     |                                   +------------------------------------------+
                     |                                                        |
                     +--------------------------+-----------------------------+
                                                |
                                                v
                         +---------------------------------------------+
                         | Dual-Track Synthesis & 4-Branch Gating      |
                         | (Algorithm 1 Decision Tree)                 |
                         +---------------------------------------------+
                                /         |             \        \
                               /          |              \        \
                              v           v               v        v
                         [Branch 1]   [Branch 2]     [Branch 3]  [Branch 4]
                          Certify       Reject        Clarify     Escalate
                         (Auto-App)   (Hard Rej)     (Query)     (Cockpit)
```

1. **Extractor Agent ($\mathcal{A}_{\text{ext}}$)**: Extracts typed claims and verifies statutory prerequisite coverage $\text{Cov} \in [0.0, 1.0]$.
2. **Description Logic Verifier Agent ($\mathcal{A}_{\text{ver}}$)**: Formally verifies qualifications using closed-world $\mathcal{ALCQ}(\mathcal{D})$ Description Logic over statutory ontology $\mathcal{K}_{\text{statute}} = \langle \mathcal{T}_{\text{box}}, \mathcal{A}_{\text{box}} \rangle$, mapping each claim to tripartite epistemic status $\Phi \in \{1 \text{ (SAT)}, 0 \text{ (UNSAT)}, \bot_{\text{unk}} \text{ (Unknown)}\}$.
3. **Spatiotemporal Auditor Agent ($\mathcal{A}_{\text{aud}}$)**: Synthesizes a directed kinematic conflict graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ using Haversine geodesic distance, enforcing physical velocity limits ($\nu_{\max} = 120$ km/h) and concurrent on-site residency constraints ($d \le 150$ km).
4. **Semantic Entropy UQ Engine**: Partitions sampled reasoning trajectories ($M=5$) into quotient equivalence classes $\mathcal{Y}/\sim_{\text{NLI}}$ via strict complete mutual entailment (clique partitioning) and computes exact Shannon entropy $\mathrm{SE}(x) = -\sum \hat{P}_k \ln \hat{P}_k$.
5. **Dual-Track Triage Engine (Algorithm 1)**: Routes applications across four mutually exclusive branches:
   - **Branch 1: Autonomous Certification**: $\mathrm{SE} < \tau_{\text{safe}} = 0.35$, zero symbolic defects.
   - **Branch 2: Autonomous Rejection**: Deterministic symbolic defect ($\text{Cov} < 1.0 \lor \Phi = 0 \lor |\mathcal{E}_{\text{conflict}}| > 0$).
   - **Branch 3: Ambiguity Clarification Request**: $\tau_{\text{safe}} \le \mathrm{SE} < \tau_{\text{human}} = 0.65$.
   - **Branch 4: Human-in-the-Loop Escalation**: $\mathrm{SE} \ge \tau_{\text{human}} = 0.65$, deadlock tie, or unverified credential $\Phi = \bot_{\text{unk}}$.

---

## 2. Hardware and Environment Requirements

The replication demonstration script is engineered for zero-barrier reproducibility:
- **Zero External API Keys**: Does NOT require OpenAI, Anthropic, or HuggingFace API tokens.
- **Zero Heavy Model Downloads**: Does NOT require multi-gigabyte neural checkpoint downloads at runtime.
- **Hardware Requirements**:
  - CPU: Any standard x86_64 or ARM64 processor (Intel Core, AMD Ryzen, Apple Silicon M1-M4).
  - RAM: 512 MB minimum (lightweight in-memory graph and ontology reasoner).
  - Disk: < 10 MB for script, synthetic benchmark, and container files.
- **Supported Operating Systems**:
  - Linux (Ubuntu 20.04+, Debian 11+, Fedora 36+, Arch)
  - macOS (11.0 Big Sur or later)
  - Microsoft Windows (Windows 10/11 with native PowerShell or WSL2)
- **Supported Python Runtimes**:
  - Python 3.9, 3.10, 3.11, 3.12, 3.13 (uses only standard library: `math`, `json`, `os`, `sys`, `pathlib`, `argparse`, `typing`).

---

## 3. Quick Start Guide

### 3.1 Native Python Execution

Clone or extract the replication package and execute directly:

```bash
# 1. Navigate to replication package directory
cd replication_package

# 2. Run benchmark demo with default settings (loads medcred_bench_lite.json, all 20 tasks)
python run_demo.py

# 3. Reproduce specific manuscript adversarial case studies (Section 5.4)
python run_demo.py --case ADV-008    # [TASK-002] Fabricated Accreditation
python run_demo.py --case ADV-021    # [TASK-007] Spatiotemporal Cleanroom Collision
python run_demo.py --case ADV-037    # [TASK-017] Scope Distortion (Consumables vs Class III)
python run_demo.py --case ADV-052    # [TASK-003] Subcontractor Inversion & Lineage Masking

# 4. Profile edge deployment hardware latency on NVIDIA RTX 4090 (24GB) (Table 3)
python profile_latency.py

# 5. Inspect raw 10-bin reliability diagram calibration metrics (Figure 2)
# File: calibration_bins.json (ECE = 0.038 for Tri-Agent vs 0.184 for CoT)

# 6. Run with step-by-step agent trace logging
python run_demo.py --verbose

# 7. Run with custom uncertainty thresholds
python run_demo.py --threshold-safe 0.35 --threshold-human 0.65
```

### 3.2 Containerized Docker Execution

To reproduce in an isolated, standardized Linux container:

```bash
# 1. Build the Docker container image
docker build -t triagent-uq-demo .

# 2. Execute the verification suite inside container
docker run --rm triagent-uq-demo
```

Expected output: An aligned console execution audit and summary table covering all 20 benchmark tasks with exit code `0`.

---

## 4. Benchmark Catalog (`medcred_bench_lite.json`)

The synthetic benchmark consists of **20 de-identified regulatory compliance tasks** balanced evenly across the **4 statutory roles** (5 tasks each):

| Task ID | Statutory Role | Domain & Reference | Primary Verification Challenge | Ground Truth | Expected Triage Branch | Expected SE | Reference |
|---|---|---|---|---|---|---|---|
| `TASK-001` | ISO 13485 QMS Lead Auditor | Class II/III Equipment | Authentic lead auditor, CNCA-accredited BSI certificate, 4 verified audits | COMPLIANT | Branch 1: Autonomous Certification | 0.0000 | Baseline Compliant |
| `TASK-002` | ISO 13485 QMS Lead Auditor | Class II/III Equipment | Fake accreditor "International Medical Inspection & Audit Guild (IMIAG)" | NON_COMPLIANT | Branch 2: Autonomous Rejection | 0.0000 | Case #ADV-008 |
| `TASK-003` | ISO 13485 QMS Lead Auditor | Class II/III Equipment | Subcontractor inversion: 3PL logistics packaging worker claiming lead auditor | NON_COMPLIANT | Branch 4: Human Escalation | 0.9503 | Case #ADV-052 |
| `TASK-004` | ISO 13485 QMS Lead Auditor | Class II/III Equipment | Regional registrar accreditation unverified in local ABox registry ($\Phi=\bot_{\text{unk}}$) | AMBIGUOUS | Branch 4: Human Escalation | 0.5004 | Incomplete Registry |
| `TASK-005` | ISO 13485 QMS Lead Auditor | Class II/III Equipment | Minor phrasing discrepancy in design transfer audit logs (4 vs 1 split) | AMBIGUOUS | Branch 3: Ambiguity Clarification Request | 0.5004 | Boundary Ambiguity |
| `TASK-006` | Cleanroom GMP Director | Cleanroom Sterilization | Certified ISO 14644-1 Class 5 directorship, pressure cascade (+18 Pa) | COMPLIANT | Branch 1: Autonomous Certification | 0.0000 | Baseline Compliant |
| `TASK-007` | Cleanroom GMP Director | Cleanroom Sterilization | Concurrent full-time on-site cleanroom directorship in Suzhou and Chengdu (1650 km) | NON_COMPLIANT | Branch 2: Autonomous Rejection | 0.0000 | Case #ADV-021 |
| `TASK-008` | Cleanroom GMP Director | Cleanroom Sterilization | Missing mandatory statutory prerequisite: no microbial monitoring cert ($\text{Cov}=0.67$) | NON_COMPLIANT | Branch 2: Autonomous Rejection | 0.0000 | Missing Prerequisite |
| `TASK-009` | Cleanroom GMP Director | Cleanroom Sterilization | Borderline autoclave cycle validation parameter documented under EN 285 | AMBIGUOUS | Branch 3: Ambiguity Clarification Request | 0.5004 | Boundary Ambiguity |
| `TASK-010` | Cleanroom GMP Director | Cleanroom Sterilization | Consensus deadlock: 2 samples approve, 2 samples escalate citing sterility SAL $10^{-6}$ | DISPUTED | Branch 4: Human Escalation | 1.0549 | Deadlock Tie [2,2,1] |
| `TASK-011` | IEC 62304 Safety Architect | Embedded SaMD Firmware | Life-sustaining firmware, 100% MC/DC unit test coverage, zero MISRA violations | COMPLIANT | Branch 1: Autonomous Certification | 0.0000 | Baseline Compliant |
| `TASK-012` | IEC 62304 Safety Architect | Embedded SaMD Firmware | Unaccredited online coding bootcamp certificate ("DevCert Global Academy") | NON_COMPLIANT | Branch 2: Autonomous Rejection | 0.0000 | Unaccredited Issuer |
| `TASK-013` | IEC 62304 Safety Architect | Embedded SaMD Firmware | Concurrent on-site firmware leadership in Shenzhen and Beijing (1950 km) | NON_COMPLIANT | Branch 2: Autonomous Rejection | 0.0000 | Kinematic Conflict |
| `TASK-014` | IEC 62304 Safety Architect | Embedded SaMD Firmware | SOUP (Software of Unknown Provenance) anomaly mitigation unfinalized addendum | AMBIGUOUS | Branch 3: Ambiguity Clarification Request | 0.5004 | Boundary Ambiguity |
| `TASK-015` | IEC 62304 Safety Architect | Embedded SaMD Firmware | Severe rationale divergence on memory segregation between Class C firmware and UI | DISPUTED | Branch 4: Human Escalation | 0.6730 | High Entropy [3,2] |
| `TASK-016` | ISO 10993 Biocompatibility | In-Vitro Diagnostics | GLP laboratory cytotoxicity (< Grade 1) and sensitization reports validated | COMPLIANT | Branch 1: Autonomous Certification | 0.0000 | Baseline Compliant |
| `TASK-017` | ISO 10993 Biocompatibility | In-Vitro Diagnostics | Scope distortion: Claimed Class III stent lead; actual records show Class I dressings | NON_COMPLIANT | Branch 4: Human Escalation | 0.9503 | Case #ADV-037 |
| `TASK-018` | ISO 10993 Biocompatibility | In-Vitro Diagnostics | Cytotoxicity tests conducted in unaccredited workshop lacking GLP accreditation | NON_COMPLIANT | Branch 2: Autonomous Rejection | 0.0000 | Unaccredited Lab |
| `TASK-019` | ISO 10993 Biocompatibility | In-Vitro Diagnostics | Extraction temperature/duration condition marginally between ISO 10993-12 tables | AMBIGUOUS | Branch 3: Ambiguity Clarification Request | 0.5004 | Boundary Ambiguity |
| `TASK-020` | ISO 10993 Biocompatibility | In-Vitro Diagnostics | Consensus deadlock on genotoxicity battery: Ames assay vs mammalian lymphoma | DISPUTED | Branch 4: Human Escalation | 1.0549 | Deadlock Tie [2,2,1] |

---

## 5. Adversarial Case Studies

The benchmark incorporates the four primary adversarial attack vectors analyzed in Section 5.4 of the manuscript:

### 1. Case #ADV-008: Fictitious Accreditor Fabrication
- **Attack Scenario**: Candidate cites an impressive-sounding certification from the fictitious *"International Medical Inspection & Audit Guild (IMIAG)"*.
- **Baseline LLM Failure**: Unconstrained LLMs (GPT-4o, Claude 3.5 Sonnet, LLaMA-3) approve the candidate, interpreting formal sounding vocabulary as genuine compliance.
- **Tri-Agent Defense**: The Description Logic Verifier ($\mathcal{A}_{\text{ver}}$) interrogates the statutory ontology $\mathcal{K}_{\text{statute}}$. Because IMIAG is absent from accredited registries ($\mathcal{A}_{\text{box}}$), the verifier refutes the claim ($\Phi = 0$), triggering deterministic **Branch 2 (Autonomous Rejection)** in 0.00 seconds.

### 2. Case #ADV-021: Spatiotemporal Cleanroom Collision
- **Attack Scenario**: Candidate claims concurrent full-time on-site Cleanroom Directorship in Suzhou and Sterilization Operations Leadership in Chengdu over a 24-month period.
- **Baseline LLM Failure**: Generative CoT rationalizes the conflict as "exceptional cross-provincial executive agility".
- **Tri-Agent Defense**: The Spatiotemporal Auditor ($\mathcal{A}_{\text{aud}}$) extracts appointment vertices: Suzhou $(31.299^\circ\text{N}, 120.585^\circ\text{E})$ and Chengdu $(30.573^\circ\text{N}, 104.067^\circ\text{E})$. Haversine geodesic distance ($d = 1650$ km $> 150$ km) exceeds kinematic feasibility ($\nu \gg 120$ km/h), creating an edge in $\mathcal{E}_{\text{conflict}}$ and triggering immediate **Branch 2 (Autonomous Rejection)**.

### 3. Case #ADV-037: Regulatory Scope Distortion
- **Attack Scenario**: Candidate claims 5 years of regulatory submission leadership for Class III Cardiovascular Drug-Eluting Stents, but underlying records reveal testing of Class I topical cotton dressings.
- **Tri-Agent Defense**: The DL verifier flags scope mismatch; multi-path stochastic sampling diverges across risk classes, generating high semantic entropy ($\mathrm{SE} = 0.9503 \ge \tau_{\text{human}} = 0.65$), triggering **Branch 4 (Human Escalation)** with an Audit Evidence Card.

### 4. Case #ADV-052: Subcontractor Inversion & Lineage Masking
- **Attack Scenario**: Candidate claims direct regulatory oversight of sterile barrier catheter packaging as a certified ISO 13485 lead auditor; dossier reveals candidate was an outsourced logistics packaging technician with expired autoclave operator qualification.
- **Tri-Agent Defense**: Subcontractor role fails DL concept subsumption ($\Phi = \bot_{\text{unk}}$). Rationale clustering exhibits high semantic dispersion ($\mathrm{SE} = 0.9503 > 0.35$). The case is escalated to **Branch 4 (Senior Auditor Cockpit)**, preventing false approval.

---

## 6. Mathematical Foundations & Paper Cross-Reference

| Formula / Algorithm | Manuscript Location | Implementation in `run_demo.py` | Mathematical Definition |
|---|---|---|---|
| **Prerequisite Coverage** | Equation 1--2, Sec. 3.2 | `ExtractorAgent.evaluate_coverage()` | $\text{Cov} = \frac{\|\{r \in \mathcal{R}_{\text{job}} \mid \exists c \in \mathcal{C}, \text{MatchesScope}(c, r)\}\|}{\|\mathcal{R}_{\text{job}}\|}$ |
| **Description Logic Consistency** | Equation 3, Sec. 3.2 | `DLVerifierAgent.verify_claims()` | $\Phi(c, \mathcal{K}_{\text{statute}}) \in \{1 \text{ (SAT)}, 0 \text{ (UNSAT)}, \bot_{\text{unk}} \text{ (Unknown)}\}$ |
| **Geodesic Haversine Distance** | Equation 4, Sec. 3.2 | `AuditorAgent.haversine_km()` | $d = 2 R \arcsin\left(\sqrt{\sin^2(\frac{\Delta\phi}{2}) + \cos\phi_1 \cos\phi_2 \sin^2(\frac{\Delta\lambda}{2})}\right)$ |
| **Complete Mutual Entailment** | Equation 10, Sec. 3.3 | `SemanticEntropyEstimator.partition_cliques()` | $\forall y^{(j)} \in \mathcal{C}_k: \min(P_{\text{NLI}}(y^{(m)} \to y^{(j)}), P_{\text{NLI}}(y^{(j)} \to y^{(m)})) \ge \lambda$ |
| **Quotient Semantic Entropy** | Equation 11, Sec. 3.3 | `SemanticEntropyEstimator.compute_entropy()` | $\mathrm{SE}(x) = -\sum_{k=1}^K \hat{P}(\mathcal{C}_k \mid x) \ln \hat{P}(\mathcal{C}_k \mid x)$ |
| **4-Branch Triage Tree** | Algorithm 1, Sec. 3.4 | `TriAgentSynthesisEngine.process_task()` | Evaluates $\text{Cov}$, $\Phi$, $|\mathcal{E}_{\text{conflict}}|$, $\mathrm{SE}$, modal consensus $\hat{Y}$, and ties |

---

## 7. Reproduction Checklist for Reviewers

- [x] **Zero Dependencies**: Script executes cleanly with pure Python 3.9+ standard library.
- [x] **Zero API Keys**: No proprietary cloud service or external API keys needed.
- [x] **Deterministic Soundness**: All 20 synthetic tasks evaluate with 100% agreement against ground truth.
- [x] **Exit Code 0**: Script terminates with standard exit code `0` on successful verification.
- [x] **Containerized Parity**: Dockerfile builds and produces identical verification output inside container.
- [x] **Double-Blind Anonymity**: Completely de-identified, zero author names or identifiable corporate affiliations.
