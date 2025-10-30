# Privacy Impact Assessment — Québécois French Corpus Blueprint

> **Version:** 1.0.0  
> **Maintainer:** Data Governance Office

## 1. Purpose & Scope
- **Project summary:**
- **Datasets involved:**
- **Systems consuming the corpus:**
- **Processing activities:**

## 2. Personal Data Mapping
Describe all direct and indirect identifiers present in the corpus.

| Field | Description | Identifier Type | Retention | Notes |
| ----- | ----------- | --------------- | --------- | ----- |
| `region_qc` | Region code within Québec | Quasi-Identifier | 5 years | Apply coarsening for sparse regions |
| `age_group` | Demographic bucket | Quasi-Identifier | 5 years | Merge rare groups into broader bins |
| `gender` | Self-declared gender | Sensitive Attribute | 5 years | Mask if l-diversity < 2 |
| `source_corpus_id` | Provenance identifier | Operational | 5 years | Required for audit trail |

## 3. Risk Matrix
Evaluate the probability and impact of re-identification or misuse.

| Scenario | Probability | Impact | Mitigation |
| -------- | ----------- | ------ | ---------- |
| Linkage attack using regional data | Medium | High | Enforce k-anonymity ≥ 5, aggregate rural regions |
| Temporal inference via `creation_date` | Low | Medium | Truncate timestamps to month granularity |
| Exposure of dialect-specific slurs | Medium | Medium | Apply linguistic redaction policy |

## 4. Anonymization Protocol
1. Execute `enforce_k_anonymity` with `k ≥ 5` on (`region_qc`, `age_group`).
2. Apply `ensure_l_diversity` with `l ≥ 2` for the `gender` field.
3. Review outputs using `audit_privacy_levels` and archive reports.
4. Store de-identified corpus separately from raw data with access logging.

## 5. Residual Risk & Approval
- **Residual risk summary:**
- **Mitigation effectiveness:**
- **Legal review (Law 25 compliance):**
- **Security review:**
- **Approving authority:**
- **Date of approval:**

---

### Appendix — Change Log

| Date | Author | Description |
| ---- | ------ | ----------- |
| 2025-01-10 | Privacy Office | Initial template |
