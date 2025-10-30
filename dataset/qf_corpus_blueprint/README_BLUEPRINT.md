# Québec French Corpus Blueprint

This blueprint provides a ready-to-use scaffold for building a structured
Québec French (QF) corpus that is compatible with the
`pi-llm-optimizer` distillation pipeline. It includes:

- JSON Schemas describing both the record format and split-level dataset cards.
- Controlled vocabularies for dialect tags, register labels, and
diachronic time periods.
- Python utilities for validation, normalization/enrichment, phonetic
  projection, distractor generation, and balance reporting.
- Example data to get started quickly.
- Compliance-oriented documentation for privacy impact assessment (PIA)
and licensing.

## Repository layout

```text
schema/               # JSON Schemas for records and dataset cards
vocab/                # Controlled vocabularies referenced by the schema
scripts/              # Validation, normalization, and analytics tools
examples/             # Sample JSONL dataset compatible with the schema
docs/                 # Compliance & licensing guidance
```

## Quick start

**Note:** Run these commands from the repository root so the relative paths resolve correctly.

1. **Validate a dataset** against the schema:

   ```bash
   pip install jsonschema
   python dataset/qf_corpus_blueprint/scripts/validate_jsonl.py \
     --schema dataset/qf_corpus_blueprint/schema/qf_record.schema.json \
     --data dataset/qf_corpus_blueprint/examples/sample.jsonl
   ```

2. **Normalize & enrich** Québec French data to generate Montreal French
   (MF) equivalents, lemmas, and sociolinguistic defaults:

   ```bash
   python dataset/qf_corpus_blueprint/scripts/ingest_normalize.py \
     --in-file dataset/qf_corpus_blueprint/examples/sample.jsonl \
     --out-file dataset/qf_corpus_blueprint/examples/enriched.jsonl \
     --default-register Casual \
     --default-region Montréal \
     --time-period 2001-2050
   ```

3. **Generate phonetic projections** suitable for MFA/Prosodylab pipelines:

   ```bash
   python - <<'PY'
   from dataset.qf_corpus_blueprint.scripts.g2p_pipeline import ProsodylabG2PPipeline

   pipeline = ProsodylabG2PPipeline()
   print(pipeline.process_text("Chu ben content d'te voir."))
   PY
   ```

4. **Report balance & coverage** across register, time period, and region
   axes:

   ```bash
   python dataset/qf_corpus_blueprint/scripts/balance_report.py \
     --data dataset/qf_corpus_blueprint/examples/enriched.jsonl
   ```

## Extending the blueprint

- Add new dialect tags by updating `vocab/dialect_tags.json` and adjusting
  the pattern-to-tag mappings in `vocab/dialect_rules.json` (no code changes
  required; the normalizer loads the JSON rules at runtime).
- Customize orthographic cleanup or MF normalization by editing
  `rules/normalization.yaml`. Rules are grouped by linguistic category,
  and each rule can add dialect tags for downstream analysis.
- Expand the JSON Schema with additional sociolinguistic attributes as
  your research requires.
- Integrate the generated dataset into the distillation workflow by
  pointing the existing labelling and training scripts to the enriched
  JSONL output. Use `schema/dataset.card.schema.json` to validate split
  metadata before publication.

## Compliance considerations

Refer to `docs/PIA_CHECKLIST.md` and `docs/LICENSE_SUGGESTION.md` for
privacy and licensing guardrails that align with open, research-driven
corpora.
