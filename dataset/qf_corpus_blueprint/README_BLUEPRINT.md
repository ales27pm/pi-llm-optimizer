# Québec French Corpus Blueprint

This blueprint provides a ready-to-use scaffold for building a structured
Québec French (QF) corpus that is compatible with the
`pi-llm-optimizer` distillation pipeline. It includes:

- A JSON Schema describing the required record format.
- Controlled vocabularies for dialect tags, register labels, and
diachronic time periods.
- Python utilities for validation, normalization/enrichment, and balance
reporting.
- Example data to get started quickly.
- Compliance-oriented documentation for privacy impact assessment (PIA)
and licensing.

## Repository layout

```
schema/               # JSON Schema for individual QF records
vocab/                # Controlled vocabularies referenced by the schema
scripts/              # Validation, normalization, and analytics tools
examples/             # Sample JSONL dataset compatible with the schema
docs/                 # Compliance & licensing guidance
```

## Quick start

1. **Validate a dataset** against the schema:

   ```bash
   pip install jsonschema
   python scripts/validate_jsonl.py \
     --schema schema/qf_record.schema.json \
     --data examples/sample.jsonl
   ```

2. **Normalize & enrich** Québec French data to generate Montreal French
   (MF) equivalents and sociolinguistic defaults:

   ```bash
   python scripts/ingest_normalize.py \
     --in-file examples/sample.jsonl \
     --out-file examples/enriched.jsonl \
     --default-register Casual \
     --default-region Montréal \
     --time-period 2001-2050
   ```

3. **Report balance & coverage** across register, time period, and region
   axes:

   ```bash
   python scripts/balance_report.py --data examples/enriched.jsonl
   ```

## Extending the blueprint

- Add new dialect tags by updating `vocab/dialect_tags.json` and adjusting
  the pattern-to-tag mappings in `vocab/dialect_rules.json` (no code changes
  required; the normalizer loads the JSON rules at runtime).
- Customize orthographic cleanup or MF normalization by editing
  `vocab/normalization_rules.json` to add or refine regex substitutions.
- Expand the JSON Schema with additional sociolinguistic attributes as
  your research requires.
- Integrate the generated dataset into the distillation workflow by
  pointing the existing labelling and training scripts to the enriched
  JSONL output.

## Compliance considerations

Refer to `docs/PIA_CHECKLIST.md` and `docs/LICENSE_SUGGESTION.md` for
privacy and licensing guardrails that align with open, research-driven
corpora.
