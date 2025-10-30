from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.qf_corpus_blueprint.scripts.anonymize import (  # noqa: E402
    audit_privacy_levels,
    enforce_k_anonymity,
    ensure_l_diversity,
)
from dataset.qf_corpus_blueprint.scripts.distractor_generator import (  # noqa: E402
    generate_distractors,
)
from dataset.qf_corpus_blueprint.scripts.g2p_pipeline import ProsodylabG2PPipeline  # noqa: E402
from dataset.qf_corpus_blueprint.scripts.normalizer import (  # noqa: E402
    NeuralLemmatizer,
    NormalizationEngine,
    normalize_text_qf,
)


def test_yaml_normalization_pipeline_rules():
    engine = NormalizationEngine.from_yaml()
    lemmatizer = NeuralLemmatizer("none")
    payload = normalize_text_qf("Chu ben content d'te voir.", engine=engine, lemmatizer=lemmatizer)
    assert payload["output"] == "je suis bien content de te voir."
    assert "Contraction_QF" in payload["rule_tags"]
    assert payload["lemmatizer"]["lemmas"] == []


def test_g2p_pipeline_fallback():
    pipeline = ProsodylabG2PPipeline()
    result = pipeline.process_text("Chu ben content d'te voir.")
    assert result.phone_set == "Prosodylab_QF_v2"
    assert result.phonetic
    assert 0.0 < result.alignment_confidence <= 1.0
    assert result.segments


def test_distractor_generation_without_model():
    payload = generate_distractors(
        "retomber en amour",
        "tomber amoureux de nouveau",
        candidate_count=3,
        max_similarity=0.6,
    )
    assert payload["expression"] == "retomber en amour"
    assert len(payload["distractors"]) == 2
    for entry in payload["distractors"]:
        assert 0 <= entry["similarity_score"] <= 0.6


def test_anonymization_pipeline_enforces_k_and_l():
    records = [
        {
            "id": idx,
            "sociolinguistic_parameters": {
                "region_qc": "Gaspésie",
                "age_group": "18_25",
                "gender": "F",
            },
        }
        for idx in range(3)
    ] + [
        {
            "id": 100 + idx,
            "sociolinguistic_parameters": {
                "region_qc": "Montréal",
                "age_group": "26_40",
                "gender": gender,
            },
        }
        for idx, gender in enumerate(["F", "M", "NB", "F", "M"])
    ]

    k_safe = enforce_k_anonymity(records, k=5)
    assert all(
        entry["sociolinguistic_parameters"]["region_qc"] == "unspecified"
        for entry in k_safe
        if entry["id"] < 3
    )

    l_safe = ensure_l_diversity(k_safe, l_threshold=2)
    audit = audit_privacy_levels(l_safe)
    assert audit["min_bucket_size"] >= 2
    assert any("unspecified" in bucket for bucket in audit["buckets"].values())
