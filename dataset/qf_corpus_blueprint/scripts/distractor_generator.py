"""Dialect-aware distractor generator for QFrCoRE-style benchmarks."""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

try:  # pragma: no cover - optional heavy dependency
    from transformers import pipeline
except Exception:  # pragma: no cover - fallback path
    pipeline = None

LOGGER = logging.getLogger("qf.distractor")

DEFAULT_NEGATIVE_POOL = [
    "changer de travail sans raison",
    "partir en vacances sur un coup de tête",
    "réparer une voiture ancienne",
    "apprendre une nouvelle langue",
    "tomber malade soudainement",
    "gagner à la loterie locale",
]


@dataclass
class DistractorCandidate:
    text: str
    bleu: float
    rouge: float
    bert_score: float

    @property
    def weighted_similarity(self) -> float:
        return 0.2 * self.bleu + 0.3 * self.rouge + 0.5 * self.bert_score


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().replace("'", " ").split() if token]


def _bleu(reference: str, candidate: str) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not cand_tokens:
        return 0.0
    overlap = sum(1 for token in cand_tokens if token in ref_tokens)
    precision = overlap / len(cand_tokens)
    brevity_penalty = min(1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1)))
    return round(precision * brevity_penalty, 3)


def _rouge_l(reference: str, candidate: str) -> float:
    ref_tokens = _tokenize(reference)
    cand_tokens = _tokenize(candidate)
    if not ref_tokens or not cand_tokens:
        return 0.0
    dp = [[0] * (len(cand_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(cand_tokens) + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    recall = lcs / len(ref_tokens)
    precision = lcs / len(cand_tokens)
    if recall + precision == 0:
        return 0.0
    return round((2 * recall * precision) / (recall + precision), 3)


def _bert_score(reference: str, candidate: str) -> float:
    try:  # pragma: no cover - optional dependency path
        from bert_score import score
    except Exception:
        ref_tokens = set(_tokenize(reference))
        cand_tokens = set(_tokenize(candidate))
        if not cand_tokens:
            return 0.0
        return round(len(ref_tokens & cand_tokens) / len(cand_tokens), 3)
    precision, _, _ = score([candidate], [reference], lang="fr")  # type: ignore[arg-type]
    return round(float(precision[0]), 3)


def _generate_with_model(prompt: str, *, model: Optional[str] = None, num_candidates: int = 6) -> List[str]:
    if pipeline is None or model is None:
        return []
    try:
        generator = pipeline(
            "text-generation",
            model=model,
            max_new_tokens=48,
            do_sample=True,
            top_p=0.95,
        )
        outputs = generator(prompt, num_return_sequences=num_candidates)
    except Exception as exc:  # pragma: no cover - optional dependency path
        LOGGER.warning("Transformer generation failed, falling back to heuristics: %s", exc)
        return []
    results: List[str] = []
    for item in outputs:
        text = item.get("generated_text", "")
        text = text.replace(prompt, "").strip()
        if text:
            results.append(text.split("\n")[0].strip())
    return results


def _heuristic_candidates(expression: str, *, count: int) -> List[str]:
    random.seed(expression)
    pool = list(DEFAULT_NEGATIVE_POOL)
    random.shuffle(pool)
    return pool[:count]


def generate_distractors(
    expression: str,
    definition_correct: str,
    *,
    model: Optional[str] = None,
    candidate_count: int = 6,
    max_similarity: float = 0.45,
) -> Dict[str, object]:
    prompt = (
        f"Expression québécoise: {expression}\n"
        "Fournis une définition plausible mais incorrecte en une courte phrase."
    )
    candidates = _generate_with_model(prompt, model=model, num_candidates=candidate_count)
    if len(candidates) < 2:
        candidates.extend(_heuristic_candidates(expression, count=candidate_count))
    scored: List[DistractorCandidate] = []
    for text in candidates:
        bleu = _bleu(definition_correct, text)
        rouge = _rouge_l(definition_correct, text)
        bert = _bert_score(definition_correct, text)
        scored.append(DistractorCandidate(text=text, bleu=bleu, rouge=rouge, bert_score=bert))
    filtered = [cand for cand in scored if cand.weighted_similarity <= max_similarity]
    if not filtered:
        filtered = sorted(scored, key=lambda cand: cand.weighted_similarity)[:2]
    payload = {
        "expression": expression,
        "definition_correct": definition_correct,
        "distractors": [
            {"text": cand.text, "similarity_score": round(cand.weighted_similarity, 3)} for cand in filtered[:2]
        ],
        "validation_metrics": {
            "BLEU": [cand.bleu for cand in filtered[:2]],
            "ROUGE": [cand.rouge for cand in filtered[:2]],
            "BERTScore": [cand.bert_score for cand in filtered[:2]],
        },
    }
    return payload


__all__ = ["generate_distractors"]
