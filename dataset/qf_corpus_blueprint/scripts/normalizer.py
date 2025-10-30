"""Advanced normalization pipeline for Québecois French text."""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

LOGGER = logging.getLogger("qf.normalizer")

SCRIPT_DIR = Path(__file__).resolve().parent
RULES_PATH = SCRIPT_DIR.parent / "rules" / "normalization.yaml"

INVISIBLE_CHARACTERS = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\ufeff",  # zero-width no-break space (BOM)
}

SEPARATOR_CHARACTERS = {
    "\u00a0",  # non-breaking space
    "\u2028",  # line separator
    "\u2029",  # paragraph separator
}

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class AppliedRule:
    """Metadata for a rule application."""

    category: str
    pattern: str
    replacement: str
    tag: Optional[str]
    count: int
    description: Optional[str] = None


class NormalizationRule:
    """Internal representation of a compiled normalization rule."""

    def __init__(
        self,
        category: str,
        pattern: str,
        replacement: str,
        *,
        tag: Optional[str] = None,
        flags: Sequence[str] | None = None,
        description: Optional[str] = None,
    ) -> None:
        self.category = category
        self.pattern = pattern
        self.replacement = replacement
        self.tag = tag
        self.description = description
        flag_value = 0
        if flags:
            for flag in flags:
                try:
                    flag_value |= getattr(re, flag)
                except AttributeError as exc:  # pragma: no cover - configuration guard
                    raise ValueError(f"Unsupported regex flag '{flag}' in YAML configuration") from exc
        self._compiled = re.compile(pattern, flag_value)

    def apply(self, text: str) -> Tuple[str, Optional[AppliedRule]]:
        matches = list(self._compiled.finditer(text))
        if not matches:
            return text, None
        substituted = self._compiled.sub(self.replacement, text)
        return substituted, AppliedRule(
            category=self.category,
            pattern=self.pattern,
            replacement=self.replacement,
            tag=self.tag,
            count=len(matches),
            description=self.description,
        )


class NormalizationEngine:
    """Loads YAML rules and executes them sequentially."""

    def __init__(self, rules: Sequence[NormalizationRule]) -> None:
        self.rules = list(rules)

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "NormalizationEngine":
        path = path or RULES_PATH
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        default_flags: Sequence[str] = raw.get("metadata", {}).get("default_regex_flags", [])
        rules: List[NormalizationRule] = []
        for category, entries in raw.items():
            if category == "metadata":
                continue
            for entry in entries or []:
                flags = entry.get("flags") or default_flags
                rules.append(
                    NormalizationRule(
                        category=category,
                        pattern=entry["pattern"],
                        replacement=entry["replacement"],
                        tag=entry.get("tag"),
                        flags=flags,
                        description=entry.get("description"),
                    )
                )
        return cls(rules)

    def normalize(self, text: str) -> Tuple[str, List[AppliedRule], List[str]]:
        clean_text = unicodedata.normalize("NFC", text)
        for separator in SEPARATOR_CHARACTERS:
            clean_text = clean_text.replace(separator, " ")
        for invisible in INVISIBLE_CHARACTERS:
            clean_text = clean_text.replace(invisible, "")
        clean_text = CONTROL_CHAR_PATTERN.sub("", clean_text)

        applied: List[AppliedRule] = []
        tags: List[str] = []
        current = clean_text
        for rule in self.rules:
            current, rule_application = rule.apply(current)
            if rule_application:
                applied.append(rule_application)
                if rule_application.tag:
                    tags.append(rule_application.tag)
        current = re.sub(r"\s+", " ", current).strip()
        return current, applied, tags


class NeuralLemmatizer:
    """Optional neural lemmatizer backed by spaCy or Stanza."""

    def __init__(self, backend: str = "spacy", model: Optional[str] = None) -> None:
        self.backend = backend
        self.model_name = model
        self._pipeline = None
        self._initialise()

    def _initialise(self) -> None:
        backend = (self.backend or "").lower()
        if backend == "spacy":
            try:
                import spacy
            except ImportError:
                LOGGER.info("spaCy not available; neural lemmatization disabled")
                return
            model_name = self.model_name or "fr_core_news_md"
            try:
                self._pipeline = spacy.load(model_name)
            except OSError:
                LOGGER.warning(
                    "spaCy model '%s' could not be loaded. Install it with 'python -m spacy download %s'",
                    model_name,
                    model_name,
                )
                self._pipeline = None
        elif backend == "stanza":
            try:
                import stanza
            except ImportError:
                LOGGER.info("Stanza not available; neural lemmatization disabled")
                return
            model_name = self.model_name or "fr"
            try:
                stanza.download(model_name, processors="tokenize,pos,lemma", verbose=False)
            except Exception:  # pragma: no cover - network issues
                LOGGER.debug("Stanza resources may already exist or cannot be downloaded")
            self._pipeline = stanza.Pipeline(lang=model_name, processors="tokenize,pos,lemma", verbose=False)
        elif backend in {"", "none"}:
            LOGGER.info("Neural lemmatizer explicitly disabled")
        else:  # pragma: no cover - configuration guard
            raise ValueError(f"Unsupported lemmatizer backend '{self.backend}'")

    def is_available(self) -> bool:
        return self._pipeline is not None

    def lemmatize(self, text: str) -> Dict[str, Iterable[str]]:
        if not self.is_available():
            return {"backend": None, "lemmas": []}
        backend = (self.backend or "").lower()
        if backend == "spacy":
            doc = self._pipeline(text)
            lemmas = [token.lemma_ for token in doc]
        else:  # stanza
            doc = self._pipeline(text)
            lemmas = [word.lemma for sentence in doc.sentences for word in sentence.words]
        return {"backend": backend, "lemmas": lemmas}


def normalize_text_qf(
    text: str,
    *,
    engine: Optional[NormalizationEngine] = None,
    lemmatizer: Optional[NeuralLemmatizer] = None,
) -> Dict[str, object]:
    """Normalize text with YAML rules followed by an optional neural lemmatizer."""

    engine = engine or NormalizationEngine.from_yaml()
    normalized_text, applied_rules, rule_tags = engine.normalize(text)

    lemma_payload: Dict[str, Iterable[str]]
    if lemmatizer is None:
        lemmatizer = NeuralLemmatizer(backend="none")
    lemma_payload = lemmatizer.lemmatize(normalized_text)

    return {
        "output": normalized_text,
        "applied_rules": [rule.__dict__ for rule in applied_rules],
        "rule_tags": rule_tags,
        "lemmatizer": lemma_payload,
    }


__all__ = [
    "AppliedRule",
    "NormalizationEngine",
    "NeuralLemmatizer",
    "normalize_text_qf",
]
