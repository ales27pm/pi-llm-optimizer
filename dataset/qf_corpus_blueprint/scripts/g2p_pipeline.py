"""Québécois French G2P pipeline compatible with Montreal Forced Aligner."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:  # pragma: no cover - allow import as script or package
    from .normalizer import NormalizationEngine
except ImportError:  # pragma: no cover
    from normalizer import NormalizationEngine

LOGGER = logging.getLogger("qf.g2p")

FALLBACK_MAPPING = {
    "a": "a",
    "â": "ɑ",
    "b": "b",
    "c": "k",
    "d": "d",
    "e": "ə",
    "é": "e",
    "è": "ɛ",
    "ê": "ɛ",
    "f": "f",
    "g": "g",
    "h": "",
    "i": "i",
    "j": "ʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ô": "o",
    "p": "p",
    "q": "k",
    "r": "ʁ",
    "s": "s",
    "t": "t",
    "u": "y",
    "û": "y",
    "v": "v",
    "w": "w",
    "x": "ks",
    "y": "i",
    "z": "z",
    "'": "",
    " ": " ",
}


@dataclass
class AlignmentResult:
    orthography: str
    phonetic: str
    phone_set: str
    alignment_confidence: float
    segments: List[Dict[str, float]]


class ProsodylabG2PPipeline:
    """High-level wrapper around MFA with a robust fallback."""

    def __init__(
        self,
        *,
        phone_set: str = "Prosodylab_QF_v2",
        g2p_model_path: Optional[Path] = None,
        dictionary_path: Optional[Path] = None,
    ) -> None:
        self.phone_set = phone_set
        self.g2p_model_path = g2p_model_path
        self.dictionary_path = dictionary_path
        self.engine = NormalizationEngine.from_yaml()
        self._mfa_g2p = None
        self._attempt_mfa_initialisation()

    def _attempt_mfa_initialisation(self) -> None:
        try:
            from montreal_forced_aligner.g2p.generator import PyniniValidator, PyniniWordGenerator
            from montreal_forced_aligner.models import G2PModel
        except Exception:  # pragma: no cover - optional dependency
            LOGGER.info("Montreal Forced Aligner is not available; falling back to internal G2P map")
            return

        model_path = self.g2p_model_path
        if model_path is None:
            try:
                model = G2PModel("prosodylab_qc")
            except Exception:  # pragma: no cover - optional download failure
                LOGGER.warning("Unable to locate default MFA Prosodylab model; fallback map will be used")
                return
        else:
            model = G2PModel(model_path)

        self._mfa_g2p = PyniniWordGenerator(model, validator=PyniniValidator(model.fst_path))
        LOGGER.info("Loaded MFA G2P model %s", model.identifier)

    def _fallback_g2p(self, text: str) -> str:
        phonemes: List[str] = []
        for char in text.lower():
            symbol = FALLBACK_MAPPING.get(char, char)
            if not symbol:
                continue
            if symbol.isspace():
                phonemes.append("|")
            else:
                phonemes.extend(symbol.split())
        cleaned: List[str] = [phone for phone in phonemes if phone and phone != "|"]
        return " ".join(cleaned)

    def _run_g2p(self, text: str) -> str:
        if self._mfa_g2p is None:
            return self._fallback_g2p(text)
        try:  # pragma: no cover - depends on optional lib
            phones = self._mfa_g2p(text)
        except Exception as exc:
            LOGGER.exception("MFA G2P generation failed: %s", exc)
            return self._fallback_g2p(text)
        return " ".join(phones)

    def _align(self, phones: List[str]) -> List[Dict[str, float]]:
        if not phones:
            return []
        duration = max(0.01, 1.0 / max(len(phones), 1))
        return [
            {"phone": phone, "start": index * duration, "end": (index + 1) * duration}
            for index, phone in enumerate(phones)
        ]

    def process_text(self, text: str) -> AlignmentResult:
        normalized, _, _ = self.engine.normalize(text)
        phonetic = self._run_g2p(normalized)
        phones = [phone for phone in phonetic.split(" ") if phone]
        segments = self._align(phones)
        confidence = 0.97 if self._mfa_g2p is not None else 0.75
        return AlignmentResult(
            orthography=normalized,
            phonetic=phonetic,
            phone_set=self.phone_set,
            alignment_confidence=confidence,
            segments=segments,
        )

    def process_record(self, record: Dict[str, str]) -> Dict[str, object]:
        text = record.get("text_orthographic_raw") or record.get("text", "")
        if not text:
            raise ValueError("Record must contain a text field for G2P processing")
        result = self.process_text(text)
        payload = {
            "text_orthographic_raw": text,
            "text_normalized_mf": result.orthography,
            "phonetic_transcription_QF": result.phonetic,
            "phone_set": result.phone_set,
            "alignment_confidence": result.alignment_confidence,
            "segments": result.segments,
        }
        payload.update({k: v for k, v in record.items() if k not in payload})
        return payload


__all__ = ["ProsodylabG2PPipeline", "AlignmentResult"]
