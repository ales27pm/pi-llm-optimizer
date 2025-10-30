# -*- coding: utf-8 -*-
"""
Prompt templates for LLaMA-style instruction-following Chat.
"""
from typing import Dict, Optional

SYSTEM_FALLBACK = "Réponds comme un ami québécois naturel et sympathique."


def llama_inst(system: str, user: str, assistant: Optional[str] = None) -> Dict[str, str]:
    sys = system or SYSTEM_FALLBACK
    prompt = f"[INST] <<SYS>> {sys} <</SYS>> {user} [/INST]"
    if assistant is None:
        return {"prompt": prompt}
    return {"prompt": f"{prompt} {assistant}"}


def extract_fields(sample: Dict) -> Dict[str, str]:
    return {
        "system": sample.get("system_hint") or SYSTEM_FALLBACK,
        "user": sample.get("user", ""),
        "assistant": sample.get("assistant") or sample.get("response") or "",
    }
