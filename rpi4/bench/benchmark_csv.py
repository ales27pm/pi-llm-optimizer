from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class BenchmarkRow:
    timestamp: datetime
    tokens_per_second: float
    init_ms: float
    decode_ms_per_token: float
    embed_ms: float

class BenchmarkCSVWriter:
    HEADER = "timestamp,tokens_per_second,init_ms,decode_ms_per_token,embed_ms"

    @staticmethod
    def _fmt(x: float) -> str:
        return f"{x:.2f}"

    def render(self, samples: List[BenchmarkRow]) -> str:
        lines = [self.HEADER]
        for s in samples:
            ts = (
                s.timestamp.astimezone(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            )
            lines.append(",".join([
                ts,
                self._fmt(s.tokens_per_second),
                self._fmt(s.init_ms),
                self._fmt(s.decode_ms_per_token),
                self._fmt(s.embed_ms),
            ]))
        return "\n".join(lines) + "\n"

    def write(self, samples: List[BenchmarkRow], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(samples), encoding="utf-8")
