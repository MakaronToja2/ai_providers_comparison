"""
Per-instance trajectory log: one JSON line per event (header, API call attempt, tool
result, summary). Lines are flushed as they are written so a crash leaves a readable
partial file.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Tool results are stored in full in the results database; the trajectory keeps a preview.
TOOL_RESULT_PREVIEW_CHARS = 2000


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TrajectoryRecorder:
    """Writes a JSONL trajectory. With no path it records nothing."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._partial: Optional[Path] = None
        self._file = None
        self._seq = 0

    def _open(self) -> None:
        if self.path is None or self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the log of an earlier attempt (for example one that ended in an API error)
        if self.path.exists():
            n = 1
            while self.path.with_suffix(f".prev{n}.jsonl").exists():
                n += 1
            self.path.rename(self.path.with_suffix(f".prev{n}.jsonl"))
        self._partial = self.path.with_suffix(".jsonl.partial")
        self._file = open(self._partial, "w", encoding="utf-8")

    def _write(self, record: Dict[str, Any]) -> None:
        if self.path is None:
            return
        self._open()
        self._seq += 1
        record = {"seq": self._seq, "ts": utc_now_iso(), **record}
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def header(self, **fields: Any) -> None:
        self._write({"type": "header", **fields})

    def api_call(self, **fields: Any) -> None:
        self._write({"type": "api_call", **fields})

    def tool_result(self, *, content: str, **fields: Any) -> None:
        self._write({
            "type": "tool_result",
            **fields,
            "content_chars": len(content),
            "content_preview": content[:TOOL_RESULT_PREVIEW_CHARS],
        })

    def summary(self, **fields: Any) -> None:
        """Write the closing record and move the file to its final name."""
        self._write({"type": "summary", **fields})
        if self._file is not None:
            self._file.close()
            self._file = None
            self._partial.rename(self.path)
