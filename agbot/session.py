from __future__ import annotations

import json
import time
from pathlib import Path


class Session:
    """Append-only JSONL log of everything that happens in a session."""

    def __init__(self, directory: Path, slug: str = "session") -> None:
        directory.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in slug.lower())
        safe = safe.strip("-")[:40] or "session"
        self.path = directory / f"{ts}-{safe}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self.log("session_start", file=str(self.path))

    def log(self, kind: str, **data) -> None:
        record = {"ts": time.time(), "kind": kind, **data}
        self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def log_message(self, message: dict) -> None:
        self.log("message", message=message)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


def latest_session_file(directory: Path) -> Path | None:
    files = sorted(Path(directory).glob("*.jsonl"))
    return files[-1] if files else None


def load_messages(path: Path) -> list[dict]:
    """Load conversation messages from a session log (system prompt excluded)."""
    messages: list[dict] = []
    if not Path(path).is_file():
        return messages
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if record.get("kind") == "message":
            msg = record.get("message")
            if isinstance(msg, dict) and msg.get("role") != "system":
                messages.append(msg)
    return messages
