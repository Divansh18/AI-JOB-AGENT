"""On-disk response cache keyed by (prompt_hash, model).

Re-running a stage during development must never re-bill. Cache entries are
plain JSON files so they can be inspected and deleted by hand.
"""

from __future__ import annotations

import json
from pathlib import Path


class ResponseCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, prompt_hash: str, model: str) -> Path:
        return self.dir / f"{model.replace('/', '_')}__{prompt_hash[:32]}.json"

    def get(self, prompt_hash: str, model: str) -> dict | None:
        path = self._path(prompt_hash, model)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, prompt_hash: str, model: str, payload: dict) -> None:
        try:
            self._path(prompt_hash, model).write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass  # a cache miss is never fatal

    def clear(self) -> int:
        n = 0
        for path in self.dir.glob("*.json"):
            path.unlink(missing_ok=True)
            n += 1
        return n
