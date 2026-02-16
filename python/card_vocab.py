from __future__ import annotations

import json
from pathlib import Path


class CardVocab:
    """Stable mapping from card name -> integer id.

    0 is reserved for padding/unknown.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name_to_id: dict[str, int] = {}
        self.next_id = 1
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.name_to_id = {str(k): int(v) for k, v in data.get("name_to_id", {}).items()}
        self.next_id = int(data.get("next_id", max(self.name_to_id.values(), default=0) + 1))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"name_to_id": self.name_to_id, "next_id": self.next_id}, indent=2, sort_keys=True)
        )

    def encode(self, name: str | None) -> int:
        if not name:
            return 0
        name = str(name)
        v = self.name_to_id.get(name)
        if v is not None:
            return v
        v = self.next_id
        self.next_id += 1
        self.name_to_id[name] = v
        # Save lazily; caller can also save periodically.
        self.save()
        return v
