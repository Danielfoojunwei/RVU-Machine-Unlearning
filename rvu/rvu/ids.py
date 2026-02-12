from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class IdGenerator:
    """Monotonic-ish ID generator with deterministic test mode."""

    deterministic: bool = False
    prefix: str = "a"
    _counter: int = field(default=0, init=False)

    def new_id(self) -> str:
        if self.deterministic:
            self._counter += 1
            return f"{self.prefix}-{self._counter:08d}"
        return uuid4().hex
