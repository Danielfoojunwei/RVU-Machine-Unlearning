from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PrivKind(str, Enum):
    FILE_WRITE = "FILE_WRITE"
    FILE_READ = "FILE_READ"
    NET_REQUEST = "NET_REQUEST"
    CONFIG_EDIT = "CONFIG_EDIT"
    MEMORY_WRITE = "MEMORY_WRITE"


@dataclass(frozen=True)
class PrivAction:
    kind: PrivKind
    scope: str
    args_hash: str
    deps: set[str] = field(default_factory=set)
