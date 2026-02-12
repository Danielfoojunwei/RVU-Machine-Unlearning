from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Policy:
    allowed_requesters: frozenset[str] = frozenset({"admin"})

    def authz(self, k_set: set[str], requester: str) -> bool:
        return bool(k_set) and requester in self.allowed_requesters


@dataclass(frozen=True)
class ControlPlane:
    sandbox_root: str = "sandbox"
    allowlist_paths: tuple[str, ...] = ("sandbox",)
    immutable_config: dict[str, str] = field(default_factory=lambda: {"version": "1"})

    def export(self) -> dict[str, object]:
        return {
            "sandbox_root": self.sandbox_root,
            "allowlist_paths": list(self.allowlist_paths),
            "immutable_config": dict(self.immutable_config),
        }
