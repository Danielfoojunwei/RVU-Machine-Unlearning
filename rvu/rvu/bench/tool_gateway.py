from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rvu.runtime.privilege import PrivAction, PrivKind


@dataclass(frozen=True)
class ToolPolicy:
    allowed_tools: set[str]
    allowed_write_roots: set[Path]
    allowed_http_hosts: set[str]


@dataclass
class ToolEvent:
    tool_name: str
    args: dict[str, Any]
    result: str
    deps: set[str]


@dataclass
class ToolGateway:
    policy: ToolPolicy
    trace: list[ToolEvent] = field(default_factory=list)

    def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        deps: set[str],
        invoker: Callable[[dict[str, Any]], str],
    ) -> tuple[str, PrivAction]:
        if tool_name not in self.policy.allowed_tools:
            raise PermissionError(f"tool {tool_name} not allowed")
        if tool_name == "file_write":
            path = Path(str(args["path"]))
            if not any(path.is_relative_to(root) for root in self.policy.allowed_write_roots):
                raise PermissionError(f"path {path} outside of allowed roots")
        if tool_name == "http_get":
            host = str(args["host"])
            if host not in self.policy.allowed_http_hosts:
                raise PermissionError(f"host {host} outside allowlist")

        result = invoker(args)
        self.trace.append(ToolEvent(tool_name=tool_name, args=args, result=result, deps=deps))

        action_kind = PrivKind.FILE_WRITE if tool_name == "file_write" else PrivKind.MEMORY_WRITE
        action = PrivAction(
            kind=action_kind,
            scope=tool_name,
            args_hash=hashlib.sha256(json.dumps(args, sort_keys=True).encode("utf-8")).hexdigest(),
            deps=deps,
        )
        return result, action
