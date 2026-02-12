from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    revision: str
    endpoint: str | None = None
    dtype: str = "bfloat16"
    max_tokens: int = 1024


@dataclass(frozen=True)
class BenchConfig:
    output_root: Path
    sandbox_root: Path
    model: ModelConfig
    run_agentdojo: bool = True
    run_injecagent: bool = True
    run_bipia: bool = True


@dataclass(frozen=True)
class RunMetadata:
    commit_hash: str
    gpu_name: str
    machine: str


def load_config(path: Path) -> BenchConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model_raw = raw["model"]
    return BenchConfig(
        output_root=Path(raw["output_root"]),
        sandbox_root=Path(raw["sandbox_root"]),
        model=ModelConfig(
            provider=model_raw["provider"],
            model_id=model_raw["model_id"],
            revision=model_raw["revision"],
            endpoint=model_raw.get("endpoint"),
            dtype=model_raw.get("dtype", "bfloat16"),
            max_tokens=int(model_raw.get("max_tokens", 1024)),
        ),
        run_agentdojo=bool(raw.get("run_agentdojo", True)),
        run_injecagent=bool(raw.get("run_injecagent", True)),
        run_bipia=bool(raw.get("run_bipia", True)),
    )
