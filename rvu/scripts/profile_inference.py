from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rvu.bench.config import load_config
from rvu.bench.model import build_planner

PROMPT = "Summarize the following bullet list in two sentences: apples, bananas, oranges."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    planner = build_planner(config.model)

    start = time.perf_counter()
    lengths: list[int] = []
    for _ in range(args.repeats):
        text = planner.generate(PROMPT)
        lengths.append(len(text.split()))
    elapsed = time.perf_counter() - start

    payload = {
        "repeats": args.repeats,
        "elapsed_s": elapsed,
        "avg_latency_s": elapsed / args.repeats,
        "avg_output_tokens_approx": sum(lengths) / len(lengths),
        "tok_per_s_approx": (sum(lengths) / elapsed) if elapsed > 0 else 0.0,
        "model": config.model.model_id,
        "revision": config.model.revision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
