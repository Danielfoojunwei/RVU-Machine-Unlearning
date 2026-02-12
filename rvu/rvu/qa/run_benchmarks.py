from __future__ import annotations

import argparse
from pathlib import Path

from rvu.bench.config import load_config
from rvu.bench.runner import run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_config(path=args.config)
    run_dir = run_benchmarks(config=config, smoke=args.smoke)
    print(run_dir)


if __name__ == "__main__":
    main()
