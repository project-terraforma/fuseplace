"""Run both rule-based and ML conflation, then evaluate against golden labels."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.utils.io import DEFAULT_DATA_PATH, REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based + ML conflation and evaluate")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    steps = [
        ("Rule-based", ["python3", "-m", "scripts.conflation.rule_based_selection", "--input", str(input_path)]),
        ("ML (Random Forest)", ["python3", "-m", "scripts.conflation.ml_selection", "--input", str(input_path)]),
        ("Evaluate", ["python3", "-m", "scripts.conflation.evaluate_methods"]),
    ]

    for name, cmd in steps:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error: {name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\nDone. Results in {REPORTS_DIR / 'conflation'}")


if __name__ == "__main__":
    main()
