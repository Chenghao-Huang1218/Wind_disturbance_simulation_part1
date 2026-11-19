#!/usr/bin/env python3
"""Freeze a selected X4 10 m/s weight, or open its independently sampled test."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training.formal_v2 import evaluate_frozen, freeze_evaluation  # noqa: E402
from wind_field_simulation.training.xml_parameters import PROJECT_ROOT  # noqa: E402


def main() -> None:
    """Expose explicit freeze and one-time formal test phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    freeze = stages.add_parser("freeze")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--checkpoint", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--baseline", type=Path, default=PROJECT_ROOT /
                        "src/wind_field_simulation/assets/checkpoints/x4_950_wind_v1/candidate.pt")
    test = stages.add_parser("test")
    test.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze_evaluation(args.manifest, args.checkpoint, args.baseline, args.protocol)
    else:
        evaluate_frozen(args.protocol)


if __name__ == "__main__":
    main()
