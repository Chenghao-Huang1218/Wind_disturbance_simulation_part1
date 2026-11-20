#!/usr/bin/env python3
"""Reproduce the X4-950 randomized drag development and independent test."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training import randomized_v3 as study  # noqa: E402


def main() -> int:
    """Dispatch study stages that write separate immutable output directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "train", "evaluate"):
        command = commands.add_parser(stage)
        command.add_argument("--output-dir", type=Path, required=True)
        if stage == "prepare":
            command.add_argument("--reference", type=Path, default=study.DEFAULT_REFERENCE)
        else:
            command.add_argument("--manifest", type=Path, required=True)
            if stage == "train":
                command.add_argument("--steps", type=int, default=6000)
            else:
                command.add_argument("--checkpoint", type=Path, required=True)
                command.add_argument("--baseline", type=Path,
                                     default=study.DEFAULT_REFERENCE.with_name("candidate.pt"))
    args = parser.parse_args()
    if args.stage == "prepare":
        path = study.prepare(args.reference, args.output_dir)
    elif args.stage == "train":
        path = study.train(args.manifest, args.output_dir, steps=args.steps)
    else:
        path = study.evaluate(args.manifest, args.checkpoint, args.baseline, args.output_dir)
    print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
