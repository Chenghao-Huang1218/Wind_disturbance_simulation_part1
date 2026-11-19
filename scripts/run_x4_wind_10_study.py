#!/usr/bin/env python3
"""Reproduce the grouped 0–10 m/s X4 synthetic-reference study."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_field_simulation.training import study_v2  # noqa: E402
from wind_field_simulation.training.xml_parameters import DEFAULT_SCENE, PROJECT_ROOT  # noqa: E402


def main() -> None:
    """Run one explicit, non-overwriting study stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    export = stages.add_parser("export")
    export.add_argument("--xml-path", type=Path, default=DEFAULT_SCENE)
    export.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pinn/x4_950_reference_wind_10.json")
    export.add_argument("--output-dir", type=Path, required=True)
    prepare = stages.add_parser("prepare")
    prepare.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pinn/x4_950_wind_10_study.json")
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    train = stages.add_parser("train")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "export":
        print(study_v2.export_wide_reference(args.xml_path, args.config, args.output_dir))
    elif args.stage == "prepare":
        study_v2.prepare_development(args.config, args.reference, args.output_dir)
    else:
        study_v2.train_study(args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
