"""
Thin interface for using the PeskaVLP pretraining objective with this project.

This does not call PeskaVLP's model or dataset builders. It launches the local
train_frozen_vis.py entrypoint, which builds this repository's VLP model and
PretrainDataset, then enables --training_method peskavlp. In that mode,
train_frozen_vis.py feeds local embeddings into PeskaVLP's copied losses:
SSL_VL_Loss_new, hier_infonce, and hier_infonce_dtw.

Examples:
    python train_peskavlp.py --dry-run --epochs 50
    python train_peskavlp.py --epochs 50 --main_csv_path surglavi_level_csv/all_video.csv
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional, Sequence


CLIP_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_SCRIPT = CLIP_ROOT / "train_frozen_vis.py"


def build_command(
    *,
    train_script: Optional[os.PathLike] = None,
    python_executable: Optional[str] = None,
    train_args: Optional[Sequence[str]] = None,
) -> List[str]:
    script = Path(train_script) if train_script is not None else DEFAULT_TRAIN_SCRIPT
    script = script.expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Missing local training script: {script}")

    command = [
        python_executable or sys.executable,
        str(script),
    ]
    if train_args:
        command.extend(train_args)
    command.extend(["--training_method", "peskavlp"])
    return command


def run_training(
    *,
    train_script: Optional[os.PathLike] = None,
    python_executable: Optional[str] = None,
    train_args: Optional[Sequence[str]] = None,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess:
    command = build_command(
        train_script=train_script,
        python_executable=python_executable,
        train_args=train_args,
    )

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if dry_run:
        print("cwd:", CLIP_ROOT)
        print("cmd:", " ".join(shlex.quote(part) for part in command))
        return subprocess.CompletedProcess(command, 0)

    return subprocess.run(command, cwd=str(CLIP_ROOT), env=merged_env, check=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch local train_frozen_vis.py with PeskaVLP pretraining losses. "
            "Unknown args are passed through to train_frozen_vis.py."
        )
    )
    parser.add_argument(
        "--train-script",
        default=str(DEFAULT_TRAIN_SCRIPT),
        help="Local training script to launch.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable used to launch local training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without starting training.",
    )
    args, train_args = parser.parse_known_args(argv)
    args.train_args = train_args
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_training(
        train_script=args.train_script,
        python_executable=args.python_executable,
        train_args=args.train_args,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
