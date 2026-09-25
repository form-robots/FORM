"""Train known-form Sys-ID in NCLaw; launcher supplied by Cheng-Hsi Hsiao."""

import subprocess
import sys
from pathlib import Path

from nclaw.constants import ENVS, PYTHON_PATH, RENDER
from nclaw.utils import dict_to_hydra, get_root, get_script_parser


def main():
    root = get_root(__file__)
    python_path = sys.executable if PYTHON_PATH is None else PYTHON_PATH
    parser = get_script_parser()
    base_args = vars(parser.parse_args())

    for env in ENVS:
        # Keep each environment's analytic material classes, optimizing their parameters.
        args = base_args | {
            "env": env,
            "env.blob.material.elasticity.requires_grad": True,
            "env.blob.material.elasticity.random": True,
            "env.blob.material.plasticity.requires_grad": True,
            "env.blob.material.plasticity.random": True,
            "render": RENDER,
            "sim": "low",
            "name": Path(env) / "train" / "sysid",
        }
        command = [python_path, root / "train.py", *dict_to_hydra(args)]
        subprocess.run([str(part) for part in command], check=True)


if __name__ == "__main__":
    main()
