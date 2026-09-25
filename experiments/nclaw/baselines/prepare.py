"""Fetch pinned NCLaw and install the baseline launcher and evaluation patch."""

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "upstream.json").read_text())
DEFAULT_DEST = HERE.parents[2] / "out" / "nclaw_baselines" / "NCLaw"


def git(dest, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(dest), *map(str, args)],
        check=check,
        capture_output=True,
        text=True,
    )


def prepare(dest: Path, cuda12=False):
    dest = dest.resolve()
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", PIN["repository"], str(dest)],
            check=True,
        )
        git(dest, "checkout", "--detach", PIN["commit"])
    head = git(dest, "rev-parse", "HEAD").stdout.strip()
    if head != PIN["commit"]:
        raise RuntimeError(f"Expected NCLaw {PIN['commit']}, found {head}; use a fresh directory.")

    launcher = dest / "experiments/scripts/train/sysid.py"
    content = (HERE / "sysid.py").read_bytes()
    if launcher.exists() and launcher.read_bytes() != content:
        raise RuntimeError(f"Refusing to overwrite a different launcher: {launcher}")

    patches = [HERE / "upstream.patch"]
    if cuda12:
        patches.append(HERE / "cuda12.patch")
    pending = []
    for patch in patches:
        forward = git(dest, "apply", "--check", patch, check=False)
        if forward.returncode == 0:
            pending.append(patch)
        elif git(dest, "apply", "--reverse", "--check", patch, check=False).returncode != 0:
            raise RuntimeError(f"NCLaw files conflict with {patch.name}:\n{forward.stderr}")
    for patch in pending:
        git(dest, "apply", patch)
    launcher.write_bytes(content)
    print(f"Prepared {dest}\nUpstream revision: {head}")
    print("Install NCLaw in a separate environment; see the baseline README.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--cuda12", action="store_true", help="Apply bundled Warp CUDA 12 compatibility fixes"
    )
    args = parser.parse_args()
    prepare(args.dest, cuda12=args.cuda12)


if __name__ == "__main__":
    main()
