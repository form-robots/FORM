"""Reconstruct pressing observations, with outputs separate from frozen data."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

from reproduce.run import ROOT, read
from reproduce.planning import save


def pressing(output, episodes, *, profiles_only=False):
    from experiments.robotics import press_hardware_measured as measured
    from experiments.robotics import press_hardware_variable_stokes as interior

    reference = ROOT / 'out/press_weakform_restart_20260914'
    output.mkdir(parents=True, exist_ok=False)
    profiles = output / 'profiles'
    profiles.mkdir()
    protocol = read(reference / 'protocol.json')
    save(profiles / 'protocol.json', protocol)
    reconstructed = output / 'interior'
    reconstructed.mkdir()
    save(reconstructed / 'protocol.json', protocol)
    for episode in episodes:
        if profiles_only:
            dest = profiles / episode
            dest.mkdir()
            source = reference / 'observation_inputs' / episode
            for name in ['camera.json', 'profile_observations.npz']:
                shutil.copy2(source / name, dest / name)
        else:
            measured.prepare(profiles, episode)
        samples = np.load(profiles / episode / 'profile_observations.npz')['samples']
        measured.assemble(profiles, episode, samples, read(profiles / episode / 'camera.json'),
                          baseline_correction=True)
        interior.reconstruct(profiles, reconstructed, episode, substeps=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', default=['ep0003', 'ep0007', 'ep0011'])
    parser.add_argument('--profiles-only', action='store_true', help='Start from saved surface profiles')
    args = parser.parse_args()
    pressing(args.out, args.episodes, profiles_only=args.profiles_only)
