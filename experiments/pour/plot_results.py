"""Plot the frozen pouring predictions and hardware measurements."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reproduce.run import ROOT


def plot(output):
    source = ROOT / 'out/pour_hardware_receiver_remap_review_20260911/summary.csv'
    rows = list(csv.DictReader(source.open()))
    target = np.array([float(r['target_ml']) for r in rows])
    means = np.array([float(r['mean_ml']) for r in rows])
    deviation = np.array([float(r['sample_sd_ml']) for r in rows])
    predicted = np.array([float(r['mpm_receiver_ml']) for r in rows])
    water_path = ROOT / 'out/pour_water_figure_20260915/water_receiver_readings.csv'
    water = list(csv.DictReader(water_path.open()))
    fig, ax = plt.subplots(figsize=(5.5, 3.8), layout='constrained')
    ax.plot([50, 170], [50, 170], ':', color='.45', label='Target')
    ax.plot(target, predicted, '--s', color='#2375aa', markerfacecolor='white', label='Glycerin: MPM')
    ax.errorbar(target, means, yerr=deviation, fmt='o-', color='#c96932', capsize=3,
                label='Glycerin: hardware, mean ± SD (5 trials)')
    ax.plot([float(r['target_ml']) for r in water], [float(r['receiver_ml']) for r in water],
            '^-', color='#243c58', markerfacecolor='white', label='Water: hardware (1 trial)')
    ax.set(xlabel='Target volume (mL)', ylabel='Measured / predicted volume (mL)', xlim=(50, 170))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'pouring.{suffix}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'out/reproduced/pouring-plot')
    plot(parser.parse_args().out)
