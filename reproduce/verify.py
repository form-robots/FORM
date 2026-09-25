"""Recompute inexpensive paper estimates and check bundled numerical references."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import json
from pathlib import Path

import numpy as np
from shapely.geometry import shape

from reproduce.run import ROOT, identify, read


def verify(output):
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for kind in ['elastic', 'plastic', 'hardware', 'fluid']:
        with (output / f'{kind}.log').open('w') as log, contextlib.redirect_stdout(log):
            identify(kind, output / kind)
    for material in 'AB':
        actual = read(output / 'elastic' / material / 'identification.json')['E_pa']
        expected = read(ROOT / f'out/strip_texture_20260912/fit_{material}/identification.json')['E_pa']
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        results[f'elastic_{material}_E_pa'] = actual
        actual = read(output / 'plastic' / material / 'identification.json')
        expected = read(ROOT / f'out/press_separated_20260913/monotonic320/separated_{material}/identification.json')
        for key in ['E_pa', 'yield_pa']:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12)
            results[f'plastic_{material}_{key}'] = actual[key]
    for material in ['play_doh', 'butter_slime', 'plasticine']:
        actual = read(output / 'hardware' / material / 'identification.json')['values']
        expected = read(ROOT / f'out/press_weakform_restart_20260914/{material}/identification.json')['values']
        for key in ['E_pa', 'yield_pa']:
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12)
        results[material] = actual
    eta = read(output / 'fluid/identification.json')['eta']
    np.testing.assert_allclose(eta, 3.4392377844275503, rtol=1e-12)
    results['viscosity_Pa_s'] = eta
    base = ROOT / 'out/shaping_hardware_iou_recheck_20260915/phototextured'
    for name, relative, expected in [
        ('red', 'top_four_color_cleaned/scan0003_scored_shapes.geojson', .7565534771499216),
        ('yellow', 'top_four_color_cleaned/scan0001_scored_shapes.geojson', .7235865570367305),
        ('gray', 'gray_material/scored_shapes.geojson', .7782817837969541)]:
        features = read(base / relative)['features']
        polygons = {f['properties']['role']: shape(f['geometry']) for f in features}
        target = polygons.pop('target')
        reconstructed, = polygons.values()
        iou = reconstructed.intersection(target).area / reconstructed.union(target).area
        np.testing.assert_allclose(iou, expected, rtol=1e-12)
        results[f'{name}_shape_IoU'] = iou
    source = ROOT / 'out/pour_hardware_receiver_remap_review_20260911'
    rows = list(csv.DictReader((source / 'command_photo_mapping.csv').open()))
    summary = {int(r['target_ml']): r for r in csv.DictReader((source / 'summary.csv').open())}
    errors = []
    for target in [60, 80, 100, 120, 140, 160]:
        values = np.array([float(r['receiver_ml']) for r in rows if int(r['target_ml']) == target])
        assert len(values) == 5
        np.testing.assert_allclose(values.mean(), float(summary[target]['mean_ml']), rtol=1e-12)
        np.testing.assert_allclose(values.std(ddof=1), float(summary[target]['sample_sd_ml']), rtol=1e-12)
        errors.append(abs(values.mean() - target))
    results['pouring_maximum_mean_error_ml'] = max(errors)
    assert round(max(errors), 1) == 3.8
    report = dict(status='passed', checks=results,
                  scope='Coefficient solves and saved-shape/measurement metrics. No new GPU rollout or baseline training.',
                  numerical_relative_tolerance=1e-12)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'out/verification' /
                        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    verify(parser.parse_args().out)
