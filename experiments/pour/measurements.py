"""Read photographed receiver graduations using the frozen paper annotations."""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from reproduce.run import ROOT, read

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def target_assignment(row, annotations):
    default_target=40+20*row['slot']
    overrides=annotations['mapping']['overrides']
    stem=Path(row['image']).stem
    if stem in overrides:
        assert row['seed']==1
    return dict(target_ml=overrides.get(stem,default_target),
        original_filename_target_ml=default_target,
        mapping_status='user_proposed_reassignment_unverified' if stem in overrides
                       else 'provisional_filename_order')

def measure_receiver(im, a):
    x=a['read_x_px']
    rgb=np.asarray(im,dtype=float)
    # The abrupt loss of blue at the amber/white interface. A vertical median
    # removes narrow red graduation strokes before edge selection.
    ratio=np.median(rgb[:,x:x+31,2]/np.maximum(rgb[:,x:x+31,0],1.),axis=1)
    profile=gaussian_filter1d(median_filter(ratio,size=17),2.)
    ticks={float(v):float(y) for v,y in a['ticks_y_px'].items()}
    lo,hi=int(ticks[250]+15),int(ticks[50]-12)
    y=int(lo+np.argmax(-np.gradient(profile)[lo:hi]))
    # Contrast quartiles document the visible transition width.
    above=float(np.median(profile[y-35:y-15]))
    below=float(np.median(profile[y+15:y+35]))
    assert above-below>.08, (a,y,above,below)
    ys=np.arange(y-15,y+16)
    edges=[int(ys[np.argmin(abs(profile[ys]-(above*(1-f)+below*f)))]) for f in [.25,.75]]
    positions=np.array([ticks[v] for v in sorted(ticks,reverse=True)])
    volumes=np.array(sorted(ticks,reverse=True))
    assert np.all(np.diff(positions)>0)
    assert positions[0]<y<positions[-1]
    ml=float(np.interp(y,positions,volumes))
    j=int(np.searchsorted(positions,y))
    return dict(receiver_ml=ml, receiver_read_low_ml=ml-12.5,
                receiver_read_high_ml=ml+12.5,receiver_meniscus_y_px=y,
                receiver_transition_low_y_px=min(edges),receiver_transition_high_y_px=max(edges),
                receiver_read_x_px=x,receiver_upper_tick_ml=float(volumes[j-1]),
                receiver_upper_tick_y_px=float(positions[j-1]),
                receiver_lower_tick_ml=float(volumes[j]),receiver_lower_tick_y_px=float(positions[j]),
                optical_contrast_blue_red=above-below)

def measure(output):
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'out/pour_hardware_receiver_remap_review_20260911'
    annotations = read(source / 'annotations.json')
    reference = list(csv.DictReader((source / 'command_photo_mapping.csv').open()))
    rows = []
    for row in reference:
        path = ROOT / row['image_path']
        with Image.open(path) as image:
            measured = measure_receiver(image, annotations['receiver'][path.stem])
        assignment = target_assignment(dict(seed=int(row['seed']), slot=int(row['slot']), image=path.name), annotations)
        rows.append(dict(image=row['image_path'], seed=int(row['seed']), **assignment, **measured))
    write_csv(output / 'readings.csv', rows)
    summary = []
    for target in [60, 80, 100, 120, 140, 160]:
        values = np.array([r['receiver_ml'] for r in rows if r['target_ml'] == target])
        assert len(values) == 5
        summary.append(dict(target_ml=target, mean_ml=float(values.mean()),
                            sample_sd_ml=float(values.std(ddof=1)), n=5))
    write_csv(output / 'summary.csv', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'out/reproduced/pour-measurements')
    measure(parser.parse_args().out)
