"""Thin strip, textured stereo observations, and frozen-parameter replay.

The old dot-marker experiment is unchanged. All outputs are written into a
fresh directory; truth is available to rendering/evaluation, not identification.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.elastic.strip_bending_study import PROTOCOL, record


def protocol():
    p=copy.deepcopy(PROTOCOL)
    p.update(size=[.012,.004,.100],grid=192,fine_grid=256,pusher_length=.016,
             inputs=['calibrated stereo pixels','net pusher force','known specimen metadata'],
             selection='Width normal to observed face reduced from 20 to 4 mm to support plane stress; unchanged E, nu, in-plane geometry, motion, and timing. Main/fine grids resolve width with 3.2/4.27 cells. No selection on recovered E.')
    return p


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['init','truth','fine','prediction','fine_prediction'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--material',choices=['A','B'],default='A')
    ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args()
    if a.stage=='init':
        a.out.mkdir(parents=True,exist_ok=True)
        path=a.out/'protocol.json'
        if path.exists(): raise FileExistsError(path)
        path.write_text(json.dumps(protocol(),indent=2)+'\n')
        return
    p=json.loads((a.out/'protocol.json').read_text())
    E=p['E_pa'][a.material]
    if 'prediction' in a.stage:
        E=json.loads((a.out/f'fit_{a.material}/identification.json').read_text())['E_pa']
    grid=p['fine_grid'] if 'fine' in a.stage else p['grid']
    record(a.out,f'{a.stage}_{a.material}',E,grid,a.device,states=a.stage=='truth')


if __name__=='__main__': main()
