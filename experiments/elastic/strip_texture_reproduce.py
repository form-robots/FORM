"""Run the textured strip experiment from scratch into a new output directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(name,'1')
    from experiments.elastic.strip_texture_study import protocol,record
    from experiments.elastic.strip_texture_observe import render
    from experiments.elastic.strip_texture_identify import track,select_reconstruction
    from experiments.elastic.strip_texture_dic import refine
    from experiments.elastic.strip_camera_identify import identify,save
    from experiments.elastic.strip_texture_audit import evaluate,isolated
    from experiments.elastic.strip_texture_report import summarize,views,visuals,write_report,provenance
    import json
    a.out.mkdir(parents=True,exist_ok=False);p=protocol();save(a.out/'protocol.json',p)
    for k in ['A','B']:
        record(a.out,f'truth_{k}',p['E_pa'][k],p['grid'],a.device,states=True)
        render(a.out,k)
        track(a.out/f'inputs_{k}',a.out/f'flow_{k}')
        refine(a.out/f'inputs_{k}',a.out/f'flow_{k}',a.out/f'fit_{k}')
    select_reconstruction(a.out)
    for k in ['A','B']:
        identify(a.out/f'inputs_{k}',a.out/f'fit_{k}')
        path=a.out/f'fit_{k}/identification.json';fit=json.loads(path.read_text())
        fit['inputs']=['textured stereo pixels','camera calibration','timestamps','pusher force','known initial geometry, density, Poisson ratio and elastic family']
        save(path,fit);E=fit['E_pa']
        record(a.out,f'prediction_{k}',E,p['grid'],a.device,states=False)
        record(a.out,f'fine_{k}',p['E_pa'][k],p['fine_grid'],a.device,states=False)
        record(a.out,f'fine_prediction_{k}',E,p['fine_grid'],a.device,states=False)
    evaluate(a.out);isolated(a.out);summarize(a.out)
    for k in ['A','B']:views(a.out,k)
    visuals(a.out);write_report(a.out);provenance(a.out)


if __name__=='__main__':main()
