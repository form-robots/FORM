"""Use frozen material fits with asymmetric initial silhouettes in 3D MPM."""
import argparse
from pathlib import Path
import numpy as np
from experiments.robotics import press_hardware_forward as forward
from experiments.robotics.press_hardware_refine import poly
from experiments.robotics.press_hardware_observe import read,save

ORIGINAL_PARTICLES=forward.initial_particles


def particles(geometry,spacing=.0015,floor=.05):
    if 'profile_coefficients0' not in geometry:
        return ORIGINAL_PARTICLES(geometry,spacing,floor)
    c=np.array(geometry['profile_coefficients0']);h=geometry['h0_m']
    max_r=np.sqrt(poly(c)(np.linspace(0,1,1001)).max())
    axis=np.arange(-max_r,max_r+spacing/2,spacing)
    nz=max(3,round(h/spacing));z=(np.arange(nz)+.5)*h/nz
    x=np.stack(np.meshgrid(axis,axis,z,indexing='ij'),-1).reshape(-1,3)
    r=np.sqrt(poly(c)(x[:,2]/h));x=x[np.linalg.norm(x[:,:2],axis=1)<r]
    x+=np.array([.09,.09,floor])
    return x.astype('float32'),np.full(len(x),geometry['volume_m3']/len(x),np.float32)


def run(out,material,episode=None,**kwargs):
    forward.initial_particles=particles
    if episode is not None:
        original=out.resolve();case=original/'cases'/episode
        case.mkdir(exist_ok=True,parents=True)
        p=read(original/'protocol.json');p['heldout'][material]=episode
        save(case/'protocol.json',p)
        for name in [episode,material]:
            dest=case/name
            if name==material:
                dest.mkdir(exist_ok=True)
                if not (dest/'fit.json').exists():(dest/'fit.json').symlink_to(original/name/'fit.json')
            elif not dest.exists():dest.symlink_to(original/name,target_is_directory=True)
        out=case
    forward.run(out,material,**kwargs)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--out',type=Path,required=True);a.add_argument('--material',required=True)
    a.add_argument('--episode');a.add_argument('--device',default='cuda:0');a.add_argument('--grid',type=int,default=80)
    a.add_argument('--friction',type=float,default=.15);a.add_argument('--bulk',type=float)
    a.add_argument('--tag',default='validation');a.add_argument('--duration',type=float,default=13.4)
    a.add_argument('--summary-only',action='store_true');args=a.parse_args();run(**vars(args))
