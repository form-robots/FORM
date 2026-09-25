"""Hardware audit of the manuscript's separated linear pressing estimator.

Uses the exact hencky_columns and solve_columns functions used for simulated
pressing. Elastic and saturated proportional intervals are explicit hypotheses;
their support must be checked rather than inferred from a positive least square.
No plastic-history replay, parameter-ratio search, or forward dynamics in fit.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from experiments.robotics.plate_separated_identify import hencky_columns, solve_columns
from experiments.robotics.press_hardware_moving_tests import fields


def assemble(root, episode, width=.2):
    d = dict(np.load(root/episode/'motion.npz'))
    geo = json.loads((root/episode/'geometry.json').read_text())
    t = d['time']; x=d['x']; v=d['velocity']; h=d['height']; weights=d['weights']; V0=geo['volume_m3']; mass=1000*V0
    w,g,wt=fields(x,h,np.gradient(h,t,edge_order=2))
    M=mass*np.einsum('tni,tmni,n->tm',v,w,weights)
    C=mass*(np.einsum('tni,tmnij,tnj,n->tm',v,g,v,weights)+np.einsum('tni,tmni,n->tm',v,wt,weights))
    body=-9.81*mass*np.einsum('tmn,n->tm',w[...,2],weights)
    E,Y=hencky_columns(d['F'],.45)
    L=np.gradient(d['F'],t,axis=0,edge_order=2)@np.linalg.inv(d['F'])
    D=(L+L.swapaxes(-1,-2))/2
    D-=np.eye(3)*np.trace(D,axis1=-2,axis2=-1)[...,None,None]/3
    instantaneous=[V0*np.einsum('tnij,tmnij,n->tm',stress,g,weights) for stress in [E,Y,D]]
    centers=np.arange(.25,10.31,.05);aE=[];aY=[];aD=[];bs=[];force=[]
    for center in centers:
        u=np.clip((t-center+width/2)/width,0,1);chi=np.sin(np.pi*u)**4
        derivative=4*np.pi/width*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        aE.append(np.trapezoid(chi[:,None]*instantaneous[0],t,axis=0))
        aY.append(np.trapezoid(chi[:,None]*instantaneous[1],t,axis=0))
        aD.append(np.trapezoid(chi[:,None]*instantaneous[2],t,axis=0))
        bs.append(np.trapezoid(chi[:,None]*(C+body-d['force'][:,None])+derivative[:,None]*M,t,axis=0))
        force.append(float(np.trapezoid(chi*d['force'],t)/np.trapezoid(chi,t)))
    d['rate_column']=np.array(aD).ravel()
    return dict(elastic=np.array(aE).ravel(),plastic=np.array(aY).ravel(),b=np.array(bs).ravel(),centers=np.repeat(centers,3)),d,np.array(force)


def fit(root,episode,out):
    start=time.perf_counter();system,d,force=assemble(root,episode)
    centers=system['centers'][::3]
    hold_force=float(np.median(d['force'][(d['time']>=3)&(d['time']<=10)]))
    # Common observation-based rule, never selected from forward agreement:
    # early ramp at 5–20% of held load; late hold at 6–10 seconds.
    early=(centers<1.5)&(force>=.05*hold_force)&(force<=.20*hold_force)
    out.mkdir(parents=True,exist_ok=False)
    diagnostic=dict(early_window_centers_s=centers[early].tolist(),early_window_forces_N=force[early].tolist(),hold_force_N=hold_force)
    if early.sum()<2:
        result=dict(status='Insufficient early windows for the predeclared elastic interval rule',diagnostic=diagnostic)
    else:
        config=dict(elastic_centers=[float(centers[early].min()),float(centers[early].max())],plastic_centers=[6.,10.])
        values,details=solve_columns(**system,config=config)
        E=values['E_pa'];Y=values['yield_pa'];unitE,_=hencky_columns(d['F'],.45)
        # Additional Method-section-compatible dictionary hypothesis: late
        # stress = Y*direction(H) + C*dev(D), assuming proportional saturated
        # flow and negligible elastic strain rate. C=2G*T for the solver law.
        # Both columns are fixed before solving; no return map enters this fit.
        from scipy.optimize import nnls
        late_system,late_d,_=assemble(root,episode,width=.6)
        use=(late_system['centers']>=6)&(late_system['centers']<=10)
        AA=np.c_[late_system['plastic'][use],late_d['rate_column'][use]]
        scales=np.linalg.norm(AA,axis=0);bb=late_system['b'][use]
        coeff,res=nnls(AA/np.maximum(scales,1e-30),bb);coeff/=np.maximum(scales,1e-30)
        flow=dict(Y_Pa=float(coeff[0]),C_Pa_s=float(coeff[1]),tau_s=float(coeff[1]/(2*E/(2*1.45))),
                  relative_weak_residual=float(res/np.linalg.norm(bb)),
                  normalized_column_condition_number=float(np.linalg.cond(AA/np.maximum(scales,1e-30))),
                  scope='Linear late-flow hypothesis with fixed H-direction and D columns; requires proportional saturated flow and negligible elastic strain rate. C=2G*T, not standard Newtonian viscosity.')
        trial_norm=E*np.linalg.norm(unitE,axis=(-2,-1))
        early_t=(d['time']>=config['elastic_centers'][0]-.1)&(d['time']<=config['elastic_centers'][1]+.1)
        late_t=(d['time']>=6)&(d['time']<=10)
        fractions=(trial_norm>Y)@d['weights']
        diagnostic.update(elastic_interval_trial_above_Y_reference_fraction_max=float(fractions[early_t].max()),
                          late_interval_trial_below_Y_reference_fraction_mean=float(np.mean(1-fractions[late_t])),
                          consistency_scope='Necessary elastic/saturation check using inferred E,Y and total Hencky strain. Passing would not prove actual plastic state or proportionality.')
        # Test coaxial/proportional total strain direction in the late interval.
        _,direction=hencky_columns(d['F'],.45);ids=np.flatnonzero(late_t);ref=direction[ids[0]]
        cos=np.einsum('tnij,nij->tn',direction[ids],ref)
        diagnostic['late_direction_cosine_reference_volume_mean']=float(np.mean(cos@d['weights']))
        result=dict(status='Conditional linear estimates; interval hypotheses require assessment',values=values,diagnostics=details,
                    linear_rate_hypothesis=flow,
                    interval_audit=diagnostic,config=config,nu_assumed=.45,
                    estimator='Exact manuscript simulation hencky_columns and solve_columns, applied to reconstructed hardware motion.',
                    forward_simulation_calls=0,constitutive_history_replays=0,ratio_search_evaluations=0,
                    limitations=['Early elastic and late fully yielded proportional intervals are hypotheses, not measured plastic states.',
                                 'Axisymmetric baseline-referenced minimum-dissipation interior is reconstructed, not tracked material points.',
                                 'Fixed nu=0.45, density=1000 kg/m3, nominal stress-free initial state; friction/adhesion not measured.'])
    result['wall_s']=time.perf_counter()-start
    (out/'identification.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'weak_system.npz',**system,force=force)
    print(episode,json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--episode',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();fit(a.root,a.episode,a.out)
