"""Position-controlled bending with measured reaction force; unfitted recovery.

Two fixed-corotated materials share the full pusher motion. A bonded upper grip
holds a vertical strip. The pusher is a frictionless, separable cylindrical SDF.
Only the loading phase identifies E; the complete identified-model replay starts
from the initial specimen and predicts the subsequent hold, withdrawal and release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.elastic.block_drop_study import ROOT, save, sha
from ident.weakform.grid_assembly import _bspline_weights_1d

PROTOCOL = dict(
    E_pa={"A": 80000., "B": 240000.}, known_nu=.45, density=1000.,
    material="fixed-corotated elasticity, no viscosity or plasticity",
    domain=.24, grid=96, fine_grid=128, dt=.00002, tick=.002,
    size=[.012, .020, .100], center=[.120, .120, .130],
    grip_center=[.120, .120, .1775], grip_half=[.025, .030, .0125],
    grip_lower_z=.165, pusher_radius=.010, pusher_length=.032,
    pusher_initial=[.140, .120, .090], pusher_travel=.024, park_offset=.050,
    sdf_cell=.0005, contact_band=.0005,
    settle_end=.30, bend_end=.90, hold_end=1.05, withdraw_end=1.17, end=2.0,
    fit_interval=[.32, .88], window_frames=26, window_stride=13,
    test_transitions=[[.112, .150], [.120, .152], [.128, .154]],
    force_sensor="net grid reaction impulse / tick; raw 3D vector",
    known=["density", "geometry", "Poisson ratio", "constitutive family", "grip and tool motion"],
    inputs=["particle x", "particle v", "particle F", "net pusher force", "reference volume"],
    excluded=["true E", "true stress", "grip force", "hold/withdrawal/recovery observations"],
    selection="Geometry and material pair fixed before pilot; retain development runs.",
)


def init(root):
    root.mkdir(parents=True, exist_ok=False)
    save(root/"protocol.json", PROTOCOL)
    sources = [Path(__file__), ROOT/"experiments/elastic/block_drop_study.py",
               ROOT/"src/ident/weakform/elastic_grid.py", ROOT/"src/ident/weakform/grid_assembly.py",
               ROOT/"src/warpmpm/core/solver.py", ROOT/"src/warpmpm/kernels/mpm_solver_warp.py",
               ROOT/"src/warpmpm/kernels/mpm_utils.py", ROOT/"src/warpmpm/kernels/warp_utils.py"]
    for path in sources:
        target=root/"source"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    save(root/"source_hashes.json", {str(f.relative_to(ROOT)):sha(f) for f in sources})


def pusher_position(t, p):
    pose=np.array(p["pusher_initial"], dtype=float)
    if t <= p["settle_end"]:
        return pose
    if t <= p["bend_end"]:
        u=(t-p["settle_end"])/(p["bend_end"]-p["settle_end"])
        pose[0]-=p["pusher_travel"]*(3*u*u-2*u*u*u)
    elif t <= p["hold_end"]:
        pose[0]-=p["pusher_travel"]
    elif t <= p["withdraw_end"]:
        u=(t-p["hold_end"])/(p["withdraw_end"]-p["hold_end"])
        pose[0]+=-p["pusher_travel"]+(p["pusher_travel"]+p["park_offset"])*(3*u*u-2*u*u*u)
    else:
        pose[0]+=p["park_offset"]
    return pose


def cylinder(p):
    from warpmpm.geometry import SDFData
    cell=p["sdf_cell"]
    extent=max(p["pusher_radius"],p["pusher_length"]/2)+.005
    extent=np.ceil(extent/cell)*cell
    axis=np.linspace(-extent,extent,round(2*extent/cell)+1)
    xyz=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),axis=-1)
    d=np.stack([np.linalg.norm(xyz[...,[0,2]],axis=-1)-p["pusher_radius"],
                abs(xyz[...,1])-p["pusher_length"]/2],axis=-1)
    values=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)
    grads=np.stack(np.gradient(values,cell,edge_order=2),axis=-1)
    return SDFData(values.astype(np.float32),grads.astype(np.float32),np.full(3,-extent),
                   cell,float(values.max()))


def specimen(p, grid):
    size=np.asarray(p["size"])
    n=np.rint(size/(p["domain"]/grid/2)).astype(int)
    axes=[(np.arange(k)+.5)*s/k-s/2+c for k,s,c in zip(n,size,p["center"],strict=True)]
    x=np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3).astype(np.float32)
    return x,np.full(len(x),size.prod()/len(x),dtype=np.float32)


def record(root, name, E, grid, device, states=True):
    from warpmpm.core.solver import GridConfig, Solver
    from warpmpm.materials import elastic
    p=json.loads((root/"protocol.json").read_text())
    case=root/name
    case.mkdir(exist_ok=False)
    x,vol=specimen(p,grid)
    save(case/"config.json",dict(E_pa=E,grid=grid,device=device,particles=len(x),
                                 protocol_sha256=sha(root/"protocol.json")))
    sim=Solver(GridConfig(grid,p["domain"]),device=device,inversion_policy="raise").load_particles(x,vol)
    sim.set_material(elastic(E=E,nu=p["known_nu"],density=p["density"]),
                     rpic_damping=0.,grid_v_damping_scale=1.)
    grip=sim.add_box(p["grip_center"],p["grip_half"])
    tool=sim.add_sdf_collider(cylinder(p),center=p["pusher_initial"],band=p["contact_band"],
                              surface="separable",friction=0.)
    times=np.arange(round(p["end"]/p["tick"])+1)*p["tick"]
    X=np.lib.format.open_memmap(case/"x.npy",mode="w+",dtype="float32",shape=(len(times),len(x),3))
    V=np.lib.format.open_memmap(case/"v.npy",mode="w+",dtype="float32",shape=X.shape)
    if states:
        F=np.lib.format.open_memmap(case/"F.npy",mode="w+",dtype="float32",shape=(*X.shape[:2],3,3))
    tip=x[:,2] < x[:,2].min()+.004
    np.save(case/"tip_mask.npy",tip)
    np.save(case/"vol0.npy",vol)
    np.save(case/"time.npy",times)
    data=[]
    start=time.monotonic()
    min_j=np.inf
    for i,t in enumerate(times):
        if i:
            begin=pusher_position(times[i-1],p)
            end=pusher_position(t,p)
            sim.set_sdf_pose(tool,center=begin,velocity=(end-begin)/p["tick"])
            sim.reset_sdf_force(tool)
            sim.reset_tool_force(grip)
            sim.step(p["dt"],substeps=round(p["tick"]/p["dt"]))
            force=sim.sdf_wrench(tool,p["tick"])["force"]
            grip_force=sim.tool_force(grip,p["tick"])
        else:
            force=grip_force=np.zeros(3)
        xf,vf,ff=sim.x(),sim.v(),sim.F()
        if not all(np.isfinite(a).all() for a in [xf,vf,ff,force,grip_force]):
            raise RuntimeError(f"Nonfinite state {name} at {t}")
        J=np.linalg.det(ff)
        min_j=min(min_j,float(J.min()))
        if J.min()<=0:
            raise RuntimeError(f"Inversion {name} at {t}")
        X[i],V[i]=xf,vf
        if states:
            F[i]=ff
        momentum=np.sum(vf.astype(float)*(vol*p["density"])[:,None],axis=0)
        data.append([t,*pusher_position(t,p),*force,*grip_force,*xf[tip].mean(0),*momentum])
        if i%100==0:
            print(f"{name}: t={t:.3f}, tool Fx={force[0]:.4f} N, tip x={xf[tip,0].mean():.5f}",flush=True)
    X.flush(); V.flush()
    if states:
        F.flush()
    data=np.asarray(data)
    np.savetxt(case/"signals.csv",data,delimiter=",",comments="",header=
        "time,tool_x,tool_y,tool_z,reaction_x,reaction_y,reaction_z,grip_x,grip_y,grip_z,tip_x,tip_y,tip_z,momentum_x,momentum_y,momentum_z")
    mass=float(vol.sum(dtype=float)*p["density"])
    expected=np.cumsum((-data[1:,4:7]-data[1:,7:10]+np.array([0,0,-mass*9.81]))*p["tick"],axis=0)
    momentum_error=data[1:,13:16]-data[0,13:16]-expected
    result=dict(wall_s=time.monotonic()-start,frames=len(times),mass_kg=mass,min_J=min_j,
                inverted_count=sim.inverted_count(),
                cumulative_momentum_max_error_kg_m_s=float(np.linalg.norm(momentum_error,axis=1).max()),
                cumulative_momentum_scale_kg_m_s=float(np.linalg.norm(data[:,13:16],axis=1).max()),
                all_frames_completed=True)
    result["hashes"]={f.name:sha(f) for f in case.glob("*.npy")}
    result["hashes"]["signals.csv"]=sha(case/"signals.csv")
    save(case/"completion.json",result)
    print(json.dumps(result),flush=True)


def virtual_fields(x, grid, p):
    """Nodal bending test interpolated with the same quadratic MPM B-splines.

    w_x = interpolated psi(z); w_z = -(x-x0) interpolated psi'(z).
    It is e_x throughout tool contact and zero throughout the fixed grip.
    This makes the boundary term the measured net material-side pusher Fx.
    """
    dx=p["domain"]/grid
    z=x[:,2]/dx
    base=np.floor(z-.5).astype(int)
    weights,dweights=_bspline_weights_1d(z-base)
    nodes=(base[:,None]+np.arange(3))*dx
    fields=[]
    offset=x[:,0]-p["center"][0]
    for lo,hi in p["test_transitions"]:
        u=np.clip((nodes-lo)/(hi-lo),0,1)
        s=1-10*u**3+15*u**4-6*u**5
        q=(-30*u**2+60*u**3-30*u**4)/(hi-lo)
        si=(weights*s).sum(axis=1)
        qi=(weights*q).sum(axis=1)
        ds=(dweights*s).sum(axis=1)/dx
        dq=(dweights*q).sum(axis=1)/dx
        w=np.zeros_like(x,dtype=float)
        grad=np.zeros((len(x),3,3))
        w[:,0]=si; w[:,2]=-offset*qi
        grad[:,0,2]=ds; grad[:,2,0]=-qi; grad[:,2,2]=-offset*dq
        fields.append((w,grad))
    return fields


def identify(root, name):
    from ident.weakform.elastic_grid import corotated_cauchy_columns, _temporal_window
    p=json.loads((root/"protocol.json").read_text())
    case=root/name
    grid=json.loads((case/"config.json").read_text())["grid"]
    times=np.load(case/"time.npy")
    frames=np.flatnonzero((times>=p["fit_interval"][0])&(times<=p["fit_interval"][1]))
    X,V,F=(np.load(case/f"{k}.npy",mmap_mode="r") for k in ["x","v","F"])
    vol=np.load(case/"vol0.npy").astype(float); mass=vol*p["density"]
    signals=np.genfromtxt(case/"signals.csv",delimiter=",",names=True)
    nu=p["known_nu"]
    a=[]; load=[]; mom=[]; contact=[]
    start=time.monotonic()
    for i in frames:
        x,v,f=X[i].astype(float),V[i].astype(float),F[i].astype(float)
        sm,sl=corotated_cauchy_columns(f)
        tau=(np.linalg.det(f)*vol)[:,None,None]*(sm/(2*(1+nu))+sl*nu/((1+nu)*(1-2*nu)))
        # Stored force is averaged over the preceding interval. Center it on
        # observation time by averaging the neighboring intervals (both loading).
        force=-.5*(signals["reaction_x"][i]+signals["reaction_x"][i+1])
        ar=[]; br=[]; mr=[]
        for w,grad in virtual_fields(x,grid,p):
            ar.append(np.einsum("pij,pij->",tau,grad))
            br.append(float(np.sum(mass*(-9.81*w[:,2]+np.einsum("pi,pij,pj->p",v,grad,v)))))
            mr.append(float(np.sum(mass*np.sum(v*w,axis=1))))
        a.append(ar); load.append(br); mom.append(mr); contact.append(force)
    a,load,mom,contact=map(np.asarray,(a,load,mom,contact))
    A=[]; b=[]; bf=[]; centers=[]
    nw=p["window_frames"]
    chi,dchi=_temporal_window(nw,p["tick"],2)
    for j in range(0,len(frames)-nw+1,p["window_stride"]):
        slc=slice(j,j+nw)
        A.extend(np.sum(chi[:,None]*a[slc],axis=0)*p["tick"])
        inertia=np.sum(chi[:,None]*load[slc]+dchi[:,None]*mom[slc],axis=0)*p["tick"]
        boundary=float(chi@contact[slc]*p["tick"])
        b.extend(inertia+boundary)
        bf.extend([boundary]*3)
        centers.extend([float(times[frames[j+nw//2]])]*3)
    A,b,bf=np.asarray(A),np.asarray(b),np.asarray(bf)
    E=float(np.linalg.lstsq(A[:,None],b,rcond=None)[0][0])
    if E<=0 or not np.isfinite(E):
        raise RuntimeError("Nonphysical stiffness fit")
    estimates=[float(A[k::3]@b[k::3]/(A[k::3]@A[k::3])) for k in range(3)]
    result=dict(E_pa=E,known_nu=nu,rows=len(A),wall_s=time.monotonic()-start,
                residual_relative_l2=float(np.linalg.norm(A*E-b)/np.linalg.norm(b)),
                per_virtual_field_E_pa=estimates,
                force_omitted_diagnostic_E_pa=float(A@(b-bf)/(A@A)),
                inputs=p["inputs"],excluded=p["excluded"],fit_interval=p["fit_interval"],
                true_E_used=False,recovery_used=False)
    np.savez(case/"weak_system.npz",a_E=A,b=b,boundary=bf,time=centers)
    save(case/"identification.json",result)
    print(json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["init","truth","fit","prediction","fine"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--material",choices=["A","B"])
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.stage=="init":
        init(args.out); return
    p=json.loads((args.out/"protocol.json").read_text())
    label=args.material
    if label is None:
        parser.error("--material required")
    if args.stage=="fit":
        identify(args.out,f"truth_{label}"); return
    E=p["E_pa"][label]
    if args.stage=="prediction":
        E=json.loads((args.out/f"truth_{label}/identification.json").read_text())["E_pa"]
    record(args.out,f"{args.stage}_{label}",E,p["fine_grid"] if args.stage=="fine" else p["grid"],
           args.device,states=args.stage!="prediction")
    if args.stage in ["truth","fine"]:
        identify(args.out,f"{args.stage}_{label}")


if __name__=="__main__":
    main()
