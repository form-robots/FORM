"""Minimum-dissipation reconstruction with the observed profile volume ratio.

This is a kinematic sensitivity, not measured material-point correspondence.
It relaxes the layer-only interior map with divergence-free streamfunctions,
while retaining the explicit assumption of spatially uniform dilatation.
"""
from pathlib import Path
import argparse
import shutil
import numpy as np
from experiments.robotics import press_hardware_stokes as stokes
from experiments.robotics.press_hardware_refine import poly
from experiments.robotics.press_hardware_moving_tests import fields
from experiments.robotics.press_hardware_observe import read, save


def velocity(r, z, h, hd, c, cd):
    volume = np.pi*h*np.mean(c)
    v, L, V, G = stokes.velocity(r, z, h, hd, c, cd, volume)
    dilation = hd/h + np.mean(cd)/np.mean(c)
    v[:, 0] += r*dilation/2
    L[:, 0, 0] += dilation/2
    L[:, 1, 1] += dilation/2
    return v, L, V, G


def coefficients(h, hd, c, cd):
    zz, wz = np.polynomial.legendre.leggauss(24)
    rr, wr = np.polynomial.legendre.leggauss(5)
    s, q = np.meshgrid((zz+1)/2, (rr+1)/2, indexing='ij')
    weights = ((wz[:, None]/2)*poly(c)(s)/np.mean(c)*(wr[None, :]*q)).ravel()
    r = q.ravel()*np.sqrt(poly(c)(s.ravel()))
    _, L, _, G = velocity(r, h*s.ravel(), h, hd, c, cd)
    D = (L+L.swapaxes(1, 2))/2
    DG = (G+G.swapaxes(1, 2))/2
    A = (DG*np.sqrt(weights)[:, None, None, None]).reshape(-1, 12)
    b = -(D*np.sqrt(weights)[:, None, None]).ravel()
    lhs = A.T@A
    co = np.linalg.solve(lhs+1e-8*np.trace(lhs)/12*np.eye(12), A.T@b)
    return co, float(np.linalg.norm(A@co-b)/max(np.linalg.norm(b), 1e-12))


def flow(q, s, h, hd, c, cd, co):
    a = poly(c); ad = poly(cd); A = a.antiderivative(); Ad = ad.antiderivative()
    radius = np.sqrt(a(s))
    v, L, V, G = velocity(q*radius, h*s, h, hd, c, cd)
    dv = V@co
    v += dv; L += G@co
    cumulative = (A(s)-A(0))/np.mean(c)
    sd = -(Ad(s)-Ad(0)-cumulative*np.mean(cd))/a(s)+dv[:, 2]/h
    qd = (dv[:, 0]-q*radius*a.derivative()(s)/(2*a(s))*dv[:, 2]/h)/radius
    return qd, sd, v, L, np.c_[q*radius, np.zeros(len(q)), h*s]


def mass_flow(q, m, h, hd, c, cd, co):
    """Cumulative-volume coordinates remove the initially spherical cap pole."""
    a = poly(c); A = a.antiderivative()
    lo = np.zeros_like(m); hi = np.ones_like(m)
    for _ in range(36):
        s = (lo+hi)/2
        lower = (A(s)-A(0))/np.mean(c) < m
        lo = np.where(lower, s, lo); hi = np.where(lower, hi, s)
    s = (lo+hi)/2; radius = np.sqrt(a(s))
    v, L, V, G = velocity(q*radius, h*s, h, hd, c, cd)
    dv = V@co; v += dv; L += G@co
    md = a(s)/np.mean(c)*dv[:, 2]/h
    qd = (dv[:, 0]-q*radius*a.derivative()(s)/(2*a(s))*dv[:, 2]/h)/radius
    return qd, md, v, L, np.c_[q*radius, np.zeros(len(q)), h*s]


def check():
    c = np.array([.06, .7, 1.1, .8, .04])*.02**2
    cd = np.array([.1, -.03, .02, -.02, .07])*.02**2
    h = .04; hd = -.002; r = np.array([.003, .008, .01]); z = np.array([.006, .017, .032]); eps = 1e-7
    v, L, _, _ = velocity(r, z, h, hd, c, cd)
    errors = []
    for dim in [0, 2]:
        vp = velocity(r+(eps if dim == 0 else 0), z+(eps if dim == 2 else 0), h, hd, c, cd)[0]
        vm = velocity(r-(eps if dim == 0 else 0), z-(eps if dim == 2 else 0), h, hd, c, cd)[0]
        errors.append(float(abs((vp-vm)/(2*eps)-L[:, :, dim]).max()))
    div_error = float(abs(np.trace(L, axis1=1, axis2=2)-(hd/h+np.mean(cd)/np.mean(c))).max())
    # The geometric boundary must be advected by the reconstructed velocity.
    s = np.linspace(.01, .99, 30); a = poly(c); rr = np.sqrt(a(s))
    vv = velocity(rr, h*s, h, hd, c, cd)[0]
    boundary_error = float(abs(vv[:, 0]-(poly(cd)(s)+a.derivative()(s)*(vv[:, 2]-hd*s)/h)/(2*rr)).max())
    assert max(errors) < 1e-6 and div_error < 1e-12 and boundary_error < 1e-12
    return dict(gradient_errors=errors, divergence_error=div_error, boundary_advection_error=boundary_error)


def reconstruct(source, out, ep, substeps=20):
    d = dict(np.load(source/ep/'motion.npz')); geo = read(source/ep/'geometry.json'); p = read(source/'protocol.json')
    t = d['time']; h = d['height']; cs = d['profile_coefficients']
    hd = np.gradient(h, t, edge_order=2); cd = np.gradient(cs, t, axis=0, edge_order=2)
    co, ratios = zip(*(coefficients(hh, dh, c, dc) for hh, dh, c, dc in zip(h, hd, cs, cd)))
    co = np.array(co); c0 = np.array(geo['profile_coefficients0']); volume = geo['volume_m3']
    zz, wz = np.polynomial.legendre.leggauss(24); rr, wr = np.polynomial.legendre.leggauss(4)
    s, q = np.meshgrid((zz+1)/2, (rr+1)/2, indexing='ij')
    weights = ((wz[:, None]/2)*poly(c0)(s)/np.mean(c0)*(wr[None, :]*q)).ravel()
    s = s.ravel(); q = q.ravel(); A0 = poly(c0).antiderivative()
    m = (A0(s)-A0(0))/np.mean(c0); F = np.tile(np.eye(3), (len(q), 1, 1))
    Xs = []; Fs = []; Vs = []; max_drift = 0.
    def at(i, alpha):
        interval = t[i+1]-t[i]
        return [(1-alpha)*h[i]+alpha*h[i+1], (h[i+1]-h[i])/interval,
                (1-alpha)*cs[i]+alpha*cs[i+1], (cs[i+1]-cs[i])/interval,
                (1-alpha)*co[i]+alpha*co[i+1]]
    for i in range(len(t)):
        _, _, v, _, x = mass_flow(q, m, h[i], hd[i], cs[i], cd[i], co[i])
        Xs.append(x); Fs.append(F.copy()); Vs.append(v)
        if i == len(t)-1:
            break
        dt = (t[i+1]-t[i])/substeps
        for j in range(substeps):
            hh, dh, c, dc, cc = at(i, j/substeps)
            qd, md, _, _, _ = mass_flow(q, m, hh, dh, c, dc, cc)
            qm = q+dt/2*qd; mm = m+dt/2*md
            hh, dh, c, dc, cc = at(i, (j+.5)/substeps)
            qd, md, _, L, _ = mass_flow(qm, mm, hh, dh, c, dc, cc)
            q += dt*qd; m += dt*md
            if not (q.min() > 0 and q.max() < 1.001 and m.min() > 0 and m.max() < 1):
                raise RuntimeError((ep, i, j, 'domain exit', q.min(), q.max(), m.min(), m.max()))
            inc = np.linalg.solve(np.eye(3)-dt*L/2, np.eye(3)+dt*L/2)
            F = inc@F
        expected = h[i+1]*np.mean(cs[i+1])/(h[0]*np.mean(cs[0]))
        J = np.linalg.det(F); max_drift = max(max_drift, float(abs(J/expected-1).max()))
        # Exact uniform-volume-ratio constraint; quantify numerical correction.
        F *= np.cbrt(expected/J)[:, None, None]
    x = np.array(Xs); F = np.array(Fs); v = np.array(Vs)
    w, g, wt = fields(x, h, hd)
    axw = w.copy(); axw[..., :2] = 0
    axg = g.copy(); axg[..., :2, :] = 0
    axwt = wt.copy(); axwt[..., :2] = 0
    w = np.concatenate([w, axw], axis=1); g = np.concatenate([g, axg], axis=1); wt = np.concatenate([wt, axwt], axis=1)
    mass = volume*p['density_kg_m3']
    M = mass*np.einsum('tni,tmni,n->tm', v, w, weights)
    C = mass*(np.einsum('tni,tmnij,tnj,n->tm', v, g, v, weights)+np.einsum('tni,tmni,n->tm', v, wt, weights))
    body = -9.81*mass*np.einsum('tmn,n->tm', w[..., 2], weights)
    target = []
    for begin in d['window_begin']:
        u = np.clip((t-begin)/.6, 0, 1); chi = np.sin(np.pi*u)**4; dc = 4*np.pi/.6*np.sin(np.pi*u)**3*np.cos(np.pi*u)
        target.append(np.sum(chi[:, None]*(C+body-d['force'][:, None])+dc[:, None]*M, axis=0)*p['export_dt_s'])
    d.update(x=x, F=F, velocity=v, weights=weights, test_gradient=g, target=target)
    dest = out/ep; dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'motion.npz', **d)
    for name in ['geometry.json', 'load_only.npz', 'camera.json']:
        shutil.copy2(source/ep/name, dest/name)
    save(dest/'stokes_audit.json', dict(max_step_J_correction=max_drift, mean_dissipation_ratio=float(np.mean(ratios)), substeps=substeps,
        scope='Uniform dilatation with minimum-dissipation interior; no contact slip penalty; correspondence remains a reconstruction prior'))
    print(ep, 'variable Stokes complete', max_drift, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', nargs='+', default=[f'ep{i:04}' for i in range(12)])
    parser.add_argument('--substeps', type=int, default=20)
    args = parser.parse_args(); args.out.mkdir(exist_ok=True, parents=True)
    p = read(args.source/'protocol.json'); p['interior'] = 'Minimum-dissipation streamfunctions; spatially uniform volume ratio from reconstructed profiles; no observed material identities'
    save(args.out/'protocol.json', p); save(args.out/'analytic_checks.json', check())
    for ep in args.episodes:
        reconstruct(args.source, args.out, ep, args.substeps)
