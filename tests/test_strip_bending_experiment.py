"""Boundary traces and interpolation of the instrumented bending probe."""
import numpy as np

from experiments.elastic.strip_bending_study import PROTOCOL, virtual_fields
from ident.weakform.elastic_grid import bspline_stencil


def test_bending_weights_use_only_net_pusher_force_and_eliminate_grip():
    p=PROTOCOL
    for grid in [96,128]:
        z=np.r_[np.linspace(.079,.102,17),np.linspace(.165,.190,17)]
        x=np.column_stack([np.linspace(.08,.16,len(z)),np.full(len(z),.12),z])
        for w,grad in virtual_fields(x,grid,p):
            np.testing.assert_allclose(w[:17],np.tile([1.,0.,0.],(17,1)),atol=1e-12)
            np.testing.assert_allclose(grad[:17],0.,atol=1e-10)
            np.testing.assert_allclose(w[17:],0.,atol=1e-12)
            np.testing.assert_allclose(grad[17:],0.,atol=1e-10)


def test_bending_weight_gradient_matches_finite_difference():
    rng=np.random.default_rng(3)
    x=np.column_stack([rng.uniform(.09,.15,21),rng.uniform(.11,.13,21),
                       rng.uniform(.108,.158,21)])
    for dim in range(3):
        delta=np.zeros(3); delta[dim]=1e-7
        plus=virtual_fields(x+delta,96,PROTOCOL)
        minus=virtual_fields(x-delta,96,PROTOCOL)
        for (_,grad),(wp,_),(wm,_) in zip(virtual_fields(x,96,PROTOCOL),plus,minus,strict=True):
            np.testing.assert_allclose((wp-wm)/2e-7,grad[:,:,dim],rtol=1e-6,atol=1e-6)


def test_bending_weights_equal_full_mpm_nodal_interpolation():
    p=PROTOCOL; grid=96
    rng=np.random.default_rng(7)
    x=rng.uniform([.085,.11,.095],[.15,.13,.175],(17,3))
    ids,N,gradN,valid=bspline_stencil(x,grid,p['domain'])
    assert valid.all()
    nx=ids//(grid*grid)*p['domain']/grid
    nz=(ids%grid)*p['domain']/grid
    for (lo,hi),(w,grad) in zip(p['test_transitions'],virtual_fields(x,grid,p),strict=True):
        u=np.clip((nz-lo)/(hi-lo),0,1)
        s=1-10*u**3+15*u**4-6*u**5
        q=(-30*u**2+60*u**3-30*u**4)/(hi-lo)
        nodal=np.zeros((*ids.shape,3)); nodal[:,:,0]=s
        nodal[:,:,2]=-(nx-p['center'][0])*q
        np.testing.assert_allclose(np.einsum('ps,psi->pi',N,nodal),w,atol=1e-12)
        np.testing.assert_allclose(np.einsum('psj,psi->pij',gradN,nodal),grad,atol=1e-10)
