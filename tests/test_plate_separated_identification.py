"""Physical checks for the restricted proportional-loading direct estimator."""
import numpy as np
import pytest
from experiments.robotics.plate_separated_identify import hencky_columns, solve_columns
from experiments.robotics.plate_observable_field import unit_stress


@pytest.mark.parametrize('yield_pa',[1000.,10000.])
def test_columns_match_independent_return_map_in_their_regimes(yield_pa):
    E,nu=80000.,.3
    strain=np.linspace(0,.35,201)
    F=np.array([np.diag(np.exp(np.array([.35,.35,-1.])*s)) for s in strain])[:,None]
    reference=E*unit_stress(F,yield_pa/E,nu,True)
    reference-=np.eye(3)*np.trace(reference,axis1=-2,axis2=-1)[...,None,None]/3
    elastic,saturated=hencky_columns(F,nu)
    norm=np.linalg.norm(elastic,axis=(-2,-1))[:,0]*E
    initial=norm<.7*yield_pa;late=norm>1.2*yield_pa
    np.testing.assert_allclose(E*elastic[initial],reference[initial],atol=2e-8)
    np.testing.assert_allclose(yield_pa*saturated[late],reference[late],atol=2e-8)
    angle=.6;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    rotated_E,rotated_Y=hencky_columns(R@F@R.T,nu)
    np.testing.assert_allclose(rotated_E,R@elastic@R.T,atol=1e-13)
    np.testing.assert_allclose(rotated_Y[1:],R@saturated[1:]@R.T,atol=1e-12)


def test_mixed_transition_rows_do_not_enter_direct_fits():
    centers=np.repeat([.4,.5,.9,1.6,1.7],3)
    elastic=np.linspace(-.0002,-.0001,15);plastic=np.linspace(-.001,-.002,15)
    rhs=np.where(centers<.6,80000*elastic,1000*plastic)
    rhs[(centers>.6)&(centers<1.5)]=1e8  # deliberately incompatible transition
    result,_=solve_columns(elastic,plastic,rhs,centers,dict(elastic_centers=[.4,.6],plastic_centers=[1.6,1.78]))
    assert result['E_pa']==pytest.approx(80000.)
    assert result['yield_pa']==pytest.approx(1000.)
