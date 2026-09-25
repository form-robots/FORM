"""Analytical checks of the ball dynamics used by the putting experiment."""
import numpy as np
from experiments.elastic.putting import Ball,pose,particles

def test_club_volume_is_independent_of_grid_resolution():
    p=dict(domain=.24,head_size=[.012,.028,.016],head_center=[0.,0.,-.0675],
           shaft_width=.012,shaft_depth=.008,shaft_top=.0075)
    exact=.012*.028*.016+.012*.008*.067
    for grid in [160,200,240,320]:
        q,v=particles(p,grid)
        assert np.isfinite(q).all() and (v>0).all()
        np.testing.assert_allclose(v.sum(dtype=np.float64),exact,rtol=1e-7)

def ball_parameters():
    return dict(ball_initial=[.118,.11,.028],ball_mass=.002,ball_radius=.008,floor=.02,
                gravity=9.81,rolling_resistance=.025,floor_restitution=0.,
                floor_static_friction=.6,floor_sliding_friction=.45)

def test_rolling_deceleration_and_stopping_distance():
    p=ball_parameters();p['ball_initial']=[0.,0.,p['floor']+p['ball_radius']]
    b=Ball(p);b.v[0]=.3;b.omega[1]=.3/p['ball_radius'];energies=[]
    a=p['rolling_resistance']*p['gravity']/1.4
    for _ in range(25000):
        b.step(np.zeros(3),np.zeros(3),.0001);energies.append(b.energy())
    assert abs(b.x[0]-.3**2/(2*a))<.0001
    assert np.linalg.norm(b.v)<1e-8
    assert np.max(np.diff(energies))<1e-15

def test_sliding_sphere_reaches_rolling_speed_from_momentum():
    p=ball_parameters();p['rolling_resistance']=0.
    b=Ball(p);b.v[0]=.3
    for _ in range(3000):b.step(np.zeros(3),np.zeros(3),.0001)
    np.testing.assert_allclose(b.v[0],.3*5/7,atol=1e-8)
    np.testing.assert_allclose(b.v[0],p['ball_radius']*b.omega[1],atol=1e-8)

def test_rendered_orientation_follows_actual_rolling_motion():
    from scipy.spatial.transform import Rotation
    p=ball_parameters();b=Ball(p);b.v[0]=.3;b.omega[1]=.3/p['ball_radius'];start=b.x[0]
    for _ in range(1000):b.step(np.zeros(3),np.zeros(3),.0001)
    expected=Rotation.from_rotvec([0,(b.x[0]-start)/p['ball_radius'],0]).as_matrix()
    np.testing.assert_allclose(Rotation.from_quat(b.quat).as_matrix(),expected,atol=1e-10)

def test_swing_has_backswing_contact_arc_and_withdrawal():
    from experiments.elastic.putting import protocol
    p=protocol();u=[.55,12.]
    start=p['settle']+p['backswing_duration']+p['backswing_pause']
    c0,R0=pose(0.,u,p);cb,Rb=pose(start,u,p)
    cf,Rf=pose(start+u[0],u,p);cl,Rl=pose(3.,u,p)
    head=np.array(p['head_center'])
    assert (cb+Rb@head)[0] < (c0+R0@head)[0]-.06
    assert (cf+Rf@head)[0] > (c0+R0@head)[0]+.05
    np.testing.assert_allclose(cl-cf,[0.,0.,p['lift_height']],atol=1e-12)
    np.testing.assert_allclose(Rl,Rf,atol=1e-12)
    # The undeformed head corners clear the floor throughout the full arc.
    import itertools
    corners=np.array(list(itertools.product([-1,1],repeat=3)))*np.array(p['head_size'])/2+head
    for t in np.linspace(0,3,601):
        c,R=pose(t,u,p)
        assert (corners@R.T+c)[:,2].min()>p['floor']
