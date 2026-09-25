"""Checks for task geometry, prescribed motion, and aperture crossing evaluation."""
import json
import numpy as np
from experiments.elastic.strip_insertion_study import rotation,translation,tilt,wall_violation
from experiments.elastic.strip_insertion_report import audit_case


def test_motion_has_hanging_start_and_fixed_final_pose():
    p=dict(align_start=.3,align_end=3.3,settle_end=3.8,advance_end=6.8,advance=.054)
    np.testing.assert_allclose(rotation(tilt(0,20,p))@np.array([1.,0,0]),[0,0,-1],atol=1.e-12)
    assert tilt(3.3,20,p)==20 and tilt(7.3,20,p)==20
    assert translation(3.8,p)==0 and translation(6.8,p)==.054
    t=np.linspace(3.8,6.8,10001);x=np.array([translation(tt,p) for tt in t])
    assert np.max(np.diff(x)/np.diff(t))<=.02700001
    np.testing.assert_allclose(rotation(40).T@rotation(40),np.eye(3),atol=1.e-12)


def test_aperture_penalty_covers_whole_material_and_has_margin():
    p=dict(wall_x=.2,wall_thickness=.006,center_y=.15,opening_z=.15,opening_width=.02,opening_height=.04)
    assert wall_violation(np.array([[.2,.15,.15]]),p,.002)==0
    assert wall_violation(np.array([[.2,.15,.169]]),p,.002)>.0009
    # A correct tip cannot conceal another part of the strip intersecting the wall.
    assert wall_violation(np.array([[.2,.15,.15],[.2,.15,.19]]),p,.002)>.021
    assert wall_violation(np.array([[.18,.15,.19]]),p,.002)==0


def test_depth_alone_does_not_count_bypassing_aperture_as_success(tmp_path):
    p=dict(wall_x=.2,wall_thickness=.006,center_y=.15,opening_z=.15,opening_width=.02,opening_height=.04)
    np.save(tmp_path/'tip_mask.npy',np.array([True,True]))
    np.save(tmp_path/'time.npy',np.array([0.,1.]))
    (tmp_path/'result.json').write_text(json.dumps(dict(success=True)))
    x=np.array([[[.19,.15,.15],[.19,.15,.19]],[[.21,.15,.15],[.21,.15,.19]]])
    np.save(tmp_path/'x.npy',x)
    assert not audit_case(tmp_path,p)['success']
    x[:,:,2]=.15;np.save(tmp_path/'x.npy',x)
    assert audit_case(tmp_path,p)['success']
