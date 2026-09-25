"""Independent geometry, aperture crossing and surface-rendering checks."""
import json
import numpy as np
from experiments.elastic.rod_insertion_study import particles,wall_violation,rotation,translation
from experiments.elastic.rod_insertion_report import audit_case,surface


def test_round_bore_rejects_square_corner():
    p=dict(wall_x=.2,wall_thickness=.006,center_y=.15,opening_z=.15,opening_radius=.006)
    assert wall_violation(np.array([[.2,.15,.15]]),p,0)==0
    assert wall_violation(np.array([[.2,.155,.155]]),p,0)>.001
    assert wall_violation(np.array([[.18,.155,.155]]),p,0)==0
    assert wall_violation(np.array([[.2,.155,.15]]),p,.002)>.0009


def test_particle_volume_and_render_preserve_rigid_geometry():
    p=dict(size=[.06,.008,.008],radius=.004,grip_length=.015,domain=.32,grip_x=.12,center_y=.16)
    x,vol,ref,_=particles(p,320,[.15,25])
    assert np.all(np.linalg.norm(ref[:,1:],axis=1)<.004)
    np.testing.assert_allclose(vol.sum(),np.pi*.004**2*.06,rtol=1e-6)
    mesh,idx,w=surface(ref,p)
    moved=np.einsum('nk,nkj->nj',w,x[idx])
    expected=mesh.points@rotation(25).T+[.12,.16,.15]
    np.testing.assert_allclose(moved,expected,atol=2e-8)


def test_crossing_outside_round_bore_cannot_pass(tmp_path):
    p=dict(wall_x=.2,wall_thickness=.006,center_y=.15,opening_z=.15,opening_radius=.006)
    np.save(tmp_path/'tip_mask.npy',np.array([True,True]));np.save(tmp_path/'time.npy',np.array([0.,1.]))
    (tmp_path/'result.json').write_text(json.dumps(dict(success=True)))
    x=np.array([[[.19,.15,.15],[.19,.155,.155]],[[.22,.15,.15],[.22,.155,.155]]])
    np.save(tmp_path/'x.npy',x)
    assert not audit_case(tmp_path,p)['success']
    x[:,:,1:]=.15;np.save(tmp_path/'x.npy',x)
    assert audit_case(tmp_path,p)['success']
