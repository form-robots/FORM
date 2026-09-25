"""Observation geometry and mechanics checks for the camera-only identifier."""
import cv2
import numpy as np

from experiments.elastic.strip_camera_identify import associate,dots,projected,spline_basis
from experiments.elastic.strip_camera_observe import CAMERA,calibration
from experiments.elastic.strip_camera_reconstruction import virtual_fields
from ident.weakform.elastic_grid import corotated_cauchy_columns


def test_stereo_camera_projection_round_trip():
    cams=calibration(CAMERA)
    q=np.array([[.115,.11,.082],[.124,.108,.145],[.101,.112,.095]])
    uv=[projected(q,c['P']).T for c in cams]
    homogeneous=cv2.triangulatePoints(np.asarray(cams[0]['P']),np.asarray(cams[1]['P']),*uv)
    np.testing.assert_allclose((homogeneous[:3]/homogeneous[3]).T,q,atol=1e-12)
    assert np.max(abs(uv[0]-uv[1]))>10


def test_image_markers_are_measured_instead_of_returning_queries():
    im=np.full((90,120),225,np.uint8)
    truth=np.array([[25.,31.],[60.,62.],[98.,25.]])
    for x,y in truth.astype(int): cv2.circle(im,(x,y),3,20,-1,cv2.LINE_AA)
    queries=truth+np.array([1.1,-.8])
    measured,ok=associate(queries,dots(im)[::-1],5.)
    assert ok.all()
    np.testing.assert_allclose(measured,truth,atol=.1)
    assert np.linalg.norm(measured-queries)>1


def test_plane_stress_transverse_stretch_annuls_normal_stress():
    nu=.45; ratio=nu/(1-2*nu)
    g=np.array([[[1.04,.03],[-.01,.98]],[[.99,-.25],[.26,.965]]])
    d=np.linalg.det(g); f=np.tile(np.eye(3),(len(g),1,1))
    for i in range(len(g)): f[i][np.ix_([0,2],[0,2])]=g[i]
    f[:,1,1]=(1+ratio*d)/(1+ratio*d*d)
    mu,lam=corotated_cauchy_columns(f)
    sigma=mu/(2*(1+nu))+lam*nu/((1+nu)*(1-2*nu))
    np.testing.assert_allclose(sigma[:,1,1],0,atol=1e-14)


def test_spatial_reconstruction_derivatives_reproduce_affine_motion():
    p=dict(center=[.12,.12,.13],size=[.012,.020,.1],spline_degree_x=2,spline_spacing_z=.008)
    x,z=np.meshgrid(np.linspace(.114,.126,9),np.linspace(.08,.18,41),indexing='ij')
    q=np.column_stack([x.ravel(),np.full(x.size,.11),z.ravel()])
    M,Mx,Mz=spline_basis(q,p)
    u=np.column_stack([.02*q[:,0]-.03*q[:,2],.04*q[:,0]+.01*q[:,2]])
    c=np.linalg.lstsq(M,u,rcond=None)[0]
    np.testing.assert_allclose(M@c,u,atol=1e-12)
    np.testing.assert_allclose(Mx@c,np.tile([.02,.04],(len(q),1)),atol=1e-10)
    np.testing.assert_allclose(Mz@c,np.tile([-.03,.01],(len(q),1)),atol=1e-10)


def test_camera_weak_fields_preserve_measured_force_and_exclude_grip():
    p=dict(domain=.24,center=[.12,.12,.13],test_transitions=[[.112,.150],[.120,.152],[.128,.154]])
    points=np.array([[.103,.11,.095],[.122,.11,.170]])
    for w,grad in virtual_fields(points,96,p):
        np.testing.assert_allclose(w,[[1,0,0],[0,0,0]],atol=1e-12)
        np.testing.assert_allclose(grad,0,atol=1e-10)
