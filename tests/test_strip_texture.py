"""Image-only geometry and strain-sensitive correspondence checks."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.elastic.strip_camera_observe import CAMERA,calibration
from experiments.elastic.strip_texture_identify import RECT,rectification,warp_points
from experiments.elastic.strip_texture_dic import correlate


def test_plane_rectification_agrees_across_calibrated_cameras():
    p={'surface_reference_y':.118}
    rect=np.array([[420.,330.],[450.,800.],[490.,500.]])
    world=np.column_stack([RECT['xmin']+rect[:,0]/RECT['scale'],np.full(3,.118),RECT['zmax']-rect[:,1]/RECT['scale']])
    for c in calibration(CAMERA):
        h=rectification(c,p)
        expected=np.column_stack([world,np.ones(3)])@np.asarray(c['P']).T
        np.testing.assert_allclose(warp_points(rect,h),expected[:,:2]/expected[:,2,None],atol=1e-10)
        np.testing.assert_allclose(warp_points(warp_points(rect,h),np.linalg.inv(h)),rect,atol=1e-10)


def test_affine_dic_recovers_motion_with_rotation_strain_and_brightness_change():
    rng=np.random.default_rng(1)
    im=gaussian_filter(rng.uniform(50,230,(200,220)),1).astype('float32')
    M=np.array([[1.01,.04,1.3],[-.03,.99,2.5]],dtype='float32')
    current=cv2.warpAffine(im,M,(220,200))*1.08-7
    q=np.array([[60.,60.],[100.,100.],[150.,120.]])
    truth=q@M[:,:2].T+M[:,2]
    measured,affine,_=correlate(im,current,q,truth+.4)
    np.testing.assert_allclose(measured,truth,atol=.04)
    np.testing.assert_allclose(affine,np.broadcast_to(M[:,:2],affine.shape),atol=.008)
