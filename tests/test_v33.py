"""v33 regression: no subject-measurement calibration, synthetic planar checks."""
from pathlib import Path
import numpy as np
import cv2
from core.geometry import detect_paper,DetectionFailure
from core.measurement import analyze_frame
from core.fuse_2d import Observation,fuse


def occluded_synthetic(seed=12):
    """Independent synthetic floor and paper with an occluded top-right corner.

    The 2D paper corners are the only known truth. There is no 3D foot model.
    """
    h,w=680,880
    rng=np.random.default_rng(seed)
    floor=np.full((h,w,3),160,np.float32)
    noise=cv2.GaussianBlur(rng.normal(0,9,(h,w)).astype(np.float32),(5,5),1)
    image=np.uint8(np.clip(floor+noise[:,:,None],0,255))
    corners=np.array([[270,175],[615,192],[604,442],[250,414]],np.int32)
    cv2.fillConvexPoly(image,corners,(234,236,237))
    cv2.ellipse(image,(499,305),(164,66),-1,0,360,(126,151,189),-1)
    cv2.rectangle(image,(555,180),(700,276),(39,38,43),-1)
    return image,corners


def test_occluded_a4_corner_extrapolation_agrees_with_synthetic_truth():
    photo,truth=occluded_synthetic()
    corners,meta=detect_paper(photo)
    assert meta['method']=='occluded_white_core_lines'
    assert np.mean(np.linalg.norm(corners-truth,axis=1))<8
    assert meta['refined'] in (True,False)


def test_occluded_a4_is_diagnostic_not_authoritative_metric():
    photo,_=occluded_synthetic()
    f=analyze_frame(photo,'side','occluded-synthetic')
    assert f.paper_method=='occluded_white_core_lines'
    assert f.paper_inferred_corner
    assert not f.accepted
    assert f.paper_corners is not None


def test_soft_bright_blob_without_four_sides_rejected():
    rng=np.random.default_rng(32)
    h,w=650,900
    im=np.uint8(np.clip(145+rng.normal(0,3,(h,w,1)),0,255))
    im=np.broadcast_to(im,(h,w,3)).copy()
    cv2.ellipse(im,(455,320),(220,115),11,0,360,(238,238,238),-1)
    try:
        detect_paper(im)
    except DetectionFailure:
        return
    raise AssertionError('Bright non-rectangular patch was accepted as A4')


def test_small_three_view_disagreement_does_not_delete_one_valid_view():
    obs=[Observation(str(i),'length',v,True,True,True) for i,v in enumerate((180,185.5,181))]
    result=fuse(obs)
    assert len(result.used_views)==3
    assert abs(result.accepted_mm-np.mean([180,185.5,181]))<1e-6
    assert result.spread_mm==5.5


def test_large_outlier_still_removed():
    obs=[Observation(str(i),'length',v,True,True,True) for i,v in enumerate((180,181,260))]
    result=fuse(obs)
    assert result.used_views==['0','1']
    assert result.rejected['2']=='inconsistent_with_other_views'


def test_real_a4_recovery_if_private_samples_present():
    # Optional regression with the private photos; production deployments
    # deliberately omit those files.
    root=Path(__file__).resolve().parents[3]
    p=root/'samples'/'side'/'11701.jpg'
    if not p.is_file():
        import pytest
        pytest.skip('private photo absent from deploy-safe source tree')
    f=analyze_frame(cv2.imread(str(p)),'side','11701')
    assert f.paper_method=='occluded_white_core_lines'
    assert f.paper_corners is not None
    assert f.paper_inferred_corner
    assert f.mm is not None
    assert not f.accepted


def test_occluded_real_photo_underexposure_keeps_same_a4_if_private_data_present():
    root=Path(__file__).resolve().parents[3]
    p=root/'samples'/'side'/'11701.jpg'
    if not p.is_file():
        import pytest
        pytest.skip('private photo absent from deploy-safe source tree')
    from core.geometry import resize_image
    original=resize_image(cv2.imread(str(p)))
    a,_=detect_paper(original)
    dark=cv2.convertScaleAbs(original,alpha=0.90)
    b,_=detect_paper(dark)
    assert np.mean(np.linalg.norm(a-b,axis=1))<6
