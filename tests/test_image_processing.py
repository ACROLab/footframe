"""Standalone tests: never rely on a particular person's foot size."""
import json
import numpy as np
import cv2
from core.image_geometry import _paper_proof, DetectionFailure
from core.homography import rectify
from core.image_processing import resize_image
from core.measurement_2d import top_width, side_length
from core.color_segmentation import foot_mask
from core.outlier_rejection import Observation, fuse
from app import RequestHandler,WEB,STATIC_FILES,process_uploads


def test_screen_is_mobile_and_declares_scope():
    page = (WEB / 'index.html').read_text(encoding='utf-8')
    script = (WEB / 'app.js').read_text(encoding='utf-8')
    assert 'viewport' in page
    assert '2D 투영 추정치' in page
    assert 'top-input' in page and 'side-input' in page
    assert "fetch('/api/jobs'" in script
    assert "/api/jobs/${id}" in script
    assert "/web/foot_measurement.webp" in STATIC_FILES
    assert '/web/styles.css' in STATIC_FILES


def test_reference_dimensions_for_landscape_and_portrait():
    img=np.full((500,600,3),200,np.uint8)
    landscape=np.array([[60,100],[500,100],[500,400],[60,400]],float)
    _,_,dims=rectify(img,landscape)
    assert dims==(297,210)
    portrait=np.array([[100,30],[380,30],[380,460],[100,460]],float)
    _,_,dims=rectify(img,portrait)
    assert dims==(210,297)


def test_blank_floor_has_no_verified_paper():
    im=np.full((700,950,3),136,np.uint8)
    q=np.array([[100,100],[550,100],[550,500],[100,500]],float)
    assert _paper_proof(im,q) is None


def test_skin_mask_excludes_clean_white_paper():
    image=np.full((420,594,3),255,np.uint8)
    assert foot_mask(image).sum()==0


def test_width_extracts_supported_forefoot():
    m=np.zeros((594,420),np.uint8)
    cv2.ellipse(m,(210,200),(90,135),0,0,360,255,-1)
    value,reason=top_width(m)
    assert reason=='projected_forefoot_width'
    assert 80<value<93


def test_side_length_column_support_rejects_single_noise_pixel():
    m=np.zeros((420,594),np.uint8)
    cv2.rectangle(m,(40,140),(535,350),255,-1)
    m[180,5]=255;m[200,580]=255
    value,reason=side_length(m)
    assert abs(value-247.5)<1
    assert reason=='floor_plane_projected_extent_not_physical_length'


def test_no_false_fusion_when_sides_disagree():
    r=fuse([Observation('a','length',170,True,True,True),Observation('b','length',250,True,True,True)])
    assert r.accepted_mm is None


def test_invalid_uploads_are_rejected():
    try:process_uploads([b'not-an-image'],[b'also-not-an-image'])
    except ValueError:return
    raise AssertionError('Invalid bytes accepted as photos')


def test_side_shallow_oblique_rejected_without_reference_value():
    """A top-down-like thin mask is insufficient to establish side-view length."""
    import numpy as np
    import cv2
    from core.measurement_2d import side_length
    mask=np.zeros((420,594),np.uint8)
    cv2.ellipse(mask,(295,230),(230,68),0,0,360,255,-1)
    mm,why=side_length(mask)
    assert mm is None
    assert why=='shallow_oblique_view_not_a_side_profile'


def test_side_genuine_tall_profile_keeps_previous_extent():
    import numpy as np
    import cv2
    from core.measurement_2d import side_length
    mask=np.zeros((420,594),np.uint8)
    cv2.ellipse(mask,(295,204),(230,122),0,0,360,255,-1)
    cv2.rectangle(mask,(275,0),(350,205),255,-1)
    mm,why=side_length(mask)
    assert mm is not None
    assert why=='floor_plane_projected_extent_not_physical_length'
