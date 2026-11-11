"""L02 Segmentation + L05 Image Geometry: width/length measurement in the A4 plane.
Includes analyze_frame() / analyze_batch(), the overall 2D pipeline entry points.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from .image_processing import resize_image
from .image_geometry import detect_paper, DetectionFailure
from .homography import rectify
from .color_segmentation import foot_mask
from .outlier_rejection import Observation, fuse

@dataclass
class FrameAnalysis:
    view_id: str
    kind: str
    accepted: bool
    mm: float | None
    reason: str
    corner_score: float | None
    refined: bool
    paper_corners: list | None
    mask_pixels: int
    image: np.ndarray | None=None
    paper_method: str | None=None
    paper_inferred_corner: bool=False

    def json(self):
        d=asdict(self);d.pop('image',None);return d


def top_width(mask,px_per_mm=2.):
    """Broad forefoot cross-sections instead of calf or individual toes."""
    h,w=mask.shape
    rows=[]
    for y in range(int(h*.14),int(h*.63)):
        xx=np.where(mask[y]>0)[0]
        if len(xx)>=.20*w:
            # A row spanning paper borders likely includes floor, not foot.
            if xx.min()<3 or xx.max()>w-4:continue
            rows.append(float((xx[-1]-xx[0])/px_per_mm))
    if len(rows)<30:return None,'forefoot_not_separable'
    widths=np.array(rows)
    # Avoid single-pixel spikes and choose repeat-supported broad region.
    width=float(np.percentile(widths,95))
    if not (35<=width<=165):return None,'width_out_of_physical_range'
    return width,'projected_forefoot_width'


def side_length(mask,px_per_mm=2.):
    """Toe-heel silhouette extent on A4 plane; biased by foot height, not ground truth."""
    h,w=mask.shape
    yy,xx=np.where(mask>0)
    good=(yy>h*.30)&(yy<h*.91)
    xy=xx[good]
    if len(xy)<.045*w*h:return None,'insufficient_visible_foot'
    # Reject shallow oblique/top-down silhouettes incorrectly submitted as a side
    # profile. This uses only image geometry, never a reference foot dimension.
    # Such masks do not establish an independent side-view toe-to-heel extent.
    ys_all, xs_all = np.where(mask > 0)
    if len(ys_all):
        vertical_coverage = (float(ys_all.max()) - float(ys_all.min())) / h
        upper_margin = float(ys_all.min()) / h
        if vertical_coverage < 0.48 and upper_margin > 0.18:
            return None, 'shallow_oblique_view_not_a_side_profile'
    # Repeated-column support retains true toe/heel boundary while rejecting
    # isolated single-pixel noise. Quantile trimming cut real tips in photos.
    occupancy=np.bincount(xy,minlength=w)
    repeated=np.where(occupancy>=6)[0]
    if len(repeated)<50:return None,'insufficient_endpoint_support'
    left=float(repeated[0]);right=float(repeated[-1])
    length=(right-left)/px_per_mm
    if left<=3 or right>=w-4:
        return None,'toe_or_heel_at_paper_border'
    if not (120<=length<=335):return None,'length_out_of_physical_range'
    return length,'floor_plane_projected_extent_not_physical_length'


def analyze_frame(bgr,kind,view_id):
    if kind not in ('top','side'):raise ValueError('kind must be top or side')
    im=resize_image(bgr)
    try:
        quad,info=detect_paper(im)
        rect,hom,dim=rectify(im,quad)
        mask=foot_mask(rect)
        if kind=='top':
            if dim[1]<=dim[0]:
                mm,reason=None,'paper_orientation_not_suitable_for_top'
            else:mm,reason=top_width(mask)
        else:
            if dim[0]<=dim[1]:
                mm,reason=None,'paper_orientation_not_suitable_for_side'
            else:mm,reason=side_length(mask)
        overlay=im.copy()
        cv2.polylines(overlay,[np.rint(quad).astype(np.int32)],True,(0,170,255),3)
        if info['method']=='occluded_white_core_lines':
            x,y=np.rint(quad[1]).astype(int)
            cv2.circle(overlay,(x,y),10,(255,30,200),3)
            cv2.putText(overlay,'INFERRED CORNER', (max(8,x-185),max(16,y-16)),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(255,30,200),2,cv2.LINE_AA)
        cv2.putText(overlay,f'A4 {info["method"]} / {"OK" if mm else "NO METRIC"}',(12,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        # An occluded fourth edge is enough for a DIAGNOSTIC paper hypothesis,
        # but not for a trusted metric observation until it is independently
        # cross-validated. Do not let one inferred corner displace existing
        # complete-border measurements in the multi-photo fusion.
        inferred=(info['method']=='occluded_white_core_lines')
        metric_valid=(mm is not None and not inferred)
        if inferred and mm is not None:
            reason='inferred_paper_corner_diagnostic_only'
        return FrameAnalysis(view_id,kind,metric_valid,round(mm,2) if mm is not None else None,reason,info['score'],info['refined'],quad.round(1).tolist(),int(np.count_nonzero(mask)),overlay,info['method'],inferred)
    except (DetectionFailure,cv2.error,ValueError) as exc:
        return FrameAnalysis(view_id,kind,False,None,str(exc),None,False,None,0,im)


def analyze_batch(tops,sides):
    """All photos contribute to ONE physically scoped measurement per kind."""
    items=[]
    for i,im in enumerate(tops):items.append(analyze_frame(im,'top',f'top-{i+1}'))
    for i,im in enumerate(sides):items.append(analyze_frame(im,'side',f'side-{i+1}'))
    res={}
    for kind,key in [('top','width'),('side','length')]:
        obs=[Observation(o.view_id,key,o.mm if o.mm is not None else 0.,o.corner_score is not None,o.mask_pixels>0,o.accepted,0.,o.reason if not o.accepted else None) for o in items if o.kind==kind]
        # All values remain PROJECTION estimates; fusion never removes height parallax.
        r=fuse(obs)
        res[key]=r.as_dict()
    return items,res
