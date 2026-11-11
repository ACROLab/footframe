"""L04 Image Formation + L05 Image Geometry: A4 quadrilateral verification and selection."""
from __future__ import annotations
import cv2
import numpy as np
from .ransac import refine_quad, order_quad
from .edge_detection import detect_a4
from .hough_transform import occluded_paper_proposals
from .color_segmentation import _candidate_contours

class DetectionFailure(ValueError):
    pass


def _paper_proof(image, quad):
    h,w=image.shape[:2]
    q=order_quad(quad)
    if not np.isfinite(q).all() or np.min(q)<-1 or q[:,0].max()>w+1 or q[:,1].max()>h+1:
        return None
    area=cv2.contourArea(q.astype(np.float32))/(h*w)
    if not (.055<area<.45):return None
    lens=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    if min(lens)<.13*min(h,w):return None
    if not cv2.isContourConvex(q.astype(np.float32)):return None
    if max(lens[0]/lens[2],lens[2]/lens[0],lens[1]/lens[3],lens[3]/lens[1])>2.0:return None
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(np.float32)
    l=lab[:,:,0]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    m=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(m,np.rint(q).astype(np.int32),1)
    inner=cv2.erode(m,np.ones((21,21),np.uint8))>0
    if inner.sum()<800:return None
    avg=cv2.blur(l,(21,21));std=np.sqrt(np.maximum(0,cv2.blur(l*l,(21,21))-avg*avg))
    smooth=float(np.mean(std[inner]<4))
    # False floor+clothes quadrilaterals, especially 11701, have no mostly smooth A4.
    if smooth<.34:return None
    # Skin and hair should not count as paper.
    paperlike=inner & (std<4) & (hsv[:,:,1]<83)
    fraction=float(np.mean(paperlike[inner]))
    if fraction<.24:return None
    # Actual paper needs a verifiable bright-inside boundary on >= 3 edges.
    center=q.mean(axis=0)
    supports=[]; inside_neutral=[]
    for a,b in zip(q,np.roll(q,-1,axis=0)):
        tangent=b-a
        normal=np.array([-tangent[1],tangent[0]],float)
        normal/=max(np.linalg.norm(normal),1)
        if np.dot(normal,center-.5*(a+b))<0:normal=-normal
        pts=a[None,:]+np.linspace(.14,.86,43)[:,None]*tangent[None,:]
        inside=np.rint(pts+8*normal).astype(int)
        outside=np.rint(pts-8*normal).astype(int)
        ok=(inside[:,0]>0)&(inside[:,0]<w)&(inside[:,1]>0)&(inside[:,1]<h)&(outside[:,0]>0)&(outside[:,0]<w)&(outside[:,1]>0)&(outside[:,1]<h)
        if ok.sum()<18: supports.append(0.);continue
        iy,ix=inside[ok,1],inside[ok,0]
        oy,ox=outside[ok,1],outside[ok,0]
        diff=(l[iy,ix]-l[oy,ox])
        supports.append(float(np.mean(diff>8)))
        inside_neutral.append(float(np.mean(hsv[iy,ix,1]<83)))
    # Reject quadrilaterals whose nominal side is actually a clothes/skin
    # boundary: low outward white contrast AND non-paper color on the inside.
    # Permit partly occluded sides only when independent paper evidence remains.
    if any(c<.32 and neutral<.55 for c,neutral in zip(supports,inside_neutral)):
        return None
    nstrong=sum(a>.35 for a in supports)
    if nstrong<3:return None
    return float(100*smooth+18*fraction+6*sum(supports)+10*nstrong)


def detect_paper(image):
    """Detect sheet automatically; returns verified A4 corners and diagnostics.

    Experimental LSD proposals are always checked against independent smoothness,
    brightness direction and geometric conditions. We do not fabricate a paper.
    """
    if image is None or image.ndim!=3 or image.shape[2]!=3:raise DetectionFailure('색상 이미지가 필요합니다')
    candidates=[]
    p=detect_a4(image,max_lines=16)
    if p is not None:candidates.append((p.corners,'line_intersections'))
    candidates += [(q,'contour') for q in _candidate_contours(image)]
    valid=[]
    for q,source in candidates:
        score=_paper_proof(image,q)
        if score is not None:valid.append((score,q,source))
    # A partially covered paper can produce a perfectly plausible but false
    # four-line intersection along the edge of clothing. Do not trust that
    # hypothesis if independent line refinement cannot support it.
    initial=max(valid,key=lambda x:x[0]) if valid else None
    initial_refined=refine_quad(image,initial[1]) if initial else None
    if initial is None or not initial_refined.accepted:
        fallback=[]
        for q in occluded_paper_proposals(image):
            score=_paper_proof(image,q)
            if score is not None:
                fallback.append((score,q,'occluded_white_core_lines'))
        if fallback:
            best_fallback=max(fallback,key=lambda x:x[0])
            # When the primary is geometrically unstable, prefer a stronger
            # independent white-border hypothesis with locally supported fit.
            if initial is None or best_fallback[0]>initial[0]+2.0:
                fr=refine_quad(image,best_fallback[1])
                if initial is None or fr.accepted:
                    initial=best_fallback
                    initial_refined=fr
    if initial is None:
        raise DetectionFailure('A4의 독립된 네 경계를 확인할 수 없습니다. 발이 종이를 가려도 되지만 최소한 3개 테두리의 증거가 필요합니다.')
    score,q,source=initial
    result=initial_refined
    if result.accepted and _paper_proof(image,result.corners) is not None:
        q=result.corners
    return order_quad(q), {'score':round(score,2),'method':source,
                           'refined':bool(result.accepted),'refinement':result.reason}
