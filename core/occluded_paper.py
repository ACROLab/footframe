"""Conservative four-border reconstruction for partly covered *blank white A4*.

Two independent sources are required: a connected, low-saturation bright core
produces visible mask-boundary lines, and an LSD segment supports the partly
occluded fourth side. This is a proposal stage, never standalone validation.
No person-specific dimensions, clicks, markers, or inferred 3D geometry.
"""
from __future__ import annotations
import cv2
import numpy as np


def _line_segment(p):
    p = np.asarray(p, dtype=np.float64).reshape(4)
    a, b = p[:2], p[2:]
    length = np.linalg.norm(a - b)
    if length < 35:
        return None
    return a, b, float(length)


def _intersect(s1, s2):
    a, b, _ = s1
    c, d, _ = s2
    u, v = b-a, d-c
    A = np.column_stack((u, -v))
    det = np.linalg.det(A)
    if abs(det) < 0.10*np.linalg.norm(u)*np.linalg.norm(v):
        return None
    return a + np.linalg.solve(A,c-a)[0]*u


def _orientation(segment):
    a, b, _ = segment
    v = b-a
    # Robust to endpoint ordering.
    return abs(np.degrees(np.arctan2(v[1],v[0])) % 180)


def occluded_paper_proposals(image, *, max_proposals=24):
    """Yield (TL,TR,BR,BL) candidates supported by connected white pixels.

    Uses image-percentile brightness, which is NOT a skin/foot-length calibration.
    Contour lines MUST exist on three distinct margins of the white region;
    the fourth line MUST independently arise from the intensity LSD detector.
    This deliberately cannot infer a sheet from an arbitrary smooth patch.
    """
    if image is None or image.ndim != 3:
        return []
    h,w = image.shape[:2]
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    l = lab[:,:,0]
    # A4's white cores are near the top tail of image brightness, even under
    # moderately dim lighting. A low saturation check rejects colored floor.
    threshold = float(np.percentile(l,97)) - 6.
    white = np.uint8((l > threshold)&(hsv[:,:,1] < 85))*255
    white = cv2.morphologyEx(white,cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    components, _ = cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    components = sorted(components,key=cv2.contourArea,reverse=True)[:10]

    # Original-frame, independent intensity lines. We do not generate the
    # missing border by extending a component bounding box alone.
    grayscale = cv2.GaussianBlur(l,(5,5),0)
    raw = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(grayscale)[0]
    if raw is None:
        return []
    lsd = [_line_segment(line) for line in raw.reshape(-1,4)]
    lsd = [x for x in lsd if x is not None]
    proposals=[]
    for c in components:
        area = cv2.contourArea(c)/(h*w)
        if not (.025 < area < .28):
            continue
        x,y,bw,bh = cv2.boundingRect(c)
        if min(bw,bh) < .16*min(h,w):
            continue
        component_mask = np.zeros((h,w),np.uint8)
        cv2.drawContours(component_mask,[c],-1,255,-1)
        edges = cv2.Canny(component_mask,30,90)
        lines = cv2.HoughLinesP(edges,1,np.pi/360,threshold=30,
                               minLineLength=max(50,int(.10*min(h,w))),maxLineGap=14)
        if lines is None:
            continue
        src = [_line_segment(z) for z in lines.reshape(-1,4)]
        src = [z for z in src if z is not None]
        if len(src) < 3:
            continue
        # Use up to three geometrically distinct lines on the visible white
        # core's top/bottom/left. The occluded right edge may be outside it.
        horizontals=[s for s in src if _orientation(s)<27 or _orientation(s)>153]
        verticals=[s for s in src if 63<_orientation(s)<117]
        top=sorted((s for s in horizontals if min(s[0][1],s[1][1]) < y+.27*bh),
                   key=lambda s:-s[2])[:3]
        bottom=sorted((s for s in horizontals if max(s[0][1],s[1][1]) > y+.70*bh),
                      key=lambda s:-s[2])[:3]
        left=sorted((s for s in verticals if min(s[0][0],s[1][0]) < x+.16*bw),
                    key=lambda s:-s[2])[:3]
        # Require a distinct line support to the right of the bright core,
        # with its centre near the paper-height range; the material proof is
        # applied later to the complete quadrilateral.
        right=[]
        for s in lsd:
            if not (63<_orientation(s)<117):
                continue
            mid=(s[0]+s[1])*.5
            if not (mid[0]>x+.83*bw and mid[0]<x+bw+.22*w):
                continue
            if not (y-.12*h<mid[1]<y+bh+.12*h):
                continue
            if s[2]<.32*bh:
                continue
            right.append(s)
        right=sorted(right,key=lambda s:-s[2])[:6]
        if not (top and bottom and left and right):
            continue
        for t in top:
            for b in bottom:
                for le in left:
                    for ri in right:
                        quad=[_intersect(t,le),_intersect(t,ri),
                              _intersect(b,ri),_intersect(b,le)]
                        if any(q is None for q in quad):
                            continue
                        q=np.asarray(quad,np.float64)
                        if not np.isfinite(q).all():
                            continue
                        if q[:,0].min()<5 or q[:,0].max()>w-5 or q[:,1].min()<5 or q[:,1].max()>h-5:
                            continue
                        if not cv2.isContourConvex(q.astype(np.float32)):
                            continue
                        quad_area=cv2.contourArea(q.astype(np.float32))/(h*w)
                        if not .055<quad_area<.45:
                            continue
                        lens=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
                        if min(lens)<.16*min(h,w):
                            continue
                        if max(lens[0]/lens[2],lens[2]/lens[0],lens[1]/lens[3],lens[3]/lens[1])>1.85:
                            continue
                        # The observed white component must substantially lie
                        # inside the proposed sheet. This is NOT a length prior.
                        region=np.zeros((h,w),np.uint8)
                        cv2.fillConvexPoly(region,np.round(q).astype(np.int32),1)
                        if cv2.countNonZero(cv2.bitwise_and(component_mask,region)) < .87*cv2.countNonZero(component_mask):
                            continue
                        # Rank by visual-edge length, not closeness to any
                        # particular physical foot dimension.
                        support=(t[2]+b[2]+le[2]+ri[2]) / (2*(bw+bh))
                        proposals.append((support,q))
    proposals.sort(key=lambda p:-p[0])
    return [q for _,q in proposals[:max_proposals]]
