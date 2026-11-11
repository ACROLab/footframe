"""Automatic, conservative 2D A4 border refinement.

Input: an *already automatically detected* paper quadrilateral (TL,TR,BR,BL),
plus original BGR image. Works as an optional post-processing step to v29.
No user clicks, no hardcoded foot sizes or camera parameters, no 3D model.

A4 is a planar known-size object, not a calibration of above-plane feet.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class A4Refinement:
    corners: np.ndarray
    accepted: bool
    reason: str
    residual_before: float
    residual_after: float
    edge_inliers: tuple[int,int,int,int]
    edge_sigma_px: tuple[float,float,float,float]


def order_quad(corners):
    p=np.asarray(corners,dtype=np.float64).reshape(4,2)
    centre=p.mean(axis=0)
    angles=np.arctan2(p[:,1]-centre[1],p[:,0]-centre[0])
    p=p[np.argsort(angles)]
    k=np.argmin(p.sum(axis=1))
    p=np.roll(p,-k,axis=0)
    # Force top-right as second point, not bottom-left.
    if p[1,0]<p[3,0]:p=p[[0,3,2,1]]
    return p


def _image_fields(bgr):
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    gray=cv2.GaussianBlur(lab[:,:,0],(5,5),0)
    return gray,hsv


def _safe_bilinear(a,xy):
    """Vectorized bilinear sample from a single-channel image at xy=(x,y)."""
    h,w=a.shape[:2]
    x=np.clip(xy[:,0],0,w-1.01).astype(np.float32)
    y=np.clip(xy[:,1],0,h-1.01).astype(np.float32)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=x0+1;y1=y0+1
    dx=x-x0;dy=y-y0
    return (a[y0,x0]*(1-dx)*(1-dy)+a[y0,x1]*dx*(1-dy)
            +a[y1,x0]*(1-dx)*dy+a[y1,x1]*dx*dy)


def _line_residuals(gray,hsv,quad,*,search_px=12, samples=145):
    """Extract signed paper-to-background candidate edge samples.

    A skin obstruction is masked using color only to stop the fit following
    the calf silhouette. It does not fill invisible paper or fabricate edges.
    """
    h,w=gray.shape
    center=quad.mean(axis=0)
    all_points=[]
    all_weights=[]
    all_scores=[]
    for j in range(4):
        a,b=quad[j],quad[(j+1)%4]
        tangent=(b-a)/max(1.,np.linalg.norm(b-a))
        n=np.array([-tangent[1],tangent[0]],float)
        if np.dot(n,center-(a+b)/2)<0:n=-n
        leng=float(np.linalg.norm(b-a))
        ts=np.linspace(.06,.94,max(50,min(samples,int(leng/2))),dtype=np.float64)
        mids=a[None,:]+ts[:,None]*(b-a)[None,:]
        offsets=np.arange(-search_px,search_px+1,dtype=np.float64)
        ps=mids[:,None,:]+offsets[None,:,None]*n[None,None,:]
        # Compare both sides of every candidate edge: paper is brighter inside.
        # Use a finite difference that tolerates 1-2px blur and mild shadows.
        outer=ps-4*n[None,None,:]
        inner=ps+4*n[None,None,:]
        inside=(outer[:,:,0]>2)&(outer[:,:,0]<w-3)&(outer[:,:,1]>2)&(outer[:,:,1]<h-3)&(inner[:,:,0]>2)&(inner[:,:,0]<w-3)&(inner[:,:,1]>2)&(inner[:,:,1]<h-3)
        score=_safe_bilinear(gray,inner.reshape(-1,2)).reshape(len(ts),len(offsets))-_safe_bilinear(gray,outer.reshape(-1,2)).reshape(len(ts),len(offsets))
        # Check that inner side has paper-like saturation; keep permissive to
        # handle overexposure and grey paper shadows.
        inside_sample=np.rint(inner).astype(int)
        yi=np.clip(inside_sample[:,:,1],0,h-1);xi=np.clip(inside_sample[:,:,0],0,w-1)
        s=hsv[yi,xi,1].astype(np.float32)
        brightness=_safe_bilinear(gray,inner.reshape(-1,2)).reshape(len(ts),len(offsets))
        score=np.where(inside&(s<95)&(brightness>105),score,-1e6)
        best=np.argmax(score,axis=1)
        strengths=score[np.arange(len(ts)),best]
        valid=strengths>10
        points=ps[np.arange(len(ts)),best][valid]
        strengths=np.clip(strengths[valid],0,100)
        all_points.append(points)
        all_weights.append(strengths)
        all_scores.append(float(np.median(np.clip(np.max(score,axis=1),-35,80))))
    return all_points,all_weights,all_scores


def _robust_line(points,weights,*,seed=91):
    """Local RANSAC + orthogonal total least squares; no sklearn dependency."""
    p=np.asarray(points,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if len(p)<18:return None,None,None
    rng=np.random.default_rng(seed)
    best=None;best_score=-1.
    for _ in range(180):
        i,j=rng.choice(len(p),size=2,replace=False)
        d=p[j]-p[i]; d=d/ max(np.linalg.norm(d),1e-12)
        n=np.array([-d[1],d[0]])
        res=np.abs((p-p[i])@n)
        inliers=res<2.3
        score=np.sum(w[inliers])
        if score>best_score:
            best_score=score;best=inliers
    if best is None or best.sum()<18:return None,None,None
    q=p[best];ww=w[best];ww=np.maximum(1.,ww)
    centroid=np.average(q,axis=0,weights=ww)
    X=q-centroid
    cov=(X*ww[:,None]).T@X/ww.sum()
    _,V=np.linalg.eigh(cov)
    normal=V[:,0]
    sigma=np.sqrt(np.average((X@normal)**2,weights=ww))
    if sigma>2.5:return None,None,None
    return (normal,-float(normal@centroid)),int(best.sum()),float(max(sigma,0.2))


def _intersection(l1,l2):
    A=np.array([l1[0],l2[0]],dtype=float)
    b=-np.array([l1[1],l2[1]],dtype=float)
    if abs(np.linalg.det(A))<0.10:return None
    return np.linalg.solve(A,b)


def _support_evidence(gray,hsv,q):
    points,weights,scores=_line_residuals(gray,hsv,q,search_px=3,samples=120)
    return float(np.mean([np.mean(w) if len(w)>8 else 0. for w in weights])),[len(p) for p in points]


def refine_quad(image,quad,*,search_px=12,max_corner_shift_px=18.) -> A4Refinement:
    """Return original points if evidence is insufficient; never force a fit.

    Any assumed unit is image pixels, so resize before calling if desired.
    Only accept adjustment when both geometry & boundary quality improve.
    """
    p=order_quad(quad)
    def fail(reason,before=0,after=0,n=(0,0,0,0),sig=(0.,0.,0.,0.)):
        return A4Refinement(p.copy(),False,reason,float(before),float(after),tuple(n),tuple(sig))
    im=np.asarray(image)
    if im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<120:return fail('invalid_image')
    h,w=im.shape[:2]
    if (p[:,0].min()<3 or p[:,0].max()>w-4 or p[:,1].min()<3 or p[:,1].max()>h-4):return fail('paper_touches_border')
    gray,hsv=_image_fields(im)
    points,weights,_=_line_residuals(gray,hsv,p,search_px=search_px)
    lines=[];ns=[];sig=[]
    for j,(edge_points,weight) in enumerate(zip(points,weights)):
        line,n,sigma=_robust_line(edge_points,weight,seed=73+j)
        if line is None:return fail('insufficient_visible_edge',n=tuple(len(x) for x in points))
        # Important: the visible line support should cover enough of the edge.
        t=p[(j+1)%4]-p[j]
        f=np.dot(edge_points-p[j],t)/max(1.,np.dot(t,t))
        if np.percentile(f,90)-np.percentile(f,10)<.30:return fail('insufficient_spread')
        lines.append(line);ns.append(n);sig.append(sigma)
    refined=[]
    for j in range(4):
        x=_intersection(lines[(j-1)%4],lines[j])
        if x is None:return fail('nearly_parallel_edges',n=ns,sig=sig)
        refined.append(x)
    refined=np.array(refined,np.float64)
    shifts=np.linalg.norm(refined-p,axis=1)
    if np.any(shifts>max_corner_shift_px):return fail('corner_shift_too_large',n=ns,sig=sig)
    if not cv2.isContourConvex(refined.astype(np.float32)):return fail('nonconvex_quad',n=ns,sig=sig)
    old_area=cv2.contourArea(p.astype(np.float32));new_area=cv2.contourArea(refined.astype(np.float32))
    if not .85<new_area/max(old_area,1)<1.15:return fail('area_changed_too_much',n=ns,sig=sig)
    before,bn=_support_evidence(gray,hsv,p)
    after,an=_support_evidence(gray,hsv,refined)
    if after+0.25<before:return fail('edge_contrast_worse',before,after,ns,sig)
    return A4Refinement(refined,True,'accepted',float(before),float(after),tuple(ns),tuple(sig))


def sensitivity_mm(q, endpoint_pairs, *, sigma_corner_px=2., N=500, seed=49):
    """Paper homography sensitivity ONLY; no coverage claim about foot height.

    Endpoint coordinates are in the image plane and must come from an independent
    automatic detector. This experiment estimates corner-localization sensitivity,
    not a confidence interval for the *true* foot length.
    """
    q=order_quad(q).astype(np.float32)
    ep=np.asarray(endpoint_pairs,dtype=np.float32).reshape(-1,2,2)
    if len(ep)!=1:raise ValueError('Supply exactly one independently detected endpoint pair per sensitivity run')
    dst=np.array([[0,0],[210,0],[210,297],[0,297]],np.float32)
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(N):
        pert=q+rng.normal(0,sigma_corner_px,(4,2)).astype(np.float32)
        H=cv2.getPerspectiveTransform(pert,dst)
        if not np.isfinite(H).all():continue
        for pair in ep:
            proj=cv2.perspectiveTransform(pair.reshape(1,2,2),H).reshape(2,2)
            values.append((float(np.linalg.norm(proj[0]-proj[1])),len(values)%len(ep)))
    # Standard deviation of a known image segment's physical projection.
    lengths=np.array([a for a,b in values])
    if len(lengths)==0:raise ValueError('homography failed')
    return {'projected_median_mm':float(np.median(lengths)),
            'corner_jitter_p05_mm':float(np.percentile(lengths,5)),
            'corner_jitter_p95_mm':float(np.percentile(lengths,95)),
            'sample_count':int(len(lengths)),
            'scope':'Only random A4 corner perturbations; excludes toe/heel localization, lens and height parallax.'}
