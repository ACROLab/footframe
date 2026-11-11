"""L02 Edge Detection + L06 Linear Algebra: LSD edge segments and A4 candidates.

No training or photo-specific coordinates. This detector is conservative: only
high-contrast, geometrically consistent candidates are returned. Never accept proposals
solely because an orange quad can be drawn around a foot.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from .ransac import order_quad


@dataclass
class Proposal:
    corners: np.ndarray
    confidence: float
    edges_supported: int
    reason: str


def line_geom(segment):
    p=np.array(segment[:2],dtype=float);q=np.array(segment[2:],dtype=float)
    v=q-p;length=np.linalg.norm(v)
    if length<1:return None
    theta=np.degrees(np.arctan2(v[1],v[0]))%180
    normal=np.array([-v[1],v[0]])/length
    return p,q,theta,normal,float(length)


def signed_edge_support(gray,seg,offset=6):
    geom=line_geom(seg)
    if geom is None:return 0.
    p,q,theta,n,length=geom;h,w=gray.shape
    t=np.linspace(.18,.82,21)
    c=p[None,:]+t[:,None]*(q-p)[None,:]
    pa=np.rint(c+n*offset).astype(int);pb=np.rint(c-n*offset).astype(int)
    ok=(pa[:,0]>=0)&(pa[:,0]<w)&(pa[:,1]>=0)&(pa[:,1]<h)&(pb[:,0]>=0)&(pb[:,0]<w)&(pb[:,1]>=0)&(pb[:,1]<h)
    if ok.sum()<15:return 0.
    a=gray[pa[ok,1],pa[ok,0]].astype(float);b=gray[pb[ok,1],pb[ok,0]].astype(float)
    return float(abs(np.median(a-b)))


def intersect(a,b):
    pa,qa,*_=line_geom(a);pb,qb,*_=line_geom(b)
    va=qa-pa;vb=qb-pb
    A=np.stack((va,-vb),axis=1)
    if abs(np.linalg.det(A))<.06*np.linalg.norm(va)*np.linalg.norm(vb):return None
    t=np.linalg.solve(A,pb-pa)
    return pa+t[0]*va


def proposal_score(gray,hsv,quad,local_std):
    h,w=gray.shape
    region=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(region,np.rint(quad).astype(int),1)
    inner=cv2.erode(region,np.ones((17,17),np.uint8))
    # paper area can be occluded but must contain enough near-white pixels;
    # the highly textured floor alone is not a valid A4.
    if inner.sum()<.06*h*w:return None
    c=gray[inner>0];S=hsv[:,:,1][inner>0]
    bright=float(np.mean((c>145)&(S<83)))
    if bright<.18:return None
    # Wood-grain and other flat surfaces frequently form plausible rectangles.
    # Real blank A4 interior is mostly smooth even with an occluding foot.
    paper_like=(inner>0)&(gray>145)&(hsv[:,:,1]<83)
    if np.count_nonzero(paper_like)<600:return None
    smooth=float(np.mean(local_std[paper_like]<4.))
    if smooth<.67:return None
    center=quad.mean(axis=0);supports=[]
    skin_mask=((hsv[:,:,0]<30)&(hsv[:,:,1]>20)&(hsv[:,:,2]>45))
    for j in range(4):
        a,b=quad[j],quad[(j+1)%4];d=b-a;length=np.linalg.norm(d)
        if length<55:return None
        n=center-(a+b)/2;n=n/max(np.linalg.norm(n),1)
        t=np.linspace(.10,.90,34);c=a[None,:]+t[:,None]*d[None,:]
        inn=np.rint(c+9*n[None,:]).astype(int);out=np.rint(c-9*n[None,:]).astype(int)
        ok=(inn[:,0]>=0)&(inn[:,0]<w)&(inn[:,1]>=0)&(inn[:,1]<h)&(out[:,0]>=0)&(out[:,0]<w)&(out[:,1]>=0)&(out[:,1]<h)
        inn=inn[ok];out=out[ok]
        if len(inn)<18:return None
        diff=gray[inn[:,1],inn[:,0]].astype(float)-gray[out[:,1],out[:,0]].astype(float)
        skin=skin_mask[inn[:,1],inn[:,0]]
        supports.append((float(np.percentile(diff,44)),float(np.mean(diff>11)),float(np.mean(skin))))
    good=sum(1 for contrast,portion,skin in supports if contrast>11 and portion>.45)
    if good<3:return None
    if any(skin>.68 for c,f,skin in supports):return None
    contrast=float(np.sum(np.clip([x[0] for x in supports],-20,80)))
    return (contrast*.15+bright*18+good*6+smooth*14),good


def detect_a4(image,*,max_lines=30):
    if image is None or image.ndim!=3:return None
    im=image
    h,w=im.shape[:2]
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)
    gray=cv2.GaussianBlur(lab[:,:,0]+1.2*(128-lab[:,:,2]),(7,7),0)
    gray=np.uint8(np.clip(gray,0,255))
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    gl=gray.astype(np.float32)
    local_mean=cv2.blur(gl,(13,13))
    local_std=np.sqrt(np.maximum(0.,cv2.blur(gl*gl,(13,13))-local_mean*local_mean))
    raw=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    if raw is None:return None
    lines=raw.reshape(-1,4)
    nearH=[];nearV=[]
    for p in lines:
        g=line_geom(p)
        if g is None:continue
        a,b,ang,n,length=g
        if length<.09*min(h,w):continue
        contrast=signed_edge_support(gray,p)
        if contrast<9:continue
        # The sample pictures can be rotated; flexible orientation windows.
        if ang<35 or ang>145:nearH.append((length*contrast,p))
        elif 48<ang<132:nearV.append((length*contrast,p))
    if len(nearH)<2 or len(nearV)<2:return None
    nearH.sort(key=lambda z:z[0],reverse=True)
    nearV.sort(key=lambda z:z[0],reverse=True)
    hs=[e[1] for e in nearH[:max_lines]]
    vs=[e[1] for e in nearV[:max_lines]]
    best=None
    for ia,a in enumerate(hs):
        for ib,b in enumerate(hs):
            if ia==ib:continue
            ay=np.mean(a[1::2]);by=np.mean(b[1::2])
            if ay+min(h,w)*.10>by:continue
            for ic,c in enumerate(vs):
                tl=intersect(a,c);bl=intersect(b,c)
                if tl is None or bl is None:continue
                for id,d in enumerate(vs):
                    if ic==id:continue
                    tr=intersect(a,d);br=intersect(b,d)
                    if tr is None or br is None:continue
                    p=np.array((tl,tr,br,bl),dtype=float)
                    if not np.isfinite(p).all():continue
                    if (p[:,0].min()<w*.015 or p[:,0].max()>w*.985 or
                        p[:,1].min()<h*.015 or p[:,1].max()>h*.985):continue
                    if not cv2.isContourConvex(p.astype(np.float32)):continue
                    area=cv2.contourArea(p.astype(np.float32))/(h*w)
                    if not .085<area<.34:continue
                    lens=np.linalg.norm(p-np.roll(p,-1,axis=0),axis=1)
                    opp=max(lens[0],lens[2])/min(lens[0],lens[2]);opp2=max(lens[1],lens[3])/min(lens[1],lens[3])
                    if opp>1.9 or opp2>1.9:continue
                    mean_horizontal=.5*(lens[0]+lens[2]);mean_vertical=.5*(lens[1]+lens[3])
                    if not .38<mean_horizontal/max(mean_vertical,1)<2.6:continue
                    score=proposal_score(gray,hsv,p,local_std)
                    if score is None:continue
                    val,valid=score
                    if best is None or val>best.confidence:best=Proposal(p,val,valid,'unverified_proposal')
    return best
