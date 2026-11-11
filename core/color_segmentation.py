"""L02 Image Processing + L03 Color: thresholding, morphology, connected components.
Produces white-paper contour proposals and the foot silhouette.
"""
from __future__ import annotations
import cv2
import numpy as np

def _candidate_contours(image):
    """Independent color/contour hypothesis for white sheets. Never trusts the mask alone."""
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    l=lab[:,:,0]
    h,w=l.shape
    sm=cv2.GaussianBlur(l,(13,13),0)
    # Adaptive local brightness + low saturation, robust to moderately dark pictures.
    local=cv2.GaussianBlur(sm,(0,0),sigmaX=31)
    mask=np.uint8(((sm>local+7)&(hsv[:,:,1]<83)))*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((17,17),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in sorted(cnts,key=cv2.contourArea,reverse=True)[:12]:
        area=cv2.contourArea(c)/(h*w)
        if not .045<area<.55:continue
        hull=cv2.convexHull(c)
        peri=cv2.arcLength(hull,True)
        approx=cv2.approxPolyDP(hull,.03*peri,True)
        if len(approx)==4:
            out.append(approx.reshape(4,2).astype(float))
    return out


def foot_mask(rect):
    """Reference-free color segmentation; no body-size/skin-tone training."""
    hsv=cv2.cvtColor(rect,cv2.COLOR_BGR2HSV)
    lab=cv2.cvtColor(rect,cv2.COLOR_BGR2LAB)
    skin=(hsv[:,:,0]<30)&(hsv[:,:,1]>24)&(hsv[:,:,2]>33)&(lab[:,:,1]>122)
    m=np.uint8(skin)*255
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    n,idx,stats,_=cv2.connectedComponentsWithStats(m)
    if n<=1:return np.zeros_like(m)
    which=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    if stats[which,cv2.CC_STAT_AREA] <.045*m.size:return np.zeros_like(m)
    mm=np.uint8(idx==which)*255
    mm=cv2.morphologyEx(mm,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    return mm
