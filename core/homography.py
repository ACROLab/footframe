"""L05 Image Geometry + L06 Homography: A4 paper-plane perspective rectification.
This does NOT compensate for the foot height above the paper plane.
"""
from __future__ import annotations
import cv2
import numpy as np
from .ransac import order_quad

def rectify(image,quad,px_per_mm=2.0):
    """Warp only flat PAPER plane, not above-floor foot surfaces."""
    q=order_quad(quad).astype(np.float32)
    lens=np.linalg.norm(q-np.roll(q,-1,axis=0),axis=1)
    horizontal=float((lens[0]+lens[2])/2)
    vertical=float((lens[1]+lens[3])/2)
    if horizontal>=vertical:
        dimensions=(297.,210.)
    else: dimensions=(210.,297.)
    pw,ph=dimensions
    W,H=int(round(pw*px_per_mm)),int(round(ph*px_per_mm))
    dst=np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32)
    Hmat=cv2.getPerspectiveTransform(q,dst)
    return cv2.warpPerspective(image,Hmat,(W,H)),Hmat,dimensions
