"""L02 Image Processing: image resize / interpolation."""
from __future__ import annotations
import cv2
import numpy as np

def resize_image(image, longest=950):
    h,w=image.shape[:2]
    if max(h,w)<=longest:return image.copy()
    s=longest/max(h,w)
    return cv2.resize(image, (round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)
