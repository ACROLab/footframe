"""Synthetic-only, independent corner-localization benchmark (no foot-size truth).

Recreates the deterministic white-A4/occluder/floor scene from tests/test_v33.py.
This script tests v33 in isolation, not a 3D model or real-foot accuracy.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from core.geometry import detect_paper


def build_scene(seed=12):
    h,w=680,880
    rng=np.random.default_rng(seed)
    floor=np.full((h,w,3),160,np.float32)
    noise=cv2.GaussianBlur(rng.normal(0,9,(h,w)).astype(np.float32),(5,5),1)
    image=np.uint8(np.clip(floor+noise[:,:,None],0,255))
    paper=np.array([[270,175],[615,192],[604,442],[250,414]],np.int32)
    cv2.fillConvexPoly(image,paper,(234,236,237))
    cv2.ellipse(image,(499,305),(164,66),-1,0,360,(126,151,189),-1)
    cv2.rectangle(image,(555,180),(700,276),(39,38,43),-1)
    return image,paper


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('synthetic_v33'))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    im,truth=build_scene()
    pred,info=detect_paper(im)
    error=np.linalg.norm(pred-truth,axis=1)
    overlay=im.copy()
    cv2.polylines(overlay,[truth.astype(np.int32)],True,(0,255,0),3)
    cv2.polylines(overlay,[np.rint(pred).astype(np.int32)],True,(255,0,255),2)
    cv2.imwrite(str(args.out/'synthetic_occluded_a4.jpg'),overlay)
    report={'case':'single deterministic 2D artificial paper/occluder',
            'ground_truth_corners':truth.tolist(),'detected_corners':pred.round(2).tolist(),
            'corner_error_px':error.round(3).tolist(),
            'mean_corner_error_px':round(float(error.mean()),3),
            'paper_method':info['method'],
            'scope':'Synthetic A4 corner error only, not real-foot accuracy'}
    (args.out/'synthetic_corner_error.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
