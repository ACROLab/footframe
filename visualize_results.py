"""Reproducible per-image visual audit, without any ground truth used for inference.

python visualize_results.py --samples ../../samples --out ../../reports/v1_visual_audit
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from core.image_processing import resize_image
from core.image_geometry import detect_paper, DetectionFailure
from core.homography import rectify
from core.measurement_2d import analyze_frame
from core.color_segmentation import foot_mask


def draw(path, kind, out):
    original = resize_image(cv2.imread(str(path)))
    analysis = analyze_frame(original, kind, path.name)
    scale = 2.0  # only A4 metric rectification px/mm, not a foot-size calibration
    out.mkdir(parents=True, exist_ok=True)
    source=analysis.image
    cv2.imwrite(str(out/(path.stem+'_source.jpg')),source)
    details={'filename':path.name, **analysis.json()}
    if analysis.paper_corners is None:
        return details
    paper=np.asarray(analysis.paper_corners,dtype=np.float32)
    rect,_,_=rectify(original,paper,px_per_mm=scale)
    mask=foot_mask(rect)
    overlay=rect.copy()
    overlay[mask>0]=cv2.addWeighted(overlay[mask>0],0.60,np.broadcast_to(np.array([60,210,70],dtype=np.uint8),overlay[mask>0].shape),.40,0)
    h,w=mask.shape
    cv2.rectangle(overlay,(0,0),(w-1,h-1),(0,220,220),2)
    if kind=='side':
        yy,xx=np.where(mask>0)
        good=(yy>h*.30)&(yy<h*.91)
        occupancy=np.bincount(xx[good],minlength=w)
        supported=np.where(occupancy>=6)[0]
        if len(supported):
            for x in [int(supported[0]),int(supported[-1])]:
                cv2.line(overlay,(x,0),(x,h-1),(20,20,255),2)
    else:
        rows=[]
        for y in range(int(h*.14),int(h*.63)):
            xx=np.where(mask[y]>0)[0]
            if len(xx)>=.20*w and xx.min()>=3 and xx.max()<=w-4:
                rows.append((float(xx[-1]-xx[0]),y,int(xx.min()),int(xx.max())))
        if rows:
            length,y,left,right=min(rows,key=lambda r:abs(r[0]-np.percentile([v[0] for v in rows],95)))
            cv2.line(overlay,(left,y),(right,y),(20,20,255),3)
            cv2.circle(overlay,(left,y),5,(255,0,255),-1)
            cv2.circle(overlay,(right,y),5,(255,0,255),-1)
    status=f'{kind}: '+(f'{analysis.mm:.2f} mm' if analysis.mm is not None else f'REJECT: {analysis.reason}')
    cv2.putText(overlay,status[:70],(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
    cv2.imwrite(str(out/(path.stem+'_rectified.jpg')),overlay)
    cv2.imwrite(str(out/(path.stem+'_mask.png')),mask)
    return details

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    frames=[]
    for kind in ('top','side'):
        for path in sorted((args.samples/kind).glob('*.jpg')):
            frames.append(draw(path,kind,args.out))
    (args.out/'frame_diagnostics.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'Created {len(frames)} per-image visual audits in {args.out}')
