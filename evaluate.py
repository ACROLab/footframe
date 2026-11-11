"""CLI real-photo evaluation. Reference values ONLY affect report comparisons.

Example: python evaluate.py --top photo1.jpg photo2.jpg --side s1.jpg s2.jpg
No user data/foot lengths are encoded into the estimator.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import cv2
from core.measurement_2d import analyze_batch
p=ArgumentParser()
p.add_argument('--top',nargs='+',required=True,help='top view photographs')
p.add_argument('--side',nargs='+',required=True,help='side view photographs')
p.add_argument('--reference-length',type=float,default=None,help='ground truth for report ONLY')
p.add_argument('--reference-width',type=float,default=None,help='ground truth for report ONLY')
p.add_argument('--out',default='evaluation.json')
a=p.parse_args()

def load(paths):
    photos=[]
    for filename in paths:
        im=cv2.imread(str(filename))
        if im is None:raise SystemExit(f'File could not be read: {filename}')
        photos.append(im)
    return photos

frames,fusion=analyze_batch(load(a.top),load(a.side))
report={'version':'footframe-v1-cv-only','frames':[f.json() for f in frames],'fusion':fusion,'reference_validation_only':{'length_mm':a.reference_length,'width_mm':a.reference_width}}
for kind,value in [('length',a.reference_length),('width',a.reference_width)]:
    estimate=fusion[kind]['accepted_mm']
    report['reference_validation_only'][f'{kind}_error_mm']=round(estimate-value,3) if estimate is not None and value is not None else None
Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
for kind in ('length','width'):
    result=fusion[kind]
    print(kind,('%.2f mm' % result['accepted_mm']) if result['accepted_mm'] is not None else 'not enough valid photos',result['status'],result['used_views'])
print('Report:',a.out)
