"""FootFrame rebuilt standalone 2D web app. Python stdlib HTTP + OpenCV.

No Flask dependency: available offline and deployable with `python app.py`.
Uploaded photographs are kept only in memory, not saved to disk.
"""
from __future__ import annotations
import base64
from email.parser import BytesParser
from email.policy import default
import html
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import traceback
import urllib.parse
import cv2
import numpy as np
from core.measurement import analyze_batch

MAX_BYTES=34_000_000
MAX_IMAGE_BYTES=12_000_000
MAX_PER_KIND=4
HOME=Path(__file__).parent

HTML=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FootFrame · 2D 발 치수 분석</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#182338;background:#f6f8fb}
*{box-sizing:border-box}body{margin:0}.nav{background:white;border-bottom:1px solid #e4e8f0;display:flex;align-items:center;justify-content:space-between;padding:23px max(5%,calc((100% - 980px)/2))}
.brand{font-size:24px;font-weight:850;letter-spacing:-1px}.brand em{color:#245dd2;font-style:normal}.tag{font-size:12px;color:#6d7b8d}
main{width:min(900px,92%);margin:50px auto 85px}h1{font-size:clamp(28px,6vw,42px);letter-spacing:-1.6px;margin:8px 0 18px}
.lead{color:#596779;line-height:1.8;margin-bottom:36px;font-size:16px}.eyebrow{font-size:12px;letter-spacing:1.6px;font-weight:800;color:#245dd2}
.card{background:#fff;border:1px solid #e3e8ef;box-shadow:0 10px 40px #172c5610;border-radius:22px;padding:30px;margin:17px 0}
h2{font-size:20px;margin:0 0 12px}p{line-height:1.7}.hint{font-size:13px;color:#66758b}.file{width:100%;padding:19px;background:#f6f9ff;border:1px dashed #90a9d2;border-radius:12px;font-size:15px}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:18px}.note{background:#f1f6ff;border-radius:14px;padding:15px;color:#345071;font-size:13px;line-height:1.7}
button{border:0;background:#235fdf;color:white;font-size:17px;font-weight:800;padding:17px 30px;border-radius:13px;cursor:pointer;width:100%;margin-top:18px}button:disabled{opacity:.55;cursor:wait}
.metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px}.metric{padding:23px;border:1px solid #e1e9f4;border-radius:15px;background:#f6f9ff}
.metric small{display:block;color:#67768d}.metric strong{display:block;font-size:31px;margin-top:8px;letter-spacing:-1px}.error{padding:16px;border-radius:12px;background:#fff3ec;color:#9b4a2c;font-size:14px;line-height:1.6}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}.frame{border:1px solid #e6e9ef;border-radius:13px;padding:12px}.frame img{display:block;width:100%;border-radius:10px;object-fit:contain}.frame h3{font-size:14px}.pill{color:#2461af;font-size:12px;font-weight:700}.bad{color:#ac6a39}.muted{color:#64748b;font-size:13px;line-height:1.6}
footer{color:#8290a4;font-size:12px;text-align:center;padding-bottom:45px}
@media(max-width:650px){.steps,.metrics{grid-template-columns:1fr}.card{padding:22px}main{margin-top:32px}}
</style></head><body><header class="nav"><span class="brand">Foot<em>Frame</em></span><span class="tag">2D Computer Vision</span></header>
<main><div class="eyebrow">AUTOMATIC 2D MEASUREMENT</div><h1>A4 한 장으로 시작하는<br>발 치수 분석</h1><p class="lead">발과 A4를 함께 촬영한 사진을 여러 장 업로드하면, 종이 경계와 발 윤곽을 자동 검출하고 같은 종류의 관측값을 통합합니다. 수동으로 꼭짓점을 찍을 필요가 없습니다.</p>
<div class="note">촬영: A4는 바닥에 그대로 두고, <b>위쪽 사진 2~4장</b>은 거의 수직으로, <b>측면 사진 2~4장</b>은 발 옆에서 찍어 주세요. 사진은 서버 메모리에서 분석하며 저장하지 않습니다.</div>
<form id="form"><div class="steps"><div class="card"><h2>01 · 위쪽 사진</h2><p class="hint">발볼 폭 추정 · A4 네 모서리가 최대한 보이도록</p><input class="file" name="top" id="top" type="file" accept="image/jpeg,image/png,image/webp" multiple required><p class="hint" id="topCount">최대 4장</p></div>
<div class="card"><h2>02 · 측면 사진</h2><p class="hint">발 길이 투영 추정 · A4는 바닥에 그대로</p><input class="file" name="side" id="side" type="file" accept="image/jpeg,image/png,image/webp" multiple required><p class="hint" id="sideCount">최대 4장</p></div></div><button id="go" type="submit">자동 분석 시작</button></form>
<section id="result" style="display:none"><div class="card"><h2>통합 분석 결과</h2><div class="metrics"><div class="metric"><small>발볼 · A4 평면 투영</small><strong id="width">—</strong></div><div class="metric"><small>발 길이 · A4 평면 투영</small><strong id="length">—</strong></div></div><p class="muted" id="summary"></p><div class="error">중요: 길이는 바닥 평면 투영치입니다. 발의 실제 높이·카메라 렌즈·시점으로 인한 시차가 제거되지 않았으므로 <b>실제 신발 사이즈를 확정하는 용도로 사용하면 안 됩니다.</b></div></div>
<div class="card"><h2>사진별 자동 검출 진단</h2><div class="grid" id="frames"></div></div></section>
</main><footer>FootFrame · OpenCV 2D · experimental measurement (v33 algorithm)</footer>
<script>
const form=document.getElementById('form');const btn=document.getElementById('go');
for(const kind of ['top','side']) document.getElementById(kind).addEventListener('change',e=>document.getElementById(kind+'Count').textContent=e.target.files.length+'장 선택');
form.addEventListener('submit',async e=>{e.preventDefault();const tops=document.getElementById('top').files;const sides=document.getElementById('side').files;
if(!tops.length||!sides.length||tops.length>4||sides.length>4){alert('위쪽과 측면 사진을 각각 1~4장 선택해 주세요.');return;}
btn.disabled=true;btn.textContent='사진 분석 중...';
try{const r=await fetch('/analyze',{method:'POST',body:new FormData(form)});const data=await r.json();if(!r.ok)throw Error(data.error||'분석 오류');
const fmt=x=>x==null?'검출 불확실':x.toFixed(1)+' mm';
document.getElementById('width').textContent=fmt(data.fusion.width.accepted_mm);
document.getElementById('length').textContent=fmt(data.fusion.length.accepted_mm);
document.getElementById('summary').textContent=`발볼 사용 사진 ${data.fusion.width.used_views.length}장 · 발 길이 사용 사진 ${data.fusion.length.used_views.length}장. 유효 사진 사이의 차이는 실제 정확도 증거가 아닙니다.`;
const g=document.getElementById('frames');g.replaceChildren();for(const f of data.frames){const el=document.createElement('div');el.className='frame';const h=document.createElement('h3');h.textContent=f.view_id+' · '+(f.mm==null?'치수 계산 보류':(f.accepted?'융합 후보: ':'보조 추정(융합 제외): ')+f.mm.toFixed(1)+' mm');const t=document.createElement('p');t.className=f.accepted?'pill':'pill bad';t.textContent=f.reason;el.append(h);if(f.overlay){const im=new Image();im.src='data:image/jpeg;base64,'+f.overlay;im.alt='A4 자동 경계 진단 이미지';el.append(im)}el.append(t);g.append(el)}
document.getElementById('result').style.display='block';document.getElementById('result').scrollIntoView({behavior:'smooth'});
}catch(e){alert(e.message)}finally{btn.disabled=false;btn.textContent='자동 분석 시작';}
});
</script></body></html>'''


def process_uploads(top_bytes,side_bytes):
    def decode(raw):
        if not 100<len(raw)<=MAX_IMAGE_BYTES:raise ValueError('개별 사진 파일이 비어 있거나 너무 큽니다')
        im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if im is None or min(im.shape[:2])<120:raise ValueError('읽을 수 없는 이미지 또는 너무 낮은 해상도입니다')
        if max(im.shape[:2])>9000:raise ValueError('너무 큰 이미지입니다')
        return im
    if not (1<=len(top_bytes)<=MAX_PER_KIND and 1<=len(side_bytes)<=MAX_PER_KIND):
        raise ValueError('위쪽 및 측면 사진을 각각 1~4장 올려 주세요')
    items,res=analyze_batch([decode(v) for v in top_bytes],[decode(v) for v in side_bytes])
    frames=[]
    for f in items:
        record=f.json()
        if f.image is not None:
            im=f.image
            h,w=im.shape[:2]
            if max(h,w)>900:
                scale=900/max(h,w)
                im=cv2.resize(im,(round(w*scale),round(h*scale)))
            ok,jpeg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,69])
            if ok:record['overlay']=base64.b64encode(jpeg.tobytes()).decode('ascii')
        frames.append(record)
    return {'version':'footframe-v33-cv-only','frames':frames,'fusion':res,'disclaimer':'Paper-plane projected dimensions only; no 3D height/lens compensation. v29 source unavailable; this is a rebuilt estimator.'}


class RequestHandler(BaseHTTPRequestHandler):
    def send_data(self,status,content,ctype):
        body=content if isinstance(content,bytes) else content.encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self' data: 'unsafe-inline'; connect-src 'self'")
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path in ('/','/index.html'):
            self.send_data(200,HTML,'text/html; charset=utf-8')
        elif self.path=='/health':
            self.send_data(200,json.dumps({'status':'ok','version':'footframe-v33-cv-only'}),'application/json; charset=utf-8')
        else:self.send_data(404,'Not found','text/plain')
    def do_POST(self):
        if self.path!='/analyze':return self.send_data(404,'Not found','text/plain')
        try:
            clen=int(self.headers.get('Content-Length','0'))
            if not 0<clen<MAX_BYTES:raise ValueError('업로드 용량은 총 34MB 미만이어야 합니다')
            ctype=self.headers.get('Content-Type','')
            if 'multipart/form-data' not in ctype:raise ValueError('multipart/form-data 형식이 필요합니다')
            raw=self.rfile.read(clen)
            header=('Content-Type: '+ctype+'\r\nMIME-Version: 1.0\r\n\r\n').encode('ascii')
            msg=BytesParser(policy=default).parsebytes(header+raw)
            parts={'top':[],'side':[]}
            for part in msg.iter_parts():
                field=part.get_param('name',header='content-disposition')
                if field in parts:
                    if len(parts[field])>=MAX_PER_KIND:raise ValueError('각 종류별 최대 4장입니다')
                    parts[field].append(part.get_payload(decode=True) or b'')
            result=process_uploads(parts['top'],parts['side'])
            self.send_data(200,json.dumps(result,ensure_ascii=False,allow_nan=False),'application/json; charset=utf-8')
        except (ValueError,cv2.error) as exc:
            self.send_data(400,json.dumps({'error':str(exc)},ensure_ascii=False),'application/json; charset=utf-8')
        except Exception:
            traceback.print_exc()
            self.send_data(500,json.dumps({'error':'서버 내부 처리 오류가 발생했습니다. 서버 로그를 확인해 주세요.'},ensure_ascii=False),'application/json; charset=utf-8')

if __name__=='__main__':
    host=os.getenv('HOST','0.0.0.0');port=int(os.getenv('PORT','8000'))
    print(f'FootFrame CV-only (v33 algorithm): http://{host}:{port}',flush=True)
    ThreadingHTTPServer((host,port),RequestHandler).serve_forever()
