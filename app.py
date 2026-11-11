"""FootFrame V1: minimal static web UI and 2D OpenCV measurement API.

Python standard library HTTP server; photos are decoded in memory only.
"""
from __future__ import annotations
import base64
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import traceback
import cv2
import numpy as np
from core.measurement_2d import analyze_batch

MAX_BYTES=34_000_000
MAX_IMAGE_BYTES=12_000_000
MAX_PER_KIND=4
HOME=Path(__file__).parent

# User-facing layout is separated from the CV analysis code in web/.
WEB = HOME / 'web'
STATIC_FILES = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/web/styles.css': ('styles.css', 'text/css; charset=utf-8'),
    '/web/app.js': ('app.js', 'application/javascript; charset=utf-8'),
    '/web/favicon.svg': ('favicon.svg', 'image/svg+xml'),
}



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
    return {'version':'footframe-v1-cv-only','frames':frames,'fusion':res,'disclaimer':'Paper-plane projected dimensions only; no 3D height/lens compensation. this is an experimental planar estimator, not clinical sizing.'}


class RequestHandler(BaseHTTPRequestHandler):
    def send_data(self,status,content,ctype):
        body=content if isinstance(content,bytes) else content.encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type',ctype)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self' data: blob:; connect-src 'self'; form-action 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[self.path]
            self.send_data(200, (WEB / filename).read_bytes(), content_type)
        elif self.path == '/health':
            self.send_data(200, json.dumps({'status':'ok','version':'footframe-v1-cv-only'}), 'application/json; charset=utf-8')
        else:
            self.send_data(404, 'Not found', 'text/plain; charset=utf-8')
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
    print(f'FootFrame Version 1 - OpenCV only: http://{host}:{port}',flush=True)
    ThreadingHTTPServer((host,port),RequestHandler).serve_forever()
