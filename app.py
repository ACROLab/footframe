"""FootFrame V1 - 2D OpenCV API, plus real analysis-stage progress polling.

The original POST /analyze endpoint remains available for backwards compatibility.
The web UI uses POST /api/jobs -> GET /api/jobs/<id> -> GET /api/jobs/<id>/result.
Photos and results are kept in temporary process memory, never written to disk.
"""
from __future__ import annotations
import base64
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback
from uuid import uuid4

import cv2
import numpy as np
from core.measurement_2d import analyze_batch

MAX_BYTES = 34_000_000
MAX_IMAGE_BYTES = 12_000_000
MAX_PER_KIND = 4
HOME = Path(__file__).parent
WEB = HOME / 'web'
STATIC_FILES = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/index.html': ('index.html', 'text/html; charset=utf-8'),
    '/web/styles.css': ('styles.css', 'text/css; charset=utf-8'),
    '/web/app.js': ('app.js', 'application/javascript; charset=utf-8'),
    '/web/favicon.svg': ('favicon.svg', 'image/svg+xml'),
    '/web/foot_measurement.webp': ('foot_measurement.webp', 'image/webp'),
}

# The deployment may restart between requests; jobs in progress then become unavailable.
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 15 * 60
MAX_ACTIVE_JOBS = 3
MAX_RETAINED_JOBS = 20
JOB_PATH = re.compile(r'^/api/jobs/([0-9a-f]{32})(/result)?$')


def parse_uploads(request):
    clen = int(request.headers.get('Content-Length', '0'))
    if not 0 < clen < MAX_BYTES:
        raise ValueError('업로드 용량은 총 34MB 미만이어야 합니다')
    ctype = request.headers.get('Content-Type', '')
    if 'multipart/form-data' not in ctype or 'boundary=' not in ctype:
        raise ValueError('multipart/form-data 형식이 필요합니다')
    raw = request.rfile.read(clen)
    header = ('Content-Type: ' + ctype + '\r\nMIME-Version: 1.0\r\n\r\n').encode('ascii')
    msg = BytesParser(policy=default).parsebytes(header + raw)
    parts = {'top': [], 'side': []}
    for part in msg.iter_parts():
        field = part.get_param('name', header='content-disposition')
        if field in parts:
            if len(parts[field]) >= MAX_PER_KIND:
                raise ValueError('각 종류별 최대 4장입니다')
            data = part.get_payload(decode=True) or b''
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError('개별 사진 파일이 너무 큽니다')
            parts[field].append(data)
    if not (1 <= len(parts['top']) <= MAX_PER_KIND and 1 <= len(parts['side']) <= MAX_PER_KIND):
        raise ValueError('위쪽 및 측면 사진을 각각 1~4장 올려 주세요')
    return parts


def process_uploads(top_bytes, side_bytes, progress=None):
    """Run the original V1 measurement pipeline. Optional progress(stage,index,total,kind)."""
    def decode(raw):
        if not 100 < len(raw) <= MAX_IMAGE_BYTES:
            raise ValueError('개별 사진 파일이 비어 있거나 너무 큽니다')
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None or min(im.shape[:2]) < 120:
            raise ValueError('읽을 수 없는 이미지 또는 너무 낮은 해상도입니다')
        if max(im.shape[:2]) > 9000:
            raise ValueError('너무 큰 이미지입니다')
        return im
    if not (1 <= len(top_bytes) <= MAX_PER_KIND and 1 <= len(side_bytes) <= MAX_PER_KIND):
        raise ValueError('위쪽 및 측면 사진을 각각 1~4장 올려 주세요')
    total = len(top_bytes) + len(side_bytes)
    tops, sides = [], []
    for index, raw in enumerate(top_bytes):
        if progress: progress('decode', index, total, 'top')
        tops.append(decode(raw))
    for index, raw in enumerate(side_bytes, start=len(top_bytes)):
        if progress: progress('decode', index, total, 'side')
        sides.append(decode(raw))

    items, result = analyze_batch(tops, sides, progress=progress)
    if progress: progress('overlays', total, total, None)
    frames = []
    for f in items:
        record = f.json()
        if f.image is not None:
            im = f.image
            h, w = im.shape[:2]
            if max(h, w) > 900:
                scale = 900 / max(h, w)
                im = cv2.resize(im, (round(w * scale), round(h * scale)))
            ok, jpeg = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 69])
            if ok:
                record['overlay'] = base64.b64encode(jpeg.tobytes()).decode('ascii')
        frames.append(record)
    return {
        'version': 'footframe-v1-cv-only',
        'frames': frames,
        'fusion': result,
        'disclaimer': 'A4 paper-plane projected dimensions only; height and lens distortion are not corrected.',
    }


def clean_jobs():
    now = time.monotonic()
    for key, job in list(JOBS.items()):
        if job['status'] in ('finished', 'failed') and now - job['created'] > JOB_TTL_SECONDS:
            del JOBS[key]
    completed = sorted(
        ((key, job) for key, job in JOBS.items() if job['status'] in ('finished', 'failed')),
        key=lambda kv: kv[1]['created'],
    )
    for key, _ in completed[:-MAX_RETAINED_JOBS]:
        JOBS.pop(key, None)


def run_job(job_id, top_bytes, side_bytes):
    def update(stage, index=0, total=0, kind=None):
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is None: return
            job['status'] = 'running'
            job['stage'] = stage
            job['completed'] = 0 if stage == 'decode' else index
            job['total'] = total
            job['kind'] = kind
    try:
        update('decode', 0, len(top_bytes) + len(side_bytes), None)
        result = process_uploads(top_bytes, side_bytes, progress=update)
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(status='finished', stage='finished',
                           completed=job['total'], kind=None, result=result)
    except (ValueError, cv2.error) as exc:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(status='failed', error=str(exc))
    except Exception:
        traceback.print_exc()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job:
                job.update(status='failed', error='서버 내부 처리 오류가 발생했습니다. 다시 시도해 주세요.')


class RequestHandler(BaseHTTPRequestHandler):
    def send_data(self, status, content, ctype):
        body = content if isinstance(content, bytes) else content.encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy',
            "default-src 'none'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data: blob:; connect-src 'self'; form-action 'self'; "
            "base-uri 'none'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def json_data(self, status, data):
        self.send_data(status, json.dumps(data, ensure_ascii=False, allow_nan=False),
                       'application/json; charset=utf-8')

    def do_GET(self):
        if self.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[self.path]
            return self.send_data(200, (WEB / filename).read_bytes(), content_type)
        if self.path == '/health':
            return self.json_data(200, {'status': 'ok', 'version': 'footframe-v1-cv-only'})
        match = JOB_PATH.fullmatch(self.path)
        if not match:
            return self.send_data(404, 'Not found', 'text/plain; charset=utf-8')
        job_id, is_result = match.group(1), match.group(2)
        with JOBS_LOCK:
            clean_jobs()
            job = JOBS.get(job_id)
            if job is None:
                status, response = 404, {'error':'분석 작업을 찾을 수 없습니다. 다시 시작해 주세요.'}
            elif is_result:
                if job['status'] == 'finished':
                    status, response = 200, job['result']
                    del JOBS[job_id]
                elif job['status'] == 'failed':
                    status, response = 400, {'error':job['error']}
                    del JOBS[job_id]
                else:
                    status, response = 409, {'error':'아직 분석 중입니다.'}
            else:
                status = 200
                response = {key: job[key] for key in
                            ('status', 'stage', 'completed', 'total', 'kind')}
                if job['status'] == 'failed':
                    response['error'] = job.get('error', '분석 실패')
        return self.json_data(status, response)

    def do_POST(self):
        if self.path not in ('/analyze', '/api/jobs'):
            return self.send_data(404, 'Not found', 'text/plain; charset=utf-8')
        try:
            parts = parse_uploads(self)
            if self.path == '/analyze':
                return self.json_data(200, process_uploads(parts['top'], parts['side']))
            with JOBS_LOCK:
                clean_jobs()
                active = sum(j['status'] in ('queued', 'running') for j in JOBS.values())
                if active >= MAX_ACTIVE_JOBS:
                    return self.json_data(429, {'error':'현재 분석 요청이 많습니다. 잠시 후 다시 시도해 주세요.'})
                job_id = uuid4().hex
                JOBS[job_id] = {
                    'created':time.monotonic(), 'status':'queued', 'stage':'queued',
                    'completed':0, 'total':len(parts['top'])+len(parts['side']),
                    'kind':None,
                }
            thread = threading.Thread(
                target=run_job, args=(job_id, parts['top'], parts['side']),
                daemon=True, name='footframe-analysis',
            )
            thread.start()
            return self.json_data(202, {'job_id':job_id})
        except (ValueError, cv2.error) as exc:
            return self.json_data(400, {'error':str(exc)})
        except Exception:
            traceback.print_exc()
            return self.json_data(500, {'error':'서버 내부 처리 오류가 발생했습니다. 서버 로그를 확인해 주세요.'})


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    print(f'FootFrame Version 1 - OpenCV with progress: http://{host}:{port}', flush=True)
    ThreadingHTTPServer((host, port), RequestHandler).serve_forever()
