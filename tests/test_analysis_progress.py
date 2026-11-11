"""V1 progress endpoint and UI wiring regression checks."""
from pathlib import Path
import time
from uuid import uuid4
import app


def test_progress_interface_is_real_stage_based():
    html = (app.WEB / 'index.html').read_text(encoding='utf8')
    js = (app.WEB / 'app.js').read_text(encoding='utf8')
    assert 'id="analysis-overlay"' in html
    assert 'data-step="homography"' in html
    assert 'data-step="paper"' in html
    assert "fetch('/api/jobs'" in js
    assert '/api/jobs/${id}' in js
    assert '/api/jobs/${id}/result' in js
    assert 'completed/total' in js
    assert 'setInterval' not in js  # The UI does not fake stage transitions over time.
    assert (app.WEB / 'foot_measurement.webp').stat().st_size > 1000


def test_background_job_records_actual_stage_changes(monkeypatch):
    jobid = uuid4().hex
    stages = []
    with app.JOBS_LOCK:
        app.JOBS[jobid] = {
            'created': time.monotonic(), 'status': 'queued', 'stage': 'queued',
            'completed': 0, 'total': 2, 'kind': None,
        }

    def mocked_analyzer(top_bytes, side_bytes, progress=None):
        for stage, count, kind in [('preprocess', 0, 'top'), ('paper', 0, 'top'),
                                   ('photo_done', 1, 'top'), ('fusion', 2, None)]:
            stages.append(stage)
            progress(stage, count, 2, kind)
        return {'fusion': {}, 'frames': []}

    monkeypatch.setattr(app, 'process_uploads', mocked_analyzer)
    try:
        app.run_job(jobid, [b'fake'], [b'fake'])
        with app.JOBS_LOCK:
            job = app.JOBS[jobid]
            assert job['status'] == 'finished'
            assert job['completed'] == 2
            assert job['stage'] == 'finished'
            assert job['result'] == {'fusion': {}, 'frames': []}
        assert stages == ['preprocess', 'paper', 'photo_done', 'fusion']
    finally:
        with app.JOBS_LOCK:
            app.JOBS.pop(jobid, None)
