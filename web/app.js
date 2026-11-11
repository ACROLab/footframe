'use strict';
const MIN_FILES = 2;
const MAX_FILES = 4;
const MAX_IMAGE_BYTES = 12_000_000;
const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);
const kinds = ['top', 'side'];
const form = document.getElementById('measure-form');
const button = document.getElementById('analyze-button');
const errorBox = document.getElementById('form-error');
const resultSection = document.getElementById('results');
const selections = {top: [], side: []};
const previewUrls = {top: [], side: []};
let isSubmitting = false;

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function clearPreviews(kind) {
  for (const url of previewUrls[kind]) URL.revokeObjectURL(url);
  previewUrls[kind] = [];
  document.getElementById(`${kind}-preview`).replaceChildren();
}

function updateSelection(kind, files) {
  const input = document.getElementById(`${kind}-input`);
  let fresh = Array.from(files);
  let problem = '';
  if (fresh.length > MAX_FILES) {
    problem = '각 방향의 사진은 최대 4장까지 선택할 수 있습니다.';
  } else if (fresh.some(file => !ACCEPTED_TYPES.has(file.type))) {
    problem = 'JPG, PNG, WEBP 형식의 이미지만 사용할 수 있습니다.';
  } else if (fresh.some(file => file.size > MAX_IMAGE_BYTES || file.size === 0)) {
    problem = '사진 한 장의 크기는 12MB 이하여야 합니다.';
  }
  if (problem) {
    input.value = '';
    fresh = [];
  }
  showError(problem);
  selections[kind] = fresh;
  clearPreviews(kind);
  const status = document.getElementById(`${kind}-status`);
  const counter = document.getElementById(`${kind}-counter`);
  const preview = document.getElementById(`${kind}-preview`);
  status.textContent = fresh.length ? `${fresh.length}장 선택됨` : '선택된 사진 없음';
  status.classList.toggle('has-files', fresh.length >= MIN_FILES);
  counter.textContent = `${fresh.length} / ${MAX_FILES}`;
  for (const file of fresh) {
    const item = document.createElement('div');
    item.className = 'preview-item';
    const image = new Image();
    const url = URL.createObjectURL(file);
    previewUrls[kind].push(url);
    image.src = url;
    image.alt = '';
    const name = document.createElement('span');
    name.textContent = file.name;
    name.title = file.name;
    item.append(image, name);
    preview.append(item);
  }
  button.disabled = isSubmitting || kinds.some(k => selections[k].length < MIN_FILES);
}

for (const kind of kinds) {
  const input = document.getElementById(`${kind}-input`);
  const zone = document.getElementById(`${kind}-zone`);
  input.addEventListener('change', () => updateSelection(kind, input.files));
  zone.addEventListener('dragenter', e => {e.preventDefault(); zone.classList.add('is-dragging');});
  zone.addEventListener('dragover', e => {e.preventDefault(); zone.classList.add('is-dragging');});
  zone.addEventListener('dragleave', e => {if (!zone.contains(e.relatedTarget)) zone.classList.remove('is-dragging');});
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('is-dragging');
    if (!e.dataTransfer) return;
    const files = Array.from(e.dataTransfer.files);
    updateSelection(kind, files);
  });
}

const REASONS = {
  ok: '분석 완료',
  no_a4: 'A4 용지를 찾지 못했습니다',
  paper_not_found: 'A4 용지를 찾지 못했습니다',
  insufficient_valid_views: '유효 사진이 부족합니다',
  two_views_disagree: '사진 간 측정값 차이가 큽니다',
  shallow_oblique_view_not_a_side_profile: '측면 촬영 각도가 적절하지 않습니다',
  inferred_paper_corner_diagnostic_only: 'A4 일부가 가려져 최종 계산에서 제외했습니다',
  projected_forefoot_width: '발볼 측정 완료',
  floor_plane_projected_extent_not_physical_length: '발 길이 추정 완료',
  paper_orientation_not_suitable_for_top: '위쪽 촬영 방향이 적절하지 않습니다',
  paper_orientation_not_suitable_for_side: '측면 촬영 방향이 적절하지 않습니다',
  forefoot_not_separable: '발볼 윤곽을 충분히 찾지 못했습니다',
  insufficient_visible_foot: '발 윤곽이 충분히 보이지 않습니다',
  insufficient_endpoint_support: '발 양 끝 위치를 확인할 수 없습니다',
  toe_or_heel_at_paper_border: '발의 끝부분이 A4 경계에 닿아 있습니다',
  width_out_of_physical_range: '발볼 계산값이 허용 범위를 벗어났습니다',
  length_out_of_physical_range: '발 길이 계산값이 허용 범위를 벗어났습니다',
};
function reasonText(reason) {
  if (!reason) return '측정 보류';
  return REASONS[reason] || (reason === 'accepted' ? '분석 완료' : '측정 조건을 충족하지 못했습니다');
}
function valueText(number) {
  return typeof number === 'number' && Number.isFinite(number)
    ? Number(number.toFixed(2)).toLocaleString('ko-KR', {maximumFractionDigits:2})
    : '—';
}

function showMetric(key, fusionItem) {
  const value = document.getElementById(`${key}-value`);
  const unit = document.getElementById(`${key}-unit`);
  const detail = document.getElementById(`${key}-detail`);
  const numeric = fusionItem?.accepted_mm;
  const ok = typeof numeric === 'number' && Number.isFinite(numeric);
  value.textContent = valueText(numeric);
  unit.hidden = !ok;
  const n = Array.isArray(fusionItem?.used_views) ? fusionItem.used_views.length : 0;
  detail.textContent = ok ? `${n}장의 유효 사진을 통합한 결과` : '측정 보류 · 해당 방향의 사진을 다시 촬영해 주세요';
}
function populateDiagnostics(frames) {
  const grid = document.getElementById('diagnostic-grid');
  grid.replaceChildren();
  document.getElementById('detail-count').textContent = `${frames.length}장`;
  for (const f of frames) {
    const card = document.createElement('article');
    card.className = 'diagnostic-item';
    if (f.overlay) {
      const image = new Image();
      image.src = `data:image/jpeg;base64,${f.overlay}`;
      image.alt = `${f.view_id} A4 경계 검출`; 
      image.loading = 'lazy';
      card.append(image);
    }
    const head = document.createElement('div');
    head.className = 'diagnostic-title';
    const title = document.createElement('span');
    title.textContent = f.view_id || '사진';
    const state = document.createElement('span');
    state.className = `diagnostic-label${f.accepted ? '' : ' rejected'}`;
    state.textContent = f.accepted ? '유효' : '제외';
    head.append(title,state);
    const desc = document.createElement('p');
    desc.className = 'diagnostic-desc';
    const size = typeof f.mm === 'number' && Number.isFinite(f.mm) ? ` · ${valueText(f.mm)} mm` : '';
    desc.textContent = `${reasonText(f.reason)}${size}`;
    card.append(head,desc);
    grid.append(card);
  }
}
// Server stage updates are authoritative. There is no fake countdown or time-based percentage.
const analysisOverlay = document.getElementById('analysis-overlay');
const analysisPanel = analysisOverlay.querySelector('.analysis-panel');
const analysisDescription = document.getElementById('analysis-description');
const analysisPhotoLabel = document.getElementById('analysis-photo-label');
const analysisCount = document.getElementById('analysis-count');
const analysisProgress = document.getElementById('analysis-progress');
const analysisProgressFill = document.getElementById('analysis-progress-fill');
const stepNodes = Array.from(document.querySelectorAll('#analysis-steps li'));
const stepOrder = ['preprocess','paper','homography','segmentation','fusion'];
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const stageDescriptions = {
  queued: '사진을 분석할 준비를 하고 있습니다.',
  decode: '사진 파일을 확인하고 영상 데이터로 변환하고 있습니다.',
  preprocess: '이미지 크기와 밝기 정보를 정리하고 있습니다.',
  paper: '흰색 영역과 직선을 검사해 A4 용지의 모서리를 찾고 있습니다.',
  homography: 'A4의 네 모서리를 기준으로 원근을 보정하고 있습니다.',
  segmentation: 'HSV와 Lab 색상 정보로 발의 윤곽을 분리하고 있습니다.',
  measurement: '발 윤곽에서 발 길이 또는 발볼을 계산하고 있습니다.',
  photo_done: '현재 사진의 분석이 완료되어 다음 사진을 준비하고 있습니다.',
  fusion: '사진별 측정값의 차이를 확인하고 이상치를 걸러내고 있습니다.',
  overlays: '분석 결과와 A4 검출 이미지를 정리하고 있습니다.',
  finished: '사진 분석이 끝났습니다. 결과를 불러오고 있습니다.',
};
function showAnalysisOverlay() {
  analysisOverlay.hidden = false;
  document.body.classList.add('is-analyzing');
  updateAnalysisProgress({stage:'queued',completed:0,total:selections.top.length+selections.side.length,kind:null});
  analysisPanel.focus({preventScroll:true});
}
function hideAnalysisOverlay() {
  analysisOverlay.hidden = true;
  document.body.classList.remove('is-analyzing');
}
function updateAnalysisProgress(data) {
  const total = Math.max(0, Number(data.total) || 0);
  const completed = Math.min(total, Math.max(0, Number(data.completed) || 0));
  const stage = String(data.stage || 'queued');
  const isFinal = ['fusion','overlays','finished'].includes(stage);
  const currentPhoto = isFinal ? total : Math.min(total,completed+1);
  const label = data.kind === 'top' ? '위쪽 사진' : data.kind === 'side' ? '측면 사진' : '전체 사진';
  analysisPhotoLabel.textContent = isFinal ? '사진별 분석 완료 · 결과 통합 중' :
    stage === 'queued' || stage === 'decode' ? '사진 불러오는 중' : `${label} · ${currentPhoto}번째 사진`;
  analysisCount.textContent = `${completed} / ${total}장 완료`;
  analysisProgress.setAttribute('aria-valuenow', String(total ? Math.round(100*completed/total) : 0));
  analysisProgressFill.style.width = `${total ? 100*completed/total : 0}%`;
  analysisDescription.textContent = stageDescriptions[stage] || '사진을 분석하고 있습니다.';
  const mapped = ['queued','decode','preprocess'].includes(stage) ? 'preprocess' :
    ['segmentation','measurement','photo_done'].includes(stage) ? 'segmentation' :
    ['fusion','overlays','finished'].includes(stage) ? 'fusion' : stage;
  const currentIndex = stepOrder.indexOf(mapped);
  for (const item of stepNodes) {
    const index = stepOrder.indexOf(item.dataset.step);
    item.classList.toggle('is-active',index===currentIndex);
    item.classList.toggle('is-done',index>=0 && index<currentIndex);
  }
}
async function readResponseJson(response) {
  try { return await response.json(); }
  catch { throw new Error('서버의 응답을 읽지 못했습니다. 다시 시도해 주세요.'); }
}
async function requestAnalysis(files) {
  const response = await fetch('/api/jobs', {method:'POST', body:files});
  const data = await readResponseJson(response);
  if (!response.ok) throw new Error(data.error || '분석 요청을 시작하지 못했습니다.');
  if (!/^[0-9a-f]{32}$/.test(data.job_id || '')) throw new Error('분석 작업 번호를 받지 못했습니다.');
  const id = data.job_id;
  let pollingErrors = 0;
  while (true) {
    let status;
    try {
      const stepResponse = await fetch(`/api/jobs/${id}`, {cache:'no-store'});
      status = await readResponseJson(stepResponse);
      if (!stepResponse.ok) throw new Error(status.error || '분석 상태를 확인할 수 없습니다.');
      pollingErrors = 0;
    } catch (err) {
      if (++pollingErrors >= 5) throw err;
      analysisDescription.textContent = '연결 상태를 확인하고 있습니다. 분석 결과를 다시 요청합니다.';
      await wait(1200);
      continue;
    }
    updateAnalysisProgress(status);
    if (status.status === 'failed') throw new Error(status.error || '분석 중 오류가 발생했습니다.');
    if (status.status === 'finished') {
      const resultResponse = await fetch(`/api/jobs/${id}/result`, {cache:'no-store'});
      const result = await readResponseJson(resultResponse);
      if (!resultResponse.ok) throw new Error(result.error || '결과를 불러오지 못했습니다.');
      return result;
    }
    await wait(650);
  }
}

function setBusy(busy) {
  isSubmitting = busy;
  button.disabled = busy || kinds.some(k => selections[k].length < MIN_FILES);
  button.querySelector('.button-label').textContent = busy ? '사진 분석 중…' : '발 치수 분석하기';
  button.setAttribute('aria-busy', String(busy));
}

form.addEventListener('submit', async e => {
  e.preventDefault();
  if (isSubmitting) return;
  if (kinds.some(k => selections[k].length < MIN_FILES)) {
    showError('위쪽과 측면 사진을 각각 2장 이상 선택해 주세요.');
    return;
  }
  const totalBytes = kinds.reduce((sum, k) => sum + selections[k].reduce((n, file) => n + file.size, 0), 0);
  if (totalBytes > 33_000_000) {
    showError('전체 사진의 용량이 너무 큽니다. 사진 수를 줄이거나 파일 크기를 줄여 주세요.');
    return;
  }
  showError('');
  setBusy(true);
  showAnalysisOverlay();
  try {
    const fd = new FormData();
    for (const kind of kinds) for (const file of selections[kind]) fd.append(kind, file, file.name);
    const result = await requestAnalysis(fd);
    showMetric('length',result.fusion?.length);
    showMetric('width',result.fusion?.width);
    populateDiagnostics(result.frames || []);
    document.getElementById('diagnostics').open = false;
    resultSection.hidden = false;
    hideAnalysisOverlay();
    resultSection.scrollIntoView({behavior:'smooth',block:'start'});
  } catch (err) {
    hideAnalysisOverlay();
    showError(err instanceof Error ? err.message : '사진을 분석할 수 없습니다. 다시 시도해 주세요.');
    errorBox.scrollIntoView({behavior:'smooth',block:'nearest'});
  } finally { setBusy(false); }
});

document.getElementById('reset-button').addEventListener('click', () => {
  form.reset();
  for (const kind of kinds) updateSelection(kind, []);
  resultSection.hidden = true;
  document.getElementById('diagnostics').open = false;
  showError('');
  document.getElementById('measure').scrollIntoView({behavior:'smooth',block:'start'});
});
