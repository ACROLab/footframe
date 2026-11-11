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
  try {
    const fd = new FormData();
    for (const kind of kinds) for (const file of selections[kind]) fd.append(kind, file, file.name);
    const response = await fetch('/analyze', {method:'POST', body:fd});
    let result;
    try { result = await response.json(); }
    catch { throw new Error('서버 응답을 읽지 못했습니다. 다시 시도해 주세요.'); }
    if (!response.ok) throw new Error(result.error || '분석 중 문제가 발생했습니다.');
    showMetric('length',result.fusion?.length);
    showMetric('width',result.fusion?.width);
    populateDiagnostics(result.frames || []);
    document.getElementById('diagnostics').open = false;
    resultSection.hidden = false;
    resultSection.scrollIntoView({behavior:'smooth',block:'start'});
  } catch (err) {
    showError(err instanceof Error ? err.message : '사진을 분석할 수 없습니다. 다시 시도해 주세요.');
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
