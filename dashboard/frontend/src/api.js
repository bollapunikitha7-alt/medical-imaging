const BASE = import.meta.env.VITE_API_URL || '';

async function get(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function postImage(path, file, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const fd = new FormData();
  fd.append('image', file);
  const r = await fetch(`${BASE}${path}${qs ? `?${qs}` : ''}`, { method: 'POST', body: fd });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(t || `${path}: ${r.status}`);
  }
  return r.json();
}

export const api = {
  models: () => get('/api/models'),
  metrics: () => get('/api/metrics'),
  datasets: () => get('/api/dataset-info'),
  scoreDist: (model, method) => get(`/api/score-distribution?model=${model}&method=${method}`),
  predict: (file, model, method) => postImage('/api/predict', file, { model, method }),
  compare: (file, method) => postImage('/api/compare', file, { method }),
  attentionUrl: (file, model) => ({ file, model }),
  alerts: () => get('/api/alerts'),
  ack: async (id) => {
    const r = await fetch(`${BASE}/api/alerts/${encodeURIComponent(id)}/ack`, { method: 'POST' });
    if (!r.ok) throw new Error(`ack: ${r.status}`);
    return r.json();
  },
};
