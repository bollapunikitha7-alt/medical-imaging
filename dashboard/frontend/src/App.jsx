import { useEffect, useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar
} from 'recharts';
import { api } from './api.js';

const MODELS = [
  { id: 'jepa', label: 'JEPA (Self-supervised)' },
  { id: 'mae', label: 'MAE (Self-supervised)' },
  { id: 'supervised', label: 'Supervised (Baseline)' },
];
const METHODS = ['mahalanobis', 'energy'];
const BASE = import.meta.env.VITE_API_URL || '';

function fmt(x, d = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(d);
}

function Decision({ v }) {
  if (!v) return <span className="pill na">—</span>;
  return <span className={`pill ${v === 'ID' ? 'id' : 'ood'}`}>{v}</span>;
}

function useBackend() {
  const [models, setModels] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [datasets, setDatasets] = useState(null);
  const [error, setError] = useState('');
  useEffect(() => {
    (async () => {
      try {
        const [m, met, ds] = await Promise.all([
          api.models().catch(() => null),
          api.metrics().catch(() => null),
          api.datasets().catch(() => null),
        ]);
        setModels(m); setMetrics(met); setDatasets(ds);
      } catch (e) { setError(String(e.message || e)); }
    })();
  }, []);
  return { models, metrics, datasets, error };
}

function Inference({ onActivity }) {
  const [model, setModel] = useState('jepa');
  const [method, setMethod] = useState('mahalanobis');
  const [emethod, setEmethod] = useState('rollout');
  const [etarget, setEtarget] = useState('energy');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [out, setOut] = useState(null);
  const [rows, setRows] = useState(null);
  const [attn, setAttn] = useState('');
  const [dist, setDist] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState('');

  const onFile = (f) => {
    if (!f) return;
    if (preview) URL.revokeObjectURL(preview);
    setFile(f); setOut(null); setRows(null); setAttn(''); setErr('');
    setPreview(URL.createObjectURL(f));
  };

  const clear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview('');
    setOut(null); setRows(null); setAttn(''); setDist(null); setErr('');
    const el = document.getElementById('up');
    if (el) el.value = '';
  };

  const score = async () => {
    if (!file) return;
    setBusy(true); setErr('');
    try {
      const r = await api.predict(file, model, method);
      setOut(r);
      try {
        const d = await api.scoreDist(model, method);
        setDist(d);
      } catch { /* optional */ }
    } catch (e) { setErr(String(e.message || e)); }
    finally { setBusy(false); if (onActivity) onActivity(); }
  };

  const compare = async () => {
    if (!file) return;
    setBusy(true); setErr('');
    try { setRows((await api.compare(file, method)).rows || []); }
    catch (e) { setErr(String(e.message || e)); }
    finally { setBusy(false); if (onActivity) onActivity(); }
  };

  const attention = async () => {
    if (!file) return;
    setBusy(true); setErr('');
    try {
      const fd = new FormData();
      fd.append('image', file);
      const r = await fetch(`${BASE}/api/attention?model=${model}&view=split&method=${emethod}&target=${etarget}`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(await r.text());
      setAttn(await r.json());
    } catch (e) { setErr(String(e.message || e)); }
    finally { setBusy(false); if (onActivity) onActivity(); }
  };

  return (
    <div className="grid c2">
      <div className="card">
        <h3>Single-image OOD scoring</h3>
        <div className="sub">Higher score = more OOD. Threshold = ID 95th percentile (ID-only).</div>
        <div className="row">
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <hr className="sep" />
        <div className="drop" onClick={() => document.getElementById('up').click()}>
          {preview ? <img src={preview} alt="upload" /> : 'Click to upload PNG / JPG (max 10 MB)'}
          <input id="up" type="file" accept=".png,.jpg,.jpeg" hidden
            onChange={(e) => onFile(e.target.files[0])} />
        </div>
        {file && (
          <div className="row" style={{ marginTop: 8 }}>
            <span className="sub" style={{ margin: 0 }}>{file.name}</span>
            <button className="btn ghost" onClick={() => document.getElementById('up').click()}>Re-upload</button>
            <button className="btn ghost" onClick={clear}>Remove</button>
          </div>
        )}
        <hr className="sep" />
        <div className="row">
          <button className="btn" disabled={!file || busy} onClick={score}>Score</button>
          <button className="btn ghost" disabled={!file || busy} onClick={compare}>Compare 3 models</button>
          <button className="btn ghost" disabled={!file || busy} onClick={attention}>Explain</button>
          <select value={emethod} onChange={(e) => setEmethod(e.target.value)} title="Explanation method">
            <option value="rollout">Attention rollout</option>
            <option value="gradcam">Grad-CAM</option>
          </select>
          {emethod === 'gradcam' && (
            <select value={etarget} onChange={(e) => setEtarget(e.target.value)} title="Grad-CAM target">
              <option value="energy">Energy OOD score</option>
              <option value="logit">Predicted logit</option>
            </select>
          )}
        </div>
        {emethod === 'gradcam' && (
          <div className="sub" style={{ marginTop: 6 }}>
            Grad-CAM needs gradients: it explains the Energy score or predicted logit.
            Mahalanobis distance has no gradients and is not explainable this way.
          </div>
        )}
        {err && <><hr className="sep" /><div className="err">{err}</div></>}
      </div>

      <div className="card">
        <h3>Result</h3>
        <div className="sub">Raw score is primary; normalized = P(ID ≤ raw).</div>
        {!out && !rows && <div className="missing">No score yet — upload and press Score.</div>}
        {out && (
          <>
            <div className={`hero ${out.decision === 'ID' ? 'ok' : 'bad'}`}>
              <div><span className="hero-label">Decision</span><span className="hero-dec">{out.decision}</span></div>
              <div><span className="hero-label">OOD score</span><span className="hero-num">{fmt((out.normalized || 0) * 100, 1)}%</span></div>
              <div><span className="hero-label">Raw vs threshold</span><span className="hero-num small">{fmt(out.raw)} / {fmt(out.threshold)}</span></div>
            </div>
            <p className="explain">
              Raw {fmt(out.raw)} {out.decision === 'OOD' ? '≥' : '<'} threshold {fmt(out.threshold)} → <b>{out.decision}</b>.
              Score = % of ID validation images scoring below this one. Higher = more OOD.
            </p>
            <div className="kv">
              <span>Model / method</span><b>{out.model} / {out.method}</b>
              <span>Raw</span><b>{fmt(out.raw)}</b>
              <span>Normalized</span><b>{fmt(out.normalized)}</b>
              <span>Percentile</span><b>{fmt(out.percentile, 1)}%</b>
              <span>Threshold</span><b>{fmt(out.threshold)}</b>
              <span>Decision</span><span><Decision v={out.decision} /></span>
            </div>
            <hr className="sep" />
            <div className="meter"><div style={{ width: `${Math.min(100, (out.normalized || 0) * 100)}%` }} /></div>
          </>
        )}
        {rows && (
          <>
            <hr className="sep" />
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={rows.filter((r) => !r.error).map((r) => ({ name: r.model, score: +(r.normalized * 100).toFixed(1) }))}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => [`${v}%`, 'OOD score']} />
                <Bar dataKey="score" fill="#0e63c9" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <table>
              <thead><tr><th>Model</th><th>Raw</th><th>Norm</th><th>Decision</th></tr></thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td>{r.model}</td>
                    <td>{r.error ? '—' : fmt(r.raw)}</td>
                    <td>{r.error ? r.error : fmt(r.normalized)}</td>
                    <td>{r.error ? '—' : <Decision v={r.decision} />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {attn && (
          <>
            <hr className="sep" />
            <div className="sub">
              Method: <b>{attn.method || 'attention-rollout (not Grad-CAM)'}</b>
              {' · '}{attn.model || ''}{attn.is_gradcam
                ? ` · gradient of ${attn.target || 'energy'} w.r.t. last-block tokens, 14×14 patches, bilinear 224×224`
                : ' · rollout of 12 ViT blocks, 14×14 patches, bilinear 224×224'}
            </div>
            <div className="grid c3">
              {[
                ['Original X-ray', attn.original],
                [`Heatmap — ${attn.is_gradcam ? 'Grad-CAM' : 'attention rollout'}`, attn.heatmap],
                ['Overlay', attn.overlay],
              ].map(([label, src]) => (
                <div key={label}>
                  <div className="sub" style={{ marginBottom: 4 }}><b>{label}</b></div>
                  <img src={src} alt={label} style={{ width: '100%', borderRadius: 10, border: '1px solid var(--line)' }} />
                </div>
              ))}
            </div>
          </>
        )}
        {dist && <div className="sub" style={{ marginTop: 8 }}>ID dist n={dist.id_scores?.length}, threshold {fmt(dist.threshold)}</div>}
      </div>
    </div>
  );
}

function MetricsView({ metrics }) {
  if (!metrics) return <div className="missing">Metrics not loaded — start backend on :8000.</div>;
  const b = metrics.baseline?.data;
  const o = metrics.ood_detection?.data;
  const t = metrics.threshold?.data;
  const r = metrics.robustness?.data;
  const miss = (k) => !metrics[k]?.found;

  const idRows = useMemo(() => {
    if (!b?.methods) return [];
    return Object.entries(b.methods).map(([k, v]) => ({ name: k, ...v.id, best: v.best_epoch }));
  }, [b]);

  const oodRows = useMemo(() => {
    if (!b?.methods) return [];
    const out = [];
    for (const [m, v] of Object.entries(b.methods)) {
      for (const ds of ['Montgomery', 'Shenzhen']) {
        const d = v.ood?.[ds];
        if (d) out.push({ model: m, ds, ...d });
      }
    }
    return out;
  }, [b]);

  const robustChart = useMemo(() => {
    if (!r?.robustness) return [];
    const sev = [0, 1, 2, 3, 4, 5];
    return sev.map((s) => ({
      sev: `L${s}`,
      JEPA: r.robustness.JEPA?.contrast?.[s]?.auc,
      MAE: r.robustness.MAE?.contrast?.[s]?.auc,
      Supervised: r.robustness.Supervised?.contrast?.[s]?.auc,
    }));
  }, [r]);

  return (
    <div className="grid">
      <div className="grid c3">
        <div className="card"><h3>Experiment</h3><div className="sub">{b?.experiment_info?.date || '—'}</div>
          <div className="kv"><span>Train / val</span><b>{b?.experiment_info?.id_train_samples} / {b?.experiment_info?.id_val_samples}</b>
            <span>Montgomery / Shenzhen</span><b>{b?.experiment_info?.ood_montgomery_samples} / {b?.experiment_info?.ood_shenzhen_samples}</b>
            <span>Seed / epochs</span><b>{b?.experiment_info?.seed} / {b?.experiment_info?.epochs}</b></div></div>
        <div className="card"><h3>Files</h3><div className="sub">Honest status — never fake numbers</div>
          <div className="kv">
            <span>baseline</span><b>{metrics.baseline?.found ? 'found' : 'Not computed'}</b>
            <span>ood_detection</span><b>{metrics.ood_detection?.found ? `ID n=${o?.id_samples}` : 'Not computed'}</b>
            <span>robustness</span><b>{metrics.robustness?.found ? 'found' : 'Not computed'}</b>
            <span>threshold</span><b>{metrics.threshold?.found ? 'found' : 'Not computed'}</b>
          </div></div>
        <div className="card"><h3>Reading guide</h3><div className="sub">Higher AUROC = better · Lower FPR@95 = better</div>
          <div className="kv"><span>Energy</span><b>from logits</b><span>Mahalanobis</span><b>embedding distance</b><span>AUPR / OOD-Acc</span><b>Not computed</b></div></div>
      </div>

      <div className="card">
        <h3>What do these numbers mean?</h3>
        <div className="sub">Hover any column header for the one-line definition</div>
        <div className="kv">
          <span><b>AUROC ↑</b></span><b>P(a random OOD image scores higher than a random ID image). 0.5 = chance, 1.0 = perfect.</b>
          <span><b>FPR@95 ↓</b></span><b>% of OOD wrongly accepted when the threshold keeps 95% of ID. Lower = safer.</b>
          <span><b>Energy ↑</b></span><b>−logsumexp of classifier logits. Confident ID → low; unfamiliar → high.</b>
          <span><b>Mahalanobis ↑</b></span><b>Distance to nearest class Gaussian fitted on ID embeddings. Far = OOD.</b>
          <span><b>ECE / NLL ↓</b></span><b>Calibration: does predicted confidence match accuracy? Lower = better calibrated.</b>
          <span><b>Threshold</b></span><b>ID 95th percentile, fitted on ID calibration images only — OOD never touches it.</b>
        </div>
      </div>

      <div className="card">
        <h3>ID classification (5% split)</h3>
        <div className="sub">Frozen encoder + linear probe</div>
        {miss('baseline') ? <div className="missing">Not computed</div> : (
          <>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={idRows.map((x) => ({ name: x.name, AUROC: +x.auroc.toFixed(3), Accuracy: +x.accuracy.toFixed(3) }))}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis domain={[0.7, 1]} tick={{ fontSize: 11 }} />
                <Tooltip /><Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="AUROC" fill="#0e63c9" radius={[6, 6, 0, 0]} />
                <Bar dataKey="Accuracy" fill="#0e9f6e" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <table><thead><tr><th title="Method">Method</th><th title="Area under ROC — higher is better">AUROC</th><th title="Fraction correct">Acc</th><th title="Expected calibration error — lower is better">ECE</th><th title="Negative log-likelihood — lower is better">NLL</th><th title="Epoch with best validation AUROC">Best ep</th></tr></thead>
              <tbody>{idRows.map((x) => <tr key={x.name}><td>{x.name}</td><td>{fmt(x.auroc)}</td><td>{fmt(x.accuracy)}</td><td>{fmt(x.ece)}</td><td>{fmt(x.nll)}</td><td>{x.best}</td></tr>)}</tbody></table>
          </>
        )}
      </div>

      <div className="card">
        <h3>OOD detection (AUROC / FPR@95)</h3>
        <div className="sub">Energy vs Mahalanobis · Montgomery (138) · Shenzhen (662)</div>
        {miss('baseline') ? <div className="missing">Not computed</div> : (
          <>
            <div className="grid c2">
              {['Montgomery', 'Shenzhen'].map((ds) => (
                <div key={ds}>
                  <div className="sub" style={{ marginBottom: 4 }}><b>{ds} — AUROC by method</b></div>
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={oodRows.filter((x) => x.ds === ds).map((x) => ({
                      name: x.model, Energy: +x.energy_auroc.toFixed(3), Mahalanobis: +x.mahal_auroc.toFixed(3),
                    }))}>
                      <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                      <Tooltip /><Legend wrapperStyle={{ fontSize: 12 }} />
                      <Bar dataKey="Energy" fill="#c77414" radius={[6, 6, 0, 0]} />
                      <Bar dataKey="Mahalanobis" fill="#0e63c9" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
            <table><thead><tr><th>Model</th><th>Set</th><th title="Energy AUROC — higher is better">Energy AUC</th><th title="Energy FPR@95 — lower is better">Energy FPR95</th><th title="Mahalanobis AUROC — higher is better">Mahal AUC</th><th title="Mahalanobis FPR@95 — lower is better">Mahal FPR95</th></tr></thead>
              <tbody>{oodRows.map((x, i) => <tr key={i}><td>{x.model}</td><td>{x.ds}</td><td>{fmt(x.energy_auroc)}</td><td>{fmt(x.energy_fpr95)}</td><td>{fmt(x.mahal_auroc)}</td><td>{fmt(x.mahal_fpr95)}</td></tr>)}</tbody></table>
          </>
        )}
      </div>

      <div className="grid c2">
        <div className="card">
          <h3>Robustness — contrast severity 0→5 (AUC)</h3>
          <div className="sub">5% run: MAE holds, JEPA collapses at L5</div>
          {miss('robustness') ? <div className="missing">Not computed</div> : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={robustChart}>
                <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="sev" /><YAxis domain={[0.4, 1]} />
                <Tooltip /><Legend />
                <Line type="monotone" dataKey="JEPA" stroke="#0e63c9" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="MAE" stroke="#0e9f6e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Supervised" stroke="#d64545" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="card">
          <h3>Threshold @95% ID (detection rate)</h3>
          <div className="sub">ID-only thresholds; OOD never used for fitting</div>
          {!t?.results ? <div className="missing">Not computed</div> : (
            <>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={Object.entries(t.results).map(([k, v]) => ({
                  name: k,
                  'Mont-E': +((v.ood?.Montgomery?.energy?.ood_detection_rate || 0) * 100).toFixed(1),
                  'Mont-M': +((v.ood?.Montgomery?.mahal?.ood_detection_rate || 0) * 100).toFixed(1),
                  'Shen-E': +((v.ood?.Shenzhen?.energy?.ood_detection_rate || 0) * 100).toFixed(1),
                  'Shen-M': +((v.ood?.Shenzhen?.mahal?.ood_detection_rate || 0) * 100).toFixed(1),
                }))}>
                  <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 11 }} /><YAxis tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => [`${v}%`, 'detected']} /><Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Mont-E" fill="#c77414" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Mont-M" fill="#0e63c9" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Shen-E" fill="#e6a23c" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="Shen-M" fill="#0e9f6e" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <table><thead><tr><th>Model</th><th>Thr (E / M)</th><th>Mont E/M %</th><th>Shen E/M %</th></tr></thead>
              <tbody>{Object.entries(t.results).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{fmt(v.threshold?.energy, 2)} / {fmt(v.threshold?.mahal, 1)}</td>
                  <td>{fmt((v.ood?.Montgomery?.energy?.ood_detection_rate || 0) * 100, 1)} / {fmt((v.ood?.Montgomery?.mahal?.ood_detection_rate || 0) * 100, 1)}</td>
                  <td>{fmt((v.ood?.Shenzhen?.energy?.ood_detection_rate || 0) * 100, 1)} / {fmt((v.ood?.Shenzhen?.mahal?.ood_detection_rate || 0) * 100, 1)}</td></tr>
              ))}</tbody></table>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function AlertsView() {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState('');
  const load = async () => {
    setErr('');
    try { setItems((await api.alerts()).alerts || []); }
    catch (e) { setErr(String(e.message || e)); setItems(null); }
  };
  useEffect(() => { load(); }, []);
  const ack = async (id) => {
    try { await api.ack(id); await load(); }
    catch (e) { setErr(String(e.message || e)); }
  };
  return (
    <div className="card">
      <h3>Alerts</h3>
      <div className="sub">Backend-generated from real results — OOD crossings, failures, missing checkpoints. No emails or external messages; research flags only, not diagnoses.</div>
      <div className="row"><button className="btn ghost" onClick={load}>Refresh</button></div>
      {err && <><hr className="sep" /><div className="err">{err}</div></>}
      {items === null && !err && <div className="missing">Loading…</div>}
      {items !== null && items.length === 0 && <div className="missing">No active alerts.</div>}
      {items !== null && items.length > 0 && (
        <table><thead><tr><th>Severity</th><th>Message</th><th>Model / method</th><th>Seen</th><th></th></tr></thead>
          <tbody>{items.map((a) => (
            <tr key={a.id}>
              <td><span className={`pill ${a.severity === 'error' ? 'ood' : 'na'}`}>{a.severity}</span></td>
              <td style={{ textAlign: 'left', maxWidth: 420 }}>{a.message}<div className="sub" style={{ margin: '4px 0 0' }}>{a.id} · {a.created_at} · results {a.results}{a.repeat_count > 1 ? ` · ×${a.repeat_count}` : ''}</div></td>
              <td>{a.model || '—'} / {a.method || '—'}</td>
              <td>{(a.last_seen || '').slice(0, 19).replace('T', ' ')}</td>
              <td><button className="btn ghost" onClick={() => ack(a.id)}>Acknowledge</button></td>
            </tr>
          ))}</tbody></table>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState('infer');
  const { models, metrics, datasets, error } = useBackend();
  const demo = models?.demo;
  const [alertCount, setAlertCount] = useState(0);
  const refreshAlerts = async () => {
    try { setAlertCount(((await api.alerts()).alerts || []).length); }
    catch { /* backend offline */ }
  };
  useEffect(() => { refreshAlerts(); }, []);
  useEffect(() => { if (tab === 'alerts') refreshAlerts(); }, [tab]);

  return (
    <div className="app">
      <aside className="side">
        <div className="brand">
          <h2>JEPA-Med-OOD</h2>
          <p>Chest X-Ray OOD Detection · ViT-S</p>
        </div>
        <nav className="nav">
          {[['infer', 'Inference'], ['metrics', 'Metrics (5%)'], ['data', 'Datasets & Models'], ['alerts', `Alerts${alertCount ? ` (${alertCount})` : ''}`]].map(([id, label]) => (
            <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div className="status">
          <b>Backend :8000</b>
          <span className={`dot ${models ? 'on' : 'off'}`} />
          {models ? 'Connected' : 'Offline'}
          {demo && ' · Demo data'}
        </div>
        <div className="foot">Research prototype — not a medical device.</div>
      </aside>

      <div className="main">
        <header className="chead">
          <h1>{tab === 'infer' ? 'Single-image OOD scoring' : tab === 'metrics' ? 'Results · 5% run (210 ID / 138 + 662 OOD)' : tab === 'alerts' ? 'Alerts · backend-generated flags' : 'Datasets & checkpoints'}</h1>
          <span className="spacer" />
          <div className="badges">
            <span className="badge">ID-ONLY 95% THR</span>
            {demo && <span className="badge demo">DEMO DATA — NOT REAL</span>}
          </div>
        </header>

        <div className="content">
          {error && <div className="err">{error}</div>}

        {tab === 'infer' && <Inference onActivity={refreshAlerts} />}

        {tab === 'metrics' && <MetricsView metrics={metrics} />}

        {tab === 'alerts' && <AlertsView key="alerts" />}

        {tab === 'data' && (
          <div className="grid c2">
            <div className="card">
              <h3>Datasets</h3>
              <div className="sub">Roles enforced: ID train/test only · OOD test only</div>
              {!datasets?.datasets ? <div className="missing">Not loaded</div> : (
                <table><thead><tr><th>Set</th><th>Role</th><th>Counts</th></tr></thead>
                  <tbody>{datasets.datasets.map((d, i) => (
                    <tr key={i}><td>{d.name}</td><td>{d.role}</td>
                      <td>{d.total ? `${d.normal} N / ${d.tuberculosis} TB = ${d.total}` : d.images}</td></tr>
                  ))}</tbody></table>
              )}
              <hr className="sep" />
              <div className="sub">Expected folders: <code>Datasets/CXR/TB_Chest_Radiography_Database</code>, <code>Montgomery TB CXR/images</code>, <code>Shenzhen TB CXR/images/images</code></div>
            </div>
            <div className="card">
              <h3>Models</h3>
              <div className="sub">Checkpoint availability (live)</div>
              {!models?.models ? <div className="missing">Backend offline</div> : (
                <table><thead><tr><th>Model</th><th>Encoder</th><th>Head</th><th>Detail</th></tr></thead>
                  <tbody>{models.models.map((m) => (
                    <tr key={m.id}><td>{m.label}</td><td>{m.encoder ? '✓' : 'Checkpoint not found'}</td>
                      <td>{m.head ? '✓' : '—'}</td><td>{m.detail}</td></tr>
                  ))}</tbody></table>
              )}
              <hr className="sep" />
              <div className="sub">{models?.disclaimer || 'Research prototype for OOD detection. Not a medical device and not for diagnosis.'}</div>
            </div>
          </div>
        )}

          <div className="footer">Higher score = more OOD · normalized = P(ID ≤ raw) · Backend <code>:8000</code> · Frontend <code>:5173</code> · Not for clinical use.</div>
        </div>
      </div>
    </div>
  );
}
