import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const HISTORY_KEY = 'agroscan.scan-history.v1';
const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000';
const AIRFLOW_URL = import.meta.env.VITE_AIRFLOW_URL || 'http://localhost:8080';
const readHistory = () => {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); } catch { return []; }
};
const pretty = (value) => value?.split(/_+/).filter((part) => part && part.toLowerCase() !== 'bell').join(' ') || 'Unknown';
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;

function Icon({ name, size = 20 }) {
  const paths = {
    leaf: <><path d="M20 4c-8 0-14 4-14 11a5 5 0 0 0 5 5c7 0 11-6 9-16Z"/><path d="M4 21c3-6 7-9 13-12"/></>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
    scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="4"/></>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    pulse: <><path d="M3 12h4l3-8 4 16 3-8h4"/></>,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M20 16v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3"/></>,
    arrow: <><path d="M7 17 17 7M7 7h10v10"/></>,
    check: <><path d="m5 12 4 4L19 6"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(readHistory);
  const [ready, setReady] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [showAllHistory, setShowAllHistory] = useState(false);
  const [activeConsole, setActiveConsole] = useState('mlflow');
  const [activePage, setActivePage] = useState('overview');
  const [feedbackClass, setFeedbackClass] = useState('');
  const [feedbackSaved, setFeedbackSaved] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [feedbackError, setFeedbackError] = useState('');
  const [clearBusy, setClearBusy] = useState(false);
  const [clearError, setClearError] = useState('');
  const pageTitles = { overview: 'Overview', analyze: 'New analysis', history: 'Scan history', performance: 'Model performance', mlops: 'MLOps consoles' };

  const refreshAnalytics = async () => {
    try {
      const response = await fetch('/dashboard');
      if (response.ok) setAnalytics(await response.json());
    } catch { /* retain the latest available dashboard summary */ }
  };

  useEffect(() => {
    let alive = true;
    const check = async () => {
      try {
        const response = await fetch('/ready');
        if (!response.ok) throw new Error('Model unavailable');
        const data = await response.json();
        if (alive) setReady(data);
      } catch { if (alive) setReady(null); }
    };
    check();
    refreshAnalytics();
    const timer = window.setInterval(() => { check(); refreshAnalytics(); }, 30000);
    return () => { alive = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!file) { setPreview(''); return undefined; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const selectFile = (picked) => {
    if (!picked) return;
    if (!picked.type.startsWith('image/')) { setError('Choose a JPG, PNG, or other supported image file.'); return; }
    setError(''); setFile(picked); setResult(null);
  };

  const analyze = async () => {
    if (!file) return;
    setBusy(true); setError('');
    try {
      const form = new FormData(); form.append('file', file);
      const response = await fetch('/predict', { method: 'POST', body: form });
      if (!response.ok) {
        let message = `Analysis failed (${response.status}).`;
        try { const body = await response.json(); message = body.detail || message; } catch { /* use fallback */ }
        throw new Error(message);
      }
      const data = await response.json();
      const item = { ...data, filename: file.name, createdAt: new Date().toISOString() };
      setResult(item);
      setFeedbackClass(data.predicted_class); setFeedbackSaved(false); setFeedbackError('');
      const next = [item, ...readHistory()].slice(0, 12);
      setHistory(next); localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
    } catch (e) { setError(e.message || 'Could not reach AgroScan. Check the API connection.'); }
    finally { setBusy(false); }
  };

  const reset = () => { setFile(null); setResult(null); setError(''); if (inputRef.current) inputRef.current.value = ''; };
  const exportHistory = () => {
    if (!history.length) return;
    const rows = [['timestamp', 'image', 'predicted_class', 'confidence', 'low_confidence', 'model_version']];
    history.forEach((item) => rows.push([item.createdAt, item.filename, item.predicted_class, item.confidence, item.low_confidence, item.model_version || '']));
    const csv = rows.map((row) => row.map((value) => `"${String(value ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'agroscan-scan-history.csv'; link.click(); URL.revokeObjectURL(url);
  };
  const clearHistory = async () => {
    if (clearBusy || !window.confirm('Clear all live scan and feedback data for this AgroScan server? This resets scan counts and live metrics for every dashboard page and user.')) return;
    setClearBusy(true); setClearError('');
    try {
      const response = await fetch('/history', { method: 'DELETE' });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Could not clear server scan data (${response.status}).`);
      }
      localStorage.removeItem(HISTORY_KEY);
      setHistory([]); setResult(null); setShowAllHistory(false);
      setFeedbackSaved(false); setFeedbackError('');
      await refreshAnalytics();
    } catch (e) { setClearError(e.message || 'Could not clear scan data.'); }
    finally { setClearBusy(false); }
  };
  const current = result || history[0];
  const confidence = Number(current?.confidence || 0);
  const daily = analytics?.daily_predictions || [];
  const maxDaily = Math.max(1, ...daily.map((item) => item.count));
  const submitFeedback = async (actualClass) => {
    if (!current?.prediction_id || feedbackBusy || feedbackSaved) return;
    setFeedbackBusy(true); setFeedbackError('');
    try {
      const response = await fetch('/feedback', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prediction_id: current.prediction_id, actual_class: actualClass }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Could not save feedback (${response.status}).`);
      }
      setFeedbackSaved(true);
      await refreshAnalytics();
    } catch (e) { setFeedbackError(e.message || 'Could not save feedback.'); }
    finally { setFeedbackBusy(false); }
  };

  return <div className="app-shell">
    <aside className="sidebar">
      <button className="brand" onClick={() => setActivePage("overview")} aria-label="AgroScan home"><span className="brand-mark"><Icon name="leaf" size={23}/></span><span>agro<span className="brand-light">scan</span></span></button>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="nav-list" aria-label="Main navigation">
        <button className={`nav-item ${activePage === 'overview' ? 'active' : ''}`} onClick={() => setActivePage('overview')}><Icon name="grid"/><span>Overview</span></button>
        <button className={`nav-item ${activePage === 'analyze' ? 'active' : ''}`} onClick={() => setActivePage('analyze')}><Icon name="scan"/><span>New analysis</span></button>
        <button className={`nav-item ${activePage === 'history' ? 'active' : ''}`} onClick={() => setActivePage('history')}><Icon name="clock"/><span>Scan history</span></button>
        <button className={`nav-item ${activePage === 'performance' ? 'active' : ''}`} onClick={() => setActivePage('performance')}><Icon name="pulse"/><span>Model performance</span></button>
        <button className={`nav-item ${activePage === 'mlops' ? 'active' : ''}`} onClick={() => setActivePage('mlops')}><Icon name="grid"/><span>MLOps consoles</span></button>
      </nav>
      <div className="sidebar-bottom">
        <div className="api-card"><span className={`status-dot ${ready ? 'online' : ''}`}/><div><strong>{ready ? 'System operational' : 'Model offline'}</strong><small>{ready ? 'Inference API connected' : 'Waiting for model readiness'}</small></div></div>
        <div className="sidebar-foot"><span className="mini-leaf"><Icon name="leaf" size={14}/></span><span>Plant health, made clearer.</span></div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="crumb"><span>Workspace</span><b>/</b><strong>{pageTitles[activePage]}</strong></div><div className="topbar-right"><span className="live-indicator"><i className={ready ? 'online' : ''}/>{ready ? 'API connected' : 'API unavailable'}</span><span className="avatar">A</span></div></header>
      <div className="content" data-page={activePage}>
        <div className="page-view overview-view">
        <section className="welcome-row"><div><div className="eyebrow"><span className="eyebrow-line"/> FIELD INTELLIGENCE</div><h1>Plant health, at a glance.</h1><p className="subtitle">Turn a leaf photo into a clearer next step.</p></div><div className="date-chip"><Icon name="pulse" size={17}/><span>Live model status</span><span className={`status-dot ${ready ? 'online' : ''}`}/></div></section>

        <section className="metrics-grid" aria-label="Dashboard summary">
          <article className="metric-card"><div className="metric-top"><span className="metric-icon green"><Icon name="scan"/></span><span className="metric-tag">ALL API SCANS</span></div><div className="metric-value">{analytics?.total_predictions ?? '—'}<span className="metric-unit"> scans</span></div><div className="metric-caption">Prediction log · updates every 30 seconds</div></article>
          <article className="metric-card"><div className="metric-top"><span className="metric-icon blue"><Icon name="pulse"/></span><span className={`health-pill ${ready ? '' : 'offline'}`}><i/> {ready ? 'Operational' : 'Offline'}</span></div><div className="metric-value metric-status">{ready ? 'Ready to analyze' : 'Model unavailable'}</div><div className="metric-caption">FastAPI inference service</div></article>
          <article className="metric-card"><div className="metric-top"><span className="metric-icon amber"><Icon name="leaf"/></span><span className="metric-tag">MODEL</span></div><div className="metric-value metric-model">{ready?.mlflow_model_name || 'AgroScan vision'}</div><div className="metric-caption">{ready?.model_version ? `Version ${ready.model_version}` : ready ? 'Pepper · potato · tomato' : 'Waiting for model metadata'}</div></article>
        </section>

        <section className="analytics-grid" aria-label="Live analytics">
          <article className="panel analytics-panel activity-chart-panel"><div className="analytics-heading"><div><div className="section-kicker">USAGE</div><h2>Prediction activity</h2></div><span className="analytics-period">LAST 7 DAYS</span></div><div className="bar-chart">{daily.map((item) => <div className="chart-column" key={item.date}><span className="chart-count">{item.count || ''}</span><div className="chart-track"><i style={{ height: `${Math.max(item.count ? 8 : 2, item.count / maxDaily * 100)}%` }}/></div><span className="chart-day">{new Date(`${item.date}T00:00:00`).toLocaleDateString([], { weekday: 'short' })}</span></div>)}</div><div className="chart-foot"><span><i/> API predictions</span><span>{analytics ? `${analytics.total_predictions} logged in total` : 'Waiting for API activity'}</span></div></article>
          <article className="panel analytics-panel mix-panel"><div className="analytics-heading"><div><div className="section-kicker">FIELD SIGNAL</div><h2>Most detected classes</h2></div><span className="analytics-period">LATEST 10K</span></div>{analytics?.class_distribution?.length ? <div className="mix-list">{analytics.class_distribution.slice(0, 5).map((item, index) => <div className="mix-item" key={item.class}><div className="mix-label"><span>{pretty(item.class)}</span><b>{item.count}</b></div><div className="mix-track"><i className={`mix-${index}`} style={{ width: `${Math.max(4, item.share * 100)}%` }}/></div></div>)}</div> : <div className="analytics-empty">Predictions will appear here after the first scan.</div>}<div className="chart-foot"><span>Distribution from API prediction logs</span></div></article>
          <article className="panel analytics-panel quality-panel"><div className="analytics-heading"><div><div className="section-kicker">MODEL OPERATIONS</div><h2>Quality & performance</h2></div><span className={`quality-indicator ${analytics?.monitoring?.status || 'unknown'}`}><i/>{analytics?.monitoring?.status || 'No report'}</span></div><div className="quality-stats"><div><span>Average latency</span><strong>{analytics?.average_latency_ms != null ? `${analytics.average_latency_ms} ms` : '—'}</strong></div><div><span>p95 latency</span><strong>{analytics?.p95_latency_ms != null ? `${analytics.p95_latency_ms} ms` : '—'}</strong></div><div><span>Average confidence</span><strong>{analytics?.average_confidence != null ? percent(analytics.average_confidence) : '—'}</strong></div><div><span>Low confidence</span><strong>{analytics?.low_confidence_rate != null ? percent(analytics.low_confidence_rate) : '—'}</strong></div></div><div className="benchmark"><span>Held-out test benchmark</span><strong>{analytics?.evaluation?.test_accuracy != null ? `${(analytics.evaluation.test_accuracy * 100).toFixed(1)}% accuracy` : 'Not available'}</strong><small>Macro F1 {analytics?.evaluation?.test_macro_f1 != null ? (analytics.evaluation.test_macro_f1 * 100).toFixed(1) : '—'}% · evaluation report</small></div>{analytics?.monitoring?.log_file && <div className="monitor-source">Monitor report source: {analytics.monitoring.log_file}</div>}</article>
        </section>
        </div>

        <div className="page-view analyze-view">
          <section className="page-intro"><div className="eyebrow"><span className="eyebrow-line"/> IMAGE ANALYSIS</div><h1>Analyze a leaf</h1><p className="subtitle">Upload a clear image for a prediction and confidence breakdown.</p></section>
        <section className="work-grid" id="analyze">
          <article className="panel analyze-panel">
            <div className="panel-heading"><div><div className="section-kicker">IMAGE ANALYSIS</div><h2>Start a new scan</h2><p>Upload a clear photo of a single leaf.</p></div><span className="step-badge">01 <span>/ 02</span></span></div>
            <div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onDragEnter={(e) => { e.preventDefault(); setDragging(true); }} onDragOver={(e) => e.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files[0]); }}>
              <input ref={inputRef} type="file" accept="image/*" onChange={(e) => selectFile(e.target.files[0])} aria-label="Choose a leaf image" />
              {preview ? <div className="preview-wrap"><img src={preview} alt="Selected leaf preview"/><div className="file-chip"><Icon name="image" size={15}/><span>{file.name}</span><button className="icon-button" onClick={(e) => { e.stopPropagation(); reset(); }} aria-label="Remove selected image"><Icon name="close" size={16}/></button></div></div> : <button className="drop-content" onClick={() => inputRef.current?.click()}><span className="upload-icon"><Icon name="upload" size={23}/></span><strong>Drop your leaf image here</strong><span className="drop-secondary">or <em>browse files</em> from your device</span><span className="file-types">JPG, PNG · Max 10 MB</span></button>}
            </div>
            {error && <div className="error-message" role="alert"><span>!</span>{error}</div>}
            <div className="analyze-actions"><span className="privacy-note"><Icon name="check" size={15}/> Images are processed by your AgroScan API</span><button className="primary-button" disabled={!file || busy || !ready} onClick={analyze}>{busy ? <><span className="spinner"/>Analyzing</> : <>Analyze image <Icon name="arrow" size={17}/></>}</button></div>
            {!ready && <div className="offline-note">Connect the API and load a model to enable analysis.</div>}
          </article>

          <article className="panel result-panel" aria-live="polite">
            <div className="panel-heading result-heading"><div><div className="section-kicker">LATEST RESULT</div><h2>{current ? 'Analysis insight' : 'Your result will appear here'}</h2><p>{current ? current.filename : 'Start a scan to see the prediction.'}</p></div>{current && <span className={`result-state ${current.low_confidence ? 'uncertain' : 'confirmed'}`}><i/>{current.low_confidence ? 'Review result' : 'High confidence'}</span>}</div>
            {current ? <div className="result-content"><div className={`result-hero ${current.low_confidence ? 'caution' : ''}`}><span className="result-leaf"><Icon name="leaf" size={20}/></span><div><span className="result-label">{current.low_confidence ? 'POSSIBLE MATCH' : /healthy/i.test(current.predicted_class) ? 'HEALTHY LEAF' : 'PREDICTED CONDITION'}</span><h3>{pretty(current.predicted_class)}</h3><p>{current.low_confidence ? (current.message || 'Try a closer, well-lit photo of one leaf.') : 'Review the result alongside what you can see in the field.'}</p></div></div><div className="confidence-row"><div><span>Model confidence</span><strong>{percent(confidence)}</strong></div><div className="confidence-track"><span style={{ width: percent(confidence) }}/></div></div><div className="top-predictions"><div className="predictions-title">OTHER POSSIBLE MATCHES <span>TOP {current.top_k?.length || 1}</span></div>{(current.top_k || [{ class: current.predicted_class, confidence }]).map((guess, i) => <div className="prediction-row" key={`${guess.class}-${i}`}><span className={`prediction-rank ${i === 0 ? 'first' : ''}`}>{String(i + 1).padStart(2, '0')}</span><span className="prediction-name">{pretty(guess.class)}</span><span className="prediction-percent">{percent(guess.confidence)}</span></div>)}</div></div> : <div className="empty-result"><div className="empty-illustration"><span><Icon name="leaf" size={34}/></span><i/><i/><i/></div><strong>Insight starts with a leaf</strong><p>Upload an image and AgroScan will return its best match with confidence scores.</p></div>}
            <div className="result-foot"><Icon name="pulse" size={15}/><span>{current?.model_version ? `Model version ${current.model_version}` : 'Predictions are decision support, not a diagnosis.'}</span></div>
            {current?.prediction_id && <div className="feedback-card"><div><strong>Can you verify this prediction?</strong><span>Your feedback helps measure live accuracy.</span></div>{feedbackSaved ? <div className="feedback-success">Feedback saved. Live metrics updated.</div> : <><div className="feedback-actions"><button disabled={feedbackBusy} onClick={() => submitFeedback(current.predicted_class)}>Correct</button><label>Actual class<select value={feedbackClass || current.predicted_class} onChange={(e) => setFeedbackClass(e.target.value)}>{(ready?.classes || analytics?.model?.classes || []).map((name) => <option key={name} value={name}>{pretty(name)}</option>)}</select></label><button className="feedback-correction" disabled={feedbackBusy} onClick={() => submitFeedback(feedbackClass || current.predicted_class)}>{feedbackBusy ? 'Saving…' : 'Submit correction'}</button></div>{feedbackError && <span className="feedback-error" role="alert">{feedbackError}</span>}</>}</div>}
          </article>
        </section>
        </div>

        <div className="page-view history-view">
        <section className="page-intro"><div className="eyebrow"><span className="eyebrow-line"/> YOUR WORKSPACE</div><h1>Scan history</h1><p className="subtitle">Review recent scans and clear the shared live data used throughout the dashboard.</p></section>
        <section className="panel activity-panel" id="activity"><div className="activity-heading"><div><div className="section-kicker">RECENT ACTIVITY</div><h2>Recent scans</h2></div><div className="history-tools"><span className="local-badge"><span/> Browser history</span><button onClick={exportHistory} disabled={!history.length}>Export CSV</button><button onClick={clearHistory} disabled={clearBusy}>{clearBusy ? 'Clearing…' : 'Clear all scans'}</button></div></div>{history.length ? <div className="activity-table"><div className="table-head"><span>PLANT / CONDITION</span><span>IMAGE</span><span>CONFIDENCE</span><span>WHEN</span><span>STATUS</span></div>{history.slice(0, showAllHistory ? 12 : 5).map((item, i) => <div className="table-row" key={`${item.createdAt}-${i}`}><span className="table-condition"><i className="table-leaf"><Icon name="leaf" size={16}/></i><strong>{pretty(item.predicted_class)}</strong></span><span className="table-file">{item.filename}</span><span className="table-confidence">{percent(item.confidence)}</span><span className="table-date">{new Date(item.createdAt).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}</span><span><span className={`table-status ${item.low_confidence ? 'review' : ''}`}><i/>{item.low_confidence ? 'Review' : 'Analyzed'}</span></span></div>)}</div> : <div className="empty-activity"><span><Icon name="clock" size={18}/></span><div><strong>No browser scans yet</strong><p>New results will appear here after analysis. Server scan totals are shown on Overview.</p></div></div>}{clearError && <div className="feedback-error" role="alert">{clearError}</div>}<div className="table-note">This page lists recent scans saved in this browser. Clear all scans also resets server prediction logs, verified feedback, and generated monitoring summaries for every dashboard page. Saved offline benchmark metrics and the drift reference baseline remain because they describe evaluation data, not live scans. {history.length > 5 && <button className="text-button" onClick={() => setShowAllHistory(!showAllHistory)}>{showAllHistory ? "Show fewer" : "View all scans"}</button>}</div></section>
        </div>
        <div className="page-view performance-view">
          <section className="page-intro"><div className="eyebrow"><span className="eyebrow-line"/> LIVE INFERENCE METRICS</div><h1>Model performance</h1><p className="subtitle">Metrics update from current prediction traffic and verified feedback.</p></section>
          <section className="performance-summary live-performance-summary">
            <article className="panel score-card"><span>LIVE PREDICTIONS</span><strong>{analytics?.total_predictions ?? 'N/A'}</strong><small>Logged by the inference API</small></article>
            <article className="panel score-card"><span>VERIFIED LABELS</span><strong>{analytics?.live_performance?.labeled_predictions ?? 0}</strong><small>Predictions checked by a user</small></article>
            <article className="panel score-card"><span>LIVE ACCURACY</span><strong>{analytics?.live_performance?.accuracy != null ? percent(analytics.live_performance.accuracy) : 'Awaiting labels'}</strong><small>Only verified predictions count</small></article>
            <article className="panel score-card"><span>LIVE MACRO F1</span><strong>{analytics?.live_performance?.macro_f1 != null ? percent(analytics.live_performance.macro_f1) : 'Awaiting labels'}</strong><small>Calculated from verified labels</small></article>
            <article className="panel score-card"><span>MEAN CONFIDENCE</span><strong>{analytics?.average_confidence != null ? percent(analytics.average_confidence) : 'N/A'}</strong><small>Current prediction log window</small></article>
            <article className="panel score-card"><span>LOW-CONFIDENCE RATE</span><strong>{analytics?.low_confidence_rate != null ? percent(analytics.low_confidence_rate) : 'N/A'}</strong><small>Current prediction log window</small></article>
            <article className="panel score-card"><span>AVERAGE LATENCY</span><strong>{analytics?.average_latency_ms != null ? `${analytics.average_latency_ms} ms` : 'N/A'}</strong><small>Per-image inference time</small></article>
            <article className="panel score-card"><span>P95 LATENCY</span><strong>{analytics?.p95_latency_ms != null ? `${analytics.p95_latency_ms} ms` : 'N/A'}</strong><small>Current prediction log window</small></article>
          </section>
          <section className="panel alert-panel" aria-live="polite">
            <div className="performance-heading"><div><div className="section-kicker">LIVE MODEL WATCH</div><h2>Signals and explanations</h2><p>Alerts recalculate with dashboard data. A drift signal is a change to inputs or predictions; verified accuracy is the direct quality signal.</p></div><span className="dataset-badge">{analytics?.live_alerts?.filter((item) => item.severity === 'warning').length || 0} active warnings</span></div>
            {analytics?.live_alerts?.length ? <div className="alert-list">{analytics.live_alerts.map((item) => <article className={`signal-card ${item.severity}`} key={item.id}><div className="signal-top"><span className="signal-category">{item.category}</span><strong>{item.title}</strong></div><p>{item.explanation}</p><div className="signal-evidence"><b>Evidence</b> {item.evidence}</div><div className="signal-action"><b>Next step</b> {item.action}</div></article>)}</div> : <div className="signal-clear"><strong>No configured warnings are active.</strong><p>Live measurements are below their warning thresholds. Keep collecting verified feedback; this does not guarantee future performance.</p></div>}
            <div className="signal-footnote">Thresholds come from configs/monitoring.yaml. Accuracy alerts require at least {analytics?.alert_thresholds?.min_labeled_predictions ?? 20} verified predictions; drift alerts require at least {analytics?.alert_thresholds?.min_predictions ?? 50} recent predictions and a reference baseline.</div>
          </section>
          <section className="panel performance-panel"><div className="performance-heading"><div><div className="section-kicker">ONLINE EVALUATION</div><h2>Verified class-level results</h2><p>Correctness is computed only after a user confirms or corrects a prediction.</p></div><span className="dataset-badge">{analytics?.live_performance?.labeled_predictions || 0} verified</span></div>
            {analytics?.live_performance?.per_class?.length ? <div className="performance-table-wrap"><table className="performance-table"><thead><tr><th>Disease class</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>Labels</th></tr></thead><tbody>{analytics.live_performance.per_class.map((row) => <tr key={row.class}><td>{pretty(row.class)}</td><td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td><span className="f1-cell"><i><b style={{ width: percent(row.f1) }}/></i>{percent(row.f1)}</span></td><td>{row.support}</td></tr>)}</tbody></table></div> : <div className="live-empty"><span className="upload-icon"><Icon name="check" size={20}/></span><strong>No verified predictions yet</strong><p>After a scan, mark the result as correct or submit the actual class. Accuracy and F1 will populate from that feedback.</p><button className="primary-button" onClick={() => setActivePage('analyze')}>Analyze a leaf <Icon name="arrow" size={15}/></button></div>}
            <div className="performance-note">Prediction confidence and class mix are live signals, not ground-truth accuracy. Verify predictions before using accuracy and F1 as quality measures.</div>
          </section>
          <details className="panel offline-benchmark"><summary>View saved offline evaluation benchmark</summary><div className="benchmark-grid"><div><span>Validation accuracy</span><strong>{analytics?.evaluation?.validation_accuracy != null ? percent(analytics.evaluation.validation_accuracy) : 'N/A'}</strong></div><div><span>Held-out test accuracy</span><strong>{analytics?.evaluation?.test_accuracy != null ? percent(analytics.evaluation.test_accuracy) : 'N/A'}</strong></div><div><span>Test macro F1</span><strong>{analytics?.evaluation?.test_macro_f1 != null ? percent(analytics.evaluation.test_macro_f1) : 'N/A'}</strong></div><div><span>Existing v5 test accuracy</span><strong>{analytics?.evaluation?.baseline?.test_accuracy != null ? percent(analytics.evaluation.baseline.test_accuracy) : 'N/A'}</strong></div></div><p>These are saved evaluation reports, not live traffic metrics. Source: reports/metrics.json and reports/v5_plantdoc_metrics.json.</p></details>
        </div>
        <div className="page-view mlops-view">
          <section className="page-intro"><div className="eyebrow"><span className="eyebrow-line"/> EXPERIMENTS & ORCHESTRATION</div><h1>MLOps consoles</h1><p className="subtitle">Experiment tracking and pipeline orchestration dashboards.</p></section>
          <section className="panel console-panel" id="mlops"><div className="console-heading"><div><div className="section-kicker">CONNECTED SERVICES</div><h2>Open a console</h2><p>Embed the service here or open it at full size.</p></div><div className="console-tabs" role="tablist" aria-label="MLOps dashboard"><button role="tab" aria-selected={activeConsole === 'mlflow'} className={activeConsole === 'mlflow' ? 'selected' : ''} onClick={() => setActiveConsole('mlflow')}>MLflow</button><button role="tab" aria-selected={activeConsole === 'airflow'} className={activeConsole === 'airflow' ? 'selected' : ''} onClick={() => setActiveConsole('airflow')}>Airflow</button></div></div><div className="console-toolbar"><span><i className="console-dot"/>{activeConsole === 'mlflow' ? 'Experiment tracking' : 'Pipeline orchestration'} · {activeConsole === 'mlflow' ? MLFLOW_URL : AIRFLOW_URL}</span><a href={activeConsole === 'mlflow' ? MLFLOW_URL : AIRFLOW_URL} target="_blank" rel="noreferrer">Open in new tab <Icon name="arrow" size={13}/></a></div><div className="console-frame-wrap"><iframe key={activeConsole} title={`${activeConsole === 'mlflow' ? 'MLflow tracking' : 'Airflow orchestration'} dashboard`} src={activeConsole === 'mlflow' ? MLFLOW_URL : AIRFLOW_URL} loading="lazy" referrerPolicy="no-referrer"/></div><div className="console-note">If the console does not load here, make sure its service is running and allows embedding. The new-tab link is always available.</div></section>
        </div>
        <footer className="page-footer"><span>AgroScan <span>·</span> Plant health intelligence</span><span>For decision support. Consult an agronomist for treatment decisions.</span></footer>
      </div>
    </main>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
