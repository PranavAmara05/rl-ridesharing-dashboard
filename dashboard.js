// ═══════════════════════════════════════════════════════════════════════════
// DARM+DPRS+DQN — Dashboard JS (v3) — Full Features
// ═══════════════════════════════════════════════════════════════════════════
const API = '';
let charts = {};
let es = null;
let selectedModels = new Set();
let simRunning = false;
let simInterval = null;

// ═══ Tab Navigation ═══
document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('p-' + tab.dataset.tab).classList.add('active');
    if (tab.dataset.tab === 'models') refreshModels();
    if (tab.dataset.tab === 'compare') refreshCompareChips();
  };
});

// Simulation Constants & Helpers
const GRID_SIZE = 15;
const VEH_COLORS = { 
  idle: '#3b82f6', 
  enroute: '#f59e0b', 
  occupied: '#f59e0b', 
  dispatching: '#8b5cf6',
  carrying: '#10b981'
};

// Animation state for smooth transit
let vehPositions = {}; // { vid: {x, y, targetX, targetY} }

function l2g(a) {
    const g = [];
    for (let r = 0; r < GRID_SIZE; r++) {
        g[r] = [];
        for (let c = 0; c < GRID_SIZE; c++) g[r][c] = 0;
    }
    if (!a) return g;
    for (let z = 0; z < a.length; z++) {
        const r = Math.floor(z / GRID_SIZE), c = z % GRID_SIZE;
        if (g[r]) g[r][c] = a[z];
    }
    return g;
}


// ═══ Slider Bindings ═══
const SL = {
  'cfg-lr':        ['v-lr',        v => `${(10**parseFloat(v)).toExponential(1)}`],
  'cfg-gamma':     ['v-gamma',     v => parseFloat(v).toFixed(3)],
  'cfg-eps-start': ['v-eps-start', v => parseFloat(v).toFixed(2)],
  'cfg-eps-end':   ['v-eps-end',   v => parseFloat(v).toFixed(2)],
  'cfg-eps-decay': ['v-eps-decay', v => parseInt(v).toLocaleString()],
  'cfg-batch':     ['v-batch',     v => v],
  'cfg-replay':    ['v-replay',    v => parseInt(v).toLocaleString()],
  'cfg-b1':        ['v-b1',        v => v],
  'cfg-b2':        ['v-b2',        v => v],
  'cfg-b3':        ['v-b3',        v => v],
  'cfg-b4':        ['v-b4',        v => v],
  'cfg-b5':        ['v-b5',        v => v],
  'cfg-fleet':     ['v-fleet',     v => v],
  'cfg-warmup':    ['v-warmup',    v => v],
  'cfg-rchange':   ['v-rchange',   v => `${(parseFloat(v)*100).toFixed(0)}%`],
  'cmp-steps':     ['v-cmp-steps', v => v],
};
Object.entries(SL).forEach(([id, [dId, fmt]]) => {
  const el = document.getElementById(id), d = document.getElementById(dId);
  if (!el || !d) return;
  const up = () => d.textContent = fmt(el.value);
  el.addEventListener('input', up);
  up();
});

// ═══ Build Config ═══
function buildConfig() {
  const rchangeEnabled = document.getElementById('cfg-rchange-enable')?.checked;
  const num = (id, fallback = 0) => {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = parseInt(el.value);
    return Number.isFinite(v) ? v : fallback;
  };
  return {
    steps: parseInt(document.getElementById('cfg-steps').value) || 500,
    algorithm: document.getElementById('cfg-algo').value,
    learning_rate: Math.pow(10, parseFloat(document.getElementById('cfg-lr').value)),
    gamma: parseFloat(document.getElementById('cfg-gamma').value),
    eps_start: parseFloat(document.getElementById('cfg-eps-start').value),
    eps_end: parseFloat(document.getElementById('cfg-eps-end').value),
    eps_decay: parseInt(document.getElementById('cfg-eps-decay').value),
    batch_size: parseInt(document.getElementById('cfg-batch').value),
    replay_size: parseInt(document.getElementById('cfg-replay').value),
    beta1: parseFloat(document.getElementById('cfg-b1').value),
    beta2: parseFloat(document.getElementById('cfg-b2').value),
    beta3: parseFloat(document.getElementById('cfg-b3').value),
    beta4: parseFloat(document.getElementById('cfg-b4').value),
    beta5: parseFloat(document.getElementById('cfg-b5').value),
    fleet_size: parseInt(document.getElementById('cfg-fleet').value),
    warmup_steps: parseInt(document.getElementById('cfg-warmup').value),
    use_dataset: document.getElementById('cfg-dataset').checked,
    dataset_path: document.getElementById('cfg-dataset-path').value.trim(),
    dataset_max_files: parseInt(document.getElementById('cfg-dataset-files').value) || 1,
    dataset_max_rows: parseInt(document.getElementById('cfg-dataset-rows').value) || 200000,
    dataset_time_bin: 1,
    dataset_demand_scale: parseFloat(document.getElementById('cfg-dataset-scale').value) || 1.0,
    enable_osrm: document.getElementById('cfg-osrm').checked,
    osrm_base_url: document.getElementById('cfg-osrm-url').value.trim(),
    osrm_profile: document.getElementById('cfg-osrm-profile').value.trim(),
    osrm_timeout_sec: 2.0,
    enable_congestion: document.getElementById('cfg-congestion').checked,
    use_time_features: document.getElementById('cfg-use-time-features').checked,
    rider_change_prob: rchangeEnabled ? parseFloat(document.getElementById('cfg-rchange').value) : 0,
    rider_cancel_prob: rchangeEnabled ? parseFloat(document.getElementById('cfg-rchange').value) / 3 : 0,
    eval_interval: num('cfg-eval-interval', 0),
    eval_steps: num('cfg-eval-steps', 0),
    eval_seed: num('cfg-eval-seed', 0),
  };
}

function setField(id, value) {
  const el = document.getElementById(id);
  if (!el || value === undefined || value === null) return;
  if (el.type === 'checkbox') {
    el.checked = !!value;
    el.dispatchEvent(new Event('change'));
    return;
  }
  el.value = value;
  el.dispatchEvent(new Event('input'));
}

function applyConfigDefaults(cfg) {
  if (!cfg) return;
  setField('cfg-algo', cfg.algorithm || 'full');
  setField('cfg-lr', Math.log10(cfg.learning_rate || 0.0005));
  setField('cfg-gamma', cfg.gamma ?? 0.95);
  setField('cfg-eps-start', cfg.eps_start ?? 1.0);
  setField('cfg-eps-end', cfg.eps_end ?? 0.05);
  setField('cfg-eps-decay', cfg.eps_decay ?? 5000);
  setField('cfg-batch', cfg.batch_size ?? 64);
  setField('cfg-replay', cfg.replay_size ?? 5000);
  setField('cfg-b1', cfg.beta1 ?? 10);
  setField('cfg-b2', cfg.beta2 ?? -1);
  setField('cfg-b3', cfg.beta3 ?? -5);
  setField('cfg-b4', cfg.beta4 ?? 12);
  setField('cfg-b5', cfg.beta5 ?? -8);
  setField('cfg-fleet', cfg.fleet_size ?? 150);
  setField('cfg-warmup', cfg.warmup_steps ?? 20);
  setField('cfg-dataset', cfg.use_dataset ?? true);
  setField('cfg-dataset-path', cfg.dataset_path || '');
  setField('cfg-dataset-files', cfg.dataset_max_files ?? 1);
  setField('cfg-dataset-rows', cfg.dataset_max_rows ?? 100000);
  setField('cfg-dataset-scale', cfg.dataset_demand_scale ?? 0.1);
  setField('cfg-osrm', cfg.enable_osrm ?? false);
  setField('cfg-osrm-url', cfg.osrm_base_url || 'http://router.project-osrm.org');
  setField('cfg-osrm-profile', cfg.osrm_profile || 'driving');
  setField('cfg-congestion', cfg.enable_congestion ?? false);
  setField('cfg-use-time-features', cfg.use_time_features ?? false);
  setField('cfg-rchange-enable', (cfg.rider_change_prob || 0) > 0);
  setField('cfg-rchange', cfg.rider_change_prob ?? 0);
  setField('cfg-eval-interval', cfg.eval_interval ?? 200);
  setField('cfg-eval-steps', cfg.eval_steps ?? 100);
  setField('cfg-eval-seed', cfg.eval_seed ?? 123);
}

// Toggle rider-change slider enabled state
const rToggle = document.getElementById('cfg-rchange-enable');
const rSlider = document.getElementById('cfg-rchange');
if (rToggle && rSlider) {
  const apply = () => { rSlider.disabled = !rToggle.checked; };
  rToggle.addEventListener('change', apply);
  apply();
}

// Toggle dataset fields enabled state
const dToggle = document.getElementById('cfg-dataset');
const dFields = [
  document.getElementById('cfg-dataset-path'),
  document.getElementById('cfg-dataset-files'),
  document.getElementById('cfg-dataset-rows'),
  document.getElementById('cfg-dataset-scale'),
];
if (dToggle) {
  const apply = () => {
    dFields.forEach(el => { if (el) el.disabled = !dToggle.checked; });
  };
  dToggle.addEventListener('change', apply);
  apply();
}

// Toggle OSRM URL fields
const oToggle = document.getElementById('cfg-osrm');
const oFields = [
  document.getElementById('cfg-osrm-url'),
  document.getElementById('cfg-osrm-profile'),
];
if (oToggle) {
  const apply = () => {
    oFields.forEach(el => { if (el) el.disabled = !oToggle.checked; });
  };
  oToggle.addEventListener('change', apply);
  apply();
}

// ═══ Charts ═══
const CC = {ar:'#10b981',profit:'#8b5cf6',wait:'#06b6d4',loss:'#f59e0b',eps:'#ef4444',qmax:'#818cf8',v_ar:'#22c55e',vprofit:'#a855f7',vwait:'#38bdf8',vocc:'#fbbf24'};
function mkCh(id, label, color, dash=false) {
  const el = document.getElementById(id);
  if (!el) return null;
  return new Chart(el, {
    type:'line', data:{labels:[],datasets:[{label,data:[],borderColor:color,borderWidth:1.5,fill:{target:'origin',above:color+'15'},pointRadius:0,tension:.3,borderDash:dash?[6,4]:undefined}]},
    options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},
      scales:{x:{display:false},y:{ticks:{color:'#64748b',font:{size:8},maxTicksLimit:4},grid:{color:'rgba(255,255,255,.04)'}}}}
  });
}
function initCharts(){charts={ar:mkCh('ch-ar','AR',CC.ar),profit:mkCh('ch-profit','$',CC.profit),wait:mkCh('ch-wait','W',CC.wait),loss:mkCh('ch-loss','L',CC.loss),eps:mkCh('ch-eps','ε',CC.eps),qmax:mkCh('ch-qmax','Q',CC.qmax),val_ar:mkCh('ch-val-ar','vAR',CC.v_ar,true),val_profit:mkCh('ch-val-profit','v$',CC.vprofit,true),val_wait:mkCh('ch-val-wait','vW',CC.vwait,true),val_occ:mkCh('ch-val-occ','vOcc',CC.vocc,true)};}
function pushCh(ch,vals){if(!ch||!vals||!vals.length)return;ch.data.labels=vals.map((_,i)=>i);ch.data.datasets[0].data=vals;ch.update('none');}

// ═══ SSE + Polling Fallback ═══
let pollTimer = null;
function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    const d = await api('/api/train/status');
    if (d) onUpdate(d);
  }, 1000);
}
function connectSSE() {
  if (es) try{es.close();}catch(e){}
  es = new EventSource(API+'/api/stream');
  es.onmessage = e => { try{const d=JSON.parse(e.data);if(d.type==='train_update')onUpdate(d);}catch(err){} };
  es.onerror = () => { setPill('RECONNECTING','error'); startPolling(); setTimeout(connectSSE,3000); };
  es.onopen = () => { if(pollTimer){clearInterval(pollTimer);pollTimer=null;} };
}
function setPill(t,c){const p=document.getElementById('status-pill');p.className='status-'+c;p.innerHTML=`<span class="dot"></span>${t}`;}

function onUpdate(d) {
  if(d.model_id)document.getElementById('sim-model-pill').textContent=d.model_id;
  if(d.running&&!d.paused)setPill('TRAINING','training');
  else if(d.paused)setPill('PAUSED','paused');
  else if(d.phase==='done')setPill('DONE','done');
  else if(d.phase==='error')setPill('ERROR','error');
  else setPill(d.phase.toUpperCase(),'idle');

  const pct=d.total_steps>0?(d.step/d.total_steps*100):0;
  document.getElementById('prog-fill').style.width=pct.toFixed(1)+'%';
  document.getElementById('prog-step').textContent=`Step ${d.step.toLocaleString()} / ${d.total_steps.toLocaleString()}`;
  document.getElementById('prog-phase').textContent=pct>0?`${pct.toFixed(1)}% — ${d.phase}`:d.phase;

  const m=d.metrics||{};
  tx('m-ar',m.ar!=null?(m.ar*100).toFixed(1)+'%':'—');
  tx('m-profit',m.profit!=null?'$'+m.profit.toFixed(1):'—');
  tx('m-wait',m.wait!=null?m.wait.toFixed(2)+'m':'—');
  tx('m-occ',m.occ!=null?(m.occ*100).toFixed(1)+'%':'—');
  tx('m-idle',m.idle_frac!=null?(m.idle_frac*100).toFixed(1)+'%':'—');
  tx('m-km',m.km!=null?m.km.toFixed(2):'—');
  tx('m-loss',m.loss!=null?m.loss.toFixed(4):'—');
  tx('m-eps',m.eps!=null?m.eps.toFixed(3):'—');
  tx('m-qmax',m.qmax!=null?m.qmax.toFixed(2):'—');
  tx('m-val-ar',m.val_ar!=null?(m.val_ar*100).toFixed(1)+'%':'—');
  tx('m-val-profit',m.val_profit!=null?'$'+m.val_profit.toFixed(1):'—');
  tx('m-val-wait',m.val_wait!=null?m.val_wait.toFixed(2)+'m':'—');
  tx('m-val-occ',m.val_occ!=null?(m.val_occ*100).toFixed(1)+'%':'—');

  const cfg=d.config||{};
  tx('val-interval',cfg.eval_interval!=null?cfg.eval_interval:'—');
  tx('val-steps',cfg.eval_steps!=null?cfg.eval_steps:'—');
  tx('val-seed',cfg.eval_seed!=null?cfg.eval_seed:'—');

  const h=d.history||{};
  pushCh(charts.ar,h.ar);pushCh(charts.profit,h.profit);pushCh(charts.wait,h.wait);
  pushCh(charts.loss,h.loss);pushCh(charts.eps,h.eps);pushCh(charts.qmax,h.qmax);
  pushCh(charts.val_ar,h.val_ar);pushCh(charts.val_profit,h.val_profit);pushCh(charts.val_wait,h.val_wait);pushCh(charts.val_occ,h.val_occ);

  document.getElementById('btn-train').disabled=d.running;
  document.getElementById('btn-pause').disabled=!d.running;
  document.getElementById('btn-stop').disabled=!d.running;
  document.getElementById('btn-pause').textContent=d.paused?'▶ Resume':'⏸ Pause';
}
function tx(id,v){const e=document.getElementById(id);if(e)e.textContent=v;}

// ═══ Train Controls ═══
document.getElementById('btn-train').onclick=async()=>{
  const btn=document.getElementById('btn-train');
  btn.disabled=true; btn.textContent='⏳ Starting...';
  const r=await api('/api/train/start','POST',buildConfig());
  if(r&&r.error){
    toast(r.error,'error');
    btn.disabled=false; btn.textContent='▶ Start';
  } else if(!r){
    // Response lost but training may have started — poll to check
    toast('Connecting to server...','info');
    setTimeout(async()=>{
      const st=await api('/api/train/status');
      if(st&&st.running){toast('Training started!','success');onUpdate(st);}
      else{btn.disabled=false;btn.textContent='▶ Start';toast('Failed to start — check server','error');}
    },1500);
  } else {
    toast('Training started!','success');
  }
};
document.getElementById('btn-pause').onclick=()=>api('/api/train/pause','POST');
document.getElementById('btn-stop').onclick=()=>api('/api/train/stop','POST');

// ═══ Save — always visible inline ═══
document.getElementById('btn-save').onclick=async()=>{
  const name=document.getElementById('save-name').value.trim()||`Model_${Date.now()}`;
  const r=await api('/api/models/save','POST',{name,description:''});
  if(r&&!r.error){
    toast(`"${name}" saved!`,'success');
    document.getElementById('save-name').value='';
  } else {
    toast(r?.error||'Save failed — train a model first','error');
  }
};

// ═══ Models Tab ═══
document.getElementById('btn-refresh-models').onclick=refreshModels;

async function refreshModels(){
  const models=await api('/api/models')||[];
  const c=document.getElementById('model-list');
  document.getElementById('model-count').textContent=`${models.length} model${models.length!==1?'s':''}`;
  if(!models.length){c.innerHTML=`<div class="empty-state" style="grid-column:1/-1"><div class="empty-icon">💾</div><div class="empty-text">No saved models yet</div><div class="empty-sub">Train → Save → Models appear here</div></div>`;return;}
  c.innerHTML=models.map(m=>{
    const fm=m.final_metrics||{},cfg=m.config||{};
    return `<div class="model-card fade-in">
      <div class="mc-name">${esc(m.name)}</div>
      <div class="mc-date">📅 ${m.timestamp} · ⏱ ${m.steps_trained} steps · 🔧 ${cfg.algorithm||'full'}</div>
      ${m.description?`<div class="mc-desc">"${esc(m.description)}"</div>`:''}
      <div class="mc-stats">
        <div class="mc-stat"><div class="mc-l">AR</div><div class="mc-v">${fm.ar!=null?(fm.ar*100).toFixed(1)+'%':'—'}</div></div>
        <div class="mc-stat"><div class="mc-l">Profit</div><div class="mc-v">${fm.profit!=null?'$'+fm.profit.toFixed(0):'—'}</div></div>
        <div class="mc-stat"><div class="mc-l">Wait</div><div class="mc-v">${fm.wait!=null?fm.wait.toFixed(1)+'m':'—'}</div></div>
        <div class="mc-stat"><div class="mc-l">Loss</div><div class="mc-v">${fm.loss!=null?fm.loss.toFixed(3):'—'}</div></div>
      </div>
      <div style="font-size:9px;color:var(--t3);margin-top:6px">Fleet:${cfg.fleet_size||'?'} · LR:${cfg.learning_rate?cfg.learning_rate.toExponential(1):'?'} · γ:${cfg.gamma||'?'}</div>
      <div class="mc-actions">
        <button class="btn btn-primary btn-sm" onclick="loadModel('${m.id}','${esc(m.name)}')">📂 Load</button>
        <button class="btn btn-danger btn-sm" onclick="deleteModel('${m.id}')">🗑 Delete</button>
      </div></div>`;
  }).join('');
}

window.loadModel=async(id,name)=>{const r=await api('/api/models/load','POST',{id});if(r&&!r.error)toast(`Loaded: ${name}`,'success');else toast(r?.error||'Load failed','error');};
window.deleteModel=async id=>{if(!confirm('Delete this model?'))return;await api(`/api/models/${id}`,'DELETE');toast('Deleted','success');refreshModels();refreshCompareChips();};

// ═══ Compare Tab ═══
async function refreshCompareChips(){
  const models=await api('/api/models')||[];
  const c=document.getElementById('cmp-chips');
  if(!models.length){c.innerHTML=`<div style="color:var(--t3);font-size:11px">No saved models yet</div>`;updateCmpBtn();return;}
  c.innerHTML=models.map(m=>{
    const a=selectedModels.has(m.id)?' active':'';
    const fm=m.final_metrics||{};
    return `<div class="cmp-chip${a}" data-id="${m.id}" onclick="toggleCmp('${m.id}')">${esc(m.name)} ${fm.ar!=null?'('+((fm.ar*100).toFixed(0))+'%)':''}</div>`;
  }).join('');
  updateCmpBtn();
}

window.toggleCmp=id=>{
  if(selectedModels.has(id))selectedModels.delete(id);else selectedModels.add(id);
  document.querySelectorAll('#cmp-chips .cmp-chip').forEach(c=>c.classList.toggle('active',selectedModels.has(c.dataset.id)));
  updateCmpBtn();
};

function updateCmpBtn(){
  const n=selectedModels.size;
  document.getElementById('btn-run-compare').disabled=n<2;
  document.getElementById('cmp-sel-count').textContent=n<2?`Select ${2-n} more`:`${n} selected ✓`;
}

document.getElementById('btn-run-compare').onclick=async()=>{
  if(selectedModels.size<2)return;
  const btn=document.getElementById('btn-run-compare');
  btn.textContent='⏳ Evaluating...';btn.disabled=true;
  const r=await api('/api/compare','POST',{model_ids:[...selectedModels],eval_steps:parseInt(document.getElementById('cmp-steps').value)});
  btn.textContent='🔬 Run Comparison';btn.disabled=selectedModels.size<2;
  if(!r||r.error){toast('Failed','error');return;}
  renderCmp(r);
};

function renderCmp(res){
  document.getElementById('compare-results').style.display='block';
  const names=Object.keys(res),cols=['#6366f1','#10b981','#f59e0b','#ef4444','#06b6d4','#8b5cf6','#f472b6','#a3e635'];
  const cd=document.getElementById('compare-charts');
  cd.innerHTML=['ar','profit','wait','occ'].map(k=>`<div class="chart-wrap" style="height:180px"><canvas id="cmp-${k}"></canvas></div>`).join('');
  const T={ar:'Accept Rate',profit:'Profit ($)',wait:'Wait (min)',occ:'Occupancy'};
  ['ar','profit','wait','occ'].forEach(k=>{
    new Chart(document.getElementById('cmp-'+k),{
      type:'bar',data:{labels:names,datasets:[{data:names.map(n=>res[n][k]),backgroundColor:names.map((_,i)=>cols[i%cols.length]+'cc'),borderColor:names.map((_,i)=>cols[i%cols.length]),borderWidth:1}]},
      options:{responsive:true,maintainAspectRatio:false,animation:{duration:400},plugins:{legend:{display:false},title:{display:true,text:T[k],color:'#94a3b8',font:{size:11,weight:'600'}}},
        scales:{x:{ticks:{color:'#94a3b8',font:{size:9}}},y:{ticks:{color:'#64748b',font:{size:8}},grid:{color:'rgba(255,255,255,.04)'},beginAtZero:true}}}
    });
  });
  const bAR=Math.max(...names.map(n=>res[n].ar)),bP=Math.max(...names.map(n=>res[n].profit)),bW=Math.min(...names.map(n=>res[n].wait));
  document.getElementById('cmp-tbody').innerHTML=names.map((n,i)=>{const r2=res[n];return `<tr><td style="color:${cols[i%cols.length]};font-weight:700">${n}</td><td class="${r2.ar===bAR?'best':''}">${(r2.ar*100).toFixed(1)}%</td><td class="${r2.profit===bP?'best':''}">$${r2.profit.toFixed(2)}</td><td class="${r2.wait===bW?'best':''}">${r2.wait.toFixed(2)}m</td><td>${r2.occ!=null?(r2.occ*100).toFixed(1)+'%':'—'}</td><td>${r2.idle_frac!=null?(r2.idle_frac*100).toFixed(1)+'%':'—'}</td><td>${r2.km!=null?r2.km.toFixed(2):'—'}</td><td>${r2.steps_trained}</td></tr>`;}).join('');
}

document.getElementById('btn-run-ablation').onclick = async () => {
  const btn = document.getElementById('btn-run-ablation');
  const status = document.getElementById('ablation-status');
  const out = document.getElementById('ablation-results');
  btn.disabled = true;
  status.textContent = 'Running...';
  out.innerHTML = '';
  const r = await api('/api/ablation', 'POST', {
    train_steps: parseInt(document.getElementById('cfg-ablation-train').value) || 300,
    eval_steps: parseInt(document.getElementById('cfg-ablation-eval').value) || 150,
  });
  btn.disabled = false;
  if (!r || r.error) {
    status.textContent = r?.error || 'Failed';
    toast(r?.error || 'Ablation failed', 'error');
    return;
  }
  status.textContent = 'Done';
  const rows = Object.entries(r.components || {}).map(([name, m]) =>
    `<tr><td>${esc(name)}</td><td>${(m.ar*100).toFixed(1)}%</td><td>$${m.profit.toFixed(2)}</td><td>${m.wait.toFixed(2)}m</td><td>${(m.occ*100).toFixed(1)}%</td><td>${(m.idle_frac*100).toFixed(1)}%</td><td>${m.km.toFixed(2)}</td></tr>`
  ).join('');
  const sensitivity = Object.entries(r.sensitivity || {}).map(([name, m]) =>
    `<tr><td>${esc(name)}</td><td>${(m.ar*100).toFixed(1)}%</td><td>$${m.profit.toFixed(2)}</td><td>${m.wait.toFixed(2)}m</td></tr>`
  ).join('');
  out.innerHTML = `
    <div style="margin-bottom:8px"><a href="${r.plot}" target="_blank" style="color:#38bdf8">Open ablation plot</a></div>
    <table class="compare-table" style="width:100%;margin-bottom:10px">
      <thead><tr><th>Config</th><th>Accept Rate</th><th>Profit</th><th>Wait</th><th>Occupancy</th><th>Cruising</th><th>Distance</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <table class="compare-table" style="width:100%">
      <thead><tr><th>Sensitivity</th><th>Accept Rate</th><th>Profit</th><th>Wait</th></tr></thead>
      <tbody>${sensitivity}</tbody>
    </table>`;
};

// ═══════════════════════════════════════════════════════════════
// SIMULATE TAB — Real vehicle visualization from /api/sim/state
// ═══════════════════════════════════════════════════════════════
const canvas = document.getElementById('city');
const ctx = canvas ? canvas.getContext('2d') : null;

document.getElementById('btn-sim-run').onclick = async () => {
  const steps = parseInt(document.getElementById('sim-steps').value) || 200;
  document.getElementById('btn-sim-run').disabled = true;
  document.getElementById('btn-sim-stop').disabled = false;
  document.getElementById('sim-overlay').style.display = 'none';
  simRunning = true;
  document.getElementById('sim-log').innerHTML = '<div style="color:var(--a2)">▶ Starting simulation...</div>';

  // Update model pill to show "Training/New" since we're starting fresh training
  document.getElementById('sim-model-pill').textContent = '⚙️ Training (New)';
  document.getElementById('sim-model-pill').style.background = 'linear-gradient(135deg,#f59e0b,#ea580c)';

  // First stop any leftover training
  await api('/api/train/stop', 'POST');
  // Small delay to let server reset
  await new Promise(r => setTimeout(r, 500));

  // Start training which also drives the simulation
  const startRes = await api('/api/train/start', 'POST', { ...buildConfig(), steps });
  if (!startRes || startRes.error) {
    toast(startRes?.error || 'Failed to start simulation', 'error');
    document.getElementById('btn-sim-run').disabled = false;
    document.getElementById('btn-sim-stop').disabled = true;
    simRunning = false;
    document.getElementById('sim-model-pill').textContent = 'No model active';
    document.getElementById('sim-model-pill').style.background = 'linear-gradient(135deg,var(--a3),var(--a2))';
    return;
  }
  document.getElementById('sim-log').innerHTML = '<div style="color:var(--a2)">▶ Simulation running...</div>';

  // Wait a moment for the training thread to begin
  await new Promise(r => setTimeout(r, 800));

  // Poll both status AND vehicle positions
  simInterval = setInterval(async () => {
    if (!simRunning) { clearInterval(simInterval); return; }
    const [st, simState] = await Promise.all([
      api('/api/train/status'),
      api('/api/sim/state'),
    ]);
    if (!st) return;

    // Progress
    const pct = st.total_steps > 0 ? st.step / st.total_steps * 100 : 0;
    document.getElementById('sim-prog').style.width = pct + '%';
    document.getElementById('sim-step-text').textContent = `Step ${st.step} / ${st.total_steps}`;

    // Metrics
    const m = st.metrics || {};
    const sm = (simState && simState.metrics) ? simState.metrics : {};
    tx('sm-ar', m.ar != null ? (m.ar * 100).toFixed(1) + '%' : '—');
    tx('sm-profit', m.profit != null ? '$' + m.profit.toFixed(0) : '—');
    tx('sm-wait', m.wait != null ? m.wait.toFixed(1) + 'm' : '—');
    tx('sm-occ', sm.occ != null ? (sm.occ * 100).toFixed(1) + '%' : '—');
    tx('sm-idle', sm.idle_frac != null ? (sm.idle_frac * 100).toFixed(1) + '%' : '—');
    tx('sm-km', sm.km != null ? sm.km.toFixed(2) : '—');
    tx('sm-trips', st.step || '—');

    // Draw city grid and heatmap with REAL vehicle data
    if (simState && simState.vehicles && simState.vehicles.length > 0) {
      window.lastSimState = simState;
      drawCityGrid(simState);
      drawHeatmap(simState);
    }

    // Update canvas/grid info
    if (simState) {
      tx('sim-time-info', `t=${simState.t || 0}`);
      tx('sim-vehicle-count', `${(simState.vehicles || []).length} vehicles`);
    }

    // Fleet status panel
    if (simState && simState.vehicles) {
      const vehs = simState.vehicles;
      const idle = vehs.filter(v => v.status === 'idle').length;
      const carrying = vehs.filter(v => (v.passengers || 0) > 0).length;
      const dispatching = vehs.filter(v => v.status === 'dispatching').length;
      const pending = (simState.pending || []).length;
      tx('sm-idle-count', idle);
      tx('sm-carry-count', carrying);
      tx('sm-dispatch-count', dispatching);
      tx('sm-pending-count', pending);
    }

    // Event log
    if (simState && simState.events && simState.events.length) {
      const log = document.getElementById('sim-log');
      const events = simState.events;
      const latest = events.slice(-20).reverse(); // Show more history
      let html = '';
      for (const ev of latest) {
        let color = 'var(--t2)';
        if (ev.type === 'accept') color = 'var(--a2)';
        else if (ev.type === 'reject') color = 'var(--a4)';
        else if (ev.type === 'dqn') color = '#818cf8';
        else if (ev.type === 'sys') color = 'var(--t3)';
        else if (ev.type === 'cancel') color = 'var(--a4)';
        else if (ev.type === 'pickup') color = 'var(--a2)';
        else if (ev.type === 'dropoff') color = '#818cf8';
        
        html += `<div style="color:${color};border-bottom:1px solid var(--bd);padding:2px 0;font-size:9px">t=${ev.t} ${ev.msg || ev.type}</div>`;
      }
      html += `<div style="color:var(--a3);padding:2px 0;font-size:9px">Step ${st.step} · Total: ${simState.total_req||0} req</div>`;
      log.innerHTML = html;
    }

    // Stop check — only stop if phase is 'done' or 'error', NOT 'starting'
    if (st.phase === 'done' || st.phase === 'error' || (!st.running && st.step > 0 && st.phase !== 'starting')) {
      clearInterval(simInterval);
      document.getElementById('btn-sim-run').disabled = false;
      document.getElementById('btn-sim-stop').disabled = true;
      simRunning = false;
      document.getElementById('sim-log').innerHTML += '<div style="color:var(--a3);margin-top:4px">⏹ Simulation complete</div>';
    }
  }, 500);
};

document.getElementById('btn-sim-stop').onclick = async () => {
  simRunning = false;
  await api('/api/train/stop', 'POST');
  if (simInterval) clearInterval(simInterval);
  document.getElementById('btn-sim-run').disabled = false;
  document.getElementById('btn-sim-stop').disabled = true;
};

function drawCityGrid(simState) {
  if (!simState || !simState.vehicles) return;
  const canvas = document.getElementById('city');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cw = w / GRID_SIZE, ch = h / GRID_SIZE;

  // 1. Demand Heatmap background
  const dg = l2g(simState.demand);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, 0, w, h);
  
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const intensity = dg[r][c] * 0.4;
      ctx.fillStyle = `rgba(245,158,11,${intensity})`;
      ctx.fillRect(c * cw, r * ch, cw, ch);
    }
  }

  // 2. Grid lines - darker and more visible
  ctx.strokeStyle = 'rgba(148,163,184,0.15)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= GRID_SIZE; i++) {
    ctx.beginPath(); ctx.moveTo(i * cw, 0); ctx.lineTo(i * cw, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * ch); ctx.lineTo(w, i * ch); ctx.stroke();
  }

  // 3. Pending requests (cyan diamonds)
  const pn = simState.pending || [];
  for (const r of pn) {
    ctx.fillStyle = '#06b6d4';
    ctx.beginPath(); ctx.arc((r.c + 0.5) * cw, (r.r + 0.5) * ch, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0891b2';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // 4. Vehicles
  const vs = simState.vehicles;
  for (const v of vs) {
    const vid = v.id || v.vid;
    const tx = (v.c + 0.5) * cw, ty = (v.r + 0.5) * ch;
    
    // Interpolate position for smooth transit
    if (!vehPositions[vid]) vehPositions[vid] = {x: tx, y: ty};
    const pos = vehPositions[vid];
    pos.x += (tx - pos.x) * 0.2; // Smooth glide toward target
    pos.y += (ty - pos.y) * 0.2;
    const vx = pos.x, vy = pos.y;

    // Route lines
    if (v.route && v.route.length > 0) {
     ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(vx, vy);
      for (const s of v.route) {
        ctx.lineTo((s.c + 0.5) * cw, (s.r + 0.5) * ch);
      }
      ctx.stroke();
    }

    const col = VEH_COLORS[v.status] || '#3b82f6';
    const px = v.passengers || v.pax || 0;
    const rd = 6 + px * 2;
    
    ctx.shadowBlur = 10;
    ctx.shadowColor = col;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(vx, vy, rd, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;

    // Passenger count
    if (px > 0) {
      ctx.fillStyle = '#fff';
      ctx.font = `bold ${Math.max(9, rd - 1)}px sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(px, vx, vy);
    }

    // Route stops
    if (v.route) {
      for (const s of v.route) {
        const sx = (s.c + 0.5) * cw, sy = (s.r + 0.5) * ch;
        ctx.fillStyle = s.type === 'pickup' ? '#10b981' : s.type === 'dropoff' ? '#ef4444' : '#8b5cf6';
        ctx.beginPath(); ctx.arc(sx, sy, 3, 0, Math.PI * 2); ctx.fill();
      }
    }
  }
  
  // Legend text
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px sans-serif';
  ctx.fillText(`t=${simState.t || 0}  ${GRID_SIZE}×${GRID_SIZE}  ${vs.length} vehicles`, 12, h - 8);
}

function drawHeatmap(simState) {
  const hm = document.getElementById('sim-hm');
  if (!hm) return;
  const hmx = hm.getContext('2d');
  const w = hm.width, h = hm.height;
  const qw = w / GRID_SIZE, qh = h / GRID_SIZE;
  hmx.clearRect(0, 0, w, h);
  const qg = l2g(simState.zone_q);
  for (let r = 0; r < GRID_SIZE; r++) {
    for (let c = 0; c < GRID_SIZE; c++) {
      const q = qg[r][c];
      const i = Math.floor(q * 255);
      hmx.fillStyle = `rgb(${Math.floor(i * 0.4)},${Math.floor(i * 0.6)},${i})`;
      hmx.fillRect(c * qw, r * qh, qw, qh);
    }
  }
}

canvas.onmousemove = e => {
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const gx = Math.floor(mx / (r.width / GRID_SIZE)), gy = Math.floor(my / (r.height / GRID_SIZE));
  
  const vs = (simRunning && simInterval) ? [] : []; // This is a placeholder, I need the actual state
  // To make this work, I should store the last simState globally
  const currentSimState = window.lastSimState; 
  if (!currentSimState) return;

  const vehicles = currentSimState.vehicles || [];
  const near = vehicles.find(v => Math.abs(v.c - gx) <= 1 && Math.abs(v.r - gy) <= 1);
  const tip = document.getElementById('sim-tip');
  if (!tip) return;
  
  tip.style.display = 'block';
  tip.style.left = (mx + 12) + 'px';
  tip.style.top = (my + 12) + 'px';

  if (near) {
    tip.innerHTML = `<b style="color:var(--a3)">V${near.id}</b> ${near.status}<br>Pax: ${near.pax} | $${near.profit.toFixed(0)}`;
  } else {
    const qg = l2g(currentSimState.zone_q);
    const q = (qg[Math.min(gy, GRID_SIZE - 1)] || [])[Math.min(gx, GRID_SIZE - 1)] || 0;
    tip.innerHTML = `Zone(${gx},${gy}) Q=${q.toFixed(3)}`;
  }
};
canvas.onmouseleave = () => {
  const tip = document.getElementById('sim-tip');
  if (tip) tip.style.display = 'none';
};

// ═══ Utilities ═══
async function api(url, method = 'GET', body = null) {
  try {
    const o = { method, headers: {} };
    if (body) { o.headers['Content-Type'] = 'application/json'; o.body = JSON.stringify(body); }
    const r = await fetch(API + url, o);
    return await r.json();
  } catch (e) { return null; }
}
function toast(msg, type = 'success') {
  const t = document.createElement('div'); t.className = 'toast ' + type; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 3000);
}
function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }

// ═══ Init ═══
window.onload = () => {
  initCharts();
  api('/api/config').then(applyConfigDefaults);
  connectSSE();
  startPolling(); // Always poll as fallback
  api('/api/train/status').then(d => { if (d) onUpdate(d); });
};
