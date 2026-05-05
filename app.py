"""
NBA Predictor v3 — PostgreSQL Edition
Paste this entire file as app.py on GitHub.
Set these Railway environment variables:
  BALLDONTLIE_KEY = your balldontlie API key
  DATABASE_URL    = auto-set by Railway PostgreSQL plugin
  ODDS_API_KEY    = optional, from the-odds-api.com
"""

from flask import Flask, jsonify
import requests, json, time, threading, math, os
from datetime import datetime, date
import numpy as np

# ── PostgreSQL ───────────────────────────────────────────────────────────────
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DB_URL = os.environ.get("DATABASE_URL", "")
    USE_DB = bool(DB_URL)
except ImportError:
    USE_DB = False

# ── ML ───────────────────────────────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    import warnings; warnings.filterwarnings('ignore')
    HAS_ML = True
except ImportError:
    HAS_ML = False

app = Flask(__name__)

# ── CONFIG ───────────────────────────────────────────────────────────────────
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
API_KEY      = os.environ.get("BALLDONTLIE_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
DEFAULT_RATING = 1500
K_FACTOR       = 20
HOME_ADV_ELO   = 100

# ── HTML PAGE ────────────────────────────────────────────────────────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>🏀 NBA Quant Predictor</title>
<style>
:root{
  --bg:#08080f;--card:#111118;--card2:#18181f;--border:#252530;
  --accent:#ff6b35;--accent2:#4fc3f7;--green:#4caf50;--red:#ef5350;
  --gold:#ffd700;--purple:#9c6fff;--text:#e8e8f0;--muted:#6868a0;--live:#ff3333;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
header{background:linear-gradient(135deg,#0a0a1a,#150a28);border-bottom:1px solid var(--border);
  padding:18px 28px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:40px;height:40px;background:var(--accent);border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:0 0 24px rgba(255,107,53,.4)}
.logo h1{font-size:20px;font-weight:800;letter-spacing:-.5px}
.logo span{color:var(--accent)}
.header-meta{font-size:11px;color:var(--muted);text-align:right}
.live-dot{display:inline-block;width:8px;height:8px;background:var(--live);
  border-radius:50%;animation:pulse 1.5s infinite;margin-right:5px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
nav{display:flex;gap:2px;padding:0 28px;background:var(--bg);border-bottom:1px solid var(--border);overflow-x:auto}
.tab{padding:12px 18px;background:none;border:none;color:var(--muted);font-size:13px;
  font-weight:600;cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover{color:var(--text)}
.wrap{max-width:1380px;margin:0 auto;padding:22px 28px}
.page{display:none}.page.active{display:block}
.pills{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.pill{background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:6px 14px;font-size:12px;display:flex;align-items:center;gap:7px}
.pill .lbl{color:var(--muted)}.pill .val{font-weight:700}
.pill .val.live{color:var(--live)}.pill .val.grn{color:var(--green)}.pill .val.ml{color:var(--purple)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}
.gcard{background:var(--card);border:1px solid var(--border);border-radius:14px;
  overflow:hidden;transition:transform .2s,border-color .2s}
.gcard:hover{transform:translateY(-2px);border-color:var(--accent)}
.gcard.live{border-color:var(--live);box-shadow:0 0 24px rgba(255,51,51,.12)}
.gcard.final{opacity:.72}
.ch{padding:10px 14px;background:var(--card2);display:flex;justify-content:space-between;
  align-items:center;border-bottom:1px solid var(--border)}
.gstatus{font-size:11px;font-weight:700;letter-spacing:.5px;padding:3px 9px;border-radius:10px}
.gstatus.live{background:rgba(255,51,51,.18);color:var(--live);animation:pulse 1.5s infinite}
.gstatus.final{background:rgba(104,104,160,.18);color:var(--muted)}
.gstatus.upcoming{background:rgba(79,195,247,.12);color:var(--accent2)}
.cbadge{font-size:10px;padding:3px 9px;border-radius:10px;font-weight:600}
.cbadge.High{background:rgba(76,175,80,.18);color:var(--green)}
.cbadge.Medium{background:rgba(255,193,7,.18);color:#ffc107}
.cbadge.Low{background:rgba(239,83,80,.18);color:var(--red)}
.cbadge.NA{background:rgba(104,104,160,.18);color:var(--muted)}
.sb{padding:18px 14px;display:flex;align-items:center;gap:10px}
.tb{flex:1;text-align:center}
.tabbr{font-size:26px;font-weight:900;letter-spacing:-1px;line-height:1}
.tname{font-size:10px;color:var(--muted);margin-top:3px}
.tscore{font-size:40px;font-weight:900;margin-top:6px;letter-spacing:-2px}
.tscore.win{color:var(--accent)}
.vs{display:flex;flex-direction:column;align-items:center;gap:3px;color:var(--muted);font-size:11px;min-width:36px}
.prob-wrap{padding:0 14px 14px}
.prob-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}
.prob-row{display:flex;align-items:center;gap:7px}
.prob-pct{font-size:15px;font-weight:800;min-width:42px;text-align:center}
.prob-pct.h{color:var(--accent2)}.prob-pct.a{color:var(--accent)}
.pbar{flex:1;height:10px;background:var(--border);border-radius:10px;overflow:hidden}
.pfill{height:100%;border-radius:10px;transition:width .8s;background:linear-gradient(90deg,var(--accent2),var(--accent))}
.prob-teams{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:3px}
.model-row{padding:0 14px 12px;display:flex;gap:8px;flex-wrap:wrap}
.model-badge{font-size:11px;padding:3px 9px;border-radius:8px;font-weight:600;border:1px solid}
.model-badge.elo{color:var(--accent2);border-color:rgba(79,195,247,.3);background:rgba(79,195,247,.08)}
.model-badge.ml{color:var(--purple);border-color:rgba(156,111,255,.3);background:rgba(156,111,255,.08)}
.model-badge.rule{color:var(--muted);border-color:var(--border);background:var(--card2)}
.wbanner{margin:0 14px 12px;background:linear-gradient(135deg,rgba(255,107,53,.08),rgba(79,195,247,.08));
  border:1px solid var(--border);border-radius:9px;padding:9px 12px;display:flex;align-items:center;gap:9px;font-size:12px}
.wlbl{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.wname{font-weight:700;font-size:13px;color:var(--gold)}
.ftable{margin:0 14px 14px;border:1px solid var(--border);border-radius:9px;overflow:hidden}
.fhdr{background:var(--card2);display:grid;grid-template-columns:1fr 64px 64px;
  padding:7px 11px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.frow{display:grid;grid-template-columns:1fr 64px 64px;padding:7px 11px;font-size:12px;
  border-top:1px solid var(--border);align-items:center}
.frow:nth-child(even){background:rgba(255,255,255,.018)}
.fname{color:var(--muted)}.fval{text-align:center;font-weight:600;font-family:monospace}
.fval.eh{color:var(--accent2)}.fval.ea{color:var(--accent)}
.empty{text-align:center;padding:70px 20px;color:var(--muted)}
.empty .ico{font-size:56px;margin-bottom:14px}
.loader{display:flex;align-items:center;justify-content:center;padding:70px;flex-direction:column;gap:14px;color:var(--muted)}
.spin{width:38px;height:38px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.etable{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.ehdr{display:grid;grid-template-columns:44px 1fr 90px 90px;padding:11px 16px;
  background:var(--card2);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:700;border-bottom:1px solid var(--border)}
.erow{display:grid;grid-template-columns:44px 1fr 90px 90px;padding:11px 16px;
  font-size:13px;border-bottom:1px solid var(--border);align-items:center;transition:background .15s}
.erow:hover{background:rgba(255,255,255,.025)}
.erow:last-child{border-bottom:none}
.erank{color:var(--muted);font-weight:700;font-size:12px}
.erank.top3{color:var(--gold)}
.ename{font-weight:600}
.ename small{display:block;color:var(--muted);font-size:10px;font-weight:400}
.erate{font-family:monospace;font-weight:700;font-size:14px;text-align:right}
.ediff{font-family:monospace;font-weight:600;text-align:right}
.ediff.pos{color:var(--green)}.ediff.neg{color:var(--red)}
.bstats{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:12px;margin-bottom:22px}
.bstat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.bstat-val{font-size:28px;font-weight:900;line-height:1}
.bstat-lbl{font-size:11px;color:var(--muted);margin-top:4px}
.bstat-sub{font-size:11px;color:var(--muted);margin-top:2px}
.bstat-val.grn{color:var(--green)}.bstat-val.red{color:var(--red)}.bstat-val.gold{color:var(--gold)}
.cal-table{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:22px}
.cal-hdr{display:grid;grid-template-columns:90px 1fr 1fr 70px 70px;padding:10px 14px;
  background:var(--card2);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:700;border-bottom:1px solid var(--border)}
.cal-row{display:grid;grid-template-columns:90px 1fr 1fr 70px 70px;padding:10px 14px;
  font-size:13px;border-bottom:1px solid var(--border);align-items:center}
.cal-row:last-child{border-bottom:none}
.gap-val{font-weight:700}.gap-val.pos{color:var(--green)}.gap-val.neg{color:var(--red)}
.gt-table{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.gt-hdr{display:grid;grid-template-columns:100px 1fr 1fr 70px 70px 60px;padding:10px 14px;
  background:var(--card2);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:700;border-bottom:1px solid var(--border)}
.gt-row{display:grid;grid-template-columns:100px 1fr 1fr 70px 70px 60px;padding:10px 14px;
  font-size:12px;border-bottom:1px solid var(--border);align-items:center}
.gt-row:last-child{border-bottom:none}
.gt-row:hover{background:rgba(255,255,255,.02)}
.correct{color:var(--green);font-weight:700}.wrong{color:var(--red);font-weight:700}
.mcard{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;margin-bottom:16px}
.mcard h3{color:var(--accent);margin-bottom:12px;font-size:15px}
.fi-bar-wrap{display:flex;align-items:center;gap:10px;margin-bottom:7px}
.fi-name{font-size:12px;color:var(--muted);min-width:120px}
.fi-bar{flex:1;height:8px;background:var(--border);border-radius:8px;overflow:hidden}
.fi-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--accent2));border-radius:8px}
.fi-val{font-size:12px;font-weight:700;min-width:40px;text-align:right;color:var(--accent2)}
.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.metric-box{background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center}
.metric-val{font-size:20px;font-weight:800}
.metric-lbl{font-size:10px;color:var(--muted);margin-top:3px;text-transform:uppercase;letter-spacing:.5px}
.infobox{background:rgba(79,195,247,.07);border:1px solid rgba(79,195,247,.2);
  border-radius:10px;padding:11px 15px;font-size:13px;color:var(--accent2);margin-bottom:18px;line-height:1.5}
.warnbox{background:rgba(255,193,7,.07);border:1px solid rgba(255,193,7,.2);
  border-radius:10px;padding:11px 15px;font-size:13px;color:#ffc107;margin-bottom:18px;line-height:1.5}
.section-title{font-size:16px;font-weight:700;margin-bottom:14px;color:var(--text)}
@media(max-width:600px){
  .wrap{padding:14px}nav{padding:0 14px}.grid{grid-template-columns:1fr}
  header{padding:12px 14px}.metric-grid{grid-template-columns:repeat(2,1fr)}
  .gt-hdr,.gt-row{grid-template-columns:80px 1fr 1fr 60px 60px}
}
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">🏀</div>
    <div>
      <h1>NBA <span>Quant</span> Predictor</h1>
      <div style="font-size:10px;color:var(--muted)">Elo · ML Ensemble · Kelly Criterion · Live Backtest</div>
    </div>
  </div>
  <div class="header-meta">
    <div><span class="live-dot"></span>Auto-refreshes every 90s</div>
    <div id="last-update">Loading...</div>
    <div id="ml-status" style="margin-top:2px"></div>
  </div>
</header>
<nav>
  <button class="tab active" onclick="showTab('games',this)">🏀 Today's Games</button>
  <button class="tab" onclick="showTab('backtest',this)">🔬 Backtest</button>
  <button class="tab" onclick="showTab('elo',this)">📈 Elo Rankings</button>
  <button class="tab" onclick="showTab('model',this)">🤖 ML Model</button>
  <button class="tab" onclick="showTab('about',this)">ℹ️ How It Works</button>
</nav>
<div class="wrap">

<div id="pg-games" class="page active">
  <div class="pills">
    <div class="pill"><span class="lbl">Games</span><span class="val" id="p-total">—</span></div>
    <div class="pill"><span class="lbl">Live</span><span class="val live" id="p-live">—</span></div>
    <div class="pill"><span class="lbl">Upcoming</span><span class="val grn" id="p-up">—</span></div>
    <div class="pill"><span class="lbl">Final</span><span class="val" id="p-fin">—</span></div>
    <div class="pill"><span class="lbl">ML Model</span><span class="val ml" id="p-ml">—</span></div>
    <div class="pill"><span class="lbl">Games in DB</span><span class="val" id="p-db">—</span></div>
  </div>
  <div id="games-wrap"><div class="loader"><div class="spin"></div><div>Fetching live data...</div></div></div>
</div>

<div id="pg-backtest" class="page">
  <div class="infobox">🔬 Results from every game the model has predicted since launch. Data persists permanently in the database — grows every game day.</div>
  <div id="backtest-wrap"><div class="loader"><div class="spin"></div><div>Running backtest...</div></div></div>
</div>

<div id="pg-elo" class="page">
  <div class="infobox">📊 Elo ratings persist in the database and update after every completed game. Teams start at 1500.</div>
  <div id="elo-wrap"><div class="loader"><div class="spin"></div><div>Loading Elo ratings...</div></div></div>
</div>

<div id="pg-model" class="page">
  <div id="model-wrap"><div class="loader"><div class="spin"></div><div>Loading model data...</div></div></div>
</div>

<div id="pg-about" class="page">
  <div style="max-width:700px">
    <div class="mcard">
      <h3>🏗️ System Architecture</h3>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">Three independent models vote on each game, weighted by calibration (Brier score). All data persists in PostgreSQL.</p>
      <div style="margin-top:14px;display:flex;flex-direction:column;gap:10px">
        <div style="background:var(--card2);border-radius:8px;padding:12px;border:1px solid var(--border)">
          <div style="color:var(--accent2);font-weight:700;font-size:13px">Elo Rating System (35% weight)</div>
          <div style="color:var(--muted);font-size:12px;margin-top:4px">Self-updating rating based on results + margin of victory. Same method as FiveThirtyEight's NBA model.</div>
        </div>
        <div style="background:var(--card2);border-radius:8px;padding:12px;border:1px solid var(--border)">
          <div style="color:var(--purple);font-weight:700;font-size:13px">ML Ensemble (50% weight) — activates after 50 games</div>
          <div style="color:var(--muted);font-size:12px;margin-top:4px">Logistic Regression + Random Forest + Gradient Boosting. Each model weighted by 1/BrierScore.</div>
        </div>
        <div style="background:var(--card2);border-radius:8px;padding:12px;border:1px solid var(--border)">
          <div style="color:var(--muted);font-weight:700;font-size:13px">Rule-Based Logistic (15% weight)</div>
          <div style="color:var(--muted);font-size:12px;margin-top:4px">Hand-tuned formula using form, point differential, home court. Baseline before ML activates.</div>
        </div>
      </div>
    </div>
    <div class="mcard">
      <h3>📐 Kelly Criterion</h3>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">
        <code style="color:var(--accent2)">f* = (bp - q) / b</code> — optimal fraction of bankroll to bet given an edge over the market. We use half-Kelly to reduce variance. Only shown when market odds available via The Odds API.
      </p>
    </div>
    <div class="mcard">
      <h3>📏 Brier Score & Calibration</h3>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">
        Brier Score = mean((predicted_prob − actual)²). Perfect = 0.0, random = 0.25. Calibration checks if 70% predictions win 70% of the time.
      </p>
    </div>
    <div class="mcard">
      <h3>⚠️ Limitations</h3>
      <div style="font-size:13px;color:var(--muted);line-height:1.9">
        <div>• No injury data — a star sitting out invalidates ratings</div>
        <div>• ML needs 50+ games; Elo converges after ~200</div>
        <div>• No rest/travel fatigue modelled</div>
        <div>• Markets are efficient — real edges close quickly</div>
        <div>• Not betting advice — quantitative modelling exercise</div>
      </div>
    </div>
  </div>
</div>

</div>
<script>
function showTab(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('pg-' + name).classList.add('active');
  btn.classList.add('active');
  if (name === 'backtest') loadBacktest();
  if (name === 'elo') loadElo();
  if (name === 'model') loadModel();
}

async function loadGames() {
  try {
    const res = await fetch('/api/games');
    const data = await res.json();
    renderGames(data.games);
    document.getElementById('last-update').textContent = 'Updated ' + data.fetched_at;
    document.getElementById('p-ml').textContent = data.ml_active ? '✅ Active' : `⏳ ${data.games_in_history}/50`;
    document.getElementById('p-db').textContent = data.games_in_history;
    document.getElementById('ml-status').innerHTML = data.ml_active
      ? '<span style="color:var(--purple)">● ML Ensemble active</span>'
      : `<span style="color:var(--muted)">Collecting: ${data.games_in_history}/50 games</span>`;
  } catch(e) {
    document.getElementById('games-wrap').innerHTML = '<div class="empty"><div class="ico">⚠️</div><h2>Connection error</h2><p>Check API key.</p></div>';
  }
}

function renderGames(games) {
  if (!games || !games.length) {
    document.getElementById('games-wrap').innerHTML = '<div class="empty"><div class="ico">🌙</div><h2>No games today</h2><p>Check back on a game day.</p></div>';
    return;
  }
  const live = games.filter(g => g.status.includes('LIVE')).length;
  const fin = games.filter(g => g.status === 'Final').length;
  document.getElementById('p-total').textContent = games.length;
  document.getElementById('p-live').textContent = live;
  document.getElementById('p-up').textContent = games.length - live - fin;
  document.getElementById('p-fin').textContent = fin;
  document.getElementById('games-wrap').innerHTML = `<div class="grid">${games.map(buildCard).join('')}</div>`;
}

function buildCard(g) {
  const p = g.prediction;
  const isLive = g.status.includes('LIVE');
  const isFinal = g.status === 'Final';
  const sc = isLive ? 'live' : (isFinal ? 'final' : 'upcoming');
  const hwin = g.home_score > g.away_score;
  const awin = g.away_score > g.home_score;
  const cf = p.confidence === 'N/A' ? 'NA' : p.confidence;
  const mlBadge = p.ml_home_prob !== null ? `<div class="model-badge ml">ML: ${p.ml_home_prob}%</div>` : '';
  const factorRows = (p.factors||[]).map(f =>
    `<div class="frow"><div class="fname">${f.name}</div><div class="fval ${f.edge==='away'?'ea':''}">${f.away}</div><div class="fval ${f.edge==='home'?'eh':''}">${f.home}</div></div>`
  ).join('');
  const scores = (isFinal || isLive)
    ? `<div class="tscore ${awin?'win':''}">${g.away_score}</div></div><div class="vs"><span style="font-weight:700;font-size:13px">@</span>${isLive?'<span style="color:var(--live);font-size:9px;font-weight:700">LIVE</span>':''}</div><div class="tb"><div class="tabbr" style="color:var(--accent)">${g.home_abbr}</div><div class="tname">${g.home_team}</div><div class="tscore ${hwin?'win':''}">${g.home_score}</div>`
    : `</div><div class="vs"><span style="font-weight:700;font-size:13px">@</span></div><div class="tb"><div class="tabbr" style="color:var(--accent)">${g.home_abbr}</div><div class="tname">${g.home_team}</div>`;
  return `<div class="gcard ${sc}">
    <div class="ch"><span class="gstatus ${sc}">${g.status}</span><span class="cbadge ${cf}">Confidence: ${p.confidence}</span></div>
    <div class="sb"><div class="tb"><div class="tabbr" style="color:var(--accent2)">${g.away_abbr}</div><div class="tname">${g.away_team}</div>${scores}</div></div>
    <div class="model-row">
      <div class="model-badge elo">Elo: ${p.elo_home_prob}%</div>${mlBadge}
      <div class="model-badge rule">Rule: ${p.rule_home_prob}%</div>
      <div class="model-badge rule" style="color:var(--gold);border-color:rgba(255,215,0,.3);background:rgba(255,215,0,.06)">Final: ${p.home_prob}%</div>
    </div>
    <div class="prob-wrap">
      <div class="prob-lbl">Win Probability</div>
      <div class="prob-row">
        <div class="prob-pct a">${p.away_prob}%</div>
        <div class="pbar"><div class="pfill" style="width:${p.home_prob}%"></div></div>
        <div class="prob-pct h">${p.home_prob}%</div>
      </div>
      <div class="prob-teams"><span>${g.away_abbr}</span><span>${g.home_abbr} (Home)</span></div>
    </div>
    <div class="wbanner"><span>🏆</span><div><div class="wlbl">Predicted Winner</div><div class="wname">${p.predicted_winner}</div></div></div>
    ${factorRows ? `<div class="ftable"><div class="fhdr"><div>Factor</div><div style="text-align:center">${g.away_abbr}</div><div style="text-align:center">${g.home_abbr}</div></div>${factorRows}</div>` : ''}
  </div>`;
}

let eloLoaded = false;
async function loadElo() {
  if (eloLoaded) return;
  const res = await fetch('/api/elo');
  const data = await res.json();
  if (!data.ratings || !data.ratings.length) {
    document.getElementById('elo-wrap').innerHTML = '<div class="warnbox">No Elo data yet — builds up as games complete.</div>';
    return;
  }
  const maxR = Math.max(...data.ratings.map(r => r.rating));
  const minR = Math.min(...data.ratings.map(r => r.rating));
  const rows = data.ratings.map(r => {
    const sign = r.diff >= 0 ? '+' : '';
    return `<div class="erow">
      <div class="erank ${r.rank<=3?'top3':''}">${r.rank<=3?['🥇','🥈','🥉'][r.rank-1]:r.rank}</div>
      <div class="ename">${r.team}<small>${r.abbr}</small></div>
      <div class="erate">${r.rating}</div>
      <div class="ediff ${r.diff>=0?'pos':'neg'}">${sign}${r.diff}</div>
    </div>`;
  }).join('');
  document.getElementById('elo-wrap').innerHTML = `<div class="etable">
    <div class="ehdr"><div>#</div><div>Team</div><div>Rating</div><div>vs 1500</div></div>${rows}</div>`;
  eloLoaded = true;
}

let btLoaded = false;
async function loadBacktest() {
  if (btLoaded) return;
  const res = await fetch('/api/backtest');
  const d = await res.json();
  if (d.error) {
    document.getElementById('backtest-wrap').innerHTML = `<div class="warnbox">⏳ ${d.error}</div>`;
    return;
  }
  const roiColor = d.roi_pct >= 0 ? 'grn' : 'red';
  const pnlColor = d.pnl >= 0 ? 'grn' : 'red';
  const calRows = (d.calibration||[]).map(c => {
    const gc = Math.abs(c.gap) < 5 ? 'pos' : (Math.abs(c.gap) < 12 ? '' : 'neg');
    return `<div class="cal-row"><div>${c.bucket}</div><div>${c.avg_predicted}%</div><div>${c.actual_win_rate}%</div><div>${c.n}</div><div class="gap-val ${gc}">${c.gap>0?'+':''}${c.gap}%</div></div>`;
  }).join('');
  const gameRows = (d.recent_games||[]).slice().reverse().map(r => {
    const correct = r.correct;
    return `<div class="gt-row">
      <div style="color:var(--muted);font-size:11px">${r.date}</div>
      <div style="font-size:11px">${r.away_team}</div>
      <div style="font-size:11px">${r.home_team}</div>
      <div style="text-align:center;font-family:monospace">${(r.predicted_prob*100).toFixed(0)}%</div>
      <div style="text-align:center;font-size:11px">${r.actual_winner}</div>
      <div class="${correct?'correct':'wrong'}" style="text-align:center">${correct?'✓':'✗'}</div>
    </div>`;
  }).join('');
  document.getElementById('backtest-wrap').innerHTML = `
    <div class="bstats">
      <div class="bstat"><div class="bstat-val gold">${d.accuracy}%</div><div class="bstat-lbl">Accuracy</div><div class="bstat-sub">Baseline: 50%</div></div>
      <div class="bstat"><div class="bstat-val">${d.brier_score}</div><div class="bstat-lbl">Brier Score</div><div class="bstat-sub">Random = 0.25</div></div>
      <div class="bstat"><div class="bstat-val">${d.log_loss}</div><div class="bstat-lbl">Log Loss</div><div class="bstat-sub">Lower is better</div></div>
      <div class="bstat"><div class="bstat-val ${roiColor}">${d.roi_pct}%</div><div class="bstat-lbl">Paper ROI</div><div class="bstat-sub">${d.bets_placed} bets</div></div>
      <div class="bstat"><div class="bstat-val ${pnlColor}">£${d.pnl}</div><div class="bstat-lbl">P&L (£100/bet)</div><div class="bstat-sub">From £10,000</div></div>
      <div class="bstat"><div class="bstat-val">${d.total_games}</div><div class="bstat-lbl">Games Total</div><div class="bstat-sub">${d.bets_won} bets won</div></div>
    </div>
    <div class="section-title">Calibration</div>
    <div class="infobox">When we say 70%, does the team win 70% of the time? 0% gap = perfectly calibrated.</div>
    <div class="cal-table">
      <div class="cal-hdr"><div>Bucket</div><div>Avg Predicted</div><div>Actual Win%</div><div>n</div><div>Gap</div></div>
      ${calRows || '<div style="padding:14px;color:var(--muted);text-align:center">Not enough data per bucket yet</div>'}
    </div>
    <div class="section-title">Recent Predictions</div>
    <div class="gt-table">
      <div class="gt-hdr"><div>Date</div><div>Away</div><div>Home</div><div>Home%</div><div>Winner</div><div>Result</div></div>
      ${gameRows || '<div style="padding:14px;color:var(--muted);text-align:center">No completed games yet</div>'}
    </div>`;
  btLoaded = true;
}

let mlLoaded = false;
async function loadModel() {
  if (mlLoaded) return;
  const res = await fetch('/api/model');
  const d = await res.json();
  if (!d.trained) {
    document.getElementById('model-wrap').innerHTML = `
      <div class="warnbox">⏳ ML model not yet trained. Need ${d.games_needed} more completed games (have ${d.games_collected}).<br>
      Retrains automatically every 10 games after that.</div>`;
    return;
  }
  const mets = d.metrics;
  const fi = d.feature_importance;
  const maxFI = Math.max(...Object.values(fi));
  const fiRows = Object.entries(fi).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`
    <div class="fi-bar-wrap">
      <div class="fi-name">${k.replace(/_/g,' ')}</div>
      <div class="fi-bar"><div class="fi-fill" style="width:${(v/maxFI)*100}%"></div></div>
      <div class="fi-val">${(v*100).toFixed(1)}%</div>
    </div>`).join('');
  document.getElementById('model-wrap').innerHTML = `
    <div class="mcard">
      <h3>📊 Model Performance (Test Set)</h3>
      <div class="metric-grid">
        ${['lr','rf','gb','ensemble'].map(m => {
          const lblMap = {lr:'Logistic Reg',rf:'Random Forest',gb:'Grad Boost',ensemble:'Ensemble'};
          const colMap = {lr:'var(--muted)',rf:'var(--accent2)',gb:'var(--purple)',ensemble:'var(--gold)'};
          return `<div class="metric-box">
            <div class="metric-val" style="color:${colMap[m]}">${((mets[m]?.accuracy||0)*100).toFixed(1)}%</div>
            <div class="metric-lbl">${lblMap[m]}</div>
            <div style="font-size:10px;color:var(--muted);margin-top:2px">Brier: ${mets[m]?.brier||'—'}</div>
          </div>`;
        }).join('')}
      </div>
      <div style="margin-top:12px;font-size:12px;color:var(--muted)">Trained on <strong style="color:var(--text)">${d.n_games}</strong> games.</div>
    </div>
    <div class="mcard">
      <h3>🔑 Feature Importance (Random Forest)</h3>
      <p style="color:var(--muted);font-size:12px;margin-bottom:12px">Which inputs matter most?</p>
      ${fiRows}
    </div>`;
  mlLoaded = true;
}

loadGames();
setInterval(loadGames, 90000);
</script>
</body>
</html>"""

# ── DATABASE ─────────────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DB_URL, sslmode='require')

def init_db():
    if not USE_DB:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS game_history (
                        game_id TEXT PRIMARY KEY,
                        home_team TEXT, away_team TEXT,
                        home_elo REAL, away_elo REAL,
                        home_form REAL, away_form REAL,
                        home_avg_diff REAL, away_avg_diff REAL,
                        home_won BOOLEAN, home_score INT, away_score INT,
                        predicted_home_prob REAL,
                        game_date TEXT,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    CREATE TABLE IF NOT EXISTS elo_ratings (
                        team TEXT PRIMARY KEY,
                        rating REAL,
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
            conn.commit()
    except Exception as e:
        print(f"DB init error: {e}")

def db_load_history():
    if not USE_DB:
        return _mem_history
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM game_history ORDER BY game_date ASC")
                rows = cur.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        print(f"DB load history error: {e}")
        return []

def db_save_game(record):
    if not USE_DB:
        _mem_history.append(record)
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO game_history
                    (game_id, home_team, away_team, home_elo, away_elo,
                     home_form, away_form, home_avg_diff, away_avg_diff,
                     home_won, home_score, away_score, predicted_home_prob, game_date)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (game_id) DO NOTHING
                """, (
                    record['game_id'], record['home_team'], record['away_team'],
                    record['home_elo'], record['away_elo'],
                    record['home_form'], record['away_form'],
                    record['home_avg_diff'], record['away_avg_diff'],
                    record['home_won'], record['home_score'], record['away_score'],
                    record['predicted_home_prob'], record['date']
                ))
            conn.commit()
    except Exception as e:
        print(f"DB save game error: {e}")

def db_load_elo():
    if not USE_DB:
        return dict(_mem_elo)
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT team, rating FROM elo_ratings")
                return {r['team']: r['rating'] for r in cur.fetchall()}
    except Exception as e:
        print(f"DB load elo error: {e}")
        return {}

def db_save_elo(ratings):
    if not USE_DB:
        _mem_elo.update(ratings)
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                for team, rating in ratings.items():
                    cur.execute("""
                        INSERT INTO elo_ratings (team, rating, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (team) DO UPDATE SET rating=EXCLUDED.rating, updated_at=NOW()
                    """, (team, rating))
            conn.commit()
    except Exception as e:
        print(f"DB save elo error: {e}")

# In-memory fallback when no DB
_mem_history = []
_mem_elo = {}

# ── CACHE ─────────────────────────────────────────────────────────────────────
_cache = {}
_cache_lock = threading.Lock()

def cached_get(url, params=None, ttl=120):
    key = url + str(sorted((params or {}).items()))
    now = time.time()
    with _cache_lock:
        if key in _cache and now - _cache[key]["ts"] < ttl:
            return _cache[key]["data"]
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = API_KEY
    try:
        r = requests.get(url, params=params, headers=h, timeout=10)
        if r.status_code == 200:
            data = r.json()
            with _cache_lock:
                _cache[key] = {"data": data, "ts": now}
            return data
    except Exception as e:
        print(f"API error: {e}")
    return None

# ── ELO HELPERS ───────────────────────────────────────────────────────────────
def elo_expected(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))

def elo_update(ratings, home, away, home_won, margin=None):
    hr = ratings.get(home, DEFAULT_RATING)
    ar = ratings.get(away, DEFAULT_RATING)
    he = elo_expected(hr + HOME_ADV_ELO, ar)
    mov = max(1.0, (abs(margin) ** 0.8) / 7.5) if margin else 1.0
    k = K_FACTOR * mov
    ratings[home] = hr + k * ((1 if home_won else 0) - he)
    ratings[away] = ar + k * ((0 if home_won else 1) - (1 - he))
    return ratings

def elo_win_prob(ratings, home, away):
    hr = ratings.get(home, DEFAULT_RATING)
    ar = ratings.get(away, DEFAULT_RATING)
    p = elo_expected(hr + HOME_ADV_ELO, ar)
    return round(p, 4), round(1 - p, 4)

# ── ML HELPERS ────────────────────────────────────────────────────────────────
_ml_model = None

def build_features(r):
    return [
        r.get('home_elo', 1500) - r.get('away_elo', 1500),
        r.get('home_form', 0.5),
        r.get('away_form', 0.5),
        r.get('home_form', 0.5) - r.get('away_form', 0.5),
        r.get('home_avg_diff', 0),
        r.get('away_avg_diff', 0),
        r.get('home_avg_diff', 0) - r.get('away_avg_diff', 0),
        1.0,
    ]

def train_ml(history):
    if not HAS_ML or len(history) < 50:
        return None
    try:
        X = np.array([build_features(r) for r in history])
        y = np.array([1 if r['home_won'] else 0 for r in history])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        lr.fit(Xtr_s, ytr); rf.fit(Xtr, ytr); gb.fit(Xtr, ytr)
        lrp = lr.predict_proba(Xte_s)[:,1]; rfp = rf.predict_proba(Xte)[:,1]; gbp = gb.predict_proba(Xte)[:,1]
        lb = brier_score_loss(yte, lrp); rb = brier_score_loss(yte, rfp); gbb = brier_score_loss(yte, gbp)
        wl, wr, wg = 1/(lb+1e-9), 1/(rb+1e-9), 1/(gbb+1e-9); tw = wl+wr+wg
        ep = (wl*lrp + wr*rfp + wg*gbp) / tw
        eb = brier_score_loss(yte, ep)
        fi = dict(zip(['elo_diff','home_form','away_form','form_diff','home_diff','away_diff','diff_diff','home_adv'], rf.feature_importances_))
        return {
            'models': {'lr': lr, 'rf': rf, 'gb': gb}, 'scaler': sc,
            'weights': {'lr': wl/tw, 'rf': wr/tw, 'gb': wg/tw},
            'metrics': {
                'lr': {'accuracy': round(accuracy_score(yte, lr.predict(Xte_s)), 4), 'brier': round(lb, 4)},
                'rf': {'accuracy': round(accuracy_score(yte, rf.predict(Xte)), 4), 'brier': round(rb, 4)},
                'gb': {'accuracy': round(accuracy_score(yte, gb.predict(Xte)), 4), 'brier': round(gbb, 4)},
                'ensemble': {'accuracy': round(accuracy_score(yte, (ep>0.5).astype(int)), 4), 'brier': round(eb, 4)},
            },
            'feature_importance': {k: round(v, 4) for k,v in fi.items()},
            'n_games': len(history),
            'weights_display': {'lr': round(wl/tw, 3), 'rf': round(wr/tw, 3), 'gb': round(wg/tw, 3)},
        }
    except Exception as e:
        print(f"ML train error: {e}")
        return None

def ml_predict(model_data, record):
    if not model_data:
        return None, None
    try:
        x = np.array(build_features(record)).reshape(1, -1)
        xs = model_data['scaler'].transform(x)
        m = model_data['models']; w = model_data['weights']
        p = w['lr']*m['lr'].predict_proba(xs)[0][1] + w['rf']*m['rf'].predict_proba(x)[0][1] + w['gb']*m['gb'].predict_proba(x)[0][1]
        return round(p, 4), round(1-p, 4)
    except:
        return None, None

# ── DATA FETCHING ─────────────────────────────────────────────────────────────
def get_todays_games():
    data = cached_get(f"{BALLDONTLIE_BASE}/games", {"dates[]": date.today().isoformat(), "per_page": 30}, ttl=60)
    return data["data"] if data and "data" in data else []

def get_recent_games(team_id, n=10):
    data = cached_get(f"{BALLDONTLIE_BASE}/games", {"team_ids[]": team_id, "per_page": n, "seasons[]": 2024}, ttl=300)
    return data["data"] if data and "data" in data else []

def get_all_teams():
    data = cached_get(f"{BALLDONTLIE_BASE}/teams", {"per_page": 30}, ttl=86400)
    return {t["id"]: t for t in data["data"]} if data and "data" in data else {}

def win_pct(games, tid):
    wins, total = 0, 0
    for g in games:
        if g["status"] != "Final": continue
        home = g["home_team"]["id"] == tid
        won = (g["home_team_score"] > g["visitor_team_score"]) if home else (g["visitor_team_score"] > g["home_team_score"])
        wins += int(won); total += 1
    return wins/total if total > 0 else 0.5

def avg_point_diff(games, tid):
    diffs = []
    for g in games:
        if g["status"] != "Final": continue
        home = g["home_team"]["id"] == tid
        d = (g["home_team_score"]-g["visitor_team_score"]) if home else (g["visitor_team_score"]-g["home_team_score"])
        diffs.append(d)
    return sum(diffs)/len(diffs) if diffs else 0

def logistic(x):
    return 1 / (1 + math.exp(-x))

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
elo_ratings = {}
ml_model_data = None
_processed_ids = set()

def init_state():
    global elo_ratings, ml_model_data, _processed_ids
    elo_ratings = db_load_elo()
    history = db_load_history()
    _processed_ids = {r['game_id'] for r in history}
    if len(history) >= 50:
        print(f"Training ML on {len(history)} games...")
        ml_model_data = train_ml(history)
        if ml_model_data:
            print(f"ML ready. Ensemble accuracy: {ml_model_data['metrics']['ensemble']['accuracy']*100:.1f}%")

# ── ELO UPDATER ───────────────────────────────────────────────────────────────
def process_completed_games(games):
    global elo_ratings, ml_model_data, _processed_ids
    changed = False
    new_records = []
    for g in games:
        if g["status"] != "Final": continue
        gid = str(g["id"])
        if gid in _processed_ids: continue
        home = g["home_team"]["full_name"]; away = g["visitor_team"]["full_name"]
        hs = g.get("home_team_score") or 0; as_ = g.get("visitor_team_score") or 0
        home_won = hs > as_; margin = abs(hs - as_)
        hr = get_recent_games(g["home_team"]["id"], 10); ar = get_recent_games(g["visitor_team"]["id"], 10)
        hform = win_pct(hr, g["home_team"]["id"]); aform = win_pct(ar, g["visitor_team"]["id"])
        hdiff = avg_point_diff(hr, g["home_team"]["id"]); adiff = avg_point_diff(ar, g["visitor_team"]["id"])
        h_elo = elo_ratings.get(home, DEFAULT_RATING); a_elo = elo_ratings.get(away, DEFAULT_RATING)
        record = {"game_id": gid, "home_team": home, "away_team": away,
                  "home_elo": h_elo, "away_elo": a_elo, "home_form": hform, "away_form": aform,
                  "home_avg_diff": hdiff, "away_avg_diff": adiff, "home_won": home_won,
                  "home_score": hs, "away_score": as_, "date": g.get("date","")[:10]}
        hep, _ = elo_win_prob(elo_ratings, home, away)
        mlp, _ = ml_predict(ml_model_data, record)
        record["predicted_home_prob"] = (mlp*0.6 + hep*0.4) if mlp else hep
        elo_ratings = elo_update(elo_ratings, home, away, home_won, margin)
        db_save_game(record)
        _processed_ids.add(gid)
        new_records.append(record)
        changed = True
    if changed:
        db_save_elo(elo_ratings)
        history = db_load_history()
        if len(history) >= 50 and len(history) % 10 == 0:
            ml_model_data = train_ml(history)
            if ml_model_data:
                print(f"ML retrained on {len(history)} games. Accuracy: {ml_model_data['metrics']['ensemble']['accuracy']*100:.1f}%")

# ── PREDICTION ────────────────────────────────────────────────────────────────
def predict_game(game):
    hname = game["home_team"]["full_name"]; aname = game["visitor_team"]["full_name"]
    hid = game["home_team"]["id"]; aid = game["visitor_team"]["id"]
    hr = get_recent_games(hid, 10); ar = get_recent_games(aid, 10)
    hform = win_pct(hr, hid); aform = win_pct(ar, aid)
    hdiff = avg_point_diff(hr, hid); adiff = avg_point_diff(ar, aid)
    h_elo = elo_ratings.get(hname, DEFAULT_RATING); a_elo = elo_ratings.get(aname, DEFAULT_RATING)
    elo_hp, _ = elo_win_prob(elo_ratings, hname, aname)
    rec = {"home_elo": h_elo, "away_elo": a_elo, "home_form": hform, "away_form": aform, "home_avg_diff": hdiff, "away_avg_diff": adiff}
    ml_hp, _ = ml_predict(ml_model_data, rec)
    live_factor = 0; period = game.get("period", 0)
    if period and period > 0:
        hs = game.get("home_team_score") or 0; as_ = game.get("visitor_team_score") or 0
        live_factor = (hs - as_) * min(period/4.0, 1.0) * 0.6
    score = (hform-aform)*30 + (hdiff-adiff)*0.4 + 1.5 + live_factor
    rule_hp = logistic(score/15)
    final_h = (ml_hp*0.50 + elo_hp*0.35 + rule_hp*0.15) if ml_hp else (elo_hp*0.60 + rule_hp*0.40)
    final_a = 1 - final_h
    margin_val = abs(final_h - final_a)
    conf = "High" if margin_val > 0.25 else ("Medium" if margin_val > 0.12 else "Low")
    factors = [
        {"name": "Elo Rating", "home": str(int(h_elo)), "away": str(int(a_elo)), "edge": "home" if h_elo > a_elo else "away"},
        {"name": "Recent Form (L10)", "home": f"{hform*100:.0f}%", "away": f"{aform*100:.0f}%", "edge": "home" if hform > aform else "away"},
        {"name": "Avg Point Diff", "home": f"{hdiff:+.1f}", "away": f"{adiff:+.1f}", "edge": "home" if hdiff > adiff else "away"},
        {"name": "Home Court", "home": "+3 pts", "away": "—", "edge": "home"},
    ]
    if period and period > 0:
        hs = game.get("home_team_score") or 0; as_ = game.get("visitor_team_score") or 0
        factors.append({"name": f"Live (Q{period})", "home": str(hs), "away": str(as_), "edge": "home" if hs>as_ else ("away" if as_>hs else "even")})
    return {
        "home_prob": round(final_h*100, 1), "away_prob": round(final_a*100, 1),
        "elo_home_prob": round(elo_hp*100, 1),
        "ml_home_prob": round(ml_hp*100, 1) if ml_hp else None,
        "rule_home_prob": round(rule_hp*100, 1),
        "confidence": conf, "predicted_winner": hname if final_h > 0.5 else aname,
        "factors": factors, "home_elo": int(h_elo), "away_elo": int(a_elo),
        "market": None, "kelly_home": None, "kelly_away": None, "ml_active": ml_hp is not None,
    }

# ── BACKTEST ──────────────────────────────────────────────────────────────────
def compute_backtest(history):
    if len(history) < 5:
        return {"error": f"Need at least 5 completed games. Have {len(history)} so far — check back after more game days."}
    correct = 0; total = 0; brier_scores = []; log_losses = []
    bankroll = 10000; pnl = 0; bets = 0; bets_won = 0
    recent_games = []
    for r in history:
        prob = r.get("predicted_home_prob", 0.5)
        if prob is None: prob = 0.5
        actual = 1 if r["home_won"] else 0
        pred_win = prob > 0.5
        correct += int(pred_win == r["home_won"]); total += 1
        brier_scores.append((prob - actual)**2)
        p = max(min(prob, 0.999), 0.001)
        log_losses.append(-(actual*math.log(p) + (1-actual)*math.log(1-p)))
        conf = abs(prob - 0.5)
        if conf > 0.05:
            bets += 1
            if pred_win == r["home_won"]:
                pnl += 90; bets_won += 1; bankroll += 90
            else:
                pnl -= 100; bankroll -= 100
        recent_games.append({
            "date": r.get("date",""),
            "home_team": r.get("home_team",""),
            "away_team": r.get("away_team",""),
            "predicted_prob": round(prob, 3),
            "actual_winner": r.get("home_team","") if r["home_won"] else r.get("away_team",""),
            "correct": pred_win == r["home_won"],
        })
    # Calibration
    buckets = {f"{i*10}-{(i+1)*10}%": {"p":[], "a":[]} for i in range(5, 10)}
    for r in history:
        prob = r.get("predicted_home_prob", 0.5) or 0.5
        p_winner = max(prob, 1-prob)
        actual_correct = (prob > 0.5) == r["home_won"]
        for i in range(5, 10):
            if i*0.1 <= p_winner < (i+1)*0.1:
                key = f"{i*10}-{(i+1)*10}%"
                buckets[key]["p"].append(p_winner); buckets[key]["a"].append(int(actual_correct))
                break
    calibration = []
    for key, vals in buckets.items():
        if vals["p"]:
            calibration.append({
                "bucket": key,
                "avg_predicted": round(sum(vals["p"])/len(vals["p"])*100, 1),
                "actual_win_rate": round(sum(vals["a"])/len(vals["a"])*100, 1),
                "n": len(vals["p"]),
                "gap": round((sum(vals["a"])/len(vals["a"]) - sum(vals["p"])/len(vals["p"]))*100, 1)
            })
    acc = correct/total if total > 0 else 0
    roi = (pnl/(bets*100))*100 if bets > 0 else 0
    return {
        "total_games": total, "accuracy": round(acc*100, 2),
        "brier_score": round(float(np.mean(brier_scores)), 4),
        "log_loss": round(float(np.mean(log_losses)), 4),
        "bets_placed": bets, "bets_won": bets_won,
        "pnl": round(pnl, 2), "roi_pct": round(roi, 2),
        "starting_bankroll": 10000, "final_bankroll": round(bankroll, 2),
        "calibration": calibration,
        "recent_games": recent_games,
        "baseline_accuracy": 50.0,
    }

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return HTML_PAGE, 200, {'Content-Type': 'text/html'}

@app.route("/api/games")
def api_games():
    games = get_todays_games()
    try:
        process_completed_games(games)
    except Exception as e:
        print(f"Process error: {e}")
    results = []
    for g in games:
        try:
            pred = predict_game(g)
        except Exception as e:
            print(f"Predict error: {e}")
            pred = {"home_prob":50,"away_prob":50,"confidence":"N/A","predicted_winner":"TBD",
                    "factors":[],"home_elo":1500,"away_elo":1500,"market":None,"kelly_home":None,
                    "kelly_away":None,"ml_active":False,"elo_home_prob":50,"ml_home_prob":None,"rule_home_prob":50}
        status = g.get("status","")
        ds = "Final" if status=="Final" else (f"Q{g['period']} LIVE" if g.get("period",0)>0 else status)
        results.append({
            "id": g["id"],
            "home_team": g["home_team"]["full_name"], "home_abbr": g["home_team"]["abbreviation"],
            "away_team": g["visitor_team"]["full_name"], "away_abbr": g["visitor_team"]["abbreviation"],
            "home_score": g.get("home_team_score") or 0, "away_score": g.get("visitor_team_score") or 0,
            "status": ds, "period": g.get("period",0), "prediction": pred,
        })
    results.sort(key=lambda g: (0 if "LIVE" in g["status"] else (2 if g["status"]=="Final" else 1)))
    history = db_load_history()
    return jsonify({"games": results, "fetched_at": datetime.now().strftime("%H:%M:%S"),
                    "ml_active": ml_model_data is not None, "games_in_history": len(history)})

@app.route("/api/elo")
def api_elo():
    all_teams = get_all_teams()
    name_abbr = {t["full_name"]: t["abbreviation"] for t in all_teams.values()}
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: -x[1])
    return jsonify({"ratings": [
        {"rank": i+1, "team": name, "rating": round(r, 1),
         "abbr": name_abbr.get(name, name[:3].upper()),
         "diff": round(r - DEFAULT_RATING, 1)}
        for i, (name, r) in enumerate(sorted_ratings)
    ]})

@app.route("/api/backtest")
def api_backtest():
    history = db_load_history()
    return jsonify(compute_backtest(history))

@app.route("/api/model")
def api_model():
    history = db_load_history()
    if not ml_model_data:
        return jsonify({"trained": False, "games_collected": len(history), "games_needed": max(0, 50-len(history))})
    return jsonify({
        "trained": True,
        "metrics": ml_model_data.get("metrics", {}),
        "feature_importance": ml_model_data.get("feature_importance", {}),
        "n_games": ml_model_data.get("n_games", 0),
        "weights": ml_model_data.get("weights_display", {}),
    })

# ── STARTUP ───────────────────────────────────────────────────────────────────
def background_updater():
    """Runs every 2 hours to catch completed games even with no site visitors."""
    import time
    while True:
        try:
            games = get_todays_games()
            if games:
                process_completed_games(games)
                print(f"Background update: checked {len(games)} games")
        except Exception as e:
            print(f"Background updater error: {e}")
        time.sleep(7200)  # every 2 hours

if __name__ == "__main__":
    print("\n🏀 NBA Predictor v3 — PostgreSQL Edition")
    init_db()
    init_state()
    # Start background updater thread
    import threading
    updater = threading.Thread(target=background_updater, daemon=True)
    updater.start()
    print("Background updater started (checks every 2 hours)")
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
