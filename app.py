"""
NBA Predictor v2 — Quant-Grade
Run: python app.py  →  open http://localhost:5000
"""

from flask import Flask, jsonify, request
import requests
import json
import time
import threading
import math
import os
from datetime import datetime, date, timedelta

from elo import (load_ratings, save_ratings, update_elo,
                 get_elo_win_prob, get_all_ratings, DEFAULT_RATING)
from ml_model import (load_history, save_history, add_game_to_history,
                      train_model, load_model, predict_with_ml,
                      build_feature_vector)
from backtest import (kelly_criterion, run_backtest, load_backtest,
                      compute_calibration)


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

/* HEADER */
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
.header-meta strong{color:var(--accent2)}
.live-dot{display:inline-block;width:8px;height:8px;background:var(--live);
  border-radius:50%;animation:pulse 1.5s infinite;margin-right:5px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* TABS */
nav{display:flex;gap:2px;padding:0 28px;background:var(--bg);border-bottom:1px solid var(--border)}
.tab{padding:12px 18px;background:none;border:none;color:var(--muted);font-size:13px;
  font-weight:600;cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover{color:var(--text)}

/* LAYOUT */
.wrap{max-width:1380px;margin:0 auto;padding:22px 28px}
.page{display:none}.page.active{display:block}

/* PILLS */
.pills{display:flex;gap:10px;margin-bottom:18px;flex-wrap:wrap}
.pill{background:var(--card);border:1px solid var(--border);border-radius:20px;
  padding:6px 14px;font-size:12px;display:flex;align-items:center;gap:7px}
.pill .lbl{color:var(--muted)}.pill .val{font-weight:700}
.pill .val.live{color:var(--live)}.pill .val.grn{color:var(--green)}
.pill .val.ml{color:var(--purple)}

/* GRID */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}

/* GAME CARD */
.gcard{background:var(--card);border:1px solid var(--border);border-radius:14px;
  overflow:hidden;transition:transform .2s,border-color .2s}
.gcard:hover{transform:translateY(-2px);border-color:var(--accent)}
.gcard.live{border-color:var(--live);box-shadow:0 0 24px rgba(255,51,51,.12)}
.gcard.final{opacity:.72}

/* Card header */
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

/* Scoreboard */
.sb{padding:18px 14px;display:flex;align-items:center;gap:10px}
.tb{flex:1;text-align:center}
.tabbr{font-size:26px;font-weight:900;letter-spacing:-1px;line-height:1}
.tname{font-size:10px;color:var(--muted);margin-top:3px}
.tscore{font-size:40px;font-weight:900;margin-top:6px;letter-spacing:-2px}
.tscore.win{color:var(--accent)}
.vs{display:flex;flex-direction:column;align-items:center;gap:3px;color:var(--muted);
  font-size:11px;min-width:36px}

/* Prob bar */
.prob-wrap{padding:0 14px 14px}
.prob-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:7px}
.prob-row{display:flex;align-items:center;gap:7px}
.prob-pct{font-size:15px;font-weight:800;min-width:42px;text-align:center}
.prob-pct.h{color:var(--accent2)}.prob-pct.a{color:var(--accent)}
.pbar{flex:1;height:10px;background:var(--border);border-radius:10px;overflow:hidden}
.pfill{height:100%;border-radius:10px;transition:width .8s;
  background:linear-gradient(90deg,var(--accent2),var(--accent))}
.prob-teams{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-top:3px}

/* Model breakdown */
.model-row{padding:0 14px 12px;display:flex;gap:8px;flex-wrap:wrap}
.model-badge{font-size:11px;padding:3px 9px;border-radius:8px;font-weight:600;border:1px solid}
.model-badge.elo{color:var(--accent2);border-color:rgba(79,195,247,.3);background:rgba(79,195,247,.08)}
.model-badge.ml{color:var(--purple);border-color:rgba(156,111,255,.3);background:rgba(156,111,255,.08)}
.model-badge.rule{color:var(--muted);border-color:var(--border);background:var(--card2)}

/* Winner banner */
.wbanner{margin:0 14px 12px;background:linear-gradient(135deg,rgba(255,107,53,.08),rgba(79,195,247,.08));
  border:1px solid var(--border);border-radius:9px;padding:9px 12px;display:flex;align-items:center;gap:9px;font-size:12px}
.wlbl{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.wname{font-weight:700;font-size:13px;color:var(--gold)}

/* Kelly */
.kelly-wrap{margin:0 14px 12px;display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kelly-box{background:var(--card2);border:1px solid var(--border);border-radius:9px;padding:9px 11px}
.kelly-title{font-size:10px;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.kelly-edge{font-size:16px;font-weight:800;color:var(--green)}
.kelly-edge.neg{color:var(--red)}
.kelly-sub{font-size:11px;color:var(--muted);margin-top:2px}

/* Factors */
.ftable{margin:0 14px 14px;border:1px solid var(--border);border-radius:9px;overflow:hidden}
.fhdr{background:var(--card2);display:grid;grid-template-columns:1fr 64px 64px;
  padding:7px 11px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:600}
.frow{display:grid;grid-template-columns:1fr 64px 64px;padding:7px 11px;font-size:12px;
  border-top:1px solid var(--border);align-items:center}
.frow:nth-child(even){background:rgba(255,255,255,.018)}
.fname{color:var(--muted)}.fval{text-align:center;font-weight:600;font-family:monospace}
.fval.eh{color:var(--accent2)}.fval.ea{color:var(--accent)}

/* EMPTY */
.empty{text-align:center;padding:70px 20px;color:var(--muted)}
.empty .ico{font-size:56px;margin-bottom:14px}

/* LOADER */
.loader{display:flex;align-items:center;justify-content:center;padding:70px;flex-direction:column;gap:14px;color:var(--muted)}
.spin{width:38px;height:38px;border:3px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

/* ELO TABLE */
.etable{background:var(--card);border:1px solid var(--border);border-radius:14px;overflow:hidden}
.ehdr{display:grid;grid-template-columns:44px 1fr 80px 90px 90px;padding:11px 16px;
  background:var(--card2);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:700;border-bottom:1px solid var(--border)}
.erow{display:grid;grid-template-columns:44px 1fr 80px 90px 90px;padding:11px 16px;
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
.ebar-wrap{display:flex;align-items:center;gap:6px}
.ebar{flex:1;height:4px;background:var(--border);border-radius:4px;overflow:hidden}
.efill{height:100%;background:var(--accent);border-radius:4px}

/* BACKTEST */
.bstats{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;margin-bottom:22px}
.bstat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.bstat-val{font-size:28px;font-weight:900;line-height:1}
.bstat-lbl{font-size:11px;color:var(--muted);margin-top:4px}
.bstat-sub{font-size:11px;color:var(--muted);margin-top:2px}
.bstat-val.grn{color:var(--green)}.bstat-val.red{color:var(--red)}.bstat-val.gold{color:var(--gold)}

/* Calibration */
.cal-table{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-bottom:22px}
.cal-hdr{display:grid;grid-template-columns:90px 1fr 1fr 80px 80px;padding:10px 14px;
  background:var(--card2);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px;font-weight:700;border-bottom:1px solid var(--border)}
.cal-row{display:grid;grid-template-columns:90px 1fr 1fr 80px 80px;padding:10px 14px;
  font-size:13px;border-bottom:1px solid var(--border);align-items:center}
.cal-row:last-child{border-bottom:none}
.gap-val{font-weight:700}.gap-val.pos{color:var(--green)}.gap-val.neg{color:var(--red)}

/* MODEL PAGE */
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

/* INFO BOX */
.infobox{background:rgba(79,195,247,.07);border:1px solid rgba(79,195,247,.2);
  border-radius:10px;padding:11px 15px;font-size:13px;color:var(--accent2);margin-bottom:18px;line-height:1.5}
.warnbox{background:rgba(255,193,7,.07);border:1px solid rgba(255,193,7,.2);
  border-radius:10px;padding:11px 15px;font-size:13px;color:#ffc107;margin-bottom:18px;line-height:1.5}
.section-title{font-size:16px;font-weight:700;margin-bottom:14px;color:var(--text)}

@media(max-width:600px){
  .wrap{padding:14px}.tabs{padding:0 14px}.grid{grid-template-columns:1fr}
  header{padding:12px 14px}.ehdr,.erow{grid-template-columns:36px 1fr 70px 70px}
  .ecol5{display:none}
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🏀</div>
    <div>
      <h1>NBA <span>Quant</span> Predictor</h1>
      <div style="font-size:10px;color:var(--muted)">Elo · ML Ensemble · Kelly Criterion</div>
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
  <button class="tab" onclick="showTab('elo',this)">📈 Elo Rankings</button>
  <button class="tab" onclick="showTab('backtest',this)">🔬 Backtest</button>
  <button class="tab" onclick="showTab('model',this)">🤖 ML Model</button>
  <button class="tab" onclick="showTab('about',this)">ℹ️ How It Works</button>
</nav>

<div class="wrap">

<!-- ── GAMES ── -->
<div id="pg-games" class="page active">
  <div class="pills" id="pills">
    <div class="pill"><span class="lbl">Games</span><span class="val" id="p-total">—</span></div>
    <div class="pill"><span class="lbl">Live</span><span class="val live" id="p-live">—</span></div>
    <div class="pill"><span class="lbl">Upcoming</span><span class="val grn" id="p-up">—</span></div>
    <div class="pill"><span class="lbl">Final</span><span class="val" id="p-fin">—</span></div>
    <div class="pill"><span class="lbl">ML Model</span><span class="val ml" id="p-ml">—</span></div>
  </div>
  <div id="games-wrap"><div class="loader"><div class="spin"></div><div>Fetching live data...</div></div></div>
</div>

<!-- ── ELO ── -->
<div id="pg-elo" class="page">
  <div class="infobox">📊 <strong>Elo ratings</strong> update after every completed game. Teams start at 1500. A win against a stronger opponent gives more points. The number in brackets is the change from 1500 (the neutral baseline).</div>
  <div id="elo-wrap"><div class="loader"><div class="spin"></div><div>Loading Elo ratings...</div></div></div>
</div>

<!-- ── BACKTEST ── -->
<div id="pg-backtest" class="page">
  <div class="infobox">🔬 <strong>Backtesting</strong> runs the model on every completed game in our history and measures whether it predicted correctly. This tells us if we're actually good at this, or just lucky.</div>
  <div id="backtest-wrap"><div class="loader"><div class="spin"></div><div>Running backtest...</div></div></div>
</div>

<!-- ── ML MODEL ── -->
<div id="pg-model" class="page">
  <div id="model-wrap"><div class="loader"><div class="spin"></div><div>Loading model data...</div></div></div>
</div>

<!-- ── ABOUT ── -->
<div id="pg-about" class="page">
  <div style="max-width:700px">

    <div class="mcard">
      <h3>🏗️ System Architecture</h3>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">
        Three independent models vote on each game, weighted by their historical calibration score (Brier score). Better-calibrated models get more weight.
      </p>
      <div style="margin-top:14px;display:flex;flex-direction:column;gap:10px">
        <div style="background:var(--card2);border-radius:8px;padding:12px;border:1px solid var(--border)">
          <div style="color:var(--accent2);font-weight:700;font-size:13px">Elo Rating System (35% weight)</div>
          <div style="color:var(--muted);font-size:12px;margin-top:4px">Self-updating rating based on results + margin of victory. Borrowed from chess, adopted by FiveThirtyEight for NBA.</div>
        </div>
        <div style="background:var(--card2);border-radius:8px;padding:12px;border:1px solid var(--border)">
          <div style="color:var(--purple);font-weight:700;font-size:13px">ML Ensemble (50% weight) — activates after 50 games</div>
          <div style="color:var(--muted);font-size:12px;margin-top:4px">Logistic Regression + Random Forest + Gradient Boosting, trained on our game history. Each model's weight = 1/BrierScore so better models dominate.</div>
        </div>
        <div style="background:var(--card2);border-radius:8px;padding:12px;border:1px solid var(--border)">
          <div style="color:var(--muted);font-weight:700;font-size:13px">Rule-Based Logistic (15% weight)</div>
          <div style="color:var(--muted);font-size:12px;margin-top:4px">Hand-tuned logistic regression using form, point differential, and home court. Provides a sensible baseline before ML data accumulates.</div>
        </div>
      </div>
    </div>

    <div class="mcard">
      <h3>📐 Kelly Criterion</h3>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">
        The Kelly formula <code style="color:var(--accent2)">f* = (bp - q) / b</code> tells you the optimal fraction of your bankroll to bet to maximise long-run growth, where <code style="color:var(--accent2)">b</code> = decimal odds − 1, <code style="color:var(--accent2)">p</code> = our probability, <code style="color:var(--accent2)">q</code> = 1 − p. We use <strong style="color:var(--text)">half-Kelly</strong> to reduce variance — full Kelly is theoretically optimal but causes massive drawdowns in practice. Only shown when market odds are available.
      </p>
    </div>

    <div class="mcard">
      <h3>📏 Brier Score & Calibration</h3>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">
        <strong style="color:var(--text)">Brier Score</strong> = mean((predicted_prob − actual_outcome)²). Perfect = 0.0, random guessing = 0.25. A well-calibrated model that says 70% should win 70% of the time — the calibration table shows exactly how ours performs in each probability bucket.
      </p>
    </div>

    <div class="mcard">
      <h3>⚠️ Limitations (important for interviews)</h3>
      <div style="font-size:13px;color:var(--muted);line-height:1.9">
        <div>• <strong style="color:var(--text)">No injury data</strong> — a star player sitting out completely invalidates our ratings</div>
        <div>• <strong style="color:var(--text)">Small sample</strong> — ML model needs 50+ games; Elo converges after ~200</div>
        <div>• <strong style="color:var(--text)">No rest/travel</strong> — back-to-back games and time zones matter a lot</div>
        <div>• <strong style="color:var(--text)">Market efficiency</strong> — if the edge is real, markets will close it quickly</div>
        <div>• <strong style="color:var(--text)">Not betting advice</strong> — this is a quantitative modelling exercise</div>
      </div>
    </div>

  </div>
</div>

</div><!-- /wrap -->

<script>
// ── TAB SYSTEM ──────────────────────────────────────────────────────────────
function showTab(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('pg-' + name).classList.add('active');
  btn.classList.add('active');
  const loaders = {
    elo: loadElo, backtest: loadBacktest, model: loadModel
  };
  if (loaders[name]) loaders[name]();
}

// ── GAMES ──────────────────────────────────────────────────────────────────
async function loadGames() {
  try {
    const res = await fetch('/api/games');
    const data = await res.json();
    renderGames(data.games);
    document.getElementById('last-update').textContent = 'Updated ' + data.fetched_at;
    document.getElementById('p-ml').textContent = data.ml_active ? '✅ Active' : `⏳ ${data.games_in_history}/50 games`;
    document.getElementById('ml-status').innerHTML = data.ml_active
      ? '<span style="color:var(--purple)">● ML Ensemble active</span>'
      : `<span style="color:var(--muted)">Collecting data: ${data.games_in_history}/50 games</span>`;
  } catch(e) {
    document.getElementById('games-wrap').innerHTML =
      '<div class="empty"><div class="ico">⚠️</div><h2>Connection error</h2><p>Check your API key and internet connection.</p></div>';
  }
}

function renderGames(games) {
  if (!games.length) {
    document.getElementById('games-wrap').innerHTML =
      '<div class="empty"><div class="ico">🌙</div><h2>No games today</h2><p>Check back on a game day.</p></div>';
    return;
  }
  const live = games.filter(g => g.status.includes('LIVE')).length;
  const fin = games.filter(g => g.status === 'Final').length;
  document.getElementById('p-total').textContent = games.length;
  document.getElementById('p-live').textContent = live;
  document.getElementById('p-up').textContent = games.length - live - fin;
  document.getElementById('p-fin').textContent = fin;
  document.getElementById('games-wrap').innerHTML =
    `<div class="grid">${games.map(buildCard).join('')}</div>`;
}

function buildCard(g) {
  const p = g.prediction;
  const isLive = g.status.includes('LIVE');
  const isFinal = g.status === 'Final';
  const sc = isLive ? 'live' : (isFinal ? 'final' : 'upcoming');
  const hwin = g.home_score > g.away_score;
  const awin = g.away_score > g.home_score;
  const cf = p.confidence.replace('/','') === 'N/A' ? 'NA' : p.confidence;

  // Model breakdown badges
  const mlBadge = p.ml_home_prob !== null
    ? `<div class="model-badge ml">ML: ${p.ml_home_prob}%</div>` : '';

  // Kelly section
  let kellyHtml = '';
  if (p.market && p.kelly_home) {
    const kh = p.kelly_home, ka = p.kelly_away;
    const showHome = kh.has_edge;
    const showAway = ka.has_edge;
    if (showHome || showAway) {
      kellyHtml = `<div class="kelly-wrap">` +
        (showHome ? `<div class="kelly-box">
          <div class="kelly-title">Kelly — ${g.home_abbr}</div>
          <div class="kelly-edge">+${kh.edge}% edge</div>
          <div class="kelly-sub">Bet £${kh.bet_size} of £1000 bankroll</div>
          <div class="kelly-sub">EV: £${kh.expected_value} per £100</div>
        </div>` : '') +
        (showAway ? `<div class="kelly-box">
          <div class="kelly-title">Kelly — ${g.away_abbr}</div>
          <div class="kelly-edge">+${ka.edge}% edge</div>
          <div class="kelly-sub">Bet £${ka.bet_size} of £1000 bankroll</div>
          <div class="kelly-sub">EV: £${ka.expected_value} per £100</div>
        </div>` : '') +
        `</div>`;
    }
  }

  // Market odds
  const mktHtml = p.market ? `
    <div style="padding:0 14px 10px;font-size:11px;color:var(--muted)">
      Market (${p.market.bookmaker}): ${g.away_abbr} ${p.market.away_odds}x · ${g.home_abbr} ${p.market.home_odds}x
      &nbsp;|&nbsp; Implied: ${(p.market.away_implied*100).toFixed(1)}% / ${(p.market.home_implied*100).toFixed(1)}%
    </div>` : '';

  const factorRows = (p.factors||[]).map(f => `
    <div class="frow">
      <div class="fname">${f.name}</div>
      <div class="fval ${f.edge==='away'?'ea':''}">${f.away}</div>
      <div class="fval ${f.edge==='home'?'eh':''}">${f.home}</div>
    </div>`).join('');

  const scores = (isFinal || isLive)
    ? `<div class="tscore ${awin?'win':''}">${g.away_score}</div>`
    + `</div><div class="vs"><span style="font-weight:700;font-size:13px">@</span>${isLive?'<span style="color:var(--live);font-size:9px;font-weight:700">LIVE</span>':''}</div>`
    + `<div class="tb"><div class="tabbr" style="color:var(--accent)">${g.home_abbr}</div><div class="tname">${g.home_team}</div><div class="tscore ${hwin?'win':''}">${g.home_score}</div>`
    : `</div><div class="vs"><span style="font-weight:700;font-size:13px">@</span></div><div class="tb"><div class="tabbr" style="color:var(--accent)">${g.home_abbr}</div><div class="tname">${g.home_team}</div>`;

  return `
  <div class="gcard ${sc}">
    <div class="ch">
      <span class="gstatus ${sc}">${g.status}</span>
      <span class="cbadge ${cf}">Confidence: ${p.confidence}</span>
    </div>
    <div class="sb">
      <div class="tb">
        <div class="tabbr" style="color:var(--accent2)">${g.away_abbr}</div>
        <div class="tname">${g.away_team}</div>
        ${scores}
      </div>
    </div>
    <div class="model-row">
      <div class="model-badge elo">Elo: ${p.elo_home_prob}%</div>
      ${mlBadge}
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
    <div class="wbanner">
      <span>🏆</span>
      <div><div class="wlbl">Predicted Winner</div><div class="wname">${p.predicted_winner}</div></div>
    </div>
    ${kellyHtml}
    ${mktHtml}
    ${factorRows ? `<div class="ftable">
      <div class="fhdr"><div>Factor</div><div style="text-align:center">${g.away_abbr}</div><div style="text-align:center">${g.home_abbr}</div></div>
      ${factorRows}
    </div>` : ''}
  </div>`;
}

// ── ELO ────────────────────────────────────────────────────────────────────
let eloLoaded = false;
async function loadElo() {
  if (eloLoaded) return;
  const res = await fetch('/api/elo');
  const data = await res.json();
  if (!data.ratings.length) {
    document.getElementById('elo-wrap').innerHTML =
      '<div class="warnbox">No Elo data yet — ratings build up as games are completed and processed.</div>';
    return;
  }
  const maxR = Math.max(...data.ratings.map(r => r.rating));
  const minR = Math.min(...data.ratings.map(r => r.rating));
  const rows = data.ratings.map(r => {
    const pct = ((r.rating - minR) / (maxR - minR + 1)) * 100;
    const sign = r.diff >= 0 ? '+' : '';
    return `<div class="erow">
      <div class="erank ${r.rank<=3?'top3':''}">${r.rank<=3?['🥇','🥈','🥉'][r.rank-1]:r.rank}</div>
      <div class="ename">${r.team}<small>${r.abbr}</small></div>
      <div class="erate">${r.rating}</div>
      <div class="ebar-wrap ecol5"><div class="ebar"><div class="efill" style="width:${pct}%"></div></div></div>
      <div class="ediff ${r.diff>=0?'pos':'neg'}">${sign}${r.diff}</div>
    </div>`;
  }).join('');
  document.getElementById('elo-wrap').innerHTML = `<div class="etable">
    <div class="ehdr"><div>#</div><div>Team</div><div>Rating</div><div class="ecol5">Strength</div><div>vs 1500</div></div>
    ${rows}
  </div>`;
  eloLoaded = true;
}

// ── BACKTEST ────────────────────────────────────────────────────────────────
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
    const gapClass = Math.abs(c.gap) < 5 ? 'grn' : (Math.abs(c.gap) < 12 ? '' : 'red');
    return `<div class="cal-row">
      <div>${c.bucket}</div>
      <div>${c.avg_predicted}%</div>
      <div>${c.actual_win_rate}%</div>
      <div>${c.n}</div>
      <div class="gap-val ${gapClass}">${c.gap > 0 ? '+' : ''}${c.gap}%</div>
    </div>`;
  }).join('');

  document.getElementById('backtest-wrap').innerHTML = `
    <div class="bstats">
      <div class="bstat"><div class="bstat-val gold">${d.accuracy}%</div><div class="bstat-lbl">Prediction Accuracy</div><div class="bstat-sub">Baseline: 50%</div></div>
      <div class="bstat"><div class="bstat-val">${d.brier_score}</div><div class="bstat-lbl">Brier Score</div><div class="bstat-sub">Random = 0.25</div></div>
      <div class="bstat"><div class="bstat-val">${d.log_loss}</div><div class="bstat-lbl">Log Loss</div><div class="bstat-sub">Lower is better</div></div>
      <div class="bstat"><div class="bstat-val ${roiColor}">${d.roi_pct}%</div><div class="bstat-lbl">Paper Trading ROI</div><div class="bstat-sub">${d.bets_placed} bets placed</div></div>
      <div class="bstat"><div class="bstat-val ${pnlColor}">£${d.pnl}</div><div class="bstat-lbl">P&L (£100/bet)</div><div class="bstat-sub">Started: £${d.starting_bankroll}</div></div>
      <div class="bstat"><div class="bstat-val">${d.total_games}</div><div class="bstat-lbl">Games Backtested</div><div class="bstat-sub">${d.bets_won} bets won</div></div>
    </div>
    <div class="section-title">Calibration Analysis</div>
    <div class="infobox">A perfectly calibrated model has 0% gap — when we say 70%, the team wins 70% of the time. Positive gap = we're underconfident, negative = overconfident.</div>
    <div class="cal-table">
      <div class="cal-hdr"><div>Bucket</div><div>Avg Predicted</div><div>Actual Win%</div><div>Sample</div><div>Gap</div></div>
      ${calRows || '<div style="padding:14px;color:var(--muted);text-align:center">Not enough data per bucket yet</div>'}
    </div>`;
  btLoaded = true;
}

// ── ML MODEL ────────────────────────────────────────────────────────────────
let mlLoaded = false;
async function loadModel() {
  if (mlLoaded) return;
  const res = await fetch('/api/model');
  const d = await res.json();
  if (!d.trained) {
    document.getElementById('model-wrap').innerHTML = `
      <div class="warnbox">⏳ ML model not yet trained. Collecting game data: <strong>${d.games_collected}/${d.games_collected + d.games_needed}</strong> games.<br>
      The model activates after 50 completed games and retrains every 10 games automatically.</div>`;
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
      <div style="margin-top:12px;font-size:12px;color:var(--muted)">Trained on <strong style="color:var(--text)">${d.n_games}</strong> games. Ensemble weights: LR ${(d.weights.lr*100||0).toFixed(0)}% · RF ${(d.weights.rf*100||0).toFixed(0)}% · GB ${(d.weights.gb*100||0).toFixed(0)}%</div>
    </div>
    <div class="mcard">
      <h3>🔑 Feature Importance (Random Forest)</h3>
      <p style="color:var(--muted);font-size:12px;margin-bottom:12px">Which inputs matter most to the model?</p>
      ${fiRows}
    </div>`;
  mlLoaded = true;
}

// ── INIT ────────────────────────────────────────────────────────────────────
loadGames();
setInterval(loadGames, 90000);
</script>
</body>
</html>
"""

app = Flask(__name__)

BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
API_KEY = os.environ.get("BALLDONTLIE_KEY", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

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

elo_ratings = load_ratings()
ml_model_data = load_model()

def get_todays_games():
    today = date.today().isoformat()
    data = cached_get(f"{BALLDONTLIE_BASE}/games", {"dates[]": today, "per_page": 30}, ttl=60)
    return data["data"] if data and "data" in data else []

def get_recent_games(team_id, n=15):
    data = cached_get(f"{BALLDONTLIE_BASE}/games",
                      {"team_ids[]": team_id, "per_page": n, "seasons[]": 2024}, ttl=300)
    return data["data"] if data and "data" in data else []

def get_all_teams():
    data = cached_get(f"{BALLDONTLIE_BASE}/teams", {"per_page": 30}, ttl=86400)
    if data and "data" in data:
        return {t["id"]: t for t in data["data"]}
    return {}

def get_market_odds(home_name, away_name):
    if not ODDS_API_KEY:
        return None
    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/",
            params={"apiKey": ODDS_API_KEY, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"},
            timeout=8
        )
        if r.status_code != 200:
            return None
        for event in r.json():
            h = event.get("home_team", "")
            a = event.get("away_team", "")
            if home_name.lower() in h.lower() or away_name.lower() in a.lower():
                for bm in event.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "h2h":
                            outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
                            ho = outcomes.get(h)
                            ao = outcomes.get(a)
                            if ho and ao:
                                total = 1/ho + 1/ao
                                return {
                                    "home_implied": round((1/ho)/total, 4),
                                    "away_implied": round((1/ao)/total, 4),
                                    "home_odds": ho, "away_odds": ao,
                                    "bookmaker": bm["title"]
                                }
    except Exception as e:
        print(f"Odds API: {e}")
    return None

def win_pct(games, team_id):
    wins, total = 0, 0
    for g in games:
        if g["status"] != "Final": continue
        home = g["home_team"]["id"] == team_id
        won = (g["home_team_score"] > g["visitor_team_score"]) if home \
              else (g["visitor_team_score"] > g["home_team_score"])
        wins += int(won); total += 1
    return wins / total if total > 0 else 0.5

def avg_diff(games, team_id):
    diffs = []
    for g in games:
        if g["status"] != "Final": continue
        home = g["home_team"]["id"] == team_id
        d = (g["home_team_score"] - g["visitor_team_score"]) if home \
            else (g["visitor_team_score"] - g["home_team_score"])
        diffs.append(d)
    return sum(diffs) / len(diffs) if diffs else 0

def logistic(x):
    return 1 / (1 + math.exp(-x))

def update_elo_from_games(games):
    global elo_ratings, ml_model_data
    history = load_history()
    existing_ids = {g.get("game_id") for g in history}
    changed = False
    for g in games:
        if g["status"] != "Final": continue
        gid = str(g["id"])
        if gid in existing_ids: continue
        home = g["home_team"]["full_name"]
        away = g["visitor_team"]["full_name"]
        hs = g.get("home_team_score") or 0
        as_ = g.get("visitor_team_score") or 0
        home_won = hs > as_
        margin = abs(hs - as_)
        home_recent = get_recent_games(g["home_team"]["id"], 10)
        away_recent = get_recent_games(g["visitor_team"]["id"], 10)
        hform = win_pct(home_recent, g["home_team"]["id"])
        aform = win_pct(away_recent, g["visitor_team"]["id"])
        hdiff = avg_diff(home_recent, g["home_team"]["id"])
        adiff = avg_diff(away_recent, g["visitor_team"]["id"])
        h_elo = elo_ratings.get(home, DEFAULT_RATING)
        a_elo = elo_ratings.get(away, DEFAULT_RATING)
        record = {"game_id": gid, "home_team": home, "away_team": away,
                  "home_elo": h_elo, "away_elo": a_elo,
                  "home_form": hform, "away_form": aform,
                  "home_avg_diff": hdiff, "away_avg_diff": adiff,
                  "home_won": home_won, "home_score": hs, "away_score": as_,
                  "date": g.get("date", "")[:10]}
        h_ep, _ = get_elo_win_prob(elo_ratings, home, away)
        ml_p, _ = predict_with_ml(ml_model_data, record)
        record["predicted_home_prob"] = (ml_p * 0.6 + h_ep * 0.4) if ml_p else h_ep
        elo_ratings = update_elo(elo_ratings, home, away, home_won, margin)
        history.append(record)
        existing_ids.add(gid)
        changed = True
    if changed:
        save_ratings(elo_ratings)
        save_history(history)
        if len(history) >= 50 and len(history) % 10 == 0:
            ml_model_data = train_model(history)
            print(f"Model retrained on {len(history)} games")

def predict_game(game):
    home_name = game["home_team"]["full_name"]
    away_name = game["visitor_team"]["full_name"]
    home_id = game["home_team"]["id"]
    away_id = game["visitor_team"]["id"]
    home_recent = get_recent_games(home_id, 10)
    away_recent = get_recent_games(away_id, 10)
    hform = win_pct(home_recent, home_id)
    aform = win_pct(away_recent, away_id)
    hdiff = avg_diff(home_recent, home_id)
    adiff = avg_diff(away_recent, away_id)
    home_elo = elo_ratings.get(home_name, DEFAULT_RATING)
    away_elo = elo_ratings.get(away_name, DEFAULT_RATING)
    elo_home_prob, _ = get_elo_win_prob(elo_ratings, home_name, away_name)
    record = {"home_elo": home_elo, "away_elo": away_elo,
              "home_form": hform, "away_form": aform,
              "home_avg_diff": hdiff, "away_avg_diff": adiff}
    ml_home_prob, _ = predict_with_ml(ml_model_data, record)
    live_factor = 0
    period = game.get("period", 0)
    if period and period > 0:
        hs = game.get("home_team_score") or 0
        as_ = game.get("visitor_team_score") or 0
        live_factor = (hs - as_) * min(period / 4.0, 1.0) * 0.6
    score = (hform - aform) * 30 + (hdiff - adiff) * 0.4 + 3 * 0.5 + live_factor
    rule_home_prob = logistic(score / 15)
    if ml_home_prob:
        final_home = ml_home_prob * 0.50 + elo_home_prob * 0.35 + rule_home_prob * 0.15
    else:
        final_home = elo_home_prob * 0.60 + rule_home_prob * 0.40
    final_away = 1 - final_home
    market = get_market_odds(home_name, away_name)
    kelly_home = kelly_criterion(final_home, market["home_implied"]) if market else None
    kelly_away = kelly_criterion(final_away, market["away_implied"]) if market else None
    margin_val = abs(final_home - final_away)
    confidence = "High" if margin_val > 0.25 else ("Medium" if margin_val > 0.12 else "Low")
    factors = [
        {"name": "Elo Rating", "home": str(int(home_elo)), "away": str(int(away_elo)),
         "edge": "home" if home_elo > away_elo else "away"},
        {"name": "Recent Form (L10)", "home": f"{hform*100:.0f}%", "away": f"{aform*100:.0f}%",
         "edge": "home" if hform > aform else "away"},
        {"name": "Avg Point Diff", "home": f"{hdiff:+.1f}", "away": f"{adiff:+.1f}",
         "edge": "home" if hdiff > adiff else "away"},
        {"name": "Home Court", "home": "+3 pts", "away": "—", "edge": "home"},
    ]
    if period and period > 0:
        hs = game.get("home_team_score") or 0
        as_ = game.get("visitor_team_score") or 0
        factors.append({"name": f"Live Score (Q{period})", "home": str(hs), "away": str(as_),
                        "edge": "home" if hs > as_ else ("away" if as_ > hs else "even")})
    return {
        "home_prob": round(final_home * 100, 1),
        "away_prob": round(final_away * 100, 1),
        "elo_home_prob": round(elo_home_prob * 100, 1),
        "ml_home_prob": round(ml_home_prob * 100, 1) if ml_home_prob else None,
        "rule_home_prob": round(rule_home_prob * 100, 1),
        "confidence": confidence,
        "predicted_winner": home_name if final_home > 0.5 else away_name,
        "factors": factors,
        "home_elo": int(home_elo),
        "away_elo": int(away_elo),
        "market": market,
        "kelly_home": kelly_home,
        "kelly_away": kelly_away,
        "ml_active": ml_home_prob is not None,
    }

@app.route("/")
def index():
    return HTML_PAGE, 200, {'Content-Type': 'text/html'}

@app.route("/api/games")
def api_games():
    games = get_todays_games()
    try:
        update_elo_from_games(games)
    except Exception as e:
        print(f"Elo update error: {e}")
    results = []
    for g in games:
        try:
            pred = predict_game(g)
        except Exception as e:
            print(f"Prediction error: {e}")
            pred = {"home_prob": 50, "away_prob": 50, "confidence": "N/A",
                    "predicted_winner": "TBD", "factors": [], "home_elo": 1500,
                    "away_elo": 1500, "market": None, "kelly_home": None,
                    "kelly_away": None, "ml_active": False,
                    "elo_home_prob": 50, "ml_home_prob": None, "rule_home_prob": 50}
        status = g.get("status", "")
        display_status = ("Final" if status == "Final"
                          else (f"Q{g['period']} LIVE" if g.get("period", 0) > 0 else status))
        results.append({
            "id": g["id"],
            "home_team": g["home_team"]["full_name"],
            "home_abbr": g["home_team"]["abbreviation"],
            "away_team": g["visitor_team"]["full_name"],
            "away_abbr": g["visitor_team"]["abbreviation"],
            "home_score": g.get("home_team_score") or 0,
            "away_score": g.get("visitor_team_score") or 0,
            "status": display_status,
            "period": g.get("period", 0),
            "prediction": pred,
        })
    results.sort(key=lambda g: (0 if "LIVE" in g["status"] else
                                (2 if g["status"] == "Final" else 1)))
    return jsonify({"games": results, "fetched_at": datetime.now().strftime("%H:%M:%S"),
                    "ml_active": ml_model_data is not None,
                    "games_in_history": len(load_history())})

@app.route("/api/elo")
def api_elo():
    ratings = get_all_ratings(elo_ratings)
    all_teams = get_all_teams()
    name_abbr = {t["full_name"]: t["abbreviation"] for t in all_teams.values()}
    return jsonify({"ratings": [
        {"rank": i+1, "team": name, "rating": round(rating, 1),
         "abbr": name_abbr.get(name, name[:3].upper()),
         "diff": round(rating - DEFAULT_RATING, 1)}
        for i, (name, rating) in enumerate(ratings)
    ]})

@app.route("/api/backtest")
def api_backtest():
    history = load_history()
    if len(history) < 10:
        return jsonify({"error": f"Need at least 10 completed games. Have {len(history)} so far."})
    return jsonify(run_backtest(history))

@app.route("/api/model")
def api_model():
    if not ml_model_data:
        history = load_history()
        return jsonify({"trained": False, "games_collected": len(history),
                        "games_needed": max(0, 50 - len(history))})
    return jsonify({
        "trained": True,
        "metrics": ml_model_data.get("metrics", {}),
        "feature_importance": ml_model_data.get("feature_importance", {}),
        "calibration": ml_model_data.get("calibration", {}),
        "n_games": ml_model_data.get("n_games", 0),
        "weights": ml_model_data.get("weights", {}),
    })

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    history = load_history()
    if len(history) < 50:
        return jsonify({"error": f"Need 50 games, have {len(history)}"})
    global ml_model_data
    ml_model_data = train_model(history)
    return jsonify({"success": True, "metrics": ml_model_data["metrics"]})

if __name__ == "__main__":
    print("\n🏀 NBA Predictor v2 — Quant Edition")
    print("   Open http://localhost:5000\n")
    history = load_history()
    if len(history) >= 50 and not ml_model_data:
        print(f"   Training ML model on {len(history)} games...")
        ml_model_data = train_model(history)
        print("   Model ready.\n")
    app.run(debug=False, port=5000, threaded=True)
