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

app = Flask(__name__)

BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
API_KEY = "58367162-22cb-4de9-94fc-99b41c568752"
ODDS_API_KEY = ""

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
    return HTML_PAGE

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
import os
app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), threaded=True)
