"""
Backtesting Engine + Kelly Criterion
--------------------------------------

BACKTESTING:
Simulates running the prediction model on historical games and 
measures how accurate it would have been. Key metrics:

- Accuracy: % of games where we correctly predicted the winner
- Brier Score: measures probability calibration (lower = better)
  BS = mean((predicted_prob - actual_outcome)^2)
  Perfect = 0.0, Random = 0.25
- Log Loss: penalises confident wrong predictions heavily
- ROI: return on investment if betting flat stakes on every prediction

KELLY CRITERION:
Optimal bet sizing formula from information theory (John Kelly, 1956).
Used by every serious quant trading firm.

  f* = (bp - q) / b

Where:
  f* = fraction of bankroll to bet
  b  = decimal odds minus 1 (the "edge" offered)
  p  = our estimated probability of winning
  q  = 1 - p (probability of losing)

If our model says 65% and the market implies 55%, we have edge.
Kelly tells us exactly how much to bet to maximise long-run growth.

We use HALF-KELLY (f*/2) as a risk management measure — full Kelly
is theoretically optimal but has very high variance in practice.
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta

BACKTEST_FILE = "backtest_results.json"


def kelly_criterion(our_prob, market_prob, bankroll=1000, half_kelly=True):
    """
    Calculate Kelly bet size.
    
    our_prob: our model's win probability
    market_prob: implied probability from betting market odds
    bankroll: current bankroll
    half_kelly: use half-Kelly for risk management
    """
    if market_prob >= 1.0 or market_prob <= 0:
        return {"edge": 0, "kelly_fraction": 0, "bet_size": 0, "has_edge": False}

    # Convert market probability to decimal odds
    # If market says 55% chance, they offer 1/0.55 = 1.82x payout
    decimal_odds = 1 / market_prob
    b = decimal_odds - 1  # profit per £1 bet

    p = our_prob
    q = 1 - p

    # Kelly formula
    f_star = (b * p - q) / b

    if half_kelly:
        f_star = f_star / 2

    # Edge = our probability minus market probability
    edge = our_prob - market_prob

    bet_size = max(0, f_star * bankroll)

    return {
        "edge": round(edge * 100, 2),          # as percentage
        "kelly_fraction": round(max(0, f_star), 4),
        "bet_size": round(bet_size, 2),
        "decimal_odds": round(decimal_odds, 3),
        "has_edge": edge > 0.02,               # only bet if edge > 2%
        "expected_value": round((p * b - q) * 100, 2),  # EV per £100
    }


def run_backtest(history, model_predictions=None):
    """
    Run backtest on historical game data.
    
    history: list of game records with outcomes
    model_predictions: optional pre-computed predictions, else use Elo
    
    Returns comprehensive performance statistics.
    """
    if len(history) < 10:
        return {"error": "Need at least 10 games to backtest"}

    results = []
    correct = 0
    total = 0
    brier_scores = []
    log_losses = []

    # Simulate paper trading (flat £100 per bet)
    bankroll = 10000
    bankroll_history = [bankroll]
    bet_size = 100
    pnl = 0
    bets_placed = 0
    bets_won = 0

    for i, game in enumerate(history):
        if "home_won" not in game:
            continue

        pred_prob = game.get("predicted_home_prob", 0.5)
        actual = 1 if game["home_won"] else 0

        # Accuracy
        predicted_win = pred_prob > 0.5
        correct += int(predicted_win == game["home_won"])
        total += 1

        # Brier score
        brier_scores.append((pred_prob - actual) ** 2)

        # Log loss (clip to avoid log(0))
        pred_clipped = max(min(pred_prob, 0.999), 0.001)
        ll = -(actual * np.log(pred_clipped) + (1 - actual) * np.log(1 - pred_clipped))
        log_losses.append(ll)

        # Paper trading: bet on our predicted winner if confidence > 55%
        confidence = abs(pred_prob - 0.5)
        if confidence > 0.05:
            bets_placed += 1
            if predicted_win == game["home_won"]:
                pnl += bet_size * 0.9  # -10% vig (bookmaker margin)
                bets_won += 1
                bankroll += bet_size * 0.9
            else:
                pnl -= bet_size
                bankroll -= bet_size
            bankroll_history.append(round(bankroll, 2))

        results.append({
            "game": f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
            "date": game.get("date", ""),
            "predicted_prob": round(pred_prob, 3),
            "actual": actual,
            "correct": predicted_win == game["home_won"],
        })

    accuracy = correct / total if total > 0 else 0
    avg_brier = np.mean(brier_scores)
    avg_log_loss = np.mean(log_losses)
    roi = (pnl / (bets_placed * bet_size)) * 100 if bets_placed > 0 else 0

    # Calibration buckets
    calibration = compute_calibration(results)

    summary = {
        "total_games": total,
        "accuracy": round(accuracy * 100, 2),
        "brier_score": round(float(avg_brier), 4),
        "log_loss": round(float(avg_log_loss), 4),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "bet_accuracy": round(bets_won / bets_placed * 100, 2) if bets_placed > 0 else 0,
        "pnl": round(pnl, 2),
        "roi_pct": round(roi, 2),
        "starting_bankroll": 10000,
        "final_bankroll": round(bankroll, 2),
        "bankroll_history": bankroll_history[-50:],  # last 50 data points
        "calibration": calibration,
        "results": results[-20:],  # last 20 for display
        "baseline_accuracy": 50.0,  # random guessing baseline
        "baseline_brier": 0.25,     # random 50/50 baseline
    }

    # Save
    with open(BACKTEST_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def compute_calibration(results):
    """
    Group predictions into buckets (e.g. 50-60%, 60-70%...)
    and check if actual win rates match predicted probabilities.
    A perfectly calibrated model: 70% predictions win 70% of the time.
    """
    buckets = {}
    for i in range(5, 10):
        low = i * 10
        high = low + 10
        key = f"{low}-{high}%"
        buckets[key] = {"predicted": [], "actual": []}

    for r in results:
        prob = max(r["predicted_prob"], 1 - r["predicted_prob"])  # always from winner's perspective
        actual = r["correct"]
        for i in range(5, 10):
            low = i * 10 / 100
            high = (i + 1) * 10 / 100
            if low <= prob < high:
                key = f"{i*10}-{(i+1)*10}%"
                buckets[key]["predicted"].append(prob)
                buckets[key]["actual"].append(actual)
                break

    calibration_out = []
    for key, vals in buckets.items():
        if vals["predicted"]:
            calibration_out.append({
                "bucket": key,
                "avg_predicted": round(np.mean(vals["predicted"]) * 100, 1),
                "actual_win_rate": round(np.mean(vals["actual"]) * 100, 1),
                "n": len(vals["predicted"]),
                "gap": round((np.mean(vals["actual"]) - np.mean(vals["predicted"])) * 100, 1)
            })

    return calibration_out


def load_backtest():
    if os.path.exists(BACKTEST_FILE):
        with open(BACKTEST_FILE) as f:
            return json.load(f)
    return None
