"""
Elo Rating System for NBA Teams
--------------------------------
Elo is a self-updating rating system originally designed for chess.
After every game, the winner gains points and the loser loses points.
The amount gained/lost depends on how surprising the result was.

Key parameters:
- K factor: how much a single game shifts ratings (we use 20, NBA standard)
- Home advantage: added to home team's effective rating (~100 pts = ~3pts on court)
- Starting rating: 1500 for all teams (conventional baseline)
"""

import json
import os
from datetime import datetime

ELO_FILE = "elo_ratings.json"
DEFAULT_RATING = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 100  # in Elo points, roughly equivalent to ~3 pts on court


def load_ratings():
    """Load saved Elo ratings from disk, or return defaults."""
    if os.path.exists(ELO_FILE):
        with open(ELO_FILE) as f:
            return json.load(f)
    return {}


def save_ratings(ratings):
    with open(ELO_FILE, "w") as f:
        json.dump(ratings, f, indent=2)


def expected_score(rating_a, rating_b):
    """
    Logistic formula: probability team A beats team B.
    This is the core Elo equation — derived from the logistic distribution.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(ratings, home_team, away_team, home_won, margin=None):
    """
    Update Elo ratings after a game result.
    
    margin: point differential — used for margin-of-victory multiplier
    (teams that win big gain more than teams that squeak by)
    """
    home_r = ratings.get(home_team, DEFAULT_RATING)
    away_r = ratings.get(away_team, DEFAULT_RATING)

    # Apply home advantage to expected score
    home_exp = expected_score(home_r + HOME_ADVANTAGE, away_r)
    away_exp = 1 - home_exp

    # Actual result (1 = win, 0 = loss)
    home_actual = 1.0 if home_won else 0.0
    away_actual = 1.0 - home_actual

    # Margin of victory multiplier (FiveThirtyEight method)
    # Winning by 20 should move ratings more than winning by 1
    if margin is not None:
        mov_mult = max(1.0, (abs(margin) ** 0.8) / 7.5)
    else:
        mov_mult = 1.0

    effective_k = K_FACTOR * mov_mult

    ratings[home_team] = home_r + effective_k * (home_actual - home_exp)
    ratings[away_team] = away_r + effective_k * (away_actual - away_exp)

    return ratings


def get_elo_win_prob(ratings, home_team, away_team):
    """
    Return (home_win_prob, away_win_prob) based on current Elo ratings.
    Includes home court advantage.
    """
    home_r = ratings.get(home_team, DEFAULT_RATING)
    away_r = ratings.get(away_team, DEFAULT_RATING)
    home_prob = expected_score(home_r + HOME_ADVANTAGE, away_r)
    return round(home_prob, 4), round(1 - home_prob, 4)


def get_all_ratings(ratings):
    """Return sorted list of (team, rating) pairs."""
    return sorted(ratings.items(), key=lambda x: -x[1])
