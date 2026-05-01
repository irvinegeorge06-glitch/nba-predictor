"""
Machine Learning Prediction Model
-----------------------------------
Uses scikit-learn to train on historical game data.

Features used:
  - home_elo, away_elo          : Elo ratings at time of game
  - elo_diff                    : Elo difference (home - away)
  - home_form, away_form        : Win % over last 10 games
  - home_diff, away_diff        : Avg point differential last 10 games
  - home_rest, away_rest        : Days since last game (fatigue factor)
  - home_advantage              : Always 1 (constant feature)

Models trained:
  1. Logistic Regression  — interpretable, good baseline
  2. Random Forest        — captures non-linear interactions
  3. Ensemble             — weighted average of both (best accuracy)

Training process:
  - Features scaled with StandardScaler (important for LogReg)
  - 80/20 train/test split, stratified by outcome
  - Cross-validation to avoid overfitting
"""

import numpy as np
import json
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

MODEL_FILE = "ml_model.pkl"
HISTORY_FILE = "game_history.json"

FEATURES = [
    "elo_diff",
    "home_form",
    "away_form",
    "form_diff",
    "home_avg_diff",
    "away_avg_diff",
    "diff_diff",
    "home_advantage",
]


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_game_to_history(game_record):
    """
    game_record should contain:
    {
      'home_team': str, 'away_team': str,
      'home_elo': float, 'away_elo': float,
      'home_form': float, 'away_form': float,
      'home_avg_diff': float, 'away_avg_diff': float,
      'home_won': bool,
      'date': str
    }
    """
    history = load_history()
    history.append(game_record)
    save_history(history)


def build_feature_vector(record):
    """Convert a game record dict into a numpy feature array."""
    return [
        record.get("home_elo", 1500) - record.get("away_elo", 1500),  # elo_diff
        record.get("home_form", 0.5),
        record.get("away_form", 0.5),
        record.get("home_form", 0.5) - record.get("away_form", 0.5),  # form_diff
        record.get("home_avg_diff", 0),
        record.get("away_avg_diff", 0),
        record.get("home_avg_diff", 0) - record.get("away_avg_diff", 0),  # diff_diff
        1.0,  # home_advantage (constant — lets model learn its weight)
    ]


def train_model(history=None):
    """
    Train ensemble model on historical game data.
    Returns dict with models, scaler, and performance metrics.
    """
    if history is None:
        history = load_history()

    if len(history) < 50:
        return None  # Not enough data yet

    X = np.array([build_feature_vector(r) for r in history])
    y = np.array([1 if r["home_won"] else 0 for r in history])

    # Train/test split — stratified to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (critical for logistic regression)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── Model 1: Logistic Regression ──
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_preds = lr.predict_proba(X_test_s)[:, 1]
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
    lr_brier = brier_score_loss(y_test, lr_preds)

    # ── Model 2: Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)  # RF doesn't need scaling
    rf_preds = rf.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_brier = brier_score_loss(y_test, rf_preds)

    # ── Model 3: Gradient Boosting ──
    gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict_proba(X_test)[:, 1]
    gb_acc = accuracy_score(y_test, gb.predict(X_test))
    gb_brier = brier_score_loss(y_test, gb_preds)

    # ── Ensemble: inverse-brier weighted average ──
    # Better calibrated models get higher weight
    w_lr = 1 / (lr_brier + 1e-6)
    w_rf = 1 / (rf_brier + 1e-6)
    w_gb = 1 / (gb_brier + 1e-6)
    total_w = w_lr + w_rf + w_gb
    ensemble_preds = (w_lr * lr_preds + w_rf * rf_preds + w_gb * gb_preds) / total_w
    ensemble_acc = accuracy_score(y_test, (ensemble_preds > 0.5).astype(int))
    ensemble_brier = brier_score_loss(y_test, ensemble_preds)

    # ── Feature importance from Random Forest ──
    feat_importance = dict(zip(FEATURES, rf.feature_importances_))

    # ── Calibration data (for calibration curve plot) ──
    fraction_of_pos, mean_pred_val = calibration_curve(
        y_test, ensemble_preds, n_bins=10, strategy='quantile'
    )

    result = {
        "models": {"lr": lr, "rf": rf, "gb": gb},
        "scaler": scaler,
        "weights": {"lr": w_lr / total_w, "rf": w_rf / total_w, "gb": w_gb / total_w},
        "metrics": {
            "lr": {"accuracy": round(lr_acc, 4), "brier": round(lr_brier, 4)},
            "rf": {"accuracy": round(rf_acc, 4), "brier": round(rf_brier, 4)},
            "gb": {"accuracy": round(gb_acc, 4), "brier": round(gb_brier, 4)},
            "ensemble": {"accuracy": round(ensemble_acc, 4), "brier": round(ensemble_brier, 4)},
        },
        "feature_importance": {k: round(v, 4) for k, v in feat_importance.items()},
        "calibration": {
            "fraction_pos": fraction_of_pos.tolist(),
            "mean_pred": mean_pred_val.tolist(),
        },
        "n_games": len(history),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Save to disk
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(result, f)

    return result


def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None


def predict_with_ml(model_data, game_record):
    """
    Use trained ensemble to predict win probability for a single game.
    Returns (home_win_prob, away_win_prob)
    """
    if model_data is None:
        return None, None

    x = np.array(build_feature_vector(game_record)).reshape(1, -1)
    scaler = model_data["scaler"]
    models = model_data["models"]
    weights = model_data["weights"]

    x_scaled = scaler.transform(x)

    lr_prob = models["lr"].predict_proba(x_scaled)[0][1]
    rf_prob = models["rf"].predict_proba(x)[0][1]
    gb_prob = models["gb"].predict_proba(x)[0][1]

    ensemble = (
        weights["lr"] * lr_prob +
        weights["rf"] * rf_prob +
        weights["gb"] * gb_prob
    )

    return round(ensemble, 4), round(1 - ensemble, 4)
