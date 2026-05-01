# 🏀 NBA Live Predictor

AI-powered NBA match winner predictions with live data.

## How to Run

### Option 1 — Easiest
Double-click `launch.py` (if Python is associated with .py files)
OR open a terminal in this folder and run:
```
python launch.py
```

### Option 2 — Direct
```
pip install flask requests
python app.py
```
Then open http://localhost:5000 in your browser.

## What It Does

- **Fetches today's NBA games** in real-time
- **Predicts win probability** for each team using:
  - Last 10 games win % (form)
  - Average point differential
  - Home court advantage (~3pts historically)
  - Live score weighting (as game progresses)
- **Auto-refreshes every 90 seconds**
- **Hot/Cold Teams** tab shows current momentum rankings
- **Confidence rating** tells you how decisive each prediction is

## Data Sources
- BallDontLie API (free, no API key needed)
- Live scores update every ~90 seconds during games

## Requirements
- Python 3.8+
- Internet connection
- Packages: flask, requests (auto-installed by launch.py)
