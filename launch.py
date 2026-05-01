#!/usr/bin/env python3
"""NBA Predictor v2 — run this to start the app"""
import subprocess, sys, os, time, webbrowser, threading

def open_browser():
    time.sleep(1.8)
    webbrowser.open("http://localhost:5000")

print("=" * 55)
print("  🏀  NBA QUANT PREDICTOR v2")
print("=" * 55)

# Install deps if needed
try:
    import flask, requests, sklearn
except ImportError:
    print("Installing packages (one-time)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "flask", "requests", "scikit-learn", "numpy", "scipy"])

print("\nStarting at http://localhost:5000")
print("Opening browser automatically...")
print("Press CTRL+C to stop\n")
threading.Thread(target=open_browser, daemon=True).start()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "app.py"])
