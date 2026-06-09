import os

# Automatically fallback to 10000 if PORT is not set, which Render assigns
port = os.environ.get("PORT", "10000")

# Force Gunicorn to bind to 0.0.0.0 and the correct port
bind = f"0.0.0.0:{port}"

# Use 1 worker to prevent OOM errors on Render free tier
workers = 1

# Optional: increase timeout for long-running inference (e.g., initial model load)
timeout = 120
