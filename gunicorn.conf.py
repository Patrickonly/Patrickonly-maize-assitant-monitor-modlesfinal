import os

# ─── Render requires binding to 0.0.0.0:$PORT ─────────────────────────────────
port_env = os.environ.get('PORT')
bind = f"0.0.0.0:{port_env if port_env else '10000'}"

# Single worker to stay within free-tier RAM (512 MB)
workers = 1
threads = 2

# Long timeout for slow model inference on free-tier CPUs
timeout = 300
graceful_timeout = 120

# Don't preload — port binds FIRST, models load in background thread
preload_app = False

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
