import os

port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Single worker to stay within free-tier RAM
workers = 1

# Graceful timeout — gives the worker time to load models on cold start
graceful_timeout = 300

# Hard kill after 5 min (safety net)
timeout = 300

# Preload app so models load BEFORE the first request (faster cold start)
preload_app = True
