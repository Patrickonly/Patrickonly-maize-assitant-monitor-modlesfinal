import os

# Port binding
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Single worker to stay within free-tier RAM
workers = 1

# Threads per worker for handling concurrent requests
threads = 2

# Graceful timeout — gives worker time to load models on first request
graceful_timeout = 120

# Hard kill after 5 min — allows time for inference on slow free-tier CPUs
timeout = 300

# Don't preload — we want the port to bind FIRST, then models load on first request
preload_app = False

# Access log
accesslog = "-"
errorlog = "-"
loglevel = "info"
