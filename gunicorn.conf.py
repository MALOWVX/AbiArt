import os
import sys

# Get port from environment, defaulting to 5000 if unset or invalid
raw_port = os.environ.get("PORT", "5000").strip()
port = raw_port if raw_port.isdigit() else "5000"

# Intercept and sanitize any literal "$PORT" strings in CLI arguments (e.g. from runners that don't expand shell variables)
for i in range(len(sys.argv)):
    arg = sys.argv[i]
    if "$PORT" in arg:
        sys.argv[i] = arg.replace("$PORT", port)
    elif "${PORT}" in arg:
        sys.argv[i] = arg.replace("${PORT}", port)
    elif "${PORT:-5000}" in arg:
        sys.argv[i] = arg.replace("${PORT:-5000}", port)

# Default Gunicorn configurations
bind = f"0.0.0.0:{port}"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = 300
keepalive = 5
accesslog = "-"
errorlog = "-"
