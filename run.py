"""
AbiArt Production Server Runner
===============================
Resolves $PORT safely across all cloud platforms (Railway, Render, Fly.io, etc.)
without relying on shell variable expansion.
"""

import os
import sys

def main():
    raw_port = os.environ.get("PORT", "5000").strip()
    port = raw_port if raw_port.isdigit() else "5000"
    workers = os.environ.get("WEB_CONCURRENCY", "2")
    # A generation holds its request open while the notebook computes (up to ~5 min). With threads, the
    # site keeps answering other requests meanwhile (second image of a batch, other users, page loads).
    threads = os.environ.get("WEB_THREADS", "8")

    print(f"[AbiArt] Starting production server on 0.0.0.0:{port} with {workers} workers x {threads} threads...")

    # Build gunicorn command arguments
    cmd = [
        sys.executable,
        "-m", "gunicorn",
        "app:app",
        "--bind", f"0.0.0.0:{port}",
        "--workers", workers,
        "--threads", threads,
        "--timeout", "300"
    ]

    # Replace current process with gunicorn
    if sys.platform != "win32":
        os.execv(sys.executable, cmd)
    else:
        import subprocess
        sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
