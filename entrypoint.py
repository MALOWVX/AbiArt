"""
AbiArt Docker Entrypoint
========================
Intercepts any command passed by container orchestrators (Railway, Render, etc.),
sanitizes literal "$PORT" strings, and executes the target process safely.
"""

import os
import sys
import subprocess

def main():
    raw_port = os.environ.get("PORT", "5000").strip()
    port = raw_port if raw_port.isdigit() else "5000"
    os.environ["PORT"] = port

    args = sys.argv[1:]

    # Default to python run.py if no arguments given
    if not args:
        args = [sys.executable, "run.py"]
    else:
        # If a single string containing spaces was passed (e.g. "gunicorn app:app --bind ...")
        if len(args) == 1 and " " in args[0]:
            import shlex
            args = shlex.split(args[0])

        # Replace any literal $PORT expressions
        sanitized = []
        for a in args:
            clean = (
                a.replace("$PORT", port)
                 .replace("${PORT}", port)
                 .replace("${PORT:-5000}", port)
            )
            sanitized.append(clean)
        args = sanitized

    print(f"[Entrypoint] Executing: {' '.join(args)} (resolved PORT={port})")
    sys.stdout.flush()

    if sys.platform != "win32":
        os.execvp(args[0], args)
    else:
        sys.exit(subprocess.call(args))

if __name__ == "__main__":
    main()
