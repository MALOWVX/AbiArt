#!/usr/bin/env python
"""
AbiArt Gunicorn Wrapper
=======================
Guarantees that any direct call to 'gunicorn' automatically resolves literal '$PORT'
to the environment variable PORT (or 5000 default).
"""

import os
import sys

def main():
    raw_port = os.environ.get("PORT", "5000").strip()
    port = raw_port if raw_port.isdigit() else "5000"

    real_bin = "/usr/local/bin/gunicorn_bin"
    if not os.path.exists(real_bin):
        # Fallback if gunicorn_bin was not moved
        real_bin = sys.executable
        args = [sys.executable, "-m", "gunicorn"]
    else:
        args = [real_bin]

    for arg in sys.argv[1:]:
        clean = (
            arg.replace("$PORT", port)
               .replace("${PORT}", port)
               .replace("${PORT:-5000}", port)
        )
        args.append(clean)

    if sys.platform != "win32":
        os.execv(args[0], args)
    else:
        import subprocess
        sys.exit(subprocess.call(args))

if __name__ == "__main__":
    main()
