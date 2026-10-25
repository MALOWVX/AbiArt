FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (certificates for TLS/SSL and curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN chmod +x start.sh entrypoint.py gunicorn_wrapper.py

# Replace system gunicorn with self-sanitizing wrapper
RUN if [ -f /usr/local/bin/gunicorn ]; then \
      mv /usr/local/bin/gunicorn /usr/local/bin/gunicorn_bin && \
      cp gunicorn_wrapper.py /usr/local/bin/gunicorn && \
      chmod +x /usr/local/bin/gunicorn; \
    fi

# Railway / Render injects $PORT at runtime
EXPOSE 5000

# Intercept all startup commands through entrypoint to guarantee $PORT is sanitized
ENTRYPOINT ["python", "entrypoint.py"]
CMD ["python", "run.py"]
