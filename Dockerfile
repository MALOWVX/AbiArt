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
RUN chmod +x start.sh

# Railway / Render injects $PORT at runtime
EXPOSE 5000

# Use run.py which dynamically binds to the exact $PORT provided at runtime
CMD ["python", "run.py"]

