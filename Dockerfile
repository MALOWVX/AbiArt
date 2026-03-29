FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Railway injects $PORT at runtime
EXPOSE ${PORT:-5000}

# Shell form so $PORT is expanded by the shell
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 300 app:app
