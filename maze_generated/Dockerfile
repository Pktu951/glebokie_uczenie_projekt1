# ════════════════════════════════════════════════════════════════════
# SMA Maze Solver — Dockerfile
# Slime Mould Algorithm for Maze Pathfinding
# Przedmiot: Głębokie Nauczanie
# ════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Metadata
LABEL maintainer="student"
LABEL description="SMA Maze Solver - Slime Mould Algorithm pathfinding with web GUI"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy/matplotlib if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/')" || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
