# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install git (useful for workspace operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .
COPY html/ ./html/

# Workspace directory (mount or let it be created)
RUN mkdir -p /workspace

ENV COPILOT_WEB_HOST=0.0.0.0
ENV COPILOT_WEB_PORT=8765
ENV COPILOT_WORKSPACE=/workspace
# COPILOT_BIN defaults to "copilot" — mount the binary via volume or set this var

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/healthz')"

CMD ["python3", "app.py"]
