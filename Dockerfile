# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package.json .
RUN npm install

COPY frontend/ .
RUN npm run build
# Output: /app/frontend/dist/


# ── Stage 2: Python backend + Nginx ──────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    nginx \
    mdbtools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
COPY train_layer3.py train_layer4.py preprocess_cascade.py ./

# Copy built frontend from Stage 1
COPY --from=frontend-build /app/frontend/dist /app/static/frontend

# Nginx config
RUN rm -f /etc/nginx/sites-enabled/default
COPY frontend/docker/nginx.conf /etc/nginx/sites-enabled/failurescope.conf

# Entrypoint
COPY frontend/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Data volume (models, trajectories, NTSB data)
VOLUME ["/opt/failurescope_data"]

# Ports: 80 (Nginx → serves frontend + proxies API)
EXPOSE 80

ENV PYTHONPATH=/app
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODEL=mistral:7b

ENTRYPOINT ["/entrypoint.sh"]