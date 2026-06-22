# Pinned to match the Python used to verify the dependency set locally.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_PORT=8000

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code (no PDF / no ingestion: the container only reads Qdrant).
COPY src/ ./src/
COPY app.py chainlit.md ./

# Run as an unprivileged user.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# exec replaces the shell so Chainlit becomes PID 1 and receives signals
# directly (graceful shutdown); the shell is only used to expand ${APP_PORT}.
CMD ["sh", "-c", "exec chainlit run app.py --host 0.0.0.0 --port ${APP_PORT:-8000}"]
