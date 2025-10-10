# Minimal runtime image for Voxtral Wyoming STT (stub)
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WYOMING_HOST=0.0.0.0 \
    WYOMING_PORT=10300 \
    VOXTRAL_LANGUAGE=en-US \
    AUDIO_SAMPLE_RATE=16000 \
    LOG_LEVEL=INFO

# Install system dependencies (if needed later, keep minimal for now)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

# Install package
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser
USER appuser

EXPOSE 10300

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["voxtral-wyoming"]
