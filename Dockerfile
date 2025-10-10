# Minimal runtime image for Voxtral Wyoming STT (stub)
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WYOMING_HOST=0.0.0.0 \
    WYOMING_PORT=10300 \
    VOXTRAL_LANGUAGE=en-US \
    AUDIO_SAMPLE_RATE=16000 \
    LOG_LEVEL=INFO

# Install system dependencies and uv (global)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && uv --version

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

# Create non-root user and take ownership before creating venv
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser && chown -R 10001:10001 /app
USER appuser

# Create and populate uv-managed virtual environment as non-root
RUN uv sync --no-dev

# Ensure virtualenv is on PATH for runtime
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 10300

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["voxtral-wyoming"]
