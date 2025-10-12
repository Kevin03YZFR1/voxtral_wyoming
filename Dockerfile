FROM python:3.13-slim AS runtime

# Install system dependencies and uv (global)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv \
    && uv --version

WORKDIR /app

# Create non-root user early
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser

# Create HuggingFace cache directory with proper ownership for volume mount
RUN mkdir -p /home/appuser/.cache/huggingface && chown -R 10001:10001 /home/appuser/.cache

# Create output directory for audio files with proper ownership
RUN mkdir -p /output/audio && chown -R 10001:10001 /output

# Copy dependency files first (for better layer caching)
COPY pyproject.toml uv.lock /app/
RUN chown -R 10001:10001 /app

USER appuser

# Install dependencies only (this layer will be cached unless dependency files change)
RUN uv sync --no-dev --no-install-project

# Copy project files needed for package installation (changes here won't invalidate the dependency installation layer)
COPY --chown=10001:10001 README.md LICENSE /app/
COPY --chown=10001:10001 src /app/src

# Install the project itself (fast since dependencies are already installed)
RUN uv sync --no-dev

# Ensure virtualenv is on PATH for runtime
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["voxtral-wyoming"]
