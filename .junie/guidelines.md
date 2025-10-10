# Voxtral Wyoming STT — Project Guidelines

These guidelines capture the foundational context, constraints, and conventions for this repository. They are intended to keep all contributors aligned as we build an offline Speech-to-Text (STT) service that uses Mistral’s Voxtral models and speaks the Wyoming protocol for seamless Home Assistant (HA) integration.

## TL;DR
- Build a local-only STT service using Mistral Voxtral.
- Expose it via the Wyoming protocol so Home Assistant can use it for Assist (no HA addon required).
- Ship as Docker (compose if multiple services are needed). No cloud APIs.

## Goals and Non‑Goals

### Goals
- Offline-only STT based on Mistral’s Voxtral models (no remote API calls).
- Wyoming protocol compatibility to integrate with Home Assistant (HA Assist).
- Containerized deployment (Docker); support Docker Compose if multiple services emerge.
- Cross-platform support where practical (x86_64 and arm64; CPU first, optional GPU acceleration).
- Clean, maintainable Python code with tests and clear documentation.

### Non‑Goals (for now)
- No cloud inference or dependencies on SaaS.
- No Home Assistant add-on packaging; the Wyoming protocol integration should be sufficient.
- No TTS or wake word (only STT); those could be separate projects/services.

## High-Level Architecture
- Wyoming STT server: Listens on a TCP port and implements the Wyoming protocol for STT.
- Audio pre-processing: Resampling/format conversion, optional VAD.
- Voxtral transcription backend: Loads local Voxtral model weights and performs inference.
- Result formatting: Returns transcription text and metadata via Wyoming.

We will keep the backend swappable via an internal interface (e.g., ITranscriber) so model runners can evolve without touching the protocol layer.

## Wyoming Protocol (for STT)
- We will implement a Wyoming service of type "stt" using the community Python package `wyoming` when feasible (commonly used in HA-compatible projects).
- Default port: 10300 (configurable via env/config). This mirrors de-facto conventions used by Wyoming-based STT services.
- Audio expectations (initial targets; configurable):
  - PCM 16-bit, mono
  - 16 kHz sample rate (we can add 8/22.05/24/32/48 kHz support by resampling)
  - Linear PCM frames (little-endian)
- Flow: client connects → handshake/metadata → sends audio frames (or stream) → server returns final transcription (and optional interim/partial results if we add streaming).
- Languages: default `en-US`; configurable (we’ll expose a `language` or `locale` setting). Actual Voxtral language coverage to be verified against released models.

References (background):
- Wyoming protocol implementations in Python exist in projects like `wyoming-whisper`. We will follow their server patterns but swap the model with Voxtral.

## Voxtral Model Strategy
Because model distribution and runtimes may evolve, we will keep the approach modular:
- Primary requirement: run Voxtral locally.
- Candidate runtimes:
  - PyTorch + Transformers (if/when Voxtral provides an ASR pipeline)
  - Vendor or community runtime for Voxtral weights (to be confirmed per release notes)
- Device selection: `cpu`, `cuda` (NVIDIA), `mps` (Apple Silicon) when available.
- Mixed precision: configurable (e.g., float32/float16/bfloat16) depending on runtime support.
- Memory/perf knobs: batch size, chunk size, beam size/greedy decoding, VAD settings.

We will NOT use cloud APIs. The container should be able to run fully offline once the model files are present on disk.

## Docker and Compose
- Single-service container by default (STT server). Add Compose if we later split components (e.g., a model downloader or cache service).
- Suggested base: `python:3.11-slim` (or `3.12-slim` once runtimes confirm support).
- Optimize image size: multi-stage builds; only ship runtime deps and model runner.
- GPU support (optional):
  - NVIDIA: document `--gpus all` and CUDA versions if used.
  - Apple Silicon: CPU or Metal (MPS) via PyTorch if supported.
- Security:
  - Run as non-root inside the container.
  - Network egress not required during runtime; avoid contacting external hosts.
- Volumes:
  - Mount a host directory at `/models` (or similar) for Voxtral weights.
  - Optional: mount `/data` for logs/cache.

## Home Assistant Integration (Wyoming)
- HA → Settings → Voice Assistants → Add Wyoming service.
- Provide host/IP of the container and port 10300.
- Select language and audio format consistent with our configuration.
- After connection, HA Assist should be able to send audio and receive transcripts from this service.

## Codebase Conventions
- Language/version: Python 3.11+.
- Packaging: `pyproject.toml` with PEP 621 metadata; src-layout (`src/voxtral_wyoming`).
- Lint/format: `ruff` + `black`. Typing: strict-ish `mypy` where feasible.
- Logging: `logging` stdlib; structured context where helpful. No audio contents in logs by default.
- Tests: `pytest` with unit tests for preprocessing and integration tests for end-to-end transcription (using a short audio sample). Consider smoke tests for container.
- CI (future): run lint, type-check, unit tests, and a minimal container build.

## Initial Repository Layout (proposed)
- `src/voxtral_wyoming/`
  - `__init__.py`
  - `server.py` (Wyoming server and protocol handling)
  - `audio.py` (I/O, resampling, framing)
  - `vad.py` (optional voice activity detection)
  - `transcriber/`
    - `__init__.py`
    - `base.py` (ITranscriber interface)
    - `voxtral.py` (Voxtral-backed implementation)
- `tests/` (unit and integration tests)
- `docker/` (optional helper files)
- `Dockerfile`
- `docker-compose.yml` (if/when needed)
- `README.md` (setup, usage, HA integration)
- `LICENSE` (already present)
- `.gitignore`, `.editorconfig`, `pyproject.toml`

Note: We may start with a minimal runnable skeleton and iterate.

## Observability
- Log levels: DEBUG (verbose for dev), INFO (default), WARN/ERROR.
- Optional metrics endpoint (future) or simple counters printed on shutdown.
- Timing and latency logs for transcription path (without leaking audio content).

## Privacy & Security
- Strictly local inference; no outbound calls from the server.
- Do not log raw audio or full transcripts at INFO level; gate sensitive output behind DEBUG.
- Avoid storing audio by default; any persistence must be explicit and documented.

## Performance Expectations
- Phase 1: functional correctness (batch transcription per request).
- Phase 2: streaming partial results (if supported by the runtime) for better UX.
- Keep CPU-only viable; add GPU acceleration paths where available to reduce latency.

## Roadmap (initial)
1. Project scaffolding and runnable stub service (Wyoming server + placeholder transcriber interface).
2. Integrate Voxtral local model runner; load weights from mounted volume; basic transcription.
3. Improve audio preprocessing (resampling, chunking) and add optional VAD.
4. Dockerize with sensible defaults, non-root user, and documented GPU options.
5. End-to-end tests and container smoke tests; documentation for HA setup.
6. Performance tuning and streaming interim results (if supported by Voxtral runtime).

## Contribution Workflow
- Branch: feature/..., fix/..., chore/...; small, reviewable PRs.
- Include tests and documentation updates with code changes where applicable.
- Keep commits atomic; write clear messages.

## Licensing
- This repo includes a LICENSE file. Ensure third-party dependencies and model licenses are compatible with redistribution and intended use.

## Open Questions / To Validate
- Public availability and licensing of Voxtral model weights for local use.
- Best-supported runtime for Voxtral across Linux/macOS (CPU/GPU) and arm64/x86_64.
- Exact Wyoming feature set we’ll support initially (batch-only vs. streaming partials).

## References (for maintainers)
- Home Assistant Wyoming integrations (e.g., existing Whisper-based servers) can serve as protocol references.
- Mistral model documentation for Voxtral (weights, runtime, and API shape) as they are released/updated.

---
This document is a living guide. Please update it as decisions are made and the implementation evolves.
