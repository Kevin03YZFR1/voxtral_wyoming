# Voxtral Wyoming STT (Early)

Offline Speech-to-Text (STT) service intended to run Mistral's Voxtral models and expose a Wyoming-compatible interface for Home Assistant Assist. This version provides a runnable server with a stub protocol and transcriber; the full Wyoming protocol and Voxtral backend integration will follow.

## Status
- Phase 1 (this release): runnable server with a stub protocol and an optional Voxtral backend. Wyoming protocol option is recognized but still falls back to stub behavior.
- Voxtral backend: implemented for offline local inference with Mistral's Voxtral model files (no Whisper, no cloud calls).
- Next steps: implement full Wyoming protocol handling using the `wyoming` Python package.

## Requirements
- Python 3.11+
- Docker (optional but recommended for deployment)

## Install (dev)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run (dev)
```bash
voxtral-wyoming --host 0.0.0.0 --port 10300 --language en-US --sample-rate 16000 --protocol wyoming --backend dummy --max-seconds 60 --log-level INFO
```

### Quick test: transcribe a sample file
With the server running in one terminal, in another terminal run:
```bash
voxtral-wyoming-sample --host 127.0.0.1 --port 10300 \
  --url https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3
```
This downloads the sample MP3, tries to convert it to PCM16 mono 16 kHz using ffmpeg (if available),
sends it to the server, and prints the transcript from the server's response.

If ffmpeg is not available, the script will send the MP3 bytes as-is (works with the stub server). Use `--no-convert` to explicitly skip conversion.

The server currently falls back to a simple stub behavior: it accepts a TCP connection, reads bytes until the client closes the socket, and returns a one-line JSON response containing a fixed transcript.

Example client:
```bash
# Send some bytes and get a JSON response
echo -n "fake-audio" | nc 127.0.0.1 10300
```

Response:
```json
{"service": "voxtral-wyoming-stub", "protocol": "stub-json", "language": "en-US", "text": "Hello from Voxtral Wyoming stub"}
```

## Configuration
Configuration can be set with CLI options or environment variables.

- WYOMING_HOST (default: 0.0.0.0)
- WYOMING_PORT (default: 10300)
- VOXTRAL_LANGUAGE (default: en-US)
- AUDIO_SAMPLE_RATE (default: 16000)
- WYOMING_PROTOCOL (default: wyoming)
- VOXTRAL_BACKEND (default: dummy)  # dummy|voxtral
- AUDIO_MAX_SECONDS (default: 60)
- LOG_LEVEL (default: INFO)

### Using the Voxtral backend (offline)
The Voxtral backend uses Mistral's Voxtral model locally. No network calls are made at runtime.

Requirements when using `--backend voxtral`:
- Install dependencies: torch, transformers, numpy (not installed by default).
- Have the model files available locally either in the HF cache or at a directory path.

Key environment variables:
- VOXTRAL_MODEL_PATH or VOXTRAL_MODEL_DIR: local path to the Voxtral model directory (recommended), e.g. `/models/Voxtral-Mini-3B-2507`
- VOXTRAL_DEVICE: cpu|cuda|mps (default: cpu)
- VOXTRAL_DTYPE: fp32|fp16|bf16 (default: fp32; CPU forces fp32)
- VOXTRAL_MAX_NEW_TOKENS: default 500

Example run:
```bash
VOXTRAL_BACKEND=voxtral \
VOXTRAL_MODEL_PATH=/models/Voxtral-Mini-3B-2507 \
VOXTRAL_DEVICE=cpu \
voxtral-wyoming --protocol stub --language en-US --sample-rate 16000
```

Notes:
- Loading uses `local_files_only=True`. If you supply a repo ID instead of a path, ensure the model is already present in your local HF cache.
- Audio input is expected as PCM16 mono. The server stub reads raw bytes from the TCP connection; the Wyoming protocol will be added later.

## Docker
Build the image:
```bash
docker build -t voxtral-wyoming:early .
```

Run the container:
```bash
docker run --rm -it -p 10300:10300 --name voxtral-wyoming voxtral-wyoming:early
```

## Home Assistant (Wyoming)
The CLI now recognizes `--protocol wyoming`, but actual Wyoming protocol handling is not implemented yet and will fall back to the stub behavior. Once complete, you will be able to add this service in Home Assistant via Settings → Voice Assistants → Add Wyoming service, pointing to the host and port configured above.

## Development
- Code style: black + ruff (config in pyproject.toml)
- Tests: pytest (minimal tests included)

## License
MIT (see LICENSE)
