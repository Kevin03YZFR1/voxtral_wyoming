# Voxtral Wyoming STT (Early)

Offline Speech-to-Text (STT) service intended to run Mistral's Voxtral models and expose a Wyoming-compatible interface for Home Assistant Assist. This version provides a runnable server with a stub protocol and transcriber; the full Wyoming protocol and Voxtral backend integration will follow.

## Status
- Phase 1 (this release): runnable skeleton with a stub transcriber and a simple TCP server; CLI supports selecting `--protocol wyoming|stub` (wyoming currently falls back to stub).
- Next steps: implement Wyoming protocol handling using the `wyoming` Python package and integrate a local Voxtral model backend.

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
