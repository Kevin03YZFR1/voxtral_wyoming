# Voxtral Wyoming STT (Stub)

Offline Speech-to-Text (STT) service intended to run Mistral's Voxtral models and expose a Wyoming-compatible interface for Home Assistant Assist. This initial version provides a runnable server stub and project scaffolding. The stub accepts a TCP connection and returns a fixed transcript; the full Wyoming protocol and Voxtral backend integration will follow.

## Status
- Phase 1 (this release): runnable skeleton with a stub transcriber and a simple TCP server.
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
voxtral-wyoming --host 0.0.0.0 --port 10300 --language en-US --sample-rate 16000 --log-level INFO
```

The stub server listens for a TCP connection. It reads bytes until the client closes the socket and then returns a one-line JSON response containing a fixed transcript.

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
- LOG_LEVEL (default: INFO)

## Docker
Build the image:
```bash
docker build -t voxtral-wyoming:stub .
```

Run the container:
```bash
docker run --rm -it -p 10300:10300 --name voxtral-wyoming voxtral-wyoming:stub
```

## Home Assistant (Wyoming)
The current version does not yet speak the Wyoming protocol. Once Wyoming support is added, you will be able to add this service in Home Assistant via Settings → Voice Assistants → Add Wyoming service, pointing to the host and port configured above.

## Development
- Code style: black + ruff (config in pyproject.toml)
- Tests: pytest (minimal tests included)

## License
MIT (see LICENSE)
