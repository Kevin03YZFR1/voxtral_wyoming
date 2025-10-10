# Voxtral Wyoming STT

Offline Speech-to-Text (STT) service using Mistral's Voxtral models with Wyoming protocol compatibility for Home Assistant Assist integration.

## Features

- 🎯 **Offline-only**: Local inference with Mistral's Voxtral model files (no cloud APIs)
- 🔌 **Wyoming Protocol**: Full compatibility with Home Assistant Assist
- 🐳 **Docker Ready**: Containerized deployment with non-root user
- 🎵 **Audio Format Support**: Automatic conversion of MP3, OGG, FLAC, WAV to PCM16 (requires ffmpeg)
- ⚡ **Device Flexibility**: CPU, CUDA (NVIDIA), or MPS (Apple Silicon) support

## Docker Deployment

### Building the Image

```bash
docker build -t voxtral-wyoming:latest .
```

### Running the Container

```bash
# Basic run
docker run --rm -it -p 10300:10300 voxtral-wyoming:latest

# With volume mount for Voxtral model
docker run --rm -it \
  -p 10300:10300 \
  -v /path/to/voxtral/models:/models:ro \
  -e MODEL_ID=/models/Voxtral-Mini-3B-2507 \
  voxtral-wyoming:latest

# With GPU support (NVIDIA)
docker run --rm -it --gpus all \
  -p 10300:10300 \
  -v /path/to/voxtral/models:/models:ro \
  -e MODEL_ID=/models/Voxtral-Mini-3B-2507 \
  -e DEVICE=cuda \
  voxtral-wyoming:latest
```

### Adding ffmpeg for Audio Format Support

To enable server-side audio format conversion, add ffmpeg to your Dockerfile:

```dockerfile
# Add this line after the base image declaration
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
```

## Home Assistant Integration

1. Start the Voxtral Wyoming server on your network (or in Docker)
2. In Home Assistant, go to **Settings** → **Voice Assistants** → **Add Voice Assistant**
3. Select **Wyoming Protocol**
4. Enter the server host and port (e.g., `192.168.1.100:10300`)
5. Select language and audio settings
6. Test with Home Assistant Assist

The Wyoming protocol is fully implemented and compatible with Home Assistant's Assist feature.

## Configuration

Configuration can be set via environment variables or CLI options:

### Server Configuration
- `HOST` (default: 0.0.0.0) - Bind host
- `PORT` (default: 10300) - Bind port
- `LANGUAGE` (default: en-US) - Language/locale hint
- `MODEL_ID` ID Voxtral model to use: "mistralai/Voxtral-Mini-3B-2507" (default) or "mistralai/Voxtral-Small-24B-2507" (or other compatible variant from Hugging Face)
- `DEVICE` (default: cuda) - Device: cpu|cuda|mps (automatically falls back to CPU if device fails)
- `DATA_TYPE` (default: fp32) - Data type: fp32|fp16|bf16 (CPU forces fp32)
- `LOG_LEVEL` (default: INFO) - Logging level
- `MAX_SECONDS` (default: 60) - Maximum audio duration in seconds
- `SAMPLE_RATE` (default: 16000) - Expected audio sample rate in Hz
- `MAX_NEW_TOKENS` (default: 128) - Maximum generation length

## Development

### Installation (Development)

```bash
uv venv
source .venv/bin/activate
uv sync
```

Requires Python and uv to be installed. Optionally, ffmpeg for audio conversion.

### Running the Server (Development)

```bash
voxtral-wyoming --host 0.0.0.0 --port 10300 --language en-US

# Available options:
#   --host HOST           Bind host (default: 0.0.0.0)
#   --port PORT           Bind port (default: 10300)
#   --language LANG       Language/locale hint (default: en-US)
#   --sample-rate RATE    Expected audio sample rate in Hz (default: 16000)
#   --max-seconds SEC     Clamp incoming audio to max seconds (default: 60)
#   --log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
```

### Testing with Sample Audio

Use the example client to test transcription:

```bash
# Test with a sample audio file from HuggingFace
python examples/client_sample.py \
  --host 127.0.0.1 \
  --port 10300 \
  --url https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3

# Or test with a local audio file
python examples/client_sample.py \
  --host 127.0.0.1 \
  --port 10300 \
  --file /path/to/audio.wav

# Options:
#   --host HOST          Server host (default: 127.0.0.1)
#   --port PORT          Server port (default: 10300)
#   --url URL            Download and transcribe audio from URL
#   --file FILE          Transcribe local audio file
#   --language LANG      Language hint (default: en-US)
#   --no-convert         Skip ffmpeg conversion (send raw bytes)
```

The client will automatically attempt to convert audio to PCM16 mono 16 kHz using ffmpeg if available.


### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=voxtral_wyoming --cov-report=term-missing
```

## Troubleshooting

### Model Not Found

If you see "model not found" errors:
1. Ensure `MODEL_ID` points to a valid model id from Hugging Face
2. Or ensure the model is in your HuggingFace cache (run `huggingface-cli download mistralai/Voxtral-Mini-3B-2507`)

### Audio Format Errors

If you see audio format errors:
1. Install ffmpeg: `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)
2. Or convert audio to PCM16 manually before sending:
   ```bash
   ffmpeg -i input.mp3 -f s16le -acodec pcm_s16le -ac 1 -ar 16000 output.pcm
   ```

### GPU Not Working

If CUDA/GPU is not detected:
1. Ensure NVIDIA drivers and CUDA toolkit are installed
2. Ensure PyTorch with CUDA support is installed: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
3. Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. The server will automatically fall back to CPU if GPU initialization fails

### Connection Issues with Home Assistant

If Home Assistant can't connect:
1. Ensure the server is running and listening on the correct host/port
2. Check firewall settings (port 10300 must be accessible)
3. Verify network connectivity between Home Assistant and the server
4. Check server logs for connection attempts

## Contributing

Contributions are welcome!
