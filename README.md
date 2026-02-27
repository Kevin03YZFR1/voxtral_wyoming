# Voxtral Wyoming STT

Offline Speech-to-Text (STT) service using [Mistral's Voxtral](https://mistral.ai/news/voxtral) model with [Wyoming protocol](https://github.com/OHF-Voice/wyoming) compatibility for Home Assistant Assist integration.

The goal is to provide a powerful drop-in alternative to the popular Whisper STT option. Especially for non-English languages, Voxtral will hopefully set the new state of the art.

## Features

- 🎯 **Offline-only**: Local inference with Mistral's Voxtral model files (no cloud APIs)
- 🔌 **Wyoming Protocol**: Full compatibility with Home Assistant Assist
- 🐳 **Docker Ready**: Containerized deployment with non-root user
- ⚡ **Device Flexibility**: CPU, CUDA (NVIDIA), or MPS (Apple Silicon) support
- 💬 **Dual Mode Support**: Choose between optimized transcribe-only mode or chat mode with custom system prompts for domain-specific context
- 🎵 **Audio Format Support**: Automatic conversion of MP3, OGG, FLAC, WAV to PCM16 (requires ffmpeg)

## Docker Compose Deployment (Recommended)

For easier deployment and configuration management, use Docker Compose:

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` to configure your setup:**
   ```bash
   # Edit configuration values as needed
   vim .env
   ```

See the short configuration overview below or checkout the `.env.example` for detailed documentation of all options.

3. **Start the service:**
   ```bash
   # Build and start in detached mode
   docker compose up --build -d

   # View logs
   docker compose logs -f
   ```

**GPU Support:**

To enable NVIDIA GPU support, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then set `DEVICE=cuda` in your `.env` file (or leave it as `DEVICE=auto` for automatic detection).

**Local Model Files:**

If you have pre-downloaded Voxtral models, uncomment the model volume mount in `docker-compose.yml`:

```yaml
volumes:
  - ./models:/models:ro
```

Then set `MODEL_ID=/models/Voxtral-Mini-3B-2507` in your `.env` file.

**Audio Saving:**

To save all received audio input as WAV files (one per transcription request), set `SAVE_AUDIO=true` in your `.env` file. The audio files will be saved to the directory specified by `AUDIO_SAVE_DIR` (default: `./output/audio/`).

The docker-compose.yml file includes a bind mount for the audio directory:

```yaml
volumes:
  - ./output/audio:/output/audio
```

Audio files are automatically saved to `./output/audio/` on your host machine with timestamp-based filenames that include the first 100 characters of the transcribed text (e.g., `audio_20251011_203145_123456_Hello_world_this_is_a_test.wav`). Special characters in the transcription are replaced with underscores for filesystem safety.

**⚠️ Warning:** Audio files may contain sensitive information. Ensure proper access controls are in place when enabling this feature.

## Docker Deployment (Alternative without Docker Compose)

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

## Home Assistant Integration

First of all, make sure that you've started the Voxtral Wyoming server as described above.

You don't need to install any HA addon, but just configure a new Wyoming integration:
1. In Home Assistant, go to **Settings** → **Devices & services** → **Add integration**
2. Select **Wyoming Protocol**
3. Enter the server host and port as configured during your server setup and confirm

Now you can choose `voxtral-wyoming` as the Speech-to-text option within any of your configured Assistants on **Settings** → **Voice assistants**.

## Configuration

All configuration is done via environment variables. The easiest way is to use a `.env` file:

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` to configure your setup:**
   ```bash
   vim .env
   ```

### Key Configuration Options

- `HOST` (default: 0.0.0.0) - Bind host
- `PORT` (default: 10300) - Bind port
- `MODEL_ID` ID Voxtral model to use
  - "mistralai/Voxtral-Mini-3B-2507" (default)
  - or "mistralai/Voxtral-Small-24B-2507"
  - or [other compatible variant from Hugging Face](https://huggingface.co/models?other=voxtral)
      - [e.g. a quantized one](https://huggingface.co/models?other=base_model:quantized:mistralai/Voxtral-Mini-3B-2507)
- `DEVICE` (default: auto) - Device selection:
  - `auto`: Automatically detect best device (cuda for NVIDIA GPU, mps for Apple Silicon, or cpu as fallback) - RECOMMENDED
  - `cuda`: Force NVIDIA GPU usage
  - `mps`: Force Apple Silicon GPU usage
  - `cpu`: Force CPU usage
  - Note: Automatically falls back to CPU if manually specified device is unavailable
- `DATA_TYPE` (default: auto-detect) - Data type override (optional, leave unset for auto-detection)
  - Override to bf16 or fp16 when loading fp32 models on GPU for memory savings
- `LOG_LEVEL` (default: INFO) - Logging level
- `MAX_SECONDS` (default: 60) - Maximum audio duration in seconds
- `MAX_NEW_TOKENS` (default: 128) - Maximum generation length
- `USE_CHAT_MODE` (default: false) - Enable chat mode with system prompts instead of transcribe-only mode
  - **false**: Optimized transcribe-only mode (faster, recommended for most users)
  - **true**: Chat mode with system prompt support (allows domain-specific context guidance)
- `SYSTEM_PROMPT` (default: smart home context) - System prompt for chat mode (only used when `USE_CHAT_MODE=true`). Customize to provide context about smart home commands, domain-specific vocabulary, or command structure expectations.
- `SAVE_AUDIO` (default: false) - Save all received audio input as WAV files (one per request)
- `AUDIO_SAVE_DIR` (default: ./output/audio/) - Directory where audio files will be saved
- `LANGUAGE_FALLBACK` (default: en-US) - Fallback language/locale hint. Will get overridden by the configuration of your Home Assistant Voice Assistant.
- `SAMPLE_RATE_FALLBACK` (default: 16000) - Expected audio sample rate in Hz. Again just a fallback value which will get replaced by the information which Home Assistant provides through the Wyoming protocol.

See `.env.example` for detailed documentation of all options.

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
# Use default .env file
voxtral-wyoming

# Or specify a custom environment file
voxtral-wyoming /path/to/custom.env

# Or use environment variables directly
HOST=0.0.0.0 PORT=10300 LANGUAGE_FALLBACK=en-US voxtral-wyoming
```

All configuration is done via environment variables loaded from a `.env` file (default) or a custom environment file specified as the only command-line argument.

### Testing with Sample Audio

Use the example client to test transcription:

```bash
# Use default .env file
python examples/client_sample.py

# Or specify a custom environment file
python examples/client_sample.py /path/to/custom.env

# Or use environment variables directly
HOST=127.0.0.1 PORT=10300 SAMPLE_URL=https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3 python examples/client_sample.py
```

The client will automatically attempt to convert audio to PCM16 mono using ffmpeg if available and enabled.

## Troubleshooting

### voxtral-wyoming exited with code 137
The server process probably got killed by your OS or the docker engine as it was using too many resources. If you
are using the dockerized version, you can try to use the python variant directly. Otherwise, you probably need to
change settings on your local setup.

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

### Adding ffmpeg for Audio Format Support (Optional)

To enable server-side audio format conversion, add ffmpeg to your Dockerfile:

```dockerfile
# Add this line after the base image declaration
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
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

## Hardware & Performance
Running Voxtral is relatively hardware-intensive. It is highly recommended to use a GPU with at least 10GB VRAM for optimal performance. It also does not need to be the latest top model, though.

Personally, I'm currently using a RTX 3090. Having reduced memory requirements by setting `DATA_TYPE=bf16`, most STT requests are handled in ~0.5s while using ~9GB VRAM.

Check out the `DATA_TYPE` parameter, if you're having memory troubles.

I can't really recommend running the model on CPU only. Anyway, if you want to give it a shot, I suggest using [one of the quantized models](https://huggingface.co/models?other=base_model:quantized:mistralai/Voxtral-Mini-3B-2507) for the `MODEL_ID` option to further reduce required resources.


## Online Alternative
If you do not want to host Voxtral on your own, but rather use Mistral's online API, [ha-openai-whisper-stt-api is a nice HA addon provided by fabio-garavini](https://github.com/fabio-garavini/ha-openai-whisper-stt-api).

## Contributing
Contributions are welcome!
