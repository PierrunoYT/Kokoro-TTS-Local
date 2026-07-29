# Kokoro TTS Local

A local implementation of the Kokoro Text-to-Speech model, featuring dynamic module loading, automatic dependency management, and a web interface.

## Features

- Local text-to-speech synthesis using the Kokoro-82M model
- Multiple voice support with easy voice selection (54 voices available across 9 languages)
- Automatic model and voice downloading from Hugging Face
- **Offline mode support** - Run completely offline after initial setup
- Phoneme output support and visualization
- Interactive CLI and web interface
- Voice listing functionality
- Cross-platform support (Windows, Linux, macOS)
- Real-time generation progress display
- Multiple output formats (WAV, MP3, AAC)
- Enhanced security and code quality features
- Centralized configuration management
- Comprehensive dependency validation
- Memory management and optimization
- Thread-safe operations for multi-user scenarios

## Prerequisites

- Python 3.10–3.12 (**Python 3.13+ is not supported** — core dependencies such as `misaki` and `numpy<2.0` do not provide packages for 3.13 yet)
- FFmpeg (optional, for MP3/AAC conversion)
- CUDA-compatible GPU (optional, for faster generation)
- Git (for version control and package management)

## Installation

1. Clone the repository and create a Python virtual environment (use Python 3.10–3.12):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate
# If your default python is 3.13+, use a supported version instead, e.g.:
# py -3.12 -m venv venv

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
# If your default python3 is 3.13+, use e.g.: python3.12 -m venv venv
```

2. Install the project and its dependencies:
```bash
pip install -e .
```

3. (Optional) For Japanese voices (`jf_*`/`jm_*`), download the UniDic dictionary data (~1 GB, one-time). Without it, Japanese G2P fails:
```bash
python -m unidic download
```

This adds the following console commands to your virtual environment:
- `kokoro-tts` — command-line interface (`python -m kokoro_tts_local.tts_demo`)
- `kokoro-tts-web` — Gradio web interface (`python -m kokoro_tts_local.gradio_interface`)
- `kokoro-tts-chinese` — Mandarin CLI (`python -m kokoro_tts_local.chinese_tts_demo`)
- `kokoro-tts-setup` — Mandarin model and voice setup
- `kokoro-tts-check` — dependency diagnostics

**Alternative Installation (Simplified):**
For a simpler setup, you can also install the official Kokoro package directly:
```bash
pip install kokoro soundfile
apt-get install espeak-ng  # On Linux
# or brew install espeak  # On macOS
```

4. (Optional) For GPU acceleration, install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.8 (for RTX 50-series cards)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

You can verify CUDA support is enabled with:
```python
import torch
print(torch.cuda.is_available())  # Should print True if CUDA is available
```

The system will automatically download required models and voice files on first run.

## Docker Quick Start

This project can be run in a CPU-first Docker setup with runtime model and voice downloads.

### Build and Run with Docker

**Linux/macOS (bash/zsh):**
```bash
docker build -t kokoro-tts-local:cpu .
docker run --rm -it \
   -p 127.0.0.1:7860:7860 \
   -e KOKORO_TTS_USERNAME=admin -e KOKORO_TTS_PASSWORD=change-me \
   -v kokoro-data:/data \
   kokoro-tts-local:cpu
```

**Windows (PowerShell):**
```powershell
docker build -t kokoro-tts-local:cpu .
docker run --rm -it `
   -p 127.0.0.1:7860:7860 `
   -e KOKORO_TTS_USERNAME=admin -e KOKORO_TTS_PASSWORD=change-me `
   -v kokoro-data:/data `
   kokoro-tts-local:cpu
```

Open `http://localhost:7860` in your browser.

### Run with Docker Compose

```bash
export KOKORO_TTS_USERNAME=admin KOKORO_TTS_PASSWORD=change-me
docker compose up --build
```

### Docker Notes

- First startup can take longer because model and voice files are downloaded from Hugging Face.
- The `kokoro-data` volume persists checkpoints, configs, voices, presets, generated audio, and the Hugging Face cache under `/data`.
- The Docker image pre-installs `en_core_web_sm` during build to avoid non-root runtime initialization errors.
- This initial Docker support is CPU-first. GPU and pre-baked model image variants are intentionally out of scope for this first implementation.
- To force offline mode after assets are downloaded, set `HF_HUB_OFFLINE=1` in your Docker environment.

## Offline Mode

After the initial setup, you can run Kokoro-TTS-Local completely offline without an internet connection.

### Quick Start - Offline Mode

**Linux/macOS:**
```bash
export HF_HUB_OFFLINE=1
kokoro-tts
```

**Windows (PowerShell):**
```powershell
$env:HF_HUB_OFFLINE="1"
kokoro-tts
```

**Windows (Command Prompt):**
```cmd
set HF_HUB_OFFLINE=1
kokoro-tts
```

### Requirements for Offline Mode

Before enabling offline mode, ensure you have:
1. Run the application at least once with internet connection
2. Downloaded the model file (`kokoro-v1_0.pth`)
3. Downloaded the config file (`config.json`)
4. Downloaded at least one voice file in the `voices/` directory

### Testing Offline Mode

Use the provided test script to verify your offline setup:

```bash
export HF_HUB_OFFLINE=1  # Enable offline mode
python test_offline.py   # Run the test
```

The script checks:
- Offline mode environment variables are set
- Required files exist (`kokoro-v1_0.pth`, `config.json`, `voices/`)
- All required Python packages are installed
- Model initializes correctly
- Voices can be listed
- Speech can be generated and saved

For detailed offline usage instructions, set `HF_HUB_OFFLINE=1` before running and use `test_offline.py` to verify your setup.

## Configuring File Locations

By default, data is stored in a stable per-user `kokoro-tts-local` directory: `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, and `$XDG_DATA_HOME` or `~/.local/share` on Unix. It does not depend on the current working directory.

| Environment Variable | Description | Default |
|---|---|---|
| `KOKORO_BASE_DIR` | Base directory for models, config, voices, outputs, and presets. | Platform user data directory |
| `KOKORO_MODEL_DIR` | Directory to look for/download the model (`.pth`) and `config.json` files. | `KOKORO_BASE_DIR` |
| `KOKORO_VOICES_DIR` | Directory to look for/download voice (`.pt`) files. | `KOKORO_BASE_DIR/voices` |
| `KOKORO_CONFIG_PATH` | Full path to `config.json`. | `KOKORO_MODEL_DIR/config.json` |

**Linux/macOS:**
```bash
export KOKORO_BASE_DIR="$HOME/.local/share/kokoro-tts-local"
kokoro-tts
```

**Windows (PowerShell):**
```powershell
$env:KOKORO_BASE_DIR = "$HOME\.kokoro-tts"
kokoro-tts
```

Explicit `model_path` arguments passed to `build_model()` are still resolved relative to the current working directory, so existing scripts continue to work unchanged.

## Usage

You can use either the command-line interface or the web interface:

### Command Line Interface

Run the interactive CLI:
```bash
kokoro-tts
```

The CLI provides an interactive menu with the following options:
1. List available voices - Shows all available voice options
2. Generate speech - Interactive process to:
   - Select a voice from the numbered list
   - Enter text to convert to speech
   - Adjust speech speed (0.5-2.0)
3. Exit - Quit the program

Example session:
```
=== Kokoro TTS Menu ===
1. List available voices
2. Generate speech
3. Exit
Select an option (1-3): 2

Available voices:
1. af_alloy
2. af_aoede
3. af_bella
...

Select a voice number (or press Enter for default 'af_bella'): 3

Enter the text you want to convert to speech
(or press Enter for default text)
> Hello, world!

Enter speech speed (0.1-3.0, default 1.0): 1.2

Generating speech for: 'Hello, world!'
Using voice: af_bella
Speed: 1.2x
...
```

### Web Interface

For a more user-friendly experience, launch the web interface:

```bash
kokoro-tts-web
```

Then open your browser to the URL shown in the console (typically http://localhost:7860).

The web interface provides:
- Easy voice selection from a dropdown menu
- Text input field with examples
- Speed control slider (0.5–2.0x)
- Output format selection (WAV, MP3, AAC)
- Real-time generation progress
- Audio playback in the browser
- Download options for generated audio
- **Speed Dial presets** — save, load, and delete frequently used voice/text/speed combinations

### Dependency Validation

Before running the application, you can validate your system setup:

```bash
kokoro-tts-check
```

This will check:
- Python version compatibility
- All required dependencies and their versions
- CUDA availability and GPU detection
- System memory and disk space
- Audio system functionality

### Configuration Management

Application data locations can be inspected with the package path helpers:

```python
from kokoro_tts_local.models import get_base_dir, get_model_dir, get_voices_dir

data_dir = get_base_dir()
models_dir = get_model_dir()
voices_dir = get_voices_dir()
```

All three paths honor the `KOKORO_*` overrides described above.

## Available Voices

The system includes 54 different voices across 9 languages:

### 🇺🇸 American English (20 voices)
**Language code: 'a'**

**Female voices (af_*):**
- af_heart: ❤️ Premium quality voice (Grade A)
- af_alloy: Clear and professional (Grade C)
- af_aoede: Smooth and melodic (Grade C+)
- af_bella: 🔥 Warm and friendly (Grade A-)
- af_jessica: Natural and engaging (Grade D)
- af_kore: Bright and energetic (Grade C+)
- af_nicole: 🎧 Professional and articulate (Grade B-)
- af_nova: Modern and dynamic (Grade C)
- af_river: Soft and flowing (Grade D)
- af_sarah: Casual and approachable (Grade C+)
- af_sky: Light and airy (Grade C-)

**Male voices (am_*):**
- am_adam: Strong and confident (Grade F+)
- am_echo: Resonant and clear (Grade D)
- am_eric: Professional and authoritative (Grade D)
- am_fenrir: Deep and powerful (Grade C+)
- am_liam: Friendly and conversational (Grade D)
- am_michael: Warm and trustworthy (Grade C+)
- am_onyx: Rich and sophisticated (Grade D)
- am_puck: Playful and energetic (Grade C+)
- am_santa: Holiday-themed voice (Grade D-)

### 🇬🇧 British English (8 voices)
**Language code: 'b'**

**Female voices (bf_*):**
- bf_alice: Refined and elegant (Grade D)
- bf_emma: Warm and professional (Grade B-)
- bf_isabella: Sophisticated and clear (Grade C)
- bf_lily: Sweet and gentle (Grade D)

**Male voices (bm_*):**
- bm_daniel: Polished and professional (Grade D)
- bm_fable: Storytelling and engaging (Grade C)
- bm_george: Classic British accent (Grade C)
- bm_lewis: Modern British accent (Grade D+)

### 🇯🇵 Japanese (5 voices)
**Language code: 'j'**

**Female voices (jf_*):**
- jf_alpha: Standard Japanese female (Grade C+)
- jf_gongitsune: Based on classic tale (Grade C)
- jf_nezumi: Mouse bride tale voice (Grade C-)
- jf_tebukuro: Glove story voice (Grade C)

**Male voices (jm_*):**
- jm_kumo: Spider thread tale voice (Grade C-)

### 🇨🇳 Mandarin Chinese (8 voices)
**Language code: 'z'**

**Female voices (zf_*):**
- zf_xiaobei: Chinese female voice (Grade D)
- zf_xiaoni: Chinese female voice (Grade D)
- zf_xiaoxiao: Chinese female voice (Grade D)
- zf_xiaoyi: Chinese female voice (Grade D)

**Male voices (zm_*):**
- zm_yunjian: Chinese male voice (Grade D)
- zm_yunxi: Chinese male voice (Grade D)
- zm_yunxia: Chinese male voice (Grade D)
- zm_yunyang: Chinese male voice (Grade D)

**Note:** Run `python -m kokoro_tts_local.setup_chinese_tts` to download the Chinese model and voice files automatically. For full usage details see [CHINESE_TTS_GUIDE.md](CHINESE_TTS_GUIDE.md) or [README_CHINESE_TTS.md](README_CHINESE_TTS.md).

### 🇪🇸 Spanish (3 voices)
**Language code: 'e'**

**Female voices (ef_*):**
- ef_dora: Spanish female voice

**Male voices (em_*):**
- em_alex: Spanish male voice
- em_santa: Spanish holiday voice

### 🇫🇷 French (1 voice)
**Language code: 'f'**

**Female voices (ff_*):**
- ff_siwis: French female voice (Grade B-)

### 🇮🇳 Hindi (4 voices)
**Language code: 'h'**

**Female voices (hf_*):**
- hf_alpha: Hindi female voice (Grade C)
- hf_beta: Hindi female voice (Grade C)

**Male voices (hm_*):**
- hm_omega: Hindi male voice (Grade C)
- hm_psi: Hindi male voice (Grade C)

### 🇮🇹 Italian (2 voices)
**Language code: 'i'**

**Female voices (if_*):**
- if_sara: Italian female voice (Grade C)

**Male voices (im_*):**
- im_nicola: Italian male voice (Grade C)

### 🇧🇷 Brazilian Portuguese (3 voices)
**Language code: 'p'**

**Female voices (pf_*):**
- pf_dora: Portuguese female voice

**Male voices (pm_*):**
- pm_alex: Portuguese male voice
- pm_santa: Portuguese holiday voice

**Note:** Quality grades (A to F) indicate the overall quality based on training data quality and duration. Higher grades generally produce better speech quality.

## Project Structure

```
.
├── src/kokoro_tts_local/
│   ├── models.py             # Model registry, downloads, and inference
│   ├── gradio_interface.py   # Authenticated web interface
│   ├── tts_demo.py           # General CLI
│   ├── chinese_tts_demo.py   # Mandarin-focused CLI
│   ├── chinese_config.py     # Mandarin text and voice metadata
│   ├── setup_chinese_tts.py  # Mandarin asset setup
│   ├── speed_dial.py         # Durable preset storage
│   └── paths.py              # Platform-aware data paths
├── .github/workflows/        # CI and assistant workflows
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── test_critical_fixes.py
├── test_offline.py
├── LICENSE
└── requirements.txt
```

## Model Information

The project uses the latest Kokoro model from Hugging Face:
- Repository: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Model file: `kokoro-v1_0.pth` (downloaded automatically)
- Sample rate: 24kHz
- Voice files: Located in the `voices/` directory (downloaded automatically)
- Available voices: 54 voices across 9 languages
- Languages: American English ('a'), British English ('b'), Japanese ('j'), Mandarin Chinese ('z'), Spanish ('e'), French ('f'), Hindi ('h'), Italian ('i'), Brazilian Portuguese ('p')
- Model size: 82M parameters

## Troubleshooting

Common issues and solutions:

### Quick System Check

First, run the dependency checker to identify potential issues:
```bash
kokoro-tts-check
```

This will automatically detect and report:
- Missing or incompatible dependencies
- CUDA/GPU configuration issues
- System resource problems
- Audio system issues

### Common Issues

1. **Installation Fails with `No matching distribution found for mishkal-hebrew`**
   - **Problem:** installation aborts with `Could not find a version that satisfies the requirement mishkal-hebrew>=0.3.2 (from versions: none)` on any Python version, or `kokoro-tts` later fails with `ModuleNotFoundError: No module named 'kokoro'`
   - **Cause:** All `mishkal-hebrew` releases were deleted from PyPI (the project was renamed to `phonikud`). The aborted install means none of the other dependencies got installed either.
   - **Solution:** Pull the latest version of this repo (`mishkal-hebrew` has been removed) and re-run `pip install -e .`

2. **Installation Fails on Python 3.13+**
   - **Problem:** `pip install -e .` fails with numpy meson/ninja build errors or `Ignored the following versions that require a different python version` messages, or later `ModuleNotFoundError: No module named 'kokoro'`
   - **Cause:** Several core dependencies (`misaki`, `numpy<2.0`) do not support Python 3.13 yet
   - **Solution:** Create the virtual environment with Python 3.10–3.12, e.g. on Windows: `py -3.12 -m venv venv`

3. **Japanese Voices Produce Wrong/Garbled Speech**
   - **Problem:** Selecting a Japanese voice (e.g. `jf_alpha`) produces nonsense audio, or Japanese G2P fails with a dictionary error
   - **Cause 1:** `fugashi`/`unidic` ship without dictionary data; it must be downloaded separately
   - **Solution:** Run `python -m unidic download` (~1 GB, one-time) in your virtual environment
   - **Cause 2:** Language routing — if you have already run `unidic download` but the web UI still produces garbled audio, the text may be running through the English G2P pipeline. This was a bug in `gradio_interface.py` (fixed) where the voice's language was misdetected; make sure you are on an up-to-date version.

4. **Offline Mode / Network Connection Issues**
   - **Problem:** Getting "Failed to resolve 'huggingface.co'" errors even with cached files
   - **Solution:** Enable offline mode with `export HF_HUB_OFFLINE=1` (Linux/macOS) or `$env:HF_HUB_OFFLINE="1"` (Windows)
   - **Verify:** Run `python test_offline.py` to confirm your offline setup is working

5. **Model Download Issues**
   - Ensure stable internet connection
   - Check Hugging Face is accessible
   - Verify sufficient disk space
   - Try clearing the `.cache/huggingface` directory

6. **CUDA/GPU Issues**
   - Verify CUDA installation with `nvidia-smi`
   - Update GPU drivers
   - Install PyTorch with CUDA support using the appropriate command:
     ```bash
     # For CUDA 11.8
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

     # For CUDA 12.1
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

     # For CUDA 12.6
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

     # For CUDA 12.8 (for RTX 50-series cards)
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
     ```
   - Verify CUDA is available in PyTorch:
     ```python
     import torch
     print(torch.cuda.is_available())  # Should print True
     ```
   - Fall back to CPU if needed

7. **Audio Output Issues**
   - Check system audio settings
   - Verify output directory permissions
   - Install FFmpeg for MP3/AAC support
   - Try different output formats

8. **Voice File Issues**
   - Delete and let system redownload voice files
   - Check `voices/` directory permissions
   - Verify voice file integrity
   - Try using a different voice

9. **Web Interface Issues**
   - Check port 7860 availability
   - Try different browser
   - Clear browser cache
   - Check network firewall settings

For any other issues:
1. Check the console output for error messages
2. Verify all prerequisites are installed
3. Ensure virtual environment is activated
4. Check system resource usage
5. Try reinstalling dependencies

## Contributing

Feel free to contribute by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Helping with documentation
4. Testing different voices and reporting issues
5. Suggesting new features or optimizations
6. Testing on different platforms and reporting results

## License

Apache 2.0 - See LICENSE file for details
