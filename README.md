# Kokoro TTS

A local implementation of the Kokoro Text-to-Speech system, based on the [Kokoro-82M model](https://huggingface.co/hexgrad/Kokoro-82M).

## Features

- High-quality English text-to-speech synthesis
- Multiple voice styles
- Adjustable speech speed
- Local inference without internet dependency (after initial model download)
- Automatic espeak-ng installation using [espeakng-loader](https://github.com/thewh1teagle/espeakng-loader)

## Prerequisites

- Python 3.8 or higher
- Git LFS (for model download)

## Installation

1. Set up Python environment:
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

The espeakng-loader package will automatically handle the installation of espeak-ng.

## Project Structure

- `tts_demo.py` - Demo script showing basic usage
- `models.py` - Model implementation and utilities
- `requirements.txt` - Python dependencies

## Usage

Run the demo script:
```bash
python tts_demo.py
```

This will:
1. Automatically install espeak-ng if needed
2. Download the Kokoro model
3. Generate a sample audio file

## Credits

- Original model: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- espeak-ng loader: [espeakng-loader](https://github.com/thewh1teagle/espeakng-loader)
- Based on papers:
  - [arxiv: 2306.07691](https://arxiv.org/abs/2306.07691)
  - [arxiv: 2203.02395](https://arxiv.org/abs/2203.02395)

## License

Apache-2.0 License (following the original model's license) 