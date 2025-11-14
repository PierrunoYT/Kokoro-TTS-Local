# Kokoro Chinese TTS Guide
## 科克罗中文文本转语音指南

Complete guide for setting up and using the Kokoro-82M-v1.1_zh Chinese TTS model locally.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Available Chinese Voices](#available-chinese-voices)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The **Kokoro-82M-v1.1_zh** is a fine-tuned Mandarin Chinese TTS model for high-quality speech synthesis.

### Key Features

- 8 Chinese voices (4 female + 4 male)
- Natural Mandarin pronunciation
- Adjustable speech speed (0.5x - 2.0x)
- Automatic text normalization
- Offline operation after setup
- Cross-platform support

---

## Installation

### Prerequisites

- Python 3.8+
- ~1GB free disk space
- Internet connection (for initial download)

### Automated Setup (Recommended)

```bash
python setup_chinese_tts.py
```

This script automatically downloads the model and all voice files.

### Manual Setup

1. **Download the model:**
   ```bash
   # From Hugging Face
   git clone https://huggingface.co/hexgrad/Kokoro-82M
   # Place kokoro-82M-v1.1_zh.pth in project root
   ```

2. **Download voice files** to `voices/` directory:
   - Female: `zf_xiaobei.pt`, `zf_xiaoni.pt`, `zf_xiaoxiao.pt`, `zf_xiaoyi.pt`
   - Male: `zm_yunjian.pt`, `zm_yunxi.pt`, `zm_yunxia.pt`, `zm_yunyang.pt`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### Interactive CLI

```bash
python chinese_tts_demo.py
```

The interactive menu provides:
1. List available voices
2. Generate speech from custom text
3. Generate from sample texts
4. Help information
5. Exit

### Python API

```python
from chinese_tts_demo import load_chinese_model, generate_chinese_speech, save_audio
import torch

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_chinese_model('kokoro-82M-v1.1_zh.pth', device)

# Generate speech
text = "你好，世界！这是一个测试。"
audio, _ = generate_chinese_speech(model, text, 'zf_xiaobei', device, speed=1.0)

# Save audio
if audio is not None:
    save_audio(audio, 'output.wav')
```

---

## Available Chinese Voices

### Female Voices (女性声音)

| Voice ID | Name | Description | Quality |
|----------|------|-------------|---------| 
| `zf_xiaobei` | 晓蓓 | Young, energetic | B |
| `zf_xiaoni` | 晓妮 | Clear, friendly | B+ |
| `zf_xiaoxiao` | 晓晓 | Soft, gentle | B |
| `zf_xiaoyi` | 晓艺 | Professional, articulate | A- |

### Male Voices (男性声音)

| Voice ID | Name | Description | Quality |
|----------|------|-------------|---------| 
| `zm_yunjian` | 云健 | Strong, confident | B- |
| `zm_yunxi` | 云析 | Warm, professional | B+ |
| `zm_yunxia` | 云夏 | Calm, steady | B |
| `zm_yunyang` | 云阳 | Resonant, deep | B- |

### Recommendations

- **Natural speech**: `zf_xiaoyi` (female) or `zm_yunxi` (male)
- **Energetic content**: `zf_xiaobei` (female) or `zm_yunjian` (male)
- **Gentle/soft content**: `zf_xiaoxiao` (female) or `zm_yunxia` (male)

---

## Troubleshooting

### "WARNING - words count mismatch"

**Cause**: Wrong phonemizer language configuration.

**Solution**: Use `chinese_tts_demo.py` (not `tts_demo.py`). The code automatically initializes the Chinese phonemizer.

### "Model file not found"

**Solution**: Run `python setup_chinese_tts.py` or download manually:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-82M-v1.1_zh.pth', local_dir='.')"
```

### "Voice file not found"

**Solution**: Run `python setup_chinese_tts.py` to download all voice files automatically.

### "No Chinese phonemizer support"

**Solution**: TTS works without phonemizer (no phoneme visualization). To install:
```bash
pip install phonemizer espeakng-loader
# Then install espeak-ng for your platform
```

### Out of memory errors

**Solution**:
- System auto-falls back to CPU
- Use shorter text segments
- Close other applications
- Use already-loaded voice files

---

## Advanced Usage

### Text Processing

The system automatically handles Chinese character validation, normalization, punctuation, and text segmentation. Use utilities for manual processing:

```python
from chinese_config import ChineseTextProcessor

# Check if text is Chinese
is_chinese = ChineseTextProcessor.is_chinese("你好")

# Normalize text
normalized = ChineseTextProcessor.normalize_chinese_text("你好  ，  世界  ！")

# Split long text
segments = ChineseTextProcessor.split_chinese_text("长文本...", max_length=100)
```

### Batch Processing

```python
texts = ["你好，世界", "欢迎使用中文文本转语音", "这是一个测试"]

for i, text in enumerate(texts):
    audio, _ = generate_chinese_speech(model, text, 'zf_xiaobei', device)
    if audio is not None:
        save_audio(audio, f'output_{i}.wav')
```

### Performance Tips

- **First run**: Slower due to model loading
- **Voice caching**: Faster subsequent generations
- **GPU**: ~3x faster with CUDA
- **Memory**: ~400MB when loaded

**Typical generation times (with GPU):**
- Short text (< 30 chars): ~0.5s
- Medium text (30-100 chars): ~1-2s
- Long text (100+ chars): ~2-5s

### Offline Usage

After initial setup, run offline:

```bash
# Linux/macOS
export HF_HUB_OFFLINE=1

# Windows PowerShell
$env:HF_HUB_OFFLINE="1"

# Windows CMD
set HF_HUB_OFFLINE=1

python chinese_tts_demo.py
```

## FAQ

**Q: Can I use this with English TTS?**  
A: Yes, but use different scripts: `tts_demo.py` for English, `chinese_tts_demo.py` for Chinese.

**Q: Can I mix Chinese and English text?**  
A: The system is optimized for pure Chinese text. Mixed text may have lower quality.

**Q: How do I improve audio quality?**  
A: Try different voices, adjust speed, ensure sufficient disk space, and use GPU if available.

**Q: Is there a REST API?**  
A: Not yet, but you can modify `gradio_interface.py` to support Chinese.

## Additional Resources

- **Kokoro Project**: https://github.com/hexgrad/kokoro
- **Model Repository**: https://huggingface.co/hexgrad/Kokoro-82M
- **Main README**: See [README.md](README.md) for general project information

---

**Version**: 1.0 | **Last Updated**: 2024

