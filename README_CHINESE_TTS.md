# Kokoro Chinese TTS - Quick Reference

Quick start guide for Chinese TTS. For complete documentation, see [CHINESE_TTS_GUIDE.md](CHINESE_TTS_GUIDE.md).

## Quick Start

```bash
# 1. Setup (downloads model and voices)
python setup_chinese_tts.py

# 2. Run interactive demo
python chinese_tts_demo.py
```

## Python API

```python
from chinese_tts_demo import load_chinese_model, generate_chinese_speech, save_audio
import torch

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_chinese_model('kokoro-82M-v1.1_zh.pth', device)

# Generate speech
audio, _ = generate_chinese_speech(
    model, 
    "你好，世界！",  # Your Chinese text
    'zf_xiaobei',    # Voice ID
    device,
    speed=1.0
)

# Save audio
if audio is not None:
    save_audio(audio, 'output.wav')
```

## Available Voices

**Female (女性)**: `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`  
**Male (男性)**: `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

**Recommended**: `zf_xiaoyi` (female) or `zm_yunxi` (male) for natural speech.

## Key Features

- ✅ 8 Chinese voices (4 female + 4 male)
- ✅ Natural Mandarin pronunciation
- ✅ Adjustable speed (0.5x - 2.0x)
- ✅ Offline operation after setup
- ✅ Cross-platform support

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python setup_chinese_tts.py` |
| Voice files missing | Run `python setup_chinese_tts.py` |
| "words count mismatch" warning | Use `chinese_tts_demo.py` (not `tts_demo.py`) |
| Out of memory | System auto-falls back to CPU |

## Documentation

- **Complete Guide**: [CHINESE_TTS_GUIDE.md](CHINESE_TTS_GUIDE.md)
- **Main README**: [README.md](README.md)

---

**Version**: 1.0 | **Model**: Kokoro-82M-v1.1_zh
