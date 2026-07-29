"""Platform-aware application data paths with environment overrides."""

import os
import sys
from pathlib import Path


def get_base_dir() -> Path:
    configured = os.environ.get("KOKORO_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return (root / "kokoro-tts-local").resolve()


def get_voices_dir() -> Path:
    configured = os.environ.get("KOKORO_VOICES_DIR")
    return Path(configured).expanduser().resolve() if configured else get_base_dir() / "voices"


def get_model_dir() -> Path:
    configured = os.environ.get("KOKORO_MODEL_DIR")
    return Path(configured).expanduser().resolve() if configured else get_base_dir()


def get_config_path(is_chinese_model: bool = False) -> Path:
    configured = os.environ.get("KOKORO_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    name = "config-v1_1-zh.json" if is_chinese_model else "config.json"
    return get_model_dir() / name
