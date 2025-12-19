"""
Chinese TTS Configuration Module for Kokoro-82M-v1.1_zh
========================================================

This module provides specialized configuration and utilities for the Kokoro Chinese TTS model.
It handles Chinese-specific phonemization, text processing, and voice management.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Chinese language code
CHINESE_LANG_CODE = 'z'

# Chinese Model Configuration
CHINESE_MODEL_CONFIG = {
    "model_name": "Kokoro-82M-v1.1_zh",
    "model_file": "kokoro-82M-v1.1_zh.pth",
    "repo_id": "hexgrad/Kokoro-82M",
    "language_code": 'z',
    "description": "Kokoro 82M Chinese (Mandarin) TTS Model v1.1",
    "phonemizer": "espeak-zh",  # Specialized Chinese phonemizer
    "sample_rate": 24000,
    "voice_prefix": ["zf_", "zm_"],  # Chinese female (zf_) and male (zm_) voices
}

# Chinese Voice Files - 8 voices total (4 female + 4 male)
CHINESE_VOICES = {
    # Female voices
    "zf_xiaobei": {
        "name": "晓蓓",
        "gender": "Female",
        "description": "Young, energetic female voice",
        "language": "Mandarin Chinese",
        "file": "zf_xiaobei.pt"
    },
    "zf_xiaoni": {
        "name": "晓妮",
        "gender": "Female",
        "description": "Clear, friendly female voice",
        "language": "Mandarin Chinese",
        "file": "zf_xiaoni.pt"
    },
    "zf_xiaoxiao": {
        "name": "晓晓",
        "gender": "Female",
        "description": "Soft, gentle female voice",
        "language": "Mandarin Chinese",
        "file": "zf_xiaoxiao.pt"
    },
    "zf_xiaoyi": {
        "name": "晓艺",
        "gender": "Female",
        "description": "Professional, articulate female voice",
        "language": "Mandarin Chinese",
        "file": "zf_xiaoyi.pt"
    },
    # Male voices
    "zm_yunjian": {
        "name": "云健",
        "gender": "Male",
        "description": "Strong, confident male voice",
        "language": "Mandarin Chinese",
        "file": "zm_yunjian.pt"
    },
    "zm_yunxi": {
        "name": "云析",
        "gender": "Male",
        "description": "Warm, professional male voice",
        "language": "Mandarin Chinese",
        "file": "zm_yunxi.pt"
    },
    "zm_yunxia": {
        "name": "云夏",
        "gender": "Male",
        "description": "Calm, steady male voice",
        "language": "Mandarin Chinese",
        "file": "zm_yunxia.pt"
    },
    "zm_yunyang": {
        "name": "云阳",
        "gender": "Male",
        "description": "Resonant, deep male voice",
        "language": "Mandarin Chinese",
        "file": "zm_yunyang.pt"
    }
}

class ChineseTextProcessor:
    """Handle Chinese-specific text processing and normalization"""
    
    @staticmethod
    def is_chinese(text: str) -> bool:
        """Check if text contains Chinese characters"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Unicode range for CJK unified ideographs
                return True
        return False
    
    @staticmethod
    def normalize_chinese_text(text: str) -> str:
        """Normalize Chinese text for TTS processing"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper spacing around punctuation
        import re
        # Add space after Chinese punctuation, removing any existing spaces first
        text = re.sub(r'\s*([。，！？；：""''（）【】《》])\s*', r'\1 ', text)
        # Clean up any double spaces that may have been created
        text = ' '.join(text.split())
        
        return text.strip()
    
    @staticmethod
    def split_chinese_text(text: str, max_length: int = 100) -> List[str]:
        """Split Chinese text into proper segments for TTS processing
        
        Args:
            text: Chinese text to split
            max_length: Maximum characters per segment
            
        Returns:
            List of text segments
        """
        segments = []
        current_segment = ""
        
        for char in text:
            current_segment += char
            
            # Split on punctuation or max length
            if char in '。！？；，\n' or len(current_segment) >= max_length:
                if current_segment.strip():
                    segments.append(current_segment.strip())
                current_segment = ""
        
        # Add remaining text
        if current_segment.strip():
            segments.append(current_segment.strip())
        
        return segments


class ChineseTTSConfig:
    """Specialized configuration manager for Chinese TTS"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file or "chinese_tts_config.json").resolve()
        self.chinese_voices_dir = Path("voices_chinese").resolve()
        self._config = self._load_default_config()
        self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values for Chinese TTS"""
        return {
            "model": CHINESE_MODEL_CONFIG,
            "voices": CHINESE_VOICES,
            "phonemizer": {
                "backend": "espeak-ng",
                "language": "zh",  # Chinese language code for espeak
                "preserve_punctuation": True,
                "strip": False
            },
            "text_processing": {
                "normalize": True,
                "split_long_text": True,
                "max_segment_length": 100,
                "min_segment_length": 10
            },
            "audio": {
                "sample_rate": 24000,
                "default_speed": 1.0,
                "min_speed": 0.5,
                "max_speed": 2.0
            },
            "paths": {
                "voices_dir": "voices_chinese",
                "models_dir": ".",
                "output_dir": "outputs"
            }
        }
    
    def _load_config_file(self):
        """Load configuration from file if it exists"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._merge_config(file_config)
                logger.info(f"Loaded Chinese TTS configuration from {self.config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load Chinese config file {self.config_file}: {e}")
    
    def _merge_config(self, file_config: Dict[str, Any]):
        """Merge file configuration with default configuration"""
        def merge_dict(default: Dict, override: Dict):
            for key, value in override.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    merge_dict(default[key], value)
                else:
                    default[key] = value
        
        merge_dict(self._config, file_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file"""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            logger.info(f"Chinese TTS configuration saved to {self.config_file}")
        except IOError as e:
            logger.error(f"Failed to save Chinese TTS configuration: {e}")
    
    def get_voices_list(self) -> List[str]:
        """Get list of available Chinese voices"""
        return list(CHINESE_VOICES.keys())
    
    def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific voice"""
        return CHINESE_VOICES.get(voice_name)
    
    def ensure_voices_directory(self):
        """Ensure Chinese voices directory exists"""
        self.chinese_voices_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Chinese voices directory ready: {self.chinese_voices_dir}")
    
    def validate_chinese_model(self, model_path: str) -> bool:
        """Validate Chinese model file"""
        model_file = Path(model_path).resolve()
        if not model_file.exists():
            logger.error(f"Chinese model file not found: {model_file}")
            return False
        
        # Basic file size check (should be > 100MB)
        if model_file.stat().st_size < 100 * 1024 * 1024:
            logger.warning(f"Model file size seems too small: {model_file.stat().st_size}")
        
        return True


# Global configuration instance for Chinese TTS
chinese_config = ChineseTTSConfig()


# Convenience functions
def get_chinese_config(key: str, default: Any = None) -> Any:
    """Get Chinese TTS configuration value"""
    return chinese_config.get(key, default)


def get_chinese_voices() -> List[str]:
    """Get list of available Chinese voices"""
    return list(CHINESE_VOICES.keys())


def get_chinese_voice_info(voice_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific Chinese voice"""
    return CHINESE_VOICES.get(voice_name)


def is_chinese_text(text: str) -> bool:
    """Check if text is in Chinese"""
    return ChineseTextProcessor.is_chinese(text)


def normalize_chinese(text: str) -> str:
    """Normalize Chinese text"""
    return ChineseTextProcessor.normalize_chinese_text(text)


def split_chinese_text(text: str, max_length: int = 100) -> List[str]:
    """Split Chinese text into segments"""
    return ChineseTextProcessor.split_chinese_text(text, max_length)

