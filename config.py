"""
Centralized Configuration System for Kokoro TTS Local
----------------------------------------------------
This module provides centralized configuration management for all components
of the Kokoro TTS Local application.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TTSConfig:
    """Centralized configuration manager for TTS application"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file or "tts_config.json").resolve()
        self._config = self._load_default_config()
        self._load_config_file()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values"""
        return {
            # Audio settings
            "audio": {
                "sample_rate": 24000,
                "max_text_length_cli": 10000,
                "max_text_length_web": 5000,
                "min_speed": 0.1,
                "max_speed": 3.0,
                "default_speed": 1.0,
                "supported_formats": ["wav", "mp3", "aac"]
            },
            
            # Model settings
            "model": {
                "default_model_path": "kokoro-v1_0.pth",
                "repo_id": "hexgrad/Kokoro-82M",
                "repo_version": "main",
                "default_language": "a",
                "max_generation_time": 300,
                "min_generation_time": 60,
                "max_retries": 3,
                "retry_delay": 2
            },
            
            # Path settings
            "paths": {
                "voices_dir": "voices",
                "outputs_dir": "outputs",
                "cache_dir": ".cache",
                "config_file": "config.json",
                "speed_dial_file": "speed_dial.json"
            },
            
            # Web interface settings
            "web": {
                "server_name": "0.0.0.0",
                "server_port": 7860,
                "share": False
            },
            
            # CLI settings
            "cli": {
                "default_output_file": "output.wav"
            },
            
            # Language codes mapping
            "language_codes": {
                'a': 'American English',
                'b': 'British English',
                'j': 'Japanese',
                'z': 'Mandarin Chinese',
                'e': 'Spanish',
                'f': 'French',
                'h': 'Hindi',
                'i': 'Italian',
                'p': 'Brazilian Portuguese'
            },
            
            # Voice files list
            "voice_files": [
                # American English Female voices (11 voices)
                "af_heart.pt", "af_alloy.pt", "af_aoede.pt", "af_bella.pt", "af_jessica.pt",
                "af_kore.pt", "af_nicole.pt", "af_nova.pt", "af_river.pt", "af_sarah.pt", "af_sky.pt",
                
                # American English Male voices (9 voices)
                "am_adam.pt", "am_echo.pt", "am_eric.pt", "am_fenrir.pt", "am_liam.pt",
                "am_michael.pt", "am_onyx.pt", "am_puck.pt", "am_santa.pt",
                
                # British English Female voices (4 voices)
                "bf_alice.pt", "bf_emma.pt", "bf_isabella.pt", "bf_lily.pt",
                
                # British English Male voices (4 voices)
                "bm_daniel.pt", "bm_fable.pt", "bm_george.pt", "bm_lewis.pt",
                
                # Japanese voices (5 voices)
                "jf_alpha.pt", "jf_gongitsune.pt", "jf_nezumi.pt", "jf_tebukuro.pt", "jm_kumo.pt",
                
                # Mandarin Chinese voices (8 voices)
                "zf_xiaobei.pt", "zf_xiaoni.pt", "zf_xiaoxiao.pt", "zf_xiaoyi.pt",
                "zm_yunjian.pt", "zm_yunxi.pt", "zm_yunxia.pt", "zm_yunyang.pt",
                
                # Spanish voices (3 voices)
                "ef_dora.pt", "em_alex.pt", "em_santa.pt",
                
                # French voices (1 voice)
                "ff_siwis.pt",
                
                # Hindi voices (4 voices)
                "hf_alpha.pt", "hf_beta.pt", "hm_omega.pt", "hm_psi.pt",
                
                # Italian voices (2 voices)
                "if_sara.pt", "im_nicola.pt",
                
                # Brazilian Portuguese voices (3 voices)
                "pf_dora.pt", "pm_alex.pt", "pm_santa.pt"
            ]
        }
    
    def _load_config_file(self):
        """Load configuration from file if it exists"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                self._merge_config(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
    
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
        """Get configuration value using dot notation (e.g., 'audio.sample_rate')"""
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
            logger.info(f"Configuration saved to {self.config_file}")
        except IOError as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_path(self, path_key: str) -> Path:
        """Get a path from configuration and return as resolved Path object"""
        path_str = self.get(f"paths.{path_key}")
        if path_str:
            return Path(path_str).resolve()
        raise ValueError(f"Path key '{path_key}' not found in configuration")
    
    def validate_sample_rate(self, rate: int) -> int:
        """Validate and normalize sample rate to acceptable values
        
        Returns the rate if valid, otherwise returns the default sample rate.
        """
        valid_rates = [16000, 22050, 24000, 44100, 48000]
        if rate not in valid_rates:
            default_rate = self.get("audio.sample_rate", 24000)
            logger.warning(
                f"Invalid sample rate {rate}. Valid rates are {valid_rates}. "
                f"Using default rate: {default_rate}"
            )
            return default_rate
        return rate
    
    def validate_language(self, lang: str) -> str:
        """Validate language code"""
        valid_langs = list(self.get("language_codes", {}).keys())
        if lang not in valid_langs:
            logger.warning(f"Invalid language code '{lang}'. Using default.")
            logger.info(f"Supported language codes: {', '.join(valid_langs)}")
            return self.get("model.default_language", "a")
        return lang
    
    def validate_speed(self, speed: float) -> float:
        """Validate speech speed is within acceptable range"""
        min_speed = self.get("audio.min_speed", 0.1)
        max_speed = self.get("audio.max_speed", 3.0)
        
        if speed < min_speed:
            logger.warning(f"Speed {speed} too low, using minimum {min_speed}")
            return min_speed
        elif speed > max_speed:
            logger.warning(f"Speed {speed} too high, using maximum {max_speed}")
            return max_speed
        
        return speed

# Global configuration instance
config = TTSConfig()

# Convenience functions for backward compatibility
def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return config.get(key, default)

def set_config(key: str, value: Any):
    """Set configuration value"""
    config.set(key, value)

def save_config():
    """Save configuration to file"""
    config.save()

def get_path(path_key: str) -> Path:
    """Get a path from configuration"""
    return config.get_path(path_key)