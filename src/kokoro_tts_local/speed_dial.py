"""
Speed Dial Module for Kokoro-TTS-Local
--------------------------------------
Manages speed dial presets for quick access to frequently used voice and text combinations.

This module provides functions to:
- Load speed dial presets from a JSON file
- Save new presets to the JSON file
- Delete presets from the JSON file
- Validate preset data
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from .paths import get_base_dir

# Define the path for the speed dial presets file
SPEED_DIAL_FILE = get_base_dir() / "speed_dial.json"

_presets_lock = threading.RLock()


def _load_presets(strict: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load and validate presets, optionally propagating read errors."""
    if not SPEED_DIAL_FILE.exists():
        return {}

    try:
        with open(SPEED_DIAL_FILE, 'r', encoding='utf-8') as f:
            presets = json.load(f)

        if not isinstance(presets, dict):
            raise ValueError(
                "expected a JSON object, "
                f"got {type(presets).__name__}"
            )

        validated_presets = {}
        for name, preset in presets.items():
            if not isinstance(name, str) or not isinstance(preset, dict):
                if strict:
                    raise ValueError(f"invalid preset entry: {name!r}")
                print(f"Skipping invalid preset entry: {name!r}")
                continue
            if validate_preset(preset):
                validated_presets[name] = preset
            elif strict:
                raise ValueError(f"invalid preset data: {name!r}")
        return validated_presets
    except (json.JSONDecodeError, OSError, ValueError) as e:
        print(f"Error loading speed dial presets: {e}")
        if strict:
            raise
        return {}


def _atomic_write_presets(presets: Dict[str, Dict[str, Any]]) -> bool:
    """Durably replace the presets file without exposing partial JSON."""
    parent = SPEED_DIAL_FILE.parent
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=str(parent),
            prefix=".speed_dial.",
            suffix=".tmp"
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, SPEED_DIAL_FILE)
        return True
    except (OSError, TypeError, ValueError) as e:
        print(f"Error writing speed dial presets: {e}")
        return False
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass

def load_presets() -> Dict[str, Dict[str, Any]]:
    """
    Load speed dial presets from the JSON file.

    Returns:
        Dictionary of presets where keys are preset names and values are preset data
    """
    with _presets_lock:
        return _load_presets()

def save_preset(name: str, voice: str, text: str, format: str = "wav", speed: float = 1.0) -> bool:
    """
    Save a new speed dial preset.
    
    Args:
        name: Name of the preset
        voice: Voice to use
        text: Text to convert to speech
        format: Output format (default: "wav")
        speed: Speech speed (default: 1.0)
        
    Returns:
        True if successful, False otherwise
    """
    import re
    
    # Validate preset name
    if not isinstance(name, str) or len(name.strip()) == 0:
        print("Preset name must be a non-empty string")
        return False
    
    if len(name) > 50:
        print("Preset name is too long (max 50 characters)")
        return False
    
    # Only allow safe characters in preset names
    if not re.match(r'^[a-zA-Z0-9_\- ]+$', name):
        print("Preset name contains invalid characters")
        return False
    
    # Create preset data
    preset = {
        "voice": voice,
        "text": text,
        "format": format,
        "speed": speed
    }
    
    # Validate preset data
    if not validate_preset(preset):
        return False
    
    with _presets_lock:
        try:
            presets = _load_presets(strict=True)
        except (json.JSONDecodeError, OSError, ValueError):
            return False

        presets[name] = preset
        return _atomic_write_presets(presets)

def delete_preset(name: str) -> bool:
    """
    Delete a speed dial preset.
    
    Args:
        name: Name of the preset to delete
        
    Returns:
        True if successful, False otherwise
    """
    with _presets_lock:
        try:
            presets = _load_presets(strict=True)
        except (json.JSONDecodeError, OSError, ValueError):
            return False

        if name not in presets:
            return False

        del presets[name]
        return _atomic_write_presets(presets)

def validate_preset(preset: Dict[str, Any]) -> bool:
    """
    Validate a preset's data structure with security checks.
    
    Args:
        preset: Preset data to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    
    # Check required fields
    required_fields = ["voice", "text"]
    for field in required_fields:
        if field not in preset:
            print(f"Preset missing required field: {field}")
            return False
    
    # Check field types and validate content
    voice = preset.get("voice")
    if not isinstance(voice, str):
        print("Preset voice must be a string")
        return False
    
    # Validate voice name (alphanumeric, underscore, dash only)
    if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
        print("Preset voice contains invalid characters")
        return False
    
    text = preset.get("text")
    if not isinstance(text, str):
        print("Preset text must be a string")
        return False
    
    # Validate text length and content
    if len(text) > 10000:
        print("Preset text is too long (max 10,000 characters)")
        return False
    
    if len(text.strip()) == 0:
        print("Preset text cannot be empty")
        return False
    
    # Optional fields with validation
    if "format" not in preset:
        preset["format"] = "wav"
    else:
        format_val = preset["format"]
        if not isinstance(format_val, str):
            print("Preset format must be a string")
            return False
        # Only allow safe audio formats
        if format_val not in ["wav", "mp3", "aac"]:
            print("Preset format must be wav, mp3, or aac")
            return False
    
    if "speed" not in preset:
        preset["speed"] = 1.0
    else:
        speed = preset["speed"]
        if not isinstance(speed, (int, float)):
            print("Preset speed must be a number")
            return False
        # Validate speed range
        if speed < 0.1 or speed > 3.0:
            print("Preset speed must be between 0.1 and 3.0")
            return False
    
    return True

def get_preset_names() -> List[str]:
    """
    Get a list of all preset names.
    
    Returns:
        List of preset names
    """
    presets = load_presets()
    return list(presets.keys())

def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific preset by name.
    
    Args:
        name: Name of the preset to get
        
    Returns:
        Preset data or None if not found
    """
    presets = load_presets()
    return presets.get(name)
