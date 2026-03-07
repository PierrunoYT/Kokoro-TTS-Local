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
from pathlib import Path
from typing import Dict, List, Optional, Any

# Define the path for the speed dial presets file
SPEED_DIAL_FILE = Path("speed_dial.json")

def load_presets() -> Dict[str, Dict[str, Any]]:
    """
    Load speed dial presets from the JSON file.
    
    Returns:
        Dictionary of presets where keys are preset names and values are preset data
    """
    if not SPEED_DIAL_FILE.exists():
        # If file doesn't exist, return an empty dictionary
        return {}
    
    try:
        with open(SPEED_DIAL_FILE, 'r', encoding='utf-8') as f:
            presets = json.load(f)

        if not isinstance(presets, dict):
            print(
                "Error loading speed dial presets: "
                f"expected a JSON object, got {type(presets).__name__}"
            )
            return {}
        
        # Validate the loaded presets
        validated_presets = {}
        for name, preset in presets.items():
            if not isinstance(name, str) or not isinstance(preset, dict):
                print(f"Skipping invalid preset entry: {name!r}")
                continue
            if validate_preset(preset):
                validated_presets[name] = preset
        
        return validated_presets
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading speed dial presets: {e}")
        return {}

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
    
    # Load existing presets
    presets = load_presets()
    
    # Add or update the preset
    presets[name] = preset
    
    # Save presets to file
    try:
        with open(SPEED_DIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Error saving speed dial preset: {e}")
        return False

def delete_preset(name: str) -> bool:
    """
    Delete a speed dial preset.
    
    Args:
        name: Name of the preset to delete
        
    Returns:
        True if successful, False otherwise
    """
    # Load existing presets
    presets = load_presets()
    
    # Check if preset exists
    if name not in presets:
        return False
    
    # Remove the preset
    del presets[name]
    
    # Save presets to file
    try:
        with open(SPEED_DIAL_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Error deleting speed dial preset: {e}")
        return False

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
