"""
Kokoro-TTS Local Generator
-------------------------
A Gradio interface for the Kokoro-TTS-Local text-to-speech system.
Supports multiple voices and audio formats, with cross-platform compatibility.

Key Features:
- Multiple voice models support (54 voices across 8 languages)
- Real-time generation with progress logging
- WAV, MP3, and AAC output formats
- Network sharing capabilities
- Cross-platform compatibility (Windows, macOS, Linux)

Dependencies:
- kokoro: Official Kokoro TTS library
- gradio: Web interface framework
- soundfile: Audio file handling
- pydub: Audio format conversion
"""

import gradio as gr
import os
import sys
import platform
from datetime import datetime
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
import argparse
from typing import Union, List, Optional, Tuple, Dict, Any
from models import (
    list_available_voices, build_model,
    generate_speech, download_voice_files, EnhancedKPipeline
)
import speed_dial

# Constants
MAX_TEXT_LENGTH = 5000
DEFAULT_SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
DEFAULT_SPEED = 1.0

# Define path type for consistent handling
PathLike = Union[str, Path]

# Configuration validation
def validate_sample_rate(rate: int) -> int:
    """Validate sample rate is within acceptable range"""
    valid_rates = [16000, 22050, 24000, 44100, 48000]
    if rate not in valid_rates:
        print(f"Warning: Unusual sample rate {rate}. Valid rates are {valid_rates}")
        return 24000  # Default to safe value
    return rate

# Global configuration
CONFIG_FILE = Path("tts_config.json")  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = Path("outputs")    # Directory for generated audio files
SAMPLE_RATE = validate_sample_rate(24000)  # Validated sample rate

# Initialize model globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None

LANG_MAP = {
    "af_": "a", "am_": "a",
    "bf_": "b", "bm_": "b",
    "jf_": "j", "jm_": "j",
    "zf_": "z", "zm_": "z",
    "ef_": "e", "em_": "e",
    "ff_": "f",
    "hf_": "h", "hm_": "h",
    "if_": "i", "im_": "i",
    "pf_": "p", "pm_": "p",
}
pipelines = {}

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        global model
        if model is None:
            print("Initializing model and downloading voices...")
            model = build_model(None, device)

        voices = list_available_voices()
        if not voices:
            print("No voices found after initialization. Attempting to download...")
            download_voice_files()  # Try downloading again
            voices = list_available_voices()

        print("Available voices:", voices)
        return voices
    except Exception as e:
        print(f"Error getting voices: {e}")
        return []

def get_pipeline_for_voice(voice_name: str) -> EnhancedKPipeline:
    """
    Determine the language code from the voice prefix and return the associated pipeline.
    """
    prefix = voice_name[:3].lower()
    lang_code = LANG_MAP.get(prefix, "a")
    if lang_code not in pipelines:
        print(f"[INFO] Creating pipeline for lang_code='{lang_code}'")
        pipelines[lang_code] = EnhancedKPipeline(lang_code=lang_code, model=True)
    return pipelines[lang_code]

def convert_audio(input_path: PathLike, output_path: PathLike, format: str) -> Optional[PathLike]:
    """Convert audio to specified format.

    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        format: Output format ('wav', 'mp3', or 'aac')

    Returns:
        Path to output file or None on error
    """
    try:
        # Normalize paths
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # For WAV format, just return the input path
        if format.lower() == "wav":
            return input_path

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert format
        audio = AudioSegment.from_wav(str(input_path))

        # Select proper format and options
        if format.lower() == "mp3":
            audio.export(str(output_path), format="mp3", bitrate="192k")
        elif format.lower() == "aac":
            audio.export(str(output_path), format="aac", bitrate="192k")
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Verify file was created
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise IOError(f"Failed to create {format} file")

        return output_path

    except (IOError, FileNotFoundError, ValueError) as e:
        print(f"Error converting audio: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error converting audio: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_tts_with_logs(voice_name: str, text: str, format: str, speed: float = 1.0) -> Optional[PathLike]:
    """Generate TTS audio with progress logging and memory management.

    Args:
        voice_name: Name of the voice to use
        text: Text to convert to speech
        format: Output format ('wav', 'mp3', 'aac')

    Returns:
        Path to generated audio file or None on error
    """
    global model
    import psutil
    import gc

    try:
        # Check available memory before processing
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < 1.0:  # Less than 1GB available
            print(f"Warning: Low memory available ({available_gb:.1f}GB). Consider closing other applications.")
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Initialize model if needed
        if model is None:
            print("Initializing model...")
            model = build_model(None, device)

        # Create output directory
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Dynamic text length limit based on available memory
        MAX_CHARS = MAX_TEXT_LENGTH
        if available_gb < 2.0:  # Less than 2GB available
            MAX_CHARS = min(MAX_CHARS, 2000)  # Reduce limit for low memory
            print(f"Reduced text limit to {MAX_CHARS} characters due to low memory")
        
        if len(text) > MAX_CHARS:
            print(f"Warning: Text exceeds {MAX_CHARS} characters. Truncating to prevent memory issues.")
            text = text[:MAX_CHARS] + "..."

        # Generate base filename from text
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}"
        wav_path = DEFAULT_OUTPUT_DIR / f"{base_name}.wav"

        # Generate speech
        print(f"\nGenerating speech for: '{text}'")
        print(f"Using voice: {voice_name}")

        # Validate voice path using Path for consistent handling
        voice_path = Path("voices").resolve() / f"{voice_name}.pt"
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        try:
            if voice_name.startswith(tuple(LANG_MAP.keys())):
                pipeline = get_pipeline_for_voice(voice_name)
                generator = pipeline(text, voice=voice_path, speed=speed, split_pattern=r'\n+')
            else:
                generator = model(text, voice=voice_path, speed=speed, split_pattern=r'\n+')

            all_audio = []
            max_segments = 100  # Safety limit for very long texts
            segment_count = 0

            for gs, ps, audio in generator:
                segment_count += 1
                if segment_count > max_segments:
                    print(f"Warning: Reached maximum segment limit ({max_segments})")
                    break

                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    print(f"Generated segment: {gs}")
                    if ps:  # Only print phonemes if available
                        print(f"Phonemes: {ps}")

            if not all_audio:
                raise Exception("No audio generated")
        except Exception as e:
            raise Exception(f"Error in speech generation: {e}")

        # Combine audio segments and save
        if not all_audio:
            raise Exception("No audio segments were generated")

        # Handle single segment case without concatenation
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            try:
                final_audio = torch.cat(all_audio, dim=0)
            except RuntimeError as e:
                raise Exception(f"Failed to concatenate audio segments: {e}")

        # Save audio file
        try:
            sf.write(wav_path, final_audio.numpy(), SAMPLE_RATE)
        except Exception as e:
            raise Exception(f"Failed to save audio file: {e}")

        # Convert to requested format if needed
        if format.lower() != "wav":
            output_path = DEFAULT_OUTPUT_DIR / f"{base_name}.{format.lower()}"
            return convert_audio(wav_path, output_path, format.lower())

        return wav_path

    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_interface(server_name="127.0.0.1", server_port=7860):
    """Create and launch the Gradio interface."""

    # Get available voices
    voices = get_available_voices()
    if not voices:
        print("No voices found! Please check the voices directory.")
        return

    # Get speed dial presets
    preset_names = speed_dial.get_preset_names()

    # Create interface
    with gr.Blocks(title="Kokoro TTS Generator", fill_height=True) as interface:
        gr.Markdown("# Kokoro TTS Generator")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## TTS Controls")
            
            with gr.Column(scale=1):
                gr.Markdown("## Speed Dial")
                
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                # Main TTS controls
                
                voice = gr.Dropdown(
                    choices=voices,
                    value=voices[0] if voices else None,
                    label="Voice"
                )
                text = gr.Textbox(
                    lines=3,
                    placeholder="Enter text to convert to speech...",
                    label="Text"
                )

            with gr.Column(scale=1):
                # Speed dial section
                preset_dropdown = gr.Dropdown(
                    choices=preset_names,
                    value=preset_names[0] if preset_names else None,
                    label="Saved Presets",
                    interactive=True
                )
                preset_name = gr.Textbox(
                    placeholder="Enter preset name...",
                    label="New Preset Name"
                )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                with gr.Row():
                    format = gr.Radio(
                        choices=["wav", "mp3", "aac"],
                        value="wav",
                        label="Output Format"
                    )
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Speed"
                    )

            with gr.Column(scale=1):
                load_preset = gr.Button("Load")
                save_preset = gr.Button("Save Current")

        with gr.Row():
            with gr.Column(scale=2):
                generate = gr.Button("Generate Speech")

            with gr.Column(scale=1):
                delete_preset = gr.Button("Delete")

        with gr.Row():
            # Output section
            output = gr.Audio(label="Generated Audio")

        # Function to load a preset
        def load_preset_fn(preset_name):
            if not preset_name:
                return None, None, None, None

            preset = speed_dial.get_preset(preset_name)
            if not preset:
                return None, None, None, None

            return preset["voice"], preset["text"], preset["format"], preset["speed"]

        # Function to save a preset
        def save_preset_fn(name, voice, text, format, speed):
            if not name or not voice or not text:
                return gr.update(value="Please provide a name, voice, and text")

            success = speed_dial.save_preset(name, voice, text, format, speed)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return gr.update(choices=preset_names, value=name)
            else:
                return gr.update(choices=preset_names)

        # Function to delete a preset
        def delete_preset_fn(name):
            if not name:
                return gr.update(value="Please select a preset to delete")

            success = speed_dial.delete_preset(name)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return gr.update(choices=preset_names, value=None)
            else:
                return gr.update(choices=preset_names)

        # Connect the buttons to their functions
        load_preset.click(
            fn=load_preset_fn,
            inputs=preset_dropdown,
            outputs=[voice, text, format, speed]
        )

        save_preset.click(
            fn=save_preset_fn,
            inputs=[preset_name, voice, text, format, speed],
            outputs=preset_dropdown
        )

        delete_preset.click(
            fn=delete_preset_fn,
            inputs=preset_dropdown,
            outputs=preset_dropdown
        )

        # Connect the generate button
        generate.click(
            fn=generate_tts_with_logs,
            inputs=[voice, text, format, speed],
            outputs=output
        )

    # Launch interface
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False
    )

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    global model

    try:
        print("Cleaning up resources...")

        # Clean up model resources
        if model is not None:
            print("Releasing model resources...")

            # Clear voice dictionary to release memory
            if hasattr(model, 'voices') and model.voices is not None:
                try:
                    voice_count = len(model.voices)
                    for voice_name in list(model.voices.keys()):
                        try:
                            # Release each voice explicitly
                            model.voices[voice_name] = None
                        except:
                            pass
                    model.voices.clear()
                    print(f"Cleared {voice_count} voice references")
                except Exception as ve:
                    print(f"Error clearing voices: {type(ve).__name__}: {ve}")

            # Clear model attributes that might hold tensors
            for attr_name in dir(model):
                if not attr_name.startswith('__') and hasattr(model, attr_name):
                    try:
                        attr = getattr(model, attr_name)
                        # Handle specific tensor attributes
                        if isinstance(attr, torch.Tensor):
                            if attr.is_cuda:
                                print(f"Releasing CUDA tensor: {attr_name}")
                                setattr(model, attr_name, None)
                        elif hasattr(attr, 'to'):  # Module or Tensor-like object
                            setattr(model, attr_name, None)
                    except:
                        pass

            # Delete model reference
            try:
                del model
                model = None
                print("Model reference deleted")
            except Exception as me:
                print(f"Error deleting model: {type(me).__name__}: {me}")

        # Clear CUDA memory explicitly
        if torch.cuda.is_available():
            try:
                # Get initial memory usage
                try:
                    initial = torch.cuda.memory_allocated()
                    initial_mb = initial / (1024 * 1024)
                    print(f"CUDA memory before cleanup: {initial_mb:.2f} MB")
                except:
                    pass

                # Free memory
                print("Clearing CUDA cache...")
                torch.cuda.empty_cache()

                # Force synchronization
                try:
                    torch.cuda.synchronize()
                except:
                    pass

                # Get final memory usage
                try:
                    final = torch.cuda.memory_allocated()
                    final_mb = final / (1024 * 1024)
                    freed_mb = (initial - final) / (1024 * 1024)
                    print(f"CUDA memory after cleanup: {final_mb:.2f} MB (freed {freed_mb:.2f} MB)")
                except:
                    pass
            except Exception as ce:
                print(f"Error clearing CUDA memory: {type(ce).__name__}: {ce}")

        # Final garbage collection
        try:
            import gc
            collected = gc.collect()
            print(f"Garbage collection completed: {collected} objects collected")
        except Exception as gce:
            print(f"Error during garbage collection: {type(gce).__name__}: {gce}")

        print("Cleanup completed")

    except Exception as e:
        print(f"Error during cleanup: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# Register cleanup for normal exit
import atexit
atexit.register(cleanup_resources)

# Register cleanup for signals
import signal
import sys

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    cleanup_resources()
    sys.exit(0)

# Register for common signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, signal_handler)
    except (ValueError, AttributeError):
        # Some signals might not be available on all platforms
        pass

def parse_arguments():
    """Parse command line arguments for host and port configuration."""
    parser = argparse.ArgumentParser(
        description="Kokoro TTS Local Generator - Gradio Web Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to run the server on"
    )
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_arguments()
        create_interface(server_name=args.host, server_port=args.port)
    finally:
        # Ensure cleanup even if Gradio encounters an error
        cleanup_resources()
