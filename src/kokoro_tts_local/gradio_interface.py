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
import uuid
import ipaddress
import shutil
from pathlib import Path
import soundfile as sf
from pydub import AudioSegment
import torch
import numpy as np
import argparse
import threading
from typing import Union, List, Optional, Tuple, Dict, Any
from contextlib import closing
from .models import (
    list_available_voices, build_model,
    download_voice_files, EnhancedKPipeline,
    get_safe_voice_path, get_language_code_from_voice, shutdown_pipelines,
    get_base_dir
)
from . import speed_dial

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
CONFIG_FILE = get_base_dir() / "tts_config.json"  # Stores user preferences and paths
DEFAULT_OUTPUT_DIR = get_base_dir() / "outputs"    # Directory for generated audio files
SAMPLE_RATE = validate_sample_rate(24000)  # Validated sample rate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_available_voices():
    """Get list of available voice models."""
    try:
        # Initialize model to trigger voice downloads
        build_model(None, device)

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
    lang_code = get_language_code_from_voice(voice_name)
    return build_model(None, device, lang_code=lang_code)

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
            audio.export(str(output_path), format="adts", codec="aac", bitrate="192k")
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

def generate_tts_with_logs(
    voice_name: str, text: str, format: str, speed: float = 1.0
) -> Tuple[Optional[PathLike], str]:
    """Generate TTS audio with progress logging and memory management.

    Args:
        voice_name: Name of the voice to use
        text: Text to convert to speech
        format: Output format ('wav', 'mp3', 'aac')
        speed: Speech speed multiplier

    Returns:
        ``(path, status)``. The status string is rendered in the UI: a remote
        user has no access to this process's console, so a bare ``None`` left
        them unable to tell a missing ffmpeg from a still-running request.
    """
    import psutil
    import gc

    notes = []
    try:
        # Check available memory before processing
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)

        if available_gb < 1.0:  # Less than 1GB available
            message = (f"Low memory available ({available_gb:.1f}GB). "
                       "Consider closing other applications.")
            print(f"Warning: {message}")
            notes.append(message)
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Create output directory
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Validate input text
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        # Validate speed server-side. The slider's bounds are advisory: a
        # crafted request can send any float, and speed=1e-9 asks the vocoder
        # to stretch the utterance by a factor of a billion. NaN is rejected
        # explicitly because it fails both comparisons.
        try:
            speed = float(speed)
        except (TypeError, ValueError):
            raise ValueError(f"Speed must be a number, got {speed!r}")
        if not (MIN_SPEED <= speed <= MAX_SPEED):
            raise ValueError(
                f"Speed must be between {MIN_SPEED} and {MAX_SPEED}, got {speed}"
            )

        # Dynamic text length limit based on available memory
        MAX_CHARS = MAX_TEXT_LENGTH
        if available_gb < 2.0:  # Less than 2GB available
            MAX_CHARS = min(MAX_CHARS, 2000)  # Reduce limit for low memory
            message = f"Reduced text limit to {MAX_CHARS} characters due to low memory"
            print(message)
            notes.append(message)

        if len(text) > MAX_CHARS:
            message = (f"Text exceeded {MAX_CHARS} characters and was truncated; "
                       "only the beginning was synthesized.")
            print(f"Warning: {message}")
            notes.append(message)
            text = text[:MAX_CHARS] + "..."

        # Generate base filename from text
        output_format = format.lower()
        if output_format not in {'wav', 'mp3', 'aac'}:
            raise ValueError(f"Unsupported format: {format}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tts_{timestamp}_{uuid.uuid4().hex}"
        wav_path = DEFAULT_OUTPUT_DIR / f"{base_name}.wav"

        # Generate speech
        print(f"\nGenerating speech for: '{text}'")
        print(f"Using voice: {voice_name}")

        # Validate voice path using safe helper to prevent path traversal
        voice_path = get_safe_voice_path(voice_name)
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        try:
            pipeline = get_pipeline_for_voice(voice_name)

            all_audio = []
            max_segments = 100  # Safety limit for very long texts
            segment_count = 0

            # The generator owns the model-family lock until it is closed, so
            # it must not outlive this loop: hitting max_segments would
            # otherwise hold the lock through file writing and conversion, and
            # block shutdown, which drains by acquiring every family lock.
            with closing(pipeline.iter_speech(
                text, voice=str(voice_path), speed=speed, split_pattern=r'\n+'
            )) as generator:
                for gs, ps, audio in generator:
                    segment_count += 1
                    if segment_count > max_segments:
                        message = (f"Reached the {max_segments}-segment limit; "
                                   "the audio is truncated.")
                        print(f"Warning: {message}")
                        notes.append(message)
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
            raise RuntimeError(f"Speech generation failed: {e}") from e

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
            if isinstance(final_audio, torch.Tensor):
                final_audio = final_audio.detach().cpu().numpy()
            sf.write(wav_path, final_audio, SAMPLE_RATE)
        except Exception as e:
            raise Exception(f"Failed to save audio file: {e}")

        # Convert to requested format if needed
        if output_format != "wav":
            output_path = DEFAULT_OUTPUT_DIR / f"{base_name}.{output_format}"
            converted = convert_audio(wav_path, output_path, output_format)
            if converted is None:
                raise RuntimeError(
                    f"Conversion to {output_format} failed. This usually means "
                    "FFmpeg is not installed or not on PATH."
                )
            wav_path.unlink(missing_ok=True)
            return converted, _status(f"Generated {segment_count} segment(s) as "
                                      f"{output_format}.", notes)

        return wav_path, _status(f"Generated {segment_count} segment(s) as wav.", notes)

    except Exception as e:
        print(f"Error generating speech: {e}")
        import traceback
        traceback.print_exc()
        return None, _status(f"{type(e).__name__}: {e}", notes)


def _status(headline: str, notes: List[str]) -> str:
    """Join a headline with any warnings collected during the request."""
    return "\n".join([headline] + [f"- {note}" for note in notes])

def create_interface(server_name="127.0.0.1", server_port=7860, auth=None):
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
                        minimum=0.1,
                        maximum=3.0,
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

        with gr.Row():
            # Every handler reports here. Without it the only failure signal
            # was an empty audio player, identical for empty input, a missing
            # voice file, absent ffmpeg and CUDA OOM.
            status = gr.Textbox(
                label="Status",
                value="Ready.",
                lines=3,
                interactive=False
            )

        # Function to load a preset. Returns gr.update() (leave unchanged)
        # rather than None on the error paths: None is not a valid Slider
        # value and would send speed=None into the pipeline on the next run.
        def load_preset_fn(preset_name):
            unchanged = (gr.update(), gr.update(), gr.update(), gr.update())
            if not preset_name:
                return (*unchanged, "Select a preset to load.")

            preset = speed_dial.get_preset(preset_name)
            if not preset:
                return (*unchanged, f"Preset {preset_name!r} could not be loaded.")

            return (preset["voice"], preset["text"], preset["format"],
                    preset["speed"], f"Loaded preset {preset_name!r}.")

        # Function to save a preset. The status string goes to the status box,
        # never into the dropdown — a dropdown whose selected value is an error
        # message will feed that message back to get_preset() on the next Load.
        def save_preset_fn(name, voice, text, format, speed):
            if not name or not voice or not text:
                return gr.update(), "Provide a name, a voice, and text to save a preset."

            success = speed_dial.save_preset(name, voice, text, format, speed)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return (gr.update(choices=preset_names, value=name),
                        f"Saved preset {name!r}.")
            return (gr.update(choices=preset_names),
                    f"Could not save preset {name!r}. Check the console for details.")

        # Function to delete a preset
        def delete_preset_fn(name):
            if not name:
                return gr.update(), "Select a preset to delete."

            success = speed_dial.delete_preset(name)

            # Update the dropdown with the new preset list
            preset_names = speed_dial.get_preset_names()

            if success:
                return (gr.update(choices=preset_names, value=None),
                        f"Deleted preset {name!r}.")
            return (gr.update(choices=preset_names),
                    f"Could not delete preset {name!r}. Check the console for details.")

        # Connect the buttons to their functions
        load_preset.click(
            fn=load_preset_fn,
            inputs=preset_dropdown,
            outputs=[voice, text, format, speed, status]
        )

        save_preset.click(
            fn=save_preset_fn,
            inputs=[preset_name, voice, text, format, speed],
            outputs=[preset_dropdown, status]
        )

        delete_preset.click(
            fn=delete_preset_fn,
            inputs=preset_dropdown,
            outputs=[preset_dropdown, status]
        )

        # Connect the generate button
        generate.click(
            fn=generate_tts_with_logs,
            inputs=[voice, text, format, speed],
            outputs=[output, status]
        )

    # Launch interface
    launch_kwargs = dict(
        server_name=server_name,
        server_port=server_port,
        share=False
    )
    if auth is not None:
        launch_kwargs["auth"] = auth
    interface.launch(**launch_kwargs)

def cleanup_resources():
    """Properly clean up resources when the application exits"""
    try:
        print("Cleaning up resources...")
        shutdown_pipelines()

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
                try:
                    torch.cuda.synchronize()
                except:
                    pass
                torch.cuda.empty_cache()

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

import atexit
import signal
import sys

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    raise SystemExit(0)

def register_cleanup_handlers():
    """Register process handlers once, and only from the main thread."""
    if threading.current_thread() is not threading.main_thread():
        return False
    atexit.register(cleanup_resources)
    supported = {signal.SIGINT}
    if hasattr(signal, 'SIGTERM'):
        supported.add(signal.SIGTERM)
    for sig in supported:
        try:
            signal.signal(sig, signal_handler)
        except (ValueError, OSError, RuntimeError):
            pass
    return True

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
    parser.add_argument(
        "--username",
        type=str,
        default=os.environ.get("KOKORO_TTS_USERNAME"),
        help="Username for Gradio authentication "
             "(or set KOKORO_TTS_USERNAME env var)"
    )
    parser.add_argument(
        "--password",
        type=str,
        default=os.environ.get("KOKORO_TTS_PASSWORD"),
        help="Password for Gradio authentication "
             "(or set KOKORO_TTS_PASSWORD env var; preferred to avoid "
             "leaking via shell history / process list)"
    )
    return parser.parse_args()

def main() -> None:
    cleanup_registered = False
    try:
        cleanup_registered = register_cleanup_handlers()
        args = parse_arguments()
        auth = None
        if args.username and args.password:
            auth = [(args.username, args.password)]
        elif args.username or args.password:
            raise ValueError("--username and --password must both be set")
        try:
            is_loopback = ipaddress.ip_address(args.host).is_loopback
        except ValueError:
            is_loopback = args.host.lower() == 'localhost'
        if not is_loopback and auth is None:
            raise ValueError("Non-loopback binding requires username and password")
        create_interface(server_name=args.host, server_port=args.port, auth=auth)
    finally:
        # Ensure cleanup even if Gradio encounters an error
        cleanup_resources()
        if cleanup_registered:
            atexit.unregister(cleanup_resources)

if __name__ == "__main__":
    main()
