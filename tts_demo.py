import torch
from typing import Optional, Tuple, List
from models import build_model, load_voice, generate_speech, list_available_voices
import argparse
from tqdm.auto import tqdm
import soundfile as sf
from pathlib import Path
import numpy as np

# Constants
SAMPLE_RATE = 24000
DEFAULT_MODEL_PATH = 'kokoro-v1_0.pth'
DEFAULT_OUTPUT_FILE = 'output.wav'
DEFAULT_LANGUAGE = 'a'  # Now documented: 'a' for American English, 'b' for British English
DEFAULT_TEXT = "Hello, welcome to this text-to-speech test."

# Configure tqdm for better Windows console support
tqdm.monitor_interval = 0  # Disable monitor thread to prevent encoding issues

def load_and_validate_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load the requested voice.
    
    Args:
        voice_name: Name of the voice to load
        device: Device to load the voice on ('cuda' or 'cpu')
        
    Returns:
        Loaded voice tensor
    """
    return load_voice(voice_name, device)

def main() -> None:
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Kokoro TTS Demo')
        parser.add_argument('--text', type=str, help='Text to synthesize (optional)')
        parser.add_argument('--voice', type=str, default='af_bella', help='Voice to use (default: af_bella)')
        parser.add_argument('--list-voices', action='store_true', help='List all available voices')
        parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help=f'Path to model file (default: {DEFAULT_MODEL_PATH})')
        parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_FILE, help=f'Output WAV file (default: {DEFAULT_OUTPUT_FILE})')
        parser.add_argument('--lang', type=str, default=DEFAULT_LANGUAGE, help=f'Language code (default: {DEFAULT_LANGUAGE})')
        parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier (default: 1.0)')
        args = parser.parse_args()

        if args.list_voices:
            voices = list_available_voices()
            print("\nAvailable voices:")
            for voice in voices:
                print(f"- {voice}")
            return

        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Build model and load voice with progress indication
        print("\nLoading model...")
        with tqdm(total=1, desc="Building model") as pbar:
            model = build_model(args.model, device)
            pbar.update(1)
        
        # Get text input
        if args.text:
            text = args.text
        else:
            print("\nEnter the text you want to convert to speech (or press Enter for default text):")
            text = input("> ").strip()
            if not text:
                text = DEFAULT_TEXT
        
        print(f"\nGenerating speech for: '{text}'")
        
        # Use the generator API
        all_audio = []
        generator = model(text, voice=args.voice, speed=args.speed, split_pattern=r'\n+')
        
        with tqdm(desc="Generating speech") as pbar:
            for gs, ps, audio in generator:
                if audio is not None:
                    # Convert numpy array to tensor if needed
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    all_audio.append(audio)
                    try:
                        print(f"\nGenerated segment: {gs}")
                        print(f"Phonemes: {ps}")
                    except UnicodeEncodeError:
                        print("\nGenerated segment: [Unicode display error]")
                        print("Phonemes: [Unicode display error]")
                    pbar.update(1)
        
        # Combine all audio segments
        if all_audio:
            final_audio = torch.cat(all_audio, dim=0)
            try:
                output_path = Path(args.output)
                sf.write(output_path, final_audio, SAMPLE_RATE)
                print(f"\nAudio saved to {output_path.absolute()}")
            except Exception as e:
                print(f"Error saving output: {e}")
                print("Audio generation was successful, but saving failed.")
        else:
            print("Error: Failed to generate audio")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 