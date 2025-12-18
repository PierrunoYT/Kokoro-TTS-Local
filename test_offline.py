#!/usr/bin/env python3
"""
Test script for verifying offline mode functionality of Kokoro-TTS-Local
"""
import os
import sys
from pathlib import Path
import torch

# Constants
REQUIRED_FILES = {
    'model': 'kokoro-v1_0.pth',
    'config': 'config.json',
    'voices_dir': 'voices'
}

DEFAULT_TEST_TEXT = "Hello, this is a test of offline mode."
DEFAULT_VOICE = "af_bella"
TEST_OUTPUT = "test_offline_output.wav"

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_status(item: str, status: bool, details: str = ""):
    """Print a status line with check or cross mark"""
    mark = "[PASS]" if status else "[FAIL]"
    print(f"  {mark} {item}")
    if details:
        print(f"      {details}")

def check_offline_mode() -> bool:
    """Check if offline mode is enabled"""
    print_header("Checking Offline Mode Configuration")
    
    hf_offline = os.environ.get("HF_HUB_OFFLINE", "0")
    transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "0")
    
    offline_enabled = hf_offline == "1" or transformers_offline == "1"
    
    print_status("HF_HUB_OFFLINE", hf_offline == "1", f"Value: {hf_offline}")
    print_status("TRANSFORMERS_OFFLINE", transformers_offline == "1", f"Value: {transformers_offline}")
    print_status("Offline Mode Status", offline_enabled, 
                "Enabled" if offline_enabled else "Not enabled - will attempt network access")
    
    return offline_enabled

def check_required_files() -> dict:
    """Check if all required files exist"""
    print_header("Checking Required Files")
    
    results = {}
    
    # Check model file
    model_path = Path(REQUIRED_FILES['model']).resolve()
    model_exists = model_path.exists()
    results['model'] = model_exists
    print_status("Model file", model_exists, str(model_path))
    
    # Check config file
    config_path = Path(REQUIRED_FILES['config']).resolve()
    config_exists = config_path.exists()
    results['config'] = config_exists
    print_status("Config file", config_exists, str(config_path))
    
    # Check voices directory
    voices_dir = Path(REQUIRED_FILES['voices_dir']).resolve()
    voices_exists = voices_dir.exists() and voices_dir.is_dir()
    results['voices_dir'] = voices_exists
    print_status("Voices directory", voices_exists, str(voices_dir))
    
    # Check for voice files
    if voices_exists:
        voice_files = list(voices_dir.glob("*.pt"))
        results['voice_count'] = len(voice_files)
        has_voices = len(voice_files) > 0
        print_status("Voice files", has_voices, 
                    f"Found {len(voice_files)} voice file(s)")
        
        # Check for default voice
        default_voice_path = voices_dir / f"{DEFAULT_VOICE}.pt"
        default_voice_exists = default_voice_path.exists()
        results['default_voice'] = default_voice_exists
        print_status(f"Default voice ({DEFAULT_VOICE})", default_voice_exists,
                    str(default_voice_path) if default_voice_exists else "Not found")
    else:
        results['voice_count'] = 0
        results['default_voice'] = False
        print_status("Voice files", False, "Voices directory not found")
    
    return results

def check_dependencies() -> dict:
    """Check if required Python packages are installed"""
    print_header("Checking Dependencies")
    
    results = {}
    required_packages = {
        'torch': 'PyTorch',
        'kokoro': 'Kokoro TTS',
        'soundfile': 'SoundFile',
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            results[package] = True
            print_status(name, True, f"Package '{package}' installed")
        except ImportError:
            results[package] = False
            print_status(name, False, f"Package '{package}' not found")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    results['cuda'] = cuda_available
    print_status("CUDA Support", cuda_available, 
                "GPU acceleration available" if cuda_available else "Using CPU")
    
    return results

def test_model_initialization() -> bool:
    """Test if model can be initialized"""
    print_header("Testing Model Initialization")
    
    try:
        from models import build_model
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Using device: {device}")
        
        model_path = Path(REQUIRED_FILES['model']).resolve()
        print(f"  Model path: {model_path}")
        
        # Build model
        print("  Initializing model...")
        model = build_model(str(model_path), device)
        
        if model is None:
            print_status("Model initialization", False, "Model returned None")
            return False
        
        print_status("Model initialization", True, "Model loaded successfully")
        return True
        
    except Exception as e:
        print_status("Model initialization", False, f"Error: {type(e).__name__}: {str(e)}")
        return False

def test_voice_listing() -> bool:
    """Test if voices can be listed"""
    print_header("Testing Voice Listing")
    
    try:
        from models import list_available_voices
        
        voices = list_available_voices()
        
        if not voices:
            print_status("Voice listing", False, "No voices found")
            return False
        
        print_status("Voice listing", True, f"Found {len(voices)} voice(s)")
        print("\n  Available voices:")
        for i, voice in enumerate(voices[:10], 1):  # Show first 10
            print(f"    {i}. {voice}")
        if len(voices) > 10:
            print(f"    ... and {len(voices) - 10} more")
        
        return True
        
    except Exception as e:
        print_status("Voice listing", False, f"Error: {type(e).__name__}: {str(e)}")
        return False

def test_speech_generation() -> bool:
    """Test if speech can be generated"""
    print_header("Testing Speech Generation")
    
    try:
        from models import build_model, list_available_voices
        import soundfile as sf
        import numpy as np
        
        # Get available voices
        voices = list_available_voices()
        if not voices:
            print_status("Speech generation", False, "No voices available")
            return False
        
        # Use default voice if available, otherwise use first voice
        voice = DEFAULT_VOICE if DEFAULT_VOICE in voices else voices[0]
        print(f"  Using voice: {voice}")
        print(f"  Test text: '{DEFAULT_TEST_TEXT}'")
        
        # Initialize model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = build_model(str(Path(REQUIRED_FILES['model']).resolve()), device)
        
        if model is None:
            print_status("Speech generation", False, "Failed to load model")
            return False
        
        # Generate speech
        print("  Generating speech...")
        voice_path = Path("voices").resolve() / f"{voice}.pt"
        
        all_audio = []
        generator = model(DEFAULT_TEST_TEXT, voice=str(voice_path), speed=1.0, split_pattern=r'\n+')
        
        for gs, ps, audio in generator:
            if audio is not None:
                audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio).float()
                all_audio.append(audio_tensor)
        
        if not all_audio:
            print_status("Speech generation", False, "No audio generated")
            return False
        
        # Concatenate audio segments
        if len(all_audio) == 1:
            final_audio = all_audio[0]
        else:
            final_audio = torch.cat(all_audio, dim=0)
        
        # Save test output
        output_path = Path(TEST_OUTPUT).resolve()
        sf.write(str(output_path), final_audio.numpy(), 24000)
        
        if not output_path.exists():
            print_status("Speech generation", False, "Failed to save output file")
            return False
        
        file_size = output_path.stat().st_size
        print_status("Speech generation", True, 
                    f"Generated {file_size:,} bytes to {output_path.name}")
        
        return True
        
    except Exception as e:
        print_status("Speech generation", False, f"Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def cleanup():
    """Clean up test files"""
    try:
        output_path = Path(TEST_OUTPUT)
        if output_path.exists():
            output_path.unlink()
            print(f"\n  Cleaned up test file: {TEST_OUTPUT}")
    except Exception as e:
        print(f"\n  Warning: Could not clean up test file: {e}")

def main():
    """Run all offline mode tests"""
    print("\n" + "=" * 60)
    print("  KOKORO TTS OFFLINE MODE TEST")
    print("=" * 60)
    
    # Track test results
    tests_passed = 0
    tests_failed = 0
    
    # Check offline mode
    offline_enabled = check_offline_mode()
    if not offline_enabled:
        print("\n[WARNING] Offline mode is not enabled!")
        print("  To enable offline mode, set the environment variable:")
        print("    Linux/macOS:  export HF_HUB_OFFLINE=1")
        print("    Windows PS:   $env:HF_HUB_OFFLINE=\"1\"")
        print("    Windows CMD:  set HF_HUB_OFFLINE=1")
        print("\n  Continuing tests (may require network access)...\n")
    
    # Check required files
    file_results = check_required_files()
    all_files_present = all([
        file_results.get('model', False),
        file_results.get('config', False),
        file_results.get('voices_dir', False),
        file_results.get('voice_count', 0) > 0
    ])
    
    if not all_files_present:
        print("\n[PREREQUISITE FAILED] Required files are missing")
        print("  Please run the application with network access first to download:")
        print("    - Model file (kokoro-v1_0.pth)")
        print("    - Config file (config.json)")
        print("    - At least one voice file in voices/ directory")
        print("\n  Run: python tts_demo.py")
        print("       or: python gradio_interface.py")
        return 1
    
    # Check dependencies
    dep_results = check_dependencies()
    all_deps_present = all([
        dep_results.get('torch', False),
        dep_results.get('kokoro', False),
        dep_results.get('soundfile', False),
        dep_results.get('numpy', False),
        dep_results.get('tqdm', False)
    ])
    
    if not all_deps_present:
        print("\n[PREREQUISITE FAILED] Required dependencies are missing")
        print("  Please install required packages:")
        print("    pip install -r requirements.txt")
        return 1
    
    # Test model initialization
    if test_model_initialization():
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test voice listing
    if test_voice_listing():
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Test speech generation
    if test_speech_generation():
        tests_passed += 1
    else:
        tests_failed += 1
    
    # Print summary
    print_header("Test Summary")
    total_tests = tests_passed + tests_failed
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {tests_passed}")
    print(f"  Failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n[SUCCESS] All tests passed!")
        print("  Your offline setup is working correctly.")
        cleanup()
        return 0
    else:
        print(f"\n[FAILURE] {tests_failed} test(s) failed")
        print("  Please review the errors above and fix any issues.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n\n[UNEXPECTED ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)

