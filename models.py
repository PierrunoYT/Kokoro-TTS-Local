"""Models module for Kokoro TTS Local"""
from typing import Optional, Tuple, List
import torch
from kokoro import KPipeline
import os
import json
import codecs
from pathlib import Path
import numpy as np
import shutil
import threading
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings from pre-trained model
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

# Set environment variables for proper encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Check if offline mode is enabled via environment variable
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
if OFFLINE_MODE:
    logger.info("Running in OFFLINE mode - will only use locally cached files")
    # Ensure the environment variable is set for the kokoro library as well
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Setup for safer cleanup
import atexit
import signal
import sys

# Track whether patches have been applied
_patches_applied = {
    'json_load': False
}

class EnhancedKPipeline(KPipeline):
    """Enhanced KPipeline with improved voice loading and error handling"""
    
    def __init__(self, lang_code: str = 'a', model: bool = True):
        super().__init__(lang_code=lang_code, model=model)
        self.device = 'cpu'  # Default device
        if not hasattr(self, 'voices'):
            self.voices = {}
    
    def load_voice(self, voice_path: str) -> torch.Tensor:
        """Load voice model with improved error handling and path validation"""
        voice_path = Path(voice_path).resolve()
        
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        
        voice_name = voice_path.stem
        
        try:
            logger.info(f"Loading voice: {voice_name} from {voice_path}")
            voice_model = torch.load(str(voice_path), weights_only=True, map_location='cpu')
            
            if voice_model is None:
                raise ValueError(f"Failed to load voice model from {voice_path}")
            
            # Move model to device and store in voices dictionary
            self.voices[voice_name] = voice_model.to(self.device)
            logger.info(f"Successfully loaded voice: {voice_name}")
            return self.voices[voice_name]
            
        except Exception as e:
            logger.error(f"Error loading voice {voice_name}: {e}")
            raise

def _cleanup_patches() -> None:
    """Restore original functions that were patched"""
    try:
        if _patches_applied['json_load'] and _original_json_load is not None:
            restore_json_load()
            _patches_applied['json_load'] = False
            logger.info("Restored original json.load function")
    except Exception as e:
        logger.warning(f"Error restoring json.load: {e}")

# Register cleanup for normal exit
atexit.register(_cleanup_patches)

# Register cleanup for signals
for sig in [signal.SIGINT, signal.SIGTERM]:
    try:
        signal.signal(sig, lambda signum, frame: (
            logger.info(f"Received signal {signum}, cleaning up..."),
            _cleanup_patches(),
            sys.exit(1)
        ))
    except (ValueError, AttributeError) as e:
        # Some signals might not be available on all platforms
        logger.warning(f"Could not register signal handler: {e}")

# List of available voice files (54 voices across 8 languages)
VOICE_FILES = [
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

# Language code mapping for different languages
LANGUAGE_CODES = {
    'a': 'American English',
    'b': 'British English',
    'j': 'Japanese',
    'z': 'Mandarin Chinese',
    'e': 'Spanish',
    'f': 'French',
    'h': 'Hindi',
    'i': 'Italian',
    'p': 'Brazilian Portuguese'
}


def patch_json_load() -> None:
    """Patch json.load to handle UTF-8 encoded files with special characters"""
    global _patches_applied, _original_json_load
    original_load = json.load
    _original_json_load = original_load  # Store for restoration

    def custom_load(fp, *args, **kwargs):
        try:
            # Try reading with UTF-8 encoding
            if hasattr(fp, 'buffer'):
                content = fp.buffer.read().decode('utf-8')
            else:
                content = fp.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise
        except UnicodeDecodeError:
            # If UTF-8 fails, try with utf-8-sig for files with BOM
            fp.seek(0)
            content = fp.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8-sig', errors='replace')
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise

    json.load = custom_load
    _patches_applied['json_load'] = True
    return original_load  # Return original for restoration

# Store the original load function for potential restoration
_original_json_load = None

def restore_json_load() -> None:
    """Restore the original json.load function"""
    global _original_json_load, _patches_applied
    if _original_json_load is not None and _patches_applied['json_load']:
        json.load = _original_json_load
        _original_json_load = None
        _patches_applied['json_load'] = False

def load_config(config_path: str) -> dict:
    """Load configuration file with proper encoding handling"""
    config_path = Path(config_path).resolve()
    
    try:
        with codecs.open(str(config_path), 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Fallback to utf-8-sig if regular utf-8 fails
        with codecs.open(str(config_path), 'r', encoding='utf-8-sig') as f:
            return json.load(f)

# Initialize espeak-ng
phonemizer_available = False  # Global flag to track if phonemizer is working
current_phonemizer_lang = None  # Track current phonemizer language

def initialize_phonemizer(language: str = 'en-us') -> bool:
    """Initialize phonemizer for a specific language
    
    Args:
        language: Language code for phonemizer (e.g., 'en-us', 'zh')
        
    Returns:
        True if initialization successful, False otherwise
    """
    global phonemizer_available, current_phonemizer_lang
    
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        from phonemizer import phonemize
        import espeakng_loader

        # Make library available first
        library_path = espeakng_loader.get_library_path()
        data_path = espeakng_loader.get_data_path()
        espeakng_loader.make_library_available()

        # Set up espeak-ng paths
        EspeakWrapper.library_path = library_path
        EspeakWrapper.data_path = data_path

        # Verify espeak-ng is working with specified language
        try:
            test_text = 'test' if language in ['en-us', 'en-gb'] else '测试'
            test_phonemes = phonemize(test_text, language=language)
            if test_phonemes:
                phonemizer_available = True
                current_phonemizer_lang = language
                logger.info(f"Phonemizer successfully initialized for language: {language}")
                return True
            else:
                logger.warning("Phonemization returned empty result")
                return False
        except Exception as e:
            # Continue without espeak functionality - be more specific about error types
            if "espeak" in str(e).lower():
                logger.warning(f"eSpeak not found: {e}")
            else:
                logger.warning(f"Phonemizer initialization error: {e}")
            return False

    except ImportError as e:
        logger.warning(f"Phonemizer packages not installed: {e}")
        logger.info("If you want phoneme visualization, manually install required packages:")
        logger.info("pip install espeakng-loader phonemizer-fork")
        return False

# Initialize default English phonemizer
try:
    initialize_phonemizer('en-us')
except Exception as e:
    logger.warning(f"Could not initialize default phonemizer: {e}")

# Initialize pipeline globally with thread safety
_pipeline = None
_pipeline_lock = threading.RLock()  # Reentrant lock for thread safety
_voice_cache_lock = threading.RLock()  # Separate lock for voice cache operations
_download_lock = threading.Lock()  # Lock for download operations

def download_voice_files(voice_files: Optional[List[str]] = None, repo_version: str = "main", required_count: int = 1) -> List[str]:
    """Download voice files from Hugging Face with enhanced progress tracking.

    Args:
        voice_files: Optional list of voice files to download. If None, download all VOICE_FILES.
        repo_version: Version/tag of the repository to use (default: "main")
        required_count: Minimum number of voices required (default: 1)

    Returns:
        List of successfully downloaded voice files

    Raises:
        ValueError: If fewer than required_count voices could be downloaded
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import hashlib
    import time
    
    # Use absolute path for voices directory
    voices_dir = Path("voices").resolve()
    voices_dir.mkdir(exist_ok=True)

    # Import here to avoid startup dependency
    from huggingface_hub import hf_hub_download
    downloaded_voices = []
    failed_voices = []

    # If specific voice files are requested, use those. Otherwise use all.
    files_to_download = voice_files if voice_files is not None else VOICE_FILES
    total_files = len(files_to_download)

    logger.info(f"Downloading voice files... ({total_files} total files)")

    # Check for existing voice files first
    existing_files = []
    for voice_file in files_to_download:
        voice_path = voices_dir / voice_file
        if voice_path.exists() and voice_path.stat().st_size > 0:
            logger.info(f"Voice file {voice_file} already exists")
            downloaded_voices.append(voice_file)
            existing_files.append(voice_file)

    # Remove existing files from the download list
    files_to_download = [f for f in files_to_download if f not in existing_files]
    if not files_to_download and downloaded_voices:
        logger.info(f"All required voice files already exist ({len(downloaded_voices)} files)")
        return downloaded_voices
    
    # In offline mode, only use existing files
    if OFFLINE_MODE:
        if not downloaded_voices:
            error_msg = "No voice files found locally and running in OFFLINE mode. Please download voice files first with network connection."
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif len(downloaded_voices) < required_count:
            error_msg = f"Only {len(downloaded_voices)} voice files found locally, but {required_count} were required. Running in OFFLINE mode."
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            logger.info(f"Using {len(downloaded_voices)} locally cached voice files (OFFLINE mode)")
            return downloaded_voices

    def download_single_voice(voice_file: str) -> Tuple[str, bool, str]:
        """Download a single voice file with retry logic"""
        retry_count = 3
        retry_delay = 2
        
        for attempt in range(retry_count):
            try:
                # Download with exponential backoff
                if attempt > 0:
                    delay = retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
                
                # Download directly to voices directory
                import tempfile
                temp_dir = tempfile.mkdtemp()
                try:
                    downloaded_path = hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename=f"voices/{voice_file}",
                        local_dir=temp_dir,
                        force_download=False,
                        revision=repo_version,
                        local_files_only=OFFLINE_MODE
                    )
                    
                    # Verify file integrity with basic size check
                    if Path(downloaded_path).stat().st_size == 0:
                        raise ValueError(f"Downloaded file {voice_file} has zero size")
                    
                    # Move to final location
                    voice_path = voices_dir / voice_file
                    shutil.move(downloaded_path, str(voice_path))
                    
                    return voice_file, True, f"Successfully downloaded {voice_file}"
                finally:
                    # Clean up temporary directory
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                    
            except Exception as e:
                error_msg = f"Failed to download {voice_file} (attempt {attempt+1}/{retry_count}): {e}"
                if attempt == retry_count - 1:
                    return voice_file, False, error_msg
                logger.warning(error_msg)
        
        return voice_file, False, f"Failed all {retry_count} attempts to download {voice_file}"

    # Download files with progress bar and parallel processing
    if files_to_download:
        logger.info(f"Downloading {len(files_to_download)} missing voice files...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:  # Limit concurrent downloads
            # Submit all download tasks
            future_to_voice = {
                executor.submit(download_single_voice, voice_file): voice_file
                for voice_file in files_to_download
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(files_to_download), desc="Downloading voices") as pbar:
                for future in as_completed(future_to_voice):
                    voice_file, success, message = future.result()
                    
                    if success:
                        downloaded_voices.append(voice_file)
                        logger.info(message)
                    else:
                        failed_voices.append(voice_file)
                        logger.error(message)
                    
                    pbar.update(1)

    # Report results
    if failed_voices:
        logger.warning(f"Failed to download {len(failed_voices)} voice files: {', '.join(failed_voices)}")

    if not downloaded_voices:
        error_msg = "No voice files could be downloaded. Please check your internet connection."
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif len(downloaded_voices) < required_count:
        error_msg = f"Only {len(downloaded_voices)} voice files could be downloaded, but {required_count} were required."
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        logger.info(f"Successfully processed {len(downloaded_voices)} voice files")

    return downloaded_voices

def build_model(model_path: str, device: str, repo_version: str = "main", lang_code: str = 'a') -> EnhancedKPipeline:
    """Build and return the Enhanced Kokoro pipeline with proper encoding configuration

    Args:
        model_path: Path to the model file or None to use default
        device: Device to use ('cuda' or 'cpu')
        repo_version: Version/tag of the repository to use (default: "main")
        lang_code: Language code for the model (default: 'a' for American English, 'z' for Chinese)

    Returns:
        Initialized EnhancedKPipeline instance
    """
    global _pipeline, _pipeline_lock

    # Use a lock for thread safety
    with _pipeline_lock:
        # Don't reuse pipeline if language code is different
        # (each language may need different configuration)
        if _pipeline is not None and hasattr(_pipeline, 'lang_code') and _pipeline.lang_code == lang_code:
            return _pipeline

        try:
            # Patch json loading before initializing pipeline
            patch_json_load()

            # Determine if this is a Chinese model
            is_chinese_model = lang_code == 'z' or (model_path and 'zh' in str(model_path).lower())

            # Download model if it doesn't exist
            if model_path is None:
                model_path = 'kokoro-82M-v1.1_zh.pth' if is_chinese_model else 'kokoro-v1_0.pth'

            model_path = os.path.abspath(model_path)
            if not os.path.exists(model_path):
                if OFFLINE_MODE:
                    error_msg = f"Model file {model_path} not found and running in OFFLINE mode. Please download the model first with network connection."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info(f"Downloading model file {model_path}...")
                try:
                    from huggingface_hub import hf_hub_download
                    
                    # Determine filename for download
                    filename = 'kokoro-82M-v1.1_zh.pth' if is_chinese_model else 'kokoro-v1_0.pth'
                    
                    model_path = hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename=filename,
                        local_dir=".",
                        force_download=False,
                        revision=repo_version,
                        local_files_only=OFFLINE_MODE
                    )
                    logger.info(f"Model downloaded to {model_path}")
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
                    raise ValueError(f"Could not download model: {e}") from e

            # Download config if it doesn't exist
            config_path = os.path.abspath("config.json")
            if not os.path.exists(config_path):
                if OFFLINE_MODE:
                    error_msg = f"Config file {config_path} not found and running in OFFLINE mode. Please download the config first with network connection."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info("Downloading config file...")
                try:
                    from huggingface_hub import hf_hub_download
                    config_path = hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename="config.json",
                        local_dir=".",
                        force_download=False,
                        revision=repo_version,
                        local_files_only=OFFLINE_MODE
                    )
                    logger.info(f"Config downloaded to {config_path}")
                except Exception as e:
                    logger.error(f"Error downloading config: {e}")
                    raise ValueError(f"Could not download config: {e}") from e

            # Initialize phonemizer for the appropriate language
            if is_chinese_model:
                logger.info("Initializing phonemizer for Chinese...")
                try:
                    initialize_phonemizer('zh')
                except Exception as e:
                    logger.warning(f"Could not initialize Chinese phonemizer: {e}")
            else:
                logger.info("Initializing phonemizer for English...")
                try:
                    initialize_phonemizer('en-us')
                except Exception as e:
                    logger.warning(f"Could not initialize English phonemizer: {e}")

            # Download voice files - require at least one voice
            try:
                downloaded_voices = download_voice_files(repo_version=repo_version, required_count=1)
            except ValueError as e:
                logger.error(f"Error: Voice files download failed: {e}")
                raise ValueError("Voice files download failed") from e

            # Validate language code
            supported_codes = list(LANGUAGE_CODES.keys())
            if lang_code not in supported_codes:
                logger.warning(f"Unsupported language code '{lang_code}'. Using 'a' (American English).")
                logger.info(f"Supported language codes: {', '.join(supported_codes)}")
                lang_code = 'a'

            # Initialize pipeline with validated language code
            pipeline_instance = EnhancedKPipeline(lang_code=lang_code)
            if pipeline_instance is None:
                raise ValueError("Failed to initialize EnhancedKPipeline - pipeline is None")

            # Store language code and device
            pipeline_instance.lang_code = lang_code
            pipeline_instance.device = device

            # Try to load the first available voice with improved error handling
            voice_loaded = False
            for voice_file in downloaded_voices:
                voice_path = os.path.abspath(os.path.join("voices", voice_file))
                if os.path.exists(voice_path):
                    try:
                        pipeline_instance.load_voice(voice_path)
                        logger.info(f"Successfully loaded voice: {voice_file}")
                        voice_loaded = True
                        break  # Successfully loaded a voice
                    except Exception as e:
                        logger.warning(f"Warning: Failed to load voice {voice_file}: {e}")
                        continue

            if not voice_loaded:
                logger.warning("Warning: Could not load any voice models")

            # Set the global _pipeline only after successful initialization
            _pipeline = pipeline_instance

        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            # Restore original json.load on error
            restore_json_load()
            raise

        return _pipeline

def list_available_voices() -> List[str]:
    """List all available voice models"""
    # Always use absolute path for consistency
    voices_dir = Path(os.path.abspath("voices"))

    # Create voices directory if it doesn't exist
    if not voices_dir.exists():
        print(f"Creating voices directory at {voices_dir}")
        voices_dir.mkdir(exist_ok=True)
        return []

    # Get all .pt files in the voices directory
    voice_files = list(voices_dir.glob("*.pt"))

    # If we found voice files, return them
    if voice_files:
        return [f.stem for f in sorted(voice_files, key=lambda f: f.stem.lower())]

    # If no voice files in standard location, check if we need to do a one-time migration
    # This is legacy support for older installations
    alt_voices_path = Path(".") / "voices"
    if alt_voices_path.exists() and alt_voices_path.is_dir() and alt_voices_path != voices_dir:
        print(f"Checking alternative voice location: {alt_voices_path.absolute()}")
        alt_voice_files = list(alt_voices_path.glob("*.pt"))

        if alt_voice_files:
            print(f"Found {len(alt_voice_files)} voice files in alternate location")
            print("Moving files to the standard voices directory...")

            # Process files in a batch for efficiency
            files_moved = 0
            for voice_file in alt_voice_files:
                target_path = voices_dir / voice_file.name
                if not target_path.exists():
                    try:
                        # Use copy2 to preserve metadata, then remove original if successful
                        shutil.copy2(str(voice_file), str(target_path))
                        files_moved += 1
                    except (OSError, IOError) as e:
                        print(f"Error copying {voice_file.name}: {e}")

            if files_moved > 0:
                print(f"Successfully moved {files_moved} voice files")
                return [f.stem for f in sorted(voices_dir.glob("*.pt"), key=lambda f: f.stem.lower())]

    print("No voice files found. Please run the application again to download voices.")
    return []

def get_language_code_from_voice(voice_name: str) -> str:
    """Get the appropriate language code from a voice name

    Args:
        voice_name: Name of the voice (e.g., 'af_bella', 'jf_alpha')

    Returns:
        Language code for the voice
    """
    # Extract prefix from voice name
    prefix = voice_name[:2] if len(voice_name) >= 2 else 'af'

    # Map voice prefixes to language codes
    prefix_to_lang = {
        'af': 'a', 'am': 'a',  # American English
        'bf': 'b', 'bm': 'b',  # British English
        'jf': 'j', 'jm': 'j',  # Japanese
        'zf': 'z', 'zm': 'z',  # Mandarin Chinese
        'ef': 'e', 'em': 'e',  # Spanish
        'ff': 'f', 'fm': 'f',  # French
        'hf': 'h', 'hm': 'h',  # Hindi
        'if': 'i', 'im': 'i',  # Italian
        'pf': 'p', 'pm': 'p',  # Brazilian Portuguese
    }

    return prefix_to_lang.get(prefix, 'a')  # Default to American English

def load_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load a voice model in a thread-safe manner

    Args:
        voice_name: Name of the voice to load (with or without .pt extension)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Loaded voice model tensor

    Raises:
        ValueError: If voice file not found or loading fails
    """
    pipeline = build_model(None, device)

    # Format voice path correctly - strip .pt if it was included
    voice_name = voice_name.replace('.pt', '')
    voice_path = Path("voices").resolve() / f"{voice_name}.pt"

    if not voice_path.exists():
        raise ValueError(f"Voice file not found: {voice_path}")

    # Use a lock to ensure thread safety when loading voices
    with _pipeline_lock:
        # Check if voice is already loaded
        if voice_name in pipeline.voices:
            return pipeline.voices[voice_name]

        # Load voice if not already loaded
        return pipeline.load_voice(str(voice_path))

def generate_speech(
    model: EnhancedKPipeline,
    text: str,
    voice: str,
    lang: str = 'a',
    device: str = 'cpu',
    speed: float = 1.0
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Generate speech using the Kokoro pipeline in a thread-safe manner

    Args:
        model: EnhancedKPipeline instance
        text: Text to synthesize
        voice: Voice name (e.g. 'af_bella')
        lang: Language code ('a' for American English, 'b' for British English)
        device: Device to use ('cuda' or 'cpu')
        speed: Speech speed multiplier (default: 1.0)

    Returns:
        Tuple of (audio tensor, phonemes string) or (None, None) on error
    """
    global _pipeline_lock

    try:
        if model is None:
            raise ValueError("Model is None - pipeline not properly initialized")

        # Format voice name and path
        voice_name = voice.replace('.pt', '')
        voice_path = Path("voices").resolve() / f"{voice_name}.pt"

        # Check if voice file exists
        if not voice_path.exists():
            raise ValueError(f"Voice file not found: {voice_path}")

        # Thread-safe initialization of model properties and voice loading
        with _pipeline_lock:
            # Ensure device is set
            model.device = device

            # Ensure voice is loaded before generating
            if voice_name not in model.voices:
                logger.info(f"Loading voice {voice_name}...")
                try:
                    model.load_voice(str(voice_path))
                    if voice_name not in model.voices:
                        raise ValueError("Voice load succeeded but voice not in model.voices dictionary")
                except Exception as e:
                    raise ValueError(f"Failed to load voice {voice_name}: {e}")

        # Generate speech (outside the lock for better concurrency)
        logger.info(f"Generating speech with device: {model.device}")
        generator = model(
            text,
            voice=str(voice_path),
            speed=speed,
            split_pattern=r'\n+'
        )

        # Get first generated segment and convert numpy array to tensor if needed
        for gs, ps, audio in generator:
            if audio is not None:
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                return audio, ps

        return None, None
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, AttributeError, TypeError) as e:
        logger.error(f"Error generating speech: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during speech generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None