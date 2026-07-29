"""Models module for Kokoro TTS Local"""
from typing import Optional, Tuple, List
import torch
from kokoro import KModel, KPipeline
import os
import json
import re
import contextlib
import hashlib
from pathlib import Path
import numpy as np
import shutil
import tempfile
import threading
import warnings
import logging
from .paths import get_base_dir, get_voices_dir, get_model_dir, get_config_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safe voice name regex (alphanumeric, underscore, dash only)
_VOICE_NAME_RE = re.compile(r'^[a-zA-Z0-9_-]+$')


def get_safe_voice_path(voice_name: str) -> Path:
    """Return a validated, canonical voice file path.

    Raises ValueError if the voice name contains unsafe characters or the
    resolved path escapes the voices directory (path traversal).
    """
    if not isinstance(voice_name, str):
        raise ValueError("Voice name must be a string")

    voice_name = voice_name.strip().removesuffix('.pt')

    if not _VOICE_NAME_RE.match(voice_name):
        raise ValueError(f"Invalid voice name: {voice_name!r}")

    voices_dir = get_voices_dir()
    voice_path = (voices_dir / f"{voice_name}.pt").resolve()

    # Ensure the resolved path is still inside the voices directory
    try:
        voice_path.relative_to(voices_dir)
    except ValueError as exc:
        raise ValueError(f"Voice path escapes voices directory: {voice_path}") from exc
    return voice_path


def safe_json_load(fp, **kwargs):
    """Load JSON from a file-like object with UTF-8 / BOM handling.

    Unlike monkey-patching json.load, this is a standalone helper that
    does not mutate global state. Accepts the same keyword-only options
    as ``json.load`` (cls, object_hook, parse_float, ...).
    """
    if hasattr(fp, 'seek'):
        fp.seek(0)

    if hasattr(fp, 'buffer'):
        # Use the raw byte stream directly so utf-8-sig strips any BOM
        # reliably, regardless of the platform default text encoding.
        content = fp.buffer.read().decode('utf-8-sig')
    else:
        content = fp.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8-sig')
        else:
            content = content.lstrip('\ufeff')

    try:
        return json.loads(content, **kwargs)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        raise


# Suppress warnings from pre-trained model
warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

# Console encoding is handled by console.enable_utf8_console(); setting
# PYTHONIOENCODING here would be a no-op, as the interpreter reads it only at
# startup, long before this module is imported.
# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Check if offline mode is enabled via environment variable
OFFLINE_MODE = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
if OFFLINE_MODE:
    logger.info("Running in OFFLINE mode - will only use locally cached files")
    # Ensure the environment variable is set for the kokoro library as well
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

class EnhancedKPipeline(KPipeline):
    """Enhanced KPipeline with improved voice loading and error handling"""

    def __init__(
        self,
        lang_code: str = 'a',
        model=True,
        repo_id: Optional[str] = None,
        device: str = 'cpu',
        family_lock: Optional[threading.RLock] = None,
    ):
        super().__init__(
            lang_code=lang_code,
            model=model,
            repo_id=repo_id,
            device=device
        )
        self.device = device
        self._family_lock = family_lock or threading.RLock()
        self._published = False
        self._closed = False
        if not hasattr(self, 'voices'):
            self.voices = {}

    def load_voice(self, voice_path: str) -> torch.Tensor:
        """Load voice model with improved error handling and path validation"""
        with self._family_lock:
            return self._load_voice_locked(voice_path)

    def _load_voice_locked(self, voice_path: str) -> torch.Tensor:
        voice_path = Path(voice_path).resolve()

        voice_name = voice_path.stem
        if voice_name in self.voices:
            return self.voices[voice_name]

        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

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

    def iter_speech(self, *args, **kwargs):
        """Iterate inference while exclusively owning this model family."""
        voice = kwargs.get('voice')
        if voice is None:
            raise ValueError("voice is required")
        voice_lang = get_language_code_from_voice(Path(voice).stem)
        if voice_lang != self.lang_code:
            raise ValueError(
                f"Voice language {voice_lang!r} does not match pipeline language {self.lang_code!r}"
            )

        def guarded():
            with self._family_lock:
                with _registry_lock:
                    unavailable = self._closed or _shutting_down
                if unavailable:
                    raise RuntimeError("Pipeline is closed")
                # Synthesis is pure inference: without this every segment
                # builds an autograd graph and retains its activations, which
                # is the straightest path to CUDA OOM under concurrent load.
                #
                # no_grad rather than inference_mode: the yielded tensors
                # outlive this scope (callers concatenate them afterwards), and
                # inference tensors carry escape restrictions that no_grad
                # tensors do not. Grad mode is thread-local and this generator
                # suspends at each yield, so it stays disabled in the consumer's
                # loop body too — harmless, since the family lock makes this
                # generator the only work running on this model, and the
                # consumers only convert and collect tensors.
                with torch.no_grad():
                    yield from super(EnhancedKPipeline, self).__call__(*args, **kwargs)
        return guarded()

    def __call__(self, *args, **kwargs):
        return self.iter_speech(*args, **kwargs)

    def __setattr__(self, name, value):
        if name in {'device', 'lang_code'} and getattr(self, '_published', False):
            if getattr(self, name, value) != value:
                raise AttributeError(f"{name} is immutable after pipeline publication")
        super().__setattr__(name, value)

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

VOICE_PREFIX_TO_LANGUAGE_CODE = {
    'af': 'a', 'am': 'a',
    'bf': 'b', 'bm': 'b',
    'jf': 'j', 'jm': 'j',
    'zf': 'z', 'zm': 'z',
    'ef': 'e', 'em': 'e',
    'ff': 'f',
    'hf': 'h', 'hm': 'h',
    'if': 'i', 'im': 'i',
    'pf': 'p', 'pm': 'p',
}

# Registries contain only fully-published objects. Construction is coordinated
# by per-key events, so the registry lock is never held during I/O or inference.
_registry_lock = threading.RLock()
_download_lock = threading.Lock()
_pipelines = {}
_models = {}
_building = {}
_shutting_down = False
_shutdown_complete = threading.Event()


class _Flight:
    """A single construction attempt and its terminal failure, if any."""
    def __init__(self):
        self.event = threading.Event()
        self.exception = None

    def wait(self):
        self.event.wait()
        if self.exception is not None:
            raise self.exception

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

    # Use configured voices directory (see get_voices_dir)
    voices_dir = get_voices_dir()
    voices_dir.mkdir(parents=True, exist_ok=True)

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

                voice_path = voices_dir / voice_file
                if voice_path.exists() and voice_path.stat().st_size > 0:
                    return voice_file, True, f"Voice file {voice_file} already exists"

                # The fetch targets a private temporary directory beside the
                # destination, so it needs no lock and stays parallel across
                # workers. Only the existence re-check and the atomic promotion
                # are serialized.
                temp_dir = tempfile.mkdtemp(dir=voices_dir, prefix='.voice-')
                try:
                    downloaded_path = hf_hub_download(
                        repo_id="hexgrad/Kokoro-82M",
                        filename=f"voices/{voice_file}",
                        local_dir=temp_dir,
                        force_download=False,
                        revision=repo_version,
                        local_files_only=OFFLINE_MODE
                    )
                    if Path(downloaded_path).stat().st_size == 0:
                        raise ValueError(f"Downloaded file {voice_file} has zero size")
                    with _download_lock:
                        if voice_path.exists() and voice_path.stat().st_size > 0:
                            return voice_file, True, f"Voice file {voice_file} already exists"
                        os.replace(downloaded_path, voice_path)
                    return voice_file, True, f"Successfully downloaded {voice_file}"
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

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

def build_model(
    model_path: Optional[str],
    device: str,
    repo_version: str = "main",
    lang_code: str = 'a'
) -> EnhancedKPipeline:
    """Return a full-key cached pipeline using keyed single-flight creation."""
    global _shutting_down
    if lang_code not in LANGUAGE_CODES:
        raise ValueError(f"Unsupported language code: {lang_code!r}")
    chinese = lang_code == 'z'
    repo = "hexgrad/Kokoro-82M-v1.1-zh" if chinese else "hexgrad/Kokoro-82M"
    filename = 'kokoro-v1_1-zh.pth' if chinese else 'kokoro-v1_0.pth'
    revision_dir = None
    if repo_version != "main":
        revision_id = hashlib.sha256(
            f"{repo}@{repo_version}".encode("utf-8")
        ).hexdigest()[:16]
        revision_dir = get_model_dir() / "revisions" / revision_id
    checkpoint = (
        os.path.abspath(model_path)
        if model_path
        else str((revision_dir or get_model_dir()) / filename)
    )
    if os.environ.get("KOKORO_CONFIG_PATH"):
        config_path = str(get_config_path(chinese))
    else:
        config_name = "config-v1_1-zh.json" if chinese else "config.json"
        config_path = str((revision_dir or get_model_dir()) / config_name)
    family_key = (repo, checkpoint, config_path, repo_version, device)
    pipeline_key = family_key + (lang_code,)

    while True:
        with _registry_lock:
            if _shutting_down:
                raise RuntimeError("Pipeline registry is shutting down")
            if pipeline_key in _pipelines:
                return _pipelines[pipeline_key]
            flight = _building.get(pipeline_key)
            if flight is None:
                flight = _building[pipeline_key] = _Flight()
                break
        flight.wait()

    failure = None
    try:
        # An explicitly requested checkpoint is never fetched: silently writing
        # the repository default to a caller-supplied path would disguise a
        # mistyped path as a working (but wrong) fine-tune.
        artifacts = [(config_path, 'config.json')]
        if model_path is None:
            artifacts.insert(0, (checkpoint, filename))
        elif not os.path.exists(checkpoint):
            raise ValueError(f"Model file not found: {checkpoint}")

        # Artifact operations alone use the download lock.
        with _download_lock:
            for target, remote in artifacts:
                if os.path.exists(target):
                    continue
                if OFFLINE_MODE:
                    raise ValueError(f"Required artifact not found in offline mode: {target}")
                from huggingface_hub import hf_hub_download
                parent = Path(target).parent
                parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=parent) as temp_dir:
                    source = hf_hub_download(repo_id=repo, filename=remote,
                        local_dir=temp_dir, revision=repo_version,
                        local_files_only=OFFLINE_MODE)
                    if os.path.getsize(source) == 0:
                        raise ValueError(f"Downloaded artifact is empty: {remote}")
                    os.replace(source, target)

        # Must stay outside the `with _download_lock` block above:
        # _download_lock is a plain Lock and download_voice_files acquires it.
        download_voice_files(repo_version=repo_version, required_count=1)
        with open(config_path, 'r', encoding='utf-8-sig') as config_file:
            config = json.load(config_file)

        model_flight_key = ('model',) + family_key
        model_owner = False
        while True:
            with _registry_lock:
                family = _models.get(family_key)
                if family is not None:
                    break
                model_flight = _building.get(model_flight_key)
                if model_flight is None:
                    model_flight = _building[model_flight_key] = _Flight()
                    model_owner = True
                    break
            model_flight.wait()
        if model_owner:
            model_failure = None
            try:
                lock = threading.RLock()
                kokoro_model = KModel(repo_id=repo, config=config, model=checkpoint).to(device).eval()
                family = (kokoro_model, lock)
                with _registry_lock:
                    if _shutting_down:
                        raise RuntimeError("Pipeline registry is shutting down")
                    _models[family_key] = family
            except BaseException as exc:
                model_failure = exc
                raise
            finally:
                with _registry_lock:
                    _building.pop(model_flight_key, None)
                    model_flight.exception = model_failure
                    model_flight.event.set()
        kokoro_model, family_lock = family
        pipeline = EnhancedKPipeline(lang_code=lang_code, model=kokoro_model,
            repo_id=repo, device=device, family_lock=family_lock)
        pipeline._model_path = checkpoint
        pipeline._config_path = config_path
        pipeline._repo_version = repo_version
        pipeline._published = True
        with _registry_lock:
            if _shutting_down:
                raise RuntimeError("Pipeline registry is shutting down")
            _pipelines[pipeline_key] = pipeline
        return pipeline
    except BaseException as exc:
        failure = exc
        raise
    finally:
        with _registry_lock:
            _building.pop(pipeline_key, None)
            flight.exception = failure
            flight.event.set()

def _routable_voice_names(voice_files: List[Path]) -> List[str]:
    """Return sorted voice stems whose prefix maps to a supported language.

    Names that :func:`get_language_code_from_voice` would reject are dropped
    here so they never reach a picker: selecting one downstream would raise
    rather than produce a readable message.
    """
    routable = []
    for voice_file in sorted(voice_files, key=lambda f: f.stem.lower()):
        try:
            get_language_code_from_voice(voice_file.stem)
        except ValueError as e:
            logger.warning(f"Ignoring voice file {voice_file.name}: {e}")
            continue
        routable.append(voice_file.stem)
    return routable


def list_available_voices() -> List[str]:
    """List all available voice models"""
    # Use configured voices directory (see get_voices_dir)
    voices_dir = get_voices_dir()

    # Create voices directory if it doesn't exist
    if not voices_dir.exists():
        print(f"Creating voices directory at {voices_dir}")
        voices_dir.mkdir(parents=True, exist_ok=True)
        return []

    # Get all .pt files in the voices directory
    voice_files = list(voices_dir.glob("*.pt"))

    # If we found voice files, return them
    if voice_files:
        return _routable_voice_names(voice_files)

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
                return _routable_voice_names(list(voices_dir.glob("*.pt")))

    print("No voice files found. Please run the application again to download voices.")
    return []

def get_language_code_from_voice(voice_name: str) -> str:
    """Get the appropriate language code from a voice name

    Args:
        voice_name: Name of the voice (e.g., 'af_bella', 'jf_alpha')

    Returns:
        Language code for the voice
    """
    name = Path(voice_name).stem
    if len(name) < 3 or name[2] != '_':
        raise ValueError("Voice name must begin with a two-character prefix and underscore")
    prefix = name[:2].lower()
    if prefix not in VOICE_PREFIX_TO_LANGUAGE_CODE:
        raise ValueError(f"Unknown voice language prefix: {prefix!r}")
    return VOICE_PREFIX_TO_LANGUAGE_CODE[prefix]

def load_voice(voice_name: str, device: str) -> torch.Tensor:
    """Load a voice model in a thread-safe manner

    Args:
        voice_name: Name of the voice to load (with or without .pt extension)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Loaded voice model tensor

    Raises:
        ValueError: If voice name is invalid, voice file not found, or loading fails
    """
    voice_path = get_safe_voice_path(voice_name)
    voice_name_clean = voice_path.stem

    if not voice_path.exists():
        raise ValueError(f"Voice file not found: {voice_path}")

    pipeline = build_model(None, device, lang_code=get_language_code_from_voice(voice_name_clean))

    with pipeline._family_lock:
        if voice_name_clean in pipeline.voices:
            return pipeline.voices[voice_name_clean]
        return pipeline.load_voice(str(voice_path))

def generate_speech(
    model: EnhancedKPipeline,
    text: str,
    voice: str,
    speed: float = 1.0
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Generate speech using the Kokoro pipeline in a thread-safe manner

    Convenience wrapper that collects every segment into one tensor. Callers
    that need streaming or their own error handling should use
    :meth:`EnhancedKPipeline.iter_speech` directly, as the bundled CLIs do.

    The pipeline's language must match the voice prefix; the device is fixed
    at :func:`build_model` time and cannot be changed here.

    Args:
        model: EnhancedKPipeline instance
        text: Text to synthesize
        voice: Voice name (e.g. 'af_bella')
        speed: Speech speed multiplier (default: 1.0)

    Returns:
        Tuple of (audio tensor, phonemes string) or (None, None) on error
    """
    try:
        if model is None:
            raise ValueError("Model is None - pipeline not properly initialized")

        # Validate voice name and resolve safe path
        voice_path = get_safe_voice_path(voice)
        voice_name = voice_path.stem

        # Check if voice file exists
        if not voice_path.exists():
            raise ValueError(f"Voice file not found: {voice_path}")

        with model._family_lock:
            if voice_name not in model.voices:
                logger.info(f"Loading voice {voice_name}...")
                try:
                    model.load_voice(str(voice_path))
                    if voice_name not in model.voices:
                        raise ValueError("Voice load succeeded but voice not in model.voices dictionary")
                except Exception as e:
                    raise ValueError(f"Failed to load voice {voice_name}: {e}")

        # Generate speech (outside the lock for better concurrency).
        # Voice cache mutation is already protected above; the generator
        # itself only reads from the cache.
        logger.info(f"Generating speech with device: {model.device}")

        audio_segments = []
        phoneme_segments = []
        # The generator holds the model-family lock until closed, so it must
        # not survive an exception raised mid-iteration.
        with contextlib.closing(model.iter_speech(
            text,
            voice=str(voice_path),
            speed=speed,
            split_pattern=r'\n+'
        )) as generator:
            for gs, ps, audio in generator:
                if audio is not None:
                    if isinstance(audio, np.ndarray):
                        audio = torch.from_numpy(audio).float()
                    audio_segments.append(audio)
                    if ps:
                        phoneme_segments.append(ps)

        if audio_segments:
            return torch.cat(audio_segments, dim=0), "\n".join(phoneme_segments)

        return None, None
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, AttributeError, TypeError) as e:
        logger.error(f"Error generating speech: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during speech generation: {e}")
        return None, None


def shutdown_pipelines() -> None:
    """Idempotently reject work, drain builds/inference, and clear registries.

    This is terminal, not merely idempotent: ``_shutting_down`` is never
    cleared, so no pipeline can be rebuilt in this process afterwards. It is
    intended for process teardown (atexit / signal handlers). Tests and
    embedders that need a fresh registry must reimport the module.

    Draining relies on every in-flight generator being closed. Consumers that
    stop iterating early must close the generator (see ``contextlib.closing``)
    or this call blocks on that family's lock.
    """
    global _shutting_down
    with _registry_lock:
        if _shutting_down:
            waiter = _shutdown_complete
            owner = False
        else:
            _shutting_down = True
            _shutdown_complete.clear()
            waiter = _shutdown_complete
            owner = True
            flights = list(_building.values())
    if not owner:
        waiter.wait()
        return

    try:
        # Builds never hold the registry lock while doing construction and all
        # observe shutdown before publishing either a model or pipeline.
        for flight in flights:
            flight.event.wait()
        with _registry_lock:
            families = list(_models.values())
        # Acquiring every family lock waits for lazy generators to finish.
        locks = []
        try:
            for _, lock in families:
                lock.acquire()
                locks.append(lock)
            with _registry_lock:
                for pipeline in _pipelines.values():
                    pipeline._closed = True
                    pipeline.voices.clear()
                    pipeline.model = None
                _pipelines.clear()
                _models.clear()
        finally:
            for lock in reversed(locks):
                lock.release()
    finally:
        _shutdown_complete.set()
