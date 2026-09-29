"""
Setup Script for Kokoro Chinese TTS
===================================

This script downloads and sets up the Kokoro-v1.1-zh Chinese TTS model
and all required voice files.

Usage:
    python -m kokoro_tts_local.setup_chinese_tts
"""

import os
import sys
import tempfile
from pathlib import Path
import logging
from typing import List, Tuple
from .console import enable_utf8_console
from .paths import get_model_dir, get_voices_dir, get_config_path

# Applied at import rather than in the entry point: every function in this
# module prints Chinese, so any caller — CLI, script, or test — needs the
# streams usable before the first call. No-op unless the current encoding
# cannot represent the output.
enable_utf8_console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHINESE_MODEL_FILE = "kokoro-v1_1-zh.pth"
CONFIG_FILE = "config-v1_1-zh.json"
VOICES_DIR = get_voices_dir()

def _model_path() -> Path:
    return get_model_dir() / CHINESE_MODEL_FILE

def _config_path() -> Path:
    return get_config_path(is_chinese_model=True)

CHINESE_VOICES = [
    # Female voices
    "zf_xiaobei.pt",
    "zf_xiaoni.pt",
    "zf_xiaoxiao.pt",
    "zf_xiaoyi.pt",
    # Male voices
    "zm_yunjian.pt",
    "zm_yunxi.pt",
    "zm_yunxia.pt",
    "zm_yunyang.pt"
]


def print_header():
    """Print setup header"""
    print("\n" + "="*60)
    print("  Kokoro-82M-v1.1 Chinese TTS Setup")
    print("  科克罗中文TTS设置")
    print("="*60 + "\n")


def check_dependencies() -> bool:
    """Check if required packages are installed"""
    print("检查依赖 (Checking dependencies)...")
    
    required_packages = {
        'torch': 'PyTorch',
        'huggingface_hub': 'Hugging Face Hub',
        'kokoro': 'Kokoro',
        'soundfile': 'SoundFile'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            missing.append(package)
    
    if missing:
        print(f"\n缺少必需的包 (Missing packages): {', '.join(missing)}")
        print("请运行: pip install -e .")
        return False
    
    print("✓ 所有依赖已安装 (All dependencies installed)\n")
    return True


def download_file(repo_id: str, filename: str, local_dir: str = ".") -> bool:
    """Download a file from Hugging Face Hub
    
    Args:
        repo_id: Repository ID (e.g., "hexgrad/Kokoro-82M")
        filename: File to download
        local_dir: Local directory to save to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"下载 (Downloading): {filename}...")
        
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            force_download=False
        )
        
        print(f"  ✓ 完成 (Done): {filename}")
        return True
        
    except Exception as e:
        print(f"  ✗ 错误 (Error): {e}")
        return False


def download_model() -> bool:
    """Download the Chinese TTS model"""
    print("\n下载中文TTS模型 (Downloading Chinese TTS Model)...")
    print("-" * 60)
    
    model_path = _model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    # An empty file is a failed earlier download, not an installed model
    if model_path.is_file() and model_path.stat().st_size > 0:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ 模型文件已存在 (Model already exists): {model_path}")
        print(f"  大小 (Size): {size_mb:.1f} MB")
        return True
    
    # Download
    success = download_file(
        "hexgrad/Kokoro-82M-v1.1-zh",
        CHINESE_MODEL_FILE,
        local_dir=str(model_path.parent)
    )
    
    if success and model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ 模型已下载 (Model downloaded): {size_mb:.1f} MB\n")
        return True
    else:
        print(f"✗ 模型下载失败 (Model download failed)\n")
        return False


def download_config() -> bool:
    """Download the model configuration file"""
    print("下载配置文件 (Downloading Config File)...")
    print("-" * 60)
    
    config_path = _config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    if config_path.is_file() and config_path.stat().st_size > 0:
        print(f"✓ 配置文件已存在 (Config already exists): {config_path}")
        return True
    
    try:
        from huggingface_hub import hf_hub_download

        with tempfile.TemporaryDirectory(dir=config_path.parent) as temp_dir:
            downloaded_config = hf_hub_download(
                repo_id="hexgrad/Kokoro-82M-v1.1-zh",
                filename="config.json",
                local_dir=temp_dir,
                force_download=False
            )
            if Path(downloaded_config).stat().st_size == 0:
                raise ValueError("Downloaded config file is empty")
            os.replace(downloaded_config, config_path)

        print(f"✓ 配置文件已下载 (Config downloaded)\n")
        return True
    except Exception as e:
        print(f"  ✗ 错误 (Error): {e}")
        print(f"✗ 配置文件下载失败 (Config download failed)\n")
        return False


def download_voices() -> Tuple[int, int]:
    """Download all Chinese voice files
    
    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    print("下载中文声音文件 (Downloading Chinese Voice Files)...")
    print("-" * 60)
    
    # Create voices directory
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for voice_file in CHINESE_VOICES:
        voice_path = VOICES_DIR / voice_file
        
        # Check if already exists
        if voice_path.exists() and voice_path.stat().st_size > 0:
            size_mb = voice_path.stat().st_size / (1024 * 1024)
            print(f"✓ {voice_file} ({size_mb:.1f} MB)")
            successful += 1
            continue
        
        # Download
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"下载 (Downloading): {voice_file}...")
            
            # Hugging Face preserves the repository's ``voices/`` prefix
            # under local_dir. Download to a temporary directory, then move
            # the file to the flat directory expected by the application.
            with tempfile.TemporaryDirectory(dir=VOICES_DIR) as temp_dir:
                downloaded_path = hf_hub_download(
                    repo_id="hexgrad/Kokoro-82M",
                    filename=f"voices/{voice_file}",
                    local_dir=temp_dir,
                    force_download=False
                )
                if Path(downloaded_path).stat().st_size == 0:
                    raise ValueError(f"Downloaded voice file is empty: {voice_file}")
                os.replace(downloaded_path, voice_path)
            
            size_mb = voice_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ 完成 (Done): {voice_file} ({size_mb:.1f} MB)")
            successful += 1
            
        except Exception as e:
            print(f"  ✗ 错误 (Error): {voice_file} - {e}")
            failed += 1
    
    print(f"\n✓ 成功: {successful}/{len(CHINESE_VOICES)} (Successful: {successful}/{len(CHINESE_VOICES)})")
    if failed > 0:
        print(f"✗ 失败: {failed}/{len(CHINESE_VOICES)} (Failed: {failed}/{len(CHINESE_VOICES)})")
    
    print()
    return successful, failed


def verify_setup() -> bool:
    """Verify that all required files are in place"""
    print("验证设置 (Verifying Setup)...")
    print("-" * 60)
    
    all_good = True
    
    # Check model
    model_path = _model_path()
    if model_path.exists():
        print(f"✓ 中文模型 (Chinese Model): {CHINESE_MODEL_FILE}")
    else:
        print(f"✗ 缺少模型 (Missing Model): {CHINESE_MODEL_FILE}")
        all_good = False
    
    # Check config
    config_path = _config_path()
    if config_path.exists():
        print(f"✓ 配置文件 (Config File): {CONFIG_FILE}")
    else:
        print(f"✗ 缺少配置 (Missing Config): {CONFIG_FILE}")
        all_good = False
    
    # Check voices
    print(f"\n中文声音文件 (Chinese Voice Files):")
    voice_count = 0
    for voice_file in CHINESE_VOICES:
        voice_path = VOICES_DIR / voice_file
        if voice_path.exists():
            print(f"  ✓ {voice_file}")
            voice_count += 1
        else:
            print(f"  ✗ {voice_file}")
            all_good = False
    
    print(f"\n✓ 已找到 {voice_count}/{len(CHINESE_VOICES)} 个声音文件")
    print(f"(Found {voice_count}/{len(CHINESE_VOICES)} voice files)\n")
    
    return all_good


def print_summary(success: bool, model_ok: bool, config_ok: bool, voices_count: int):
    """Print setup summary"""
    print("="*60)
    print("  设置摘要 (Setup Summary)")
    print("="*60)
    
    if success:
        print("\n✓ 设置完成！(Setup Complete!)")
        print("\n下一步 (Next Steps):")
        print("1. 运行演示: kokoro-tts-chinese")
        print("   (Run demo: kokoro-tts-chinese)")
    else:
        print("\n⚠ 设置未完成 (Setup Incomplete)")
        print("\n缺少的文件 (Missing Files):")
        if not model_ok:
            print(f"  - {CHINESE_MODEL_FILE}")
        if not config_ok:
            print(f"  - {CONFIG_FILE}")
        if voices_count < len(CHINESE_VOICES):
            print(f"  - 声音文件 ({voices_count}/{len(CHINESE_VOICES)}) (Voice files)")
    
    print("\n"+"="*60 + "\n")


def main():
    """Main setup function"""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print("请先安装依赖 (Please install dependencies first)")
        return False
    
    # Download files
    model_ok = download_model()
    config_ok = download_config()
    voice_success, voice_failed = download_voices()
    
    # Verify setup
    print()
    setup_ok = verify_setup()
    
    # Summary
    print_summary(
        setup_ok,
        model_ok,
        config_ok,
        voice_success
    )
    
    return setup_ok


def cli() -> None:
    """Run setup and return a meaningful process status."""
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n设置被用户中止 (Setup interrupted by user)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"设置错误 (Setup error): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    cli()
