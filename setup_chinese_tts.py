"""
Setup Script for Kokoro Chinese TTS
===================================

This script downloads and sets up the Kokoro-82M-v1.1_zh Chinese TTS model
and all required voice files.

Usage:
    python setup_chinese_tts.py
"""

import os
import sys
from pathlib import Path
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHINESE_MODEL_FILE = "kokoro-82M-v1.1_zh.pth"
CONFIG_FILE = "config.json"
VOICES_DIR = Path("voices").resolve()

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
        print("请运行: pip install -r requirements.txt")
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
    
    model_path = Path(CHINESE_MODEL_FILE).resolve()
    
    # Check if already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ 模型文件已存在 (Model already exists): {model_path}")
        print(f"  大小 (Size): {size_mb:.1f} MB")
        return True
    
    # Download
    success = download_file(
        "hexgrad/Kokoro-82M",
        CHINESE_MODEL_FILE,
        local_dir="."
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
    
    config_path = Path(CONFIG_FILE).resolve()
    
    # Check if already exists
    if config_path.exists():
        print(f"✓ 配置文件已存在 (Config already exists): {config_path}")
        return True
    
    # Download
    success = download_file(
        "hexgrad/Kokoro-82M",
        CONFIG_FILE,
        local_dir="."
    )
    
    if success and config_path.exists():
        print(f"✓ 配置文件已下载 (Config downloaded)\n")
        return True
    else:
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
        if voice_path.exists():
            size_mb = voice_path.stat().st_size / (1024 * 1024)
            print(f"✓ {voice_file} ({size_mb:.1f} MB)")
            successful += 1
            continue
        
        # Download
        try:
            from huggingface_hub import hf_hub_download
            
            print(f"下载 (Downloading): {voice_file}...")
            
            downloaded_path = hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename=f"voices/{voice_file}",
                local_dir=str(VOICES_DIR.parent),
                force_download=False
            )
            
            size_mb = Path(downloaded_path).stat().st_size / (1024 * 1024)
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
    model_path = Path(CHINESE_MODEL_FILE).resolve()
    if model_path.exists():
        print(f"✓ 中文模型 (Chinese Model): {CHINESE_MODEL_FILE}")
    else:
        print(f"✗ 缺少模型 (Missing Model): {CHINESE_MODEL_FILE}")
        all_good = False
    
    # Check config
    config_path = Path(CONFIG_FILE).resolve()
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
        print("1. 运行演示: python chinese_tts_demo.py")
        print("   (Run demo: python chinese_tts_demo.py)")
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


if __name__ == "__main__":
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

