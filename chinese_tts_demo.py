"""
Chinese TTS Demo - Interactive CLI for Kokoro Chinese TTS Model
================================================================

This script provides an interactive command-line interface for the Kokoro-82M-v1.1_zh
Chinese TTS model. It handles Chinese-specific text processing and voice selection.

Usage:
    python chinese_tts_demo.py

Requirements:
    - Kokoro-82M-v1.1_zh.pth model file
    - Chinese voice files in voices/ directory
    - All dependencies from requirements.txt
"""

import torch
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Union
import soundfile as sf
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import from local modules
from models import build_model, generate_speech, EnhancedKPipeline
from chinese_config import (
    ChineseTextProcessor,
    ChineseTTSConfig,
    CHINESE_VOICES,
    get_chinese_voices,
    get_chinese_voice_info
)
from config import TTSConfig

# Constants
DEFAULT_CHINESE_MODEL = "kokoro-82M-v1.1_zh.pth"
DEFAULT_CHINESE_OUTPUT = "output_chinese.wav"
SAMPLE_RATE = 24000
MIN_SPEED = 0.5
MAX_SPEED = 2.0
DEFAULT_SPEED = 1.0

# Sample Chinese texts for testing
SAMPLE_CHINESE_TEXTS = {
    "1": {
        "title": "北风与太阳 (The North Wind and the Sun)",
        "text": "你当旅行者裹着温暖的斗篷走来时,北风和太阳之争更强。他们同意,一个首先成功使旅行者脱下斗篷的人应该被认为比另一个更强大。然后,北风吹得尽力而为,但吹得越厉害,旅行者就越披上斗篷。最后,北风放弃了这一尝试。然后,太阳温暖地照耀着,旅行者立刻脱下了斗篷。因此,北风不得不承认太阳是两者中最强的一个。"
    },
    "2": {
        "title": "简短测试 (Short Test)",
        "text": "你好，这是一个中文文本转语音测试。"
    },
    "3": {
        "title": "自定义输入 (Custom Input)",
        "text": None  # Will be entered by user
    }
}


def print_chinese_header():
    """Print application header"""
    print("\n" + "="*60)
    print("  Kokoro-82M-v1.1 Chinese TTS Demo")
    print("  科克罗中文文本转语音演示")
    print("="*60 + "\n")


def print_menu():
    """Print the main menu options in Chinese"""
    print("\n" + "-"*40)
    print("  主菜单 (Main Menu)")
    print("-"*40)
    print("1. 列出可用声音 (List available voices)")
    print("2. 生成语音 (Generate speech)")
    print("3. 从样本文本生成 (Generate from sample text)")
    print("4. 帮助 (Help)")
    print("5. 退出 (Exit)")
    print("-"*40)
    return input("请选择一个选项 (Select an option) (1-5): ").strip()


def print_help():
    """Print help information"""
    print("\n" + "="*60)
    print("帮助信息 (Help Information)")
    print("="*60)
    print("""
关于本程序 (About this program):
  这是一个中文TTS演示程序，使用Kokoro-82M-v1.1中文模型。
  This is a Chinese TTS demo using the Kokoro-82M-v1.1 Chinese model.

功能 (Features):
  - 支持8个中文女性和男性声音 (Supports 8 Chinese female and male voices)
  - 可调节语速 (Adjustable speech speed)
  - 支持自定义和预设文本 (Supports custom and preset texts)
  - 自动文本处理和分割 (Automatic text processing and segmentation)

声音列表 (Voice List):
  女性声音 (Female voices):
    zf_xiaobei  - 晓蓓 (Young, energetic)
    zf_xiaoni   - 晓妮 (Clear, friendly)
    zf_xiaoxiao - 晓晓 (Soft, gentle)
    zf_xiaoyi   - 晓艺 (Professional, articulate)
  
  男性声音 (Male voices):
    zm_yunjian  - 云健 (Strong, confident)
    zm_yunxi    - 云析 (Warm, professional)
    zm_yunxia   - 云夏 (Calm, steady)
    zm_yunyang  - 云阳 (Resonant, deep)

常见问题 (FAQ):
  Q: 提示"字数不匹配" (Word count mismatch warning)?
  A: 这通常是因为英文音素化器被用于中文文本。
     请确保使用正确的中文模型和配置。
     
  Q: 生成的音频质量不好?
  A: 尝试调整语速，使用不同的声音。
     确保模型和声音文件完整。
""")
    print("="*60 + "\n")


def list_chinese_voices():
    """List all available Chinese voices with details"""
    print("\n" + "-"*60)
    print("可用声音 (Available Chinese Voices)")
    print("-"*60)
    
    voices = get_chinese_voices()
    
    # Organize by gender
    female_voices = [v for v in voices if v.startswith('zf_')]
    male_voices = [v for v in voices if v.startswith('zm_')]
    
    print("\n女性声音 (Female Voices):")
    for i, voice in enumerate(female_voices, 1):
        info = get_chinese_voice_info(voice)
        print(f"  {i}. {voice} - {info['name']} ({info['description']})")
    
    print("\n男性声音 (Male Voices):")
    for i, voice in enumerate(male_voices, 1):
        info = get_chinese_voice_info(voice)
        print(f"  {i+len(female_voices)}. {voice} - {info['name']} ({info['description']})")
    
    print("-"*60 + "\n")


def select_voice(voices: List[str]) -> str:
    """Interactive voice selection"""
    print("\n可用声音 (Available voices):")
    for i, voice in enumerate(voices, 1):
        info = get_chinese_voice_info(voice)
        print(f"{i}. {voice} - {info['name']} ({info['description']})")

    while True:
        try:
            choice = input("\n请选择一个声音编号 (Select a voice number) (or press Enter for 'zf_xiaobei'): ").strip()
            if not choice:
                return "zf_xiaobei"
            choice = int(choice)
            if 1 <= choice <= len(voices):
                return voices[choice - 1]
            print(f"无效选择。请输入1到{len(voices)}之间的数字。(Invalid choice. Please try again.)")
        except ValueError:
            print("请输入有效的数字。(Please enter a valid number.)")


def get_chinese_text_input() -> str:
    """Get Chinese text input from user"""
    print("\n请输入要转换为语音的中文文本")
    print("(Enter the Chinese text you want to convert to speech)")
    print("(or press Enter to exit)")
    text = input("> ").strip()
    return text


def get_speech_speed() -> float:
    """Get speech speed from user"""
    while True:
        try:
            speed = input(f"\n请输入语速 (Enter speech speed) ({MIN_SPEED}-{MAX_SPEED}, default {DEFAULT_SPEED}): ").strip()
            if not speed:
                return DEFAULT_SPEED
            speed = float(speed)
            if MIN_SPEED <= speed <= MAX_SPEED:
                return speed
            print(f"语速必须在 {MIN_SPEED} 和 {MAX_SPEED} 之间。(Speed must be between {MIN_SPEED} and {MAX_SPEED})")
        except ValueError:
            print("请输入有效的数字。(Please enter a valid number.)")


def select_sample_text() -> Optional[str]:
    """Select from predefined sample texts"""
    print("\n选择样本文本 (Select sample text):")
    for key, sample in SAMPLE_CHINESE_TEXTS.items():
        print(f"{key}. {sample['title']}")
        if sample["text"]:
            print(f"   {sample['text'][:50]}...")
    
    choice = input("\n请选择 (Select): ").strip()
    
    if choice in SAMPLE_CHINESE_TEXTS:
        if SAMPLE_CHINESE_TEXTS[choice]["text"]:
            return SAMPLE_CHINESE_TEXTS[choice]["text"]
        else:
            # Custom input option
            return get_chinese_text_input()
    
    return None


def load_chinese_model(model_path: str, device: str) -> EnhancedKPipeline:
    """Load the Chinese TTS model
    
    Args:
        model_path: Path to the Chinese model file
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        EnhancedKPipeline instance configured for Chinese
    """
    try:
        # Check if model file exists
        model_file = Path(model_path).resolve()
        if not model_file.exists():
            print(f"错误: 找不到模型文件 (Error: Model file not found): {model_file}")
            print(f"请确保您已下载 {DEFAULT_CHINESE_MODEL}")
            raise FileNotFoundError(f"Chinese model not found: {model_file}")
        
        # Build model with Chinese language code
        logger.info(f"加载中文模型 (Loading Chinese model): {model_path}")
        
        # Import build_model to use with Chinese config
        from models import build_model
        
        # We'll use language code 'z' for Chinese (Mandarin)
        # Create a custom pipeline for Chinese
        pipeline = build_model(model_path, device, repo_version="main")
        
        logger.info("中文模型加载成功 (Chinese model loaded successfully)")
        return pipeline
        
    except Exception as e:
        logger.error(f"加载中文模型时出错 (Error loading Chinese model): {e}")
        raise


def generate_chinese_speech(
    model: EnhancedKPipeline,
    text: str,
    voice: str,
    device: str = 'cpu',
    speed: float = 1.0
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Generate speech for Chinese text
    
    Args:
        model: EnhancedKPipeline instance
        text: Chinese text to synthesize
        voice: Voice name (e.g., 'zf_xiaobei')
        device: Device to use
        speed: Speech speed multiplier
        
    Returns:
        Tuple of (audio_data, phonemes) or (None, None) on error
    """
    try:
        # Check if text contains Chinese characters
        if not ChineseTextProcessor.is_chinese(text):
            print("警告: 文本可能不是中文 (Warning: Text may not be Chinese)")
        
        # Normalize Chinese text
        text = ChineseTextProcessor.normalize_chinese_text(text)
        logger.info(f"已规范化文本 (Normalized text): {text[:50]}...")
        
        # Generate speech
        logger.info(f"生成语音... (Generating speech...)")
        print(f"  文本: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"  声音: {voice}")
        print(f"  语速: {speed}x")
        
        # Load voice file
        voice_path = Path("voices").resolve() / f"{voice}.pt"
        if not voice_path.exists():
            print(f"错误: 找不到声音文件 (Error: Voice file not found): {voice_path}")
            return None, None
        
        # Generate using the model
        audio_segments = []
        all_phonemes = []
        
        try:
            generator = model(
                text,
                voice=str(voice_path),
                speed=speed,
                split_pattern=r'\n+'
            )
            
            for gs, ps, audio in generator:
                if audio is not None:
                    # Convert to numpy if needed
                    if isinstance(audio, torch.Tensor):
                        audio = audio.numpy()
                    audio_segments.append(audio)
                    all_phonemes.append(ps)
                    logger.info(f"生成了句段: {gs} (Generated segment: {gs})")
            
            # Concatenate all audio segments
            if audio_segments:
                if len(audio_segments) == 1:
                    final_audio = audio_segments[0]
                else:
                    final_audio = np.concatenate(audio_segments, axis=0)
                
                all_phonemes_str = " ".join(all_phonemes) if all_phonemes else ""
                return final_audio, all_phonemes_str
            else:
                print("错误: 没有生成音频 (Error: No audio was generated)")
                return None, None
                
        except Exception as e:
            logger.error(f"生成过程中出错 (Error during generation): {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
    except Exception as e:
        logger.error(f"生成语音时出错 (Error generating speech): {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_audio(audio_data: np.ndarray, output_path: str = DEFAULT_CHINESE_OUTPUT) -> bool:
    """Save generated audio to file
    
    Args:
        audio_data: Audio data as numpy array
        output_path: Path to save the audio file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing file if it exists
        if output_path.exists():
            output_path.unlink()
        
        logger.info(f"保存音频到 (Saving audio to): {output_path}")
        sf.write(str(output_path), audio_data, SAMPLE_RATE)
        print(f"✓ 音频已保存 (Audio saved to): {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存音频时出错 (Error saving audio): {e}")
        print(f"✗ 无法保存音频 (Failed to save audio): {e}")
        return False


def main():
    """Main application loop"""
    print_chinese_header()
    
    try:
        # Set up device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备 (Using device): {device}\n")
        
        # Load model
        print("初始化模型 (Initializing model)...")
        model = load_chinese_model(DEFAULT_CHINESE_MODEL, device)
        print("✓ 模型已加载 (Model loaded)\n")
        
        # Get available voices
        voices = get_chinese_voices()
        if not voices:
            print("错误: 找不到中文声音文件 (Error: No Chinese voices found)")
            print(f"请确保中文声音文件在 voices/ 目录中")
            return
        
        # Main loop
        while True:
            choice = print_menu()
            
            if choice == "1":
                # List voices
                list_chinese_voices()
                
            elif choice == "2":
                # Generate speech from user input
                voice = select_voice(voices)
                text = get_chinese_text_input()
                
                if not text:
                    print("已取消 (Cancelled)")
                    continue
                
                speed = get_speech_speed()
                
                print("\n生成中... (Generating...)")
                audio, phonemes = generate_chinese_speech(model, text, voice, device, speed)
                
                if audio is not None:
                    if save_audio(audio):
                        print("✓ 完成 (Done)")
                    else:
                        print("✗ 保存失败 (Save failed)")
                else:
                    print("✗ 生成失败 (Generation failed)")
                    
            elif choice == "3":
                # Generate from sample text
                text = select_sample_text()
                if text:
                    voice = select_voice(voices)
                    speed = get_speech_speed()
                    
                    print("\n生成中... (Generating...)")
                    audio, phonemes = generate_chinese_speech(model, text, voice, device, speed)
                    
                    if audio is not None:
                        if save_audio(audio):
                            print("✓ 完成 (Done)")
                        else:
                            print("✗ 保存失败 (Save failed)")
                    else:
                        print("✗ 生成失败 (Generation failed)")
                        
            elif choice == "4":
                # Help
                print_help()
                
            elif choice == "5":
                # Exit
                print("\n再见！(Goodbye!)")
                break
                
            else:
                print("无效选择。请重试。(Invalid choice. Please try again.)")
                
    except KeyboardInterrupt:
        print("\n\n用户中断 (User interrupted)")
    except Exception as e:
        logger.error(f"应用程序错误 (Application error): {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束 (Program ended)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

