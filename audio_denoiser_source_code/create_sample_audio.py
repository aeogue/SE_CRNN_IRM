import numpy as np
import soundfile as sf
import librosa

def create_sample_audio():
    """创建示例音频文件用于测试"""
    sample_rate = 32000
    duration = 3  # 3秒
    
    # 生成一个简单的正弦波作为测试音频
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4音
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # 添加一些谐波使声音更自然
    audio += 0.2 * np.sin(2 * np.pi * 2 * frequency * t)
    audio += 0.1 * np.sin(2 * np.pi * 3 * frequency * t)
    
    # 保存音频文件
    sf.write("sample_audio.wav", audio, sample_rate)
    print(f"✅ 示例音频文件已创建: sample_audio.wav")
    print(f"   采样率: {sample_rate} Hz")
    print(f"   时长: {duration} 秒")
    print(f"   文件大小: {len(audio)} 个采样点")

if __name__ == "__main__":
    create_sample_audio()