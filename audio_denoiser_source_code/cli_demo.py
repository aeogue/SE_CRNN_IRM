import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audio_processor import AudioProcessor
from crnn_model import RealTimeDenoiser

def demo_ai_denoise():
    """命令行版本的AI降噪演示"""
    print("🚀 ZegoAIDenoise AI降噪演示 - 命令行版本")
    print("=" * 50)
    
    try:
        # 1. 加载示例音频
        print("📥 加载示例音频...")
        audio, sr = librosa.load("sample_audio.wav", sr=32000)
        print(f"   音频信息: {len(audio)} 采样点, {sr} Hz 采样率")
        
        # 2. 添加噪声
        print("🔊 添加键盘敲击噪声...")
        noise = np.zeros_like(audio)
        click_duration = int(0.05 * sr)  # 50ms敲击声
        interval = int(0.2 * sr)  # 200ms间隔
        
        for i in range(0, len(audio), interval):
            if i + click_duration < len(audio):
                click = np.random.normal(0, 0.1, click_duration) * np.hanning(click_duration)
                noise[i:i+click_duration] += click
        
        noisy_audio = audio + noise * 5  # 噪声强度5
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        # 保存带噪音频
        sf.write("noisy_sample.wav", noisy_audio, sr)
        print("   带噪音频已保存: noisy_sample.wav")
        
        # 3. 初始化AI降噪器
        print("🧠 初始化AI降噪器...")
        denoiser = RealTimeDenoiser()
        denoiser.initialize(sr)
        print("   AI降噪器初始化完成")
        
        # 4. 进行AI降噪
        print("⚡ 进行AI降噪处理...")
        enhanced_audio = denoiser.process_frame(noisy_audio)
        
        # 保存降噪后音频
        sf.write("enhanced_sample.wav", enhanced_audio, sr)
        print("   降噪后音频已保存: enhanced_sample.wav")
        
        # 5. 生成频谱图对比
        print("📊 生成频谱图对比...")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 原始音频频谱图
        D_clean = librosa.stft(audio)
        S_db_clean = librosa.amplitude_to_db(np.abs(D_clean), ref=np.max)
        librosa.display.specshow(S_db_clean, sr=sr, x_axis='time', y_axis='hz', ax=axes[0])
        axes[0].set_title('原始音频频谱图')
        
        # 带噪音频频谱图
        D_noisy = librosa.stft(noisy_audio)
        S_db_noisy = librosa.amplitude_to_db(np.abs(D_noisy), ref=np.max)
        librosa.display.specshow(S_db_noisy, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('带噪音频频谱图')
        
        # 降噪后音频频谱图
        D_enhanced = librosa.stft(enhanced_audio)
        S_db_enhanced = librosa.amplitude_to_db(np.abs(D_enhanced), ref=np.max)
        img = librosa.display.specshow(S_db_enhanced, sr=sr, x_axis='time', y_axis='hz', ax=axes[2])
        axes[2].set_title('AI降噪后频谱图')
        
        plt.tight_layout()
        plt.savefig('spectrogram_comparison.png', dpi=150, bbox_inches='tight')
        print("   频谱图对比已保存: spectrogram_comparison.png")
        
        # 6. 计算性能指标
        print("📈 计算性能指标...")
        original_rms = np.sqrt(np.mean(audio**2))
        noisy_rms = np.sqrt(np.mean(noisy_audio**2))
        enhanced_rms = np.sqrt(np.mean(enhanced_audio**2))
        
        noise_reduction_db = 20 * np.log10(noisy_rms / enhanced_rms)
        print(f"   噪声抑制效果: {noise_reduction_db:.2f} dB")
        
        # 7. 显示结果总结
        print("\n🎉 AI降噪演示完成！")
        print("=" * 50)
        print("生成的文件:")
        print("  • sample_audio.wav     - 原始示例音频")
        print("  • noisy_sample.wav     - 添加噪声后的音频") 
        print("  • enhanced_sample.wav  - AI降噪后的音频")
        print("  • spectrogram_comparison.png - 频谱图对比")
        print(f"\n📊 性能指标:")
        print(f"   噪声抑制: {noise_reduction_db:.2f} dB")
        print(f"   处理时长: {len(audio)/sr:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_ai_denoise()