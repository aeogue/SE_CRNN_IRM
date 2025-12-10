import sys
import os
import numpy as np

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from crnn_model import RealTimeDenoiser
    print("✅ RealTimeDenoiser导入成功")
    
    # 测试初始化
    denoiser = RealTimeDenoiser()
    denoiser.initialize()
    print("✅ RealTimeDenoiser初始化成功")
    
    # 测试音频处理
    test_audio = np.random.randn(3200)  # 100ms @ 32kHz
    enhanced_audio = denoiser.process_frame(test_audio)
    print(f"✅ 音频处理成功，输入形状: {test_audio.shape}, 输出形状: {enhanced_audio.shape}")
    
    print("🎉 所有测试通过！AI降噪系统正常工作")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()