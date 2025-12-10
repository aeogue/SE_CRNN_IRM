import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
# 导入你提供的AudioProcessor
from audio_processor import AudioProcessor

class DenoiseDataset(Dataset):
    """降噪数据集：适配你提供的AudioProcessor"""
    def __init__(self, clean_audio_dir: str, noisy_audio_dir: str, 
                 sample_rate: int = 16000, seq_len: int = 100):
        """
        Args:
            clean_audio_dir: 干净语音目录
            noisy_audio_dir: 带噪语音目录（文件名与干净语音一一对应）
            sample_rate: 采样率（需与AudioProcessor一致）
            seq_len: 固定序列长度（模型输入的时间步长）
        """
        self.clean_dir = clean_audio_dir
        self.noisy_dir = noisy_audio_dir
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        
        # 初始化你提供的AudioProcessor
        self.audio_processor = AudioProcessor(
            sample_rate=sample_rate,
            frame_size=160,  # 10ms @ 16kHz
            hop_size=80     # 5ms @ 16kHz
        )
        
        # 匹配干净/带噪语音文件（仅保留.wav）
        self.file_list = [
            f for f in os.listdir(clean_audio_dir) 
            if f.endswith('.wav') and os.path.exists(os.path.join(noisy_dir, f))
        ]
        if not self.file_list:
            raise ValueError("未找到匹配的干净/带噪语音文件对")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        # 1. 加载音频文件（确保单通道、采样率一致）
        file_name = self.file_list[idx]
        clean_audio, sr_clean = sf.read(os.path.join(self.clean_dir, file_name))
        noisy_audio, sr_noisy = sf.read(os.path.join(self.noisy_dir, file_name))
        
        # 校验采样率
        if sr_clean != self.sample_rate or sr_noisy != self.sample_rate:
            raise ValueError(f"采样率不匹配：{file_name} 需为{self.sample_rate}Hz")
        
        # 转为单通道（若为立体声）
        if len(clean_audio.shape) > 1:
            clean_audio = np.mean(clean_audio, axis=1)
        if len(noisy_audio.shape) > 1:
            noisy_audio = np.mean(noisy_audio, axis=1)
        
        # 确保音频长度一致（截断到较短的长度）
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        # 2. STFT变换（使用你提供的stft方法）
        clean_stft = self.audio_processor.stft(clean_audio)  # (freq_bins, n_frames)
        noisy_stft = self.audio_processor.stft(noisy_audio)
        
        # 3. 计算幅度谱（用于特征提取和增益计算）
        clean_mag = self.audio_processor.compute_magnitude_spectrum(clean_stft)
        noisy_mag = self.audio_processor.compute_magnitude_spectrum(noisy_stft)

        # 4. 提取模型输入特征（25维：22子带+3频谱特征）
        noisy_features = self.audio_processor.compute_spectral_features(noisy_mag)  # (25, n_frames)
        
        # 5. 计算目标增益（22维巴克子带增益）
        # 目标增益 = 干净子带能量 / 带噪子带能量（限制在[0,1]）
        clean_subband = self.audio_processor.decompose_to_subbands(clean_mag)  # (22, n_frames)
        noisy_subband = self.audio_processor.decompose_to_subbands(noisy_mag)
        target_gains = clean_subband / (noisy_subband + 1e-8)  # 避免除零
        target_gains = np.clip(target_gains, 0.0, 1.0)  # 增益范围[0,1]

        # 6. 截断/补零到固定序列长度（保证批次维度一致）
        n_frames = noisy_features.shape[1]
        if n_frames < self.seq_len:
            # 补零：(25, seq_len) / (22, seq_len) / (freq_bins, seq_len)
            pad_len = self.seq_len - n_frames
            noisy_features = np.pad(noisy_features, ((0,0), (0,pad_len)), mode='constant')
            target_gains = np.pad(target_gains, ((0,0), (0,pad_len)), mode='constant')
            clean_mag = np.pad(clean_mag, ((0,0), (0,pad_len)), mode='constant')
            noisy_mag = np.pad(noisy_mag, ((0,0), (0,pad_len)), mode='constant')
        else:
            # 随机截断（数据增强）
            start_idx = np.random.randint(0, n_frames - self.seq_len + 1)
            noisy_features = noisy_features[:, start_idx:start_idx+self.seq_len]
            target_gains = target_gains[:, start_idx:start_idx+self.seq_len]
            clean_mag = clean_mag[:, start_idx:start_idx+self.seq_len]
            noisy_mag = noisy_mag[:, start_idx:start_idx+self.seq_len]

        # 7. 维度转置：(feat_dim, seq_len) → (seq_len, feat_dim)（适配模型输入）
        noisy_features = noisy_features.T  # (seq_len, 25)
        target_gains = target_gains.T      # (seq_len, 22)

        # 8. 转换为Tensor（float32）
        return {
            "noisy_features": torch.FloatTensor(noisy_features),    # [seq_len, 25]
            "target_gains": torch.FloatTensor(target_gains),        # [seq_len, 22]
            "clean_spectrum": torch.FloatTensor(clean_mag),         # [freq_bins, seq_len]
            "noisy_spectrum": torch.FloatTensor(noisy_mag)          # [freq_bins, seq_len]
        }


class CRNNDenoiseModel(nn.Module):
    """CRNN降噪模型，实现ZegoAIDenoise的神经网络部分"""
    
    def __init__(self, input_dim: int = 25, hidden_dim: int = 128, num_layers: int = 3, output_dim: int = 22):
        """
        初始化CRNN模型
        
        Args:
            input_dim: 输入特征维度 (22个子带 + 3个频谱特征)
            hidden_dim: GRU隐藏层维度
            num_layers: GRU层数
            output_dim: 输出维度 (22个子带增益)
        """
        super(CRNNDenoiseModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # 卷积层 - 提取局部特征
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        
        # GRU层 - 时序建模
        self.gru = nn.GRU(input_size=128, hidden_size=hidden_dim, 
                         num_layers=num_layers, batch_first=True, bidirectional=True)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出层 - 估计子带增益
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # 增益范围[0,1]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            
        Returns:
            子带增益估计 [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 转置为 [batch_size, input_dim, seq_len] 用于卷积
        x_conv = x.transpose(1, 2)
        
        # 卷积层
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        
        # 转置回 [batch_size, seq_len, channels]
        x_gru = x_conv.transpose(1, 2)
        
        # GRU层
        gru_out, _ = self.gru(x_gru)  # [batch_size, seq_len, hidden_dim*2]
        
        # 注意力机制
        attention_weights = F.softmax(self.attention(gru_out), dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attention_weights * gru_out, dim=1)  # [batch_size, hidden_dim*2]
        
        # 重复上下文向量以匹配序列长度
        context_expanded = context_vector.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, hidden_dim*2]
        
        # 融合GRU输出和上下文信息
        fused_features = gru_out + context_expanded
        
        # 输出层
        gains = self.output_layer(fused_features)  # [batch_size, seq_len, output_dim]
        
        return gains

class DenoiseTrainer:
    """降噪模型训练器"""
    
    def __init__(self, model: CRNNDenoiseModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
    def compute_loss(self, predicted_gains: torch.Tensor, target_gains: torch.Tensor, 
                    clean_spectrum: torch.Tensor, noisy_spectrum: torch.Tensor) -> torch.Tensor:
        """
        计算损失函数，包含平方误差和四次方误差
        
        Args:
            predicted_gains: 预测的增益
            target_gains: 目标增益
            clean_spectrum: 干净语音频谱
            noisy_spectrum: 带噪语音频谱
            
        Returns:
            总损失
        """
        # 平方误差损失
        mse_loss = F.mse_loss(predicted_gains, target_gains)
        
        # 四次方误差损失 - 强调大误差的代价
        quartic_loss = torch.mean((predicted_gains - target_gains) ** 4)
        
        # 语音保护损失 - 避免对语音的过度抑制
        enhanced_spectrum = predicted_gains.unsqueeze(-1) * noisy_spectrum.unsqueeze(2)
        speech_preservation_loss = F.mse_loss(enhanced_spectrum, clean_spectrum)
        
        # 组合损失
        total_loss = mse_loss + 0.1 * quartic_loss + 0.5 * speech_preservation_loss
        
        return total_loss
    
    def train_step(self, noisy_features: torch.Tensor, target_gains: torch.Tensor,
                  clean_spectrum: torch.Tensor, noisy_spectrum: torch.Tensor) -> float:
        """
        单步训练
        
        Returns:
            当前步骤的损失值
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        predicted_gains = self.model(noisy_features)
        
        # 计算损失
        loss = self.compute_loss(predicted_gains, target_gains, clean_spectrum, noisy_spectrum)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, noisy_features: torch.Tensor, target_gains: torch.Tensor,
                clean_spectrum: torch.Tensor, noisy_spectrum: torch.Tensor) -> float:
        """
        验证步骤
        
        Returns:
            验证损失
        """
        self.model.eval()
        with torch.no_grad():
            predicted_gains = self.model(noisy_features)
            loss = self.compute_loss(predicted_gains, target_gains, clean_spectrum, noisy_spectrum)
        return loss.item()

def create_pretrained_model() -> CRNNDenoiseModel:
    """
    创建预训练模型（模拟训练好的模型）
    
    在实际应用中，这里应该加载真实训练好的权重
    """
    model = CRNNDenoiseModel(input_dim=25, hidden_dim=128, num_layers=3, output_dim=22)
    
    # 这里可以添加预训练权重加载逻辑
    # 由于时间和资源限制，我们使用随机初始化的模型进行演示
        """加载训练好的模型"""
    #model = CRNNDenoiseModel(input_dim=25, hidden_dim=128, num_layers=3, output_dim=22)
    
    # 加载权重
    # checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

class RealTimeDenoiser:
    """实时降噪器"""
    
    def __init__(self, model_path: str = None):
        self.audio_processor = None
        self.model = None
        self.is_initialized = False
        
    def initialize(self, sample_rate: int = 32000):
        """初始化降噪器"""
        from audio_processor import AudioProcessor
        
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)
        self.model = create_pretrained_model()
        self.model.eval()
        self.is_initialized = True
        
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        处理单帧音频
        
        Args:
            audio_frame: 输入音频帧
            
        Returns:
            降噪后的音频帧
        """
        if not self.is_initialized:
            raise RuntimeError("Denoiser not initialized. Call initialize() first.")
        
        # STFT变换
        stft_matrix = self.audio_processor.stft(audio_frame)
        magnitude_spectrum = self.audio_processor.compute_magnitude_spectrum(stft_matrix)
        
        # 提取特征
        features = self.audio_processor.compute_spectral_features(magnitude_spectrum)
        
        # 转换为模型输入格式
        features_tensor = torch.FloatTensor(features.T).unsqueeze(0)  # [1, seq_len, features]
        
        # 模型推理
        with torch.no_grad():
            gains = self.model(features_tensor)  # [1, seq_len, 22]
        
        # 应用增益到频谱
        gains_np = gains.squeeze(0).numpy().T  # [22, seq_len]
        
        # 将子带增益映射回频点增益
        enhanced_stft = stft_matrix.copy()
        for i, (low_bin, high_bin) in enumerate(self.audio_processor.bark_bands):
            band_gain = gains_np[i, :]
            # 将子带增益应用到对应频点
            for bin_idx in range(low_bin, high_bin):
                if bin_idx < enhanced_stft.shape[0]:
                    enhanced_stft[bin_idx, :] *= band_gain
        
        # 逆STFT重构时域信号
        enhanced_audio = self.audio_processor.istft(enhanced_stft)
        
        return enhanced_audio

if __name__ == "__main__":
    # 测试模型
    model = CRNNDenoiseModel()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    test_input = torch.randn(2, 100, 25)  # [batch_size, seq_len, features]
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")