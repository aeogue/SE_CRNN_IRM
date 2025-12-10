import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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