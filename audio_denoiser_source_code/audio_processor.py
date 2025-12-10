import numpy as np
import librosa
import scipy.signal as signal
from typing import Tuple, List

class AudioProcessor:
    """音频信号处理器，实现ZegoAIDenoise算法的信号处理部分"""
    
    def __init__(self, sample_rate: int = 32000, frame_size: int = 320, hop_size: int = 160):
        """
        初始化音频处理器
        
        Args:
            sample_rate: 采样率，默认32kHz
            frame_size: 帧大小，10ms对应320个采样点
            hop_size: 跳跃大小，5ms对应160个采样点
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = 512
        
        # 巴克频带尺度 - 22个子带
        self.bark_bands = self._create_bark_bands()
        
    def _create_bark_bands(self) -> List[Tuple[int, int]]:
        """创建巴克频带尺度，22个子带"""
        # 巴克频带边界频率 (Hz)
        bark_edges = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 
                     1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500]
        
        # 转换为频点索引
        bands = []
        for i in range(len(bark_edges) - 1):
            low_freq = bark_edges[i]
            high_freq = bark_edges[i + 1]
            low_bin = int(low_freq * self.n_fft / self.sample_rate)
            high_bin = int(high_freq * self.n_fft / self.sample_rate)
            bands.append((low_bin, high_bin))
        
        return bands
    
    def stft(self, audio: np.ndarray) -> np.ndarray:
        """
        短时傅里叶变换
        
        Args:
            audio: 输入音频信号
            
        Returns:
            STFT频谱矩阵
        """
        return librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_size, 
                           win_length=self.frame_size, window='hann')
    
    def istft(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        逆短时傅里叶变换
        
        Args:
            stft_matrix: STFT频谱矩阵
            
        Returns:
            重构的时域信号
        """
        return librosa.istft(stft_matrix, hop_length=self.hop_size, 
                            win_length=self.frame_size, window='hann')
    
    def compute_magnitude_spectrum(self, stft_matrix: np.ndarray) -> np.ndarray:
        """计算幅度谱"""
        return np.abs(stft_matrix)
    
    def compute_power_spectrum(self, stft_matrix: np.ndarray) -> np.ndarray:
        """计算功率谱"""
        return np.abs(stft_matrix) ** 2
    
    def decompose_to_subbands(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """
        将频谱分解到巴克子带
        
        Args:
            magnitude_spectrum: 幅度谱
            
        Returns:
            子带能量特征 (22个子带 × 时间帧数)
        """
        n_frames = magnitude_spectrum.shape[1]
        subband_features = np.zeros((len(self.bark_bands), n_frames))
        
        for i, (low_bin, high_bin) in enumerate(self.bark_bands):
            # 计算每个子带的能量
            subband_energy = np.sum(magnitude_spectrum[low_bin:high_bin, :] ** 2, axis=0)
            subband_features[i, :] = subband_energy
        
        return subband_features
    
    def apply_comb_filter(self, audio: np.ndarray, pitch_period: int, M: int = 2) -> np.ndarray:
        """
        应用梳状滤波器增强语音谐波特性
        
        Args:
            audio: 输入音频
            pitch_period: 基音周期
            M: 中心抽头两侧的周期数
            
        Returns:
            滤波后的音频
        """
        # 创建梳状滤波器系数
        filter_length = 2 * M * pitch_period + 1
        filter_coeff = np.zeros(filter_length)
        filter_coeff[M * pitch_period] = 1.0  # 中心抽头
        
        # 应用滤波器
        filtered_audio = signal.lfilter(filter_coeff, [1.0], audio)
        
        return filtered_audio
    
    def estimate_pitch_period(self, audio_frame: np.ndarray) -> int:
        """
        估计基音周期
        
        Args:
            audio_frame: 音频帧
            
        Returns:
            估计的基音周期
        """
        # 使用自相关函数估计基音
        autocorr = np.correlate(audio_frame, audio_frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # 寻找第一个峰值（排除零延迟）
        min_period = int(self.sample_rate / 500)  # 500Hz对应最小周期
        max_period = int(self.sample_rate / 80)   # 80Hz对应最大周期
        
        # 在合理范围内寻找最大自相关
        if len(autocorr) > max_period:
            autocorr_region = autocorr[min_period:max_period]
            max_idx = np.argmax(autocorr_region)
            pitch_period = min_period + max_idx
        else:
            pitch_period = min_period
        
        return pitch_period
    
    def compute_spectral_features(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """
        计算频谱特征用于神经网络输入
        
        Args:
            magnitude_spectrum: 幅度谱
            
        Returns:
            组合特征向量
        """
        n_frames = magnitude_spectrum.shape[1]
        n_features = len(self.bark_bands) + 3  # 子带能量 + 频谱质心 + 频谱滚降 + 频谱平坦度
        
        features = np.zeros((n_features, n_frames))
        
        # 子带能量特征
        subband_energy = self.decompose_to_subbands(magnitude_spectrum)
        features[:len(self.bark_bands), :] = subband_energy
        
        # 频谱质心
        for t in range(n_frames):
            frame_spectrum = magnitude_spectrum[:, t]
            if np.sum(frame_spectrum) > 0:
                features[len(self.bark_bands), t] = np.sum(
                    np.arange(len(frame_spectrum)) * frame_spectrum) / np.sum(frame_spectrum)
        
        # 频谱滚降点 (85%)
        for t in range(n_frames):
            frame_spectrum = magnitude_spectrum[:, t]
            total_energy = np.sum(frame_spectrum)
            if total_energy > 0:
                cumulative_energy = np.cumsum(frame_spectrum)
                rolloff_point = np.where(cumulative_energy >= 0.85 * total_energy)[0]
                if len(rolloff_point) > 0:
                    features[len(self.bark_bands) + 1, t] = rolloff_point[0]
        
        # 频谱平坦度
        for t in range(n_frames):
            frame_spectrum = magnitude_spectrum[:, t] + 1e-8  # 避免除零
            geometric_mean = np.exp(np.mean(np.log(frame_spectrum)))
            arithmetic_mean = np.mean(frame_spectrum)
            features[len(self.bark_bands) + 2, t] = geometric_mean / arithmetic_mean
        
        return features