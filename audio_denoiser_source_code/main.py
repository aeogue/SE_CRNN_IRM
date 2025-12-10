import sys
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import WhiteNoise, Sine
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSlider, QComboBox, QFileDialog,
                            QGroupBox, QProgressBar, QMessageBox, QWidget, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor

from audio_processor import AudioProcessor
from crnn_model import RealTimeDenoiser

class AudioProcessingThread(QThread):
    """音频处理线程"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(str, str, str)  # 原始文件, 噪声文件, 降噪文件
    
    def __init__(self, audio_file, noise_type, noise_level):
        super().__init__()
        self.audio_file = audio_file
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.denoiser = None
        
    def run(self):
        try:
            # 初始化降噪器
            self.denoiser = RealTimeDenoiser()
            self.denoiser.initialize()
            
            # 加载原始音频
            self.progress_updated.emit(10)
            audio, sr = librosa.load(self.audio_file, sr=32000)
            
            # 生成噪声
            self.progress_updated.emit(30)
            noisy_audio = self.add_noise(audio, sr)
            
            # 保存带噪音频
            noisy_file = "noisy_audio.wav"
            sf.write(noisy_file, noisy_audio, sr)
            
            # 进行AI降噪
            self.progress_updated.emit(50)
            enhanced_audio = self.denoiser.process_frame(noisy_audio)
            
            # 保存降噪后音频
            self.progress_updated.emit(80)
            enhanced_file = "enhanced_audio.wav"
            sf.write(enhanced_file, enhanced_audio, sr)
            
            self.progress_updated.emit(100)
            self.processing_finished.emit(self.audio_file, noisy_file, enhanced_file)
            
        except Exception as e:
            print(f"处理错误: {e}")
            self.progress_updated.emit(0)
    
    def add_noise(self, clean_audio, sr):
        """添加指定类型的噪声"""
        if self.noise_type == "white":
            # 白噪声
            noise = np.random.normal(0, self.noise_level * 0.01, len(clean_audio))
        elif self.noise_type == "keyboard":
            # 键盘敲击声模拟
            noise = self.simulate_keyboard_noise(len(clean_audio), sr)
        elif self.noise_type == "mouse":
            # 鼠标点击声模拟
            noise = self.simulate_mouse_clicks(len(clean_audio), sr)
        elif self.noise_type == "restaurant":
            # 餐厅嘈杂声模拟
            noise = self.simulate_restaurant_noise(len(clean_audio), sr)
        else:
            noise = np.zeros_like(clean_audio)
        
        noisy_audio = clean_audio + noise * self.noise_level
        return np.clip(noisy_audio, -1.0, 1.0)
    
    def simulate_keyboard_noise(self, length, sr):
        """模拟键盘敲击声"""
        noise = np.zeros(length)
        click_duration = int(0.05 * sr)  # 50ms敲击声
        interval = int(0.2 * sr)  # 200ms间隔
        
        for i in range(0, length, interval):
            if i + click_duration < length:
                # 创建短促的敲击声
                click = np.random.normal(0, 0.1, click_duration) * np.hanning(click_duration)
                noise[i:i+click_duration] += click
        
        return noise
    
    def simulate_mouse_clicks(self, length, sr):
        """模拟鼠标点击声"""
        noise = np.zeros(length)
        click_duration = int(0.02 * sr)  # 20ms点击声
        interval = int(0.5 * sr)  # 500ms间隔
        
        for i in range(0, length, interval):
            if i + click_duration < length:
                # 创建更短促的点击声
                click = np.random.normal(0, 0.05, click_duration) * np.hanning(click_duration)
                noise[i:i+click_duration] += click
        
        return noise
    
    def simulate_restaurant_noise(self, length, sr):
        """模拟餐厅嘈杂声"""
        # 使用多个正弦波模拟人声嘈杂
        noise = np.zeros(length)
        for freq in [100, 200, 300, 400, 500]:
            sine_wave = 0.02 * np.sin(2 * np.pi * freq * np.arange(length) / sr)
            noise += sine_wave
        
        # 添加随机噪声成分
        noise += 0.01 * np.random.normal(0, 1, length)
        
        return noise

class SpectrogramWidget(FigureCanvas):
    """频谱图显示组件"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel('时间 (s)')
        self.axes.set_ylabel('频率 (Hz)')
        
    def plot_spectrogram(self, audio_file, title="频谱图"):
        """绘制音频频谱图"""
        try:
            audio, sr = librosa.load(audio_file, sr=32000)
            
            # 计算频谱图
            D = librosa.stft(audio)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            self.axes.clear()
            img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=self.axes)
            self.axes.set_title(title)
            self.fig.colorbar(img, ax=self.axes, format='%+2.0f dB')
            
            self.draw()
            
        except Exception as e:
            print(f"绘制频谱图错误: {e}")

class AIDenoiseApp(QMainWindow):
    """AI降噪演示应用程序"""
    
    def __init__(self):
        super().__init__()
        self.audio_files = {}
        self.processing_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("ZegoAIDenoise AI降噪演示")
        self.setFixedSize(1200, 800)
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("🚨 ZegoAIDenoise AI降噪演示系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
                margin: 10px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()
        
        # 文件选择
        file_layout = QVBoxLayout()
        self.load_btn = QPushButton("加载音频文件")
        self.load_btn.clicked.connect(self.load_audio_file)
        file_layout.addWidget(self.load_btn)
        
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("font-size: 12px; color: #666;")
        file_layout.addWidget(self.file_label)
        
        control_layout.addLayout(file_layout)
        
        # 噪声类型选择
        noise_layout = QVBoxLayout()
        noise_label = QLabel("噪声类型:")
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["白噪声", "键盘敲击声", "鼠标点击声", "餐厅嘈杂声"])
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_combo)
        
        control_layout.addLayout(noise_layout)
        
        # 噪声强度
        level_layout = QVBoxLayout()
        level_label = QLabel("噪声强度:")
        self.level_slider = QSlider(Qt.Horizontal)
        self.level_slider.setRange(1, 10)
        self.level_slider.setValue(5)
        self.level_value = QLabel("5")
        level_layout.addWidget(level_label)
        level_layout.addWidget(self.level_slider)
        level_layout.addWidget(self.level_value)
        self.level_slider.valueChanged.connect(self.update_level_value)
        
        control_layout.addLayout(level_layout)
        
        # 处理按钮
        process_layout = QVBoxLayout()
        self.process_btn = QPushButton("开始AI降噪")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        process_layout.addWidget(self.process_btn)
        
        control_layout.addLayout(process_layout)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # 频谱图显示区域
        spectro_group = QGroupBox("频谱图对比")
        spectro_layout = QHBoxLayout()
        
        # 创建三个频谱图显示区域
        self.original_spectro = SpectrogramWidget(self, width=4, height=3)
        self.noisy_spectro = SpectrogramWidget(self, width=4, height=3)
        self.enhanced_spectro = SpectrogramWidget(self, width=4, height=3)
        
        spectro_layout.addWidget(self.original_spectro)
        spectro_layout.addWidget(self.noisy_spectro)
        spectro_layout.addWidget(self.enhanced_spectro)
        
        spectro_group.setLayout(spectro_layout)
        main_layout.addWidget(spectro_group)
        
        # 信息显示
        info_label = QLabel("""
        💡 ZegoAIDenoise 技术特点:
        • 轻量级神经网络降噪，性能开销低
        • 采用传统算法与深度学习结合的Hybrid方法  
        • 使用22个巴克频带子带分解
        • CRNN网络模型（卷积层+GRU层）
        • 对稳态和非稳态噪声均有良好效果
        """)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #e8f4fd;
                border: 1px solid #b3d9ff;
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
                font-size: 12px;
            }
        """)
        main_layout.addWidget(info_label)
        
    def update_level_value(self, value):
        """更新噪声强度显示"""
        self.level_value.setText(str(value))
    
    def load_audio_file(self):
        """加载音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "", "音频文件 (*.wav *.mp3 *.m4a)")
        
        if file_path:
            self.audio_files['original'] = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.process_btn.setEnabled(True)
            
            # 显示原始音频频谱图
            self.original_spectro.plot_spectrogram(file_path, "原始音频")
    
    def start_processing(self):
        """开始AI降噪处理"""
        if not self.audio_files.get('original'):
            QMessageBox.warning(self, "警告", "请先选择音频文件")
            return
        
        # 禁用按钮，显示进度条
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 获取参数
        noise_type_map = {
            "白噪声": "white",
            "键盘敲击声": "keyboard", 
            "鼠标点击声": "mouse",
            "餐厅嘈杂声": "restaurant"
        }
        noise_type = noise_type_map[self.noise_combo.currentText()]
        noise_level = self.level_slider.value()
        
        # 启动处理线程
        self.processing_thread = AudioProcessingThread(
            self.audio_files['original'], noise_type, noise_level)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.processing_completed)
        self.processing_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def processing_completed(self, original_file, noisy_file, enhanced_file):
        """处理完成回调"""
        self.audio_files.update({
            'noisy': noisy_file,
            'enhanced': enhanced_file
        })
        
        # 更新频谱图显示
        self.original_spectro.plot_spectrogram(original_file, "原始音频")
        self.noisy_spectro.plot_spectrogram(noisy_file, "带噪音频")
        self.enhanced_spectro.plot_spectrogram(enhanced_file, "降噪后音频")
        
        # 重置界面
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "完成", "AI降噪处理完成！\n可以播放音频对比效果。")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序信息
    app.setApplicationName("ZegoAIDenoise Demo")
    app.setApplicationVersion("1.0")
    
    # 创建并显示主窗口
    window = AIDenoiseApp()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()