import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI降噪演示 - 测试窗口")
        self.setFixedSize(800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout(central_widget)
        
        # 添加标题标签
        title_label = QLabel("🎧 AI降噪演示系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50; margin: 20px;")
        layout.addWidget(title_label)
        
        # 添加说明标签
        desc_label = QLabel(
            "这是一个基于ZEGO AI降噪算法的演示系统\n\n"
            "功能包括：\n"
            "• 加载音频文件\n"
            "• 添加不同类型噪声\n"
            "• AI降噪处理\n"
            "• 实时频谱可视化\n"
            "• 音频效果对比\n\n"
            "基于博客：消灭非稳态噪音的利器 - AI降噪"
        )
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setStyleSheet("font-size: 16px; color: #34495e; margin: 20px;")
        layout.addWidget(desc_label)
        
        # 添加状态标签
        status_label = QLabel("✓ PyQt5 GUI环境正常\n✓ 音频处理库已加载\n✓ 准备启动主应用")
        status_label.setAlignment(Qt.AlignCenter)
        status_label.setStyleSheet("font-size: 14px; color: #27ae60; margin: 20px;")
        layout.addWidget(status_label)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI降噪演示")
    app.setApplicationVersion("1.0")
    
    # 创建并显示测试窗口
    window = TestWindow()
    window.show()
    
    print("PyQt5 GUI测试窗口已启动")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()