import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm  # 进度条（可选，需安装：pip install tqdm）

# 导入你的模型和训练器
from crnn_denoise import CRNNDenoiseModel, DenoiseTrainer 
from audio_processor import AudioProcessor

def train_denoise_model(
    clean_train_dir: str,
    noisy_train_dir: str,
    clean_val_dir: str,
    noisy_val_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    save_path: str = "denoise_crnn_best.pth",
    device: str = None
):
    """
    训练CRNN降噪模型（完整流程）
    
    Args:
        clean_train_dir: 训练集干净语音目录
        noisy_train_dir: 训练集带噪语音目录
        clean_val_dir: 验证集干净语音目录
        noisy_val_dir: 验证集带噪语音目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 初始学习率
        save_path: 最佳模型保存路径
        device: 训练设备（自动检测GPU/CPU）
    """
    # 1. 设备初始化
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用训练设备：{device}")

    # 2. 构建数据集和DataLoader
    print("加载数据集...")
    train_dataset = DenoiseDataset(clean_train_dir, noisy_train_dir)
    val_dataset = DenoiseDataset(clean_val_dir, noisy_val_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 根据CPU核心数调整
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )

    # 3. 初始化模型和训练器
    model = CRNNDenoiseModel(
        input_dim=25,    # 22子带 + 3频谱特征
        hidden_dim=128,
        num_layers=3,
        output_dim=22    # 22个子带增益
    ).to(device)
    
    trainer = DenoiseTrainer(model, learning_rate=learning_rate)
    best_val_loss = float("inf")

    # 4. 训练循环
    print(f"开始训练（共{epochs}轮）...")
    for epoch in range(epochs):
        # ---------------- 训练阶段 ----------------
        model.train()
        train_losses = []
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)")
        
        for batch in train_bar:
            # 数据移到设备
            noisy_feat = batch["noisy_features"].to(device)
            target_gain = batch["target_gains"].to(device)
            clean_spec = batch["clean_spectrum"].to(device)
            noisy_spec = batch["noisy_spectrum"].to(device)
            
            # 单步训练
            loss = trainer.train_step(noisy_feat, target_gain, clean_spec, noisy_spec)
            train_losses.append(loss)
            
            # 更新进度条
            train_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_train_loss = np.mean(train_losses)

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_losses = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)")
        
        with torch.no_grad():
            for batch in val_bar:
                noisy_feat = batch["noisy_features"].to(device)
                target_gain = batch["target_gains"].to(device)
                clean_spec = batch["clean_spectrum"].to(device)
                noisy_spec = batch["noisy_spectrum"].to(device)
                
                loss = trainer.validate(noisy_feat, target_gain, clean_spec, noisy_spec)
                val_losses.append(loss)
                val_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_val_loss = np.mean(val_losses)

        # ---------------- 学习率调整 + 模型保存 ----------------
        # 调整学习率（基于验证损失）
        trainer.scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "sample_rate": train_dataset.sample_rate
            }, save_path)
            print(f"✅ 保存最佳模型 (Val Loss: {best_val_loss:.4f})")

        # ---------------- 打印日志 ----------------
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Current LR: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

    print(f"训练完成！最佳验证损失：{best_val_loss:.4f} (模型已保存至 {save_path})")
    return model

# -------------------------- 训练入口 --------------------------
if __name__ == "__main__":
    # 配置数据集路径（替换为你的实际路径）
    CONFIG = {
        "clean_train_dir": "./dataset/train/clean",
        "noisy_train_dir": "./dataset/train/noisy",
        "clean_val_dir": "./dataset/val/clean",
        "noisy_val_dir": "./dataset/val/noisy",
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "save_path": "./denoise_crnn_best.pth"
    }
'''
dataset/
├── train/
│   ├── clean/  # 干净语音文件（如 clean_001.wav）
│   └── noisy/  # 带噪语音文件（如 clean_001.wav）
└── val/
    ├── clean/
    └── noisy/
'''
    # 启动训练
    trained_model = train_denoise_model(**CONFIG)