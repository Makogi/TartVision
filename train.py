import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 从我们自己写的库中导入所需的模块
from tart_vision.config import DEVICE, TEMP_FEATURE_DIM, NUM_CLASSES, HIDDEN_DIM, CNN_OUTPUT_DIM, IMG_CROP_SIZE
from tart_vision.models.unet import UNet
from tart_vision.models.fusion import FusionModelWithCNN
from tart_vision.data.preprocess import preprocess_features
from tart_vision.data.dataset import EndToEndDataset


def parse_args():
    parser = argparse.ArgumentParser(description="TartVision End-to-End Training Script")
    parser.add_argument('--csv_path', type=str, required=True, help='特征CSV路径')
    parser.add_argument('--seg_weights', type=str, required=True, help='预训练的UNet权重路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_path', type=str, default='weights/best_fusion_model.pth', help='模型保存路径')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 读取与预处理数据
    print(f"加载数据集: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    df = preprocess_features(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['cooking_state'])

    # 2. 定义增强流
    crop_transform = A.Compose([
        A.Resize(height=IMG_CROP_SIZE, width=IMG_CROP_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 3. 加载 U-Net (冻结状态)
    print("加载 U-Net 模型...")
    seg_model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    seg_model.load_state_dict(torch.load(args.seg_weights, map_location=DEVICE))
    seg_model.eval()

    # 4. 构建 Dataloader
    train_dataset = EndToEndDataset(train_df, seg_model, transform=crop_transform)
    val_dataset = EndToEndDataset(val_df, seg_model, transform=crop_transform)

    def collate_fn(batch):
        # 过滤掉返回值为 -1 的异常数据（比如没检测到对应轮廓）
        batch = list(filter(lambda x: x[2] != -1, batch))
        if len(batch) == 0: return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 5. 初始化主融合模型
    model = FusionModelWithCNN(TEMP_FEATURE_DIM, NUM_CLASSES, HIDDEN_DIM, CNN_OUTPUT_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6. 开始训练循环
    print(f"开始训练 (Device: {DEVICE})...")
    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in pbar:
            if batch is None: continue

            img_tensors, temp_feat, labels = batch
            img_tensors, temp_feat, labels = img_tensors.to(DEVICE), temp_feat.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(img_tensors, temp_feat)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 验证阶段
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue

                img_tensors, temp_feat, labels = batch
                img_tensors, temp_feat, labels = img_tensors.to(DEVICE), temp_feat.to(DEVICE), labels.to(DEVICE)

                outputs = model(img_tensors, temp_feat)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        current_val_accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{args.epochs} -> Val Accuracy: {current_val_accuracy:.2f}%")

        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            torch.save(model.state_dict(), args.save_path)
            print(f"==> 发现更好的模型！已保存至 {args.save_path}")

    print(f"\n训练结束！最佳验证集准确率: {best_val_accuracy:.2f}%")


if __name__ == "__main__":
    main()