# 🥧 TartVision

基于 PyTorch 的多模态（视觉 + 温度序列）端到端蛋挞烘焙状态分类框架。

通过结合 U-Net 图像分割与 MobileNetV3，并引入独创的**温度门控融合机制 (Temperature Gated Fusion)**，本项目能够有效克服烤箱内部蒸汽、油烟及光照干扰，精准预测烘焙状态。

## ⚙️ 1. 环境安装

```bash
git clone https://github.com/Makogi/TartVision.git
cd TartVision
pip install -r requirements.txt
```

## 📂 2. 数据与权重准备

在运行代码前，请确保按以下结构准备好**预训练权重**和**数据集**：

1. **下载权重**：[点击此处下载 best_model_weights.pth](https://github.com/Makogi/TartVision/releases/download/v1.0.0/best_model_weights.pth)，并放入 `weights/` 目录。
2. **准备数据**：将您的特征 CSV 表格和图片文件夹 (`raw_data`) 放入 `dataset/` 目录。（注：CSV 中的图片路径须为相对路径）

最终项目目录应如下所示：
```text
TartVision/
├── dataset/                                   # 你的数据集
│   ├── all_experiments_features_final.csv
│   └── raw_data/images/...
├── weights/                                   # 模型权重
│   └── best_model_weights.pth
├── tart_vision/                               # 核心算法库
├── train.py                                   # 训练脚本
└── ...
```

## 🚀 3. 快速开始

环境与数据就绪后，直接运行以下命令启动端到端训练：

```bash
python train.py \
    --csv_path dataset/all_experiments_features_final.csv \
    --seg_weights weights/best_model_weights.pth \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4
```

训练完成后，融合分类模型的最佳权重将自动保存在 `weights/best_fusion_model.pth`。

## 🛠️ 4. 核心特性
- **模块化解耦**: 分割网络 (U-Net) 与分类网络 (MobileNetV3) 分离，易于二次开发。
- **特征工程**: 自动计算平台期积热特征 (`time_at_100`)。
- **抗干扰流**: 内置针对烤箱场景（失焦、起雾、光线波动）的 Albumentations 数据增强流。
