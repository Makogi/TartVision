import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tart_vision.config import DEVICE, IMG_CROP_SIZE, LAYOUTS, TEMP_FEATURE_DIM

class EndToEndDataset(Dataset):
    def __init__(self, dataframe, seg_model, transform=None):
        self.dataframe = dataframe
        self.seg_model = seg_model
        self.transform = transform
        self.seg_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        self.layouts = LAYOUTS

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        target_id = row['tart_id']
        experiment_id = row['experiment_id']

        # 1. 读取原图
        full_image = Image.open(image_path).convert("RGB")
        full_image_np = np.array(full_image)

        # 2. UNet 分割找轮廓
        input_tensor = self.seg_transform(image=full_image_np)["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.seg_model(input_tensor)
            preds = (torch.sigmoid(logits) > 0.5).float()
        full_mask_resized = preds.squeeze().cpu().numpy().astype(np.uint8)

        contours, _ = cv2.findContours(full_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 3. 匹配布局并定位
        layout_key = experiment_id.split('experiment_')[-1]
        if layout_key not in self.layouts:
            layout_key = '4_center'
        config = self.layouts[layout_key]

        if len(contours) != config['num_tarts']:
            return torch.zeros(3, IMG_CROP_SIZE, IMG_CROP_SIZE), torch.zeros(TEMP_FEATURE_DIM), torch.tensor(-1, dtype=torch.long)

        centroids =[cv2.moments(c) for c in contours if cv2.contourArea(c) > 0]
        if len(centroids) != config['num_tarts']:
            return torch.zeros(3, IMG_CROP_SIZE, IMG_CROP_SIZE), torch.zeros(TEMP_FEATURE_DIM), torch.tensor(-1, dtype=torch.long)

        centroids_coords = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in centroids]
        contours_sorted = sorted(zip(contours, centroids_coords), key=lambda item: item[1][1] * 1000 + item[1][0])

        try:
            target_index = config['labels_in_order'].index(target_id)
            target_contour_resized = contours_sorted[target_index][0]
        except (ValueError, IndexError):
            return torch.zeros(3, IMG_CROP_SIZE, IMG_CROP_SIZE), torch.zeros(TEMP_FEATURE_DIM), torch.tensor(-1, dtype=torch.long)

        # 4. 裁剪目标蛋挞
        x, y, w, h = cv2.boundingRect(target_contour_resized)
        orig_h, orig_w, _ = full_image_np.shape
        scale_x, scale_y = orig_w / 256, orig_h / 256
        x_orig, y_orig = int(x * scale_x), int(y * scale_y)
        w_orig, h_orig = int(w * scale_x), int(h * scale_y)

        tart_image_crop = full_image_np[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

        if tart_image_crop.size == 0:
            return torch.zeros(3, IMG_CROP_SIZE, IMG_CROP_SIZE), torch.zeros(TEMP_FEATURE_DIM), torch.tensor(-1, dtype=torch.long)

        # 5. 数据增强
        if self.transform:
            augmented = self.transform(image=tart_image_crop)
            tart_image_tensor = augmented['image']
        else:
            tart_image_tensor = torch.zeros(3, IMG_CROP_SIZE, IMG_CROP_SIZE) # 兜底

        # 6. 获取温度与标签特征
        feats = row[['temperature', 'temp_gradient', 'time_at_100']].values.astype(np.float32)
        temp_features = torch.tensor(feats, dtype=torch.float32)
        label = torch.tensor(row['cooking_state'], dtype=torch.long)

        return tart_image_tensor, temp_features, label