import torch

# 硬件配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型超参数
TEMP_FEATURE_DIM = 3
NUM_CLASSES = 3
HIDDEN_DIM = 128
CNN_OUTPUT_DIM = 1024

# 图像处理参数
IMG_CROP_SIZE = 128

# 烤盘布局规则
LAYOUTS = {
    '4_center': {'num_tarts': 4, 'labels_in_order':['E', 'D', 'B', 'C']},
    '6_left_right': {'num_tarts': 6, 'labels_in_order':['E', 'B', 'F', 'C', 'G', 'D']},
    '6_up_down': {'num_tarts': 6, 'labels_in_order':['B', 'C', 'D', 'E', 'F', 'G']}
}