import torch
import torch.nn as nn
import timm

class TemperatureGatedFusion(nn.Module):
    def __init__(self, temp_dim, img_dim, hidden_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(temp_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, temp_feat):
        gate = self.gate_net(temp_feat)
        weighted_img_feat = img_feat * gate
        return torch.cat([weighted_img_feat, temp_feat], dim=1)

class FusionModelWithCNN(nn.Module):
    def __init__(self, temp_feature_dim, num_classes, hidden_dim, cnn_out_dim):
        super(FusionModelWithCNN, self).__init__()
        self.image_branch = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out_dummy = self.image_branch(dummy)
            self.backbone_out_dim = out_dummy.shape[1]

        self.img_projector = nn.Sequential(
            nn.Linear(self.backbone_out_dim, cnn_out_dim),
            nn.ReLU()
        )

        for param in list(self.image_branch.parameters())[:-10]:
            param.requires_grad = False

        self.fusion_gate = TemperatureGatedFusion(temp_feature_dim, cnn_out_dim, hidden_dim)

        fused_dim = cnn_out_dim + temp_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_tensor, temp_features):
        h_img_raw = self.image_branch(image_tensor)
        h_img = self.img_projector(h_img_raw)
        output_fused = self.fusion_gate(h_img, temp_features)
        output = self.classifier(output_fused)
        return output