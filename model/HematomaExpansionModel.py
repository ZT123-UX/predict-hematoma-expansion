import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
import matplotlib.pyplot as plt


class HematomaExpansionModel(nn.Module):
    def __init__(self, struct_feat_dim, image_feat_dim, hidden_dim=256):
        super().__init__()

        self.ct_resnet = models.resnet50(pretrained=True)
        self.ct_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.ct_resnet.fc = nn.Linear(2048, hidden_dim)

        self.mask_resnet = models.resnet18(pretrained=True)
        self.mask_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.mask_resnet.fc = nn.Linear(512, hidden_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim*2, nhead=8),
            num_layers=4
        )

        self.mlp = nn.Sequential(
            nn.Linear(struct_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp_image_feat = nn.Sequential(
            nn.Linear(image_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, ct_images, mask_images, struct_data, image_feat_data):
        B, S, C, H, W = ct_images.shape
        ct_feats, mask_feats = [], []

        for i in range(S):
            ct_feat = self.ct_resnet(ct_images[:, i, :, :, :])  # (B, hidden_dim)
            mask_feat = self.mask_resnet(mask_images[:, i, :, :, :])  # (B, hidden_dim)
            ct_feats.append(ct_feat)
            mask_feats.append(mask_feat)

        ct_feats = torch.stack(ct_feats, dim=1)  # (B, S, hidden_dim)
        mask_feats = torch.stack(mask_feats, dim=1)  # (B, S, hidden_dim)
        fusion_feats = torch.cat([ct_feats, mask_feats], dim=-1)  # (B, S, hidden_dim * 2)

        fusion_feats = rearrange(fusion_feats, 'b s d -> s b d')
        fusion_feats = self.transformer(fusion_feats)  # (S, B, D)
        fusion_feats = torch.mean(fusion_feats, dim=0)  # (B, D)

        struct_feats = self.mlp(struct_data)  # (B, hidden_dim)
        image_feats = self.mlp_image_feat(image_feat_data)  # (B, hidden_dim)

        fused_feats = torch.cat([fusion_feats, struct_feats], dim=1)
        output = self.fc(fused_feats)  # (B, 1)
        return output
