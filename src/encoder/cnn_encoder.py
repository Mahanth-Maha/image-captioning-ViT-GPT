import torch
import torch.nn as nn
from torchvision import models


class CNNEncoder(nn.Module):
    def __init__(self, backbone="resnet50", proj_dim=512, train_backbone=False):
        super().__init__()
        assert backbone in {"resnet18","resnet34","resnet50","resnet101","resnet152"}
        resnet_ctor = getattr(models, backbone)
        pre_weights = self._get_torchvision_weights(backbone)
        net = resnet_ctor(weights=pre_weights)
        modules = list(net.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.out_dim = net.fc.in_features
        self.proj = nn.Linear(self.out_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        if not train_backbone:
            for p in self.cnn.parameters():
                p.requires_grad = False

    def forward(self, images):
        feats = self.cnn(images)
        feats = feats.flatten(1)
        feats = self.proj(feats)
        feats = self.norm(feats)
        return feats

    def _get_torchvision_weights(self,backbone):
        try:
            return getattr(models, f"{backbone.capitalize()}_Weights").DEFAULT
        except AttributeError:
            try:
                return getattr(models, f"{backbone.upper()}_Weights").DEFAULT
            except AttributeError:
                return None