import torch.nn as nn
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, output_dim)
    
    def forward(self, x):
        return self.backbone(x)
