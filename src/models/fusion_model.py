import torch
import torch.nn as nn
from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder
from src.config import Config

class FusionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        vision_dim = config['model']['vision_encoder']['output_dim']
        text_dim = config['model']['text_encoder']['output_dim']
        hidden_dim = config['model']['fusion_model']['hidden_dim']
        num_classes = config['model']['fusion_model']['num_classes']
        
        self.vision_encoder = VisionEncoder(vision_dim)
        self.text_encoder = TextEncoder(text_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, images, input_ids, attention_mask):
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        fused = torch.cat([vision_features, text_features], dim=1)
        return self.fusion(fused)
