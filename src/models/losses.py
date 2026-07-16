import torch.nn as nn

class CustomHealthcareLoss(nn.Module):
    def __init__(self, weight: list = None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, predictions, targets):
        return self.ce_loss(predictions, targets)
