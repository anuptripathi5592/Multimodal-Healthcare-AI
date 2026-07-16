import torch.nn as nn
from transformers import BertModel, BertTokenizer

class TextEncoder(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, output_dim)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.pooler_output
        return self.fc(pooled)
