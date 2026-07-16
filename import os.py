import os
from pathlib import Path

# Base project directory
base_dir = Path(__file__).parent

# Define all files and their contents
files_content = {
    "main.py": '''import yaml
import logging
from src.config import Config
from src.train import train
from src.evaluate import evaluate
from src.predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Multimodal Healthcare AI application")
    
    # Load configuration
    config = Config("configs/config.yaml")
    
    # Train model
    logger.info("Training model...")
    train(config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluate(config)
    
    # Make predictions
    logger.info("Making predictions...")
    predict(config)
    
    logger.info("Application completed successfully")
''',

    "requirements.txt": '''torch==2.0.0
torchvision==0.15.0
transformers==4.30.0
pyyaml==6.0
numpy==1.24.0
scikit-learn==1.3.0
pillow==9.5.0
pytorch-lightning==2.0.0
''',

    ".gitignore": '''__pycache__/
*.pyc
.DS_Store
*.egg-info/
dist/
build/
.venv/
venv/
.env
data/raw/
data/processed/
artifacts/
*.log
.vscode/
''',

    "README.md": '''# Multimodal Healthcare AI

A deep learning platform combining vision and text modalities for healthcare diagnostics.

## Project Structure
- `src/` - Source code
- `configs/` - Configuration files
- `data/` - Data directory
- `tests/` - Unit tests
- `api/` - API endpoints
- `ui/` - User interface

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Features
- Vision Encoder (ResNet50)
- Text Encoder (BERT)
- Multimodal Fusion Model
- Medical Image & Text Processing
''',

    "configs/config.yaml": '''# Model Configuration
model:
  vision_encoder:
    type: "resnet50"
    pretrained: true
    output_dim: 512
  
  text_encoder:
    type: "bert"
    pretrained: true
    output_dim: 512
  
  fusion_model:
    fusion_type: "concat"
    hidden_dim: 256
    num_classes: 5

# Training Configuration
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  loss_function: "cross_entropy"
  device: "cuda"

# Data Configuration
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  num_workers: 4
  image_size: 224

# Other
seed: 42
''',

    "src/__init__.py": '''# Multimodal Healthcare AI Module
''',

    "src/config.py": '''import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
''',

    "src/train.py": '''import torch
import logging
from src.config import Config
from src.data.dataloader import get_dataloaders
from src.models.fusion_model import FusionModel

logger = logging.getLogger(__name__)

def train(config: Config):
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(config)
    
    logger.info("Initializing model...")
    model = FusionModel(config)
    
    logger.info("Training started...")
    # Training loop implementation
    logger.info("Training completed!")
''',

    "src/evaluate.py": '''import logging
from src.config import Config

logger = logging.getLogger(__name__)

def evaluate(config: Config):
    logger.info("Evaluation started...")
    # Evaluation logic
    logger.info("Evaluation completed!")
''',

    "src/predict.py": '''import logging
from src.config import Config

logger = logging.getLogger(__name__)

def predict(config: Config):
    logger.info("Prediction started...")
    # Prediction logic
    logger.info("Prediction completed!")
''',

    "src/data/__init__.py": '''# Data module
''',

    "src/data/preprocess.py": '''import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def preprocess_image(image_path: str) -> any:
    """Preprocess medical images"""
    logger.info(f"Preprocessing image: {image_path}")
    # Image preprocessing logic
    pass

def preprocess_text(text: str) -> str:
    """Preprocess medical text data"""
    logger.info("Preprocessing text")
    # Text preprocessing logic
    return text.lower().strip()
''',

    "src/data/dataset.py": '''from torch.utils.data import Dataset

class HealthcareDataset(Dataset):
    def __init__(self, images: list, texts: list, labels: list):
        self.images = images
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'text': self.texts[idx],
            'label': self.labels[idx]
        }
''',

    "src/data/dataloader.py": '''from torch.utils.data import DataLoader
from src.data.dataset import HealthcareDataset
from src.config import Config

def get_dataloaders(config: Config):
    # Load data from disk
    train_dataset = HealthcareDataset([], [], [])
    val_dataset = HealthcareDataset([], [], [])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    return train_loader, val_loader
''',

    "src/models/__init__.py": '''# Models module
''',

    "src/models/vision_encoder.py": '''import torch.nn as nn
from torchvision import models

class VisionEncoder(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Linear(2048, output_dim)
    
    def forward(self, x):
        return self.backbone(x)
''',

    "src/models/text_encoder.py": '''import torch.nn as nn
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
''',

    "src/models/fusion_model.py": '''import torch
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
''',

    "src/models/losses.py": '''import torch.nn as nn

class CustomHealthcareLoss(nn.Module):
    def __init__(self, weight: list = None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
    
    def forward(self, predictions, targets):
        return self.ce_loss(predictions, targets)
''',

    "src/utils/__init__.py": '''# Utils module
''',

    "src/utils/metrics.py": '''from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(predictions, targets):
    acc = accuracy_score(targets, predictions)
    prec = precision_score(targets, predictions, average='weighted')
    rec = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
''',

    "src/utils/seed.py": '''import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
''',

    "tests/__init__.py": '''# Tests module
''',

    "tests/test_models.py": '''import unittest
from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder

class TestModels(unittest.TestCase):
    def test_vision_encoder(self):
        encoder = VisionEncoder(output_dim=512)
        self.assertIsNotNone(encoder)
    
    def test_text_encoder(self):
        encoder = TextEncoder(output_dim=512)
        self.assertIsNotNone(encoder)

if __name__ == '__main__':
    unittest.main()
''',

    "api/__init__.py": '''# API module
''',

    "api/app.py": '''from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Prediction logic
    return jsonify({'result': 'prediction'})

if __name__ == '__main__':
    app.run(debug=True)
''',

    "ui/__init__.py": '''# UI module
''',

    "docker/Dockerfile": '''FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
''',

    "docker/.dockerignore": '''__pycache__
.git
.gitignore
.venv
venv
*.log
artifacts/
data/raw/
data/processed/
''',
}

def create_all_files():
    """Create all project files"""
    print("=" * 60)
    print("Creating Multimodal Healthcare AI Project Structure")
    print("=" * 60)
    
    # Create all files
    for file_path, content in files_content.items():
        full_path = base_dir / file_path
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Created: {file_path}")
    
    # Create empty data directories
    data_dirs = [
        "data/raw",
        "data/processed", 
        "data/interim",
        "data/splits",
        "artifacts"
    ]
    
    for dir_path in data_dirs:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    print("\n" + "=" * 60)
    print("✅ Project created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. pip install -r requirements.txt")
    print("2. python main.py")
    print("=" * 60)

if __name__ == "__main__":
    create_all_files()