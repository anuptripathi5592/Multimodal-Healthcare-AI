import os
import torch
from torchvision import transforms
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

from src.models.fusion_model import FusionModel
from src.config import Config

config = Config("configs/config.yaml")

# -------------------------------
# Download model from Hugging Face
# -------------------------------
MODEL_PATH = hf_hub_download(
    repo_id="anuptripathi5592/multimodal-healthcare-ai",
    filename="model.pth"
)

# -------------------------------
# Load Model
# -------------------------------
model = FusionModel(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

classes = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia"
]


def predict(image, text):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        logits = model(
            image_tensor,
            encoded["input_ids"],
            encoded["attention_mask"]
        )

        probs = torch.sigmoid(logits)[0]

    predictions = []
    confidences = []

    for i, prob in enumerate(probs):
        if prob > 0.5:
            predictions.append(classes[i])
            confidences.append(round(prob.item() * 100, 2))

    if not predictions:
        predictions = ["No Significant Finding"]
        confidences = [100.0]

    return {
        "prediction": predictions,
        "confidence": confidences
    }