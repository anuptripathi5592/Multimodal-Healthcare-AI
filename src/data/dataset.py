import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
import torch


class MultimodalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        available_cols = [
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
            "Pleural_Thickening",
            "Hernia",
        ]

        self.label_cols = [col for col in available_cols if col in self.data.columns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)

        text = str(row['text'])

        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        labels = torch.tensor(
            row[self.label_cols].values.astype(float),
            dtype=torch.float
        )

        return (
            image,
            encoded['input_ids'].squeeze(0),
            encoded['attention_mask'].squeeze(0),
            labels
        )
