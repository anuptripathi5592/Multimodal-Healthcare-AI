import torch
import logging
from torch import nn, optim

from src.data.dataloader import get_dataloaders
from src.models.fusion_model import FusionModel

logger = logging.getLogger(__name__)


def train(config):
    train_loader, val_loader = get_dataloaders(config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FusionModel(config).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    epochs = config['training']['epochs']

    logger.info("Training started...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, input_ids, attention_mask, labels in train_loader:

            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}"
        )

    torch.save(model.state_dict(), "artifacts/model.pth")

    logger.info("Training completed & model saved.")


