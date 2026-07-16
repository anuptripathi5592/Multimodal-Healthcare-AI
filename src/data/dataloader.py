from src.data.dataset import MultimodalDataset

def get_dataloaders(config):
    train_dataset = MultimodalDataset(config['data']['train_path'])

    print("Dataset length:", len(train_dataset))  # 👈 ADD THIS

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    return train_loader, train_loader

