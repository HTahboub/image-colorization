import torch
from data import make_dataset
from losses import CombinedLoss
from models.model import DDColor

import wandb


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, log_freq):
    pass


def evaluate(model, data_loader, device):
    pass


def train(model, criterion, optimizer, data_loader, device, epochs, log_freq):
    wandb.watch(model)
    for epoch in range(1, epochs):
        print("-" * 10 + f" Epoch {epoch:03d} " + "-" * 10)
        train_loss = train_one_epoch(
            model, criterion, optimizer, data_loader, device, epoch, log_freq
        )
        print(f"Epoch {epoch} - Loss: {train_loss}")
        wandb.log({"epoch": epoch, "loss": train_loss})

        # Evaluate the model
        val_loss = evaluate(model, data_loader, device)
        print(f"Validation Loss: {val_loss}\n")
        wandb.log({"epoch": epoch, "val_loss": val_loss})
    wandb.finish()


if __name__ == "__main__":
    wandb.init(project="ddcolor")

    batch_size = 64
    lr = 1e-3
    device = torch.device("cuda")
    train_loader, val_loader = make_dataset(batch_size, validation=True)
    model = DDColor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = CombinedLoss().to(device)
    epochs = 100
    log_freq = 10


