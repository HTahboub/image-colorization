import wandb


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    pass


def evaluate(model, data_loader, device):
    pass


def train(model, optimizer, data_loader, device, epochs, print_freq):
    wandb.init(project="ddcolor")
    wandb.watch(model)
    for epoch in range(1, epochs):
        print("-" * 10 + f" Epoch {epoch:03d} " + "-" * 10)
        train_loss = train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq
        )
        print(f"Epoch {epoch} - Loss: {train_loss}")
        wandb.log({"epoch": epoch, "loss": train_loss})

        # Evaluate the model
        val_loss = evaluate(model, data_loader, device)
        print(f"Validation Loss: {val_loss}\n")
        wandb.log({"epoch": epoch, "val_loss": val_loss})
    wandb.finish()
