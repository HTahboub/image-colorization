from glob import glob

import torch
import torchvision

import wandb
from ddcolor.data import get_transform, make_dataset
from ddcolor.losses import CombinedLoss
from ddcolor.models.model import DDColor
from ddcolor.models.utils import image_list_to_tensor, preprocess_images

TESTING = False


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    log_freq,
    samples,
    sample_log_freq,
    scheduler
):
    total_loss = 0
    for i, images in enumerate(data_loader):
        images = images.to(device)
        images, images_lab, images_rgb = preprocess_images(images)
        images_ab = images_lab[:, 1:, ...]
        optimizer.zero_grad()
        output, colored_images = model(images)
        loss = criterion(output, colored_images, images_ab, images_rgb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i == len(data_loader) // 2 and epoch > 1:
            scheduler.step()
        if i % log_freq == 0:
            print(f"Epoch {epoch} - Iteration {i} - Loss: {loss.item()}")
            wandb.log(
                {
                    "epoch": epoch,
                    "iteration": i + epoch * len(data_loader),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss": loss.item(),
                }
            )
        # visualize samples
        if i % sample_log_freq == 0:
            samples_gray, _, samples_rgb = preprocess_images(samples)
            _, sample_colored_images = model(samples_gray.to(device))
            samples_rgb = samples_rgb.cpu()
            samples_gray = samples_gray.cpu()
            sample_colored_images = sample_colored_images.cpu()
            # reverse_transform = get_transform(reverse=True) ###
            # samples_rgb = reverse_transform(samples_rgb) ###
            # samples_gray = reverse_transform(samples_gray) ###
            # sample_colored_images = reverse_transform(sample_colored_images) ###
            sample_grid = torch.stack(
                [samples_rgb, samples_gray, sample_colored_images], dim=0
            )  # 3, 5, 3, 256, 256
            sample_grid = sample_grid.permute(2, 0, 3, 1, 4)  # 3, 5, 256, 3, 256
            sample_grid = sample_grid.reshape(3, 3 * 256, 5 * 256)  # 3, 768, 1280
            wandb.log(
                {
                    "samples": [
                        wandb.Image(
                            sample_grid,
                            caption="Ground Truth, Grayscale Input, Colored",
                        )
                    ],
                    "epoch": epoch,
                }
            )
    total_loss /= len(data_loader)
    scheduler.step()
    return total_loss


def evaluate(model, data_loader, device):
    total_loss = 0
    with torch.no_grad():
        for i, images in enumerate(data_loader):
            images = images.to(device)
            images, images_lab, images_rgb = preprocess_images(images)
            images_ab = images_lab[:, 1:, ...]
            output, colored_images = model(images)
            loss = criterion(output, colored_images, images_ab, images_rgb)
            total_loss += loss.item()
    total_loss /= len(data_loader)
    return total_loss


def train(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epochs,
    log_freq,
    samples,
    sample_log_freq,
    scheduler
):
    wandb.watch(model)
    for epoch in range(1, epochs + 1):
        print("-" * 10 + f" Epoch {epoch:03d} " + "-" * 10)
        # train
        model.train()
        train_loss = train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            log_freq,
            samples,
            sample_log_freq,
            scheduler
        )
        print(f"Epoch {epoch} - Loss: {train_loss}")
        wandb.log({"epoch": epoch, "epoch_loss": train_loss})

        # validate
        model.eval()
        val_loss = evaluate(model, data_loader, device)
        print(f"Validation Loss: {val_loss}\n")
        wandb.log({"epoch": epoch, "val_loss": val_loss})

        # save checkpoint
        if not TESTING:
            torch.save(model.state_dict(), f"ddcolor_checkpoint_{epoch}.pth")

    wandb.finish()


if __name__ == "__main__":
    samples_dir = "test_images/imagenet/"
    sample_paths = glob(f"{samples_dir}/*.JPEG")
    sample_images = [torchvision.io.read_image(img) for img in sample_paths]
    samples = image_list_to_tensor(sample_images).float()
    samples = get_transform()(samples)

    wandb.init(project="ddcolor", entity="color")

    batch_size = 256
    lr = 1e-4
    device = torch.device("cuda")
    print("Loading data...")
    train_loader, val_loader = make_dataset(batch_size, testing=TESTING)
    print("Data loaded.")
    model = DDColor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(.9, .99))
    # decay lr by 0.5 after 1 epoch and every 0.5 epochs thereafter
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = CombinedLoss().to(device)
    epochs = 50
    log_freq = 10
    sample_log_freq = 150
    train(
        model,
        criterion,
        optimizer,
        train_loader,
        device,
        epochs,
        log_freq,
        samples,
        sample_log_freq,
        scheduler
    )
    print("Training done.")
    if not TESTING:
        print("Saving model...")
        torch.save(model.state_dict(), "ddcolor.pth")
        print("Model saved.")
