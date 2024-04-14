import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.utils import preprocess_images

def make_dataset(batch_size, validation=False):
    imagenet_path = "/work/vig/Datasets/imagenet/train"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_dataset = ImageFolder(imagenet_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if validation:
        imagenet_path = "/work/vig/Datasets/imagenet/val"
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.ToTensor(),
            ]
        )
        val_dataset = ImageFolder(imagenet_path, transform=transform)
        # reduce to 1024 random
        torch.manual_seed(0)
        val_dataset = torch.utils.data.Subset(val_dataset, torch.randperm(1024))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    return train_loader
