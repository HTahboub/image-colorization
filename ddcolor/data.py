import os
import pickle

import torch
from PIL import Image
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, split, transform=None):
        if split == "train":
            path = "/work/vig/Datasets/imagenet/train"
        elif split == "val":
            path = "/work/vig/Datasets/imagenet/val"
        else:
            raise ValueError("Invalid split")
        self.transform = transform
        if os.path.exists(f"imagenet_{split}.pkl"):
            with open(f"imagenet_{split}.pkl", "rb") as f:
                self.images = pickle.load(f)
        else:
            self.images = []
            classes = os.listdir(path)
            for class_ in classes:
                class_path = os.path.join(path, class_)
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)
                    # check that the image is RGB
                    with Image.open(img_path) as img:
                        if img.mode == "RGB":
                            self.images.append(img_path)
            with open(f"imagenet_{split}.pkl", "wb") as f:
                pickle.dump(self.images, f)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.images[idx])
        if self.transform:
            img = self.transform(img)
        if img.shape == (1, 256, 256):
            return self.__getitem__(torch.randint(0, len(self.images), (1,)).item())
        return img


def collate_fn(batch):
    return torch.stack(batch)


def get_transform(tensor=True, reverse=False):
    if reverse:
        raise NotImplementedError
        # transform = transforms.Compose(
        #     [
        #         transforms.Normalize(
        #             mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        #         ),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0]),
        #     ]
        # )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.RandomHorizontalFlip(),
            ]
        )
    if not tensor:
        transform.transforms.append(transforms.ToTensor())
    return transform


def make_dataset(batch_size, validation=True, testing=False):
    if testing:
        imagenet_path = "/work/vig/Datasets/imagenet/train"
        class1 = os.listdir(imagenet_path)[0]
        class1 = os.path.join(imagenet_path, class1)
        subsample = os.listdir(class1)[:64]
        subsample = [os.path.join(class1, img) for img in subsample]
        subsample = [torchvision.io.read_image(img).float() for img in subsample]
        subsample = [get_transform()(img) for img in subsample]
        subsample = torch.stack(subsample)
        train_dataset = TensorDataset(subsample)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_dataset = ImageDataset("train", transform=get_transform())
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn,
        )

        if validation:
            val_dataset = ImageDataset("val", transform=get_transform())
            # reduce to 1024 random images
            torch.manual_seed(42)
            val_dataset = torch.utils.data.Subset(
                val_dataset, torch.randperm(len(val_dataset))[:1024]
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                collate_fn=collate_fn,
            )
        return train_loader, val_loader
    return train_loader
