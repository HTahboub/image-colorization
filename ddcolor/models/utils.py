import cv2
import torch
import numpy as np
from typing import List


def preprocess_images(images: List[np.ndarray]) -> torch.Tensor:
    """Converts images of shape (H, W, 3) to a tensor of shape (B, 3, H, W) where
    the output tensors are grayscale images.

    Args:
        images (List[np.ndarray]): List of images of shape (H, W, 3), grayscale or RGB.

    Returns:
        torch.Tensor: Grayscale image tensor of shape (B, 3, H, W).
    """
    images = [cv2.resize(image, (224, 224)) for image in images]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, :1] for image in images]
    images = [
        np.concatenate((image, np.zeros_like(image), np.zeros_like(image)), axis=-1)
        for image in images
    ]
    images = [cv2.cvtColor(image, cv2.COLOR_LAB2RGB) for image in images]
    images = [torch.tensor(image).permute(2, 0, 1) for image in images]
    return torch.stack(images)


def postprocess_images(images: torch.Tensor) -> List[np.ndarray]:
    """Converts images of shape (B, 3, H, W) to a list of images of shape (H, W, 3) where
    the output images are RGB images.

    Args:
        images (torch.Tensor): Grayscale image tensor of shape (B, 3, H, W).

    Returns:
        List[np.ndarray]: List of images of shape (H, W, 3), RGB.
    """
    images = images.permute(0, 2, 3, 1).numpy()
    images = [cv2.cvtColor(image, cv2.COLOR_RGB2Lab) for image in images]
    images = [image[:, :, 0] for image in images]
    images = [
        np.stack((image, np.zeros_like(image), np.zeros_like(image)), axis=-1)
        for image in images
    ]
    images = [cv2.cvtColor(image, cv2.COLOR_LAB2RGB) for image in images]
    return images