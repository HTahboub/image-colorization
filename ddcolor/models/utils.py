import cv2
import torch
import numpy as np
from typing import List


def preprocess_images(images: List[np.ndarray]) -> torch.Tensor:
    """Converts images of shape (H, W, 3) to a tensor of shape (B, 3, H, W) where
    the output tensors are grayscale images.

    Args:
        images (List[np.ndarray]): List of images of shape (H, W, 3), grayscale or BGR.

    Returns:
        torch.Tensor: Grayscale image tensor of shape (B, 3, H, W).
    """
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[2] == 3 for image in images)
    images = [cv2.resize(image, (256, 256)) for image in images]
    images = [image.astype(np.float32) / 255.0 for image in images]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, :1] for image in images]
    images = [
        np.concatenate((image, np.zeros_like(image), np.zeros_like(image)), axis=-1)
        for image in images
    ]
    images = [cv2.cvtColor(image, cv2.COLOR_LAB2RGB) for image in images]
    images = [image * 255.0 for image in images]
    images = [torch.tensor(image).permute(2, 0, 1) for image in images]
    images = torch.stack(images)
    return images


if __name__ == "__main__":
    image_paths = [
        "test_images/sample1.png",
        "test_images/sample2.png",
    ]
    images = [cv2.imread(image) for image in image_paths]
    images = preprocess_images(images)
    assert images.shape == (len(image_paths), 3, 256, 256)
    # check that they're actually grayscale
    for image in images:
        assert torch.allclose(image[0, ...], image[1, ...])
        assert torch.allclose(image[1, ...], image[2, ...])
