from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from kornia.color import lab_to_rgb, rgb_to_lab


def image_list_to_tensor(
    images: List[torch.Tensor], size: Tuple[int, int] = (256, 256)
) -> torch.Tensor:
    """Converts a list of images to a tensor of shape (B, 3, H, W).

    Args:
        images (List[torch.Tensor]): List of images of shape (3, ?, ?).
        size (Tuple[int, int]): Size to resize the images to.

    Returns:
        torch.Tensor: Tensor of images of shape (B, 3, H, W).
    """
    assert all(image.ndim == 3 for image in images)
    assert all(image.shape[0] == 3 for image in images)
    images = [F.interpolate(image.unsqueeze(0), size=size) for image in images]
    images = torch.cat(images, dim=0)
    return images


def preprocess_images(
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts images of shape (B, 3, H, W) to a tensor of shape (B, 3, H, W) where
    the output tensors are grayscale images. Also returns the LAB and RGB images.

    Args:
        images (torch.Tensor): Tensor of images of shape (B, 3, H, W), grayscale or BGR.

    Returns:
        torch.Tensor: Grayscale image tensor of shape (B, 3, H, W).
        torch.Tensor: LAB image tensor of shape (B, 3, H, W).
        torch.Tensor: RGB image tensor of shape (B, 3, H, W).
    """
    assert images.ndim == 4
    assert images.shape[1] == 3
    images = F.interpolate(images, size=(256, 256))
    images_rgb = images.float() / 255.0
    images_lab = rgb_to_lab(images_rgb)
    images_l = images_lab[:, :1, ...]
    images = torch.cat(
        (
            images_l,
            torch.zeros_like(images_l),
            torch.zeros_like(images_l),
        ),
        dim=1,
    )
    images = lab_to_rgb(images)
    # images = images * 255.0
    return images, images_lab, images_rgb


if __name__ == "__main__":
    image_paths = [
        "test_images/sample1.png",
        "test_images/sample2.png",
    ]
    images = [torchvision.io.read_image(image_path) for image_path in image_paths]
    images = image_list_to_tensor(images)
    images, image_lab, image_rgb = preprocess_images(images)
    assert (
        images.shape
        == image_lab.shape
        == image_rgb.shape
        == (len(image_paths), 3, 256, 256)
    )
    # check that they're actually grayscale
    for image in images:
        assert torch.allclose(image[0, ...], image[1, ...], atol=1e-3)
        assert torch.allclose(image[1, ...], image[2, ...], atol=1e-3)
