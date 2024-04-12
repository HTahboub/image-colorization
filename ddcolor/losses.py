import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pixel_weight: float = 0.1,
        perceptual_weight: float = 5.0,
        adversarial_weight: float = 1.0,
        colorfulness_weight: float = 0.5,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.colorfulness_weight = colorfulness_weight
        self.pixel_loss = PixelLoss()
        self.perceptual_loss = PerceptualLoss()
        # self.adversarial_loss = AdversarialLoss()
        # self.colorfulness_loss = ColorfulnessLoss()

    def forward(
        self,
        colorized_image: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the combined loss between the colorized image and the ground truth
        image, which is a weighted sum of the pixel loss, perceptual loss, adversarial
        loss, and colorfulness loss.

        Args:
            colorized_image (torch.Tensor): The colorized image, shape (B, C, H, W)
            ground_truth (torch.Tensor): The ground truth image, shape (B, C, H, W)

        Returns:
            torch.Tensor: The combined loss between the colorized image and the ground
                truth image as a scalar tensor.
        """
        pixel_loss = self.pixel_loss(colorized_image, ground_truth)
        perceptual_loss = self.perceptual_loss(colorized_image, ground_truth)
        # adversarial_loss = self.adversarial_loss(colorized_image)
        # colorfulness_loss = self.colorfulness_loss(colorized_image)
        return (
            self.pixel_weight * pixel_loss
            + self.perceptual_weight * perceptual_loss
            # + self.adversarial_weight * adversarial_loss
            # + self.colorfulness_weight * colorfulness_loss
        )


class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, colorized_image: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Computes the pixel loss between the colorized image and the ground truth
        image, defined as L1 distance between the two images.

        Args:
            colorized_image (torch.Tensor): The colorized image, shape (B, C, H, W)
            ground_truth (torch.Tensor): The ground truth image, shape (B, C, H, W)

        Returns:
            torch.Tensor: The pixel loss between the colorized image and the ground
                truth image as a scalar tensor.
        """
        return F.l1_loss(colorized_image, ground_truth)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(weights="IMAGENET1K_FEATURES")  # TODO try vgg19?
        self.vgg.forward = lambda x: self.vgg.avgpool(self.vgg.features(x))
        self.maxpool = nn.MaxPool2d(7)  # NOTE this maxpool may not be necessary
        del self.vgg.classifier

    def forward(
        self, colorized_image: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Measures the semantic difference between the colorized image and the ground
        truth image using a pretrained VGG16 network. The perceptual loss is defined as
        the L1 distance between the features of the two images after passing through the
        VGG16.


        Args:
            colorized_image (torch.Tensor): The colorized image, shape (B, C, H, W)
            ground_truth (torch.Tensor): The ground truth image, shape (B, C, H, W)

        Returns:
            torch.Tensor: The perceptual loss between the colorized image and the ground
                truth image as a scalar tensor.
        """
        colorized_features = self.vgg(colorized_image)
        colorized_features = self.maxpool(colorized_features).squeeze((2, 3))
        ground_truth_features = self.vgg(ground_truth)
        ground_truth_features = self.maxpool(ground_truth_features).squeeze((2, 3))
        return F.l1_loss(colorized_features, ground_truth_features)


class AdversarialLoss(nn.Module):
    pass


class ColorfulnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, colorized_image: torch.Tensor) -> torch.Tensor:
        # Convert the colorized image from RGB to rg-yb space
        r, g, b = (
            colorized_image[:, 0, :, :],
            colorized_image[:, 1, :, :],
            colorized_image[:, 2, :, :],
        )
        rg = r - g
        yb = 0.5 * (r + g) - b

        # Compute the mean and standard deviation of rg and yb
        rg_mean = torch.mean(rg, dim=[1, 2])
        rg_std = torch.std(rg, dim=[1, 2])
        yb_mean = torch.mean(yb, dim=[1, 2])
        yb_std = torch.std(yb, dim=[1, 2])

        # Compute the mean and standard deviation of the pixel cloud in the rg-yb space
        std_rgyb = torch.sqrt(rg_std**2 + yb_std**2)
        mean_rgyb = torch.sqrt(rg_mean**2 + yb_mean**2)

        # Compute the colorfulness metric
        colorfulness = std_rgyb + 0.3 * mean_rgyb

        # Normalize the colorfulness to [0, 1]
        min_colorfulness = 0.0
        max_colorfulness = 109.0  # From Table 2 in the paper
        colorfulness = (colorfulness - min_colorfulness) / (
            max_colorfulness - min_colorfulness
        )

        # Return colorfulness loss
        return 1 - colorfulness.mean()


if __name__ == "__main__":
    dummy = torch.randn(4, 3, 256, 256)
    dummy_gt = torch.randn(4, 3, 256, 256)

    pixel_loss = PixelLoss()
    perceptual_loss = PerceptualLoss()
    # adversarial_loss = AdversarialLoss()
    colorfulness_loss = ColorfulnessLoss()

    loss_1 = pixel_loss(dummy, dummy_gt)
    loss_2 = perceptual_loss(dummy, dummy_gt)
    # loss_3 = adversarial_loss(dummy)
    loss_4 = colorfulness_loss(dummy)

    assert loss_1.shape == torch.Size([])
    assert loss_2.shape == torch.Size([])
    # assert loss_3.shape == torch.Size([])
    assert loss_4.shape == torch.Size([])

    print(f"Pixel loss: {loss_1:.4f}")
    print(f"Perceptual loss: {loss_2:.4f}")
    # print(f"Adversarial loss: {loss_3:.4f}")
    print(f"Colorfulness loss: {loss_4:.4f}")

    assert pixel_loss(dummy, dummy) == 0
    assert perceptual_loss(dummy, dummy) == 0
