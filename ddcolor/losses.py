import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from kornia.color import lab_to_rgb
from torchvision.models import vgg16
from torchvision.models.resnet import ResNet18_Weights


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
        self.adversarial_loss = AdversarialLoss()
        self.colorfulness_loss = ColorfulnessLoss()

    def forward(
        self,
        colorized_image: torch.Tensor,
        ground_truth: torch.Tensor,
        grayscale_image: torch.Tensor,
        ground_truth_ab: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the combined loss between the colorized image and the ground truth
        image, which is a weighted sum of the pixel loss, perceptual loss, adversarial
        loss, and colorfulness loss.

        Args:
            colorized_image (torch.Tensor): The colorized image, shape (B, 2, H, W)
            ground_truth (torch.Tensor): The ground truth image, shape (B, 3, H, W)
            grayscale_image (torch.Tensor): The grayscale image, shape (B, 1, H, W)
            ground_truth_ab (torch.Tensor): The ground truth AB channels, shape (B, 2, H, W)

        Returns:
            torch.Tensor: The combined loss between the colorized image and the ground
                truth image as a scalar tensor.
        """
        assert (
            colorized_image.shape[0]
            == ground_truth.shape[0]
            == grayscale_image.shape[0]
            == ground_truth_ab.shape[0]
            and colorized_image.shape[2:]
            == ground_truth.shape[2:]
            == grayscale_image.shape[2:]
            == ground_truth_ab.shape[2:]
        )
        assert (
            colorized_image.shape[1] == 2
            and ground_truth.shape[1] == 3
            and grayscale_image.shape[1] == 1
            and ground_truth_ab.shape[1] == 2
        )
        full_colorized_image = torch.cat((grayscale_image, colorized_image), dim=1)
        full_colorized_image = lab_to_rgb(full_colorized_image)
        full_colorized_image = full_colorized_image[:, [2, 1, 0], ...]  # RGB to BGR
        pixel_loss = self.pixel_loss(colorized_image, ground_truth_ab)
        perceptual_loss = self.perceptual_loss(full_colorized_image, ground_truth)
        adversarial_loss = self.adversarial_loss(ground_truth, full_colorized_image)
        colorfulness_loss = self.colorfulness_loss(full_colorized_image)
        return (
            self.pixel_weight * pixel_loss
            + self.perceptual_weight * perceptual_loss
            + self.adversarial_weight * adversarial_loss
            + self.colorfulness_weight * colorfulness_loss
        )


class PixelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, colorized_image: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """Computes the pixel loss between the colorized image and the ground truth
        image, defined as L1 distance between the two images. The images are assumed to
        be the AB channels of the Lab color space.

        Args:
            colorized_image (torch.Tensor): The colorized image, shape (B, 2, H, W)
            ground_truth (torch.Tensor): The ground truth image, shape (B, 2, H, W)

        Returns:
            torch.Tensor: The pixel loss between the colorized image and the ground
                truth image as a scalar tensor.
        """
        assert colorized_image.shape == ground_truth.shape
        assert len(colorized_image.shape) == 4
        assert colorized_image.shape[1] == 2
        return F.l1_loss(colorized_image, ground_truth)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(weights="IMAGENET1K_FEATURES")  # TODO try vgg19? try small?
        self.vgg.forward = lambda x: self.vgg.avgpool(self.vgg.features(x))
        self.maxpool = nn.MaxPool2d(7)  # NOTE this maxpool may not be necessary
        del self.vgg.classifier
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

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
        assert colorized_image.shape == ground_truth.shape
        assert len(colorized_image.shape) == 4
        assert colorized_image.shape[1] == 3
        colorized_features = self.vgg(colorized_image)
        colorized_features = self.maxpool(colorized_features).squeeze((2, 3))
        ground_truth_features = self.vgg(ground_truth)
        ground_truth_features = self.maxpool(ground_truth_features).squeeze((2, 3))
        return F.l1_loss(colorized_features, ground_truth_features)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

        # Load pre-trained ResNet model
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

        # Freeze the model parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Set the model to evaluation mode
        self.resnet.eval()

        # Define the loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real_images, fake_images):
        assert real_images.shape == fake_images.shape
        assert len(real_images.shape) == 4
        assert real_images.shape[1] == 3
        # Forward pass for real images
        real_outputs = self.resnet(real_images)
        real_labels = torch.ones_like(real_outputs)
        real_loss = self.criterion(real_outputs, real_labels)

        # Forward pass for fake images
        fake_outputs = self.resnet(fake_images)
        fake_labels = torch.zeros_like(fake_outputs)
        fake_loss = self.criterion(fake_outputs, fake_labels)

        # Combine the losses
        loss = (real_loss + fake_loss) / 2

        return loss


class ColorfulnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, colorized_image: torch.Tensor) -> torch.Tensor:
        assert len(colorized_image.shape) == 4
        assert colorized_image.shape[1] == 3
        # Convert the colorized image from BGR to rg-yb space
        b, g, r = (
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
    dummy = torch.randn(4, 2, 256, 256).cuda()
    dummy_gt = torch.randn(4, 3, 256, 256).cuda()
    dummy_gt_ab = torch.randn(4, 2, 256, 256).cuda()
    dummy_gray = torch.randn(4, 1, 256, 256).cuda()

    pixel_loss = PixelLoss().cuda()
    perceptual_loss = PerceptualLoss().cuda()
    adversarial_loss = AdversarialLoss().cuda()
    colorfulness_loss = ColorfulnessLoss().cuda()
    combined_loss = CombinedLoss().cuda()

    dummy_colorized_sim = lab_to_rgb(torch.concat((dummy_gray, dummy), dim=1))
    loss_1 = pixel_loss(dummy, dummy_gt_ab)
    loss_2 = perceptual_loss(dummy_colorized_sim, dummy_gt)
    loss_3 = adversarial_loss(dummy_gt, dummy_colorized_sim)
    loss_4 = colorfulness_loss(dummy_colorized_sim)
    loss = combined_loss(dummy, dummy_gt, dummy_gray, dummy_gt_ab)

    assert loss_1.shape == torch.Size([])
    assert loss_2.shape == torch.Size([])
    assert loss_3.shape == torch.Size([])
    assert loss_4.shape == torch.Size([])
    assert loss.shape == torch.Size([])

    print(f"Pixel loss: {loss_1:.4f}")
    print(f"Perceptual loss: {loss_2:.4f}")
    print(f"Adversarial loss: {loss_3:.4f}")
    print(f"Colorfulness loss: {loss_4:.4f}")
    print(f"Combined loss: {loss:.4f}")

    assert pixel_loss(dummy, dummy) == 0
    assert perceptual_loss(dummy_colorized_sim, dummy_colorized_sim) == 0

    from time import time

    start = time()
    for _ in range(100):
        pixel_loss(dummy, dummy_gt_ab)
    print(f"\nPixel loss time: {time() - start:.4f}")

    start = time()
    for _ in range(100):
        perceptual_loss(dummy_colorized_sim, dummy_gt)
    print(f"Perceptual loss time: {time() - start:.4f}")

    start = time()
    for _ in range(100):
        adversarial_loss(dummy_gt, dummy_colorized_sim)
    print(f"Adversarial loss time: {time() - start:.4f}")

    start = time()
    for _ in range(100):
        colorfulness_loss(dummy_colorized_sim)
    print(f"Colorfulness loss time: {time() - start:.4f}")

    start = time()
    for _ in range(100):
        combined_loss(dummy, dummy_gt, dummy_gray, dummy_gt_ab)
    print(f"Combined loss time: {time() - start:.4f}")
