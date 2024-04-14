from typing import Tuple, Union

import torch
import torch.nn as nn
import torchvision
from kornia.color import lab_to_rgb, rgb_to_lab
from ddcolor.models.colordecoder import ColorDecoder
from ddcolor.models.encoder import EncoderModule
from ddcolor.models.fusion import FusionModule
from ddcolor.models.pixeldecoder import PixelDecoder


class DDColor(nn.Module):
    def __init__(
        self,
        encoder_name: str = "facebook/convnext-tiny-224",
        embedding_dim: int = 256,
        num_color_queries: int = 100,
        num_color_decoder_layers: int = 3,
        num_color_decoder_heads: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.backbone = EncoderModule(encoder_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.pixel_decoder = PixelDecoder()

        self.num_color_queries = num_color_queries
        self.color_decoder = ColorDecoder(
            num_layers=num_color_decoder_layers,
            num_color_queries=num_color_queries,
            embedding_dim=embedding_dim,
            num_heads=num_color_decoder_heads,
        )

        self.fusion = FusionModule()

    def forward(
        self, grayscale_image: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the model.

        Args:
            grayscale_image: A tensor of shape (B, 3, H, W) representing a grayscale
                (one channel concatenated three times) image. Not single channel so that
                the pretrained encoder can be used.

        Returns:
            output: A tensor of shape (B, 2, H, W) representing the chrominance channels
                of the colored images.
            colored_images: A tensor of shape (B, 3, H, W) representing the colored
                images.
        """
        B, C, H, W = grayscale_image.shape
        assert C == 3

        # check that the image is actually grayscale (all channels same)
        assert torch.allclose(
            grayscale_image[:, 0, ...], grayscale_image[:, 1, ...], atol=1e-3
        )
        assert torch.allclose(
            grayscale_image[:, 1, ...], grayscale_image[:, 2, ...], atol=1e-3
        )

        over_four, over_eight, over_sixteen, over_thirtytwo = self.backbone(
            grayscale_image
        )
        assert over_four.shape == (B, C * 32, H // 4, W // 4)
        assert over_eight.shape == (B, C * 64, H // 8, W // 8)
        assert over_sixteen.shape == (B, C * 128, H // 16, W // 16)
        assert over_thirtytwo.shape == (B, C * 256, H // 32, W // 32)

        upsample_1, upsample_2, upsample_3, upsample_4 = self.pixel_decoder(
            grayscale_image, over_four, over_eight, over_sixteen, over_thirtytwo
        )
        assert upsample_1.shape == (B, 2 * self.embedding_dim, H // 16, W // 16)
        assert upsample_2.shape == (B, 2 * self.embedding_dim, H // 8, W // 8)
        assert upsample_3.shape == (B, self.embedding_dim, H // 4, W // 4)
        assert upsample_4.shape == (B, self.embedding_dim, H, W)

        color_queries = self.color_decoder(upsample_1, upsample_2, upsample_3)
        assert color_queries.shape == (B, self.num_color_queries, self.embedding_dim)

        output = self.fusion(image_embedding=upsample_4, color_embedding=color_queries)
        assert output.shape == (B, 2, H, W)

        grayscale_single = rgb_to_lab(grayscale_image)[:, :1, ...]
        assert grayscale_single.shape == (B, 1, H, W)
        if False:
            assert grayscale_single.min() >= 0  # passes
            assert grayscale_single.max() <= 100  # passes
            # test: replace output with noise to see if the model _could_ colorize
            output = output / output.max()
            output = output * torch.randn_like(output) * 50 + 50
        output = output.clamp(-128, 127)
        colored_images = torch.cat((grayscale_single, output), dim=1)
        colored_images = lab_to_rgb(colored_images)
        assert colored_images.shape == (B, 3, H, W)
        return output, colored_images


if __name__ == "__main__":
    import time
    from ddcolor.losses import CombinedLoss
    from ddcolor.models.utils import preprocess_images, image_list_to_tensor

    mul_factor = 32  # simulating batch size of 64  # TODO try 128
    images = ["test_images/sample1.png", "test_images/sample2.png"]
    images = [torchvision.io.read_image(img) for img in images] * mul_factor
    images = image_list_to_tensor(images)
    images, images_lab, images_rgb = preprocess_images(images)
    images_ab = images_lab[:, 1:, ...]

    model = DDColor()
    criterion = CombinedLoss()

    device = "cpu"
    model = model.to(device)
    images = images.to(device)
    images_rgb = images_rgb.to(device)
    images_ab = images_ab.to(device)
    criterion = criterion.to(device)
    print("Input dim:", images.shape)
    start = time.time()
    output, colored_images = model(images)
    loss = criterion(output, colored_images, images_ab, images_rgb)
    loss.backward()
    print(f"CPU Forward-Backward Time: {time.time() - start:.4f} seconds")
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    device = torch.device("cuda")
    model = model.to(device)
    images = images.to(device)
    images_rgb = images_rgb.to(device)
    images_ab = images_ab.to(device)
    criterion = criterion.to(device)
    start = time.time()
    output, colored_images = model(images)
    loss = criterion(output, colored_images, images_ab, images_rgb)
    loss.backward()
    print(f"GPU Forward-Backward Time: {time.time() - start:.4f} seconds")
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    output, colored_images = model(images)
    assert output.shape == (2 * mul_factor, 2, 256, 256)
    assert colored_images.shape == (2 * mul_factor, 3, 256, 256)
    torchvision.io.write_png(
        (colored_images[0, ...] * 255).to(torch.uint8).cpu(),
        "test_images/sample1_color.png",
    )
    torchvision.io.write_png(
        (colored_images[1, ...] * 255).to(torch.uint8).cpu(),
        "test_images/sample2_color.png",
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")  # 30,282,080
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total trainable params: {total_trainable_params}")  # 2,461,952
