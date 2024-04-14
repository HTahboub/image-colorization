from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from colordecoder import ColorDecoder
from encoder import EncoderModule
from fusion import FusionModule
from pixeldecoder import PixelDecoder


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
        self, grayscale_image: torch.Tensor, return_colored_image: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[np.ndarray]]]:
        """Forward pass of the model.

        Args:
            grayscale_image: A tensor of shape (B, 3, H, W) representing a grayscale
                (one channel concatenated three times) image. Not single channel so that
                the pretrained encoder can be used.

        Returns:
            output: A tensor of shape (B, 2, H, W) representing the chrominance channels
                of the colored image.
            colored_images: A list of B numpy arrays of shape (H, W, 3) representing the
                colored images in BGR format. Only returned if return_colored_image is
                True.
        """
        B, C, H, W = grayscale_image.shape
        assert C == 3

        # check that the image is actually grayscale (all channels same)
        assert torch.allclose(grayscale_image[:, 0, ...], grayscale_image[:, 1, ...])
        assert torch.allclose(grayscale_image[:, 1, ...], grayscale_image[:, 2, ...])

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

        if return_colored_image:
            grayscale_single = grayscale_image.detach().clone()
            grayscale_single = grayscale_single.permute(0, 2, 3, 1)
            grayscale_single = grayscale_single.cpu().numpy()
            grayscale_singles = []
            # TODO: replace with kornia.color.bgr_to_lab?
            for i in range(grayscale_single.shape[0]):
                grayscale_i = grayscale_single[i].astype(np.float32) / 255.0
                grayscale_i = cv2.cvtColor(grayscale_i, cv2.COLOR_BGR2Lab)
                grayscale_i = grayscale_i[:, :, :1]
                grayscale_singles.append(grayscale_i)
            grayscale_single = np.stack(grayscale_singles)

            output_np = output.detach().clone().cpu().numpy()
            output_np = np.transpose(output_np, (0, 2, 3, 1)).astype(np.float32)

            # scale output to proper range for AB (-128, 128)
            # TODO: does this matter?
            output_min = output_np.min()
            output_max = output_np.max()
            output_np = (output_np - output_min) / (
                output_max - output_min
            ) * 256.0 - 128.0

            # combine the input (luminance) with the output (chrominance/ab channels)
            lab = np.concatenate((grayscale_single, output_np), axis=-1)  # (B, H, W, 3)

            colored_images = []
            # convert each to BGR
            for i in range(lab.shape[0]):
                bgr_image = cv2.cvtColor(lab[i], cv2.COLOR_LAB2BGR) * 255.0
                colored_images.append(bgr_image)
            return output, colored_images
        return output


if __name__ == "__main__":
    import sys
    import time

    sys.path.append("ddcolor")
    import numpy as np
    from losses import CombinedLoss
    from utils import preprocess_images

    mul_factor = 64  # simulating batch size of 128
    images = ["test_images/sample1.png", "test_images/sample2.png"]
    images = [cv2.imread(image) for image in images] * mul_factor
    images, images_lab, images_rgb = preprocess_images(images)

    model = DDColor()
    criterion = CombinedLoss()
    images_l, images_ab = images_lab[:, :1, ...], images_lab[:, 1:, ...]

    device = "cpu"
    model = model.to(device)
    images = images.to(device)
    images_rgb = images_rgb.to(device)
    images_l = images_l.to(device)
    images_ab = images_ab.to(device)
    criterion = criterion.to(device)
    print("Input dim:", images.shape)
    start = time.time()
    output = model(images)
    loss = criterion(output, images_rgb, images_l, images_ab)
    loss.backward()
    print(f"CPU Forward-Backward Time: {time.time() - start:.4f} seconds")
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    device = torch.device("cuda")
    model = model.to(device)
    images = images.to(device)
    images_rgb = images_rgb.to(device)
    images_l = images_l.to(device)
    images_ab = images_ab.to(device)
    criterion = criterion.to(device)
    start = time.time()
    output = model(images)
    loss = criterion(output, images_rgb, images_l, images_ab)
    loss.backward()
    print(f"GPU Forward-Backward Time: {time.time() - start:.4f} seconds")
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None

    output, colored_images = model(images, return_colored_image=True)
    assert output.shape == (2 * mul_factor, 2, 256, 256)
    assert len(colored_images) == 2 * mul_factor
    assert colored_images[0].shape == (256, 256, 3)
    assert colored_images[1].shape == (256, 256, 3)
    # cv2.imwrite("test_images/sample1_colored.png", colored_images[0])
    # cv2.imwrite("test_images/sample2_colored.png", colored_images[1])

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params}")  # 30,282,080
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total trainable params: {total_trainable_params}")  # 2,461,952
