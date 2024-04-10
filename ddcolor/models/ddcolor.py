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
        # TODO pixel decoder hyperparams
        num_color_queries: int = 100,
        num_color_decoder_layers: int = 3,
        num_color_decoder_heads: int = 8,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone = EncoderModule(encoder_name)
        self.pixel_decoder = PixelDecoder(
            # TODO hyperparams
        )
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

        over_four, over_eight, over_sixteen, over_thirtytwo = self.backbone(
            grayscale_image
        )
        assert over_four.shape == (B, C * 32, H // 4, W // 4)
        assert over_eight.shape == (B, C * 64, H // 8, W // 8)
        assert over_sixteen.shape == (B, C * 128, H // 16, W // 16)
        assert over_thirtytwo.shape == (B, C * 256, H // 32, W // 32)

        upsample_1, upsample_2, upsample_3, upsample_4 = self.pixel_decoder(
            over_four, over_eight, over_sixteen, over_thirtytwo
        )
        assert upsample_1.shape == (B, 2 * self.embedding_dim, H // 16, W // 16)
        assert upsample_2.shape == (B, 2 * self.embedding_dim, H // 8, W // 8)
        assert upsample_3.shape == (B, self.embedding_dim, H // 4, W // 4)
        assert upsample_4.shape == (B, self.embedding_dim, H, W)

        color_queries = self.color_decoder(upsample_1, upsample_2, upsample_3)
        assert color_queries.shape == (B, self.num_color_queries, self.embedding_dim)

        output = self.fusion(color_queries, upsample_4)
        assert output.shape == (B, 2, H, W)

        if return_colored_image:
            grayscale_single = grayscale_image[:, :1, ...]
            # check that the image is actually grayscale (all channels same)
            assert torch.allclose(
                grayscale_image[:, 0, ...], grayscale_image[:, 1, ...]
            )
            assert torch.allclose(
                grayscale_image[:, 1, ...], grayscale_image[:, 2, ...]
            )

            # scale grayscale image to proper range for L (0, 100)
            grayscale_single = grayscale_single / 255.0 * 100.0

            # scale output to proper range for AB (-128, 128)
            output_min = output.min().item()
            output_max = output.max().item()
            output = (output - output_min) / (output_max - output_min) * 256.0 - 128.0

            # combine the input (luminance) with the output (chrominance/ab channels)
            lab = torch.cat((grayscale_single, output), dim=1)  # B, 3, H, W

            lab = lab.permute(0, 2, 3, 1).cpu().numpy()
            colored_images = []
            # convert each to BGR
            for i in range(lab.shape[0]):
                lab_image = lab[i].astype(np.float32)
                bgr_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
                colored_images.append(bgr_image)

            return output, colored_images
        return output


if __name__ == "__main__":
    import cv2
    from utils import preprocess_images

    # TODO: run when pixel decoder is done
    model = DDColor()
    images = ["test_images/sample1.png", "test_images/sample2.png"]
    images = [cv2.imread(image) for image in images]
    images = preprocess_images(images)
    output, colored_images = model(images, return_colored_image=True)
    assert output.shape == (2, 2, 224, 224)
    assert len(colored_images) == 2
    assert colored_images[0].shape == (224, 224, 3)
    assert colored_images[1].shape == (224, 224, 3)
    cv2.imwrite("test_images/sample1_colored.png", colored_images[0])
    cv2.imwrite("test_images/sample2_colored.png", colored_images[1])
