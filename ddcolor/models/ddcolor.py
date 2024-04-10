import torch
import torch.nn as nn
from encoder import EncoderModule
from colordecoder import ColorDecoder
from pixeldecoder import PixelDecoder
from fusion import FusionModule


class DDColor(nn.Module):
    def __init__(
        self,
        encoder_name="facebook/convnext-tiny-224",
        embedding_dim=256,
        num_color_decoder_layers=3,
        num_color_queries=100,
        num_color_decoder_heads=8,
    ):
        super().__init__()
        self.encoder = EncoderModule(encoder_name)
        self.color_decoder = ColorDecoder(
            num_layers=num_color_decoder_layers,
            num_color_queries=num_color_queries,
            embedding_dim=embedding_dim,
            num_heads=num_color_decoder_heads,
        )
        self.pixel_decoder = PixelDecoder()
        self.fusion = FusionModule()

    def forward(self, grayscale_image):
        """Forward pass of the model.

        Args:
            grayscale_image: A tensor of shape (B, 3, H, W) representing a grayscale
                (one channel concatenated three times) image. Not single channel so that
                the pretrained encoder can be used.

        Returns:
            torch.Tensor: Output tensor of shape (B, 2, H, W).
        """
        
