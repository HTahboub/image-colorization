from typing import Tuple

import torch
from torch import nn
from transformers import AutoImageProcessor, ConvNextModel


class EncoderModule(nn.Module):
    def __init__(self, model_name: str = "facebook/convnext-tiny-224"):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ConvNextModel.from_pretrained(model_name)

    def forward(
        self, grayscale_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder module.

        Args:
            grayscale_image (torch.Tensor): Grayscale image tensor of shape
                (B, 3, H, W).

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: dimensions
            (B, 96, H/4, W/4), (B, 192, H/8, W/8), (B, 384, H/16, W/16), and
            (B, 768, H/32, W/32), respectively, representing the hidden states of the
            encoder.
        """
        assert grayscale_image.shape[1:] == (3, 224, 224)
        inputs = self.image_processor(grayscale_image, return_tensors="pt")
        outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
        over_four, over_eight, over_sixteen, over_thirtytwo = outputs.hidden_states[1:]
        return over_four, over_eight, over_sixteen, over_thirtytwo


if __name__ == "__main__":
    import cv2
    from utils import preprocess_images

    encoder = EncoderModule()
    images = ["test_images/sample1.png", "test_images/sample2.png"]
    images = [cv2.imread(image) for image in images]
    images = preprocess_images(images)
    over_four, over_eight, over_sixteen, over_thirtytwo = encoder(images)
    assert over_four.shape == (2, 96, 224 // 4, 224 // 4)
    assert over_eight.shape == (2, 192, 224 // 8, 224 // 8)
    assert over_sixteen.shape == (2, 384, 224 // 16, 224 // 16)
    assert over_thirtytwo.shape == (2, 768, 224 // 32, 224 // 32)
