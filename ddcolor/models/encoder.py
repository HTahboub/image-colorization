from typing import Tuple

import torch
from torch import nn
from transformers import AutoImageProcessor, ConvNextModel
from transformers.utils.logging import set_verbosity_error, set_verbosity_warning


class EncoderModule(nn.Module):
    def __init__(self, model_name: str = "facebook/convnext-tiny-224"):
        super().__init__()
        set_verbosity_error()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        set_verbosity_warning
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
                (B, 768, H/32, W/32), respectively, representing the hidden states of
                the encoder.
        """
        assert grayscale_image.shape[1:] == (3, 256, 256)
        inputs = self.image_processor(
            grayscale_image, return_tensors="pt", do_resize=False
        ).to(grayscale_image.device)
        outputs = self.model(**inputs, return_dict=True, output_hidden_states=True)
        over_four, over_eight, over_sixteen, over_thirtytwo = outputs.hidden_states[1:]
        return over_four, over_eight, over_sixteen, over_thirtytwo


if __name__ == "__main__":
    import torchvision
    from utils import preprocess_images, image_list_to_tensor

    encoder = EncoderModule()
    images = ["test_images/sample1.png", "test_images/sample2.png"]
    images = [torchvision.io.read_image(image) for image in images]
    images = image_list_to_tensor(images)
    images, _, _ = preprocess_images(images)
    over_four, over_eight, over_sixteen, over_thirtytwo = encoder(images)
    assert over_four.shape == (2, 96, 256 // 4, 256 // 4)
    assert over_eight.shape == (2, 192, 256 // 8, 256 // 8)
    assert over_sixteen.shape == (2, 384, 256 // 16, 256 // 16)
    assert over_thirtytwo.shape == (2, 768, 256 // 32, 256 // 32)
