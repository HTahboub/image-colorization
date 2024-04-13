import torch
from torch import nn


class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, image_embedding: torch.Tensor, color_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the fusion module.

        Args:
            image_embedding: A tensor of shape (B, C, H, W).
            color_embedding: A tensor of shape (B, K, C).

        Returns:
            A tensor of shape (B, 2, H, W).
        """
        assert image_embedding.shape[1] == color_embedding.shape[2], f"Instead got {image_embedding.shape} and {color_embedding.shape}, image embedding shape should be (B, C, H, W) and color embedding shape should be (B, K, C), where C = {image_embedding.shape[1]} and K = {color_embedding.shape[1]}."
        assert color_embedding.shape[0] == image_embedding.shape[0], (
            f"Batch size of image embedding and color embedding must be the same. "
            f"Got {image_embedding.shape[0]} and {color_embedding.shape[0]}."
        )

        f_hat = torch.einsum("bkc,bchw -> bkhw", color_embedding, image_embedding)
        y_hat = nn.Conv2d(
            in_channels=color_embedding.shape[1], out_channels=2, kernel_size=1
        )(f_hat)
        return y_hat


if __name__ == "__main__":
    c = 512
    h = 256
    w = 256
    k = 128

    # Create an instance of the fusion module
    fusion_module = FusionModule()

    # Test batched input
    b = 4
    image_embedding_batch = torch.randn(b, c, h, w)
    color_embedding_batch = torch.randn(b, k, c)
    output_batch = fusion_module(image_embedding_batch, color_embedding_batch)
    assert output_batch.shape == (b, 2, h, w)
