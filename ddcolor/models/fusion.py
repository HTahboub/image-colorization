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
        assert image_embedding.shape[0] == color_embedding.shape[1]

        f_hat = torch.einsum("bkc,bchw -> bkhw", color_embedding, image_embedding)
        y_hat = nn.Conv2d(
            in_channels=color_embedding.shape[1], out_channels=2, kernel_size=1
        )(f_hat)
        return y_hat


if __name__ == "__main__":
    c = 512
    h = 64
    w = 100
    k = 128

    # Create an instance of the fusion module
    fusion_module = FusionModule()

    # Test batched input
    batch_size = 4
    image_embedding_batch = torch.randn(
        batch_size, c, h, w
    )  # Example shape: (B, C, H, W)
    color_embedding_batch = torch.randn(batch_size, k, c)  # Example shape: (B, K, C)

    output_batch = fusion_module(image_embedding_batch, color_embedding_batch)

    expected_batch_shape = (batch_size, 2, h, w)
    assert (
        output_batch.shape == expected_batch_shape
    ), f"Output shape {output_batch.shape} does not match the expected shape {expected_batch_shape}"
    print("Batched input test passed!")
