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
            image_embedding: A tensor of shape (C, H, W).
            color_embedding: A tensor of shape (K, C).

        Returns:
            A tensor of shape (2, H, W).
        """
        assert image_embedding.shape[0] == color_embedding.shape[1]
        
        f_hat = torch.einsum("kc,chw -> khw", color_embedding, image_embedding)
        y_hat = nn.Conv2d(
            in_channels=color_embedding.shape[0], out_channels=2, kernel_size=1
        )(f_hat)
        return y_hat


if __name__ == "__main__":
    c = 512
    h = 64
    w = 100
    k = 128
    image_embedding = torch.randn(c, h, w)  # Example shape: (C, H, W)
    color_embedding = torch.randn(k, c)  # Example shape: (K, C)

    # Create an instance of the fusion module
    fusion_module = FusionModule()

    # Run the forward pass
    output = fusion_module(image_embedding, color_embedding)

    # Check the shape of the output tensor
    expected_shape = (2, h, w)  # Adjust the expected shape based on your requirements
    assert (
        output.shape == expected_shape
    ), f"Output shape {output.shape} does not match the expected shape {expected_shape}"

    print("Output shape is correct!")
