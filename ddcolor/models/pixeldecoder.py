import torch
from torch import nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    def __init__(self, in_channels=384, out_channels=256):
        super().__init__()
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.pixel_shuffle3 = nn.PixelShuffle(2)
        self.pixel_shuffle4 = nn.PixelShuffle(2)

    def forward(
        self,
        grayscale_image: torch.Tensor,  # Shape: (B, 3, 224, 224)
        over_four: torch.Tensor,  # Shape: (B, 96, 224/4, 224/4)
        over_eight: torch.Tensor,  # Shape: (B, 192, 224/8, 224/8)
        over_sixteen: torch.Tensor,  # Shape: (B, 384, 224/16, 224/16)
        over_thirtytwo: torch.Tensor,  # Shape: (B, 384, 224/32, 224/32)
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        x = self.pixel_shuffle1(over_thirtytwo)
        x = torch.cat([x, over_sixteen], dim=1)

        x = self.pixel_shuffle2(x)
        x = torch.cat([x, over_eight], dim=1)

        x = self.pixel_shuffle3(x)
        x = torch.cat([x, over_four], dim=1)
        x = x[:, :172, :, :]

        x = self.pixel_shuffle4(x)
        grayscale_image = F.interpolate(
            grayscale_image, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, grayscale_image], dim=1)

        num_channels = x.shape[1]
        output1 = x[:, : num_channels // 4, :, :]
        output2 = x[:, num_channels // 4 : num_channels // 2, :, :]
        output3 = x[:, num_channels // 2 : num_channels * 3 // 4, :, :]
        output4 = x[:, num_channels * 3 // 4 :, :, :]

        return output1, output2, output3, output4


if __name__ == "__main__":
    batch_size = 2
    height = 224
    width = 224

    grayscale_image = torch.randn(batch_size, 3, height, width)
    over_four = torch.randn(batch_size, 96, height // 4, width // 4)
    over_eight = torch.randn(batch_size, 192, height // 8, width // 8)
    over_sixteen = torch.randn(batch_size, 384, height // 16, width // 16)
    over_thirtytwo = torch.randn(batch_size, 384, height // 32, width // 32)

    pixel_decoder = PixelDecoder()

    # Test each step of the forward method and print the intermediate output shapes
    x = pixel_decoder.pixel_shuffle1(over_thirtytwo)
    print(f"After pixel_shuffle1: {x.shape}")

    x = torch.cat([x, over_sixteen], dim=1)
    print(f"After concatenation with over_sixteen: {x.shape}")

    x = pixel_decoder.pixel_shuffle2(x)
    print(f"After pixel_shuffle2: {x.shape}")

    x = torch.cat([x, over_eight], dim=1)
    print(f"After concatenation with over_eight: {x.shape}")

    x = pixel_decoder.pixel_shuffle3(x)
    print(f"After pixel_shuffle3: {x.shape}")

    x = torch.cat([x, over_four], dim=1)
    print(f"After concatenation with over_four: {x.shape}")

    x = x[:, :172, :, :]
    print(f"After channel adjustment: {x.shape}")

    x = pixel_decoder.pixel_shuffle4(x)
    print(f"After pixel_shuffle4: {x.shape}")

    grayscale_image = F.interpolate(
        grayscale_image, size=x.shape[2:], mode="bilinear", align_corners=False
    )
    print(f"After resizing grayscale_image: {grayscale_image.shape}")

    x = torch.cat([x, grayscale_image], dim=1)
    print(f"After final concatenation: {x.shape}")

    # Call the forward method and print the output shapes
    output1, output2, output3, output4 = pixel_decoder(
        grayscale_image, over_four, over_eight, over_sixteen, over_thirtytwo
    )
    print(f"Output 1 shape: {output1.shape}")
    print(f"Output 2 shape: {output2.shape}")
    print(f"Output 3 shape: {output3.shape}")
    print(f"Output 4 shape: {output4.shape}")
