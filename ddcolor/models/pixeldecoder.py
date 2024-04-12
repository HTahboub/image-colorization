import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    def __init__(self):
        super(PixelDecoder, self).__init__()

        self.conv1x1_1 = nn.Conv2d(192, 384, kernel_size=1)
        self.conv1 = nn.Conv2d(768, 512, kernel_size=3, padding=1)

        self.conv1x1_2 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(704, 512, kernel_size=3, padding=1)

        self.conv1x1_3 = nn.Conv2d(128, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(608, 1024, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(64, 256, kernel_size=3, padding=1)

    def forward(
        self, grayscale_image, over_four, over_eight, over_sixteen, over_thirtytwo
    ):
        x = over_thirtytwo

        x = F.pixel_shuffle(x, upscale_factor=2)
        x = self.conv1x1_1(x)
        x = torch.cat([x, over_sixteen], dim=1)
        x = self.conv1(x)
        stage5_output = x

        x = F.pixel_shuffle(x, upscale_factor=2)
        x = self.conv1x1_2(x)
        x = torch.cat([x, over_eight], dim=1)
        x = self.conv2(x)
        stage6_output = x

        x = F.pixel_shuffle(x, upscale_factor=2)
        x = self.conv1x1_3(x)
        x = torch.cat([x, over_four], dim=1)
        x = self.conv3(x)
        stage7_output = x

        x = F.pixel_shuffle(x, upscale_factor=4)
        stage8_output = self.conv4(x)

        return stage5_output, stage6_output, stage7_output, stage8_output


if __name__ == "__main__":
    # Create an instance of the PixelDecoder
    pixel_decoder = PixelDecoder()

    # Set the height and width
    height = 256
    width = 256

    # Create random input tensors with the specified shapes
    grayscale_image = torch.randn(1, 3, height, width)
    over_four = torch.randn(1, 96, height // 4, width // 4)
    over_eight = torch.randn(1, 192, height // 8, width // 8)
    over_sixteen = torch.randn(1, 384, height // 16, width // 16)
    over_thirtytwo = torch.randn(1, 768, height // 32, width // 32)

    # Forward pass through the PixelDecoder
    stage5_output, stage6_output, stage7_output, stage8_output = pixel_decoder(
        grayscale_image, over_four, over_eight, over_sixteen, over_thirtytwo
    )

    # Print the shapes of the output tensors
    # print("Input over_thirtytwo shape:", over_thirtytwo.shape)
    # print("After pixel_shuffle 1 shape:", stage5_output.shape)
    # print("Input over_sixteen shape:", over_sixteen.shape)
    # print("After concatenation 1 shape:", stage5_output.shape)
    # print("Stage 5 output shape:", stage5_output.shape)
    # print("After pixel_shuffle 2 shape:", stage6_output.shape)
    # print("Input over_eight shape:", over_eight.shape)
    # print("After concatenation 2 shape:", stage6_output.shape)
    # print("Stage 6 output shape:", stage6_output.shape)
    # print("After pixel_shuffle 3 shape:", stage7_output.shape)
    # print("Input over_four shape:", over_four.shape)
    # print("After concatenation 3 shape:", stage7_output.shape)
    # print("Stage 7 output shape:", stage7_output.shape)
    # print("After pixel_shuffle 4 shape:", stage8_output.shape)
    # print("Stage 8 output shape:", stage8_output.shape)

    # Assert the shapes of the output tensors
    assert stage5_output.shape == (
        1,
        512,
        height // 16,
        width // 16,
    ), f"Stage 5 output shape mismatch: {stage5_output.shape}"
    assert stage6_output.shape == (
        1,
        512,
        height // 8,
        width // 8,
    ), f"Stage 6 output shape mismatch: {stage6_output.shape}"
    assert stage7_output.shape == (
        1,
        1024,
        height // 4,
        width // 4,
    ), f"Stage 7 output shape mismatch: {stage7_output.shape}"
    assert stage8_output.shape == (
        1,
        256,
        height,
        width,
    ), f"Stage 8 output shape mismatch: {stage8_output.shape}"

    print("All assertions passed!")
