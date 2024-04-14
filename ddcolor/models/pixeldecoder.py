import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    def __init__(self):
        super(PixelDecoder, self).__init__()

    def forward(
        self, grayscale_image, over_four, over_eight, over_sixteen, over_thirtytwo
    ):
        stage_3 = over_sixteen
        stage_2 = over_eight
        stage_1 = over_four

        # stage 5
        stage_5 = nn.PixelShuffle(upscale_factor=2)(over_thirtytwo)
        stage_5 = torch.cat([stage_5, stage_3], dim=1)
        stage_5 = nn.Conv2d(in_channels=stage_5.shape[1], out_channels=512, kernel_size=3, padding=1)(stage_5)

        assert stage_5.shape[1:] == (512, 16, 16)

        # stage 6
        stage_6 = nn.PixelShuffle(2)(over_sixteen)
        stage_6 = torch.cat([stage_6, stage_2], dim=1)
        stage_6 = nn.Conv2d(stage_6.shape[1], 512, 3, padding=1)(stage_6)

        assert stage_6.shape[1:] == (512, 32, 32)

        # stage 7
        stage_7 = nn.PixelShuffle(2)(over_eight)
        stage_7 = torch.cat([stage_7, stage_1], dim=1)
        stage_7 = nn.Conv2d(stage_7.shape[1], 256, 3, padding=1)(stage_7)

        assert stage_7.shape[1:] == (256, 64, 64)

        # stage 8
        stage_8 = nn.PixelShuffle(4)(over_four)
        stage_8 = nn.Conv2d(stage_8.shape[1], 256, 3, padding=1)(stage_8)
        assert stage_8.shape[1:] == (256, 256, 256), f"Stage 8 shape mismatch: {stage_8.shape}"

        return stage_5, stage_6, stage_7, stage_8


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
        256,
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
