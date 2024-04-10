import torch
from torch import nn


class ColorDecoder(nn.Module):
    def __init__(
        self, num_layers=3, num_color_queries=100, embedding_dim=256, num_heads=8
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ColorDecoderBlock(512, embedding_dim, num_heads),
                        ColorDecoderBlock(512, embedding_dim, num_heads),
                        ColorDecoderBlock(256, embedding_dim, num_heads),
                    ]
                )
            ]
        )
        self.color_queries = nn.Parameter(torch.zeros(num_color_queries, embedding_dim))

    def forward(self, over_sixteen, over_eight, over_four):
        """Forward pass of the color decoder.

        Args:
            over_sixteen (torch.Tensor): PixelDecoder hidden state tensor of shape
                (B, 512, H/16, W/16).
            over_eight (torch.Tensor): PixelDecoder hidden state tensor of shape
                (B, 512, H/8, W/8).
            over_four (torch.Tensor): PixelDecoder hidden state tensor of shape
                (B, 256, H/4, W/4).

        Returns:
            torch.Tensor: Color tensor of shape (B, K, C).
        """
        color_queries = self.color_queries.unsqueeze(0).expand(
            over_sixteen.size(0), -1, -1
        )
        for layer in self.layers:
            for i, block in enumerate(layer):
                if i == 0:
                    color_queries = block(color_queries, over_sixteen)
                elif i == 1:
                    color_queries = block(color_queries, over_eight)
                else:
                    color_queries = block(color_queries, over_four)
        return color_queries


class ColorDecoderBlock(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads):
        super().__init__()
        assert input_dim == 256 or input_dim == 512  # (should be 512, 512, 256)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        self.atn_1 = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True,
        )
        self.atn_2 = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, color_queries, image_features):
        """Forward pass of the color decoder block.

        Args:
            color_queries (torch.Tensor): Color queries tensor of shape (B, K, C).
            image_features (torch.Tensor): PixelDecoder hidden state tensor of shape
                (B, input_dim, Hl, Wl), where Hl and Wl are the spatial resolution.

        Returns:
            torch.Tensor: Color tensor of shape (B, K, C).
        """
        assert image_features.size(1) == self.input_dim
        assert color_queries.size(2) == self.embedding_dim
        assert image_features.size(0) == color_queries.size(0)
        B = color_queries.size()[0]

        image_features = image_features.view(B, self.input_dim, -1).permute(0, 2, 1)

        color_queries = self.atn_1(color_queries, image_features, image_features)[0]
        color_queries = color_queries.permute(1, 0, 2)
        color_queries = color_queries + color_queries
        color_queries = self.layer_norm_1(color_queries)

        color_queries = self.atn_2(color_queries, color_queries, color_queries)[0]
        color_queries = color_queries.permute(1, 0, 2)
        color_queries = color_queries + color_queries
        color_queries = self.layer_norm_2(color_queries)

        color_queries = self.mlp(color_queries)
        return color_queries


if __name__ == "__main__":
    color_decoder = ColorDecoder()
    over_sixteen = torch.randn(8, 512, 14, 14)
    over_eight = torch.randn(8, 512, 28, 28)
    over_four = torch.randn(8, 256, 56, 56)
    output = color_decoder(over_sixteen, over_eight, over_four)
    print(output.shape)  # (8, 100, 256)
    print(sum(p.numel() for p in color_decoder.parameters()))  # 2461952 params
