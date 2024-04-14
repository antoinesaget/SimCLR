# %%
import random

import numpy as np
import torch.nn as nn
from torch.nn.init import constant_, xavier_normal_


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings for the transformer
    """

    def __init__(self, length, depth):
        super(PositionalEmbedding, self).__init__()

        self.pe = nn.Embedding(length, depth)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class InputProjector(nn.Module):
    """
    Slice the input into subsequences of length window_length and project them with a 1D convolution
    """

    def __init__(self, in_channels, out_channels, window_length):
        super(InputProjector, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=window_length,
            stride=window_length,
        )

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class TFEncoder(nn.Module):
    def __init__(self, args):
        super(TFEncoder, self).__init__()
        self.training = True
        self.timeseries_length = args.timeseries_length
        self.timeseries_n_channels = args.timeseries_n_channels
        self.window_length = args.window_length
        self.projection_depth = args.projection_depth

        self.projection_max_length = self.timeseries_length // self.window_length

        self.positional_encoder = PositionalEmbedding(
            length=self.projection_max_length, depth=self.projection_depth
        )
        self.input_projector = InputProjector(
            in_channels=self.timeseries_n_channels,
            out_channels=self.projection_depth,
            window_length=self.window_length,
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.projection_depth,
                nhead=args.n_attention_heads,
                dim_feedforward=4 * self.projection_depth,
                dropout=args.dropout,
            ),
            num_layers=args.n_encoder_layers,
        )
        self.fc = nn.Linear(self.projection_depth, args.n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):
        # print(f"x.shape: {x.shape}")
        x = x.transpose(1, 2)
        # print(f"x.shape: {x.shape}")
        x = self.input_projector(x)
        # print(f"x.shape: {x.shape}")
        x += self.positional_encoder(x)

        if self.training:
            index = np.arange(x.shape[1])
            random.shuffle(index)
            n_kept = random.randint(4, 10)
            mask = index[:n_kept]
            x = x[:, mask, :]

        # print(f"x.shape: {x.shape}")
        x = self.encoder(x)
        # print(f"x.shape: {x.shape}")
        x = x.mean(dim=1)
        # print(f"x.shape: {x.shape}")
        x = self.fc(x)
        # print(f"x.shape: {x.shape}")
        return x


# %% Test PositionalEmbedding
# import torch
# length = 10
# depth = 4
# batch_size = 64
# pe = PositionalEmbedding(length, depth)
# x = torch.zeros(batch_size, length, depth)
# print(f"x.shape: {x.shape}")
# x = x.transpose(1, 2)
# print(f"x.shape: {x.shape}")
# x_pe = pe(x)
# print(f"x_pe.shape: {x_pe.shape}")
# print(f"x_pe[0]: {x_pe[0]}")
# print(f"x_pe[1]: {x_pe[1]}")
# # %% Test InputProjecter
# length = 60
# n_channels = 4
# depth = 256
# window_length = 10
# batch_size = 64
# ip = InputProjector(n_channels, depth, window_length)
# pe = PositionalEmbedding(6, 256)
# x = torch.zeros(batch_size, n_channels, length)
# x = x.transpose(1, 2)
# x_ip = ip(x)
# x_ip_pe = pe(x_ip)
# print(f"x.shape: {x.shape}")
# print(f"x_ip.shape: {x_ip.shape}")
# print(f"x_ip_pe.shape: {x_ip_pe.shape}")
# # %% Test TFEncoder
# from types import SimpleNamespace

# args = SimpleNamespace(
#     timeseries_length=60,
#     timeseries_n_channels=4,
#     window_length=10,
#     projection_depth=128,
#     n_attention_heads=4,
#     dropout=0.2,
#     n_encoder_layers=16,
#     n_classes=35,
# )
# batch_size = 64
# x = torch.zeros(batch_size, args.timeseries_length, args.timeseries_n_channels)
# model = TFEncoder(args)
# # %%
# y = model(x)

# # %%
# print

# %%
