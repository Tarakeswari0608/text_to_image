import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = F.silu(self.linear_1(x))
        return self.linear_2(x)

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time):
        residual = self.residual(x)
        x = self.conv_1(F.silu(self.groupnorm_1(x)))
        time = self.linear_time(F.silu(time)).unsqueeze(-1).unsqueeze(-1)
        x = self.conv_2(F.silu(self.groupnorm_2(x + time)))
        return x + residual

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head, n_embd, d_context=768):
        super().__init__()
        c = n_head * n_embd
        self.groupnorm = nn.GroupNorm(32, c)
        self.in_proj = nn.Conv2d(c, c, 1)
        self.attn1 = SelfAttention(n_head, c, in_proj_bias=False)
        self.attn2 = CrossAttention(n_head, c, d_context, in_proj_bias=False)
        self.ffn_1 = nn.Linear(c, 4 * c * 2)
        self.ffn_2 = nn.Linear(4 * c, c)
        self.out_proj = nn.Conv2d(c, c, 1)

    def forward(self, x, context):
        residual = x
        x = self.in_proj(self.groupnorm(x))
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).transpose(-1, -2)
        x = self.attn1(x) + x
        x = self.attn2(x, context) + x
        x_proj, gate = self.ffn_1(x).chunk(2, dim=-1)
        x = self.ffn_2(x_proj * F.gelu(gate)) + x
        x = x.transpose(-1, -2).view(n, c, h, w)
        return self.out_proj(x) + residual

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, 3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, 3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
        ])
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        skips = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skips.append(x)
        x = self.bottleneck(x, context, time)
        for layer in self.decoders:
            x = torch.cat((x, skips.pop()), dim=1)
            x = layer(x, context, time)
        return x

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(F.silu(self.norm(x)))

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.output = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        time = self.time_embedding(time)
        x = self.unet(latent, context, time)
        return self.output(x)
