import torch
from torch import nn
from torch.nn import functional as F from attention import SelfAttention class VAE_AttentionBlock(nn.Module):
def  init (self, channels): super(). init ()
self.groupnorm = nn.GroupNorm(32, channels) self.attention = SelfAttention(1, channels)
def forward(self, x): residue = x
x = self.groupnorm(x)
 
n, c, h, w = x.shape
x = x.view((n, c, h * w)) x = x.transpose(-1, -2)
x = self.attention(x)
x = x.transpose(-1, -2) x = x.view((n, c, h, w)) x += residue
return x
class VAE_ResidualBlock(nn.Module):
def  init (self, in_channels, out_channels): super(). init ()
self.groupnorm_1 = nn.GroupNorm(32, in_channels)
self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) self.groupnorm_2 = nn.GroupNorm(32, out_channels)
self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if in_channels == out_channels:
self.residual_layer = nn.Identity() else:
self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0
class VAE_Decoder(nn.Sequential): def  init (self):
super().  init  (
nn.Conv2d(4, 4, kernel_size=1, padding=0), nn.Conv2d(4, 512, kernel_size=3, padding=1), VAE_ResidualBlock(512, 512), VAE_AttentionBlock(512), VAE_ResidualBlock(512, 512),
VAE_ResidualBlock(512, 512),
VAE_ResidualBlock(512, 512),
VAE_ResidualBlock(512, 512), nn.Upsample(scale_factor=2)
nn.Conv2d(512, 512, kernel_size=3, padding=1), VAE_ResidualBlock(512, 256),
 
VAE_ResidualBlock(256, 256),
VAE_ResidualBlock(256, 256), nn.Upsample(scale_factor=2),
nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.GroupNorm(32, 128),
nn.SiLU(),
nn.Conv2d(128, 3, kernel_size=3, padding=1),
)
def forward(self, x): x /= 0.18215
for module in self: x = module(x)
return x
