import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class conv_block(nn.Module):
    def __init__(self, in_channels,  out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        init.xavier_uniform(self.conv.weight, gain=np.sqrt(2))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32)
        )
        self.down2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 64)
        )
        self.down3 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 128)
        )

        self.middle = conv_block(128, 128)

        self.up3 = nn.Sequential(
            conv_block(256, 256),
            conv_block(256, 64)
        )

        self.up2 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 32)
        )

        self.up1 = nn.Sequential(
            conv_block(64, 64),
            conv_block(64, 1)
        )

    def forward(self,  x):
        down1 = self.down1(x)
        out = F.max_pool2d(down1, 2)

        down2 = self.down2(out)
        out = F.max_pool2d(down2, 2)

        down3 = self.down3(out)
        out = F.max_pool2d(down3, 2)

        out = self.middle(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down3, out], 1)
        out = self.up3(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down2, out], 1)
        out = self.up2(out)

        out = F.upsample(out, scale_factor=2)
        out = torch.cat([down1, out], 1)
        out = self.up1(out)

        out = F.sigmoid(out)

        return out
