"""
Attention Modules for YOLOv8 Fire and Smoke Detection
Add this code to your ultralytics/nn/modules/block.py or create a new file attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['ECA', 'SA', 'GAM', 'ResBlock_CBAM']


class ECA(nn.Module):
    """Efficient Channel Attention Module"""
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        
        # Calculate kernel size adaptively
        t = int(abs((math.log(channel, 2) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        
        # 1D convolution for channel attention
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        b, c, h, w = x.size()
        y = x.view(b, c, -1).mean(dim=2).unsqueeze(-1)
        
        # 1D convolution
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # Sigmoid activation
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class SA(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, channel, reduction=16):
        super(SA, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel-wise max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(concat)
        
        return x * attention


class GAM(nn.Module):
    """Global Attention Module"""
    def __init__(self, in_channels, rate=4):
        super(GAM, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel attention
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        
        # Apply channel attention
        x = x * x_channel_att
        
        # Spatial attention
        x_spatial_att = self.spatial_attention(x).sigmoid()
        
        # Apply spatial attention
        out = x * x_spatial_att
        
        return out


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1))
        x = x * spatial_att
        
        return x


class ResBlock_CBAM(nn.Module):
    """Residual Block with CBAM Attention"""
    def __init__(self, channel, reduction=16):
        super(ResBlock_CBAM, self).__init__()
        
        # Residual block
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel)
        
        # CBAM attention
        self.cbam = CBAM(channel, reduction)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply CBAM attention
        out = self.cbam(out)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out