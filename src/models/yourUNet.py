# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourUNet.  Un réseau inspiré de UNet
mais comprenant des connexions résiduelles et denses.  Soyez originaux et surtout... amusez-vous!

'''

class ResidualBlock(nn.Module):
    """Bloc résiduel avec gestion précise des dimensions"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class DenseLayer(nn.Module):
    """Couche dense optimisée"""
    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=1),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        new_features = self.bottleneck(x)
        return torch.cat([x, new_features], 1)

class YourUNet(CNNBaseModel):
    """Version finale avec Deep Supervision et gestion d'inférence"""
    
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__(num_classes, init_weights)
        
        # Encoder
        self.enc1 = ResidualBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = nn.Sequential(
            ResidualBlock(256, 512),
            DenseLayer(512),
            nn.Conv2d(512 + 32, 512, kernel_size=1)
        )
        
        # Decoder
        self.up1 = self._up_block(512, 256)
        self.up2 = self._up_block(512, 128)
        self.up3 = self._up_block(256, 64)
        
        # Deep Supervision
        self.ds1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.ds2 = nn.Conv2d(128, num_classes, kernel_size=1)
        
        # Sortie finale
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x, deep_supervision=True):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x2 = self.enc2(x2)
        x3 = self.pool2(x2)
        x3 = self.enc3(x3)
        
        # Bridge
        x4 = self.pool3(x3)
        x4 = self.bridge(x4)
        
        # Decoder
        x = self.up1(x4)
        ds1_out = self.ds1(x)
        x = torch.cat([x, x3], dim=1)
        
        x = self.up2(x)
        ds2_out = self.ds2(x)
        x = torch.cat([x, x2], dim=1)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        
        final_out = self.final_conv(x)
        
        if deep_supervision:
            return final_out, ds1_out, ds2_out
        else:
            return final_out
'''
Fin de votre code.
'''