import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.upsamp6 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv6 = self.conv_block(1024, 512)

        self.upsamp7 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv7 = self.conv_block(512, 256)

        self.upsamp8 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv8 = self.conv_block(256, 128)

        self.upsamp9 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv9 = self.conv_block(128, 64)

        self.output_layer = nn.Conv2d(64, num_classes, 1)
        self.num_classes = num_classes

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        x6 = self.conv6(torch.cat([self.upsamp6(x5), x4], 1))
        x7 = self.conv7(torch.cat([self.upsamp7(x6), x3], 1))
        x8 = self.conv8(torch.cat([self.upsamp8(x7), x2], 1))
        x9 = self.conv9(torch.cat([self.upsamp9(x8), x1], 1))

        return self.output_layer(x9)
