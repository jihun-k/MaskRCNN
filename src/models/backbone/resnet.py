from typing import Optional
import torch.nn as nn
from torch.nn.modules import conv
from torch.nn.modules.conv import Conv2d

class Bottleneck(nn.Module):
    ''' resnet bottleneck '''
    def __init__(self, in_channels, mid_channels, out_channels, conv_bypass=False, downsample_stride=1):
        super(Bottleneck, self).__init__()
        self.conv_bypass = conv_bypass
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, affine=False, track_running_stats=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=downsample_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, affine=False, track_running_stats=False)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        if conv_bypass:
            self.bypass = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=downsample_stride)
            self.bn_bypass = nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = self.bn1(f1)
        f1 = self.conv2(f1)
        f1 = self.bn2(f1)
        f1 = self.conv3(f1)
        f1 = self.bn3(f1)
        f1 = self.relu(f1)
        if self.conv_bypass:
            f2 = self.bypass(x)
            f2 = self.bn_bypass(f2)
        else:
            f2 = x
        return f1 + f2

class ResBlock(nn.Module):
    '''
        resnet block
        bottlenect with downsample + bottleneck * repeat

    '''
    def __init__(self, in_channels, mid_channels, out_channels, repeat, downsample_stride):
        super(ResBlock, self).__init__()
        layers = [Bottleneck(in_channels, mid_channels, out_channels, conv_bypass=True, downsample_stride=downsample_stride)]
        for i in range(1, repeat):
            layers.append(Bottleneck(out_channels, mid_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResNet(nn.Module):
    ''' backbone resnet '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = ResBlock(64, 64, 256, repeat=3, downsample_stride=1)
        self.layer2 = ResBlock(256, 128, 512, repeat=4, downsample_stride=2)
        self.layer3 = ResBlock(512, 256, 1024, repeat=6, downsample_stride=2)
        self.layer4 = ResBlock(1024, 512, 2048, repeat=3, downsample_stride=2)
        
    def forward(self, x):
        ''' forward '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x