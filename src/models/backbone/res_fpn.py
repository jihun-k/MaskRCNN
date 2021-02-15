import torch.nn as nn
import torchvision


class Bottleneck(nn.Module):
    ''' resnet bottleneck '''
    def __init__(self, in_channels, mid_channels, out_channels, conv_bypass=False, downsample_stride=1):
        super(Bottleneck, self).__init__()
        self.conv_bypass = conv_bypass
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, affine=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=downsample_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, affine=False)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, affine=False)
        self.relu = nn.ReLU(inplace=True)

        for l in [self.conv1, self.conv2, self.conv3]:#, self.bn1, self.bn2, self.bn3]:
            l.weight
        if conv_bypass:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=downsample_stride, bias=False),
                nn.BatchNorm2d(out_channels, affine=False))

    def forward(self, x):
        f1 = self.conv1(x)
        f1 = self.bn1(f1)
        f1 = self.conv2(f1)
        f1 = self.bn2(f1)
        f1 = self.conv3(f1)
        f1 = self.bn3(f1)
        f1 = self.relu(f1)
        if self.conv_bypass:
            f2 = self.downsample(x)
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
        for _ in range(1, repeat):
            layers.append(Bottleneck(out_channels, mid_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Res50FPN(nn.Module):
    ''' backbone resnet '''
    def __init__(self, imagenet_pretrained=False):
        super(Res50FPN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False)#, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = ResBlock(64, 64, 256, repeat=3, downsample_stride=1)
        self.layer2 = ResBlock(256, 128, 512, repeat=4, downsample_stride=2)
        self.layer3 = ResBlock(512, 256, 1024, repeat=6, downsample_stride=2)
        self.layer4 = ResBlock(1024, 512, 2048, repeat=3, downsample_stride=2)

        self.fpn_inner2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.fpn_inner3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.fpn_inner4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1)
        self.fpn_inner5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1)

        self.fpn_layer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_layer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_layer4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_layer5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.fpn_upsample = nn.Upsample(scale_factor=(2,2),)
        self.fpn_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if imagenet_pretrained:
            
            pretrained_model = torchvision.models.resnet50(pretrained=True)
            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()
            for layer, param in pretrained_dict.items():
                if "layer" in layer:
                    split = layer.index('.')
                    layer_self = layer[:split] + ".layers" + layer[split:]
                else:
                    layer_self = layer
                    
                if layer_self in model_dict:
                    # param.requires_grad = False
                    model_dict.update({layer_self: param})
            self.load_state_dict(model_dict)
            
            return
        
        
    def forward(self, x, targets=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.fpn_inner5(c5)
        p4 = self.fpn_inner4(c4) + self.fpn_upsample(p5)
        p3 = self.fpn_inner3(c3) + self.fpn_upsample(p4)
        p2 = self.fpn_inner2(c2) + self.fpn_upsample(p3)

        f2 = self.fpn_layer2(p2)
        f3 = self.fpn_layer3(p3)
        f4 = self.fpn_layer4(p4)
        f5 = self.fpn_layer5(p5)
        f6 = self.fpn_maxpool(f5)

        features = [f2, f3, f4, f5, f6]

        return features
