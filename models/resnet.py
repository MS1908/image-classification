from torch import nn

__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50'
]


class BasicBlock(nn.Module):
    # Residual block for ResNet18 and ResNet34

    expansion = 1

    def __init__(self, inp, oup, stride=1):
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(oup * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != oup * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup * BasicBlock.expansion)
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    # Residual block for ResNet50

    expansion = 4

    def __init__(self, inp, oup, stride=1):
        super(Bottleneck, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup * Bottleneck.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inp != oup * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(oup * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.conv(x) + self.shortcut(x)
        x = self.activation(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_block, n_classes=100):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(block, 64, num_block[0], 1)
        self.layer2 = self._make_layer(block, 128, num_block[1], 2)
        self.layer3 = self._make_layer(block, 256, num_block[2], 2)
        self.layer4 = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, num_blocks, oup, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, oup, stride))
            self.in_channels = oup * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def resnet18(**kwargs):
    return ResNet(block=BasicBlock, num_block=[2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(block=BasicBlock, num_block=[3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(block=Bottleneck, num_block=[3, 4, 6, 3], **kwargs)


if __name__ == '__main__':
    import torch

    net = resnet18(n_classes=200)
    inp = torch.randn((1, 3, 64, 64))
    oup = net(inp)
    print(oup.size())
    torch.save(net.state_dict(), 'resnet_test.pth')
