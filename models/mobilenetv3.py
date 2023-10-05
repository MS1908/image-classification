import numpy as np
import torch.nn.functional as F
from torch import nn


__all__ = [
    'MobileNetV3',
    'mobilenet_v3_small_50',
    'mobilenet_v3_small_75',
    'mobilenet_v3_small',
    'mobilenet_v3_large_50',
    'mobilenet_v3_large_75',
    'mobilenet_v3_large'
]


def conv_bn(inp, oup, stride, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


def conv_1x1_bn(inp, oup, activation=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        activation(inplace=True)
    )


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            HSigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()

        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        if nl == 'RE':
            activation = nn.ReLU
        elif nl == 'HS':
            activation = HSwish
        else:
            raise NotImplementedError("This nonlinear activation is not implemented")

        if se:
            se_layer = SEModule
        else:
            se_layer = nn.Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp),
            activation(inplace=True),

            # dw
            nn.Conv2d(exp, exp, kernel_size=kernel, stride=stride, padding=padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            se_layer(exp),
            activation(inplace=True),

            # pw-linear
            nn.Conv2d(exp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_classes=1000, dropout=0., model_size='small', width_mult=1.0):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280

        if model_size == 'large':
            model_settings = [
                # [k, exp, c, se, nl, s]
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]

        elif model_size == 'small':
            model_settings = [
                # [k, exp, c, se, nl, s]
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        else:
            raise ValueError('model_size argument has to be [small, large]')

        # building first layer
        last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, activation=HSwish)]
        self.classifier = []

        # building mobile blocks
        for k, exp, c, se, nl, s in model_settings:
            output_channel = make_divisible(c * width_mult)
            exp_channel = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel

        # building last several layers
        if model_size == 'large':
            last_conv = make_divisible(960 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, activation=HSwish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(HSwish(inplace=True))
        elif model_size == 'small':
            last_conv = make_divisible(576 * width_mult)
            self.features.append(conv_1x1_bn(input_channel, last_conv, activation=HSwish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(HSwish(inplace=True))
        else:
            raise ValueError('model_size argument has to be [small, large]')

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(last_channel, n_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenet_v3_small_50(**kwargs):
    net = MobileNetV3(model_size='small', width_mult=0.5, **kwargs)
    return net


def mobilenet_v3_small_75(**kwargs):
    net = MobileNetV3(model_size='small', width_mult=0.75, **kwargs)
    return net


def mobilenet_v3_small(**kwargs):
    net = MobileNetV3(model_size='small', width_mult=1., **kwargs)
    return net


def mobilenet_v3_large_50(**kwargs):
    net = MobileNetV3(model_size='large', width_mult=0.5, **kwargs)
    return net


def mobilenet_v3_large_75(**kwargs):
    net = MobileNetV3(model_size='large', width_mult=0.75, **kwargs)
    return net


def mobilenet_v3_large(**kwargs):
    net = MobileNetV3(model_size='large', width_mult=1., **kwargs)
    return net


if __name__ == '__main__':
    import torch

    net = mobilenet_v3_small_50(n_classes=200)
    inp = torch.randn((1, 3, 64, 64))
    oup = net(inp)
    print(oup.size())
    # torch.save(net.state_dict(), 'mbv3_test.pth')
