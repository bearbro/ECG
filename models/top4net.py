import torch
import torch.nn as nn
import math

from models.top1net import make_dot


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Channel_Attention1d(nn.Module):

    def __init__(self, in_planes, inner_units_ratio=0.5):
        super(Channel_Attention1d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(in_planes*2, int(in_planes * inner_units_ratio))
        self.rule = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(int(in_planes * inner_units_ratio), in_planes*2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_avg = self.avgpool(x)
        channel_max = self.maxpool(x)

        out = torch.cat([channel_avg, channel_max], dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.rule(out)
        out = self.fc2(out)
        out = out.view(out.size(0), 2, -1)
        out = torch.sum(out, dim=1)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1)
        out = x * out
        return out


class DeepNN(nn.Module):
    def __init__(self, block, layers, num_classes=55):
        self.inplanes = 64
        super(DeepNN, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.channel_attention = Channel_Attention1d(512, inner_units_ratio=0.5)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # channel attention
        x = self.channel_attention(x)

        x1 = self.avgpool(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.maxpool2(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat([x1, x2], dim=1)
        out1 = x
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x, out1


def DeepNN34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DeepNN(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


if __name__ == '__main__':
    model = DeepNN34()
    x = torch.randn(20, 8, 5000)
    y, o = model(x)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)
    vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
    vis_graph.view()