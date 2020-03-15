import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch

from models.se_module import SELayer


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, conv_num=3):
        super(BasicBlock2d, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=(1, kernel_size),
                               stride=stride, bias=False, padding=(0, kernel_size // 2))
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride
        self.layers1 = nn.Sequential()
        self.layers1.add_module("BatchNorm2d", self.bn1)
        self.layers1.add_module('relu', self.relu)
        self.layers1.add_module('conv1', self.conv1)
        self.layers2 = None
        if downsample is not None:
            self.layers2 = nn.Sequential()
            self.layers2.add_module('downsample', self.downsample)
            for i in range(1, conv_num):
                self.layers2.add_module('downsample', self.downsample)

        for i in range(1, conv_num):
            if conv_num == i - 1:
                self.layers1.add_module('dropout',
                                        nn.Dropout(.2)
                                        )
            self.layers1.add_module("BatchNorm2d", self.bn2)
            self.layers1.add_module('relu', self.relu2)
            self.layers1.add_module('conv1',
                                    # self.conv1
                                    nn.Conv2d(planes, planes, kernel_size=(1, kernel_size),
                                              stride=1, bias=False, padding=(0, kernel_size // 2))
                                    )

        self.se = SELayer(planes, reduction=16)

    def forward(self, x):
        residual = x
        out = self.layers1(x)
        out = self.se(out)

        if self.layers2 is not None:
            residual = self.layers2(x)

        # d = residual.shape[2] - out.shape[2]
        # if d == 0:
        #     out = residual + out
        # else:
        #     out = residual[:, :, 0:-d] + out

        out = residual + out
        return out


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1, conv_num=3):
        super(BasicBlock1d, self).__init__()
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=kernel_size,
                               stride=stride, bias=False, padding=kernel_size // 2)
        self.conv12 = nn.Conv1d(planes, planes, kernel_size=kernel_size,
                                stride=1, bias=False, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.layers1 = nn.Sequential()
        self.layers1.add_module("BatchNorm1d", self.bn1)
        self.layers1.add_module('relu', self.relu)
        self.layers1.add_module('conv1', self.conv1)

        for i in range(1, conv_num):
            if conv_num == i - 1:
                self.layers1.add_module('dropout',
                                        nn.Dropout(.2)
                                        )

            self.layers1.add_module("BatchNorm1d", self.bn2)
            self.layers1.add_module('relu', self.relu)
            self.layers1.add_module('conv1_%d' % i,
                                    # self.conv12
                                    nn.Conv1d(planes, planes, kernel_size=kernel_size,
                                              stride=1, bias=False, padding=kernel_size // 2)
                                    )

        self.se = SELayer(planes, reduction=16)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size,
                               stride=1, bias=False, padding=kernel_size // 2)
        self.layers2 = nn.Sequential()
        self.layers2.add_module('BatchNorm1d', self.bn2)
        self.layers2.add_module('relu', self.relu)
        self.layers2.add_module('conv1', self.conv2)

        for i in range(1, conv_num):
            if conv_num == i - 1:
                self.layers2.add_module('dropout',
                                        nn.Dropout(.2)
                                        )
            self.layers2.add_module('BatchNorm1d', self.bn2)
            self.layers2.add_module('relu', self.relu)
            self.layers2.add_module('conv1_%d' % i,
                                    # self.conv2
                                    nn.Conv1d(planes, planes, kernel_size=kernel_size,
                                              stride=1, bias=False, padding=kernel_size // 2)
                                    )

    def forward(self, x):
        residual = x
        out = self.layers1(x)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # d = residual.shape[2] - out.shape[2]
        # if d == 0:
        #     out = residual + out
        # else:
        #     out = residual[:, :, 0:-d] + out

        out = residual + out

        residual2 = out

        out = self.layers2(out)
        out = self.se(out)

        out += residual2

        return out


class ECGNet(nn.Module):
    def __init__(self, input_channel=1, layers=[[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
                 sizes=[[3, 3, 3, 3, 3, 3], [5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7, 7]], num_classes=55):
        super(ECGNet, self).__init__()
        self.layers = layers
        self.sizes = sizes
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1, 50), stride=(1, 2),
                               padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.inplanes = 32
        self.layers = nn.Sequential()
        self.layers.add_module('layer_1', self._make_layer2d(BasicBlock2d, 32, 1, stride=(1, 2), size=15, conv_num=1))
        self.layers.add_module('layer_2', self._make_layer2d(BasicBlock2d, 32, 1, stride=(1, 2), size=15, conv_num=1))
        self.layers.add_module('layer_3', self._make_layer2d(BasicBlock2d, 32, 1, stride=(1, 2), size=15, conv_num=1))

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i, size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1, 1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size),
                                    self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1, 1), size=sizes[i][1]))
            self.inplanes *= 8
            self.layers2.add_module('layer{}_2_1'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size),
                                    self._make_layer1d(BasicBlock1d, 256, layers[i][5], stride=2, size=sizes[i][5]))

            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        self.fc = nn.Linear(256 * len(sizes) + 2, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer2d(self, block, planes, blocks=1, stride=(1, 1), size=1, conv_num=1):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=size, conv_num=conv_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_layer1d(self, block, planes, blocks=1, stride=1, size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x0, fr=None):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.layers(x0)
        xs = []
        for i in range(len(self.sizes)):
            x = self.layers1_list[i](x0)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = self.layers2_list[i](x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            xs.append(x)
        out = torch.cat(xs, dim=1)
        if fr is not None:
            out = torch.cat([out, fr], dim=1)
        out = self.fc(out)
        # out = self.sigmoid(out)
        return out


from graphviz import Digraph
from torch.autograd import Variable


# 画画用的
def make_dot(var, params=None):
    """
    画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    蓝色节点表示有梯度计算的变量Variables;
    橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled', shape='box', align='left',
                     fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # 多输出场景 multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    return dot


if __name__ == '__main__':
    model = ECGNet(1, num_classes=55)
    x = torch.randn(20, 8, 5000)
    fr = torch.randn(20, 2)
    model(x, fr)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print(num_params)

    # from tensorboardX import SummaryWriter
    #
    # writer = SummaryWriter('log')  # 建立一个保存数据用的东西
    # with SummaryWriter(comment='top1net') as w:
    #     w.add_graph(model, (x,))
    vis_graph = make_dot(model(x, fr), params=dict(model.named_parameters()))
    vis_graph.view()
