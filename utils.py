# -*- coding: utf-8 -*-
'''
@time: 2019/9/12 15:16

@ author: javis
'''
import torch
import numpy as np
import time, os
from sklearn.metrics import f1_score
from torch import nn
import pandas as pd


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)


# 打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# 多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')  # Sigmoid-BCELoss合成一步
        # BCEWithLogitsLoss损失函数把 Sigmoid 层集成到了 BCELoss 类中.
        # 该版比用一个简单的 Sigmoid 层和 BCELoss 在数值上更稳定,
        # 因为把这两个操作合并为一个层之后, 可以利用 log-sum-exp 的 技巧来实现数值稳定. http://m.elecfans.com/article/805898.html
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()


# 读取数据
def read_csv(file_path, sep=' ', channel_size=8):
    df = pd.read_csv(file_path, sep=sep)
    if channel_size == 12:
        # 计算出4个导联
        # III = II - I
        df['III'] = df['II'] - df['I']
        # aVR = -(I + II) / 2
        df['aVR'] = -(df['I'] + df['II']) / 2
        # aVL = I - II / 2
        df['aVL'] = df['I'] - df['II'] / 2
        # aVF = II - I / 2
        df['aVF'] = df['II'] - df['I'] / 2
    return df
