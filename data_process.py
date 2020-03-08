# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:44
数据预处理：
    1.构建label2index和index2label
    2.划分数据集
@ author: javis
'''
import os, torch
import numpy as np
from config import config
import pandas as pd

# 保证每次划分数据一致
np.random.seed(41)


def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def split_data(file2idx, name2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:文件的标签id
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    data = set(os.listdir(config.train_dir))
    # todo 去重
    new_data = set()
    dv = set()
    for i in data:
        file_path = os.path.join(config.train_dir, i)
        with open(file_path, 'r', encoding='utf-8') as fr:
            ss = fr.read()
        if ss in dv:
            continue  # todo 去重
        elif name2idx['窦性心律'] in file2idx[i] and name2idx['窦性心律不齐'] in file2idx[i]:
            continue  # todo  去除同时出现窦性心律和窦性心律不齐
        else:
            new_data.add(i)
            dv.add(ss)
    data = new_data

    for i in data:
        if (name2idx['不完全性右束支传导阻滞'] in file2idx[i] or name2idx['完全性右束支传导阻滞'] in file2idx[i]) \
                and name2idx['右束支传导阻滞'] not in file2idx[i]:
            file2idx[i].append(name2idx['右束支传导阻滞'])
        # todo 部分数据标记为不完全性右束支传导阻滞或完全性右束支传导阻滞，却没标记为右束支传导阻滞
        if name2idx['完全性左束支传导阻滞'] in file2idx[i] \
                and name2idx['右束支传导阻滞'] not in file2idx[i]:
            file2idx[i].append(name2idx['左束支传导阻滞'])
            file2idx[i].append(name2idx['左束支传导阻滞'])
        # todo 部分数据标记为完全性左束支传导阻滞，却没标记为左束支传导阻滞

    val = set()
    idx2file = [[] for _ in range(config.num_classes)]
    for file in data:
        for idx in file2idx[file]:
            idx2file[idx].append(file)

    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)

    return list(train), list(val)


def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    file2age = dict()
    file2sex = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        if len(arr[1]) < 1:
            arr[1] = '-999'
        file2age[id] = int(arr[1])
        file2sex[id] = {'FEMALE': 0, 'MALE': 1, '': -999}[arr[2]]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
    return file2index, file2age, file2sex


def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * config.num_classes
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)


def train(name2idx, idx2name):
    file2idx, file2age, file2sex = file2index(config.train_label, name2idx)
    train, val = split_data(file2idx, name2idx)
    wc = count_labels(train, file2idx)
    print(wc)
    # wc = np.array([3] * len(wc)) # 权重相同
    # print(wc)
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx, 'wc': wc, 'file2age': file2age,
          'file2sex': file2sex}
    torch.save(dd, config.train_data)


if __name__ == '__main__':
    pass
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train(name2idx, idx2name)
