# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
import math
import os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal


def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]


def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    sig = resample(sig, config.target_point_num)
    # # 数据增强
    if (not config.top4_catboost) and train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        # if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """

    def __init__(self, data_path, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load(data_path)
        self.train = train
        self.data = dd['train'] if train else dd['val']
        self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])
        self.file2age = dd['file2age']
        self.file2sex = dd['file2sex']

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(config.train_dir, fid)
        df = pd.read_csv(file_path, sep=' ')
        x = transform(df.values, self.train)
        age = self.file2age[fid]
        sex = self.file2sex[fid]
        fr = torch.tensor([age, sex], dtype=torch.float32)
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        if config.top4_catboost:
            # 获取其他传统特征
            r_features_file = os.path.join(config.r_train_dir, fid.replace('txt', 'fea'))
            other_f = get_other_features(df, r_features_file)  # 1684
            return x, fr, target, other_f
        return x, fr, target

    def __len__(self):
        return len(self.data)


# df 5000*8
def get_QRS_features(df, rr_idx):
    # QRS波形态的特征提取
    # 在一个导联上，以R峰为基准，向左右两边提取固定长度的数据（-0.2，0.4），【100，200】
    # 获得若千个含有R峰的片段, 将其叠加。
    # 通过评估Templates的重合度，来间接反映出病变QRS波的特征
    # 每个Templates同索引的极差(max - min)
    # 每个Templates同索引的均值(mean)
    # 每个Templates同索引的标准差(std)
    # 经过resample后输出三个等长的长度为50的序列作为特征。  重采样 ？
    rr = []
    for i in rr_idx:
        begin = i - int(config.target_point_num * 0.2 / 10)
        end = i + int(config.target_point_num * 0.4 / 10)
        if begin >= 0 and end <= config.target_point_num:
            rr.append((begin, end))
    QRS = []
    for i in range(8):
        x = df.iloc[:, i].values
        t = []
        for j in rr:
            t.append(x[j[0]:j[1]])
        t_mean = np.mean(t, axis=0)
        t_max_min = np.max(t, axis=0) - np.min(t, axis=0)
        t_std = np.std(t, axis=0)
        t_mean = signal.resample(t_mean, 50)
        t_max_min = signal.resample(t_max_min, 50)
        t_std = signal.resample(t_std, 50)
        QRS += np.concatenate([t_mean, t_max_min, t_std]).tolist()

    return QRS  # 50*3*8


# df 5000*8
def get_other_features(df, file_path):
    rr_idx = pd.read_csv(file_path, sep='\t', header=None).values[:, 5]  # R波峰值的下标
    RR = np.vsplit(df, rr_idx)[1:-2]  # RR区间
    # RR区间：min，max，mean，std，偏度(skewness)和峰度(kurtosis）6*？*通道数
    RR_min = [i.min(axis=0).to_list() for i in RR]
    RR_max = [i.max(axis=0).to_list() for i in RR]
    RR_mean = [i.mean(axis=0).to_list() for i in RR]
    RR_std = [i.std(axis=0).to_list() for i in RR]
    RR_skewness = [i.skew(axis=0).to_list() for i in RR]  # 计算偏斜度
    RR_kurtosis = [i.kurt(axis=0).to_list() for i in RR]  # 计算峰度
    # 选择中间的k个
    k = 10
    k_begin = (len(RR_min) - k) // 2
    RR_section_feature = [RR_min, RR_max, RR_mean, RR_std, RR_skewness, RR_kurtosis]
    RR_section_feature_k = []  # 6*8*k  k=10  480
    for i in range(len(RR_section_feature)):
        RR_section_feature_k += RR_section_feature[i][k_begin:k_begin + k]
    # RR_section_feature_k k*6*8
    RR_section_feature_k = np.array(RR_section_feature_k).T.reshape((-1,)).tolist()

    rr_len = [len(i) for i in RR]  # RR区间长度

    rrc = []  # 相邻RR间期差值
    for i in range(1, len(rr_len)):
        rr_i = rr_len[i] - rr_len[i - 1]
        rrc.append(rr_i)
    # pNN50:相邻RR间期差距大于50ms的比率 大于50ms的RR间期对数/总对数 1
    pNN50 = len([i for i in rrc if i * 10000 / config.target_point_num > 50]) / len(rrc)

    # R波密度: R波个数 / 记录长度 1
    rP = len(RR) / ((rr_idx[-1] - rr_idx[0]) * 0.002)
    # RMSSD: 相邻RR间期差值的均方根 1
    RMSSD = math.sqrt(sum([i ** 2 for i in rrc]) / len(rrc))
    # RR间期的采样熵:衡量RR间期变化混乱   1
    RRHX = sum([i * math.log2(i) for i in rr_len]) * -1

    QRS_features = get_QRS_features(df, rr_idx)  # 1200

    other_f = RR_section_feature_k + [pNN50, rP, RMSSD, RRHX] + QRS_features  # 480+4+1200
    return torch.tensor(other_f, dtype=torch.float32)


if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    # for i in range(len(d)):
    #     print(i)
    #     d[i]
    # d = ECGDataset(config.train_data, False)
    # for i in range(len(d)):
    #     print(i)
    #     d[i]
    print(d[0])
