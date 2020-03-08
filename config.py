# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:45

@ author: javis
'''
import os


class Config:
    # for data_process.py
    root = r'/Users/brobear/Downloads/data'
    # root = r'data'
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'testA')
    train_label = os.path.join(root, 'hf_round1_label.txt')
    test_label = os.path.join(root, 'hf_round1_subA.txt')
    arrythmia = os.path.join(root, 'hf_round1_arrythmia.txt')
    root = r'data'

    # for train
    # 训练的模型名称

    kind = 2
    train_data = os.path.join(root, ['train.pth', 'train_without_wc.pth', 'train.pth'][kind])
    model_name = ['resnet34', 'ECGNet', 'DeepNN34'][kind]
    # 在第几个epoch进行到下一个state,调整lr
    stage_epoch = [32, 64, 128]
    # 训练时的batch大小
    batch_size = 64
    # label的类别数
    num_classes = 55
    # 最大训练多少个epoch
    max_epoch = 256
    # 目标的采样长度
    target_point_num = [2048, 5000, 2048][kind]
    # 保存模型的文件夹
    ckpt = 'ckpt'
    # 保存提交文件的文件夹
    sub_dir = 'submit'
    # 初始的学习率
    lr = 1e-3
    # 保存模型当前epoch的权重
    current_w = 'current_w.pth'
    # 保存最佳的权重
    best_w = 'best_w.pth'
    # 学习率衰减 lr/=lr_decay
    lr_decay = 10

    # for test
    temp_dir = os.path.join(root, 'temp')
    top4_state = 'ckpt/DeepNN34_202003021252/best_w.pth'
    top4_catboost = False
    top4_data_val = os.path.join(root, 'top4_data_val.csv')
    top4_data_train = os.path.join(root, 'top4_data_train.csv')
    top4_catboost_model='catboost_model'
    top4_cat_features=[1079, 1080, 1081, 1082]

config = Config()
print("model:", config.model_name)

