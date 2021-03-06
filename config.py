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
    r_train_dir = os.path.join(root, 'train_FeaPosition')
    r_test_dir = os.path.join(root, 'testA_FeaPosition')
    root = r'data'

    # for train
    # 训练的模型类型
    kind = 2
    train_data = os.path.join(root, ['train.pth', 'train_without_wc.pth', 'train.pth'][kind])
    # 训练的模型名称
    model_name = ['resnet34', 'ECGNet', 'DeepNN34'][kind]
    # 在第几个epoch进行到下一个state,调整lr
    stage_epoch = [32, 64, 128]
    # 导联个数
    channel_size = 12
    # 训练时的batch大小
    batch_size = 64
    # label的类别数
    num_classes = 55
    # 最大训练多少个epoch
    max_epoch = 256
    # 目标的采样长度
    target_point_num = [2048, 5000, 5000][kind]
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
    top4_catboost = True
    top4_DeepNN = True
    top4_DeepNN_tag = True
    top4_catboost_model = 'catboost_model_only_%d_%d' % (target_point_num, channel_size)
    top4_data_val = os.path.join(root, '%s_data_train.csv' % top4_catboost_model)
    top4_data_train = os.path.join(root, '%s_data_train.csv' % top4_catboost_model)
    top4_rr_k = 4
    # 30 43 0.001  18 350 0.008
    top4_tag_list = [list(range(0, 31 + 1)) + [46], list(range(0, 20))][0]
    if top4_DeepNN:
        top4_cat_features = [] + [55 + 1024 + top4_rr_k * 6 * channel_size + 4 + 3 * 50 * channel_size + 0,
                                  55 + 1024 + top4_rr_k * 6 * channel_size + 4 + 3 * 50 * channel_size + 1]
        if top4_DeepNN_tag:
            top4_cat_features = [] + [
                len(top4_tag_list) + 1024 + top4_rr_k * 6 * channel_size + 4 + 3 * 50 * channel_size + 0,
                len(top4_tag_list) + 1024 + top4_rr_k * 6 * channel_size + 4 + 3 * 50 * channel_size + 1]
    else:
        top4_cat_features = [] + [top4_rr_k * 6 * channel_size + 4 + 3 * 50 * channel_size + 0,
                                  top4_rr_k * 6 * channel_size + 4 + 3 * 50 * channel_size + 1]


config = Config()
print("model:", config.model_name)
