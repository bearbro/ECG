来源：["合肥高新杯"心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction)

尝试复现第一和第四的模型
* top1
https://github.com/RandomWalk-xzq/Hefei_ECG_TOP1
* top4
https://zhuanlan.zhihu.com/p/98547636

使用了开源babseline:
https://github.com/JavisPeng/ecg_pytorch

系统环境：*centos7 python3.6 pytorch1.0*


# 数据预处理
数据解压放在data目录下，使用8个导联的数据，简单进行train_val数据集划分
```shell
python data_process.py
```

# 模型训练
```shell
python main.py train #从零开始训练
```

# 模型测试
模型测试，在submit文件夹下生成提交结果
```shell
python main.py test --ckpt=..model_path #加载预训练权重进行测试
```

# 修改参数
项目的设置参数在文件config.py中

# Todo
1. top1的模型仍有bug
2. ~~top4的模型传统特征还未加入~~~
