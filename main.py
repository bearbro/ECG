# -*- coding: utf-8 -*-
'''
@time: 2019/7/23 19:42

@ author: javis
'''
import torch, time, os, shutil
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

import models, utils
import numpy as np
import pandas as pd
from tensorboard_logger import Logger
from torch import nn, optim
from torch.utils.data import DataLoader

from data_process import name2index
from dataset import ECGDataset
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)

print(device)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, fr, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        fr = fr.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        if config.kind == 1:
            output = model(inputs, fr)
        elif config.kind == 2:
            output, _ = model(inputs)
        else:
            output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        # output = torch.sigmoid(output)
        f1 = utils.calc_f1(target, output)
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count


def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, fr, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            fr = fr.to(device)
            if config.kind == 1:
                output = model(inputs, fr)
            elif config.kind == 2:
                output, _ = model(inputs)
            else:
                output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            # output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    return loss_meter / it_count, f1_meter / it_count


def train(args):
    if config.kind == 2 and config.top4_catboost:
        top4_catboost_train(args)
        return
    # model
    model = getattr(models, config.model_name)(num_classes=config.num_classes)
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
    model = model.to(device)
    # data
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=6)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.01)  # L2 正则化
    w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
    criterion = utils.WeightedMultilabel(w)
    # 模型保存文件夹
    model_save_dir = '%s/%s_%s' % (config.ckpt, config.model_name, time.strftime("%Y%m%d%H%M"))
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    # 从上一个断点，继续训练
    if args.resume:
        if os.path.exists(args.ckpt):  # 这里是存放权重的目录
            model_save_dir = args.ckpt
            current_w = torch.load(os.path.join(args.ckpt, config.current_w))
            best_w = torch.load(os.path.join(model_save_dir, config.best_w))
            best_f1 = best_w['loss']
            start_epoch = current_w['epoch'] + 1
            lr = current_w['lr']
            stage = current_w['stage']
            model.load_state_dict(current_w['state_dict'])
            # 如果中断点恰好为转换stage的点
            if start_epoch - 1 in config.stage_epoch:
                stage += 1
                lr /= config.lr_decay
                utils.adjust_learning_rate(optimizer, lr)
                model.load_state_dict(best_w['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(start_epoch - 1))
    logger = Logger(logdir=model_save_dir, flush_secs=2)
    # =========>开始训练<=========
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print('#epoch:%02d stage:%d train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_f1', train_f1, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_f1', val_f1, step=epoch)
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                 'stage': stage}
        save_ckpt(state, best_f1 < val_f1, model_save_dir)
        best_f1 = max(best_f1, val_f1)
        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            best_w = os.path.join(model_save_dir, config.best_w)
            model.load_state_dict(torch.load(best_w)['state_dict'])
            print("*" * 10, "step into stage%02d lr %.3ef" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)


# 用于测试加载模型
def val(args):
    list_threhold = [0.5]
    model = getattr(models, config.model_name)(num_classes=config.num_classes)
    # if args.ckpt:
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    if config.kind == 2 and config.top4_catboost:
        val_f1 = top4_val_epoch(model, val_dataloader)
        print('catboost val_f1:%.3f\n' % val_f1)
    else:
        criterion = nn.BCEWithLogitsLoss()
        for threshold in list_threhold:
            val_loss, val_f1 = val_epoch(model, criterion, val_dataloader, threshold)
            print('threshold %.2f val_loss:%0.3e val_f1:%.3f\n' % (threshold, val_loss, val_f1))


# 提交结果使用
def test(args):
    if config.kind == 2 and config.top4_catboost:
        top4_catboost_test(args)  # todo catboost
    else:
        from dataset import transform
        from data_process import name2index
        name2idx = name2index(config.arrythmia)
        idx2name = {idx: name for name, idx in name2idx.items()}
        utils.mkdirs(config.sub_dir)
        # model
        model = getattr(models, config.model_name)(num_classes=config.num_classes)
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
        model = model.to(device)
        model.eval()
        sub_file = '%s/subA_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
        fout = open(sub_file, 'w', encoding='utf-8')
        print(sub_file)
        with torch.no_grad():
            for line in open(config.test_label, encoding='utf-8'):
                fout.write(line.strip('\n'))
                line = line.strip('\n')
                id = line.split('\t')[0]
                age = line.split('\t')[1]
                sex = line.split('\t')[2]
                if len(age) < 1:
                    age = '-999'
                age = int(age)
                sex = {'FEMALE': 0, 'MALE': 1, '': -999}[sex]
                file_path = os.path.join(config.test_dir, id)
                df = pd.read_csv(file_path, sep=' ').values
                x = transform(df).unsqueeze(0).to(device)
                fr = torch.tensor([age, sex], dtype=torch.float32)
                if config.kind == 1:
                    output = model(x, fr).squeeze().cpu().numpy()
                elif config.kind == 2:
                    output, out2 = model(x)
                    output = output.squeeze().cpu().numpy()
                else:
                    output = model(x).squeeze().cpu().numpy()

                ixs = [i for i, out in enumerate(output) if out > 0.5]
                for i in ixs:
                    fout.write("\t" + idx2name[i])
                fout.write('\n')
        fout.close()


def save_model_list(model_list, dir_path):
    for i, model in enumerate(model_list):
        model.save_model(os.path.join(dir_path, 'model_for_class_%d.dump' % i))
    print('catboost model_list saved')


def load_model_list(dir_path):
    # print('catboost load model_list ')
    model_list = []
    for i in range(config.num_classes):
        model = CatBoostClassifier()
        model.load_model(os.path.join(dir_path, 'model_for_class_%d.dump' % i))
        model_list.append(model)
    return model_list


def model_list_predict(model_list, X):
    y_pre = [[]] * len(model_list)
    for tagi, model in enumerate(model_list):
        y_pre[tagi] = model.predict(X)
    y_pre = np.array(y_pre).T
    return y_pre


def top4_make_dateset(model, dataloader, file_path, save=True):
    print('make dataset')
    if os.path.exists(file_path):
        val_df = pd.read_csv(file_path, index_col=None)
        return val_df
    model.eval()
    values = []
    with torch.no_grad():
        n = 0
        for inputs, fr, target, other_f in dataloader:
            inputs = inputs.to(device)
            output, out1 = model(inputs)
            output = output.cpu()
            out1 = out1.cpu()
            # output, out1 = torch.zeros(64, 10), torch.ones(64, 20)
            vi = torch.cat([output, out1, other_f, fr, target], dim=1)
            values.append(vi)
            n += 1
            if n == 100:
                break
    values = torch.cat(values, dim=0)
    columnslist = []
    columnslist += ['dnn1_%d' % i for i in range(output.size(1))]
    columnslist += ['dnn2_%d' % i for i in range(out1.size(1))]
    print('len_dnn_feature', len(columnslist))
    columnslist += ['other_f_%d' % i for i in range(other_f.size(1))]
    columnslist += ['sex', 'age']
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    columnslist += [idx2name[i] for i in range(target.size(1))]
    df = pd.DataFrame(values.numpy(), columns=columnslist)
    df[columnslist[-4 - 55:]] = df[columnslist[-4 - 55:]].astype(int)
    if save:
        df.to_csv(file_path, index=None)
    return df


def top4_getXy(df):
    X = df.iloc[:, :-1 * config.num_classes]
    y = df.iloc[:, -1 * config.num_classes:]
    return X, y


def top4_catboost_train(args):
    print('top4_catboost_train begin')
    model = getattr(models, config.model_name)(num_classes=config.num_classes)
    # if args.ckpt:
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    # else:
    #     model.load_state_dict(torch.load(config.ckpt.top4_state, map_location='cpu')['state_dict'])
    model = model.to(device)
    val_dataset = ECGDataset(data_path=config.train_data, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
    train_dataset = ECGDataset(data_path=config.train_data, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4)
    train_df = top4_make_dateset(model, train_dataloader, config.top4_data_train, save=False)
    X_train, y_train = top4_getXy(train_df)
    val_df = top4_make_dateset(model, val_dataloader, config.top4_data_val, save=False)
    X_validation, y_validation = top4_getXy(val_df)
    print('train_catboost finish dataset')

    model_list = [None] * y_train.shape[1]
    for tagi, model in enumerate(model_list):
        if model_list[tagi] == None:
            model = CatBoostClassifier(
                iterations=1000,
                random_seed=42,
                eval_metric='F1',
                learning_rate=0.03,  # todo 超参数选择
                task_type={'cuda': 'GPU', 'cpu': 'cpu'}[device.type],
                od_type='Iter',  # 早停
                od_wait=40
            )
            model_list[tagi] = model

        model.fit(
            X_train, y_train.iloc[:, tagi],
            cat_features=config.top4_cat_features,
            eval_set=(X_validation, y_validation.iloc[:, tagi]),
            verbose=False,  # 打印
            plot=False  # 作图
        )
    y_pred_train = model_list_predict(model_list, X_train)
    # train_f1 = utils.calc_f1(y_train.values, y_pred_train)
    train_f1 = f1_score(y_train.values, y_pred_train, average='micro')
    y_pred = model_list_predict(model_list, X_validation)
    # val_f1 = utils.calc_f1(y_validation.values, y_pred)
    val_f1 = f1_score(y_validation, y_pred, average='micro')
    save_model_list(model_list, os.path.join(config.ckpt, config.top4_catboost_model))
    print('catboost train_f1:%.3f\tval_f1:%.3f\n' % (train_f1, val_f1))


def top4_val_epoch(model, val_dataloader):
    print('top4_val_epoch')
    val_df = top4_make_dateset(model, val_dataloader, config.top4_data_val, save=False)
    X_validation, y_validation = top4_getXy(val_df)
    model_list = load_model_list(os.path.join(config.ckpt, config.top4_catboost_model))
    y_pred = model_list_predict(model_list, X_validation)
    # val_f1 = utils.calc_f1(y_validation.values, y_pred)
    val_f1 = f1_score(y_validation, y_pred, average='micro')
    return val_f1


def top4_catboost_test(args):
    print('top4_catboost_test')
    from dataset import transform
    from data_process import name2index
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    utils.mkdirs(config.sub_dir)
    # model
    model = getattr(models, config.model_name)(num_classes=config.num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'])
    model = model.to(device)
    model.eval()
    sub_file = '%s/subA_%s.txt' % (config.sub_dir, time.strftime("%Y%m%d%H%M"))
    print(sub_file)
    fout = open(sub_file, 'w', encoding='utf-8')
    if config.kind == 2:
        model_list = load_model_list(os.path.join(config.ckpt, config.top4_catboost_model))
    with torch.no_grad():
        for line in open(config.test_label, encoding='utf-8'):
            fout.write(line.strip('\n'))
            line = line.strip('\n')
            id = line.split('\t')[0]
            age = line.split('\t')[1]
            sex = line.split('\t')[2]
            if len(age) < 1:
                age = '-999'
            age = int(age)
            sex = {'FEMALE': 0, 'MALE': 1, '': -999}[sex]
            file_path = os.path.join(config.test_dir, id)
            df = pd.read_csv(file_path, sep=' ').values
            x = transform(df).unsqueeze(0).to(device)
            fr = torch.tensor([age, sex], dtype=torch.float32)
            if config.kind == 1:
                output = model(x, fr).squeeze().cpu().numpy()
            elif config.kind == 2:
                output, out1 = model(x)
                output = output.squeeze().cpu().numpy()
                out1 = out1.squeeze().cpu().numpy()
                other_f = [0, 0]  # todo
                df_values = np.concatenate((output, out1, other_f, fr))
                columnslist = []
                columnslist += ['dnn1_%d' % i for i in range(len(output))]
                columnslist += ['dnn2_%d' % i for i in range(len(out1))]
                # print('len_dnn_feature', len(columnslist))
                columnslist += ['other_f_%d' % i for i in range(len(other_f))]
                columnslist += ['sex', 'age']
                df = pd.DataFrame(df_values.reshape(1, -1), columns=columnslist)
                df[df.columns[config.top4_cat_features]] = df[df.columns[config.top4_cat_features]].astype(int)
                # for cindex in config.top4_cat_features:
                #     df[df.columns[cindex]] = df[df.columns[cindex]].astype(int)
                output = model_list_predict(model_list, df).squeeze()
                # print(output)
            else:
                output = model(x).squeeze().cpu().numpy()

            ixs = [i for i, out in enumerate(output) if out > 0.5]
            for i in ixs:
                fout.write("\t" + idx2name[i])
            fout.write('\n')
    fout.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()

    if (args.command == "train"):
        train(args)
    if (args.command == "test"):
        test(args)
    if (args.command == "val"):
        val(args)
