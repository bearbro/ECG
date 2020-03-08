import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import models
from config import config
from data_process import name2index
from dataset import ECGDataset
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)

print(device)


def make_dateset(model, dataloader, file_path, save=True):
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


def get_dateset(save=True):
    if save:
        if os.path.exists(config.top4_data_val):
            val_df = pd.read_csv(config.top4_data_val, index_col=None)
        else:
            model = getattr(models, config.model_name)(num_classes=config.num_classes)
            state = torch.load(config.top4_state, map_location='cpu')
            model.load_state_dict(state['state_dict'])
            model = model.to(device)

            val_dataset = ECGDataset(data_path=config.train_data, train=False)
            val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
            val_df = make_dateset(model, val_dataloader, config.top4_data_val,save=True)
        print("finish make val_data")

        if os.path.exists(config.top4_data_train):
            train_df = pd.read_csv(config.top4_data_train, index_col=None)
        else:
            model = getattr(models, config.model_name)(num_classes=config.num_classes)
            state = torch.load(config.top4_state, map_location='cpu')
            model.load_state_dict(state['state_dict'])
            model = model.to(device)

            train_dataset = ECGDataset(data_path=config.train_data, train=True)
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4)
            train_df = make_dateset(model, train_dataloader, config.top4_data_train,save=True)
    else:
        model = getattr(models, config.model_name)(num_classes=config.num_classes)
        state = torch.load(config.top4_state, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        model = model.to(device)

        val_dataset = ECGDataset(data_path=config.train_data, train=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4)
        val_df = make_dateset(model, val_dataloader, config.top4_data_val,save=False)

        train_dataset = ECGDataset(data_path=config.train_data, train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4)
        train_df = make_dateset(model, train_dataloader, config.top4_data_train, save=False)

    print("finish make data")
    return train_df, val_df


if __name__ == '__main__':
    train_df, val_df = get_dateset(save=False)
