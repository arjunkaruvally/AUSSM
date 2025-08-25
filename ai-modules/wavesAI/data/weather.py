from icecream import ic
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import numpy as np
import os

import lightning as L
import torch

from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler
from wavesAI.utils.time_features import time_features



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        # print(cols)
        num_train = int(len(df_raw) * (0.7 if not self.train_only else 1))
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class WeatherDataset(L.LightningDataModule):
    def __init__(self, root_path, data_path, size=None,
                 features='M', target='OT', scale=True,
                 timeenc=0, freq='h', train_only=False,
                 batch_size=32, num_workers=4, pin_memory=True):
        super().__init__()
        self.root_path = root_path
        self.size = [720, 336, 720] if size is None else size
        self.features = features
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset_creator = lambda flag: Dataset_Custom(root_path, flag, size, features, data_path,
                                        target, scale, timeenc, freq, train_only)

        self.train_set = self.dataset_creator("train")

        sample_x, sample_y, _, _ = self.train_set[0]

        self.input_dim = sample_x.shape[-1]
        self.output_dim = sample_y.shape[-1]

        self.save_hyperparameters(ignore=['_class_path'])


    def setup(self, stage: str=None) -> None:
        self.dataset_train = self.train_set
        self.dataset_val = self.dataset_creator("val")
        self.dataset_test = self.dataset_creator("test")

        def collate_batch(batch):
            x = torch.stack([ torch.FloatTensor(arr[0]) for arr in batch ])
            y = torch.stack([ torch.FloatTensor(arr[1]) for arr in batch ])

            return x, y

        self._collate_fn = collate_batch


    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn,
            drop_last=True
        )


if __name__ == '__main__':
    root_path = os.environ["DATA_ROOT"]
    data_path = 'weather.csv'
    data_set = Dataset_Custom(root_path,
                              data_path=data_path,
                              features='M',
                              scale=True,
                              timeenc=0,
                              freq='h',
                              train_only=False)

    x, y, x_mark, y_mark = data_set[0]
    ic(x.shape)
    ic(y.shape)
    ic(x_mark)

    # data_set = WeatherDataset(root_path,
    #                           data_path)
    # data_set.setup()
    #
    # train_loader = data_set.train_dataloader()
    #
    # batch = next(iter(train_loader))
    #
    # x, y = batch
    #
    # ic(x.shape)
    # ic(y.shape)
