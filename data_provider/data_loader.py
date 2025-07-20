import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='OBS_FD01_cleaned.csv',
                 time_col='dtime', scale=True, target='r_apower'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.time_col = time_col
        self.target = target
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = 1  # 输入特征数固定为1（仅r_wspd）
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler_x = StandardScaler()  # 用于输入特征 r_wspd
        self.scaler_y = StandardScaler()  # 用于目标特征 r_apower
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 仅选择 r_wspd 作为输入特征
        cols_data = ['r_wspd']
        df_data = df_raw[cols_data]
        # 单独获取目标列 r_apower
        df_target = df_raw[[self.target]]

        if self.scale:
            # 标准化输入特征 r_wspd
            train_data_x = df_data[border1s[0]:border2s[0]]
            self.scaler_x.fit(train_data_x.values)
            data_x = self.scaler_x.transform(df_data.values)
            # 标准化目标特征 r_apower
            train_data_y = df_target[border1s[0]:border2s[0]]
            self.scaler_y.fit(train_data_y.values)
            data_y = self.scaler_y.transform(df_target.values)
        else:
            data_x = df_data.values
            data_y = df_target.values

        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler_x.inverse_transform(data)

    def inverse_transform_target(self, data):
        return self.scaler_y.inverse_transform(data)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='OBS_FD01_cleaned.csv',
                 time_col='dtime', scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.time_col = time_col
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[[self.time_col]]
        df_stamp[self.time_col] = pd.to_datetime(df_stamp[self.time_col]).apply(str)
        self.data_stamp = df_stamp[self.time_col].values
        self.data_stamp = [str(x) for x in self.data_stamp]

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        seq_x_mark = self.data_stamp[s_begin]
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)