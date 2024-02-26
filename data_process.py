
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(file_name):
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + file_name
    df = pd.read_csv(path, encoding='gbk')
    # 只针对非时间列进行填充 否则会报错
    for idx, c in enumerate(df.columns):
        if idx == 0:
            continue   # 跳过时间列
        df[c].fillna(df[c].mean(), inplace=True)  #
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


# Multiple outputs data processing.
def nn_seq_mo(seq_len, B, num):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # print(test)
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, step_size, shuffle):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        seq = []
        for i in range(0, len(dataset) - seq_len - num + 1, step_size):
            train_seq = []
            train_label = []

            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)

            for j in range(i + seq_len, i + seq_len + num):
                train_label.append(load[j])

            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=num, shuffle=False)

    return Dtr, Val, Dte, m, n


# Single step scrolling data processing.
#这里Dte传出去就有问题了
def nn_seq_sss(seq_len, B):
    data = load_data('ETTh1010（17.6-17.7）.csv')
    #如果不能传入scaler，只能实例化一个再传出去
    scaler = StandardScaler()
    # 分割数据集
    train_data = data.iloc[:int(len(data) * 0.6), 1:8]  
    val_data = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), 1:8]
    test_data = data.iloc[int(len(data) * 0.8):, 1:8]

    # 对整个数据集进行标准化
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 将标准化后的数据转换回DataFrame，以便后续处理
    train = pd.DataFrame(train_scaled, columns=data.columns[1:8])
    val = pd.DataFrame(val_scaled, columns=data.columns[1:8])
    test = pd.DataFrame(test_scaled, columns=data.columns[1:8])

    m, n = np.max(train[train.columns[-1]]), np.min(train[train.columns[-1]])

    def process(dataset, batch_size, shuffle):
        load = dataset[dataset.columns[-1]]
        #不再需要归一化
        #load = (load - n) / (m - n)
        load = load.tolist()
        dataset = dataset.values.tolist()
        seqs = []
        for i in range(len(dataset) - seq_len):
            train_seq = []
            train_label = []
            for j in range(i, i + seq_len):
                x = dataset[j][:-1]  # 排除最后一列，因为它是目标
                x.append(load[j])  # 将训练目标添加到序列的末尾
                train_seq.append(x)
            train_label.append(load[i + seq_len])
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seqs.append((train_seq, train_label))

        seq = MyDataset(seqs)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, True)
    Val = process(val, B, True)
    Dte = process(test, B, False)
    print(scaler)
    print("均值:", scaler.mean_)
    print("标准差:", scaler.scale_)
    print('888888888888888888888')
    #此时传出去一个scaler
    #Dtr为训练集，Val为验证集，Dte为测试集。
    return Dtr, Val, Dte, m, n,scaler


# Multiple models single step data processing.
def nn_seq_mmss(seq_len, B, pred_step_size):
    data = load_data('data.csv')

    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    m, n = np.max(train[train.columns[1]]), np.min(train[train.columns[1]])

    def process(dataset, batch_size, step_size, shuffle):
        load = dataset[dataset.columns[1]]
        load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        load = load.tolist()
        #
        seqs = [[] for i in range(pred_step_size)]
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            train_seq = []
            for j in range(i, i + seq_len):
                x = [load[j]]
                for c in range(2, 8):
                    x.append(dataset[j][c])
                train_seq.append(x)
            for j, ind in zip(range(i + seq_len, i + seq_len + pred_step_size), range(pred_step_size)):
                #
                train_label = [load[j]]
                seq = torch.FloatTensor(train_seq)
                train_label = torch.FloatTensor(train_label).view(-1)
                seqs[ind].append((seq, train_label))
        #
        res = []
        for seq in seqs:
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
            res.append(seq)

        return res

    Dtrs = process(train, B, step_size=1, shuffle=True)
    Vals = process(val, B, step_size=1, shuffle=True)
    Dtes = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtrs, Vals, Dtes, m, n


#这是我们需要的mms：多任务多步滚动LSTM模型
def nn_seq_mms(seq_len, B, pred_step_size,scaler):
    data = load_data('ETTh1010（17.6-17.7）.csv')
    # 实例化StandardScaler()
    #scaler = StandardScaler()
    # 分割数据集
    #if scaler is None:
    #scaler = StandardScaler()
    #必须要用.iloc进行分割函数
    train_data = data.iloc[:int(len(data) * 0.6), 1:8]  # 
    val_data = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), 1:8]
    test_data = data.iloc[int(len(data) * 0.8):, 1:8]
    #对整个数据集进行标准化
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    # 将归一化后的数据转换回DataFrame，以便后续处理
    train = pd.DataFrame(train_scaled, columns=data.columns[1:8])
    val = pd.DataFrame(val_scaled, columns=data.columns[1:8])
    test = pd.DataFrame(test_scaled, columns=data.columns[1:8])
    # 使用最后一列作为训练目标来计算m和n(这是归一化的数值，暂时不做变动)
    m, n = np.max(train[train.columns[-1]]), np.min(train[train.columns[-1]])

    def process(dataset, batch_size, step_size, shuffle):
        load = dataset[dataset.columns[-1]] # 修改为使用最后一列数据
        #不再需要归一化
        #load = (load - n) / (m - n)
        dataset = dataset.values.tolist()
        load = load.tolist()
        seqs = [[] for i in range(pred_step_size)]
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            all_train_seqs = []
            for j in range(i, i + pred_step_size):
                train_seq = []
                for k in range(j, j + seq_len):
                    x = []
                    for c in range(1, 7): # 排除最后一列，因为它是目标
                        x.append(dataset[k][c])
                    x.append(load[k]) # 将训练目标添加到序列的末尾
                    train_seq.append(x)

                all_train_seqs.append(train_seq)

            for j, ind in zip(range(i + seq_len, i + seq_len + pred_step_size), range(pred_step_size)):
                train_label = [load[j]]
                seq = torch.FloatTensor(all_train_seqs[j - (i + seq_len)])
                train_label = torch.FloatTensor(train_label).view(-1)
                seqs[ind].append((seq, train_label))

        res = []
        for seq in seqs:
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
            res.append(seq)

        return res
    #关于这里是否要洗牌进行预测，只有Dte是False
    Dtrs = process(train, B, step_size=1, shuffle=True)
    Vals = process(val, B, step_size=1, shuffle=True)
    Dtes = process(test, B, step_size=pred_step_size, shuffle=False)
    #其实m,n都是归一化的数据接口了，此时不用再次返回该数值
    #print(scaler)
    #print("均值:", scaler.mean_)
    #print("标准差:", scaler.scale_)
    #print('2222222222222222')
    return Dtrs, Vals, Dtes, m, n,scaler



#多元任务训练数据抓取
def nn_seq_mtl(seq_len, B, pred_step_size):
    data = load_data('mtl_data_1.csv')
    # 比例
    train = data[:int(len(data) * 0.6)]
    val = data[int(len(data) * 0.6):int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):len(data)]
    # 归一化
    train.drop([train.columns[0]], axis=1, inplace=True)
    val.drop([val.columns[0]], axis=1, inplace=True)
    test.drop([test.columns[0]], axis=1, inplace=True)
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train.values)
    val = scaler.transform(val.values)
    test = scaler.transform(test.values)

    def process(dataset, batch_size, step_size, shuffle):
        dataset = dataset.tolist()
        seq = []
        for i in range(0, len(dataset) - seq_len - pred_step_size, step_size):
            train_seq = []
            for j in range(i, i + seq_len):
                x = []
                for c in range(len(dataset[0])):  # 前24个时刻的所有变量
                    x.append(dataset[j][c])
                train_seq.append(x)
            # 下几个时刻的所有变量
            train_labels = []
            for j in range(len(dataset[0])):
                train_label = []
                for k in range(i + seq_len, i + seq_len + pred_step_size):
                    train_label.append(dataset[k][j])
                train_labels.append(train_label)
            # tensor
            train_seq = torch.FloatTensor(train_seq)
            train_labels = torch.FloatTensor(train_labels)
            seq.append((train_seq, train_labels))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

        return seq

    Dtr = process(train, B, step_size=1, shuffle=True)
    Val = process(val, B, step_size=1, shuffle=True)
    Dte = process(test, B, step_size=pred_step_size, shuffle=False)

    return Dtr, Val, Dte, scaler


def get_mape(x, y):
    """
    :param x: true value
    :param y: pred value
    :return: mape
    """
    return np.mean(np.abs((x - y) / x))
