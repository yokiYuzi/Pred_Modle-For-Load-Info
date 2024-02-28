import os
import sys
import pandas as pd
from datetime import datetime
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process import device, get_mape, setup_seed, MyDataset
from model_train import load_data
from models import LSTM, BiLSTM, Seq2Seq, MTL_LSTM
#标准化过程和反过程
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

setup_seed(20)
plt.rcParams["text.usetex"] = False
plt.style.use('ggplot')

def load_data(file_name):
    """
    :return: dataframe
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + file_name
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    return df


def test(args, Dte, path, m, n):
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print(y[-100:])
    # print(pred[-100:])
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


#
    # Dtr, Dte, lis1, lis2 = load_data(args, flag, args.batch_size)
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    # plot
    plot(y, pred)


def list_of_groups(data, sub_len):
    groups = zip(*(iter(data),) * sub_len)
    end_list = [list(i) for i in groups]
    count = len(data) % sub_len
    end_list.append(data[-count:]) if count != 0 else end_list
    return end_list


#单任务滚动预测模型
def ss_rolling_test(args, Dte, path, m, n):
    """
    :param args:
    :param Dte:
    :param path:
    :param m:
    :param n:
    :return:
    """
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    #
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[len(sub_pred) - len(seq) + t]
                else:
                    for t in range(len(sub_pred)):
                        seq[len(seq) - len(sub_pred) + t][0] = sub_pred[t]
            else:
                seq = seq.cpu().numpy().tolist()[0]
            # print(new_seq)
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)
            # print(new_seq)
            seq = [x for x in iter(seq)][0]
            # print(new_seq)
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # print(y_pred)
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)

    y, pred = np.array(y), np.array(pred)
    y = (m - n) * y + n
    pred = (m - n) * pred + n
    print('mape:', get_mape(y, pred))
    plot(y, pred)




# 需要的多任务滚动模型
def mms_rolling_test(args, Dte, PATHS, m, n,scaler):
    print(scaler)
    print("均值:", scaler.mean_)
    print("标准差:", scaler.scale_)
    print('55555555555555555555555555')
    pred = []
    y = []
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    models = []
    for path in PATHS:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        model.load_state_dict(torch.load(path)['models'])
        model.eval()
        models.append(model)
    #是否是Dte就出现问题了呢？
    #果然是Dte就有问题
    Dte = [x for x in iter(Dte)]
    Dte = list_of_groups(Dte, args.pred_step_size)
    print(Dte)
    for sub_item in tqdm(Dte):
        sub_pred = []
        for seq_idx, (seq, label) in enumerate(sub_item, 0):
            model = models[seq_idx]
            label = list(chain.from_iterable(label.data.tolist()))
            y.extend(label)
            if seq_idx != 0:
                seq = seq.cpu().numpy().tolist()[0]
                if len(sub_pred) >= len(seq):
                    for t in range(len(seq)):
                        seq[t][0] = sub_pred[len(sub_pred) - len(seq) + t]
                else:
                    for t in range(len(sub_pred)):
                        seq[len(seq) - len(sub_pred) + t][0] = sub_pred[t]
            else:
                seq = seq.cpu().numpy().tolist()[0]
            # print(new_seq)
            seq = [seq]
            seq = torch.FloatTensor(seq)
            seq = MyDataset(seq)
            seq = DataLoader(dataset=seq, batch_size=1, shuffle=False, num_workers=0)
            # print(new_seq)
            seq = [x for x in iter(seq)][0]
            # print(new_seq)
            with torch.no_grad():
                seq = seq.to(device)
                y_pred = model(seq)
                y_pred = list(chain.from_iterable(y_pred.data.tolist()))
                # print(y_pred)
                sub_pred.extend(y_pred)

        pred.extend(sub_pred)
    #不是程序的错，是标准化的错啊555
    y, pred = np.array(y), np.array(pred)
    # 对y和pred进行反标准化处理
    #最后一列的均值
    mean_last_column =  0
    #最后一列的标准差
    std_dev_last_column = 1

    # 手动进行逆标准化
    #需要及时调整数据问题
    y_inverse = (y* std_dev_last_column) + mean_last_column
    pred_inverse = (pred* std_dev_last_column) + mean_last_column
    #y_inverse = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    #pred_inverse = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    print(y_inverse)

    # 计算MAPE
    mape = mean_absolute_percentage_error(y_inverse, pred_inverse)
    print('mape:', mape)

    # 假设值
    n_predict = 744  # 或720，根据您的需求调整
    n_actual = 2976  # 前面的实际数据点数量
      #引入原始的数据集进行绘制
    data = load_data('ETTh1010（17.6-17.7）.csv')
    # Extracting the target column as actual data points
    y_actual = data.iloc[-n_actual:, -1].values  # Extracting values from the end, backwards
    
    # Simulating prediction data points for demonstration
    #取出预测集
    pred_inverse_selected  =pred_inverse[-n_predict:]

    #np.random.randn(n_predict)  # Simulated prediction data points
    
    # The actual data points and their corresponding time index, counted backwards
    actual_time_index = data.index[-n_actual:]
    
    # The start time for prediction within the actual data
    prediction_start_time = actual_time_index[-n_predict]
    
    # Identifying every first day of the month from the start of prediction to the end of the dataset
    prediction_period = actual_time_index[-n_predict:]  # The time index for the prediction period
    #尝试打印时间戳变量
    #print(actual_time_index)
    monthly_first_day = prediction_period.to_series().resample('MS').first().index
    
    # Plotting
    plt.figure(figsize=(20, 5))
    
    # Plot the actual data
    plt.plot(actual_time_index, y_actual, label='Actual Data', linestyle='-', color='blue')
    
    # Plot the predicted data, ensuring it aligns with the actual data's end
    plt.plot(prediction_period, pred_inverse_selected, label='Predicted Data', linestyle='-', color='red')
    
    # Add a vertical line to indicate the start of prediction
    plt.axvline(x=prediction_start_time, color='green', linestyle='--', label='Start of Prediction')
    
    # Set x-ticks to the first day of each month from the start of prediction
    plt.xticks(monthly_first_day, [time.strftime('%Y-%m-%d') for time in monthly_first_day], rotation=0)
    
    # Formatting the plot
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.title('Oil Temperature Rolling Prediction')
    plt.xlabel('Date')
    plt.ylabel('OT')
    plt.legend()
    plt.tight_layout()  # Adjust layout to make room for the rotated x-tick labels
    plt.show()

    # 定义保存绘图的文件夹名称
    plots_folder = "training_plots"
    # 创建文件夹（如果它不存在）
    os.makedirs(plots_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(plots_folder, f"pred_true_{timestamp}_.png")
    plt.savefig(filename)
    plt.close()

    #计算MAE和MSE
    y_actual_front = y[:n_actual]  # 实际值的前n_actual个数据点
    y_predict_last = pred[-n_predict:]  # 预测值的最后n_predict个数据点
    y_actual_last = y[-n_predict:]  # 实际值的最后n_predict个数据点

# 数据拼接
    y_combined = np.concatenate([y_actual_front, y_predict_last])
    mae = mean_absolute_error(y_actual_last, y_predict_last)
    mse = mean_squared_error(y_actual_last, y_predict_last)

    print("MAE:", mae)
    print("MSE:", mse)

    # 累积MAE和MSE
    cumulative_mae = np.cumsum(np.abs(y_actual_last - y_predict_last)) / np.arange(1, n_predict + 1)
    cumulative_mse = np.cumsum((y_actual_last - y_predict_last) ** 2) / np.arange(1, n_predict + 1)

    # 绘图
    plt.figure(figsize=(18, 6))

    # 绘制MAE增长曲线
    plt.subplot(1, 2, 1)
    plt.plot(cumulative_mae, label='Cumulative MAE', color='blue')
    plt.title('Cumulative MAE over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('MAE')

    # 绘制MSE增长曲线
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_mse, label='Cumulative MSE', color='red')
    plt.title('Cumulative MSE over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('MSE')

    plt.legend()
    filename = os.path.join(plots_folder, f"MSE+MAE_rolling_{timestamp}_.png")
    plt.savefig(filename)

    plt.tight_layout()
    plt.close()


def mtl_test(args, Dte, scaler, path):
    print('loading models...')
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = MTL_LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size,
                     n_outputs=args.n_outputs).to(device)
    # models = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['models'])
    model.eval()
    print('predicting...')
    ys = [[] for i in range(args.n_outputs)]
    preds = [[] for i in range(args.n_outputs)]
    for (seq, targets) in tqdm(Dte):
        targets = np.array(targets.data.tolist())  # (batch_size, n_outputs, pred_step_size)
        for i in range(args.n_outputs):
            target = targets[:, i, :]
            target = list(chain.from_iterable(target))
            ys[i].extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            _pred = model(seq)
            for i in range(_pred.shape[0]):
                pred = _pred[i]
                pred = list(chain.from_iterable(pred.data.tolist()))
                preds[i].extend(pred)

    # ys, preds = [np.array(y) for y in ys], [np.array(pred) for pred in preds]
    ys, preds = np.array(ys).T, np.array(preds).T
    ys = scaler.inverse_transform(ys).T
    preds = scaler.inverse_transform(preds).T
    for ind, (y, pred) in enumerate(zip(ys, preds), 0):
        print(get_mape(y, pred))
        mtl_plot(y, pred, ind + 1)

    plt.show()


def plot(y, pred):
    # plot
    # x = [i for i in range(1, 150 + 1)]
    # # print(len(y))
    # x_smooth = np.linspace(np.min(x), np.max(x), 500)
    # y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='green', marker='*', ms=1, alpha=0.75, label='true')
    #
    # y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    # plt.plot(x_smooth, y_smooth, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    # 只画出测试集中前1000个点
    plt.plot(y[:1000], c='green', label='true')
    plt.plot(pred[:1000], c='red', label='pred')
    plt.grid(axis='y')
    plt.legend()
    plt.show()


def mtl_plot(y, pred, ind):
    # plot
    # x = [i for i in range(1, 150 + 1)]
    # # print(len(y))
    # x_smooth = np.linspace(np.min(x), np.max(x), 500)
    # y_smooth = make_interp_spline(x, y[150:300])(x_smooth)
    plt.plot(y[:1000], c='green', marker='*', ms=1, alpha=0.75, label='true' + str(ind))

    # y_smooth = make_interp_spline(x, pred[150:300])(x_smooth)
    plt.plot(pred[:1000], c='red', marker='o', ms=1, alpha=0.75, label='pred' + str(ind))
    plt.grid(axis='y')
    plt.legend(loc='upper center', ncol=6)
