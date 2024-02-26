# -*- coding:utf-8 -*-

import copy
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_process import nn_seq_mmss, nn_seq_mo, nn_seq_sss, device, setup_seed, nn_seq_mms
from models import LSTM, BiLSTM, Seq2Seq, MTL_LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

setup_seed(20)


def load_data(args, flag, batch_size):
    # 在所有路径之前初始化scaler变量，这里使用None作为默认值
    scaler = None
    if flag == 'mms':
        #这里是从控制台调用所有的参数
        #给定一个实类scaler
        scaler = StandardScaler()
        if scaler is not None:
            print(scaler)
        print('111111111111111')
        Dtr, Val, Dte, m, n,scaler= nn_seq_mms(seq_len=args.seq_len, B=batch_size, pred_step_size=args.pred_step_size,scaler=scaler)
    elif flag == 'mmss':
        Dtr, Val, Dte, m, n = nn_seq_mmss(seq_len=args.seq_len, B=batch_size, pred_step_size=args.pred_step_size)
    elif flag == 'mo' or flag == 'seq2seq':
        Dtr, Val, Dte, m, n = nn_seq_mo(seq_len=args.seq_len, B=batch_size, num=args.output_size)
    else:
        #重新给实例能赋值新scaler么？
        #Dtr为训练集，Val为验证集，Dte为测试集。
        Dtr, Val, Dte, m, n,scaler = nn_seq_sss(seq_len=args.seq_len, B=batch_size)
        print(scaler)
        print("均值:", scaler.mean_)
        print("标准差:", scaler.scale_)
        print('77777777777777777')

    return Dtr, Val, Dte, m, n,scaler


def get_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())

    return np.mean(val_loss)

#重新定义了一个能够得到所有的损失值和损失方差的值
def get_val_loss01(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    all_labels = []
    all_preds = []
    for (seq, label) in Val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            y_pred = model(seq)
            all_labels.append(label.detach().cpu().numpy())
            all_preds.append(y_pred.detach().cpu().numpy())
            
    # 将所有批次的预测和真实标签合并
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # 计算MSE和MAE
    val_mse = mean_squared_error(all_labels, all_preds)
    val_mae = mean_absolute_error(all_labels, all_preds)
    
    return val_mse, val_mae

def get_mtl_val_loss(args, model, Val):
    model.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, labels) in Val:
        seq = seq.to(device)
        labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
        preds = model(seq)
        total_loss = 0
        for k in range(args.n_outputs):
            total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
        total_loss /= preds.shape[0]

        val_loss.append(total_loss.item())

    return np.mean(val_loss)


def train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    #经典见面四个数组糊脸
    train_mse_list = []
    train_mae_list = []
    val_mse_list = []
    val_mae_list = []
    for epoch in tqdm(range(args.epochs)):
        model.train()
        y_true_train, y_pred_train = [], []
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算MSE与MAE
            #得到误差值
            y_true_train.append(label[:, -1].detach().cpu().numpy())  
            y_pred_train.append(y_pred[:, -1].detach().cpu().numpy())
        scheduler.step()
        y_true_train = np.concatenate(y_true_train, axis=0)
        y_pred_train = np.concatenate(y_pred_train, axis=0)
        train_mse = mean_squared_error(y_true_train.reshape(-1), y_pred_train.reshape(-1))  # 展平数组以匹配mean_squared_error的期望输入
        
        #train_mse = mean_squared_error(y_true_train, y_pred_train)
        train_mae = mean_absolute_error(y_true_train.reshape(-1), y_pred_train.reshape(-1))
        train_mse_list.append(train_mse)
        train_mae_list.append(train_mae)
    
        # 假设val_mse和val_mae由get_val_loss函数计算得到
        val_mse, val_mae = get_val_loss01(args, model, Val)
        val_mse_list.append(val_mse)
        val_mae_list.append(val_mae)
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            #出现该报错说明原模型不收敛
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()


    state = {'models': best_model.state_dict()}
    plots_folder = "training_plots"
    # 创建文件夹（如果它不存在）
    os.makedirs(plots_folder, exist_ok=True)
    epochs = range(1, len(train_mse_list) + 1)
    # MSE绘图
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_mse_list, 'bo-', label='Training MSE')
    plt.plot(epochs, val_mse_list, 'ro--', label='Validation MSE')
    plt.title('Training and Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    # MAE绘图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_mae_list, 'bo-', label='Training MAE')
    plt.plot(epochs, val_mae_list, 'ro--', label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # 自动生成文件名并保存绘图
    # 例如，使用当前时间戳来确保文件名唯一
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(plots_folder, f"training_plot_{timestamp}.png")
    plt.savefig(filename)
    # 清除当前图形，以便下次循环时使用新的图形
    plt.close()
    torch.save(state, path)


def seq2seq_train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    batch_size = args.batch_size
    model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=batch_size).to(device)
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_loss = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)


def mtl_train(args, Dtr, Val, path):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    model = MTL_LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size,
                     n_outputs=args.n_outputs).to(device)
    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = 10
    best_model = None
    min_val_loss = 5
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        for (seq, labels) in Dtr:
            seq = seq.to(device)
            labels = labels.to(device)  # (batch_size, n_outputs, pred_step_size)
            preds = model(seq)  # (n_outputs, batch_size, pred_step_size)
            # print(labels.shape)
            # print(preds.shape)
            total_loss = 0
            for k in range(args.n_outputs):
                total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
            total_loss /= preds.shape[0]
            train_loss.append(total_loss.item())
            total_loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        # validation
        val_loss = get_mtl_val_loss(args, model, Val)
        if epoch + 1 >= min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    torch.save(state, path)
