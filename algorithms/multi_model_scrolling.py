import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from args import mms_args_parser
from model_train import train, load_data
from model_test import mms_rolling_test

args = mms_args_parser()
path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATHS = [path + 'LSTM02\\models\\mms\\' + str(i) + '.pkl' for i in range(args.pred_step_size)]

if __name__ == '__main__':
    flag = 'mms'
    Dtrs, Vals, Dtes, m, n,scaler = load_data(args, flag, batch_size=args.batch_size)
    print(scaler)
    print("均值:", scaler.mean_)
    print("标准差:", scaler.scale_)
    print('333333333333')
    #更新过scaler参数
    #Dtr为训练集，Val为验证集，Dte为测试集。
    for Dtr, Val, path in zip(Dtrs, Vals, LSTM_PATHS):
        train(args, Dtr, Val, path)
    #这步本来是sss,我首先更换成mms看一下输出的scaler是否正确
    #如果不行我要调试sss，进行单步的预报
    #修整sss
    #依靠single_step_scrolling进行预测上的预报和调整
    Dtr, Val, Dte, m, n,scaler = load_data(args, flag='sss', batch_size=1)
    print(scaler)
    #这里的scaler应该在888后面，是8算出来的，但是值和33的一样
    print("均值:", scaler.mean_)
    print("标准差:", scaler.scale_)
    print('44444444444444')
    #这里传入888的值传到mms_test里
    mms_rolling_test(args, Dte, LSTM_PATHS, m, n,scaler)
    #能不能在这里绘制整体的预测图像呢（思路：预报后20%后向内寻找月份天数31*24个数据#
    #然后调取原数据进行拼接预测

