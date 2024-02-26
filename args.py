
import argparse
import torch



# Multiple models scrolling
#既从这个方法中调取的参数进行的训练和修改
def mms_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help='input dimension')
    #我们的总输入数据是7列
    parser.add_argument('--input_size', type=int, default=7, help='input dimension')
    #向后观测336h
    parser.add_argument('--seq_len', type=int, default=336, help='seq len')
    #每个LSTM训练器的预测还是1
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')
    #隐藏层个数
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')
    #设置学习率
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    #设置batch_size
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')
    parser.add_argument('--step_size', type=int, default=10, help='step size')
    #修改了预报的步长，每次要预报24
    parser.add_argument('--pred_step_size', type=int, default=24, help='pred step size')

    parser.add_argument('--gamma', type=float, default=0.3, help='gamma')

    args = parser.parse_args()

    return args





