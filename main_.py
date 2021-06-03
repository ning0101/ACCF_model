from model import CF_model
import argparse
import numpy as np

def parse_args():
    desc = "Tensorflow implementation of CF_recommendation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='MLP', help='[MLP / A_MLP/ C_MLP]')

    parser.add_argument('--dataset', type=str, default='100k', help='[100k / 1m / amazon]')
    parser.add_argument('--dataset_PS', type=str, default='prediction', help='[prediction/ similarity]')
    parser.add_argument('--dataset_IU', type=str, default='user', help='[item/ user]')
    parser.add_argument('--dataset_ACP', type=str, default='cos', help='[adcos/ cos/ pearson]')

    # parser.add_argument('--dataset_C_MLP', type=str, default='adcos', help='[adcos/ cos/ pearson]')


    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run[50, 100, 200, 250, 300]')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch per gpu[5, 10, 20]')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate for model')

    # parser.add_argument('--unit', type=int, default=128, help='LSTM unit')
    #接下來iteration的次數,決定loss下降速度, 可以當一張圖
    '''
    amazon                 100k                    1M
    epoch 100               epoch 100               epoch 100

    batch_size 3           batch_size 3             batch_size 3
    learning_rate 0.01    learning_rate 0.001   learning_rate 0.001
    '''
    return parser.parse_args()

def main():
    args = parse_args()
    model = CF_model(args)
    model.train()
    model.evaluate__()

    # print(evaluate_dataset[:, :50])
    # print(pre[:, :50])
    # print(evaluate_dataset[:, -50:])
    # print(pre[:, -50:])
main()

