

from models import MLP
import matplotlib.pyplot as plt
import codecs, json
import numpy as np


epoch = 35
batch_size_path = [ "experiment/1m/MLP/batch_size/item/cos/batchsize_1.npy",
                    "experiment/1m/MLP/batch_size/item/cos/batchsize_2.npy",
                    "experiment/1m/MLP/batch_size/item/cos/batchsize_3.npy",
                    "experiment/1m/MLP/batch_size/item/cos/batchsize_4.npy",
                    "experiment/1m/MLP/batch_size/item/cos/batchsize_5.npy"]

def show_history(path):
    data = np.load(path,allow_pickle=True).item()
    val_rmse_loss = data["val_root_mean_squared_error"]
 
    plt.plot(val_rmse_loss)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def show_batch_size(path):
    V_lstm = []
    for i in path:
        data = np.load(i)
        print(data.shape)
        V_lstm.append(data[epoch])

    labels = ['1', '2', '3', '4', '5']

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    ax.set_ylim([0.6, 0.8])
    # ax = plt.subplots([0.05, 2.0])
    rects2 = ax.bar(x, V_lstm, width, label='MLP-cos-item')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Batch_size')
    ax.set_title('INCF Model - cosine')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
 
    fig.tight_layout()

    plt.show()


def show_unit(path):
    F_lstm = []
    V_lstm = []
    for _, i in enumerate(path):
        data = np.load(i,allow_pickle=True).item()
        val_rmse_loss = data["val_root_mean_squared_error"]
        val_rmse_loss = np.array(val_rmse_loss)
        V_lstm.append(val_rmse_loss[-1])
        F_lstm.append(val_rmse_loss[-1])

    labels = ['32', '64', '128', '256', '512']

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    # ax.set_ylim([0.14, 0.15])
    # ax = plt.subplots([0.05, 2.0])
    rects1 = ax.bar(x - width/2, F_lstm, width, label='V-lstm')
    rects2 = ax.bar(x + width/2, V_lstm, width, label='F-lstm')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Unit')
    ax.set_title('ELR Model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
 
    fig.tight_layout()

    plt.show()


def show_learning(path):
    F_lstm = []
    V_lstm = []
    for _, i in enumerate(path):
        data = np.load(i,allow_pickle=True).item()
        val_rmse_loss = data["val_root_mean_squared_error"]
        val_rmse_loss = np.array(val_rmse_loss)
        print(len(val_rmse_loss))
        V_lstm.append(val_rmse_loss[25])
        F_lstm.append(val_rmse_loss[25])

    labels = ['0.1', '0.01', '0.001', '0.0001', '0.00001']

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    # ax.set_ylim([0.0475, 0.0525])
    ax.set_ylim([0.04, 0.06])
    # ax = plt.subplots([0.05, 2.0])
    rects1 = ax.bar(x - width/2, F_lstm, width, label='V-lstm')
    rects2 = ax.bar(x + width/2, V_lstm, width, label='F-lstm')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Learning_rate')
    ax.set_title('ELR Model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
 
    fig.tight_layout()

    plt.show()


def show_layer(path, path1):
    F_lstm = []
    V_lstm = []
    for _, i in enumerate(path):
        data = np.load(i,allow_pickle=True).item()
        val_rmse_loss = data["val_root_mean_squared_error"]
        val_rmse_loss = np.array(val_rmse_loss)
        F_lstm.append(val_rmse_loss[30])

    for _, i in enumerate(path1):
        data = np.load(i,allow_pickle=True).item()
        val_rmse_loss = data["val_root_mean_squared_error"]
        val_rmse_loss = np.array(val_rmse_loss)
        V_lstm.append(val_rmse_loss[30])

    labels = ['1', '2', '3', '4', '5']

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    ax.set_ylim([0.044, 0.06])
    # ax = plt.subplots([0.04, 0.05])
    rects1 = ax.bar(x - width/2, F_lstm, width, label='V-lstm')
    rects2 = ax.bar(x + width/2, V_lstm, width, label='F-lstm')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Learning_rate')
    ax.set_xlabel('Layers')
    ax.set_title('ELR Model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
 
    fig.tight_layout()

    plt.show()








###
# show_batch_size(batch_size_path)