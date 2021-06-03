from math import sqrt
import numpy as np

from sklearn.metrics import mean_squared_error

def MSE(y_actual, y_predicted):
    # y_actual = y_actual
    # y_predicted = y_predicted

    error = []
    # print(y_actual.shape)
    # print(y_predicted.shape)
    for i in range(len(y_actual)):
        if y_predicted[i] >= 5:
            y_predicted[i] = 5
        error.append(y_actual[i] - y_predicted[i])

    
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 Square Error
        absError.append(abs(val))#誤差絕對值 Absolute Value of Error
 
    _ = sum(squaredError) / len(squaredError)
    return _

def RMSE(y_actual, y_predicted):
    # y_actual = y_actual
    # y_predicted = y_predicted
    error = []
    for i in range(len(y_actual)):
        if y_predicted[i] >= 5:
            y_predicted[i] = 5
        error.append(y_actual[i] - y_predicted[i])
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 Square Error
        absError.append(abs(val))#誤差絕對值 Absolute Value of Error
    
    ans = sqrt(sum(squaredError)/len(squaredError))
    return ans


def mae(y_actual, y_predicted):
    """
    引數:
    y_true -- 測試集目標真實值
    y_pred -- 測試集目標預測值
    
    返回:
    mae -- MAE 評價指標
    """
    
    n = len(y_actual)
    mae = sum(np.abs(y_actual - y_predicted))/n
    return mae