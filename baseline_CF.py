from math import sqrt
import numpy as np

from sklearn.metrics import mean_squared_error

def MSE(y_actual, y_predicted):
    # y_actual = list(y_actual)
    # y_predicted = list(y_predicted)
    y_actual = y_actual[2]
    y_predicted = y_predicted[2]

    error = []
    count = 0
    # print(y_actual.shape)
    # print(y_predicted.shape)
    for i in range(len(y_actual)):
        if y_predicted[i] >= 5:
            y_predicted[i] = 5
        if y_actual[i]!=0:
            error.append(y_actual[i] - y_predicted[i])
            count = count + 1
    print(count)

    
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 Square Error
        absError.append(abs(val))#誤差絕對值 Absolute Value of Error
 
    _ = sum(squaredError) / len(squaredError)
    return _

def RMSE(y_actual, y_predicted):
    y_actual = y_actual[2]
    y_predicted = y_predicted[2]
    error = []
    for i in range(len(y_actual)):
        if y_predicted[i] >= 5:
            y_predicted[i] = 5
        if y_actual[i]!=0:
            error.append(y_actual[i] - y_predicted[i])
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 Square Error
        absError.append(abs(val))#誤差絕對值 Absolute Value of Error
    
    ans = sqrt(sum(squaredError)/len(squaredError))
    return ans



y_actual = np.load('evaluate_dataset/1m_evaluate_notchange.npy')


y_predicted = np.load('movielen1M_dataset/ml-1m-prediction/item_adcos_prediction_test_movielen1m.npy')

print(y_actual.shape)
print(y_predicted.shape)

mse = mean_squared_error(y_actual[6,:],y_predicted[7,:])
# mse = MSE(y_actual[34, :],y_predicted[33, :])
rmse = sqrt(mse)
print(mse)
print(rmse)






'''
y_item_adcos = np.load('movielen100k_dataset/ml-100k-prediction/item_adcos_prediction_test_movielen100k.npy')
y_user_adcos = np.load('movielen100k_dataset/ml-100k-prediction/user_adcos_prediction_test_movielen100k.npy')
y_user_cos = np.load('movielen100k_dataset/ml-100k-prediction/user_cos_prediction_test_movielen100k.npy')
y_item_cos = np.load('movielen100k_dataset/ml-100k-prediction/item_cos_prediction_test_movielen100k.npy')
y_user_pearson = np.load('movielen100k_dataset/ml-100k-prediction/user_pearson_prediction_test_movielen100k.npy')
y_item_pearson = np.load('movielen100k_dataset/ml-100k-prediction/item_pearson_prediction_test_movielen100k.npy')
print(y_item_adcos[0,99950:])
print(y_user_adcos[0,99950:])
print("--------------------")
print(y_user_cos[0,99950:])
print(y_item_cos[0,99950:])
print("--------------------")
print(y_user_pearson[0,99950:])
print(y_item_pearson[0,99950:])
print("====================")
print(y_item_adcos[0,:50])
print(y_user_adcos[0,:50])
print("--------------------")
print(y_user_cos[0,:50])
print(y_item_cos[0,:50])
print("--------------------")
print(y_user_pearson[0,:50])
print(y_item_pearson[0,:50])
'''


