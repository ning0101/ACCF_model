# from sklearn import metrics
# from sklearn.metrics import mean_squared_error
from math import sqrt



# mse = mean_squared_error(y_actual, y_predicted, squared=False)
# rms = sqrt(mean_squared_error(y_actual, y_predicted))

def MSE(y_actual, y_predicted):
    error = []
    for i in range(len(y_actual)):
        error.append(y_actual[i] - y_predicted[i])
    
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 Square Error
        absError.append(abs(val))#誤差絕對值 Absolute Value of Error
 
    _ = sum(squaredError) / len(squaredError)
    return _


def RMSE(y_actual, y_predicted):

    error = []
    for i in range(len(y_actual)):
        error.append(y_actual[i] - y_predicted[i])
        
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 Square Error
        absError.append(abs(val))#誤差絕對值 Absolute Value of Error
    
    ans = sqrt(sum(squaredError)/len(squaredError))
    return ans

 
# print("MSE = ", sum(squaredError) / len(squaredError))#均方誤差MSE

# print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))#均方根誤差RMSE


y_actual = [2,3,2,1,3,5]
y_predicted = [2,2,3,1,2,4]

print(RMSE(y_actual,y_predicted))