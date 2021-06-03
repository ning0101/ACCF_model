import numpy as np

evaluate = np.load('evaluate_dataset/100K_evaluate_notchange.npy')

evaluate = evaluate[round(evaluate.shape[0]*0.8)+1:, :]
print(evaluate.shape)
# last_month = evaluate[-1:]
# print(last_month.shape)



# last_month = list(last_month)
# last_month = last_month[0]
# print(len(last_month))s

# evaluate_100k = []

# for i in range(len(last_month)):
#     if last_month[i] > 0 :
#         evaluate_100k.append(i)

# print(len(evaluate_100k))

