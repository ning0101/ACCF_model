from matrix_factorization import BaselineModel, KernelMF, train_update_test_split

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# cols = ["user_id", "item_id", "rating", "timestamp"]
# movie_data = pd.read_v("test_mon7.txt")

# movie_data = movie_data.sample(frac=1)
# print(movie_data.head(10))

# np.save("mon7",movie_data)

data =  np.loadtxt("test_mon6.txt")



np.random.shuffle(data)

print(data)

np.savetxt("mon6.txt",data,delimiter="\t",fmt='%i')


