from matrix_factorization import BaselineModel, KernelMF, train_update_test_split

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error



# Movie data found here https://grouplens.org/datasets/movielens/
cols = ["user_id", "item_id", "rating", "timestamp"]
movie_data = pd.read_csv(
    "mon6.txt", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
)
ground_true = pd.read_csv(
    "mon7.txt", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
)
test_movie_X = ground_true[["user_id", "item_id"]]
test_movie_y = ground_true["rating"]

train_movie_X = movie_data[["user_id", "item_id"]]
train_movie_y = movie_data["rating"]

# Initial training
matrix_fact = KernelMF(n_epochs=10, n_factors=3, verbose=1, lr=0.1, reg=0.5)

matrix_fact.fit(test_movie_X, test_movie_y)


pred = matrix_fact.predict(test_movie_X)
# print(pred)
print(len(test_movie_y), len(pred))
rmse = mean_squared_error(test_movie_y, pred, squared=False)
print(f"\nTest RMSE: {rmse:.4f}")
print(pred[:10], test_movie_y[:10])
