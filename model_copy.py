from models import *
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error
from evaluate_standard import RMSE, MSE
import os

# from sklearn.metrics import mean_squared_error

class CF_model(object):
    def __init__(self, args):

        self.dataset = args.dataset
        self.dataset_PS = args.dataset_PS
        self.dataset_IU = args.dataset_IU
        self.dataset_ACP = args.dataset_ACP

        self.ground_true = None


        self.dataset_shape = None
        self.dataset_shape_ = None
        self.model = args.model
        self.batch = args.batch_size
        self.epoch = args.epoch

        self.train_set = None
        self.test_set = None

        self.set_groundtruth()

        self.data_pre()
        self.build_model()
    
    def set_groundtruth(self):

        self.ground_true = 'evaluate_dataset/1m_evaluate_notchange.npy'

        
    
    def set_datasetpath(self):
        if self.dataset == "1m":
            if self.model == "C_MLP":
                path1 = f'movielen1M_dataset/ml-1m-prediction/{self.dataset_IU}_{self.dataset_ACP}_prediction_test_movielen{self.dataset}.npy'
                path2 = f'movielen1M_dataset/ml-1m-prediction/{self.dataset_IU}_{self.dataset_ACP}_similarity_test_movielen{self.dataset}.npy'
                return path1, path2
            else:
                dataset_path = f'movielen1M_dataset/ml-1m-{self.dataset_PS}/{self.dataset_IU}_{self.dataset_ACP}_{self.dataset_PS}_test_movielen{self.dataset}.npy'
        elif self.dataset == "100k":
            if self.model == "C_MLP":
                path1 = f'movielen100k_dataset/ml-100k-prediction/{self.dataset_IU}_{self.dataset_ACP}_prediction_test_movielen{self.dataset}.npy'
                path2 = f'movielen100k_dataset/ml-100k-prediction/user_cos_prediction_test_movielen{self.dataset}.npy'
                return path1, path2
            else:
                dataset_path = f'movielen100k_dataset/ml-100k-{self.dataset_PS}/{self.dataset_IU}_{self.dataset_ACP}_{self.dataset_PS}_test_movielen{self.dataset}.npy'
        elif self.dataset == "amazon":
            if self.model == "C_MLP":
                path1 = f'movielen1M_dataset/ml-1m-prediction/{self.dataset_IU}_{self.dataset_ACP}_prediction_test_movielen{self.dataset}.npy'
                path2 = f'movielen1M_dataset/ml-1m-similarity/{self.dataset_IU}_{self.dataset_ACP}_similarity_test_movielen{self.dataset}.npy'
                return path1, path2
            else:
                dataset_path = f'Amazon_dataset/amazon-{self.dataset_PS}/{self.dataset_IU}_{self.dataset_ACP}_{self.dataset_PS}_test_movielen{self.dataset}.npy'
        else:
            print("no dataset")        
        return dataset_path

    def data_pre(self):
        if self.model == "C_MLP":
            path1, path2 = self.set_datasetpath()
            first = np.load(path1)
            second = np.load(path2)
            self.dataset_shape = first.shape
            self.dataset_shape_ = second.shape
            print(self.dataset_shape, self.dataset_shape_)
            #itembase_similarity:1413721 ； userbase_similarity:444153 item_prediction:100000
            x1 = first[:-1, :]
            x2 = second[:-1, :]
            self.test1 = first[round(self.dataset_shape[0]*0.8):-2, :]
            self.test2 = second[round(self.dataset_shape[0]*0.8):-2, :]
            y = first[1:, :]
            # test = tf.data.Dataset.from_tensor_slices(x1, x2)
            X = tf.data.Dataset.from_tensor_slices((x1, x2))
            Y = tf.data.Dataset.from_tensor_slices(y)
            data_set = tf.data.Dataset.zip((X, Y))
            self.train_set = data_set.take(round(self.dataset_shape[0]*0.8))
            # self.test_set = data_set.skip(round(self.dataset_shape[0]*0.8))
        else:
            dataset_path = self.set_datasetpath()
            _ = np.load(dataset_path)
            true_dataset = np.load(self.ground_true)
            self.dataset_shape = _.shape
            print(self.dataset_shape)
            #itembase_similarity:1413721 ； userbase_similarity:444153 item_prediction:100000
            x = _[:-1, :]
            y = true_dataset[1:, :]
            print("------------------------------------------")
            print(x.shape, true_dataset.shape)
            print("------------------------------------------")

            data_set = tf.data.Dataset.from_tensor_slices((x, y))
            
            # data_set.shuffle(1000)
            self.train_set = data_set.take(round(self.dataset_shape[0]*0.8))
            # self.test_set = data_set.skip(round(self.dataset_shape[0]*0.8))
            self.test_set = _[round(self.dataset_shape[0]*0.8):, :]

    def build_model(self):
        if self.model == "MLP":
            self.model = MLP(self.dataset_shape)
        if self.model == "A_MLP":
            self.model = Attention_MLP(self.dataset_shape)
        if self.model == "C_MLP":
            self.model = Combine_model(self.dataset_shape, self.dataset_shape_)
        else :
            print("no model")

    def train(self):
        callback = EarlyStopping(monitor='loss', patience=3)


        self.model.summary()

        self.train_set = self.train_set.batch(1)
        # self.test_set = self.test_set.batch(1)

        self.model.fit(self.train_set, epochs=self.epoch, callbacks=[callback]) 

        # if self.model == "C_MLP":
        #     # self.predict_ = self.model.predict(x)
        #     result = evaluate()
        #     print(result)
        # else:
        #     x, y = tf.data.experimental.get_single_element(self.test_set)
        #     result = self.model.evaluate(x, y)
        #     print(result)

    def evaluate__(self, ground_true):
        # x, y = tf.data.experimental.get_single_element(self.test_set)
        print(self.test_set.shape)
        if self.model == "C_MLP":
            self.predict_ = self.model.predict([self.test1, self.test2])
        else:
            # x, y = tf.data.experimental.get_single_element(self.test_set)
            self.predict_ = self.model.predict(self.test_set[:-1])
            print(self.predict_.shape, ground_true[1:].shape)
            self.predict_ = np.round(self.predict_, 2)
        mse = MSE(ground_true[1:], self.predict_)
        rmse = RMSE(ground_true[1:], self.predict_)
        print(mse, rmse)
 
        return self.predict_

        


