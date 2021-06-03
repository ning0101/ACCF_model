from models import *
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


data_set = np.load('movielen100k_dataset/cosine_similarity_userbasetest_movielen100k.npy')
#itembase_similarity:1413721 ； userbase_similarity:444153 item_prediction:100000
data1 = data_set[:7, :]
print(data_set.shape)
data2 = data_set[1:, :]
dataset = tf.data.Dataset.from_tensor_slices((data1, data2))

print(dataset.element_spec)
train_dataset = dataset.take(4)
test_dataset = dataset.skip(4)

model = MLP(data_set.shape)
# model = Combine_model(data_set.shape) #############未修正
# model = Attention_MLP(data_set.shape)
model.summary()


train_dataset = train_dataset.batch(1)
test_dataset = test_dataset.batch(1)
'''
def get_checkpoint_best():
    checkpoint_path = 'checkpoints/checkpoint'
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                            save_weights_only=True,
                            monitor='loss',
                            mode='min',
                            save_best_only=True,
                            verbose=2)
    return checkpoint

'''

evaluate_dataset = np.load('evaluate_dataset/100k_evaluate_notchange.npy')
evaluate_dataset = evaluate_dataset[round(evaluate_dataset.shape[0]*0.8):, :]
print(evaluate_dataset.shape)
# model.fit(train_dataset, epochs=100) 
# model.evaluate__(evaluate_dataset)
# model.evaluate(test_dataset)
# model.fit_generator(train_dataset, epochs=100) 

