import tensorflow as tf
from layers import SelfAttention
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.constraints import MinMaxNorm, NonNeg
def MLP(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(input_shape[1],), activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.45),
        tf.keras.layers.Dense(64, activation='softmax'),
        tf.keras.layers.Dense(input_shape[1]),
        ])

    model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    model.summary()
    return model

def Attention_MLP(input_shape):

    vocabulary_size = 1000
    embedding_dims = 256

    X = Input(shape=(input_shape[1],))
    # Word-Embedding Layer
    embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims, mask_zero=True)(X)

    # Optional Self-Attention Mechanisms
    embedded, attention_weights = SelfAttention(size=256,
                                                num_hops=20,
                                                use_penalization=False)(embedded)
    # Multi-Layer Perceptron
    embedded_flattened = Flatten()(embedded)
    fully_connected = Dense(units=256, activation='relu')(embedded_flattened)
    # fully_connected = Dense(units=256, activation='relu')(fully_connected)
    # fully_connected = Dense(units=256, activation='relu')(fully_connected)
    fully_connected = Dense(units=128, activation='relu')(fully_connected)
    fully_connected = Dense(units=128, activation='softmax')(fully_connected)
    # fully_connected = Dense(units=128, activation='relu')(embedded_flattened)
    # Prediction Layer
    Y = Dense(units=input_shape[1])(fully_connected)
    model = Model(inputs=X, outputs=Y)
    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    model.summary()

    return model

def Attention_MLP_combine(input_shape):

    vocabulary_size = 1000
    embedding_dims = 64
    
    X = Input(shape=(input_shape[1],))
    # Word-Embedding Layer
    embedded = Embedding(input_dim=vocabulary_size, output_dim=embedding_dims)(X)

    # Optional Self-Attention Mechanisms
    embedded, attention_weights = SelfAttention(size=128,
                                                num_hops=20,
                                                use_penalization=False)(embedded)
    # Multi-Layer Perceptron
    embedded_flattened = Flatten()(embedded)
    fully_connected = Dense(units=256, activation='relu')(embedded_flattened)
    fully_connected = Dense(units=128, activation='relu')(fully_connected)

    # Prediction Layer
    Y = Dense(units=128)(fully_connected)
    model = Model(inputs=X, outputs=Y)

    return model

def Combine_model(P_input_shape, S_input_shape):
    
    model_pre = Attention_MLP_combine(P_input_shape)
    model_sim = Attention_MLP_combine(S_input_shape)
    
    X = Concatenate()([model_pre.output, model_sim.output])
    combine = Dense(units=256, activation='relu')(X)
    combine = Dense(units=256, activation='relu')(combine)
    combine = Dense(units=128, activation='relu')(combine)
    combine = Dense(units=128, activation='softmax')(combine)
    Y = Dense(units=P_input_shape[1])(combine)

    model = Model(inputs=[model_pre.input, model_sim.input], outputs=Y)
    model.compile(loss='mae', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.MeanAbsoluteError()])
    return model
