'''train.py - Train Artificial Neural Network model and save to H5 file.'''

import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

sys.path.append('../data') 
from datset_preparation import dataset, categorize, scale, vectorize


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 


def train(X_train):
    model = Sequential()
    model.add(Dense(64, activation='relu',input_shape=(X_train.shape[1], )))
    model.add(Dropout(0.15))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model


def main():
    '''Train, evaluate and save model.'''
    train_csv = '../../data/raw/train/df_full_train.csv'
    test_csv = '../../data/raw/test/df_test.csv'
    target = 'price'
    
    X_train, y_train, X_test, y_test = dataset(train_csv, test_csv, target, scaler=True)
    
    learning_rate = 0.01
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    rmse = keras.metrics.RootMeanSquaredError()

    model = train(X_train)
    model.compile(optimizer=optimizer, loss=loss, metrics=[rmse])
    model.save_weights('model_v1.h5', save_format='h5')

    checkpoint = keras.callbacks.ModelCheckpoint(
        'ann_v1_{epoch:02d}_{val_root_mean_squared_error:.3f}.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max'
    )

    return model.fit(
        X_train,
        y_train,
        batch_size=4096,
        epochs=50,
        verbose=1,
        validation_data=(X_test,y_test),
        callback=[checkpoint]
    )

if __name__ == '__main__':
    history = train()