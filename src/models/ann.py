'''train.py - Train Artificial Neural Network model and save to H5 file.'''

import logging
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# sys.path lists directories that Python searches for modules to import
sys.path.append('../data') 
from data_preparation import dataset


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

root = '../..'# Root Directory

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f'{root}/logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 


def train(X_train):
    '''Train a model using the provided data.'''
    learning_rate = 0.1
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.MeanSquaredError()
    rmse = keras.metrics.RootMeanSquaredError()
    model = Sequential(
        [
            Dense(64, activation='relu',input_shape=(X_train.shape[1], )),
            Dense(128, activation='relu'),
            Dropout(0.15),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=[rmse])
    return model


def main():
    '''Train, evaluate and save model.'''
    train_csv = f'{root}/data/train/df_full_train.csv'
    test_csv = f'{root}/data/test/df_test.csv'
    model_path = f'{root}/models/ann_v1.keras'
    weighted_model_path = f'{root}/models/ann_v1.h5'
    ckpt_path = '../../models/checkpoints/ann_v1_{epoch:02d}_{val_root_mean_squared_error:.3f}.keras'
    target = 'price'

    X_train, y_train, X_test, y_test = dataset(train_csv, test_csv, target, scaler=True)
    model = train(X_train)

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_best_only=True,
        monitor='val_rmse',
        mode='min'
    )

    history = model.fit(
        X_train,
        y_train,
        batch_size=4096,
        epochs=45,
        verbose=1,
    #   callbacks=[checkpoint],
        validation_data=(X_test,y_test)
    )
    model.save(model_path)
    model.save(weighted_model_path)
    logger.info(f'Succesfully saved model to {weighted_model_path}.')
    return history


if __name__ == '__main__':
    history = main()
    
    