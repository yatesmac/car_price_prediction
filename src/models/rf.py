'''train.py - Train Random Forest Regression model and save to Pickle file.'''

import pickle
import logging
import sys
import os

import numpy as np

from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# sys.path lists directories that Python searches for modules to import
sys.path.append('../data') 
from data_preparation import dataset


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 


def train(X_train, y_train):
    '''Train a model using the provided data.'''
    logger.info(
        'Training the final model using Random Forest Regression.')
    params = {
        'max_depth': 20,
        'min_samples_split': 2,
        'n_estimators': 200
       }
    model = RandomForestRegressor(**params)
    return model.fit(X_train, y_train)

    
def validate(model, X_test, y_test):
    '''Validate model pyformance.'''   
    # Make predictions on validation data
    y_pred = model.predict(X_test)

    # Evaluate the model
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    return r2, rmse


def save_model(model, model_file):
    '''Save the model.'''
    logger.info('Saving the model as a pickle file')
    try: 
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f'Model saved successfully to {model_file} \n')
    
    except OSError as e:
        logger.info(e, '\n')
    

def main():
    '''Train, evaluate and save model.'''
    root = '../..'
    model_file = f'{root}/models/rf_v1.pkl'
    train_csv = f'{root}/data/train/df_full_train.csv'
    test_csv = f'{root}/data/test/df_test.csv'
    target = 'price'

    X_train, y_train, X_test, y_test = dataset(train_csv, test_csv, target, scaler=True)
    model = train(X_train, y_train)
    r2, rmse = validate(model, X_test, y_test)
    logger.info(f'R2 Score = {r2}\nRMSE = {rmse}')
    
    if not os.path.exists(model_file):
        save_model(model, model_file)


if __name__ == '__main__':
    main()