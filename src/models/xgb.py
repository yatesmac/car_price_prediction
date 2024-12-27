'''train.py - Train XGBoost Regression model and save to Pickle file.'''

import pickle
import logging
import sys

import os.path

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, root_mean_squared_error
from xgboost import XGBRegressor

sys.path.append('../data') 
from dataset_preparation import dataset, categorize, vectorize


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
        'Training the final model using XGBoost Regression.')
    params = {
        'learning_rate': 0.2,
        'max_depth': 5,
        'n_estimators': 200
       }
    model = XGBRegressor(**params)
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
    model_file = '../../models/xgb.pkl'
    train_csv = '../../data/raw/train/df_full_train.csv'
    test_csv = '../../data/raw/test/df_test.csv'
    target = 'price'

    X_train, y_train, X_test, y_test = dataset(train_csv, test_csv, target)

    model = train(X_train=X_train, y_train=y_train)
    r2, rmse = validate(model=model, X_test=X_test, y_test=y_test)
    logger.info(f'R2 Score = {r2}\nRMSE = {rmse}')
    
    if not os.path.exists(model_file):
        save_model(model, model_file)


if __name__ == '__main__':
    main()