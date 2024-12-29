'''test.py - Test Flask application script.'''

import json
import logging
import sys

import requests

from sample import sample_row_csv
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


def get_data():
    root = '../../data' # Raw Data root director
    train_csv = f'{root}/train/df_full_train.csv'
    test_csv = f'{root}/test/df_test.csv'
    target = 'price'

    sample_test_csv = sample_row_csv(test_csv)
    _, _, X_test, y_test = dataset(train_csv, sample_test_csv, target, scaler=True)
    return X_test, y_test


def main():
    url = 'http://localhost:9696/predict'
    
    X, y = get_data()
    sample_data = json.load(X)
    response = requests.post(url, json=sample_data)

    if response.status_code == 200:        
        ann = response.json()['ANN']
        xgb = response.json()['XGB']
        logger.info(
            f'Actual Value: {y:.3f} \nPredictions: \nANN: {ann} \nXGB: {xgb}')
    else:
        logger.info(
            f'Failed to retrieve prediction. Status Code: {response.status_code} \n')
        

if __name__ == '__name__':
    main()