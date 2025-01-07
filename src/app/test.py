'''test.py - Test Flask application script.'''

import json
import logging
import sys

import requests

from sample import sample_row


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    )


def main():
    url = 'http://localhost:9696/predict'
    
    root = '../../data' # Raw Data root director
    test_csv = f'{root}/test/df_test.csv'
    
    sample_test_data = sample_row(test_csv)
    # Open the JSON file with sample data
    logger.info('\n Getiing Sample Data...')
    with open(sample_test_data) as f:
        sample_data = json.load(f)

    response = requests.post(url, json=sample_data)

    if response.status_code == 200:        
        ann = response.json()['ANN']
        xgb = response.json()['XGB']
        actual = response.json()['ACTUAL']
        logger.info(
            f'Actual Value: {actual:.3f} \nPredictions: \nANN: {ann:.3f} \nXGB: {xgb:.3f}')
    else:
        logger.info(
            f'Failed to retrieve prediction. Status Code: {response.status_code} \n')
        

if __name__ == '__main__':
    main()