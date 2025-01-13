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


def url_to_json(url):
    escaped_string = url.replace("'", '"') 
    return json.dumps({"url": escaped_string})


def main():
    url = 'http://localhost:9696/predict'
    
    root = '../../data' # Raw Data root director
    test_csv = f'{root}/test/df_test.csv'
    
    # Create new sample
    # sample_url = sample_row(test_csv)
    # Use existing sample
    sample_url = f'{root}/test/random_rows/row_3.csv'
    sample_data = url_to_json(sample_url)   

    response = requests.post(url, json=sample_data)

    if response.status_code == 200:        
        ann = response.json()['ANN']
        xgb = response.json()['XGB']
        rf = response.json()['RF']
        actual = response.json()['ACTUAL']
        logger.info(
            f'Actual Value: {actual:.2f} \tPredictions: Neural Net: {ann:.2f} Random Forest: {rf:.2f} XG Boost: {xgb:.2f}')
    else:
        logger.info(
            f'Failed to retrieve prediction. Status Code: {response.status_code} \n')
        

if __name__ == '__main__':
    main()