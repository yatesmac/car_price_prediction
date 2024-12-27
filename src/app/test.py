'''test.py - Test Flask application script.'''
import json
import logging
import sys

import requests


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    )

url = 'http://localhost:9696/predict'
data = '../../data/raw/test/test.csv'

# TODO: sample random value from test data
'''
# Open the JSON file with sample row data 
with open(data) as f:
    sample_data = json.load(f)
print(sample_data)
'''
# TODO: figure out how to send both X and y data.

response = requests.post(url, json=sample_data)
if response.status_code == 200:        
    actual_value = response.json()['actual_value']
    prediction = response.json()['prediction']
    logger.info(
        f'Actual Value: {actual_value:.3f} \nPrediction: {prediction} \n')
else:
    logger.info(
        f'Failed to retrieve prediction. Status Code: {response.status_code} \n')