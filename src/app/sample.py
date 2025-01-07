'''sample.py Samples a single random row from test CSV file and writes it to a new CSV/JSON file.'''


import os
import sys
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('../../logs/debug.log', mode='a'),
        logging.StreamHandler(sys.stdout)]
    ) 


def generate_name(idx):
    '''Generate an incremental filename.'''
    base_name = 'row'
    # ext = 'csv'
    ext = 'json'
    return f"{base_name}_{idx}.{ext}"

def sample_row(input):
    '''Samples a single random row'''
    # Take first 100 rows to sample from.
    df = pd.read_csv(input, nrows=100)
    
    while True:
        root = '../../data/test/random_rows'
        random_row = df.sample(n=1)
        random_row_name = generate_name(random_row.index[0])
        random_row_path = f'{root}/{random_row_name}'
        if not os.path.exists(random_row_path):
            break
    
    # random_row.to_csv(random_row_path)
    random_row.to_json(random_row_path, orient='records', lines=True)
    logger.info(f'Generated Random Row: {random_row_name}')
    return random_row_path