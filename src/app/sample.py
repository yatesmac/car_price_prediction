'''sample.py Samples a single random row from test CSV file and writes it to a new CSV file.'''


import os

import pandas as pd
import numpy as np


def generate_name(idx):
    '''Generate an incremental filename.'''
    base_name = 'sample'
    ext = 'csv'
    return f"{base_name}_{idx}.{ext}"

def sample_row_csv(input):
    '''Samples a single random row'''
    df = pd.read_csv(input)
    
    while True:
        root = '../../data/test/random_rows'
        random_row = df.sample(n=1)
        random_row_path = generate_name(random_row.index)
        if not os.path.exists(f'{root}/{random_row_path}'):
            break
    
    random_row.to_csv(random_row_path)
    return random_row_path
    

